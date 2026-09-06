/**
 * emit.ts — the JSONL emitter's file modes. The atomic-rewrite mode is the
 * guard for the --judge --resume-from path: the resume file (paid reader
 * rows) must never be truncated while the backfill is in flight.
 */
import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { makeEmitter, emitByTypeSummary } from '../src/eval/longmemeval/emit.ts';

let dir: string;
beforeEach(() => { dir = mkdtempSync(join(tmpdir(), 'lme-emit-')); });
afterEach(() => { rmSync(dir, { recursive: true, force: true }); });

const lines = (p: string) => readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));

describe('makeEmitter', () => {
  test('truncate mode (default) replaces the file', () => {
    const p = join(dir, 'out.ndjson');
    writeFileSync(p, JSON.stringify({ question_id: 'old' }) + '\n');
    const em = makeEmitter(p, false);
    em.emit({ question_id: 'new' });
    em.close();
    expect(lines(p).map((r) => r.question_id)).toEqual(['new']);
  });

  test('append mode keeps prior rows', () => {
    const p = join(dir, 'out.ndjson');
    writeFileSync(p, JSON.stringify({ question_id: 'old' }) + '\n');
    const em = makeEmitter(p, true);
    em.emit({ question_id: 'new' });
    em.close();
    expect(lines(p).map((r) => r.question_id)).toEqual(['old', 'new']);
  });

  test('atomicRewrite: the original is untouched until close(), then replaced in one rename; the temp file is gone', () => {
    const p = join(dir, 'run.ndjson');
    writeFileSync(p, [{ question_id: 'q1', hypothesis: 'paid answer' }, { kind: 'by_type_summary' }].map((r) => JSON.stringify(r)).join('\n') + '\n');
    const em = makeEmitter(p, false, { atomicRewrite: true });
    // Mid-run (a kill here must lose nothing): the resume file still has the paid row.
    expect(lines(p).map((r) => r.question_id ?? r.kind)).toEqual(['q1', 'by_type_summary']);
    expect(existsSync(`${p}.rewrite.tmp`)).toBe(true);
    em.emit({ question_id: 'q1', hypothesis: 'paid answer', judge_correct: true });
    expect(lines(p)[0].judge_correct).toBeUndefined(); // still the old file
    em.close();
    expect(lines(p)).toEqual([{ question_id: 'q1', hypothesis: 'paid answer', judge_correct: true }]);
    expect(existsSync(`${p}.rewrite.tmp`)).toBe(false);
    em.close(); // idempotent
  });

  test('atomicRewrite is ignored in append mode (nothing to protect)', () => {
    const p = join(dir, 'out.ndjson');
    writeFileSync(p, JSON.stringify({ question_id: 'old' }) + '\n');
    const em = makeEmitter(p, true, { atomicRewrite: true });
    em.emit({ question_id: 'new' });
    expect(existsSync(`${p}.rewrite.tmp`)).toBe(false);
    em.close();
    expect(lines(p).map((r) => r.question_id)).toEqual(['old', 'new']);
  });

  test('the summary writer runs against the renamed file (rewrite → summary order)', () => {
    const p = join(dir, 'run.ndjson');
    writeFileSync(p, JSON.stringify({ question_id: 'q1' }) + '\n' + JSON.stringify({ kind: 'by_type_summary', stale: true }) + '\n');
    const em = makeEmitter(p, false, { atomicRewrite: true });
    em.emit({ question_id: 'q1', judge_correct: false });
    em.close();
    emitByTypeSummary(p, { kind: 'by_type_summary', k: 5 } as never);
    const rows = lines(p);
    expect(rows.length).toBe(2);
    expect(rows[0]).toEqual({ question_id: 'q1', judge_correct: false });
    expect(rows[1].kind).toBe('by_type_summary');
    expect(rows[1].stale).toBeUndefined();
  });
});
