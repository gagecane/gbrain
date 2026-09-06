/**
 * emit.ts — JSONL emission for the LongMemEval harness: the per-row emitter
 * (stdout or file; truncate or append) and the resume-safe `by_type_summary`
 * writer. Peeled from src/commands/eval-longmemeval.ts.
 *
 * INVARIANT: the summary is the FINAL line of the output and there is at most
 * one — any prior `kind:"by_type_summary"` line is removed before the new one
 * is appended, so a resume never stacks summaries. Its `_meta.metric_glossary`
 * is the ONE glossary block per response ([CDX-25]) and names exactly the
 * metrics the summary carries (recall_all@k, recall_any@k, and qa_accuracy
 * when the judged lane ran).
 *
 * INVARIANT: a CR inside an emitted line is corrupt input, never a silent
 * line break — both writers throw instead of splitting a JSONL record.
 */

import { closeSync, existsSync, openSync, readFileSync, renameSync, writeFileSync, writeSync } from 'node:fs';
import { buildMetricGlossaryMeta } from '../../core/eval/metric-glossary.ts';
import type { ByTypeSummaryV2 } from './metrics.ts';

export interface JsonlEmitter {
  emit(obj: object): void;
  close(): void;
}

export interface EmitterOptions {
  /**
   * Rewrite-in-place mode: the emitter writes to `<outputPath>.rewrite.tmp`
   * and renames it over `outputPath` on `close()`, so the original file is
   * never truncated while the run is in flight. Required whenever the output
   * path IS the resume file and the rows are re-emitted rather than appended
   * (the --judge backfill path): a kill mid-backfill used to leave a 0-byte
   * file and every paid reader row was lost. Ignored when `append` is true.
   */
  atomicRewrite?: boolean;
}

/**
 * `outputPath` undefined → stdout (stays open). Append mode is used by
 * --resume-from when the output path is the resume file; truncating would
 * erase the already-answered questions. When the rows are rewritten instead
 * (judge backfill), pass `atomicRewrite` — see EmitterOptions.
 */
export function makeEmitter(outputPath?: string, append: boolean = false, options: EmitterOptions = {}): JsonlEmitter {
  if (!outputPath) {
    return {
      emit(obj) {
        const json = JSON.stringify(obj);
        if (json.includes('\r')) throw new Error('CRLF in JSONL emit (corrupt input)');
        process.stdout.write(Buffer.from(json + '\n', 'utf8'));
      },
      close() { /* stdout stays open */ },
    };
  }
  const atomic = options.atomicRewrite === true && !append;
  const writePath = atomic ? `${outputPath}.rewrite.tmp` : outputPath;
  const fd = openSync(writePath, append ? 'a' : 'w');
  let closed = false;
  return {
    emit(obj) {
      const json = JSON.stringify(obj);
      if (json.includes('\r')) throw new Error('CRLF in JSONL emit (corrupt input)');
      writeSync(fd, Buffer.from(json + '\n', 'utf8'));
    },
    close() {
      if (closed) return;
      closed = true;
      closeSync(fd);
      // Atomic on POSIX: readers see either the old file or the complete new one.
      if (atomic) renameSync(writePath, outputPath);
    },
  };
}

/** Emit the by_type_summary as the final line (replacing any prior summary line) with its glossary block. */
export function emitByTypeSummary(outputPath: string | undefined, summary: ByTypeSummaryV2): void {
  const keys = [`recall_all@${summary.k}`, `recall_any@${summary.k}`, ...(summary.qa_accuracy ? ['qa_accuracy'] : [])];
  const withMeta = { ...summary, _meta: { metric_glossary: buildMetricGlossaryMeta(keys) } };
  const json = JSON.stringify(withMeta);
  if (json.includes('\r')) throw new Error('CRLF in by_type_summary emit (corrupt input)');
  if (!outputPath) {
    process.stdout.write(Buffer.from(json + '\n', 'utf8'));
    return;
  }
  let existing = '';
  if (existsSync(outputPath)) existing = readFileSync(outputPath, 'utf8');
  const kept: string[] = [];
  for (const line of existing.split('\n')) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (row && typeof row === 'object' && (row as { kind?: unknown }).kind === 'by_type_summary') continue;
    } catch {
      // Corrupt line — keep as-is; the resume loader has its own skip logic.
    }
    kept.push(line);
  }
  kept.push(json);
  writeFileSync(outputPath, kept.join('\n') + '\n', 'utf8');
}
