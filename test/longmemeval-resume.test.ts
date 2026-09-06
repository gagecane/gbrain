/**
 * resume.ts — re-scoring prior rows (pure, no engine).
 *
 *   - seedBucketsFromRows: `goldMissing` / `collisions` count EVERY question
 *     row (abstention-excluded, no-gold, not-in-dataset and error rows
 *     included) — the same set the live harness counts — so
 *     `run_config.gold_missing_from_haystack` / `slug_collisions` are
 *     identical for a fresh run and a resume of the same file; only
 *     SCORED rows feed the buckets.
 */
import { describe, test, expect } from 'bun:test';
import { seedBucketsFromRows } from '../src/eval/longmemeval/resume.ts';
import type { RecallBucket } from '../src/eval/longmemeval/metrics.ts';

describe('seedBucketsFromRows — gold_missing / slug_collisions row set', () => {
  const goldByQid = new Map<string, readonly string[]>([
    ['scored', ['s1']],
    ['abs_abs', ['s9']],
    ['nogold', []],
    ['aborted', ['s3']],
  ]);
  const rows = [
    { question_id: 'scored', question_type: 'single-session-user', hypothesis: 'h', retrieved_session_ids: ['s1'], gold_missing_from_haystack: ['s1'], slug_collision: 1 },
    { question_id: 'abs_abs', question_type: 'single-session-user', hypothesis: 'h', retrieved_session_ids: [], gold_missing_from_haystack: ['s9'], slug_collision: 0 },
    { question_id: 'nogold', question_type: 'multi-session', hypothesis: 'h', retrieved_session_ids: [], gold_missing_from_haystack: [], slug_collision: 2 },
    { question_id: 'not-in-dataset', question_type: 'multi-session', hypothesis: 'h', retrieved_session_ids: [], gold_missing_from_haystack: ['x'], slug_collision: 0 },
    { question_id: 'aborted', question_type: 'multi-session', hypothesis: '', error: 'slug_collision touches a gold session id', slug_collision: 1, slug_collision_gold: ['chat/a-b'] },
    { kind: 'by_type_summary', slug_collision: 5, gold_missing_from_haystack: ['ignored'] },
  ];

  test('counts over every question row; buckets only over scored rows; summary lines ignored', () => {
    const buckets: Record<string, RecallBucket> = {};
    const res = seedBucketsFromRows(rows, buckets, { goldByQid, k: 5, includeAbstention: false });
    // scored + abs_abs + not-in-dataset carry gold_missing; the summary line does not count.
    expect(res.goldMissing).toBe(3);
    // scored + nogold + the collision-abort error row.
    expect(res.collisions).toBe(3);
    expect(res.seeded).toBe(1);
    expect(res.excludedAbstention).toBe(1);
    expect(Object.keys(buckets)).toEqual(['single-session-user']);
    expect(buckets['single-session-user'].total).toBe(1);
    expect(buckets['single-session-user'].all_hit).toBe(1);
  });

  test('--include-abstention changes the buckets, not the integrity counters', () => {
    const buckets: Record<string, RecallBucket> = {};
    const res = seedBucketsFromRows(rows, buckets, { goldByQid, k: 5, includeAbstention: true });
    expect(res.goldMissing).toBe(3);
    expect(res.collisions).toBe(3);
    expect(res.seeded).toBe(2);
    expect(res.excludedAbstention).toBe(0);
  });
});
