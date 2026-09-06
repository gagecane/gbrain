/**
 * judge-lane.ts — harness-side orchestration for `gbrain eval longmemeval
 * --judge`: the `--max-usd` parser, the per-row `judge_config_hash` (D33),
 * backfill row selection + the mixed-config gate, the spend preflight, and
 * the concurrent judge-only backfill. Keeps src/commands/eval-longmemeval.ts
 * under its size cap; stderr writing and exit codes stay in the harness.
 *
 * INVARIANT: pure except `runJudgeBackfill`, whose only side effects are the
 * injected judge client and in-place row updates. No engine, no file I/O.
 *
 * INVARIANT (D33): a row's `judge_config_hash` covers the judge pins (model,
 * prompt version, max_tokens, temperature) AND the reader pins the row was
 * produced under (reader model, reader prompt sha, k, reader max_tokens). A
 * backfill hashes each prior row from the row's OWN recorded reader pins when
 * present, so a file answered by another reader is never relabelled as this
 * run's; rows already judged under a different hash are refused unless
 * `--allow-mixed-run-config`.
 */

import { runWithLimit } from '../../core/worker-pool.ts';
import { estimateTokens } from '../../core/search/token-budget.ts';
import { BudgetLedger, isJudgeModelPriced } from '../shared/judge-runner.ts';
import {
  JUDGE_MAX_TOKENS,
  JUDGE_PROMPT_VERSION,
  JUDGE_TEMPERATURE,
  estimateJudgeRunUsd,
  judgeConfigHash,
  judgeRow,
  stripJudgeFields,
  type JudgeLaneContext,
  type JudgePromptInput,
} from './judge.ts';
import type { LongMemEvalQuestion } from './adapter.ts';

/** `--max-usd N|off`: `off` disables the cap (and lets an unpriced judge run). */
export function parseMaxUsd(flag: string, v: string): number | null {
  const s = v.trim().toLowerCase();
  if (s === 'off' || s === 'none' || s === 'unlimited') return null;
  const n = Number(s);
  if (!Number.isFinite(n) || n < 0) throw new Error(`${flag} must be a non-negative number of USD or 'off' (got: ${v})`);
  return n;
}

/** This run's reader pins (the hash's reader half for live rows). */
export interface ReaderPins {
  model: string;
  prompt_sha: string;
  max_tokens: number;
  k: number;
}

export type RowLike = Record<string, unknown>;

/**
 * Per-row hasher: a prior row's recorded `reader_model` / `reader_prompt_sha`
 * / `reader_max_tokens` win over this run's pins; k is the run's `--top-k`
 * (already gated by `retrieval_config_hash`).
 */
export function makeJudgeConfigHasher(judgeModel: string, run: ReaderPins): (row: RowLike) => string {
  return (row) => judgeConfigHash({
    judge_model: judgeModel,
    prompt_version: JUDGE_PROMPT_VERSION,
    max_tokens: JUDGE_MAX_TOKENS,
    temperature: JUDGE_TEMPERATURE,
    reader_model: typeof row.reader_model === 'string' ? row.reader_model : run.model,
    reader_prompt_sha: typeof row.reader_prompt_sha === 'string' ? row.reader_prompt_sha : run.prompt_sha,
    context: {
      k: run.k,
      max_tokens: typeof row.reader_max_tokens === 'number' ? row.reader_max_tokens : run.max_tokens,
    },
  });
}

/** True when the row carries a judge attempt (verdict, error, or budget skip). */
export function hasJudgeAttempt(row: RowLike): boolean {
  return typeof row.judge_correct === 'boolean' || typeof row.judge_error === 'string' || row.judge_skipped === 'budget';
}

/** A judged verdict that needs no re-judge. */
export function hasSettledVerdict(row: RowLike): boolean {
  return typeof row.judge_correct === 'boolean' && typeof row.judge_error !== 'string' && row.judge_skipped !== 'budget';
}

export interface BackfillSelection {
  /** Prior rows with a reader hypothesis and no settled verdict — judged from their stored hypothesis. */
  candidates: RowLike[];
  /** Prior rows whose verdict stands. */
  settled: number;
  /** Rows already stamped with a DIFFERENT judge_config_hash than this run would stamp (refuse unless allowed). */
  mismatched: number;
  foreign: string[];
  /** Candidates produced by --retrieval-only (no reader hypothesis) — refused. */
  retrievalOnly: number;
  /** Candidates whose question_id is not in the dataset (cannot be judged: no gold). */
  missingFromDataset: number;
}

export function selectBackfillRows(
  priorRows: ReadonlyArray<RowLike>,
  ctx: { questionByQid: ReadonlyMap<string, LongMemEvalQuestion>; hashFor: (row: RowLike) => string },
): BackfillSelection {
  const sel: BackfillSelection = { candidates: [], settled: 0, mismatched: 0, foreign: [], retrievalOnly: 0, missingFromDataset: 0 };
  const foreign = new Set<string>();
  for (const row of priorRows) {
    if (row.kind === 'by_type_summary' || typeof row.question_id !== 'string') continue;
    if (typeof row.hypothesis !== 'string' || row.hypothesis === '') continue; // reader error rows are re-run, not judged
    if (typeof row.error === 'string') continue;
    if (typeof row.judge_config_hash === 'string' && row.judge_config_hash !== ctx.hashFor(row)) {
      sel.mismatched++;
      foreign.add(row.judge_config_hash);
    }
    if (hasSettledVerdict(row)) { sel.settled++; continue; }
    if (row.retrieval_only === true) { sel.retrievalOnly++; continue; }
    if (!ctx.questionByQid.has(row.question_id)) { sel.missingFromDataset++; continue; }
    sel.candidates.push(row);
  }
  sel.foreign = [...foreign].sort();
  return sel;
}

export interface JudgePreflightInput {
  judgeModel: string;
  maxUsd: number | null;
  yes: boolean;
  /** `isAvailable('chat', judgeModel)` — or true when a client is injected. */
  available: boolean;
  /** Questions the reader will answer live (hypothesis unknown → assume `readerMaxTokens`). */
  live: ReadonlyArray<{ question: string; answer: string }>;
  readerMaxTokens: number;
  /** Prior rows to judge from their stored hypothesis. */
  backfill: ReadonlyArray<RowLike>;
}

export type JudgePreflightResult =
  | { ok: true; estUsd: number | null; ledger: BudgetLedger; lines: string[] }
  | { ok: false; exitCode: 1 | 2; message: string };

/**
 * Availability → pricing → estimate → cap. Exit 1 when the judge model has
 * no usable provider; exit 2 when a budget is set against an unpriced model
 * or the estimate exceeds the cap without `--yes` (eval-cross-modal pattern).
 */
export function judgePreflight(input: JudgePreflightInput): JudgePreflightResult {
  if (!input.available) {
    return {
      ok: false, exitCode: 1,
      message: `--judge: no usable chat provider for judge model ${input.judgeModel}. Set OPENAI_API_KEY (or pass --judge-model <provider:model> for a configured provider).`,
    };
  }
  const priced = isJudgeModelPriced(input.judgeModel);
  if (!priced && input.maxUsd !== null) {
    return {
      ok: false, exitCode: 2,
      message: `--judge: --max-usd requires a priced judge model; "${input.judgeModel}" has no CANONICAL_PRICING entry (src/core/model-pricing.ts). Pick a priced model or pass --max-usd off to accept unbounded spend.`,
    };
  }
  const items = [
    ...input.live.map(q => ({ question: q.question, answer: q.answer ?? '', hypothesisTokens: input.readerMaxTokens })),
    ...input.backfill.map(r => ({
      question: typeof r.question === 'string' ? r.question : '',
      answer: typeof r.answer === 'string' ? r.answer : '',
      hypothesisTokens: estimateTokens(typeof r.hypothesis === 'string' ? r.hypothesis : ''),
    })),
  ];
  const estUsd = priced ? estimateJudgeRunUsd(input.judgeModel, items) : null;
  const lines: string[] = [];
  lines.push(
    `[longmemeval] judge: ${input.judgeModel}, ${input.live.length} live + ${input.backfill.length} backfill question(s), ` +
    `estimated judge spend ${estUsd === null ? 'unpriced' : `~$${estUsd.toFixed(4)}`}` +
    `${input.maxUsd === null ? ' (no cap: --max-usd off)' : ` (cap $${input.maxUsd.toFixed(2)})`}`,
  );
  if (input.maxUsd !== null && estUsd !== null && estUsd > input.maxUsd && !input.yes) {
    return {
      ok: false, exitCode: 2,
      message: `--judge: estimated judge spend $${estUsd.toFixed(4)} exceeds --max-usd $${input.maxUsd.toFixed(2)}; pass --yes to proceed under the cap (the run soft-stops at the cap), raise --max-usd, or lower --limit.`,
    };
  }
  return { ok: true, estUsd, ledger: new BudgetLedger(input.maxUsd, estUsd), lines };
}

export interface BackfillResult {
  judged: number;
  errors: number;
  skipped: number;
}

/**
 * Judge prior rows from their stored hypothesis, `concurrency` at a time,
 * updating each row IN PLACE (prior judge_* keys stripped first so a
 * re-judge leaves no stale field behind). The dataset's answer is the
 * reference; the row's own `answer` is the fallback.
 */
export async function runJudgeBackfill(
  candidates: ReadonlyArray<RowLike>,
  ctx: JudgeLaneContext,
  opts: { concurrency: number; questionByQid: ReadonlyMap<string, LongMemEvalQuestion>; onRow?: (row: RowLike) => void },
): Promise<BackfillResult> {
  const result: BackfillResult = { judged: 0, errors: 0, skipped: 0 };
  await runWithLimit({
    items: candidates,
    limit: Math.max(1, opts.concurrency),
    fn: async (row) => {
      const q = opts.questionByQid.get(row.question_id as string);
      const input: JudgePromptInput & RowLike = {
        ...row,
        question_id: row.question_id as string,
        question_type: q?.question_type ?? (typeof row.question_type === 'string' ? row.question_type : 'unknown'),
        question: q?.question ?? (typeof row.question === 'string' ? row.question : ''),
        answer: q?.answer ?? (typeof row.answer === 'string' ? row.answer : ''),
        hypothesis: row.hypothesis as string,
      };
      const fields = await judgeRow(input, ctx);
      stripJudgeFields(row);
      Object.assign(row, fields);
      if (typeof fields.judge_correct === 'boolean') result.judged++;
      else if (fields.judge_error) result.errors++;
      else result.skipped++;
      opts.onRow?.(row);
    },
  });
  return result;
}
