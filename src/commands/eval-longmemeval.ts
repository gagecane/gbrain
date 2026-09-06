/**
 * `gbrain eval longmemeval <dataset.jsonl>` — public LongMemEval benchmark
 * adapter and the repo's receipt producer for the strict retrieval metric.
 * Spins up an in-memory PGLite, imports each question's haystack, runs
 * hybridSearch, optionally generates an answer via the configured gateway
 * chat model, emits one JSONL row per question (plus an optional schema-v2
 * `by_type_summary`) for downstream `evaluate_qa.py` / replay tooling.
 *
 * Hermetic by design: cli.ts skips connectEngine() when this subcommand is
 * invoked, so the user's ~/.gbrain brain is never opened. Tests stub the
 * ThinkLLMClient / expandFn / readiness / embed transport seams so the full
 * pipeline runs without any API key.
 *
 * INVARIANTS (ranker wave, Phase 0):
 *   - Recall joins on RAW dataset session ids through a per-question
 *     slug→raw map (metrics.ts); a collision touching a gold id aborts the
 *     question with an error row instead of scoring it.
 *   - `recall_all@k` / `recall_any@k` are scored over the DISTINCT sessions
 *     among the top-k CHUNK rows returned at `limit: k`. `_abs` questions are
 *     emitted but stay out of the recall denominators unless
 *     `--include-abstention`.
 *   - Every retrieval pin (mode, reranker, autocut, expansion, variant
 *     budget, embedder, top-k, trajectory) PLUS the resolved `knobs_hash`
 *     is hashed into `retrieval_config_hash` on every row; resume refuses a
 *     mixed file (a snapshot differing in any non-pin knob is a different run).
 *   - Silent degradation is a gate, not a footnote: on a non-keyword-only
 *     run a row whose vector arm fell back to keyword-only
 *     (`vector_enabled:false`, `embed_unavailable`, `embed_timeout`) or whose
 *     `--expansion` did not run as configured (`expansion_failed` /
 *     `expansion_partial`) is counted (`vector_degraded_rows`,
 *     `expansion_failed_rows`) and the run exits 1 — mirroring the reranker
 *     gate. The run-end gates + `--record` run through ONE `finishRun` from
 *     both the main path and the no-op resume path.
 *   - The embed-cache transaction wraps only the embed-producing section of a
 *     question (import + search); a reader/LLM failure afterwards never rolls
 *     back committed vectors and never holds the cache write lock across a
 *     network round-trip.
 *   - Flags live in ONE table (`LME_FLAGS`) that drives parseArgs AND
 *     printHelp, so the flag-registry scan sees every literal and help can
 *     never drift from the parser.
 *   - `--judge` (Phase D) judges the reader's answer with the official
 *     LongMemEval prompts (src/eval/longmemeval/judge.ts) through the shared
 *     judge runner; a judge malfunction is a `judge_error`, never an
 *     `incorrect`, and the headline scores it as incorrect. Every judged row
 *     carries `judge_config_hash` (judge + reader pins); `--judge
 *     --resume-from` is a judge-only backfill that re-judges rows lacking a
 *     settled verdict from their stored hypothesis and rebuilds `qa_accuracy`
 *     from ALL rows. Incomplete judgments (judge_errors / skipped_budget > 0)
 *     are not publishable → exit 1 unless --allow-incomplete-judgments.
 */

import { homedir } from 'os';
import { join } from 'path';
import { execSync } from 'child_process';
import type Anthropic from '@anthropic-ai/sdk';
import { withBenchmarkBrain, resetTables } from '../eval/longmemeval/harness.ts';
import { haystackToPages } from '../eval/longmemeval/adapter.ts';
import {
  READER_MAX_TOKENS,
  READER_PROMPT_SHA,
  generateAnswer,
  rawSessionId,
  renderRetrievedAsHypothesis,
} from '../eval/longmemeval/reader.ts';
import {
  DEFAULT_JUDGE_MODEL,
  JUDGE_METHODOLOGY_NOTE,
  JUDGE_PROMPT_VERSION,
  judgeRow,
  type JudgeLaneContext,
} from '../eval/longmemeval/judge.ts';
import {
  judgePreflight,
  makeJudgeConfigHasher,
  parseMaxUsd,
  runJudgeBackfill,
  selectBackfillRows,
  type RowLike,
} from '../eval/longmemeval/judge-lane.ts';
import { anyRowJudged, buildQaAccuracy, type QaAccuracyBlock } from '../eval/longmemeval/qa-accuracy.ts';
import { emitByTypeSummary, makeEmitter } from '../eval/longmemeval/emit.ts';
import type { BudgetLedger, JudgeChatFn } from '../eval/shared/judge-runner.ts';
import {
  addRowToBucket,
  buildByTypeSummaryV2,
  buildRow,
  buildSlugToRawMap,
  collisionsTouchingGold,
  detectSlugCollisions,
  newBucket,
  sessionIdFromSlug,
  type ByTypeSummaryContext,
  type ByTypeSummaryV2,
  type LongMemEvalRow,
  type RecallBucket,
  type SlugToRawMap,
} from '../eval/longmemeval/metrics.ts';
import {
  buildRunConfig,
  loadDataset,
  loadQuestionIds,
  redactSecrets,
  retrievalConfigHash,
  type CacheReceipt,
  type DatasetQuestion,
  type RetrievalPins,
} from '../eval/longmemeval/run-config.ts';
import {
  checkResumeConfigHash,
  classifyDegradation,
  countDegradation,
  loadExpansionReplay,
  loadResumeSet,
  readJsonlRows,
  seedBucketsFromRows,
  type RowSearchMeta,
} from '../eval/longmemeval/resume.ts';
import { EmbeddingCache, installEmbedCache, type EmbedTransportFn, type InstalledEmbedCache } from '../eval/shared/embed-cache.ts';
import { importFromContent } from '../core/import-file.ts';
import { hybridSearch, type HybridSearchOpts } from '../core/search/hybrid.ts';
import { expandQuery } from '../core/search/expansion.ts';
import {
  KNOBS_HASH_VERSION,
  knobsHash,
  loadOverridesFromConfig,
  resolveSearchMode,
  type ResolvedSearchKnobs,
  type SearchMode,
} from '../core/search/mode.ts';
import { estimateTokens } from '../core/search/token-budget.ts';
import { resolveModel } from '../core/model-config.ts';
import type { ThinkLLMClient } from '../core/think/index.ts';
import { createProgress } from '../core/progress.ts';
import { getCliOptions, cliOptsToProgressOptions } from '../core/cli-options.ts';
import type { PGLiteEngine } from '../core/pglite-engine.ts';
import type { HybridSearchMeta, SearchResult } from '../core/types.ts';
import { classifyIntent, type Intent } from '../eval/longmemeval/intent.ts';
import {
  extractAndInsertClaims,
  makeAliasMap,
  resetExtractorState,
  getCacheStats,
  type AliasMap,
} from '../eval/longmemeval/extract.ts';
import { extractCandidateEntities } from '../core/think/entity-extract.ts';
import { normalizeModelId } from '../core/model-id.ts';
import { chat as gatewayChat, getEmbeddingDimensions, getEmbeddingModel, isAvailable } from '../core/ai/gateway.ts';
import { rerankerReadinessForEngine, type EngineReadiness } from '../core/ai/reranker-readiness-engine.ts';
import { describeRerankerFix } from '../core/ai/reranker-readiness.ts';
import { resolveEntitySlugWithSource, type ResolutionSource } from '../core/entities/resolve.ts';
import { formatTrajectoryBlock } from '../core/trajectory-format.ts';
import { persistRunRecord, type EvalRunRecord } from './eval-run-all.ts';

// Back-compat re-exports (these used to live here; tests + consumers import from the harness).
export { loadResumeSet } from '../eval/longmemeval/resume.ts';
export { emitByTypeSummary } from '../eval/longmemeval/emit.ts';

/**
 * v0.40.2.0 — methodology disclosure marker. Stamped on every row when
 * trajectory routing is enabled so downstream readers see the preprocessing
 * step is in the pipeline ("gbrain + Haiku-preprocess" vs "gbrain alone").
 */
const TRAJECTORY_METHODOLOGY_NOTE = 'extractor=haiku-preprocess-full-haystack-v1';

const HUGGINGFACE_URL = 'https://huggingface.co/datasets/xiaowu0162/longmemeval';

const DEFAULT_EMBED_CACHE_PATH = join(homedir(), '.cache', 'gbrain-eval', 'longmemeval-embed.sqlite');

const SEARCH_MODES_LIST: readonly SearchMode[] = ['conservative', 'balanced', 'tokenmax'];

type FloorMetric = 'recall_all' | 'recall_any';

interface ParsedArgs {
  help: boolean;
  datasetPath?: string;
  limit?: number;
  model?: string;
  retrievalOnly: boolean;
  keywordOnly: boolean;
  expansion: boolean;
  /** `--expansion-replay FILE`: serve recorded `expansion_variants` per question_id. Implies --expansion. */
  expansionReplayPath?: string;
  /** undefined = not pinned (config/bundle decides); null = legacy weighting. */
  expansionVariantBudget?: number | null;
  /** --search-pin key=value (repeatable): extra `search.*` config pins written before the run and folded into the knobs hash. */
  searchPins?: Record<string, string>;
  topK: number;
  outputPath?: string;
  mode?: SearchMode;
  /** `--reranker on|off` → `search.reranker.enabled` pin. undefined = not pinned. */
  reranker?: boolean;
  /** `--autocut on|off` → `search.autocut` pin. undefined = not pinned. */
  autocut?: boolean;
  resumeFromPath?: string;
  allowMixedRunConfig: boolean;
  questionIdsPath?: string;
  noTrajectory: boolean;
  byType: boolean;
  byTypeFloor?: number;
  byTypeFloorMetric: FloorMetric;
  includeAbstention: boolean;
  embedCache: boolean;
  embedCachePath: string;
  capturePool: boolean;
  record: boolean;
  /** `--judge`: LLM-judge each answer (implies --by-type: qa_accuracy lands on the summary line). */
  judge: boolean;
  judgeModel: string;
  /** Judge spend cap in USD; null = `--max-usd off`. */
  maxUsd: number | null;
  yes: boolean;
  judgeConcurrency: number;
  allowIncompleteJudgments: boolean;
}

interface LmeFlag {
  name: string;
  /** Value placeholder for help (`N`, `FILE`, `on|off`); absent = boolean flag. */
  arg?: string;
  help: string[];
  apply: (out: ParsedArgs, value: string) => void;
}

function parseOnOff(flag: string, v: string): boolean {
  const s = v.trim().toLowerCase();
  if (s === 'on' || s === 'true' || s === '1') return true;
  if (s === 'off' || s === 'false' || s === '0') return false;
  throw new Error(`${flag} must be on|off (got: ${v})`);
}

/**
 * The ONE flag table (eng D4): renders `printHelp` AND drives `parseArgs`.
 * Every `--flag` literal the command accepts appears here, so the
 * flag-registry generator (which scans this file's text) sees the full
 * surface and the help text can never drift from the parser.
 */
const LME_FLAGS: LmeFlag[] = [
  { name: '--limit', arg: 'N', help: ['Run only the first N questions (after --question-ids filtering).'],
    apply: (o, v) => { o.limit = Number(v); if (!Number.isInteger(o.limit) || o.limit < 1) throw new Error(`--limit must be a positive integer (got: ${v})`); } },
  { name: '--model', arg: 'M', help: ['Override answer-generation model (default: resolveModel).'],
    apply: (o, v) => { o.model = v; } },
  { name: '--retrieval-only', help: ['Skip LLM answer generation; emit retrieved sessions instead.'],
    apply: (o) => { o.retrievalOnly = true; } },
  { name: '--keyword-only', help: ['Skip vector embedding; pure keyword retrieval (no reranker, no cache).'],
    apply: (o) => { o.keywordOnly = true; } },
  { name: '--expansion', help: [
      'Enable multi-query expansion. OFF by default for EVERY mode — the per-call',
      'setting wins over the bundle, so `--mode tokenmax` alone does NOT expand;',
      'expansion fires only with this flag. One Haiku call per question, non-',
      'deterministic; each row records `expansion_variants` for replay.'],
    apply: (o) => { o.expansion = true; } },
  { name: '--expansion-replay', arg: 'FILE', help: [
      'Serve the `expansion_variants` recorded in FILE (a prior --expansion run)',
      'instead of calling the LLM, so every cell differs only in its knobs.',
      'Implies --expansion. A question_id missing from FILE is an error row',
      '(`expansion_replay_miss`) and the run exits 1 at the end.'],
    apply: (o, v) => { o.expansionReplayPath = v; o.expansion = true; } },
  { name: '--search-pin', arg: 'KEY=VALUE', help: [
      'Pin an arbitrary `search.*` config key for the run (repeatable; last wins).',
      'Generic pass-through for knob A/Bs (e.g. search.metadata_boost_gate=lexical);',
      'the value is written via engine.setConfig and folds into the knobs hash.'],
    apply: (o, v) => {
      const eq = v.indexOf('=');
      if (eq <= 0) throw new Error(`--search-pin takes KEY=VALUE (got: ${v})`);
      const key = v.slice(0, eq).trim(); const val = v.slice(eq + 1).trim();
      if (!key.startsWith('search.') || key === 'search.' || val.length === 0) throw new Error(`--search-pin key must start with "search." and carry a non-empty value (got: ${v})`);
      o.searchPins = { ...(o.searchPins ?? {}), [key]: val };
    } },
  { name: '--expansion-variant-budget', arg: 'B', help: [
      'Pin `search.expansion_variant_budget`: `legacy` (every RRF list weight 1)',
      'or a number in (0, 4] — the total RRF weight shared equally by all',
      'expansion variant lists (the original list always keeps weight 1).'],
    apply: (o, v) => {
      const s = v.trim().toLowerCase();
      if (s === 'legacy' || s === 'null') { o.expansionVariantBudget = null; return; }
      const n = Number(s);
      if (!Number.isFinite(n) || n <= 0 || n > 4) throw new Error(`--expansion-variant-budget must be legacy or a number in (0, 4] (got: ${v})`);
      o.expansionVariantBudget = n;
    } },
  { name: '--top-k', arg: 'K', help: [
      'Retrieve K chunk rows per question (default: 8). recall_*@k is scored over',
      'the distinct sessions among those K rows (k = K).'],
    apply: (o, v) => { o.topK = Number(v); if (!Number.isInteger(o.topK) || o.topK < 1) throw new Error(`--top-k must be a positive integer (got: ${v})`); } },
  { name: '--mode', arg: 'M', help: [
      'Search mode: conservative|balanced|tokenmax. Resolves through',
      'src/core/search/mode.ts so retrieval matches production under that mode.',
      'NOTE: no mode implies --expansion (see --expansion).'],
    apply: (o, v) => {
      if (!(SEARCH_MODES_LIST as readonly string[]).includes(v)) throw new Error(`--mode must be one of conservative|balanced|tokenmax (got: ${v})`);
      o.mode = v as SearchMode;
    } },
  { name: '--reranker', arg: 'on|off', help: [
      'Pin `search.reranker.enabled` for the run (beats any injected snapshot).',
      'With `on`: readiness preflight (exit 2 with the fix if the reranker cannot',
      'run) and the run exits non-zero if any row fell through un-reranked',
      '(`reranker_skipped_rows` in the summary).'],
    apply: (o, v) => { o.reranker = parseOnOff('--reranker', v); } },
  { name: '--autocut', arg: 'on|off', help: ['Pin `search.autocut` for the run (beats any injected snapshot).'],
    apply: (o, v) => { o.autocut = parseOnOff('--autocut', v); } },
  { name: '--output', arg: 'FILE', help: ['Write JSONL to FILE instead of stdout.'],
    apply: (o, v) => { o.outputPath = v; } },
  { name: '--resume-from', arg: 'FILE', help: [
      'Skip question_ids already present in FILE; resume the remaining',
      'questions. Typically the same path as --output so the run continues in',
      'append mode. Rows are re-scored from their retrieved ids + the dataset',
      'gold; a file written under different retrieval pins is refused.'],
    apply: (o, v) => { o.resumeFromPath = v; } },
  { name: '--allow-mixed-run-config', help: ['Resume even when FILE rows carry a different retrieval_config_hash.'],
    apply: (o) => { o.allowMixedRunConfig = true; } },
  { name: '--question-ids', arg: 'FILE', help: [
      'Run only the question_ids listed in FILE (one per line, # comments).',
      'Unknown ids or an empty file exit 1. Dev-slice / held-out discipline.'],
    apply: (o, v) => { o.questionIdsPath = v; } },
  { name: '--no-trajectory', help: [
      'Opt out of trajectory routing (skips the Haiku claim extractor AND the',
      'per-question intent routing). Use for like-for-like retrieval receipts.'],
    apply: (o) => { o.noTrajectory = true; } },
  { name: '--by-type', help: [
      'Emit a final schema_version:2 by_type_summary line: per question_type',
      '{total, all_hit, all_rate, any_hit, any_rate} + aggregate, excluded_abstention,',
      'mean_distinct_sessions, run_config (pins, hashes, cache stats). Resume-safe:',
      'a prior summary at the tail is REPLACED, not appended.'],
    apply: (o) => { o.byType = true; } },
  { name: '--by-type-floor', arg: 'F', help: [
      'Exit non-zero if any question_type rate < F (range [0, 1]). Gates on',
      'recall_all by default (see --by-type-floor-metric). Implies --by-type.'],
    apply: (o, v) => {
      const f = Number(v);
      if (!Number.isFinite(f) || f < 0 || f > 1) throw new Error(`--by-type-floor must be a number in [0, 1] (got: ${v})`);
      o.byTypeFloor = f; o.byType = true;
    } },
  { name: '--by-type-floor-metric', arg: 'M', help: ['Which rate --by-type-floor gates on: recall_all (default) | recall_any.'],
    apply: (o, v) => {
      if (v !== 'recall_all' && v !== 'recall_any') throw new Error(`--by-type-floor-metric must be recall_all|recall_any (got: ${v})`);
      o.byTypeFloorMetric = v;
    } },
  { name: '--include-abstention', help: [
      'Count `_abs` (abstention) questions in the recall denominators. Default:',
      'emitted with abstention:true but excluded (summary.excluded_abstention).'],
    apply: (o) => { o.includeAbstention = true; } },
  { name: '--embed-cache', arg: 'FILE', help: [
      `Content-addressed embedding cache (bun:sqlite). Default:`,
      `~/.cache/gbrain-eval/longmemeval-embed.sqlite. Hits/misses land in`,
      `run_config.cache; misses must be 0 for a like-for-like arm.`],
    apply: (o, v) => { o.embedCachePath = v; o.embedCache = true; } },
  { name: '--no-embed-cache', help: ['Disable the embedding cache for this run.'],
    apply: (o) => { o.embedCache = false; } },
  { name: '--capture-pool', help: [
      'Record `rerank_pool` per row: the post-rerank candidate pool BEFORE',
      'autocut / the limit slice ({slug, chunk_id, session_id, rrf_rank,',
      'rerank_score, alias_hit, est_tokens}) for the autocut-floor replay.'],
    apply: (o) => { o.capturePool = true; } },
  { name: '--record', help: [
      'Append an EvalRunRecord (suite longmemeval, params = run_config) to',
      '.gbrain-evals/eval-results.jsonl. Error text is secret-redacted.'],
    apply: (o) => { o.record = true; } },
  { name: '--judge', help: [
      'LLM-judge each answer against the gold with the official LongMemEval',
      'evaluate_qa.py prompts (temperature 0, max_tokens 10). Implies --by-type:',
      'the summary line gains a qa_accuracy block whose headline scores judge',
      'errors as incorrect. Incompatible with --retrieval-only. With',
      '--resume-from FILE: judge-only backfill of rows lacking a settled verdict',
      '(no reader call; rows with judge_error are re-judged), then qa_accuracy',
      'is rebuilt from ALL rows and FILE is rewritten with the judged rows.'],
    apply: (o) => { o.judge = true; o.byType = true; } },
  { name: '--judge-model', arg: 'M', help: [`Judge model (default: ${DEFAULT_JUDGE_MODEL}, the official scorer's model).`],
    apply: (o, v) => { o.judgeModel = normalizeModelId(v, 'openai'); } },
  { name: '--max-usd', arg: 'N|off', help: [
      'Cap on JUDGE spend only, in USD (default: 5) — the reader / extractor',
      'lanes are not metered here. Preflight refuses an estimate over the cap',
      'without --yes (exit 2); at run time the lane soft-stops at the cap and',
      'stamps the remaining rows judge_skipped:"budget" (not publishable). An',
      'unpriced judge model requires `off` (no cap).'],
    apply: (o, v) => { o.maxUsd = parseMaxUsd('--max-usd', v); } },
  { name: '--yes', help: ['Proceed when the judge estimate exceeds --max-usd (the cap still soft-stops the run).'],
    apply: (o) => { o.yes = true; } },
  { name: '--judge-concurrency', arg: 'N', help: [
      'Parallel judge calls during a --resume-from backfill (default: 1; live',
      'rows are judged inline after each reader call).'],
    apply: (o, v) => { o.judgeConcurrency = Number(v); if (!Number.isInteger(o.judgeConcurrency) || o.judgeConcurrency < 1) throw new Error(`--judge-concurrency must be a positive integer (got: ${v})`); } },
  { name: '--allow-incomplete-judgments', help: [
      'Exit 0 even when judge_errors > 0 or skipped_budget > 0. Default: such a',
      'run is NOT publishable (stderr FAIL line + exit 1) — re-run with --judge',
      '--resume-from FILE until both are 0.'],
    apply: (o) => { o.allowIncompleteJudgments = true; } },
];

function parseArgs(args: string[]): ParsedArgs {
  const out: ParsedArgs = {
    help: false,
    retrievalOnly: false,
    keywordOnly: false,
    expansion: false,
    topK: 8,
    allowMixedRunConfig: false,
    noTrajectory: false,
    byType: false,
    byTypeFloorMetric: 'recall_all',
    includeAbstention: false,
    embedCache: true,
    embedCachePath: DEFAULT_EMBED_CACHE_PATH,
    capturePool: false,
    record: false,
    judge: false,
    judgeModel: DEFAULT_JUDGE_MODEL,
    maxUsd: 5,
    yes: false,
    judgeConcurrency: 1,
    allowIncompleteJudgments: false,
  };
  const byName = new Map(LME_FLAGS.map(f => [f.name, f]));
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--help' || a === '-h') { out.help = true; continue; }
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      const name = eq >= 0 ? a.slice(0, eq) : a;
      const flag = byName.get(name);
      if (!flag) throw new Error(`unknown flag ${name} for 'gbrain eval longmemeval' (see --help)`);
      let value = '';
      if (flag.arg !== undefined) {
        value = eq >= 0 ? a.slice(eq + 1) : (args[++i] ?? '');
        if (value === '' || (eq < 0 && value.startsWith('--'))) throw new Error(`${name} requires a value (${flag.arg})`);
      }
      flag.apply(out, value);
      continue;
    }
    if (!out.datasetPath) { out.datasetPath = a; continue; }
  }
  if (out.judge && out.retrievalOnly) {
    throw new Error('--judge cannot be combined with --retrieval-only (there is no reader hypothesis to judge)');
  }
  return out;
}

function printHelp(): void {
  const lines: string[] = [
    `gbrain eval longmemeval <dataset.jsonl> [options]`,
    ``,
    `Run the LongMemEval benchmark against gbrain's hybrid retrieval. Spins up an`,
    `in-memory PGLite per benchmark run; the user's brain is never opened.`,
    ``,
    `Arguments:`,
    `  <dataset.jsonl>           LongMemEval dataset file (one question per line).`,
    `                            Download from ${HUGGINGFACE_URL}`,
    ``,
    `Options:`,
  ];
  const pad = 28;
  for (const f of LME_FLAGS) {
    const head = `  ${f.name}${f.arg ? ' ' + f.arg : ''}`;
    if (head.length >= pad - 1) {
      lines.push(head);
      for (const h of f.help) lines.push(' '.repeat(pad) + h);
    } else {
      lines.push(head.padEnd(pad) + f.help[0]);
      for (const h of f.help.slice(1)) lines.push(' '.repeat(pad) + h);
    }
  }
  lines.push(`  -h, --help                Show this help.`);
  lines.push(``);
  lines.push(`Row fields: recall_all_hit (every gold session in the top-k distinct sessions),`);
  lines.push(`recall_any_hit (at least one), recall_hit (DEPRECATED alias of recall_any_hit),`);
  lines.push(`abstention, distinct_sessions_in_top_k, retrieved[] (every returned chunk row),`);
  lines.push(`retrieved_session_ids, search_meta, retrieval_config_hash, reader_model, reader_prompt_sha;`);
  lines.push(`with --judge: judge_correct | judge_error | judge_skipped, judge_model, judge_raw, judge_cost_usd,`);
  lines.push(`judge_config_hash (summary: qa_accuracy).`);
  lines.push(``);
  lines.push(`Note: a full 500-question run takes ~20-60 minutes depending on flags. Use`);
  lines.push(`--limit or --question-ids during development.`);
  process.stderr.write(lines.join('\n') + '\n');
}

export interface RunOpts {
  /** Inject a chat client for tests; defaults to the gateway-routed client (#4636). */
  client?: ThinkLLMClient;
  /** Separate stub for the Haiku claim extractor (defaults to the same gateway client). */
  extractorClient?: ThinkLLMClient;
  /** Model id for the extractor's Haiku call. Defaults to a tier-utility model via resolveModel. */
  extractorModel?: string;
  /**
   * Inject a pre-built benchmark brain instead of creating one inside this
   * call (tests amortize the PGLite cold-create across many runs). MUST be
   * the result of createBenchmarkBrain(); the caller owns lifecycle.
   */
  engine?: PGLiteEngine;
  /**
   * Live nightly-probe search-mode/reranker settings copied into the isolated
   * benchmark brain (#3676). Explicit pin flags (--mode/--reranker/--autocut/
   * --expansion-variant-budget) beat the snapshot.
   */
  searchConfigSnapshot?: Record<string, string>;
  /** Test seam: the expansion function behind `--expansion` (default expandQuery). */
  expandFn?: (query: string) => Promise<string[]>;
  /**
   * Test seam: the transport the embed cache serves misses from and restores
   * on uninstall (default: the real ai-sdk embedMany). A test that installed
   * a fake transport must pass it here or the cache install would replace it.
   */
  embedTransport?: EmbedTransportFn | null;
  /** Test seam: reranker readiness probe behind `--reranker on` (default rerankerReadinessForEngine). */
  rerankerReadiness?: (engine: PGLiteEngine, model: string) => Promise<EngineReadiness>;
  /** Test seam: directory for the `--record` ledger (default <repo>/.gbrain-evals/). */
  recordDir?: string;
  /** Test seam: the judge chat client behind `--judge` (default: the gateway `chat`). */
  judgeClient?: JudgeChatFn;
  /** Test seam: base retry backoff for judge timeouts / 429s (default 500ms). */
  judgeBackoffMs?: number;
}

/** Abort one question with an error row that carries extra diagnostic fields. */
class QuestionAbort extends Error {
  constructor(message: string, readonly extra: Record<string, unknown>) {
    super(message);
    this.name = 'QuestionAbort';
  }
}

interface RunContext {
  opts: ParsedArgs;
  model: string;
  client: ThinkLLMClient;
  trajectoryEnabled: boolean;
  extractorClient: ThinkLLMClient;
  extractorModel: string;
  expandFn: (query: string) => Promise<string[]>;
  replay: Map<string, string[]> | null;
  retrievalConfigHash: string;
  /**
   * Wraps the embed-producing section of one question (import + search) in
   * ONE embed-cache transaction (eng D7) — and nothing after it, so a reader
   * failure cannot roll back committed vectors. Identity when no cache.
   */
  embedTxn: <T>(fn: () => Promise<T>) => Promise<T>;
}

interface QuestionOutcome {
  row: LongMemEvalRow;
  rerankerSkipped: boolean;
  vectorDegraded: boolean;
  expansionFailed: boolean;
}

/** Resolve the pins for this run from flags + the injected snapshot (no engine read). */
function resolvePins(opts: ParsedArgs, runOpts: RunOpts, trajectoryEnabled: boolean): { pins: RetrievalPins; knobs: ResolvedSearchKnobs } {
  // Generic --search-pin entries ride the same override plane as the injected snapshot (explicit pins win),
  // so knobs_hash / retrieval_config_hash reflect them.
  const snapshot = { ...(runOpts.searchConfigSnapshot ?? {}), ...(opts.searchPins ?? {}) };
  const knobs = resolveSearchMode({
    mode: opts.mode ?? snapshot['search.mode'],
    overrides: loadOverridesFromConfig(snapshot),
    perCall: {
      expansion: opts.expansion,
      ...(opts.expansionVariantBudget !== undefined ? { expansion_variant_budget: opts.expansionVariantBudget } : {}),
      ...(opts.reranker !== undefined ? { reranker_enabled: opts.reranker } : {}),
      ...(opts.autocut !== undefined ? { autocut: opts.autocut } : {}),
    },
  });
  let embedder = 'unconfigured';
  try { embedder = `${getEmbeddingModel()}@${getEmbeddingDimensions()}`; } catch { /* no gateway configured */ }
  const pins: RetrievalPins = {
    mode: knobs.resolved_mode,
    keyword_only: opts.keywordOnly,
    reranker: { enabled: opts.keywordOnly ? false : knobs.reranker_enabled, model: knobs.reranker_model },
    autocut: opts.keywordOnly ? false : knobs.autocut,
    expansion: opts.keywordOnly ? false : opts.expansion,
    expansion_variant_budget: knobs.expansion_variant_budget,
    embedder,
    top_k: opts.topK,
    trajectory: trajectoryEnabled,
  };
  return { pins, knobs };
}

function floorBreaches(summary: ByTypeSummaryV2, floor: number, metric: FloorMetric): string[] {
  const breaches: string[] = [];
  for (const [t, v] of Object.entries(summary.recall_by_type)) {
    const rate = metric === 'recall_all' ? v.all_rate : v.any_rate;
    // null = empty bucket; JS `null < F` is true, so guard explicitly.
    if (rate !== null && rate < floor) breaches.push(`${t}: ${metric} ${(rate * 100).toFixed(1)}% < ${(floor * 100).toFixed(1)}%`);
  }
  return breaches;
}

function gitShort(cmd: string, fallback: string): string {
  try { return execSync(cmd, { encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'] }).trim() || fallback; } catch { return fallback; }
}

export async function runEvalLongMemEval(args: string[], runOpts: RunOpts = {}): Promise<void> {
  let opts: ParsedArgs;
  try {
    opts = parseArgs(args);
  } catch (err: any) {
    process.stderr.write(`Error: ${err.message ?? err}\n`);
    process.exit(1);
    return;
  }
  if (opts.help) { printHelp(); return; }
  if (!opts.datasetPath) {
    process.stderr.write(`Error: <dataset.jsonl> is required.\n\n`);
    printHelp();
    process.exit(1);
    return;
  }

  let questions: DatasetQuestion[];
  let datasetSha256: string;
  try {
    ({ questions, sha256: datasetSha256 } = loadDataset(opts.datasetPath, HUGGINGFACE_URL));
  } catch (err: any) {
    process.stderr.write(`Error: ${err.message ?? err}\n`);
    process.exit(1);
    return;
  }
  const datasetQuestionCount = questions.length;
  // D12: duplicate question_ids → WARN + dedupe (first occurrence wins).
  {
    const seen = new Set<string>();
    const deduped: DatasetQuestion[] = [];
    for (const q of questions) {
      if (seen.has(q.question_id)) { process.stderr.write(`[longmemeval] WARN duplicate question_id ${q.question_id} — keeping the first\n`); continue; }
      seen.add(q.question_id);
      deduped.push(q);
    }
    questions = deduped;
  }
  // Gold by question_id for resume re-scoring (dataset is loaded on resume).
  const goldByQid = new Map<string, readonly string[]>(questions.map(q => [q.question_id, q.answer_session_ids ?? []]));
  /** Dataset row by id — the judge backfill takes question_type / answer from the dataset, not the row. */
  const questionByQid = new Map<string, DatasetQuestion>(questions.map(q => [q.question_id, q]));

  if (opts.questionIdsPath) {
    try {
      const ids = loadQuestionIds(opts.questionIdsPath);
      const known = new Set(questions.map(q => q.question_id));
      const unknown = ids.filter(id => !known.has(id));
      if (unknown.length > 0) throw new Error(`--question-ids: ${unknown.length} id(s) not in the dataset: ${unknown.slice(0, 5).join(', ')}${unknown.length > 5 ? ', …' : ''}`);
      const want = new Set(ids);
      questions = questions.filter(q => want.has(q.question_id));
    } catch (err: any) {
      process.stderr.write(`Error: ${err.message ?? err}\n`);
      process.exit(1);
      return;
    }
  }
  if (opts.limit && opts.limit < questions.length) {
    questions = questions.slice(0, opts.limit);
  }
  if (questions.length === 0) {
    process.stderr.write(`Error: dataset contains no questions.\n`);
    process.exit(1);
    return;
  }

  const trajectoryEnabled = !opts.noTrajectory;
  const { pins, knobs } = resolvePins(opts, runOpts, trajectoryEnabled);
  const knobsHashValue = knobsHash(knobs);
  // D33 + review: the hash covers the pins AND the resolved knobs hash, so a
  // resume cannot merge runs whose injected snapshot differs in a non-pin knob.
  const retrievalHash = retrievalConfigHash(pins, { knobs_hash: knobsHashValue, knobs_hash_version: KNOBS_HASH_VERSION });
  const degradeOpts = { keywordOnly: opts.keywordOnly, expansion: opts.expansion };

  // Resolved BEFORE the resume block: the reader model is half of every
  // row's judge_config_hash (D33), which the --judge backfill gate needs.
  const model = await resolveModel(null, {
    cliFlag: opts.model,
    configKey: 'models.eval.longmemeval',
    envVar: 'GBRAIN_MODEL',
    fallback: 'sonnet',
  });
  const judgeHashFor = makeJudgeConfigHasher(opts.judgeModel, { model, prompt_sha: READER_PROMPT_SHA, max_tokens: READER_MAX_TOKENS, k: opts.topK });
  /** This run's judge_config_hash (live rows; prior rows hash from their own recorded reader pins). */
  const runJudgeHash = judgeHashFor({});

  /** Everything the run-end block needs; filled by the main path or the no-op resume path. */
  interface RunEndState {
    buckets: Record<string, RecallBucket>;
    distinct: number[];
    excludedAbstention: number;
    goldMissing: number;
    slugCollisions: number;
    rerankerSkippedRows: number;
    vectorDegradedRows: number;
    expansionFailedRows: number;
    expansionReplayMiss: number;
    errorCount: number;
    errorMessages: string[];
    cacheReceipt: CacheReceipt | null;
    cacheSkipped?: string;
    questionsRun: number;
    runStart: number;
    /** Every question row of the run (prior rows, rewritten on a judge backfill, + new rows) for qa_accuracy. */
    qaRows: RowLike[];
    judgeEstUsd: number | null;
    judgeLedger: BudgetLedger | null;
  }

  const summaryRunConfig = (st: RunEndState): Record<string, unknown> => buildRunConfig({
    pins,
    retrieval_config_hash: retrievalHash,
    dataset_sha256: datasetSha256,
    dataset_questions: datasetQuestionCount,
    knobs_hash: knobsHashValue,
    knobs_hash_version: KNOBS_HASH_VERSION,
    cache: st.cacheReceipt,
    cache_skipped: st.cacheSkipped,
    reranker_skipped_rows: st.rerankerSkippedRows,
    vector_degraded_rows: st.vectorDegradedRows,
    expansion_failed_rows: st.expansionFailedRows,
    expansion_replay_miss: st.expansionReplayMiss,
    expansion_replay: opts.expansionReplayPath ?? null,
    gold_missing_from_haystack: st.goldMissing,
    slug_collisions: st.slugCollisions,
    excluded_abstention: st.excludedAbstention,
    question_ids_file: opts.questionIdsPath ?? null,
    errors: st.errorCount,
  });

  /**
   * The ONE run-end block: stderr summary, --by-type emission + floor gate,
   * the reranker / vector-degraded / expansion-failed / replay-miss gates,
   * --record, exit. Called from the no-op resume branch AND the main path so
   * a resume that has nothing left to do still records and still fails on
   * the gates its prior rows breach.
   */
  const finishRun = (st: RunEndState): void => {
    let exitCode = 0;
    const elapsed = Math.round((Date.now() - st.runStart) / 1000);
    process.stderr.write(`\n[longmemeval] done. ${st.questionsRun} questions in ${elapsed}s. ${st.errorCount} errors.\n`);
    if (Object.keys(st.buckets).length > 0) {
      process.stderr.write(`[longmemeval] recall_all@${opts.topK} / recall_any@${opts.topK} by question_type:\n`);
      for (const [t, v] of Object.entries(st.buckets).sort()) {
        const pa = v.total === 0 ? 0 : (v.all_hit / v.total) * 100;
        const pn = v.total === 0 ? 0 : (v.any_hit / v.total) * 100;
        process.stderr.write(`  ${t}: all ${v.all_hit}/${v.total} (${pa.toFixed(1)}%), any ${v.any_hit}/${v.total} (${pn.toFixed(1)}%)\n`);
      }
      if (st.excludedAbstention > 0) process.stderr.write(`  (excluded ${st.excludedAbstention} abstention question(s); --include-abstention counts them)\n`);
    }
    if (st.cacheReceipt) {
      const c = st.cacheReceipt;
      process.stderr.write(`[longmemeval] embed cache: ${c.hits} hits, ${c.misses} misses, ${c.bypassed} bypassed, ${c.infra_faults} infra fault(s) (canonical ${c.canonical_sha256.slice(0, 12)})\n`);
    }

    const runConfig = summaryRunConfig(st);
    // qa_accuracy (Phase D): rebuilt from ALL rows whenever the judge lane ran
    // or any row carries a verdict. Headline scores judge errors / budget
    // skips / reader errors as incorrect (D16); errors-excluded is secondary.
    let qa: QaAccuracyBlock | undefined;
    if (opts.judge || anyRowJudged(st.qaRows)) {
      qa = buildQaAccuracy(st.qaRows, {
        judgeModel: opts.judgeModel,
        judgePromptVersion: JUDGE_PROMPT_VERSION,
        judgeConfigHash: runJudgeHash,
        estCostUsd: st.judgeEstUsd,
        runCostUsd: st.judgeLedger ? st.judgeLedger.actualUsd : null,
        methodologyNote: JUDGE_METHODOLOGY_NOTE,
      });
      const pct = (x: number | null): string => (x === null ? 'n/a' : `${(x * 100).toFixed(1)}%`);
      const ci = qa.ci95_bootstrap;
      process.stderr.write(
        `[longmemeval] qa_accuracy: headline ${pct(qa.accuracy_headline)} (${qa.correct}/${qa.total_questions}; ` +
        `judge_errors ${qa.judge_errors}, skipped_budget ${qa.skipped_budget}, reader_errors ${qa.reader_errors}), ` +
        `excluding errors ${pct(qa.accuracy_excluding_errors)} (${qa.correct}/${qa.judged}), ` +
        `non-abstention ${pct(qa.accuracy_470)} (n=${qa.non_abstention_total}), abstention ${qa.abstention.correct}/${qa.abstention.total}; ` +
        `ci95 [${pct(ci.lower)}, ${pct(ci.upper)}] (${ci.label}); judge ${qa.judge_model} v=${qa.judge_prompt_version}; ` +
        `spend est ${qa.est_cost_usd === null ? 'unpriced' : `$${qa.est_cost_usd.toFixed(4)}`}, ` +
        `this run $${(qa.run_cost_usd ?? 0).toFixed(4)}, all rows $${qa.actual_cost_usd.toFixed(4)}\n`,
      );
      if (opts.judge && !qa.complete) {
        const line = `[longmemeval] ${opts.allowIncompleteJudgments ? 'WARN' : 'FAIL'} --judge: judgments incomplete ` +
          `(judge_errors ${qa.judge_errors}, skipped_budget ${qa.skipped_budget}, unjudged ${qa.unjudged}) — NOT publishable; re-run with ` +
          `--judge --resume-from FILE until all three are 0`;
        process.stderr.write(opts.allowIncompleteJudgments ? `${line} (continuing: --allow-incomplete-judgments)\n` : `${line}.\n`);
        if (!opts.allowIncompleteJudgments) exitCode = 1;
      }
    }
    let summary: ByTypeSummaryV2 | undefined;
    if (opts.byType) {
      summary = buildByTypeSummaryV2(st.buckets, {
        k: opts.topK,
        excludedAbstention: st.excludedAbstention,
        goldMissingFromHaystack: st.goldMissing,
        slugCollisions: st.slugCollisions,
        distinctSessionsInTopK: st.distinct,
        runConfig,
        ...(qa ? { qa } : {}),
      });
      emitByTypeSummary(opts.outputPath, summary);
      if (opts.byTypeFloor !== undefined) {
        const breaches = floorBreaches(summary, opts.byTypeFloor, opts.byTypeFloorMetric);
        if (breaches.length > 0) {
          process.stderr.write(`[longmemeval] FAIL --by-type-floor=${opts.byTypeFloor}: ${breaches.join(', ')}\n`);
          exitCode = 1;
        }
      }
    }
    if (opts.reranker === true && st.rerankerSkippedRows > 0) {
      process.stderr.write(`[longmemeval] FAIL --reranker on: ${st.rerankerSkippedRows} row(s) fell through un-reranked (reranker_skipped / rerank_passthrough)\n`);
      exitCode = 1;
    }
    if (!opts.keywordOnly && st.vectorDegradedRows > 0) {
      process.stderr.write(`[longmemeval] FAIL vector arm: ${st.vectorDegradedRows} row(s) scored keyword-only after a silent embed failure (vector_enabled:false / embed_unavailable / embed_timeout) — vector_degraded_rows\n`);
      exitCode = 1;
    }
    if (!opts.keywordOnly && opts.expansion && st.expansionFailedRows > 0) {
      process.stderr.write(`[longmemeval] FAIL --expansion: ${st.expansionFailedRows} row(s) did not expand as configured (expansion_failed / expansion_partial) — expansion_failed_rows\n`);
      exitCode = 1;
    }
    if (st.expansionReplayMiss > 0) {
      process.stderr.write(`[longmemeval] FAIL --expansion-replay: ${st.expansionReplayMiss} question(s) had no recorded variants (expansion_replay_miss)\n`);
      exitCode = 1;
    }

    if (opts.record) {
      const commit = gitShort('git rev-parse --short HEAD', 'unknown');
      const repoRoot = gitShort('git rev-parse --show-toplevel', process.cwd());
      const record: EvalRunRecord = {
        schema_version: 3,
        run_id: `${commit}-longmemeval-${pins.mode}-${Date.now().toString(36)}`,
        ran_at: new Date().toISOString(),
        suite: 'longmemeval',
        mode: pins.mode,
        commit,
        seed: 0,
        ...(opts.limit !== undefined ? { limit: opts.limit } : {}),
        params: {
          ...runConfig,
          questions_run: st.questionsRun,
          output: opts.outputPath ?? null,
          aggregate: summary?.aggregate ?? null,
          mean_distinct_sessions: summary?.mean_distinct_sessions ?? null,
          qa_accuracy: qa
            ? {
                accuracy_headline: qa.accuracy_headline, accuracy_excluding_errors: qa.accuracy_excluding_errors,
                judged: qa.judged, correct: qa.correct, judge_errors: qa.judge_errors, skipped_budget: qa.skipped_budget,
                judge_model: qa.judge_model, judge_config_hash: qa.judge_config_hash, complete: qa.complete,
              }
            : null,
        },
        status: exitCode === 0 ? 'completed' : 'failed',
        duration_ms: Date.now() - st.runStart,
        // Belt-and-braces: persistRunRecord redacts too; keep the caller-side scrub.
        ...(st.errorMessages.length > 0 || exitCode !== 0
          ? { error: redactSecrets([...(exitCode !== 0 ? [`exit ${exitCode}`] : []), ...st.errorMessages].join(' | ')) }
          : {}),
      };
      persistRunRecord(repoRoot, record, runOpts.recordDir);
      process.stderr.write(`[longmemeval] recorded ${record.run_id} (${record.status})\n`);
    }
    if (exitCode !== 0) process.exit(exitCode);
  };

  // --resume-from: filter out already-answered question_ids before any
  // model/brain setup so a no-op resume costs ~zero. Refuse a mixed file
  // (rows produced under different retrieval pins) unless told otherwise.
  let appendOutput = false;
  /** Prior rows (resume). With --judge they are judged in place and re-emitted ahead of the new rows. */
  let priorRows: RowLike[] = [];
  /** Prior rows the --judge backfill judges from their stored hypothesis. */
  let backfill: RowLike[] = [];
  if (opts.resumeFromPath) {
    priorRows = readJsonlRows(opts.resumeFromPath);
    const check = checkResumeConfigHash(priorRows, retrievalHash);
    if (check.mismatched > 0) {
      const msg = `[longmemeval] resume: ${check.mismatched} row(s) in ${opts.resumeFromPath} carry a different retrieval_config_hash ` +
        `(${check.foreign.map(h => h.slice(0, 12)).join(', ')} vs this run's ${retrievalHash.slice(0, 12)}).`;
      if (!opts.allowMixedRunConfig) {
        process.stderr.write(`${msg} Refusing to mix runs; pass --allow-mixed-run-config to override.\n`);
        process.exit(1);
        return;
      }
      process.stderr.write(`${msg} Continuing (--allow-mixed-run-config).\n`);
    }
    if (check.unstamped > 0) {
      process.stderr.write(`[longmemeval] resume: ${check.unstamped} row(s) carry no retrieval_config_hash (pre-stamp file) — re-scored, not refused\n`);
    }
    const done = loadResumeSet(opts.resumeFromPath);
    const before = questions.length;
    questions = questions.filter(q => !done.has(q.question_id));
    process.stderr.write(`[longmemeval] resume: ${done.size} already done; ${questions.length}/${before} remaining\n`);
    if (opts.judge) {
      // Judge-only backfill (D33): rows lacking a settled verdict are judged
      // from their stored hypothesis; rows already judged under a DIFFERENT
      // judge_config_hash are refused unless told otherwise. The output is
      // REWRITTEN (prior rows + judge fields, then new rows), never appended.
      const sel = selectBackfillRows(priorRows, { questionByQid, hashFor: judgeHashFor });
      if (sel.mismatched > 0) {
        const msg = `[longmemeval] resume: ${sel.mismatched} judged row(s) in ${opts.resumeFromPath} carry a different judge_config_hash ` +
          `(${sel.foreign.map(h => h.slice(0, 12)).join(', ')} vs this run's ${runJudgeHash.slice(0, 12)}).`;
        if (!opts.allowMixedRunConfig) {
          process.stderr.write(`${msg} Refusing to mix judge configs; pass --allow-mixed-run-config to override.\n`);
          process.exit(1);
          return;
        }
        process.stderr.write(`${msg} Continuing (--allow-mixed-run-config).\n`);
      }
      if (sel.retrievalOnly > 0) {
        process.stderr.write(`Error: --judge: ${sel.retrievalOnly} row(s) in ${opts.resumeFromPath} were produced with --retrieval-only and carry no reader hypothesis to judge.\n`);
        process.exit(1);
        return;
      }
      if (sel.missingFromDataset > 0) process.stderr.write(`[longmemeval] WARN judge backfill: ${sel.missingFromDataset} row(s) not in this dataset — left unjudged\n`);
      backfill = sel.candidates;
      process.stderr.write(`[longmemeval] judge backfill: ${backfill.length} row(s) to judge from their stored hypothesis; ${sel.settled} verdict(s) stand\n`);
    } else if (opts.outputPath && opts.resumeFromPath === opts.outputPath) appendOutput = true;
    if (questions.length === 0 && backfill.length === 0) {
      process.stderr.write(`[longmemeval] resume: nothing to do (all questions already answered${opts.judge ? ' and judged' : ''}).\n`);
      if (opts.judge && opts.outputPath && opts.outputPath !== opts.resumeFromPath) {
        // A judge resume into a DIFFERENT output still copies the prior rows
        // forward, so the new file is complete (rows + the summary below).
        const em = makeEmitter(opts.outputPath, false, { atomicRewrite: !!opts.outputPath && opts.resumeFromPath === opts.outputPath });
        for (const row of priorRows) if (row.kind !== 'by_type_summary' && typeof row.question_id === 'string') em.emit(row);
        em.close();
      }
      // Even a no-op resume runs the FULL run-end block against the prior
      // rows (CDX-3 + review): --by-type emission, the floor gate, the
      // reranker / vector-degraded / expansion gates, and --record.
      const buckets: Record<string, RecallBucket> = {};
      const seed = seedBucketsFromRows(priorRows, buckets, { goldByQid, k: opts.topK, includeAbstention: opts.includeAbstention });
      const deg = countDegradation(priorRows, degradeOpts);
      finishRun({
        buckets,
        distinct: seed.distinct,
        excludedAbstention: seed.excludedAbstention,
        goldMissing: seed.goldMissing,
        slugCollisions: seed.collisions,
        rerankerSkippedRows: deg.rerankerSkipped,
        vectorDegradedRows: deg.vectorDegraded,
        expansionFailedRows: deg.expansionFailed,
        expansionReplayMiss: 0,
        errorCount: 0,
        errorMessages: [],
        cacheReceipt: null,
        cacheSkipped: 'resume_noop',
        questionsRun: 0,
        runStart: Date.now(),
        qaRows: priorRows,
        judgeEstUsd: null,
        judgeLedger: null,
      });
      return;
    }
  }

  let replay: Map<string, string[]> | null = null;
  if (opts.expansionReplayPath) {
    try {
      replay = loadExpansionReplay(opts.expansionReplayPath);
    } catch (err: any) {
      process.stderr.write(`Error: ${err.message ?? err}\n`);
      process.exit(1);
      return;
    }
    process.stderr.write(`[longmemeval] expansion replay: ${replay.size} recorded variant set(s) from ${opts.expansionReplayPath}\n`);
  }

  // #4636: BOTH chat lanes (answer generation + trajectory claim extractor)
  // route through the configured AI gateway — the same provider routing the
  // rest of the brain uses. gateway.chat parses `provider:model` recipe ids,
  // so resolveModel's output passes through UN-stripped.
  const gatewayClient: ThinkLLMClient = {
    create: async (params) => {
      const system = typeof params.system === 'string'
        ? params.system
        : Array.isArray(params.system)
          ? params.system.map(b => ('text' in b ? b.text : '')).join('')
          : undefined;
      const messages = params.messages.map(m => ({
        role: m.role,
        content: typeof m.content === 'string'
          ? m.content
          : Array.isArray(m.content)
            ? m.content.map(b => ('text' in b ? b.text : '')).join('')
            : '',
      }));
      const result = await gatewayChat({
        model: normalizeModelId(params.model),
        system,
        messages,
        maxTokens: params.max_tokens,
      });
      return {
        id: '',
        type: 'message',
        role: 'assistant',
        // The provider-reported snapshot id (D30) when the SDK surfaced one,
        // else the requested id — mirrors what the Anthropic SDK's `message.model` carries.
        model: result.responseModel ?? result.model,
        content: [{ type: 'text', text: result.text }],
        usage: { input_tokens: result.usage.input_tokens, output_tokens: result.usage.output_tokens },
        stop_reason: result.stopReason === 'length' ? 'max_tokens' : 'end_turn',
      } as unknown as Anthropic.Message;
    },
  };
  const client: ThinkLLMClient = runOpts.client ?? gatewayClient;
  const extractorClient: ThinkLLMClient = runOpts.extractorClient ?? gatewayClient;
  const extractorModel = trajectoryEnabled
    ? await resolveModel(null, {
        cliFlag: runOpts.extractorModel,
        tier: 'utility',
        fallback: 'haiku',
      })
    : '';

  // --judge preflight (Phase D): availability → pricing → estimate → cap.
  // Runs before the brain connects so a refused run costs nothing.
  let judgeCtx: JudgeLaneContext | null = null;
  let judgeEstUsd: number | null = null;
  let judgeLedger: BudgetLedger | null = null;
  if (opts.judge) {
    const pre = judgePreflight({
      judgeModel: opts.judgeModel,
      maxUsd: opts.maxUsd,
      yes: opts.yes,
      available: runOpts.judgeClient ? true : isAvailable('chat', opts.judgeModel),
      live: questions,
      readerMaxTokens: READER_MAX_TOKENS,
      backfill,
    });
    if (!pre.ok) {
      process.stderr.write(`Error: ${pre.message}\n`);
      process.exit(pre.exitCode);
      return;
    }
    for (const line of pre.lines) process.stderr.write(`${line}\n`);
    judgeEstUsd = pre.estUsd;
    judgeLedger = pre.ledger;
    judgeCtx = {
      client: runOpts.judgeClient ?? gatewayChat,
      model: opts.judgeModel,
      ledger: pre.ledger,
      configHashFor: judgeHashFor,
      ...(runOpts.judgeBackoffMs !== undefined ? { backoffMs: runOpts.judgeBackoffMs } : {}),
    };
  }

  process.stderr.write(`[longmemeval] estimated 20-60 minutes for ${questions.length} questions; use --limit N for shorter runs\n`);
  process.stderr.write(`[longmemeval] connecting in-memory brain...\n`);
  process.stderr.write(
    `[longmemeval] starting (questions: ${questions.length}, model: ${model}, mode: ${pins.mode}, ` +
    `reranker: ${pins.reranker.enabled ? 'on' : 'off'}, autocut: ${pins.autocut ? 'on' : 'off'}, ` +
    `expansion: ${pins.expansion ? (replay ? 'replay' : 'on') : 'off'}, budget: ${pins.expansion_variant_budget ?? 'legacy'}, ` +
    `embedder: ${pins.embedder}, top-k: ${pins.top_k}, trajectory: ${trajectoryEnabled ? 'on' : 'off'}` +
    `${trajectoryEnabled ? `, extractor: ${extractorModel}` : ''}, retrieval_config_hash: ${retrievalHash.slice(0, 12)})\n`,
  );
  if (trajectoryEnabled) resetExtractorState();

  const progress = createProgress(cliOptsToProgressOptions(getCliOptions()));
  const buckets: Record<string, RecallBucket> = {};
  const distinct: number[] = [];
  let excludedAbstention = 0;
  let goldMissing = 0;
  let slugCollisions = 0;
  let rerankerSkippedRows = 0;
  let vectorDegradedRows = 0;
  let expansionFailedRows = 0;
  let expansionReplayMiss = 0;
  // Prior rows that survive into the output: a prior error row for a question
  // about to be re-run is dropped (its retry row follows it).
  const rerun = new Set(questions.map(q => q.question_id));
  const keptPrior = priorRows.filter(r => !(typeof r.error === 'string' && !r.hypothesis && rerun.has(r.question_id as string)));
  // --resume-from: prior rows feed the degradation gates whether or not
  // --by-type is set (a partial resume must still fail on rows the prior run
  // degraded). Bucket seeding (re-scored) only feeds the --by-type summary.
  if (opts.resumeFromPath) {
    const deg = countDegradation(keptPrior, degradeOpts);
    rerankerSkippedRows += deg.rerankerSkipped;
    vectorDegradedRows += deg.vectorDegraded;
    expansionFailedRows += deg.expansionFailed;
  }
  if (opts.byType && opts.resumeFromPath) {
    const seed = seedBucketsFromRows(keptPrior, buckets, { goldByQid, k: opts.topK, includeAbstention: opts.includeAbstention });
    distinct.push(...seed.distinct);
    excludedAbstention += seed.excludedAbstention;
    goldMissing += seed.goldMissing;
    slugCollisions += seed.collisions;
  }
  const runStart = Date.now();
  let errorCount = 0;
  const errorMessages: string[] = [];
  let cacheReceipt: CacheReceipt | null = null;
  let cacheSkipped: string | undefined;
  const qaRows: RowLike[] = [];

  const foldRow = (row: LongMemEvalRow): void => {
    // gold_missing / slug_collisions count EVERY question row (abstention + no-gold rows here,
    // collision-abort error rows in the catch below) — the set seedBucketsFromRows counts on resume.
    if (row.gold_missing_from_haystack.length > 0) goldMissing++;
    if (row.slug_collision > 0) slugCollisions++;
    if (row.abstention && !opts.includeAbstention) { excludedAbstention++; return; }
    if (typeof row.recall_all_hit !== 'boolean') return; // no gold → not in the denominator
    const bucket = buckets[row.question_type] ?? (buckets[row.question_type] = newBucket());
    if (addRowToBucket(bucket, row) !== 'skipped') distinct.push(row.distinct_sessions_in_top_k);
  };

  const ctx: RunContext = {
    opts, model, client, trajectoryEnabled, extractorClient, extractorModel,
    expandFn: runOpts.expandFn ?? expandQuery,
    replay,
    retrievalConfigHash: retrievalHash,
    embedTxn: (fn) => fn(),
  };

  const work = async (engine: PGLiteEngine): Promise<void> => {
    // #3676: nightly probe callers may copy audited live search config into
    // this isolated engine. Data tables stay hermetic.
    for (const [key, value] of Object.entries(runOpts.searchConfigSnapshot ?? {})) {
      if (key.trim().length > 0) await engine.setConfig(key, value);
    }
    // Explicit pins beat the snapshot. resetTables preserves `config` between
    // questions, so these fire once for the run; hybridSearch resolves them
    // through the standard chain.
    if (opts.mode) await engine.setConfig('search.mode', opts.mode);
    if (opts.reranker !== undefined) await engine.setConfig('search.reranker.enabled', opts.reranker ? 'true' : 'false');
    if (opts.autocut !== undefined) await engine.setConfig('search.autocut', opts.autocut ? 'true' : 'false');
    for (const [k, v] of Object.entries(opts.searchPins ?? {})) await engine.setConfig(k, v);
    if (opts.expansionVariantBudget !== undefined) {
      await engine.setConfig('search.expansion_variant_budget', opts.expansionVariantBudget === null ? 'legacy' : String(opts.expansionVariantBudget));
    }

    // --reranker on: preflight. A reranker that cannot run would silently
    // turn every row into RRF order (reranker_skipped) — refuse up front.
    if (opts.reranker === true) {
      const probe = runOpts.rerankerReadiness ?? rerankerReadinessForEngine;
      const r = await probe(engine, pins.reranker.model);
      if (!r.readiness.ready) {
        process.stderr.write(`[longmemeval] reranker not ready (${r.plane} plane): ${describeRerankerFix(r.readiness) ?? 'not ready'}\n`);
        process.exit(2);
        return;
      }
    }

    // Embedding cache (plan 0c): byte-identical vectors across arms. Never
    // installed on --keyword-only (no embeds happen).
    let cache: EmbeddingCache | null = null;
    let installed: InstalledEmbedCache | null = null;
    if (!opts.embedCache) cacheSkipped = 'disabled';
    else if (opts.keywordOnly) cacheSkipped = 'keyword_only';
    else {
      try {
        cache = new EmbeddingCache(opts.embedCachePath);
        installed = installEmbedCache(cache, { realTransport: runOpts.embedTransport ?? null });
        process.stderr.write(`[longmemeval] embed cache: ${cache.path} (${installed.model}@${installed.dims}, ${cache.size()} rows)\n`);
      } catch (err: any) {
        cache = null;
        cacheSkipped = `install_failed: ${redactSecrets(String(err?.message ?? err))}`;
        process.stderr.write(`[longmemeval] embed cache not installed: ${cacheSkipped}\n`);
      }
    }

    // The transaction is scoped INSIDE runOneQuestion to the import + search
    // section (the only embed producers). Wrapping the whole question here
    // would roll back a question's vectors on a reader failure and hold the
    // sqlite write lock across the LLM round-trip.
    const c = cache;
    ctx.embedTxn = c ? (fn) => c.withTransaction(fn) : (fn) => fn();

    // Rewrite-in-place (judge backfill re-emits the prior rows): write to a
    // temp file and rename on close so a kill mid-run never truncates the
    // resume file (the paid reader rows) to zero bytes.
    const emitter = makeEmitter(opts.outputPath, appendOutput, { atomicRewrite: !!opts.outputPath && opts.resumeFromPath === opts.outputPath });
    progress.start('eval.longmemeval', backfill.length + questions.length);
    try {
      // Judge-only backfill first (prior rows, no reader call), then re-emit
      // EVERY prior question row (judged or not) so the rewritten output is
      // complete; prior error rows for questions about to be re-run are
      // dropped (their retry row follows), prior summary lines always are.
      if (judgeCtx && backfill.length > 0) {
        await runJudgeBackfill(backfill, judgeCtx, {
          concurrency: opts.judgeConcurrency,
          questionByQid,
          onRow: (row) => progress.tick(1, `${String(row.question_id)} (judge)`),
        });
      }
      for (const row of keptPrior) {
        if (row.kind === 'by_type_summary' || typeof row.question_id !== 'string') continue;
        qaRows.push(row);
        if (opts.judge) emitter.emit(row);
      }
      for (const q of questions) {
        const qStart = Date.now();
        try {
          const outcome = await runOneQuestion(engine, q, ctx);
          if (judgeCtx) {
            // Judge inline from the row's hypothesis (the same path the backfill takes).
            Object.assign(outcome.row, await judgeRow({
              ...outcome.row,
              question_id: q.question_id, question_type: q.question_type, question: q.question,
              answer: q.answer ?? '', hypothesis: outcome.row.hypothesis,
            }, judgeCtx));
          }
          emitter.emit(outcome.row);
          qaRows.push(outcome.row);
          if (outcome.rerankerSkipped) rerankerSkippedRows++;
          if (outcome.vectorDegraded) vectorDegradedRows++;
          if (outcome.expansionFailed) expansionFailedRows++;
          foldRow(outcome.row);
          progress.tick(1, q.question_id);
        } catch (err: any) {
          errorCount++;
          const message = redactSecrets(String(err?.message ?? err));
          if (errorMessages.length < 3) errorMessages.push(`${q.question_id}: ${message}`);
          const extra = err instanceof QuestionAbort ? err.extra : {};
          if (extra.expansion_replay_miss === true) expansionReplayMiss++;
          if (typeof extra.slug_collision === 'number' && extra.slug_collision > 0) slugCollisions++;
          // Error rows carry question + question_type so the cross-modal
          // --batch consumer can flag them as upstream errors instead of
          // silently dropping them from the denominator.
          const errorRow: RowLike = {
            question_id: q.question_id,
            question: q.question,
            question_type: q.question_type,
            hypothesis: '',
            error: message,
            retrieval_config_hash: retrievalHash,
            ...extra,
          };
          emitter.emit(errorRow);
          qaRows.push(errorRow);
          progress.tick(1, `${q.question_id} (error)`);
        }
        if (process.env.GBRAIN_LME_DEBUG === '1') {
          process.stderr.write(`[longmemeval] ${q.question_id} ${Date.now() - qStart}ms\n`);
        }
      }
    } finally {
      progress.finish();
      emitter.close();
      if (cache) {
        try {
          const s = cache.stats();
          cacheReceipt = { path: cache.path, hits: s.hits, misses: s.misses, bypassed: s.bypassed, infra_faults: s.infra_faults, canonical_sha256: cache.canonicalSha256(), sha256: cache.fileSha256() };
        } catch (err: any) {
          cacheSkipped = `receipt_failed: ${redactSecrets(String(err?.message ?? err))}`;
        }
        installed?.uninstall();
        cache.close();
      }
    }
  };

  if (runOpts.engine) {
    // Caller owns engine lifecycle (typically a test beforeAll/afterAll).
    await work(runOpts.engine);
  } else {
    await withBenchmarkBrain(work);
  }

  if (trajectoryEnabled) {
    const xc = getCacheStats();
    const total = xc.hits + xc.misses;
    const pct = total === 0 ? 0 : (xc.hits / total) * 100;
    process.stderr.write(`[longmemeval] extractor.cache_hits: ${xc.hits} / ${total} sessions (${pct.toFixed(1)}%, cached_bodies=${xc.size})\n`);
    process.stderr.write(`[longmemeval] methodology_note: ${TRAJECTORY_METHODOLOGY_NOTE}\n`);
  }

  finishRun({
    buckets, distinct, excludedAbstention, goldMissing, slugCollisions,
    rerankerSkippedRows, vectorDegradedRows, expansionFailedRows, expansionReplayMiss,
    errorCount, errorMessages, cacheReceipt, cacheSkipped,
    questionsRun: questions.length, runStart,
    qaRows, judgeEstUsd, judgeLedger,
  });
}

function poolKey(r: SearchResult): string {
  return `${r.slug}#${r.chunk_id}`;
}

async function runOneQuestion(
  engine: PGLiteEngine,
  q: DatasetQuestion,
  ctx: RunContext,
): Promise<QuestionOutcome> {
  const { opts } = ctx;
  const slugToRaw = buildSlugToRawMap(q);
  const gold = q.answer_session_ids ?? [];
  // Pre-import aborts (pure, zero cost): both fire BEFORE resetTables /
  // import / any embed call so an abort never spends an embedding.
  //
  // Plan D32: a slug collision that touches a gold id makes the join
  // ambiguous — abort the question with an error row instead of scoring it.
  const goldCollisions = collisionsTouchingGold(slugToRaw, gold);
  if (goldCollisions.length > 0) {
    throw new QuestionAbort(
      `slug_collision touches a gold session id (${goldCollisions.map(s => `${s} <- ${(slugToRaw.get(s) ?? []).join(' / ')}`).join('; ')})`,
      { slug_collision: detectSlugCollisions(slugToRaw).length, slug_collision_gold: goldCollisions, abstention: /_abs$/.test(q.question_id) },
    );
  }
  // --expansion-replay: a question with no recorded variants is an error row
  // (exit 1 at the end) — detected here, not after the haystack was imported.
  let expansionVariants: string[] | undefined;
  let expansionReplayed = false;
  let expandFn: ((query: string) => Promise<string[]>) | undefined;
  if (!opts.keywordOnly && opts.expansion) {
    if (ctx.replay) {
      const recorded = ctx.replay.get(q.question_id);
      if (!recorded) {
        throw new QuestionAbort(`expansion_replay_miss: no recorded expansion_variants for ${q.question_id}`, { expansion_replay_miss: true });
      }
      expansionVariants = recorded;
      expansionReplayed = true;
      expandFn = async () => recorded;
    } else {
      const base = ctx.expandFn;
      expandFn = async (query) => {
        const v = await base(query);
        expansionVariants = v;
        return v;
      };
    }
  }

  await resetTables(engine);
  const adapterPages = haystackToPages(q);
  const dates = q.haystack_dates ?? [];
  const pageMeta: { slug: string; content: string; date?: string }[] = [];
  // Per-question alias map for the extractor so canonical-slug aliases never
  // leak across questions.
  const aliasMap: AliasMap = makeAliasMap();

  let meta: HybridSearchMeta | undefined;
  let pool: SearchResult[] | undefined;
  let preRerank: SearchResult[] | undefined;
  // The embed-producing section (import + search) runs in ONE embed-cache
  // transaction and COMMITS before the trajectory lookup / reader call.
  const results: SearchResult[] = await ctx.embedTxn(async () => {
    for (let i = 0; i < adapterPages.length; i++) {
      const p = adapterPages[i];
      pageMeta.push({ slug: p.slug, content: p.content, date: dates[i] });
      await importFromContent(engine, p.slug, p.content, { noEmbed: opts.keywordOnly });
      // Inline Haiku extractor populates the facts table so trajectory routing
      // has data to retrieve. Fail-open per session.
      if (ctx.trajectoryEnabled) {
        await extractAndInsertClaims({
          engine,
          client: ctx.extractorClient,
          model: ctx.extractorModel,
          sessionSlug: p.slug,
          sessionId: sessionIdFromSlug(p.slug),
          sessionBody: p.content,
          sourceId: 'default',
          aliasMap,
        });
      }
    }
    if (opts.keywordOnly) return engine.searchKeyword(q.question, { limit: opts.topK });
    const searchOpts: HybridSearchOpts = {
      limit: opts.topK,
      // Per-call wins over the bundle: expansion fires ONLY with --expansion.
      expansion: opts.expansion,
      ...(expandFn ? { expandFn } : {}),
      ...(opts.expansionVariantBudget !== undefined ? { expansionVariantBudget: opts.expansionVariantBudget } : {}),
      onMeta: (m) => { meta = m; },
      ...(opts.capturePool
        ? { onRerankPool: (p: readonly SearchResult[], pre?: readonly SearchResult[]) => { pool = [...p]; preRerank = pre ? [...pre] : undefined; } }
        : {}),
    };
    return hybridSearch(engine, q.question, searchOpts);
  });

  // Trajectory routing for temporal / knowledge_update intents. Skips for
  // 'other' or when --no-trajectory.
  let trajectoryBlock = '';
  let trajectoryPoints = 0;
  let entityResolved: string | null = null;
  let resolutionSource: ResolutionSource | null = null;
  const intent: Intent = ctx.trajectoryEnabled ? classifyIntent(q) : 'other';
  if (ctx.trajectoryEnabled && intent !== 'other') {
    try {
      const retrievedSlugs = results.map(r => r.slug);
      const candidates = extractCandidateEntities(q.question, retrievedSlugs);
      for (const cand of candidates) {
        const resolved = await resolveEntitySlugWithSource(engine, 'default', cand.raw);
        if (!resolved) continue;
        // Unlike the think production path, the harness does NOT skip
        // fallback_slugify results: the extractor and the lookup both slugify
        // free-form entity names, so they cohere on the same fallback slug
        // and there are no canonical pages in the benchmark to protect.
        const points = await Promise.race([
          engine.findTrajectory({
            entitySlug: resolved.slug,
            sourceId: 'default',
            remote: false,
            kind: 'all',
            limit: 100,
          }),
          new Promise<import('../core/engine.ts').TrajectoryPoint[]>(resolve => {
            setTimeout(() => resolve([]), 5000);
          }),
        ]);
        if (points.length === 0) continue;
        const fmt = formatTrajectoryBlock(points, resolved.slug, { intent });
        if (fmt.rendered.length === 0) continue;
        trajectoryBlock = fmt.rendered;
        trajectoryPoints = fmt.emittedPoints;
        entityResolved = resolved.slug;
        resolutionSource = resolved.source;
        break;
      }
    } catch {
      // Best-effort: any error degrades to "no block injected".
    }
  }

  // Reader pins (D30) ride every answered row: requested model, the
  // provider-reported snapshot when it differs, the system-prompt sha and the
  // output cap. --retrieval-only rows are marked so a --judge backfill can
  // refuse them (no reader hypothesis to grade).
  let hypothesis: string;
  let readerFields: Record<string, unknown>;
  if (opts.retrievalOnly) {
    hypothesis = renderRetrievedAsHypothesis(results, slugToRaw);
    readerFields = { retrieval_only: true };
  } else {
    const answer = await generateAnswer(ctx.client, q, results, pageMeta, slugToRaw, ctx.model, trajectoryBlock);
    hypothesis = answer.text;
    readerFields = {
      reader_model: ctx.model,
      reader_model_snapshot: answer.response_model,
      reader_prompt_sha: READER_PROMPT_SHA,
      reader_max_tokens: READER_MAX_TOKENS,
      reader_context_chars: answer.context_chars,
      reader_context_sessions: answer.context_sessions,
      reader_sessions_truncated: answer.sessions_truncated,
    };
  }

  // search_meta (plan 0e): `reranked` is derived — no such meta field exists.
  const degraded = meta?.degraded ?? [];
  const searchMeta: Record<string, unknown> & RowSearchMeta = {
    vector_enabled: meta?.vector_enabled ?? false,
    expansion_applied: meta?.expansion_applied ?? false,
    degraded,
    reranked: false,
    ...(meta?.autocut ? { autocut: meta.autocut } : {}),
  };
  const { rerankerSkipped, vectorDegraded, expansionFailed } = classifyDegradation(searchMeta, { keywordOnly: opts.keywordOnly, expansion: opts.expansion });
  searchMeta.reranked = results.some(r => Number.isFinite(r.rerank_score)) && !rerankerSkipped;

  const extra: Record<string, unknown> = {
    retrieval_config_hash: ctx.retrievalConfigHash,
    ...readerFields,
    search_meta: searchMeta,
    ...(expansionVariants ? { expansion_variants: expansionVariants } : {}),
    ...(expansionReplayed ? { expansion_replayed: true } : {}),
    ...(ctx.trajectoryEnabled ? {
      intent,
      trajectory_points: trajectoryPoints,
      entity_resolved: entityResolved,
      resolution_source: resolutionSource,
      methodology_note: TRAJECTORY_METHODOLOGY_NOTE,
    } : {}),
  };
  if (pool) {
    // rerank_pool (plan D24): EVERY row of the pool hybridSearch hands to
    // applyAutocut, in pool order — including unscored alias-hop / exact-lookup
    // injected rows (they carry no rerank_score; the replay's preserve
    // predicate keeps them exactly as the live cut did). `rrf_rank` is the
    // pre-rerank RRF position when the hook supplied it (cliff attribution:
    // fusion vs reranking) and otherwise the pool position; `pool_rank` is
    // always the 1-based position in the captured pool.
    const rrfRank = new Map<string, number>();
    (preRerank ?? []).forEach((r, i) => rrfRank.set(poolKey(r), i + 1));
    extra.rerank_pool = pool.map((r, i) => ({
      slug: r.slug,
      chunk_id: r.chunk_id,
      session_id: rawSessionId(r.slug, slugToRaw),
      rrf_rank: rrfRank.get(poolKey(r)) ?? i + 1,
      pool_rank: i + 1,
      ...(Number.isFinite(r.rerank_score) ? { rerank_score: r.rerank_score } : {}),
      ...(r.alias_hit === true ? { alias_hit: true } : {}),
      ...(r.exact_lookup !== undefined ? { exact_lookup: true } : {}),
      // Mirrors hybrid.ts's autocut predicates: pinned relational rows are
      // preserved through the cut AND excluded from its cliff math.
      ...(r.relational_pinned === true ? { relational_pinned: true } : {}),
      est_tokens: estimateTokens(r.chunk_text),
    }));
    // Autocut on (a decision was recorded): when the kept count equals the
    // rows we got back, the returned rows ARE the kept set — record their
    // keys so the replay validates the cut byte-for-byte (otherwise it
    // validates at count/gap level; a further limit/budget slice hides the
    // exact set).
    if (meta?.autocut && meta.autocut.kept === results.length) {
      extra.autocut_kept_keys = results.map(poolKey);
    }
  }

  const row = buildRow({ question: q, hypothesis, results, k: opts.topK, slugToRaw, mode: opts.mode, extra });
  return { row, rerankerSkipped, vectorDegraded, expansionFailed };
}

/** Schema-v2 by_type_summary (metrics.ts builder) — the ONE summary shape. */
export type ByTypeSummary = ByTypeSummaryV2;
export { buildByTypeSummaryV2 as buildByTypeSummary } from '../eval/longmemeval/metrics.ts';
export { seedRecallByTypeFromFile } from '../eval/longmemeval/resume.ts';
