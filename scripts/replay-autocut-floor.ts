/**
 * replay-autocut-floor.ts — replay the autocut weak-top floor sweep over a
 * captured rerank pool (plan Phase C / D24, rule R2).
 *
 * INVARIANT: every cell comes from ONE capture (the shipped-default arm run
 * with `--capture-pool`); no second reranker call. `--validate-live <floor>`
 * must pass (the replay reproduces the recorded live decisions byte-for-byte)
 * before any other floor cell is read. The pure core lives in
 * src/eval/shared/autocut-replay.ts; this file owns argv, file I/O, printing.
 *
 * Usage:
 *   bun run scripts/replay-autocut-floor.ts <capture.ndjson> \
 *     --floors off,0.10,0.20,0.35,0.50,0.65,0.80 [--k 5] \
 *     [--validate-live 0.35] [--split-half seed42] [--jump 0.2] [--min-keep 1] [--json]
 *
 * Exit: 0 ok · 1 validate-live mismatch or bad input · 2 usage.
 */

import { readFileSync } from 'node:fs';
import { buildMetricGlossaryMeta } from '../src/core/eval/metric-glossary.ts';
import {
  parseFloors,
  parseReplayNdjson,
  sweepFloors,
  validateLive,
  type Floor,
  type FloorSummary,
  type ReplayKnobs,
  DEFAULT_REPLAY_KNOBS,
} from '../src/eval/shared/autocut-replay.ts';

function usage(code: number): never {
  process.stderr.write(
    'usage: bun run scripts/replay-autocut-floor.ts <capture.ndjson> --floors off,0.10,0.35 [--k 5] ' +
      '[--validate-live 0.35] [--split-half <seed>] [--jump 0.2] [--min-keep 1] [--json]\n',
  );
  process.exit(code);
}

interface Args {
  file: string;
  floors: Floor[];
  k: number;
  validateLive?: Floor;
  splitSeed?: string;
  knobs: ReplayKnobs;
  json: boolean;
}

function parseArgs(argv: string[]): Args {
  let file: string | undefined;
  let floorsSpec: string | undefined;
  let k = 5;
  let validate: Floor | undefined;
  let splitSeed: string | undefined;
  let jumpRatio = DEFAULT_REPLAY_KNOBS.jumpRatio;
  let minKeep = DEFAULT_REPLAY_KNOBS.minKeep;
  let json = false;
  const need = (i: number, flag: string): string => {
    if (i + 1 >= argv.length) {
      process.stderr.write(`${flag} needs a value\n`);
      usage(2);
    }
    return argv[i + 1];
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--floors') floorsSpec = need(i++, a);
    else if (a === '--k') k = Number(need(i++, a));
    else if (a === '--validate-live') {
      // Exactly ONE floor: the capture arm ran at a single floor, so a list
      // here can only mean the caller expected a sweep — refuse rather than
      // silently validating against the first entry.
      const floors = parseFloors(need(i++, a));
      if (floors.length !== 1) {
        process.stderr.write(`--validate-live takes exactly one floor (got ${floors.length}: ${argv[i]}); the sweep is --floors\n`);
        usage(2);
      }
      validate = floors[0];
    }
    else if (a === '--split-half') splitSeed = need(i++, a);
    else if (a === '--jump') jumpRatio = Number(need(i++, a));
    else if (a === '--min-keep') minKeep = Number(need(i++, a));
    else if (a === '--json') json = true;
    else if (a === '--help' || a === '-h') usage(0);
    else if (a.startsWith('--')) {
      process.stderr.write(`unknown flag ${a}\n`);
      usage(2);
    } else if (file === undefined) file = a;
    else {
      process.stderr.write(`unexpected argument ${a}\n`);
      usage(2);
    }
  }
  if (!file || !floorsSpec) usage(2);
  if (!Number.isInteger(k) || k < 1) {
    process.stderr.write(`--k must be a positive integer\n`);
    usage(2);
  }
  if (!Number.isFinite(jumpRatio) || jumpRatio <= 0 || jumpRatio > 1) {
    process.stderr.write(`--jump must be in (0, 1]\n`);
    usage(2);
  }
  if (!Number.isInteger(minKeep) || minKeep < 1) {
    process.stderr.write(`--min-keep must be a positive integer\n`);
    usage(2);
  }
  return { file, floors: parseFloors(floorsSpec), k, validateLive: validate, splitSeed, knobs: { jumpRatio, minKeep }, json };
}

/** Every metric this script prints, as glossary keys (all carried by src/core/eval/metric-glossary.ts). */
export const REPLAY_GLOSSARY_KEYS = ['recall_all@k', 'recall_any@k', 'mean_returned_results', 'mean_returned_est_tokens'] as const;

/**
 * One `_meta.metric_glossary` block per response ([CDX-25]). Every key routes
 * through src/core/eval/metric-glossary.ts — no local fallback lines, so a
 * metric printed here without a glossary entry is a test failure, not a
 * silent gap.
 */
function glossaryBlock(): Record<string, string> {
  return buildMetricGlossaryMeta(REPLAY_GLOSSARY_KEYS);
}

function pct(x: number): string {
  return `${(100 * x).toFixed(2)}%`;
}

function renderTable(title: string, summaries: FloorSummary[], out: string[]): void {
  out.push(`## ${title}`);
  out.push('| floor | n | recall_all@k | recall_any@k | autocut applied | mean returned | mean est_tokens | mean kept pool |');
  out.push('|---|---|---|---|---|---|---|---|');
  for (const s of summaries) {
    out.push(
      `| ${s.floor} | ${s.n} | ${s.recall_all_hit} (${pct(s.recall_all_rate)}) | ${s.recall_any_hit} (${pct(s.recall_any_rate)}) | ` +
        `${s.autocut_applied} | ${s.mean_returned_results.toFixed(2)} | ${s.mean_returned_est_tokens.toFixed(0)} | ${s.mean_kept_pool.toFixed(2)} |`,
    );
  }
  out.push('');
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  let text: string;
  try {
    text = readFileSync(args.file, 'utf-8');
  } catch (err) {
    process.stderr.write(`cannot read ${args.file}: ${(err as Error).message}\n`);
    process.exit(1);
  }
  let parsed;
  try {
    parsed = parseReplayNdjson(text);
  } catch (err) {
    process.stderr.write(`replay-autocut-floor: ${(err as Error).message}\n`);
    process.exit(1);
  }
  if (parsed.rows.length === 0) {
    process.stderr.write('replay-autocut-floor: no question rows with rerank_pool found\n');
    process.exit(1);
  }

  let validateMismatches: ReturnType<typeof validateLive> = [];
  if (args.validateLive !== undefined) {
    validateMismatches = validateLive(parsed.rows, args.validateLive, args.knobs);
  }

  const sweep = sweepFloors(parsed.rows, args.floors, args.k, { knobs: args.knobs, splitSeed: args.splitSeed });

  if (args.json) {
    process.stdout.write(
      JSON.stringify(
        {
          file: args.file,
          rows: parsed.rows.length,
          skipped_error_rows: parsed.skipped_error_rows,
          skipped_non_question_rows: parsed.skipped_non_question_rows,
          validate_live: args.validateLive === undefined ? null : { floor: args.validateLive, mismatches: validateMismatches },
          ...sweep,
          _meta: { metric_glossary: glossaryBlock() },
        },
        null,
        2,
      ) + '\n',
    );
  } else {
    const out: string[] = [];
    out.push(`# autocut floor replay — ${args.file}`);
    out.push(`rows=${parsed.rows.length} skipped_error=${parsed.skipped_error_rows} k=${args.k} jump=${args.knobs.jumpRatio} minKeep=${args.knobs.minKeep}`);
    out.push('');
    if (args.validateLive !== undefined) {
      out.push(
        validateMismatches.length === 0
          ? `validate-live @ ${args.validateLive}: OK (${parsed.rows.length} rows reproduce the recorded live decision)`
          : `validate-live @ ${args.validateLive}: FAIL (${validateMismatches.length} row(s) differ)`,
      );
      for (const m of validateMismatches) out.push(`  - ${m.question_id}: ${m.reason}`);
      out.push('');
    }
    renderTable('all rows', sweep.summaries, out);
    out.push('## paired recall_all vs first floor');
    out.push('| floor | baseline | wins | losses | net | per-type net |');
    out.push('|---|---|---|---|---|---|');
    for (const p of sweep.paired) {
      const perType = Object.entries(p.by_type_net)
        .map(([t, n]) => `${t}:${n >= 0 ? '+' : ''}${n}`)
        .join(' ');
      out.push(`| ${p.floor} | ${p.baseline_floor} | ${p.wins} | ${p.losses} | ${p.net >= 0 ? '+' : ''}${p.net} | ${perType} |`);
    }
    out.push('');
    if (sweep.split) {
      renderTable(`half A (seed ${sweep.split.seed})`, sweep.split.a, out);
      renderTable(`half B (seed ${sweep.split.seed})`, sweep.split.b, out);
    }
    out.push('## top rerank score histogram');
    out.push('| bin | count |');
    out.push('|---|---|');
    for (const b of sweep.top_score_histogram) out.push(`| [${b.bin_start.toFixed(1)}, ${b.bin_end.toFixed(1)}) | ${b.count} |`);
    out.push('');
    out.push('Glossary: recall_all@k = every gold session among the distinct sessions of the first k kept chunk rows; recall_any@k = at least one; mean est_tokens = returned-window token estimate (autocut benefit metric).');
    process.stdout.write(out.join('\n') + '\n');
  }

  if (validateMismatches.length > 0) process.exit(1);
}

main();
