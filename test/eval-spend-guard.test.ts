/**
 * scripts/eval-spend-guard.sh — subprocess pins with a temp ledger.
 *
 * Hermetic: the wrapped "paid command" is a shell one-liner that writes a
 * marker file (proving it ran) and optionally an actual-cost file. Env is
 * passed to spawnSync, never mutated on process.env.
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const ROOT = process.cwd();
const SCRIPT = join(ROOT, 'scripts/eval-spend-guard.sh');

let dir: string;
let ledger: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'gbrain-spend-guard-'));
  ledger = join(dir, 'receipts', 'spend.jsonl');
  // The guard requires the ledger to EXIST (a missing ledger is not a $0
  // ledger). Every test starts from an empty, present ledger; the missing-
  // ledger behaviour has its own tests below, which remove it first.
  mkdirSync(join(dir, 'receipts'), { recursive: true });
  writeFileSync(ledger, '');
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function run(args: string[], extraEnv: Record<string, string> = {}) {
  const env: Record<string, string> = {
    PATH: process.env.PATH ?? '/usr/bin:/bin',
    HOME: dir,
    TMPDIR: dir,
    GBRAIN_EVAL_SPEND_LEDGER: ledger,
    ...extraEnv,
  };
  return spawnSync('bash', [SCRIPT, ...args], { cwd: ROOT, encoding: 'utf-8', env });
}

function ledgerRows(): Array<Record<string, unknown>> {
  if (!existsSync(ledger)) return [];
  return readFileSync(ledger, 'utf-8')
    .split('\n')
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as Record<string, unknown>);
}

describe('eval-spend-guard.sh', () => {
  test('under cap: runs the command and appends the estimate as cost', () => {
    const marker = join(dir, 'ran.txt');
    const r = run(['75', '3', '--', 'sh', '-c', `echo ok > "${marker}"`]);
    expect(r.status).toBe(0);
    expect(existsSync(marker)).toBe(true);
    expect(r.stderr).toContain('launching');
    const rows = ledgerRows();
    expect(rows.length).toBe(1);
    expect(rows[0].estimate_usd).toBe(3);
    expect(rows[0].cost_usd).toBe(3);
    expect(rows[0].exit_code).toBe(0);
    expect(String(rows[0].command)).toContain('echo ok');
    expect(String(rows[0].ts)).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
  });

  test('over cap: refuses with exit 3, never runs the command, appends nothing', () => {
    // seed the ledger at $73.50 across two rows (one with an exponent form)
    writeFileSync(
      ledger,
      '{"ts":"2026-09-04T00:00:00Z","estimate_usd":70,"cost_usd":70.25,"exit_code":0,"command":"x"}\n' +
        '{"ts":"2026-09-04T00:00:01Z","estimate_usd":3,"cost_usd":3.25e0,"exit_code":0,"command":"y"}\n',
    );
    const marker = join(dir, 'should-not-exist.txt');
    const r = run(['75', '2', '--', 'sh', '-c', `echo no > "${marker}"`]);
    expect(r.status).toBe(3);
    expect(existsSync(marker)).toBe(false);
    expect(r.stderr).toContain('REFUSED');
    expect(r.stderr).toContain('75.5');
    expect(ledgerRows().length).toBe(2);
  });

  test('exactly at cap is allowed (ledger + estimate == cap)', () => {
    writeFileSync(ledger, '{"cost_usd":72}\n');
    const r = run(['75', '3', '--', 'true']);
    expect(r.status).toBe(0);
    expect(ledgerRows().length).toBe(2);
  });

  test('actual-cost file written by the command overrides the estimate (bare number)', () => {
    const costFile = join(dir, 'actual.txt');
    const r = run(['75', '3', '--', 'sh', '-c', `printf '1.2345\\n' > "$GBRAIN_EVAL_ACTUAL_COST_FILE"`], {
      GBRAIN_EVAL_ACTUAL_COST_FILE: costFile,
    });
    expect(r.status).toBe(0);
    const rows = ledgerRows();
    expect(rows[0].estimate_usd).toBe(3);
    expect(rows[0].cost_usd).toBe(1.2345);
  });

  test('actual-cost file as JSON {cost_usd} is honored; unset env gets a scratch path exported to the child', () => {
    const r = run([
      '75',
      '3',
      '--',
      'sh',
      '-c',
      `test -n "$GBRAIN_EVAL_ACTUAL_COST_FILE" && printf '{"cost_usd": 0.5, "note":"x"}' > "$GBRAIN_EVAL_ACTUAL_COST_FILE"`,
    ]);
    expect(r.status).toBe(0);
    expect(ledgerRows()[0].cost_usd).toBe(0.5);
  });

  test('command failure: exit code propagates and is recorded', () => {
    const r = run(['75', '1', '--', 'sh', '-c', 'exit 7']);
    expect(r.status).toBe(7);
    expect(ledgerRows()[0].exit_code).toBe(7);
  });

  test('command text with quotes and backslashes yields valid JSON', () => {
    const r = run(['75', '1', '--', 'sh', '-c', 'echo "a \\"b\\" \\\\ c"']);
    expect(r.status).toBe(0);
    const rows = ledgerRows(); // JSON.parse would have thrown on a bad line
    expect(rows.length).toBe(1);
    expect(String(rows[0].command)).toContain('echo');
  });

  test('usage errors exit 2 (missing --, non-numeric cap)', () => {
    expect(run(['75', '1', 'true']).status).toBe(2);
    expect(run(['abc', '1', '--', 'true']).status).toBe(2);
    expect(run(['75', 'x', '--', 'true']).status).toBe(2);
    expect(ledgerRows().length).toBe(0);
  });

  test('ledger rows accumulate across runs and drive the next decision', () => {
    expect(run(['10', '6', '--', 'true']).status).toBe(0);
    expect(run(['10', '3', '--', 'true']).status).toBe(0);
    const r = run(['10', '2', '--', 'true']); // 9 + 2 > 10
    expect(r.status).toBe(3);
    expect(ledgerRows().length).toBe(2);
  });

  // ── fail-closed ledger integrity ─────────────────────────────────────────

  test('a truncated ledger line refuses to launch (exit 3) naming the line, and never runs the command', () => {
    writeFileSync(
      ledger,
      '{"ts":"2026-09-04T00:00:00Z","estimate_usd":3,"cost_usd":3.25,"exit_code":0,"command":"x"}\n' +
        '{"ts":"2026-09-04T00:00:01Z","estimate_usd":70,"cost_usd":70\n', // killed mid-write: no closing brace
    );
    const marker = join(dir, 'should-not-exist.txt');
    const r = run(['75', '1', '--', 'sh', '-c', `echo no > "${marker}"`]);
    expect(r.status).toBe(3);
    expect(existsSync(marker)).toBe(false);
    expect(r.stderr).toContain('unparseable line(s)');
    expect(r.stderr).toContain('line numbers: 2');
    // nothing appended
    expect(readFileSync(ledger, 'utf-8').split('\n').filter((l) => l.length > 0).length).toBe(2);
  });

  test('a string-typed cost_usd is unparseable, not silently $0 (exit 3, no launch)', () => {
    // Pre-fix this row was dropped by the grep and the ledger read as $3.25 —
    // fail OPEN. The 70 in the string row is exactly the spend that would blow the cap.
    writeFileSync(ledger, '{"cost_usd":3.25}\n{"cost_usd":"70"}\n{"cost_usd":1}\n');
    const marker = join(dir, 'should-not-exist.txt');
    const r = run(['75', '1', '--', 'sh', '-c', `echo no > "${marker}"`]);
    expect(r.status).toBe(3);
    expect(existsSync(marker)).toBe(false);
    expect(r.stderr).toContain('line numbers: 2');
  });

  test('a signed / negative ledger cost is unparseable too (a negative row would drive the total backwards)', () => {
    writeFileSync(ledger, '{"cost_usd":-70}\n{"cost_usd":+1}\n{"cost_usd":2}\n');
    const r = run(['75', '1', '--', 'true']);
    expect(r.status).toBe(3);
    expect(r.stderr).toContain('2 unparseable line(s) out of 3');
    expect(r.stderr).toContain('line numbers: 1,2');
  });

  test('blank lines are tolerated; a valid ledger with whitespace and exponent forms sums exactly', () => {
    writeFileSync(ledger, '\n{"cost_usd": 1.5 , "x":1}\n\n{ "cost_usd":2.5e0}\n   \n');
    const r = run(['5', '1', '--', 'true']);
    expect(r.status).toBe(0);
    expect(r.stderr).toContain('ledger $4.000000 (2 row(s))');
    expect(run(['5', '1', '--', 'true']).status).toBe(3); // 5 + 1 > 5
  });

  // ── unsigned-only amounts + %.6f normalization ──────────────────────────

  test('signed or malformed estimate / cap is a usage error (exit 2), nothing launches, nothing is written', () => {
    const marker = join(dir, 'should-not-exist.txt');
    for (const [cap, est] of [
      ['75', '-1'],
      ['75', '+1'],
      ['-75', '1'],
      ['75', '1e'],
      ['75', 'e5'],
      ['75', '1.2.3'],
      ['75', 'NaN'],
      ['75', 'inf'],
    ]) {
      const r = run([cap, est, '--', 'sh', '-c', `echo no > "${marker}"`]);
      expect(r.status, `${cap} ${est}`).toBe(2);
      expect(r.stderr, `${cap} ${est}`).toContain('not an unsigned number');
    }
    expect(existsSync(marker)).toBe(false);
    expect(ledgerRows().length).toBe(0);
  });

  test("estimate forms '1.', '.5' and '2e-1' are accepted and written as valid JSON numbers", () => {
    expect(run(['75', '1.', '--', 'true']).status).toBe(0);
    expect(run(['75', '.5', '--', 'true']).status).toBe(0);
    expect(run(['75', '2e-1', '--', 'true']).status).toBe(0);
    const rows = ledgerRows(); // JSON.parse would throw on `1.` or `.5`
    expect(rows.map((r) => r.estimate_usd)).toEqual([1, 0.5, 0.2]);
    expect(rows.map((r) => r.cost_usd)).toEqual([1, 0.5, 0.2]);
    const raw = readFileSync(ledger, 'utf-8');
    expect(raw).toContain('"estimate_usd":1.000000,"cost_usd":1.000000');
    expect(raw).toContain('"estimate_usd":0.500000,"cost_usd":0.500000');
    // …and the next decision sums them (1.7) correctly.
    const r = run(['2', '0.5', '--', 'true']); // 1.7 + 0.5 > 2
    expect(r.status).toBe(3);
    expect(r.stderr).toContain('ledger $1.700000');
  });

  test("a cost file of '1.' / '.5' is normalized before it is written", () => {
    const costFile = join(dir, 'actual.txt');
    const r = run(['75', '3', '--', 'sh', '-c', `printf '.5' > "$GBRAIN_EVAL_ACTUAL_COST_FILE"`], {
      GBRAIN_EVAL_ACTUAL_COST_FILE: costFile,
    });
    expect(r.status).toBe(0);
    expect(readFileSync(ledger, 'utf-8')).toContain('"cost_usd":0.500000');
    expect(ledgerRows()[0].cost_usd).toBe(0.5);
  });

  test('a negative, signed, zero, or string-typed actual cost falls back to the estimate', () => {
    const cases: Array<[string, string]> = [
      ['-2', 'unreadable'], // signed bare number: rejected by is_number, not parsed as -2
      ['+2', 'unreadable'],
      ['0', 'non-positive'],
      ['{"cost_usd":"2"}', 'unreadable'],
      ['{"cost_usd":-0.5}', 'unreadable'],
      ['{"cost_usd":0}', 'non-positive'],
      ['garbage', 'unreadable'],
    ];
    for (const [payload, note] of cases) {
      writeFileSync(ledger, '');
      const costFile = join(dir, 'actual.txt');
      const r = run(['75', '3', '--', 'sh', '-c', `printf '%s' '${payload}' > "$GBRAIN_EVAL_ACTUAL_COST_FILE"`], {
        GBRAIN_EVAL_ACTUAL_COST_FILE: costFile,
      });
      expect(r.status, payload).toBe(0);
      expect(r.stderr, payload).toContain(note);
      expect(r.stderr, payload).toContain('estimate');
      const rows = ledgerRows();
      expect(rows.length, payload).toBe(1);
      expect(rows[0].cost_usd, payload).toBe(3); // the estimate, never a negative or zero row
      rmSync(costFile, { force: true });
    }
  });

  // ── ledger existence ─────────────────────────────────────────────────────

  test('a missing ledger refuses to launch (exit 3) and prints the resolved path', () => {
    rmSync(ledger, { force: true });
    const marker = join(dir, 'should-not-exist.txt');
    const r = run(['75', '1', '--', 'sh', '-c', `echo no > "${marker}"`]);
    expect(r.status).toBe(3);
    expect(existsSync(marker)).toBe(false);
    expect(existsSync(ledger)).toBe(false); // NOT silently created
    expect(r.stderr).toContain('ledger does not exist');
    expect(r.stderr).toContain(ledger);
    expect(r.stderr).toContain('GBRAIN_EVAL_SPEND_LEDGER_INIT=1');
  });

  test('the default ledger path (no env override) is also required to exist', () => {
    rmSync(ledger, { force: true });
    const r = spawnSync('bash', [SCRIPT, '75', '1', '--', 'true'], {
      cwd: ROOT,
      encoding: 'utf-8',
      env: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: dir, TMPDIR: dir },
    });
    expect(r.status).toBe(3);
    expect(r.stderr).toContain(join(dir, 'gbrain-lme-receipts', 'spend.jsonl'));
  });

  test('GBRAIN_EVAL_SPEND_LEDGER_INIT=1 creates the ledger with a loud NEW LEDGER line, then launches', () => {
    rmSync(join(dir, 'receipts'), { recursive: true, force: true });
    const marker = join(dir, 'ran.txt');
    const r = run(['75', '1', '--', 'sh', '-c', `echo ok > "${marker}"`], { GBRAIN_EVAL_SPEND_LEDGER_INIT: '1' });
    expect(r.status).toBe(0);
    expect(existsSync(marker)).toBe(true);
    expect(r.stderr).toContain('NEW LEDGER');
    expect(r.stderr).toContain(ledger);
    expect(ledgerRows().length).toBe(1);
    // A second run with INIT still set does NOT re-announce (the file exists now).
    const r2 = run(['75', '1', '--', 'true'], { GBRAIN_EVAL_SPEND_LEDGER_INIT: '1' });
    expect(r2.status).toBe(0);
    expect(r2.stderr).not.toContain('NEW LEDGER');
    expect(ledgerRows().length).toBe(2);
  });
});
