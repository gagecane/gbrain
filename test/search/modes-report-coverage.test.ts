/**
 * #4604 — `gbrain search modes` resolves EVERY ModeBundle knob.
 *
 * buildModesReport used to hardcode a 12-knob literal array, so live
 * overrides like search.reranker.* and search.relational_retrieval were
 * invisible on the dashboard while they steered every real search. The knob
 * list now derives from KNOB_DESCRIPTIONS (a Record over every ModeBundle
 * key — the type system forces a description, and therefore a dashboard row,
 * for each new knob). Drift guard: resolved keys == bundle keys, forever.
 *
 * Also pins the #4604 honesty note: the report labels itself as brain-level
 * resolution only (per-call SearchOpts overrides are not represented).
 */

import { describe, test, expect } from 'bun:test';
import { buildModesReport, formatKnobValue, KNOB_DESCRIPTIONS, MODES_REPORT_PER_CALL_NOTE } from '../../src/core/search/modes-report.ts';
import { _exports_for_test as searchCmd } from '../../src/commands/search.ts';
import { MODE_BUNDLES } from '../../src/core/search/mode.ts';
import type { BrainEngine } from '../../src/core/engine.ts';

const stubEngine = (configRows: Record<string, string> = {}) =>
  ({
    getConfig: async (k: string) => configRows[k] ?? null,
  }) as unknown as BrainEngine;

describe('#4604 buildModesReport — full-bundle knob coverage', () => {
  test('resolved covers EVERY ModeBundle key (drift guard)', async () => {
    const report = await buildModesReport(stubEngine());
    const bundleKeys = Object.keys(MODE_BUNDLES.balanced).sort();
    expect(Object.keys(report.resolved).sort()).toEqual(bundleKeys);
    // And every row carries value/source/description.
    for (const [k, row] of Object.entries(report.resolved)) {
      expect(typeof row.source).toBe('string');
      expect(typeof row.source_detail).toBe('string');
      expect(row.description).toBe(KNOB_DESCRIPTIONS[k as keyof typeof KNOB_DESCRIPTIONS]);
    }
  });

  test('the previously-hardcoded-out knobs now resolve (reranker + relational)', async () => {
    const report = await buildModesReport(stubEngine());
    // The 12-knob literal array omitted all of these — the issue's headline.
    for (const k of [
      'reranker_enabled',
      'reranker_model',
      'reranker_top_n_in',
      'reranker_top_n_out',
      'reranker_timeout_ms',
      'relationalRetrieval',
      'relational_retrieval_depth',
      'graph_signals',
      'autocut',
      'title_boost',
    ] as const) {
      expect(report.resolved[k]).toBeDefined();
    }
  });

  test('a live config override on a formerly-invisible knob is attributed', async () => {
    const report = await buildModesReport(stubEngine({
      'search.relational_retrieval': 'false',
      'search.reranker.enabled': 'true',
    }));
    expect(report.resolved.relationalRetrieval.value).toBe(false);
    expect(report.resolved.relationalRetrieval.source).toBe('override');
    expect(report.resolved.reranker_enabled.value).toBe(true);
    expect(report.resolved.reranker_enabled.source).toBe('override');
  });

  test('per_call_note labels the report as brain-level resolution only', async () => {
    const report = await buildModesReport(stubEngine());
    expect(report.per_call_note).toBe(MODES_REPORT_PER_CALL_NOTE);
    expect(report.per_call_note).toContain('Per-call');
  });
});

describe('formatKnobValue — a legitimate null is not "(undefined)" (adversarial finding)', () => {
  test('null renders distinctly per knob; undefined keeps "(undefined)"; values stringify', () => {
    expect(formatKnobValue('expansion_variant_budget', null)).toBe('legacy (null)');
    expect(formatKnobValue('reranker_top_n_out', null)).toBe('no truncate (null)');
    expect(formatKnobValue('tokenBudget', null)).toBe('(null)');
    expect(formatKnobValue('tokenBudget', undefined)).toBe('(undefined)');
    expect(formatKnobValue('expansion_variant_budget', undefined)).toBe('(undefined)');
    expect(formatKnobValue('expansion_variant_budget', 0.5)).toBe('0.5');
    expect(formatKnobValue('expansion', false)).toBe('false');
  });

  test('`gbrain search modes` text at bundle defaults prints expansion_variant_budget = legacy (null)', async () => {
    const report = await buildModesReport(stubEngine());
    expect(report.resolved.expansion_variant_budget.value).toBeNull();
    const text = searchCmd.formatModesText(report);
    const row = text.split('\n').find((l) => /^\s+expansion_variant_budget\s+=/.test(l));
    expect(row).toBeDefined();
    expect(row).toContain('= legacy (null)');
    expect(row).not.toContain('(undefined)');
    // reranker_top_n_out's bundle-default null gets its own label too.
    const tn = text.split('\n').find((l) => /^\s+reranker_top_n_out\s+=/.test(l));
    expect(tn).toContain('= no truncate (null)');
  });

  test('a numeric config override renders the number, and the literal legacy renders legacy (null) as an override', async () => {
    const half = await buildModesReport(stubEngine({ 'search.expansion_variant_budget': '0.5' }));
    expect(searchCmd.formatModesText(half)).toMatch(/expansion_variant_budget\s+= 0\.5\s+\[/);
    const legacy = await buildModesReport(stubEngine({ 'search.expansion_variant_budget': 'legacy' }));
    expect(legacy.resolved.expansion_variant_budget.source).toBe('override');
    expect(searchCmd.formatModesText(legacy)).toMatch(/expansion_variant_budget\s+= legacy \(null\)\s+\[/);
  });
});
