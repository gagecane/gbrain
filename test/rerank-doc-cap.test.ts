/**
 * Reranker document cap — self-hosted rerankers (llama-server, TEI) have a
 * small physical batch and 500 on an oversized chunk; hosted rerankers
 * truncate server-side. applyReranker now caps each document to a char AND
 * token ceiling before the call. Hex-dense text (index pages, hashes) is the
 * hazard the token ceiling exists for: ~1 char/token, so a char cap alone
 * still overflows the batch.
 *
 * Pins:
 *  - short prose passes through untouched
 *  - a long document is cut to the char ceiling
 *  - a hex-dense document under the char ceiling is still cut to the token ceiling
 *  - the cap is applied on the applyReranker path (via the rerankerFn seam)
 */

import { describe, test, expect } from 'bun:test';
import * as rerankMod from '../src/core/search/rerank.ts';
import type { RerankerOpts } from '../src/core/search/rerank.ts';
// Namespace import on purpose: with the fix reverted, capRerankDoc is undefined and the
// tests below FAIL at the call site instead of the file crashing at import (a vacuous
// non-zero exit is not a discrimination result — see scripts/check-test-discriminates.sh).
const { applyReranker, capRerankDoc } = rerankMod;
import { estimateTokens } from '../src/core/chunkers/token-estimate.ts';
import type { SearchResult } from '../src/core/types.ts';
import type { RerankInput } from '../src/core/ai/gateway.ts';

const MAX_DOC_CHARS = 6000;
const MAX_DOC_TOKENS = 1400;

// Deterministic hex-dense text: every 8 chars is a distinct nibble run, so
// cl100k cannot merge it into long tokens.
function hexDense(chars: number): string {
  let out = '';
  let i = 0;
  while (out.length < chars) {
    out += (i++ * 2654435761 >>> 0).toString(16).padStart(8, '0') + ' ';
  }
  return out.slice(0, chars);
}

function result(chunk_text: string, i = 0): SearchResult {
  return {
    slug: `doc/${i}`,
    page_id: i + 1,
    title: `Doc ${i}`,
    type: 'note',
    chunk_text,
    chunk_source: 'compiled_truth',
    chunk_id: i + 100,
    chunk_index: 0,
    score: 1,
    stale: false,
  };
}

describe('capRerankDoc', () => {
  test('short prose passes through unchanged', () => {
    const doc = 'A short chunk about channel pricing strategy.';
    expect(capRerankDoc(doc)).toBe(doc);
  });

  test('a long document is cut to the char ceiling', () => {
    const doc = 'word '.repeat(5000); // 25k chars, ~5k tokens
    const capped = capRerankDoc(doc);
    expect(capped.length).toBeLessThanOrEqual(MAX_DOC_CHARS);
    expect(estimateTokens(capped)).toBeLessThanOrEqual(MAX_DOC_TOKENS);
  });

  test('hex-dense text under the char ceiling is still cut to the token ceiling', () => {
    const doc = hexDense(MAX_DOC_CHARS - 100);
    // Positive control: the input itself overflows the token ceiling.
    expect(estimateTokens(doc)).toBeGreaterThan(MAX_DOC_TOKENS);
    const capped = capRerankDoc(doc);
    expect(capped.length).toBeLessThan(doc.length);
    expect(estimateTokens(capped)).toBeLessThanOrEqual(MAX_DOC_TOKENS);
    expect(doc.startsWith(capped)).toBe(true); // a prefix, never rewritten
  });
});

describe('applyReranker applies the cap to every document', () => {
  test('the reranker receives capped documents', async () => {
    const results = [result(hexDense(20_000), 0), result('tiny', 1)];
    let seen: RerankInput | null = null;
    const opts: RerankerOpts = {
      enabled: true,
      topNIn: 30,
      topNOut: null,
      rerankerFn: async (input) => {
        seen = input;
        return [{ index: 0, relevanceScore: 0.9 }, { index: 1, relevanceScore: 0.1 }];
      },
    };
    await applyReranker('q', results, opts);
    expect(seen).not.toBeNull();
    const docs = seen!.documents;
    expect(docs).toHaveLength(2);
    expect(docs[0].length).toBeLessThanOrEqual(MAX_DOC_CHARS);
    expect(estimateTokens(docs[0])).toBeLessThanOrEqual(MAX_DOC_TOKENS);
    expect(docs[1]).toBe('tiny');
  });
});
