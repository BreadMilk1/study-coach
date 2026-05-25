# Phase 2.0 — Retrieval Foundation Rebuild

> Pre-flight for Phase 2 (agent-ify old four features). The Phase 1 closed loop
> returned answers that "couldn't find content actually in the PDF" — diagnosis
> showed not the model but the **retrieval + prompt** pipeline. P2.0 rebuilds
> that foundation with TDD discipline before any agent tools are added on top.

## Outcome

Cumulative delta over Phase 1 baseline (15 ground-truth queries from the
HKBU corpus, top_k = 10 for measurement headroom; production uses top_k = 5):

| Metric | Phase 1 (single-route Dense, top_k=3, bare prompt) | Phase 2.0 (Hybrid + Reranker + concat-split, top_k=5, grounded prompt + Citation 5-field) | Delta |
|---|---|---|---|
| Hit Rate @ 5 | 0.733 | **0.933** | **+27%** |
| MRR | 0.633 | **0.822** | **+30%** |

Live per-variant data, refreshed by `scripts/run_baseline_eval.py`:
see [RETRIEVAL_EVAL.md](./RETRIEVAL_EVAL.md).

## Diagnosis (Phase 1 root causes, ordered by lever size)

1. **BM25 absent**. `retriever.py` was Dense-only (33 lines) despite ROADMAP
   P1 stating "Chroma backend; reuse BM25 hybrid scoring". Dense-embedding
   blind spot on proper nouns: `ReAct`, `Cycle of Quality` returned rank
   `—` even though Topic 8 has 15 pages on ReAct.
2. **`top_k = 3`** in `graph.py`. Hard ceiling on recall; correct chunks at
   rank 4-10 invisible to the LLM.
3. **Chunks per-page**. `process_pdf` split text page-by-page; concepts
   spanning a page boundary (e.g., last sentence of page 3 → first of page
   4) were torn into two unconnected chunks.
4. **Bare prompt**. Context block was `[N] {content}` only — no source, no
   page, no system role, no grounded约束. Citations were 3-field
   `{chunk_id, source, page}`, not aligned with ARCHITECTURE.md Citation
   TypedDict (4-field `{chunk_id, page, span_start, span_end}`).
5. **`nomic-embed-text` is English-mono**. Chinese queries lost semantic
   alignment against English PDFs.

## Architecture changes

| Module | Change |
|---|---|
| `app/rag/hybrid_retriever.py` | NEW — `BM25Index` + `HybridRetriever` (BM25 ⊕ Dense via RRF, k=60); `rank_bm25` dep added |
| `app/rag/reranking_retriever.py` | NEW — `RerankingRetriever` (depth=20 → top-K) + `FastembedReranker` adapter; `fastembed` dep added; default model `jinaai/jina-reranker-v2-base-multilingual` (1.11 GB, multilingual) |
| `app/rag/document_processor.py` | `process_pdf` rewritten: concat all PDF pages → split → map each chunk back to origin page via character-offset spans |
| `app/agent/prompt.py` | NEW — shared `build_prompt` / `build_citations` / `format_context` / `SYSTEM_INSTRUCTION`; grounded约束 + source/page injection |
| `app/agent/graph.py` | Imports from `prompt.py`; top_k 3 → 5 |
| `app/agent/state.py` | `CitationRef` → `Citation`; added `source` + `span_start` + `span_end` (5-field, aligned with updated ARCHITECTURE.md) |
| `app/api/routes.py` | Removed local `_format_context`; uses shared `build_prompt` / `build_citations`; top_k 3 → 5 |
| `app/main.py` | `_build_default_retriever` now wires `Retriever → HybridRetriever → RerankingRetriever`; rebuilds BM25 in-memory from existing Chroma chunks on startup |
| `docs/ARCHITECTURE.md` | Citation TypedDict updated 4-field → 5-field (adds `source`) |
| `scripts/build_eval_queries.py` | NEW — one-shot helper, half-auto-labels HKBU 12 queries + 3 cross-lang queries via grep |
| `scripts/run_baseline_eval.py` | Generalized to multi-variant runner; results saved per-variant JSON, markdown regenerated side-by-side |
| `tests/eval/`, `tests/fixtures/eval_results/` | NEW — eval harness tests, per-variant result JSONs |

## Test impact

| | Before P2.0 | After P2.0 |
|---|---|---|
| Total tests | 19 | **36** |
| New tests added | — | retrieval eval (5), hybrid retriever (5), reranker (4), doc processor cross-page (1), graph prompt + Citation (3 modified/added) |
| Runtime | ~0.4 s | ~0.3 s (no network / model loads under pytest; reranker / Ollama mocked) |
| TDD discipline | partial | every prod line driven by failing test first (red → green → refactor) per `superpowers:test-driven-development` |

## Known limitations / next moves

1. **`context engineering` query still miss** — fixture's ground-truth pins
   only Topic 5 page 1 (chapter title); the chapter body uses `context`
   alone. Either loose-grain fixture relabel (expand to whole Topic 5 PDF)
   or accept the lexical-bias measurement.
2. **Reranker MRR -0.078 on v2 chunks vs v1 chunks** — concat-then-split
   produces longer chunks, introducing noise into the cross-encoder pairwise
   scoring (more text per pair, less focus). Real impact requires
   answer-quality eval (LLM-as-judge), deferred to P2.1+ when Judge Guard
   subgraph lands.
3. **Span fields are placeholders** (`span_start = 0`,
   `span_end = len(content)`). Character-level answer spans require Phase
   2.2 HyDE / quote-extract tools.
4. **No end-to-end answer-quality eval yet** — Hit@5 measures retrieval
   only. P2.1 should add LLM-as-judge over the full chat response.
5. **Frontend unchanged** — chat view still receives `chunk_id / source /
   page` in citations (no UI change required for P2.0);
   `span_start / span_end` are passed through but UI does not yet
   highlight.

## How to reproduce

```bash
cd study-coach/backend
uv sync
ollama pull nomic-embed-text   # if not already

# Build eval Q&A fixture (one-shot, idempotent)
uv run python scripts/build_eval_queries.py

# Run each variant (first `--variant baseline --reindex` will ingest 14 PDFs;
# subsequent calls reuse the eval Chroma index)
uv run python scripts/run_baseline_eval.py --variant baseline --reindex
uv run python scripts/run_baseline_eval.py --variant hybrid_bm25_rrf
uv run python scripts/run_baseline_eval.py --variant hybrid_bm25_rrf_reranker

# Output: docs/RETRIEVAL_EVAL.md (regenerated side-by-side from JSON results)

# Full test suite (no network / model loads needed)
uv run pytest                   # 36 passing
```
