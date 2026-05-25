# Retrieval Evaluation

Each variant evaluated against `backend/tests/fixtures/retrieval_eval_queries.json`
(15 queries: 4 short_kw / 4 vocab_gap / 4 long_specific / 3 cross_lang).

Hit Rate@5 = at least one expected (source, page) in top-5;  
MRR = 1 / rank of first hit (0 if no hit).

## Comparison

| Variant | top_k | Hit@5 | MRR |
|---|---|---|---|
| Baseline — Dense only (nomic-embed-text + Chroma) | 10 | **0.800** | **0.639** |
| +B — Hybrid (BM25 + Dense, RRF k=60) | 10 | **0.867** | **0.619** |
| +C — Hybrid + jina-reranker-v2 (depth=20→top-K) | 10 | **0.933** | **0.822** |

## Baseline — Dense only (nomic-embed-text + Chroma)

Top-K = 10.  Aggregate: Hit@5 = **0.800**, MRR = **0.639**.

| # | Class | Query (truncated) | Hit Rank | Hit@5 | MRR |
|---|---|---|---|---|---|
| 1 | short_kw | hallucinations | 1 | 1.0 | 1.000 |
| 2 | short_kw | token budget | 1 | 1.0 | 1.000 |
| 3 | short_kw | ReAct | — | 0.0 | 0.000 |
| 4 | short_kw | temperature | 3 | 1.0 | 0.333 |
| 5 | vocab_gap | What is prompt engineering and why is it important? | 1 | 1.0 | 1.000 |
| 6 | vocab_gap | How does a large language model work? | 2 | 1.0 | 0.500 |
| 7 | vocab_gap | What is context engineering and how does it improve LLM | — | 0.0 | 0.000 |
| 8 | vocab_gap | What are the key methods for evaluating LLM application | 1 | 1.0 | 1.000 |
| 9 | long_specific | Can you explain the autoregressive nature of text gener | 1 | 1.0 | 1.000 |
| 10 | long_specific | According to the text, what are the first three steps i | — | 0.0 | 0.000 |
| 11 | long_specific | What are the different tiers of importance listed for p | 1 | 1.0 | 1.000 |
| 12 | long_specific | Given that IT support assistance in this scenario invol | 1 | 1.0 | 1.000 |
| 13 | cross_lang | 什么是 prompt engineering 提示工程? | 2 | 1.0 | 0.500 |
| 14 | cross_lang | 解释少样本学习 few-shot learning | 1 | 1.0 | 1.000 |
| 15 | cross_lang | 如何评估 LLM 应用? | 4 | 1.0 | 0.250 |

## +B — Hybrid (BM25 + Dense, RRF k=60)

Top-K = 10.  Aggregate: Hit@5 = **0.867**, MRR = **0.619**.

| # | Class | Query (truncated) | Hit Rank | Hit@5 | MRR |
|---|---|---|---|---|---|
| 1 | short_kw | hallucinations | 1 | 1.0 | 1.000 |
| 2 | short_kw | token budget | 1 | 1.0 | 1.000 |
| 3 | short_kw | ReAct | 2 | 1.0 | 0.500 |
| 4 | short_kw | temperature | 3 | 1.0 | 0.333 |
| 5 | vocab_gap | What is prompt engineering and why is it important? | 1 | 1.0 | 1.000 |
| 6 | vocab_gap | How does a large language model work? | 4 | 1.0 | 0.250 |
| 7 | vocab_gap | What is context engineering and how does it improve LLM | — | 0.0 | 0.000 |
| 8 | vocab_gap | What are the key methods for evaluating LLM application | 3 | 1.0 | 0.333 |
| 9 | long_specific | Can you explain the autoregressive nature of text gener | 1 | 1.0 | 1.000 |
| 10 | long_specific | According to the text, what are the first three steps i | 5 | 1.0 | 0.200 |
| 11 | long_specific | What are the different tiers of importance listed for p | 1 | 1.0 | 1.000 |
| 12 | long_specific | Given that IT support assistance in this scenario invol | 1 | 1.0 | 1.000 |
| 13 | cross_lang | 什么是 prompt engineering 提示工程? | 2 | 1.0 | 0.500 |
| 14 | cross_lang | 解释少样本学习 few-shot learning | 1 | 1.0 | 1.000 |
| 15 | cross_lang | 如何评估 LLM 应用? | 6 | 0.0 | 0.167 |

## +C — Hybrid + jina-reranker-v2 (depth=20→top-K)

Top-K = 10.  Aggregate: Hit@5 = **0.933**, MRR = **0.822**.

| # | Class | Query (truncated) | Hit Rank | Hit@5 | MRR |
|---|---|---|---|---|---|
| 1 | short_kw | hallucinations | 1 | 1.0 | 1.000 |
| 2 | short_kw | token budget | 1 | 1.0 | 1.000 |
| 3 | short_kw | ReAct | 1 | 1.0 | 1.000 |
| 4 | short_kw | temperature | 1 | 1.0 | 1.000 |
| 5 | vocab_gap | What is prompt engineering and why is it important? | 1 | 1.0 | 1.000 |
| 6 | vocab_gap | How does a large language model work? | 2 | 1.0 | 0.500 |
| 7 | vocab_gap | What is context engineering and how does it improve LLM | — | 0.0 | 0.000 |
| 8 | vocab_gap | What are the key methods for evaluating LLM application | 1 | 1.0 | 1.000 |
| 9 | long_specific | Can you explain the autoregressive nature of text gener | 1 | 1.0 | 1.000 |
| 10 | long_specific | According to the text, what are the first three steps i | 1 | 1.0 | 1.000 |
| 11 | long_specific | What are the different tiers of importance listed for p | 1 | 1.0 | 1.000 |
| 12 | long_specific | Given that IT support assistance in this scenario invol | 1 | 1.0 | 1.000 |
| 13 | cross_lang | 什么是 prompt engineering 提示工程? | 2 | 1.0 | 0.500 |
| 14 | cross_lang | 解释少样本学习 few-shot learning | 1 | 1.0 | 1.000 |
| 15 | cross_lang | 如何评估 LLM 应用? | 3 | 1.0 | 0.333 |
