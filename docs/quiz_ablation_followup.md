# P2.3 update: does the agent loop thesis transfer to a different task?

> A follow-up to `agent_loop_vs_deterministic.md`. Same 4 models, same matrix shape,
> a new task with a markedly stricter schema. Three predictions tested, one new
> dimension surfaced by accident.

## What I went back to test

The original P2.2 post ended with three predictions about how the same ablation would run on the Quiz path:

1. The `gemma3:4b agent_loop` collapse should repeat — the model still has no tools capability in the Ollama manifest.
2. The persistence rescue effect (P2.2 §Finding 2) may *not* manifest — "quiz tools are simpler."
3. (Implied) Quiz is a "more constrained tool surface" than Plan.

The Quiz task differs from Plan in one structurally important way: it needs corpus-specific facts, not generic study-plan structure. So the experiment has more ways to surprise.

## The matrix

Same setup as P2.2 — 4 models × 2 modes × 12 queries × 3 runs + a gemma4:e4b thinking-ON appendix = 396 records, ~5 hours on a 16GB Apple Silicon Mac, ~$3 of MiniMax-M2.7 cloud judging. Same dual judge (qwen2.5:7b local + MiniMax-M2.7 cloud). Topics overlap with P2.2 by design (HyDE / BM25 / reranking / chunking / etc.) so future cross-task analyses are possible.

Full numbers at `docs/EVAL.md` §"P2.3 Quiz Agent Loop Ablation".

## Prediction 1: gemma3:4b collapse — confirmed

`gemma3:4b agent_loop` produced 27/36 `llm_call_failed` records (75%, the rest were multi-turn GRADE records dispatched to deterministic per design). Ollama API still returns 400 `does not support tools` on every request. The degrade handler caught it cleanly.

This is the cleanest carry-over from P2.2. The model has to be trained for tools; no harness fixes that.

## Prediction 2: schema rescue — mechanism inverted, not absent

This is where the predictions broke. I expected "schema rescue won't help because Quiz tools are simpler"; the actual result was "schema rescue **reverses direction** because Quiz tools are *stricter*, not simpler."

P2.2's `Milestone` schema has one required field (`title`); the rest are optional. P2.3's `QuizQuestionPersist` requires five fields, constrains `answer` to `Literal["A","B","C","D"]`, requires `options` to be a list of exactly 4 strings, and runs a custom `field_validator` checking each option starts with `"A) "/"B) "/"C) "/"D) "`. Four orthogonal constraints, all enforced by the tool wrapper before any data hits the DB.

The persistence numbers flipped:

| qwen3.5:4b | Plan (P2.2) | Quiz (P2.3) |
|---|---|---|
| deterministic | 10% | **75%** |
| agent_loop | 86% | **50%** |

Why? Two effects in opposite directions:

- **Deterministic Quiz benefits from a tolerant JSON parser**. `generate_quiz` uses a 3-tier regex (fenced ```json``` → bare array → first-`{...}`-block) that accepts what small models naturally emit even when their JSON is slightly malformed. 75% persistence even with `reasoning=False`.
- **Agent-loop Quiz can't satisfy the strict schema in 6 retries**. qwen3.5:4b hits the `persist_quiz_question` schema rejection, gets a `{"error": "..."}` ToolMessage back, retries with marginally different args, and still fails. Mean tool-error rate per run: **1.7**. Result: 25% `budget_exhausted`.

So P2.2 §Finding 2 ("tool schemas substitute for in-model reasoning") needs a qualifier. **Schema-as-harness is conditional on a 2D fit between schema strictness and model self-correction capability.** Plan's schema was strict enough to surface formatting errors and loose enough that the model could satisfy it; Quiz's schema is strict enough to reject everything and tight enough that the model can't recover.

## The hidden upside

Here's the thing that made me keep going instead of declaring the experiment "failed." When agent_loop *does* succeed (filtered to `exit_reason="natural_stop"`), its output quality is **higher** than deterministic by +0.05 to +0.16 on cloud judge across the three working model tiers. The strict schema, when satisfied, guarantees a well-formed MCQ; deterministic mode persists more often but produces some malformed quizzes that judges score down.

This is the classic precision-recall trade-off, made concrete: agent_loop optimizes per-success quality, deterministic optimizes completion rate. Mode choice depends on which axis matters for the deployment.

## Prediction 3: "quiz tools are simpler" — false

The Quiz tool surface IS narrower (2 tools vs Plan's 5). But the *schema strictness per tool* is much higher. "Simpler" turned out to be the wrong word for what's varying between the two tasks — and the variable that actually matters (schema strictness) cuts the opposite direction from my mental model.

Worth flagging this honestly: my pre-experiment intuition was wrong on the direction. The data corrected me.

## The dimension P2.2 didn't anticipate

The first time I ran the P2.3 matrix, I had `retriever=None` (a copy-paste fallout from the P2.2 spec template). The data looked weird: agent_loop refusal rate spiked to 21-39% on the well-aligned cells. qwen3.5:4b agent_loop wrote things like:

> *"I'm unable to retrieve any information about HyDE from the user's PDF source material. Could you please provide more context..."*

— 14 out of 36 runs. Meanwhile deterministic mode with no retriever silently produced quizzes from training-distribution recall.

This is a structural alignment property the original post didn't measure: agent loops with `retriever_search` give the model a clean signal when grounding is missing, and well-aligned models act on it by declining to fabricate. Deterministic prompt paths have no equivalent channel — the empty context section just gets filled in from training.

I rewired the eval to use the production retriever and the refusal rate dropped to 0% across all working cells. The pilot data is preserved at `results_no_retriever.jsonl` as a methodology baseline. It's worth flagging because the production deployment implication is concrete: for any Study Coach task where empty-corpus fabrication is a correctness risk (factual quiz on user materials), agent_loop should be the default. For tasks where corpus grounding isn't essential (study plan structure), deterministic is fine.

This is a 4th dimension to the original three predictions. It's the most directly actionable finding for the Study Coach production deployment, and the one I didn't anticipate at all.

## Where the thesis lands now

The `agent_loop_vs_deterministic` framing held up structurally — the 4-tier model capability story (gemma3 / qwen2.5 / qwen3.5 / gemma4 with thinking-on) still predicts where the agent loop helps and where it hurts. What changed:

- **Schema strictness is a second dimension that interacts with model capability.** P2.2 underweighted this. The same model can be "above the line" or "below the line" depending on the schema it's trying to satisfy.
- **Tool-feedback channels create an alignment safety property** independent of the schema-rescue mechanism. P2.2 didn't have this.
- **Mode choice is a deployment-axis question** (completion-rate vs per-success-quality), not a generic "agent loops are better" or "deterministic is better."

## What I'd do next

- **`max_iter=12`**: re-run only the qwen3.5:4b agent_loop cell at higher budget to isolate "schema strictness" from "retry budget."
- **A non-thinking cloud judge** (GPT-4o-mini) — same item as last time, still deferred.
- **Mixed-mode dispatcher**: pick deterministic vs agent_loop per-query based on whether corpus grounding is essential. This is the production move.

I have the data the original post promised would settle the schema-rescue question. The answer turned out to be more interesting than the prediction.

## Reproduce

```bash
cd study-coach/backend
uv sync
ollama pull gemma3:4b qwen3.5:4b qwen2.5:7b gemma4:e4b
uv run pytest -q                  # 202 tests pass
export MINIMAX_API_KEY=sk-...     # optional, skips cloud judge if absent
uv run python -m app.eval.p2_3_quiz_ablation.run_eval \
  --queries app/eval/p2_3_quiz_ablation/queries.json \
  --output app/eval/p2_3_quiz_ablation/output/results.jsonl \
  --runs 3 \
  --thinking-appendix
```

Total wall time ~5h on a 16GB Apple Silicon Mac. Cost ~$3 MiniMax.

---

*Sister post to [Did I build a real agent?](./agent_loop_vs_deterministic.md). Full numbers and per-cell tables at [EVAL.md §"P2.3 Quiz Agent Loop Ablation"](./EVAL.md). Raw data at `backend/app/eval/p2_3_quiz_ablation/output/`. Repo for this work: Study Coach (refactor of HKBU class project).*
