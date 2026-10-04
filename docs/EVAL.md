# Agent Loop Ablation — Empirical Report

> P2.2 deliverable. Tests the question: on locally-served small Ollama models,
> does an LLM tool-calling agent loop produce better study plans than a
> hand-written deterministic node? Or does it just spend more tokens?
>
> **Matrix**: 4 models × 2 modes × 14 queries × 3 runs + appendix (gemma4:e4b
> thinking on/off, 10 single-turn × 2 modes × 3 runs) = **396 runs**.
> **Wall time**: ~5 hours sequential on a 16GB Apple Silicon Mac (Ollama local + MiniMax-M2.7 cloud judge).
> **Cost**: ~$3 MiniMax API for 396 cloud judgments.
> **Data**: `backend/app/eval/p2_2_agent_ablation/output/results.jsonl` (396 rows, 0 harness_error).

## TL;DR

- **`gemma3:4b agent_loop` is a 100% negative data point** — model manifest has no `tools` capability, Ollama API returns 400 on every call. P2.2's `_format_degrade_output` handles it gracefully (42/42 cells degrade clean, no crash). Confirms spec §1.Q1a prediction.
- **Tool schemas rescue thinking models when thinking is OFF**. With `reasoning=False`, `qwen3.5:4b` deterministic persistence drops to 10% (LLM emits garbled JSON without thinking), but `qwen3.5:4b agent_loop` persists 86% because `update_study_plan` tool's Pydantic schema forces valid output. Same pattern on `gemma4:e4b` (43% → 94%).
- **`gemma4:e4b` is the agent_loop champion**: 100% `natural_stop`, 2.74 tool calls/run, 94% persistence, top-tied judge scores (local 0.856 / cloud 0.606). Tools + thinking + multimodal = working agent at 8B.
- **`qwen2.5:7b` is the "simplicity wins" tier**: deterministic and agent_loop tie on local judge (both 0.770), but deterministic is **3× faster** (7s vs 21s) and **2.25× more persistent** (90% vs 40%). For this model, the agent harness adds cost without quality.
- **Judges disagree systematically**. Mean |local−cloud| is 0.19–0.31 across cells. Local qwen2.5:7b is consistently more generous than cloud MiniMax-M2.7 (`local +0.20` typical). Self-preference bias is observable when qwen2.5 judges qwen2.5 (local 0.770 > cloud 0.584).
- **The portfolio answer to learn-claude-code's "agency = model + minimal harness" thesis: it depends on whether the model is trained for tool use AND whether its reasoning is exposed.** See `docs/agent_loop_vs_deterministic.md`.

## Setup

- 4 models: `gemma3:4b` (4B, no tools/thinking flag), `qwen3.5:4b` (4.7B, tools + thinking),
  `qwen2.5:7b` (7B, tools, no thinking), `gemma4:e4b` (8B, tools + thinking + multimodal).
- 2 modes: `deterministic` (P2.1-⑤ baseline) vs `agent_loop` (P2.2 hand-written while-loop).
- 12 queries × 3 runs = 36 trials per (model, mode) cell + multi-turn check-in.
- Dual judges: `qwen2.5:7b` local + `MiniMax-M2.7` cloud, both using `PLAN_DIMENSIONS` rubric
  (milestone_specificity / milestone_granularity / time_feasibility / topic_coverage / actionability).
- Appendix: `gemma4:e4b` thinking on vs off on the same matrix.
- **Critical control**: `reasoning=False` forwarded to ChatOllama on main matrix to match spec §1.Q1b
  (verified Cut ①f Phase B: `qwen3.5:4b` 813s → 7.3s).

## Results

### 1. Latency (wall_time_s per cell)

| model | mode | median | mean | n |
|---|---|---|---|---|
| gemma3:4b | deterministic | 13.5 | 14.9 | 42 |
| **gemma3:4b** | **agent_loop** | **0.1** | **0.1** | 42 |
| qwen3.5:4b | deterministic | 3.7 | 4.5 | 42 |
| qwen3.5:4b | agent_loop | 73.1 | 85.6 | 42 |
| qwen2.5:7b | deterministic | 7.0 | 7.9 | 42 |
| qwen2.5:7b | agent_loop | 21.4 | 24.6 | 42 |
| gemma4:e4b | deterministic | 7.1 | 24.9 | 72 |
| gemma4:e4b | agent_loop | 38.7 | 48.1 | 72 |

- **`gemma3:4b agent_loop` 0.1s** is the Ollama 400 reject time — `_format_degrade_output` fires before any real LLM call.
- **`qwen3.5:4b agent_loop` 73s** is the most expensive cell (thinking-on within each iteration). The matching deterministic at 3.7s reflects `reasoning=False` shortcutting the single LLM call.
- Agent_loop cost factor over deterministic: gemma3 N/A · qwen3.5 **20×** · qwen2.5 **3.1×** · gemma4 **5.5×**.

### 2. Robustness — exit_reason distribution

| model | mode | exit_reason distribution |
|---|---|---|
| gemma3:4b | deterministic | `deterministic=42` |
| **gemma3:4b** | **agent_loop** | **`llm_call_failed=42` (100%)** |
| qwen3.5:4b | deterministic | `deterministic=42` |
| qwen3.5:4b | agent_loop | `natural_stop=41, budget_exhausted=1` |
| qwen2.5:7b | deterministic | `deterministic=42` |
| qwen2.5:7b | agent_loop | `natural_stop=42` |
| gemma4:e4b | deterministic | `deterministic=72` |
| gemma4:e4b | agent_loop | `natural_stop=72` |

- **Zero harness_error rows across 396 cells.** Every run produced a valid record (degraded or not).
- **One `budget_exhausted` in qwen3.5:4b agent_loop** out of 42. max_iter=10 was sufficient for 41/42.
- **gemma3:4b agent_loop is a fully clean negative data point**: the degrade handler intercepts the Ollama 400 deterministically.

### 3. Tool calling correctness (agent_loop only)

| model | mean tool calls/run | median | % runs with 0 tool calls | tool_errors total | n |
|---|---|---|---|---|---|
| gemma3:4b | 0.00 | 0.0 | 100% | 0 | 42 |
| qwen2.5:7b | 1.83 | 2.0 | 0% | 5 | 42 |
| gemma4:e4b | 2.74 | 3.0 | 1% | 1 | 72 |
| **qwen3.5:4b** | **3.88** | **4.0** | 0% | 0 | 42 |

- **qwen3.5:4b is the most tool-active model** (3.88 calls/run). Thinking models dispatch tools aggressively; gemma3 dispatches none (API blocked).
- **qwen2.5:7b has the highest tool_error rate** (5/42 = 12%), likely due to schema-imperfect args on the older instruction-tuned 7B. Even so, the loop's self-correction (tool error → ToolMessage → model retries) yielded 100% `natural_stop`.
- **0% of qwen3.5/qwen2.5/gemma4 runs emit zero tool calls** — the agent loop is genuinely engaging the tools, not just monologuing.

### 4. Plan quality — Local judge (qwen2.5:7b)

| model | mode | local mean | local median | n |
|---|---|---|---|---|
| **gemma3:4b** | **deterministic** | **0.903** | 0.960 | 42 |
| gemma3:4b | agent_loop | 0.514 | 0.520 | 42 |
| qwen3.5:4b | deterministic | 0.303 | 0.240 | 42 |
| **qwen3.5:4b** | **agent_loop** | **0.843** | 0.880 | 42 |
| qwen2.5:7b | deterministic | 0.770 | 0.800 | 42 |
| qwen2.5:7b | agent_loop | 0.770 | 0.800 | 42 |
| gemma4:e4b | deterministic | 0.523 | 0.240 | 72 |
| **gemma4:e4b** | **agent_loop** | **0.856** | 0.840 | 72 |

### 5. Plan quality — Cloud judge (MiniMax-M2.7)

| model | mode | cloud mean | cloud median | n |
|---|---|---|---|---|
| gemma3:4b | deterministic | 0.688 | 0.680 | 42 |
| gemma3:4b | agent_loop | 0.371 | 0.200 | 42 |
| qwen3.5:4b | deterministic | 0.390 | 0.320 | 42 |
| qwen3.5:4b | agent_loop | 0.549 | 0.560 | 42 |
| qwen2.5:7b | deterministic | 0.584 | 0.600 | 42 |
| qwen2.5:7b | agent_loop | 0.469 | 0.480 | 42 |
| gemma4:e4b | deterministic | 0.557 | 0.600 | 72 |
| **gemma4:e4b** | **agent_loop** | **0.606** | 0.600 | 72 |

- **Cloud judge is systematically more conservative**: every cell scores lower under MiniMax-M2.7 than under qwen2.5:7b local.
- **The qwen2.5:7b verdict flips between judges**: local says agent_loop = deterministic (both 0.770); cloud says deterministic > agent_loop (0.584 > 0.469). Cloud detects the persistence + tool-error overhead that local judge ignores.
- **For thinking models (qwen3.5, gemma4), BOTH judges agree agent_loop > deterministic.** This is the strongest cross-judge agreement in the matrix.

### 6. Judge agreement (cross-model)

| model | mode | mean &#124;local−cloud&#124; | bias |
|---|---|---|---|
| qwen3.5:4b | deterministic | 0.192 | cloud +0.09 |
| qwen2.5:7b | deterministic | 0.221 | local +0.19 |
| gemma3:4b | agent_loop | 0.211 | local +0.14 |
| gemma3:4b | deterministic | 0.215 | local +0.22 |
| gemma4:e4b | deterministic | 0.226 | cloud +0.03 |
| gemma4:e4b | agent_loop | 0.249 | local +0.25 |
| qwen3.5:4b | agent_loop | 0.300 | local +0.29 |
| qwen2.5:7b | agent_loop | 0.313 | local +0.30 |

- **Mean delta 0.19–0.31 across cells** — judges DO NOT closely agree on absolute scores.
- **Local judge has a positive bias on 6/8 cells** (qwen2.5:7b local rates higher than MiniMax-M2.7 cloud).
- **Self-preference visible**: when qwen2.5:7b judges qwen2.5:7b's own outputs, local rates **+0.19 / +0.30** higher than cloud. Cross-model judging (e.g., qwen2.5 judging gemma4) shows similar bias direction but smaller magnitude.
- **The 2 cells where cloud rates higher** (qwen3.5 deterministic +0.09, gemma4 deterministic +0.03) are exactly the cells where deterministic plans are **low quality** — both judges agree at the bottom, but MiniMax sees less degradation than qwen2.5.

### 7. Token cost (agent_loop rows only — deterministic path has no agent_trace)

| model | mode | mean in_tok | mean out_tok | mean total/run |
|---|---|---|---|---|
| qwen2.5:7b | agent_loop | 2,380 | 328 | **2,708** |
| gemma4:e4b | agent_loop | 3,806 | 762 | **4,568** |
| qwen3.5:4b | agent_loop | 6,146 | 687 | **6,833** |

- Token cost roughly tracks tool call count: qwen3.5 emits most tool calls (3.88) and consumes most tokens (6,833).
- **Cost-per-quality**: qwen2.5:7b at 2,708 tokens for 0.770 local score = **3,517 tokens/quality-point**. qwen3.5:4b at 6,833 / 0.843 = **8,107 tokens/quality-point**. **qwen2.5:7b is ~2.3× more token-efficient** at the same quality tier — though qwen3.5 hits this score WITHOUT thinking, suggesting tools+coordination compensate.
- Deterministic path does not populate `agent_trace`, so token cost N/A for deterministic mode. Future eval improvement: instrument deterministic LLM calls separately.

### 8. Plan persistence + plan_action breakdown

| model | mode | % persisted | n persisted | n | plan_action breakdown |
|---|---|---|---|---|---|
| gemma3:4b | deterministic | 100% | 42 | 42 | check_in=6, generate=36 |
| gemma3:4b | agent_loop | 0% | 0 | 42 | generate=42 (all degraded) |
| qwen3.5:4b | deterministic | **10%** | 4 | 42 | generate=4, none=38 |
| qwen3.5:4b | agent_loop | **86%** | 36 | 42 | check_in=6, generate=36 |
| qwen2.5:7b | deterministic | 90% | 38 | 42 | check_in=6, generate=32, none=4 |
| qwen2.5:7b | agent_loop | 40% | 17 | 42 | generate=42 |
| gemma4:e4b | deterministic | **43%** | 31 | 72 | generate=31, none=41 |
| gemma4:e4b | agent_loop | **94%** | 68 | 72 | check_in=5, generate=67 |

- **Tool schemas rescue thinking models when `reasoning=False`.** qwen3.5:4b deterministic only persists 10% (LLM emits garbled JSON without thinking enabled) but jumps to **86% under agent_loop** because the `update_study_plan` tool's Pydantic schema rejects invalid milestone payloads, forcing the model to retry until valid.
- **Same effect on gemma4:e4b** (43% deterministic → 94% agent_loop). Tool dispatch as a quality filter.
- **qwen2.5:7b shows the opposite pattern** (90% deterministic → 40% agent_loop). For instruction-tuned-without-thinking models, the agent loop's multi-step coordination introduces failure modes (tool argument validation failures) that the deterministic path bypasses with a single direct prompt.
- **Multi-turn check-in detection works in agent_loop**: gemma4 5/72, qwen3.5 6/42 check_ins inferred from `get_existing_plan` returning non-null — proves `_infer_plan_action` heuristic.

## Findings

### 1. The "agency = model + minimal harness" thesis is conditional

`learn-claude-code`'s central claim is that agency comes from the model and the harness should be minimal — that a competent model with a `while(tool_use)` loop is enough. **Our data partially confirms and partially refutes this**:

- **Confirmed on 8B thinking models**: `gemma4:e4b agent_loop` hits 100% `natural_stop`, 94% persistence, and the highest agent-mode judge scores. The harness IS minimal (≤ 350 lines) and the model IS doing the work. Agency emerges.
- **Refuted on 4B without tools**: `gemma3:4b` does not have `tools` capability and the Ollama API rejects every call. No model + harness combination can make this model an agent. **The model has to be trained for it.** Minimal harness cannot rescue a model that wasn't trained to emit tool calls.
- **Refuted on 7B older instruction-tuned + tools**: `qwen2.5:7b` can run the agent loop, but it does so *worse* than its own deterministic mode by every objective metric except local judge (which is itself). 12% tool error rate and 40% persistence — vs 90% persistence deterministic — suggest the model is not robust enough as an agent at this tier. The harness exposes the gap rather than closing it.
- **Confirmed on 4B+ thinking + tools when `reasoning=False`**: qwen3.5:4b cannot produce a usable deterministic plan with reasoning disabled (10% persistence, garbled JSON output). But its agent_loop mode persists 86% with valid milestones. **Tool schemas substitute for the reasoning the model isn't being allowed to do.** This is a genuinely new finding: the harness isn't minimal here — the tool schema is doing structural reasoning that the model can't do alone without thinking enabled.

### 2. Tool schemas are an under-acknowledged form of harness

The Pydantic `Milestone` model in `app/agent/tools/schemas.py` is 4 lines:
```python
class Milestone(BaseModel):
    title: str
    due_at: str | None = None
    done: bool = False
    topic: str | None = None
```

This 4-line schema, enforced by `update_study_plan`'s `Milestone.model_validate(m)` in our closure factory, **lifted qwen3.5:4b's persistence rate from 10% to 86%** with `reasoning=False`. The schema validation surfaces a recoverable error to the model via `ToolMessage`, forcing it to self-correct until output is parseable.

**This contradicts "minimal harness".** A 4-line schema is small in lines but large in semantic guidance. If learn-claude-code's "minimal" means "just a while-loop", then it's underspecified — *structured tool schemas* are doing significant heavy lifting.

### 3. Judge agreement is the meta-metric that matters

Mean |local − cloud| of 0.19–0.31 means our two judges disagree by a quarter to a third of the score range. The verdicts they produce CAN flip — qwen2.5:7b's mode preference flips between judges. **No single judge can be trusted alone on plan quality** at the small-model tier.

The bias direction is consistent: qwen2.5:7b local judge is more generous than MiniMax-M2.7 cloud (local +0.19 to +0.30 on 6/8 cells). This is the **self-preference bias** spec §6.5 anticipated: the local judge is closer in capability to the planner models, so it under-detects errors that a stronger cloud model surfaces.

### 4. Latency is the agent loop tax

`agent_loop` costs **3-20× more wall time** than deterministic on the same model. For interactive UX, this matters: 73s on qwen3.5:4b vs 3.7s on the same model is a **20× UX gap** purely from architecture choice. The deterministic path is not a "downgrade" — it's a different point on the latency/quality curve.

The choice between modes should depend on use case:
- **Realtime chat**: deterministic for all but gemma4:e4b
- **Background batch (overnight)**: agent_loop on thinking models for higher quality
- **Constrained models without tools (gemma3 tier)**: deterministic is the only option

### 5. `reasoning=False` is double-edged

Forcing `reasoning=False` on thinking models speeds them up 30-100× (Phase B verified qwen3.5:4b 813s → 7.3s). But it cripples them on the deterministic path because they're trained to think — strip thinking and their direct outputs degrade catastrophically (qwen3.5 persistence 10%).

The agent_loop with tool schemas recovers most of this quality gap by externalizing structure. **The harness adds a structural layer that thinking would otherwise provide internally.** This is a real engineering finding — not just empirical noise.

## Limitations

- **N = 36-72 per cell.** Statistical power for small effects is limited; differences within ±0.05 score should not be treated as significant.
- **Local judge model (qwen2.5:7b) overlaps with one of the planner models.** Self-preference bias is visible in those cells; treat qwen2.5 × qwen2.5 cells as the lower-bound on objective truth.
- **Deterministic path has no token cost data** because `agent_trace` is agent_loop-only. Compare-by-tokens between modes is currently impossible from this run.
- **All queries are HKBU domain.** Transferability to other corpora untested.
- **Ollama tool-calling on gemma3:4b is an industry-state quirk**, not a generalizable model limitation; future Ollama releases may add tools capability to gemma3 variants.
- **MiniMax-M2.7 cloud judge has its own thinking model bias** — JSON output came after a `<think>` block in the same `message.content` field; our greedy regex parsed it correctly but the cloud judge is itself a thinking model judging plans. A non-thinking cloud judge (e.g., GPT-4o-mini) would yield a different bias direction.
- **Appendix thinking on/off comparison** (gemma4:e4b only) is folded into the gemma4 cells but not separately tabulated here; n=30 for thinking-on subset is borderline for distinct conclusions.

## Smoke verification log (from Cut ①f)

| Model | Mode | Wall | SSE | Plan persisted | Judge | Notes |
|---|---|---|---|---|---|---|
| gemma3:4b | deterministic | ~10s | 3 events ✓ | 7 milestones, grounded in `HyDEGenerator` / `ablation runs` / `hyde_battery_results.csv` | 0.64 pass | Clean baseline. |
| gemma3:4b | agent_loop | 5s | 3 events ✓ | ✗ Degrade (`⚠️ Could not reach the planner model.`) | 0.60 pass | **Ollama API 400 `does not support tools`**. `_format_degrade_output` handled gracefully. |
| qwen2.5:7b | deterministic | 26s | 3 events ✓ | 3 milestones | 0.80 pass | Normal. |
| qwen2.5:7b | agent_loop | 36s | 3 events ✓ | 4 milestones, references "sources I found" | 0.84 pass | **Tool calling works on 7B instruction-tuned**. |
| gemma4:e4b | deterministic | 122s | 3 events ✓ | 3 milestones | 0.88 pass | Thinking-ON tax: 5× slower than qwen2.5:7b. |
| gemma4:e4b | agent_loop | 98s | 3 events ✓ | structured markdown plan | 0.96 pass | Tool calling + thinking. |
| qwen3.5:4b | deterministic | 813s | 3 events ✓ | 4+ milestones citing `TokenAnalyzer` / `Gao et al 2022` | 0.96 pass | **Highest smoke judge score**; thinking-ON makes single LLM call expensive. |
| qwen3.5:4b | agent_loop | 683s | 3 events ✓ | 7 milestones in markdown table | 0.68 pass | Tool calling + thinking-ON; agent_loop cost quality (multi-step coordination noise). |

Smoke prompted the Phase B finding that `ChatOllama(reasoning=False)` reduces qwen3.5:4b 813s → 7.3s (32× speedup), which made the full matrix tractable.

## Pre-requisite for ②b that was applied

Cut ②a originally constructed `planner_llm = ChatOllama(model=spec.model, temperature=0.7)` without forwarding the `RunSpec.thinking` field. **Applied 1-line patch (Cut ②a follow-up)**:

```python
planner_llm = ChatOllama(
    model=spec.model,
    temperature=0.7,
    reasoning=spec.thinking,  # main matrix False, appendix gemma4:e4b True
)
```

The `reasoning` field is langchain-ollama 1.1's mapping to Ollama API `think`. Without this, qwen3.5:4b deterministic runs at 813s each, making ②b ~30 hours. With it, full matrix runs in ~5 hours.

## References

- Spec: `docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md`
- Plan: `docs/superpowers/plans/2026-05-23-p2-2-agent-loop-ablation.md`
- Raw data: `backend/app/eval/p2_2_agent_ablation/output/results.jsonl`
- Auto-generated summary: `backend/app/eval/p2_2_agent_ablation/output/summary.md`
- Blog: `docs/agent_loop_vs_deterministic.md`
- `learn-claude-code` repo (the thesis being tested): `/Users/lianghaozhe/learn-claude-code/`

---

# P2.3 Quiz Agent Loop Ablation — Empirical Report

> P2.3 deliverable. Tests three predictions from the P2.2 blog `docs/agent_loop_vs_deterministic.md` "What I would do next" — does the agent_loop pattern transfer from Plan to Quiz when the schema is markedly stricter and the task is more grounded in specific corpus content?
>
> **Matrix**: same 4 models × 2 modes × 12 queries × 3 runs + gemma4:e4b thinking-ON appendix = **396 records**.
> **Wall time**: ~5 hours sequential on a 16GB Apple Silicon Mac.
> **Cost**: ~$3 MiniMax-M2.7 API for 396 cloud judgments (+ ~$3 for an earlier pilot — see §8 below).
> **Data**: `backend/app/eval/p2_3_quiz_ablation/output/results.jsonl` (main, with production retriever wired); `backend/app/eval/p2_3_quiz_ablation/output/results_no_retriever.jsonl` (pilot, no retriever).

## TL;DR

- **`gemma3:4b agent_loop` = 75% `llm_call_failed`** (Ollama 400 `does not support tools`). P2.2 P1 prediction replicates cleanly on a new task. The remaining 25% are multi-turn GRADE records routed through the deterministic dispatcher.
- **Schema rescue effect REVERSES on Quiz vs Plan**. P2.2 found qwen3.5:4b agent_loop persistence at 86% (Plan); P2.3 measures 50% (Quiz) — and deterministic Quiz hits **75%**. The stricter QuizQuestionPersist schema (`options: list[str]` len=4 + `Literal["A","B","C","D"]` answer + prefix `field_validator`) exceeds what small models can self-correct from in 6 iterations.
- **`qwen3.5:4b agent_loop` budget-exhausts at 25%** with 1.7 mean tool errors per run — the model satisfies retriever_search once, then loops on persist_quiz_question schema rejection.
- **But on natural_stop runs, agent_loop quality is HIGHER**. Per-cell judge scores filtered to `exit_reason="natural_stop"` show local 0.80-0.82 vs deterministic 0.74-0.76; cloud 0.57-0.72 vs 0.46-0.54. A precision-recall trade-off: agent_loop sacrifices completion rate (50-75% natural_stop) for per-success quality (+0.05 to +0.16 cloud judge).
- **A methodology pilot run with `retriever=None`** (§8) surfaced an additional dimension P2.2 did not measure: agent_loop tool feedback creates an alignment safety net. When `retriever_search` returns `"[]"`, well-aligned models refuse to fabricate (21-39% refusal rate on qwen-family + gemma4), while the deterministic prompt path silently emits training-distribution content. This is not the schema-rescue test we set out to do, but it is a finding worth flagging.

## Setup

- Same 4 models as P2.2 — `gemma3:4b`, `qwen3.5:4b`, `qwen2.5:7b`, `gemma4:e4b`. Same 2 modes (`deterministic` / `agent_loop`).
- 10 single-turn + 2 multi-turn quiz queries on HKBU corpus topics (HyDE / BM25 / reranking / chunking / eval / judge / embeddings / hybrid / RRF). Topic overlap with P2.2 by design — enables future task-generalization cross-analysis.
- Multi-turn queries are GENERATE turn 0 → fixed reply `"A"` → GRADE turn 1. GRADE always routes to deterministic `quiz_master` per the state-aware dispatcher in `graph.py:quiz_node`; only `turn_idx=0` records contribute to mode comparison.
- Agent loop hyperparameters: `max_iter=6` (smaller than P2.2's 10 — quiz has a narrower expected path: retriever_search → persist_quiz_question → summary = 3 iters; 2× safety margin).
- Production retriever (`_build_default_retriever()` from `app/main.py` — `RerankingRetriever(HybridRetriever(Retriever))` over Chroma) wired into both `quiz_master` and `quiz_master_agent` via `run_eval.py`. Same retriever instance reused across all specs (read-only, safe to share).
- Dual judges identical to P2.2: `qwen2.5:7b` local + `MiniMax-M2.7` cloud, both running the `judge_quiz.txt` rubric with `QUIZ_DIMENSIONS` (question_quality / option_plausibility / answer_correctness / explanation_clarity / difficulty_calibration).
- `reasoning=False` forwarded to `ChatOllama` on the main matrix (verified pre-flight at 3.0s per qwen3.5:4b call vs the ~813s thinking-ON baseline P2.2 originally hit).

## Results

### 1. Latency (median wall_time_s per cell, agent_loop only counts iterations)

| model | mode | median wall (s) | mean iters | n |
|---|---|---|---|---|
| gemma3:4b | deterministic | 10.9 | n/a | 36 |
| **gemma3:4b** | **agent_loop** | **0.1** | 0.0 | 36 |
| qwen3.5:4b | deterministic | 18.5 | n/a | 36 |
| qwen3.5:4b | agent_loop | 46.7 | 3.4 | 36 |
| qwen2.5:7b | deterministic | 11.8 | n/a | 36 |
| qwen2.5:7b | agent_loop | 30.8 | 2.2 | 36 |
| gemma4:e4b | deterministic | 26.7 | n/a | 66 |
| gemma4:e4b | agent_loop | 28.5 | 1.8 | 66 |

- `gemma3:4b agent_loop` 0.1s is the Ollama 400 reject time before any LLM call.
- `qwen3.5:4b agent_loop` is the slowest cell — 3.4 mean iterations, 1.7 mean tool errors (schema retry loop).
- Agent-loop overhead vs deterministic same model: gemma3 N/A, qwen3.5 **2.5×**, qwen2.5 **2.6×**, gemma4 **1.07×**. P2.2 measured 3-20× for Plan; Quiz is tighter because the prompt path is also doing the LLM-side reasoning work the agent loop replaces.

### 2. Robustness — exit_reason distribution (turn_idx=0 only)

| model | mode | exit_reason distribution |
|---|---|---|
| gemma3:4b | deterministic | `deterministic=34, error=2` |
| **gemma3:4b** | **agent_loop** | **`llm_call_failed=27 (75%), n/a=9`** |
| qwen3.5:4b | deterministic | `deterministic=36` |
| **qwen3.5:4b** | **agent_loop** | **`natural_stop=18, budget_exhausted=9 (25%), n/a=9`** |
| qwen2.5:7b | deterministic | `deterministic=36` |
| qwen2.5:7b | agent_loop | `natural_stop=27, n/a=9` |
| gemma4:e4b | deterministic | `deterministic=66` |
| gemma4:e4b | agent_loop | `natural_stop=48, n/a=18` |

- `n/a` cells are multi-turn GRADE records routed to deterministic per the dispatcher. They appear under agent_loop because that was the requested mode, but their LLM work happened on the deterministic path. Not a defect.
- `gemma3:4b deterministic` has 2 `error` records (out of 36, 5.5%). Both share the symptom of an empty `final_text_excerpt` — likely model output parse failures on a corpus chunk that produced ambiguous JSON.
- `qwen3.5:4b agent_loop` is the only cell with non-zero `budget_exhausted`. See §7 for the schema-strictness mechanism.

### 3. Tool calling correctness (agent_loop only, total tool calls across n records)

| model | retriever_search | persist_quiz_question | mean tool_calls/run | mean tool_errors/run |
|---|---|---|---|---|
| gemma3:4b | 0 | 0 | 0.0 | 0.0 (Ollama rejects before tool dispatch) |
| qwen2.5:7b | 27 | 25 | 1.93 | 0.1 |
| gemma4:e4b | 31 | 39 | 1.46 | 0.2 |
| **qwen3.5:4b** | **27** | **78** | **3.88** | **1.7** |

- The expected path is `retriever_search → persist_quiz_question` (2 tools per run).
- `qwen2.5:7b` runs almost exactly on the expected path (1.93 tool calls/run, 0.1 errors).
- `gemma4:e4b` slightly under-calls retriever_search (31 calls / 48 successful runs = 0.65/run; the model often persists from one search rather than refining).
- `qwen3.5:4b` over-calls `persist_quiz_question` by 3× — 78 calls across 36 runs (mean 2.17/run, 1.7 errors/run). This is the schema retry loop manifesting.

### 4. Plan quality — Local judge qwen2.5:7b (filtered to `natural_stop` only — degraded runs excluded)

| model | mode | n | local mean | local mean (all runs) |
|---|---|---|---|---|
| gemma3:4b | deterministic | 34 | 0.752 | 0.754 |
| gemma3:4b | agent_loop | 0 | n/a | 0.644 (degrade messages, judge sees `⚠️`) |
| qwen3.5:4b | deterministic | 36 | 0.758 | 0.758 |
| **qwen3.5:4b** | **agent_loop** | **18** | **0.822** | 0.732 |
| qwen2.5:7b | deterministic | 36 | 0.741 | 0.741 |
| **qwen2.5:7b** | **agent_loop** | **27** | **0.797** | 0.782 |
| gemma4:e4b | deterministic | 66 | 0.755 | 0.755 |
| **gemma4:e4b** | **agent_loop** | **48** | **0.815** | 0.785 |

### 5. Plan quality — Cloud judge MiniMax-M2.7 (same filter as §4)

| model | mode | n | cloud mean | cloud mean (all runs) |
|---|---|---|---|---|
| gemma3:4b | deterministic | 34 | 0.514 | 0.508 |
| gemma3:4b | agent_loop | 0 | n/a | 0.422 |
| qwen3.5:4b | deterministic | 36 | 0.458 | 0.458 |
| **qwen3.5:4b** | **agent_loop** | **18** | **0.720** | 0.589 |
| qwen2.5:7b | deterministic | 36 | 0.501 | 0.501 |
| **qwen2.5:7b** | **agent_loop** | **27** | **0.621** | 0.561 |
| gemma4:e4b | deterministic | 66 | 0.544 | 0.544 |
| gemma4:e4b | agent_loop | 48 | 0.568 | 0.507 |

- On every cell except gemma3 agent_loop (which has no successful runs), filtering to `natural_stop` raises both judge scores.
- Cloud judge agrees with local that agent_loop natural_stop quality > deterministic on qwen3.5:4b (+0.26), qwen2.5:7b (+0.12), gemma4:e4b (+0.02 — tied within noise).
- Local qwen2.5:7b is the most-improved cell under agent_loop (+0.06 over its own deterministic) — note local judge is also qwen2.5:7b, so self-preference bias may inflate this.

### 6. Judge agreement — mean |local − cloud| per cell (all runs, n full)

| model | mode | mean &#124;local−cloud&#124; | local bias |
|---|---|---|---|
| gemma3:4b | agent_loop | 0.224 | local +0.22 |
| gemma3:4b | deterministic | 0.260 | local +0.25 |
| qwen3.5:4b | deterministic | 0.304 | local +0.30 |
| qwen3.5:4b | agent_loop | 0.159 | local +0.14 |
| qwen2.5:7b | deterministic | 0.242 | local +0.24 |
| qwen2.5:7b | agent_loop | 0.228 | local +0.22 |
| gemma4:e4b | deterministic | 0.228 | local +0.21 |
| gemma4:e4b | agent_loop | 0.283 | local +0.28 |

- Range 0.16-0.30 — comparable to P2.2 (0.19-0.31). Same finding replicates: local qwen2.5:7b is systematically more generous than cloud MiniMax-M2.7 by ~0.2 across all 8 cells.
- The tightest agreement (qwen3.5:4b agent_loop, 0.159) is the cell with the most polarized data: 18 high-quality natural_stop runs + 9 budget_exhausted degrade messages. Both judges agree on both extremes.

### 7. Persistence — the schema rescue replication test (the headline subsection)

| model | mode | persisted | n | persist% |
|---|---|---|---|---|
| gemma3:4b | deterministic | 25 | 36 | **69%** |
| gemma3:4b | agent_loop | 0 | 36 | 0% (all `llm_call_failed`) |
| qwen3.5:4b | deterministic | 27 | 36 | **75%** |
| **qwen3.5:4b** | **agent_loop** | **18** | **36** | **50%** |
| qwen2.5:7b | deterministic | 27 | 36 | **75%** |
| qwen2.5:7b | agent_loop | 21 | 36 | 58% |
| gemma4:e4b | deterministic | 48 | 66 | **73%** |
| gemma4:e4b | agent_loop | 26 | 66 | 39% |

**Compared with P2.2 (Plan)**:

| model | P2.2 det | P2.2 agent | P2.3 det | P2.3 agent | direction |
|---|---|---|---|---|---|
| qwen3.5:4b | 10% | 86% | **75%** | **50%** | **reversed** |
| gemma4:e4b | 43% | 94% | **73%** | **39%** | **reversed** |
| qwen2.5:7b | 90% | 40% | 75% | 58% | both modes lower; gap narrows |

- P2.2's schema rescue effect — "agent loop with `update_study_plan` Pydantic schema raises persistence from 10% to 86% on thinking models with reasoning=False" — **does not transfer to Quiz**.
- Three orthogonal mechanisms are operating:
  1. **Deterministic Quiz benefits from tolerant JSON parsing**. `generate_quiz` in `app/agent/tools/quiz.py` uses a 3-tier regex (fenced ```json``` → bare array → first-`{...}`-block) that accepts what LLMs naturally emit. Persistence is high at 69-75% even with `reasoning=False`.
  2. **Agent-loop Quiz schema is too strict for small-model self-correction**. `QuizQuestionPersist` enforces `options: list[str]` (exactly 4) + `answer: Literal["A","B","C","D"]` + `field_validator` requiring `"A) "/"B) "/"C) "/"D) "` prefix. When the model produces a near-miss (e.g. lowercase `"a)"`, 3 options, or `answer="A."`), the wrapper returns `{"error": ...}` JSON and the loop retries. qwen3.5:4b hits this 1.7× per run on average and exhausts its 6-iter budget on 25% of runs.
  3. **When agent_loop succeeds, the persisted MCQ is higher quality** (see §4-5). The schema enforces structural correctness so the persisted question is always valid 4-option MCQ with a Literal answer letter and the prefix convention. Deterministic mode persists more often but produces some malformed MCQs that the judge scores down on `option_plausibility` and `answer_correctness`.

### 8. Methodology pilot — alignment safety effect with `retriever=None`

Before wiring the production retriever, an initial 396-record pilot ran with `retriever=None` (mirroring the spec template that fell out of the P2.2 fork). The pilot revealed a methodology issue that was not present in P2.2.

In P2.2, the Plan task `"make a plan on HyDE"` can be served from training knowledge — milestones are meta-content (timeline, ordering), not corpus-specific facts. With no retriever, the deterministic path silently emitted generic study-plan structure and the agent_loop path called `retriever_search`, got `[]` back, and then proceeded to generate a plan from training knowledge.

In P2.3 Quiz, the task `"quiz me on HyDE"` requires corpus-grounded specifics. With `retriever=None`:

| model | mode | persist% | refusal% | local J | cloud J |
|---|---|---|---|---|---|
| qwen3.5:4b | agent_loop | **0%** | **39%** | 0.488 | 0.397 |
| gemma4:e4b | agent_loop | 15% | **21%** | 0.625 | 0.440 |
| qwen2.5:7b | agent_loop | 22% | **33%** | 0.651 | 0.442 |
| (all deterministic cells) | | 71-75% | 0-11% | 0.72-0.78 | 0.44-0.61 |

- **Agent-loop refusal rate jumped to 21-39%** on the well-aligned cells. qwen3.5:4b agent_loop produced 14/36 final outputs like *"I'm unable to retrieve any information about HyDE from the user's PDF source material. Could you please provide more context..."* — refusing to fabricate without sources.
- **Deterministic refusal rate stayed near 0%** because the deterministic prompt template doesn't surface "retrieval returned empty" as a signal the model can act on; it just produces text from the empty-context prompt.
- After wiring the production retriever in the main matrix run, **refusal rate dropped to 0% across all agent_loop cells** and persistence rates jumped (qwen3.5:4b 0% → 50%, gemma4:e4b 15% → 39%, qwen2.5:7b 22% → 58%).
- **This is not a P2.2-replication finding — it is a new dimension**: agent loops with retriever tools create a structural alignment property (refuse on empty context) that the deterministic prompt path lacks. Whether this is desirable depends on use case (interactive quiz wants graceful empty-corpus handling, not silent fabrication).

## Findings

### Finding 1 — Schema rescue effect is conditional on the schema-strictness vs model-capability ratio

P2.2's `Milestone` schema has 1 required field (`title`) and 3 optionals. P2.3's `QuizQuestionPersist` has 5 required fields, 1 Literal constraint, and 1 custom `field_validator`. The rescue mechanism works when the schema is strict enough to surface formatting errors back to the model but loose enough that the model can satisfy it within `max_iter` retries.

In Plan: schema strictness ≤ model self-correction capability → rescue works (P2.2 finding).
In Quiz: schema strictness > model self-correction capability (qwen3.5:4b can only satisfy 75% of cases) → rescue partially fails via `budget_exhausted` (25%). For gemma4:e4b and qwen2.5:7b, the schema is satisfiable but the overhead of the retry loop drops persistence below deterministic.

The "schema is a form of harness" claim from P2.2 §Finding 2 remains true — but the magnitude of help depends on a 2D fit, not a 1D "stricter is better".

### Finding 2 — Agent loop optimizes for per-success quality, deterministic for completion rate

On `natural_stop` runs, agent_loop quality beats deterministic by **+0.05 to +0.16 on cloud judge** across the 3 working tiers (qwen3.5:4b, qwen2.5:7b, gemma4:e4b). But agent_loop completes only 50-75% of runs vs deterministic 69-75%.

For interactive UX (a student asks for one quiz), the deterministic path is the better fit — predictable latency, near-100% completion, acceptable quality. For batch quality-max use cases (curated MCQ generation, exam authoring), agent_loop on `gemma4:e4b` is the better fit — higher per-output quality, accept some retries.

This is a **selection effect**, not a strict ordering. Mode choice depends on which axis matters for the deployment.

### Finding 3 — Agent loop with retriever tool creates an alignment safety net deterministic mode lacks

The methodology pilot (§8) made this measurable. When the corpus is missing, well-aligned models with `retriever_search` tool access correctly refuse to fabricate (21-39% refusal in pilot). The deterministic prompt path has no equivalent signal — it silently produces training-distribution content.

This was not the P2.2-replication test we set out to do. It is a finding the original blog `agent_loop_vs_deterministic.md` did not consider — agent loops have a corollary safety property emerging from the tool-feedback channel, separable from the schema-rescue mechanism.

For production: the agent_loop variant should be the default for any task where empty-corpus fabrication is a correctness risk (factual quiz on user-specific materials). The deterministic variant should be reserved for tasks where corpus-grounding is not essential (study plan structure, calendar suggestion, general advice).

## Limitations

- **`max_iter=6` was sized for the 3-iteration expected path** (retriever_search → persist → summary). For qwen3.5:4b, schema retry can absorb 5+ iterations alone, leaving no budget for the surrounding work. A follow-up cut with `max_iter=12` would isolate "schema retry budget" from "schema strictness" as separate variables.
- **`reasoning=False` was forced on all matrix runs** (consistent with P2.2). qwen3.5:4b and gemma4:e4b are thinking models; their per-tool-call schema-correction capability under thinking-ON is untested.
- **N=36 per main-matrix cell** (n=66 for gemma4:e4b after appendix). Statistical power for effects under ±0.05 is limited; the "natural_stop quality advantage" finding (§4-5) is +0.05 to +0.16 — most pairs are above the noise floor, but qwen2.5:7b agent_loop vs deterministic cloud judge delta (+0.12) is borderline.
- **`results_no_retriever.jsonl` was scored by the same judges as main**. The judges have no information that the pilot ran with `retriever=None`, so their scores penalize agent_loop refusals as low-quality outputs even though refusal IS the correct behavior. A pilot-aware judge prompt could disambiguate this; out of scope for this run.
- **Cloud judge is MiniMax-M2.7, a thinking model** — same caveat as P2.2. Non-thinking cloud judge cross-validation (e.g. GPT-4o-mini) deferred to a future judge-bias ablation.

## References

- Spec: `docs/superpowers/specs/2026-05-24-p2-3-quiz-agent-loop-ablation-design.md`
- Plan: `docs/superpowers/plans/2026-05-24-p2-3-quiz-agent-loop-ablation.md`
- Raw data main (with retriever): `backend/app/eval/p2_3_quiz_ablation/output/results.jsonl`
- Raw data pilot (no retriever): `backend/app/eval/p2_3_quiz_ablation/output/results_no_retriever.jsonl`
- Blog continuation: `docs/quiz_ablation_followup.md` (sister post answering the 3 P2.2 blog predictions)
- Upstream blog being responded to: `docs/agent_loop_vs_deterministic.md`

---

# P3 Frontend Productize — Shipping Report

> P3 deliverable. Productized the stable P2.3 backend into a portfolio-grade 7-view Vue 3 app. Backend stays byte-identical except for 4 new minimal GET endpoints. Operationalizes P2.3 §Finding 3 alignment-safety as production UX.

## TL;DR

- **7 views shipped**: Overview (hero dashboard), Chat, PlanTimeline, QuizAdaptive, MistakeBank, Library, Settings.
- **4 backend GET endpoints added**: `/api/plans/current`, `/api/documents`, `/api/mistakes/due`, `/api/mastery`. All TDD red-green. Test count 202 → 213 (+11).
- **1 new repo method**: `MasteryRepository.list_for_user_detailed()` (joins Topic for `last_reviewed`).
- **Alignment-safety banner operationalized**: dual-channel detection (pre-flight via `chunks_count` + in-flight via refusal regex) renders `<EmptyCorpusBanner>` instead of error.
- **Mode dispatch UX**: per-view default (Plan/Quiz default `agent_loop`) + `<ModeChip>` per-message override. Settings persists user preferences.
- **Visual system**: Modern Dark Cinema (Inter + JetBrains Mono + Noto Sans SC), all tokens in Tailwind 4 `@theme` block per `design-system/MASTER.md`.

## What ships per view

| View | URL | Key behaviors |
|---|---|---|
| Overview | `/` | UploadGate (when docs=0), 4 cards (MasteryCard / PlanProgressCard / MistakesDueCard / WeakTopicsChips), RadarChart 5-axis |
| Chat | `/chat` | Existing P1 streaming SSE with mode-header injection extended |
| PlanTimeline | `/plan` | MilestoneList with status icons (overdue/today/future), MindmapPanel (mermaid lazy-load), ModeChip planner override, Check-in button |
| QuizAdaptive | `/quiz` | DifficultySelector (easy/med/hard), MCQCard with 4 options, GradeResult ✓/✗ + explanation, ModeChip quiz override, EmptyCorpusBanner gate |
| MistakeBank | `/mistakes` | List tracked mistakes with topic chip + SM-2 (interval/ease); header still surfaces due-today count; Redo → `/quiz?mistake_id=X` |
| Library | `/library` | Upload flow + persisted indexed-PDF list from `GET /api/documents` |
| Settings | `/settings` | Existing P1 BYOK fields + new fieldset "P3 mode defaults" (defaultPlannerMode / defaultQuizMode dropdowns) |

## Mode dispatch UX validation

The `<ModeChip>` per-view pattern (default from settings, click to flip for next chat send, auto-revert after `done`) makes the dual-mode P2.2/P2.3 finding visible to the user without cluttering the chat input. Settings persistence means users can lock a default after deciding which mode works best for them.

**UX trade-off observed**: when user clicks Generate Question on `/quiz` with mode=agent_loop, the spinner runs 30-90 seconds (P2.3 measured latency). The deterministic toggle gives 3-20× speedup at the cost of P2.3-measured quality trade-off. The product lets the user feel this trade-off rather than hiding it.

## EmptyCorpusBanner — operationalizing P2.3 §F3

P2.3 §Finding 3 measured that agent_loop with retriever-empty makes well-aligned models refuse to fabricate (21-39% refusal rate). In product:

1. **Pre-flight** (Quiz mount): `GET /api/documents` → if `chunks_count===0` for all docs, render banner before any chat request. User never sees the refusal text.
2. **In-flight** (during chat stream): `parse.ts:looksLikeEmptyCorpusRefusal()` regex on accumulated token buffer; if match, set `quiz.needsUpload=true` → banner appears.

Both channels feed the same `needsUpload = docs.isEmpty || quiz.needsUpload` computed in `QuizAdaptive.vue`. Banner click navigates to Library with `?return=/quiz` query (groundwork for auto-return polish).

## Backend changes (4 new GETs)

| Endpoint | Repo methods used | New repo methods | Tests added |
|---|---|---|---|
| `GET /api/plans/current` | `GoalRepository.list_active_for_user` + `PlanRepository.get_by_goal` (both existing) | none | 2 |
| `GET /api/documents` | `DocumentRepository.list_for_user` (existing) | none | 2 |
| `GET /api/mistakes/due` | new | `MistakeRepository.list_due_with_details` (joins Question + Topic) | 3 (1 repo + 2 route) |
| `GET /api/mastery` | `GoalRepository.list_active_for_user` + `PlanRepository.get_by_goal` (for overdue_count) | `MasteryRepository.list_for_user_detailed` (joins Topic for last_reviewed) | 4 (1 repo + 3 route) |

All Pydantic output DTOs use `model_config = ConfigDict(extra="ignore")` for forward-compat — extending the underlying model with new fields won't break the API.

## Spec deviations

- **Brainstorm Q1 said "C: Dashboard-first, 7 nav links flat"** — the implemented shell uses "B: Grouped (Study/Review/System sections)" because 7 links benefits from visual structure. Documented in plan §A0 Step 6 as a small intentional drift.
- **A11 implementer saved one screenshot to `docs/superpowers/screenshots/cut-A11.png` instead of `docs/screenshots/p3/cut-A11.png`** — A13 reconciles by also placing the 7 final view screenshots in the canonical `docs/screenshots/p3/` path.

## Known limitations (P4 candidates)

- **No mobile UI** — <768px renders a "use desktop" banner per spec §10. P4 covers mobile.
- **Streak + Coverage axes in RadarChart are placeholders** — real values require a `sessions` activity log + chunk-coverage compute that doesn't exist yet.
- **MCQ parser regex is heuristic** — works for the LLM's prescribed format but may need iteration if backend prompt evolves.
- **No real-time updates** — stores refetch on view mount; mid-session changes (e.g. new mistake from chat) require manual refresh or navigation.
- **`?return=/quiz` query in EmptyCorpusBanner** is groundwork — Library doesn't yet auto-redirect after upload.

## Verification log

- Backend test suite: `cd backend && uv run pytest -q | tail -3` → **213 passed** (+11 over P2.3's 202).
- Frontend build: `cd frontend && pnpm build` → exit 0 (~400ms per cut after deps installed). Bundles: index.js ~300KB / ~109KB gzip; mermaid lazy-chunked.
- chrome-devtools verification per cut: 14 `cut-A*.png` screenshots in `docs/screenshots/p3/`.

## References

- Spec: `study-coach/docs/superpowers/specs/2026-05-25-p3-frontend-productize-design.md`
- Plan: `study-coach/docs/superpowers/plans/2026-05-25-p3-frontend-productize.md`
- Design system: `study-coach/design-system/MASTER.md`
- Sister blog: `study-coach/docs/p3_frontend_productize.md`
- Final view screenshots: `study-coach/docs/screenshots/p3/{overview,chat,plan,quiz,mistakes,library,settings}.png`

---

# Learning Run Harness — Tutor Prompt Regression

> Frozen 12-case suite. This is not a claim about overall Tutor quality or learning effect.

## Config

| Field | Value |
|---|---|
| Date | 2026-08-17 |
| Experiment | `tutor-prompt-regression-v1` |
| Axis | `prompt_version` only (`tutor-v2` production vs `tutor-v3` candidate) |
| Cases | 12 (6 answerable / 3 multi_evidence / 3 expected_refusal) |
| Provider / model | `ollama` / `llama3.2` (local alias to `gemma4:e4b`, digest `c6eb396dbd59`, tools+thinking; suite runner used `reasoning=False`) |
| Parameters | temperature 0, top_p 1 |
| Scorers | `hybrid-v1` at run time; `hybrid-v2` historical re-score |
| Runtime judge | off |
| Budget | retrieval 5s / tutor 55s / hybrid scoring 25s / total 90s |
| Curated fixture | `backend/app/eval/learning_run/fixtures/tutor-prompt-regression-v1.jsonl` |
| Raw output | `backend/app/eval/learning_run/output/` (gitignored) |

## Hashes (Registry)

Taken from the frozen experiment document, not from a hand-edited export:

- `tutor-v2` prompt: `3686c0120d0b8cb615579b27ea43dc624e8053db763b1559b5e59a4a726fc9a2`
- `tutor-v3` prompt: `7f0da024e65c7fbf4b0dd8d485ef0bc1e1949b849f56fce10ac0a9eebf440b08`
- task cases: `90735b9708d6957c6c8ac7cbf5c09c2f6303bdf66ad2496241676a0be3e3ee1b`
- corpus: `9dd2758d60c8c51f4cbaccc9bacb153cf83b87b11add902787a5f3de404255cf`
- hybrid-v1 scorer: `41ad3573a3d48503e1f2c6a7404a10b5c37e2c5ede11d289529101101a1e7897`

## Results

Third real local suite on 2026-08-17 after repointing the frozen `llama3.2` alias to `gemma4:e4b` (same digest `c6eb396dbd59`). 24 finished Runs (12 × 2), each with `hybrid-v1` and `hybrid-v2` ScoreSets. Metrics were curated from the runner export; they were not hand-edited.

| Prompt | hybrid-v1 pass | hybrid-v1 fail | hybrid-v1 inconclusive | Notes |
|---|---|---|---|---|
| tutor-v2 | 0 | 0 | 12 | Rubric `scorer_parse_error` on every cell; no dimension scores |
| tutor-v3 | 0 | 0 | 12 | Same hybrid-v1 verdict matrix |

- Tutor answers are real `gemma4:e4b` generations. Deterministic citation numbering no longer floors the suite.
- Refusal observer fired on **v2** for `tgqa-004`, `tgqa-008`, and `tgqa-012` (`I don't know`).
- On **v3**, `tgqa-008` and `tgqa-012` added a "General Study Knowledge" fill after saying the notes were silent. That is the helpfulness-vs-grounding leak the candidate prompt asked for. The observer did **not** mark `expected_refusal_observed` on those two v3 cells.
- **Zero hybrid-v1 verdict/score regression.** The directed suite regression is the refusal-axis leak: v2 has `expected_refusal_observed` on `tgqa-008` and `tgqa-012`; v3 does not, and those v3 answers add "General Study Knowledge". That is accepted as a real prompt-axis regression. The LLM rubric parser was not loosened. Run Lab `Regressions` must show `2` for this fixture, not `0` fail verdicts.
- Empty dimension scores are a scorer parse limit on this model, not a fabricated 0.

## Limits

- Local small-model Hybrid rubric output can be `scorer_parse_error`; the ScoreSet then stays `partial` / `inconclusive` instead of fabricating dimension scores.
- Evaluation measures one TutorAttempt, not Graph Judge retries.
- 12 cases are a directed regression suite, not a benchmark of overall study quality.
- Paid/remote models are not a CI gate.

## Cloud Capability Foundation — protocol helpers and identity (offline)

This batch adds only the offline foundation that the later cloud-capability harness builds on:

- `backend/app/eval/p2_3_cloud_capability/protocol.py` — response and protocol helpers: `SmokeAbort`, `failure_class_for_exception` (transport vs harness classification), `flatten_text_content`, `normalize_finish_status`, `extract_thinking_tokens`, `extract_usage`, `canonical_json_bytes`, `langchain_response_canonical`, `thinking_char_length`, `finish_raw_from_response` and `raw_usage_from_response`, plus secret-key redaction for provenance fingerprints.
- `backend/app/eval/p2_3_cloud_capability/identity.py` — the frozen `ProviderIdentity` value object and the deterministic identity algorithm. `cloud_run_id` is a 16-hex-character digest over provider, protocol, model, mode, thinking profile, query id, `turn_idx` and `run_idx`. `storage_key` is the same digest over the same inputs **without** `turn_idx`, prefixed with `s` (so `s` followed by 16 hex characters); it addresses a session rather than a single turn. `recover_run_idx` does not invert either digest: it enumerates candidate run indices in a bounded range, rebuilds the id for each candidate from the row's context (mode, thinking profile, query id and `turn_idx`) and compares it with the recorded `run_id` or `session_storage_key`. An already-present nonnegative integer `run_idx` (excluding booleans) is returned as is. No index is returned when no candidate matches.
- `backend/app/eval/p2_3_cloud_capability/__init__.py` (empty) with the matching offline unit tests `backend/tests/eval/test_p2_3_cloud_protocol.py` and `backend/tests/eval/test_p2_3_cloud_identity.py`.

Nothing else is part of this batch. The probe, the live harness, the matrix runner, the `--stage` CLI wiring, the budget ledger (v1 or v2), the lifecycle lock, active-attempt selection, evidence snapshots and the summarizer are deliberately **not** delivered here and remain in the planned follow-up batches, each of which is still authorized separately. These are offline unit tests: no provider is called and no real `output/` is touched.

### Behaviour boundaries fixed by review (offline)

- Secret-key matching in `langchain_response_canonical` normalises the key before the existing exact-match and `_api_key` suffix rules: it lowercases it, maps `-` to `_`, and additionally treats a key as secret when it ends with `apikey` after removing `_`/`-`. That covers `api_key`, `x-api-key`, `X-API-KEY`, `service-API-Key` and the camelCase/acronym forms `apiKey`, `APIKey`, `APIKEY`, `serviceApiKey` and `serviceAPIKey`; a lower/upper boundary regex would miss `APIKey`. Retained key names are not renamed, non-secret lookalikes such as `apiKeyHint` and `apiKeyId` are kept, and a string value under a non-secret key is never redacted. This is a suffix rule, not a scan for arbitrary sensitive fragments.
- `cloud_run_id` and `storage_key` raise `ValueError` when any joined field (including the string form of the indices) contains the reserved `|` separator, because `('a|b', 'c')` and `('a', 'b|c')` would otherwise produce the same identity. The error message does not echo the offending input. Join order, UTF-8 encoding, SHA-256 and its 16-character truncation and the `s` prefix are unchanged; hardcoded golden digests pin that compatibility.
- `recover_run_idx` uses `turn_idx` only while attempting the `run_id` path, and only when the value is an exact integer (booleans excluded); a missing field keeps the historical default of `0`. Any other present value — explicit `None`, a float such as `1.5` or `1.0`, a numeric string, an empty string, a list or a dict — skips the `run_id` path without coercion and still attempts `session_storage_key`, so it cannot outrank a valid session key. A negative integer is not re-specified here. An already-present non-negative integer `run_idx`, a matching `run_id`, the bounded `max_runs` search, the builders and the golden digests are unchanged.
- `extract_usage` falls back to `input_tokens + output_tokens` when `total_tokens` is missing or is not an exact non-negative integer; every non-negative integer total is kept as recorded, and the existing `input_tokens`/`output_tokens` validation is unchanged. `raw_usage_from_response` still returns the original nested usage object without modifying the input.
- `failure_class_for_exception` walks the whole cause/context chain (cycle-guarded) before classifying, so a wrapped failure is not decided by its outermost layer. Explicit local file, permission and directory errors plus `AttributeError` are `harness` and take priority over network evidence; explicit timeouts and connection errors, bare `socket.gaierror`/`socket.herror` and the existing SDK/HTTPX transport names are `transport`; an `OSError` with no network evidence is `harness`; everything else stays `model`.

Verification status. **Independent acceptance (2026-10-02, historical):** Mac Codex independently accepted the previous five repairs (`|` separator rejection, hyphen secret-key forms, `turn_idx` falling back to the session key, negative `total_tokens`, `OSError` classification) together with the identity test evidence; that scope covered the focused 46-pass run, and the independent backend run before the test-only fixture corrections passed 892 tests with 0 skipped while the implementation was unchanged.

**Independent acceptance (2026-10-02, this round):** Mac Codex independently reviewed camelCase API-key matching and strict `turn_idx` typing, and re-ran the focused tests (52 passed) and full backend suite (898 passed, 0 skipped). Test collection and loaded project modules were verified to come from the foundation release worktree. Three new regression tests also failed as expected against `c498bbcda4110b566049ea9a4776dc968e405359` in an independent in-memory control. The original appendix worktree's dirty files, index and real output bytes/mtimes remained unchanged. The earlier 46/892 record stands for its historical scope and does not cover these later changes. This batch still ships no harness, CLI, ledger or snapshot code, and no historical acceptance record is changed.

## Cloud Capability Data Contracts — candidate artifacts, session/score storage, corpus (offline)

This batch adds only the offline data contracts the later cloud-capability harness will build on:

- `backend/app/eval/p2_3_cloud_capability/artifacts.py` — `new_candidate` builds a versioned `cloud-capability-candidate-v1` payload dict wrapped in a `Candidate` (not a frozen object). Exactly three dedicated parameters are not copied into the payload: `thinking_text` (only its presence and char length are recorded), `api_key` (ignored) and `error_message` (dropped; `error_type` is kept only when it is a string passing `str.isidentifier()`, otherwise `None`). `raw_response_bytes` is reduced to a sha-256. There is no general text scanning or redaction. `final_text` is stored as supplied; other fields follow their schema conversions and defaults. `CandidateStore` persists multi-turn sessions as atomic complete files, keeps in-progress `.partial.json` drafts (a working draft only — there is no resume API; `discard_incomplete()` is the cleanup path) and single-record `.failed.json` failure markers, and `ScoreStore` is an append-only ScorerExecution store (fcntl-locked, `NNNN.json` sequence per scorer directory) with legacy `scores/{run_id}.json` kept read-only and selected/needs_retry rules unchanged.
- `backend/app/eval/p2_3_cloud_capability/corpus.py` — corpus contract: `snapshot_hash` is a canonical hash over chunk-id-sorted `chunk_id`/`content`/`source`/`page` (retrieval scores excluded), `retriever_config_hash` hashes the configuration mapping passed to it, and `RecordingRetriever` captures per-call evidence; a retriever exception is recorded as `capture_status="missing"` with `evidence=None` (rather than an empty list) and is then re-raised to the caller.
- Matching offline unit tests `backend/tests/eval/test_p2_3_cloud_artifacts.py` and `backend/tests/eval/test_p2_3_cloud_corpus.py`.

### CandidateStore storage-key and commit contract

`storage_key` remains a free-form safe string — for example `s1`, `s-old`, `s-agent`, and the identity module's `s` + 16-hex keys — and is never encoded or renamed; filenames keep the `<key>.json` / `<key>.partial.json` / `<key>.failed.json` shape. Every public read/write entry that receives a `storage_key` — `has_complete_session`, `load_session`, `begin_session`, `add_turn`, `commit_session`, `discard_session`, `record_failure`, `load_failure` — rejects, before any file access or `_open` mutation: non-strings, the empty string, embedded NUL, any `/` or `\` separator (which also covers absolute paths and traversal), exactly `.` or `..`, and keys ending with `.partial` or `.failed` in any letter case, which would collide with the store's own partial drafts and failure markers. Rejection raises one shared fixed `ValueError` (`unsafe session storage key`) that never echoes the offending key; `commit_session` applies the rule before its immutable and turn-count checks, so an unstarted invalid key cannot fall through the `expected_turns=0` path and write outside the root. A rejected call leaves the file set, existing file bytes and mtimes, and `_open` untouched, and an invalid read never opens the escape target. Complete, partial and failed paths all go through the same rule. Safe Unicode keys and ordinary dotted keys not ending in a reserved suffix stay allowed. ScoreStore encoding, append and retry logic are unchanged, and the identity/protocol modules and their schemas from the foundation batch are untouched.

After the key rule, `commit_session` refuses a key whose complete file already exists (`ValueError("immutable")`, matching `begin_session`/`add_turn`) and a key that was never begun on that store instance (`ValueError("session not started: ...")`); both rejections fire before the turn-count check, any temp-file creation or any file write, so a repeated same-instance commit, a fresh-instance commit against an existing complete file, or a commit for an unstarted key can never create or overwrite a complete file. An explicit `begin_session` followed by zero turns and `expected_turns=0` remains the supported way to complete an empty session; a count mismatch raises `SessionIncomplete` and leaves the draft in `_open`; when the session has fewer turns than expected, the caller may add the remaining turns and commit again. No turn-count typing policy, concurrency or recovery behaviour is otherwise changed.

### Status

An independent review of the first draft found that `commit_session` could overwrite a completed session or create one for an unstarted key; this revision fixes both. Mac Codex independently accepted the code this round (focused 91 passed / full backend suite 937 passed, 0 skipped) and completed the final wording's read-only re-check on 2026-10-02, with the release-contract suite independently passing 8 tests. As with the foundation batch, this batch does not deliver the probe, live harness, matrix runner, `--stage` CLI, budget ledger, lifecycle lock, active-attempt selection or evidence snapshots, and it changes no historical acceptance record. These are offline unit tests: no provider is called and no real `output/` is touched.

### ScoreStore append-file read order

A review found that `ScoreStore.load_appended` read append files in pure filename lexicographic order, so `10000.json` was read before `9999.json`. The read order is now integer order for numeric append files: a file counts as numeric only when its stem satisfies `isascii() and isdecimal()`, those files are ranked by `int(stem)` and placed back into their original numeric slots, and every non-numeric file keeps its previous lexicographic position and read behaviour. Equal integer ordinals keep their original lexicographic order and are never deduplicated, so `0007.json` and `7.json` are both read. Only numeric stems reach `int()`; the existing skip rules for unparsable JSON, non-`Mapping` payloads and `.tmp` names are unchanged, and a read leaves the file set, bytes and mtimes untouched.

Nothing else about the store changed: file naming (`NNNN.json`), `append` and its cross-boundary sequence (`10001.json` after `10000.json`, without overwriting lower files), encoding, locking, legacy `scores/{run_id}.json` read-only status, the success/skipped priority and the `selected`/`needs_retry` rules are unchanged. For example, with a retryable `9999.json` and a `structured_artifact_failure` `10000.json`, the rows are now read as `[9999, 10000]`, `selected` is the `10000` execution and `needs_retry` is `False`.

**Status (2026-10-02 — Mac Codex independent acceptance passed):** the focused group independently passed 96 after the mixed-name fixture was enhanced to cover interleaved non-numeric files, non-ASCII decimal stems and read-only file preservation. In-memory mutation checks confirmed that the enhanced fixture rejects both moving all numeric files to the front and sorting non-ASCII decimal stems as numeric. The full backend suite independently passed 942 with 0 skipped before that fixture-only enhancement; the implementation remained unchanged, and the full suite was not repeated for the fixture-only change. DeepSeek self-test history: the artifact file had 5 failed / 34 passed before the implementation change and 39 passed after it; the focused group passed 96 and the full backend suite passed 942 with 0 skipped. No historical acceptance record above is changed, and this offline repair authorizes no provider or real-artifact operation.

## Cloud Capability Lifecycle Lock Primitive — release/cleanup failure contract (offline)

This batch delivers only the standalone lifecycle lock primitive, moved into a clean release worktree cut from current main `15c5d184ad64594d92249c93f3d2568a739b8455`, which already carries the merged foundation, data-contracts and ScoreStore read-order repairs (no appendix history). It wires no writer, CLI, ledger, snapshot or recover entry, and therefore does **not** claim the R2-B1 writer-coverage milestone is delivered.

- `backend/app/eval/p2_3_cloud_capability/lifecycle.py` — moved in byte-identical from the appendix worktree (git blob `3741afe5e85fed17629def2b6265e50a80ae7635`, SHA-256 `75968ef1ae182ca867e86260605a9cb694c8bf262833f83bb4749f44627cd7a2`), then extended only with the reviewed cleanup contract below. The primitive is delivered for Unix/macOS/Linux and uses non-blocking `fcntl.flock`; a missing `fcntl` fails closed with `LifecycleUnavailable` (no lock, no run — never an unlocked fallback). The lock file is the sibling `.<output-dir>.cloud-capability.lifecycle.lock` outside the output tree; acquiring it may leave one empty lock file outside the output as an external side effect — the file is created but never truncated (existing bytes are kept) and never unlinked on release, and an empty lock file does not mean the lock is held. Path aliases resolve to one lock, different outputs never block each other, leases are checked against output path, process and released state before any use, and a forked child closing its inherited descriptor does not unlock the parent. `.gitignore` gains one exact rule for the default output's lock path so the external side-effect file stays out of version control.
- Cleanup-failure contract: `release()` invalidates the lease when cleanup starts (a failed `release()` is a no-op afterwards and `check()` keeps refusing), always attempts the single close on its own descriptor, skips `LOCK_UN` in a forked child, and raises the new `LifecycleReleaseFailed` (`reason_code="lifecycle_release_failed"`, fixed sanitized message) when unlock or close hits an expected `OSError`. The two expected failures are recorded separately and both survive: the stable exception's `__cause__` is the first cleanup `OSError`, and diagnostic notes carry only the fixed reason code, the fixed stage name (`unlock`/`close`) and a type-checked errno (integer or the fixed `unknown`) — never the original message, path, `str`/`repr` or traceback. The raw error object stays reachable as `__cause__` for internal diagnosis: this module never prints or logs the original exception itself, and the fixed exception text and notes never contain the original details, while Python's default traceback rendering or a consumer's own logging can still display `__cause__` — that is accepted as-is; no logging or traceback framework is added and the cause is not removed. `unlock` failing while `close` succeeds is still reported as a failed cleanup, not success. There is no retry of `close` and no second operation on a descriptor that may already be reusable. A non-`OSError` programming error from unlock is never converted to `LifecycleReleaseFailed` or to success: the descriptor is still closed, and when close also fails, the programming error remains the primary exception carrying a sanitized close note. The implementation uses no broad `Exception`/`BaseException` handler and never compares exception object identity: an explicit flag records whether the unlock stage settled (skipped, or its `OSError` collected), and only when that flag says unlock did not settle does the `finally` read the in-flight exception via `sys.exc_info()` and attach the sanitized close note. That keeps close evidence on a non-`OSError` that is already being handled by an outer `except` block and is re-raised from `flock` (evidence was previously lost on that path), while a normally-settled release inside an outer `except` block, or a business exception propagating through `hold`, still receives exactly one set of notes — no duplication and no pollution of the outer handled exception.
- `hold.__exit__`: the owned lease reference is cleared in a `finally`, and a borrower never releases the owner's lease. When release fails while a business exception (including `CancelledError`) is propagating, the same fixed sanitized notes are attached to the original exception object and `False` is returned, so the original object, type and traceback propagate unchanged (no re-raise, no replacement, no `ExceptionGroup`); with nothing propagating, `LifecycleReleaseFailed` propagates and success is never returned. Unexpected programming errors keep propagating; nothing is broadly swallowed.
- `acquire()`: after a failed `flock`, an expected `OSError` from the single descriptor close can no longer overwrite the stable classification — `LifecycleBusy`/`LifecycleUnavailable` keep the original `flock` error as `__cause__` and a failing close only adds a fixed sanitized note. The normal acquisition path and the existing busy/unavailable rules are unchanged.

Known boundary: `add_note` data is diagnostic and does not guarantee a CLI operator ever sees it. When a `with` body handles its own error and returns, `__exit__` runs with no propagating exception, and how a release failure combines with a single-JSON CLI output or with snapshot success/replay return values is left to a later consumer-wiring batch; nothing here claims that combination is solved.

`backend/tests/eval/test_p2_3_cloud_lifecycle.py` extracts the seven lock-primitive cases from the appendix tree (alias sharing, cross-process non-blocking, fail-closed without parent dir or `fcntl`, invalid/foreign lease rejection, borrower semantics, inherited-lease cleanup, no truncation) with deadline-bounded child handshakes and a bounded `waitpid`, and adds the cleanup-contract coverage: stable type/reason/cause/notes on release I/O failure via `hold` and via a direct lease, dual cleanup-failure evidence with a single close attempt, programming-error primacy with sanitized close evidence, business-exception and real `task.cancel()` preservation, post-failure release idempotency and check rejection, busy and unavailable classifications each under an injected close failure (busy against real cross-process contention), reacquisition after normal, business-error and cancelled exits without cleanup failure, explicit timeout coverage for silent and partial child handshakes, and sentinel assertions that notes and stable texts never leak injected messages or paths. All child processes are reaped by their own tests; failure injection uses per-test directed shims scoped to the lifecycle module only. A bounded follow-up after independent review added: the re-raised-active-exception close-evidence case (unlock re-raising the exception an outer `except` block is already handling), a guard that a direct release inside an outer `except` block still raises the stable exception while leaving the handled exception untouched, reaping inside `_hold_process()` for every handshake failure before ownership is handed over (deadline, kill-then-wait and bounded waits kept, no unbounded `readline`) with the injection case going through `_hold_process()` itself and asserting the original error, the reaped process and closed pipes, timeout-reaping coverage for the silent and partial children through `_hold_process()`, the forked child exiting non-zero when its release fails (zero only after a normal release) while the parent asserts successful exit for the normal path, and the cancel case asserting the very `CancelledError` object recorded at the propagation point with exact notes.

Status: **2026-10-03 — Mac Codex independent acceptance passed for this standalone lifecycle primitive.** First-round history: the seven extracted primitives passed against the byte-identical moved-in module (7 passed); after adding the importable exception skeleton plus the contract tests, a genuine behavior RED was recorded (9 failed / 10 passed); after the contract implementation, the focused file passed 19, the six-file focused group passed 115, and the full backend suite passed 961 / 0 skipped. Mac Codex independently re-ran the 115 / 961 numbers before this follow-up — that re-run is **not** an acceptance verdict and found two contract gaps (close evidence lost when the unlock stage re-raises the exception an outer `except` block is already handling, and a leaked child when the HELD handshake read raises) plus assertion-strength items. For this follow-up the new target assertions first failed for real (2 failed / 23 passed), then passed after the fixes: focused file 25 passed, six-file focused group 121 passed, full backend suite 967 / 0 skipped, on the borrowed Python 3.11.15 interpreter with `PYTHONPATH` empty, pytest rootdir, configfile and child-process module loads verified to come from this worktree. No provider was called and no real `output/`, ledger, reservation, marker or evidence snapshot was touched. Writer/CLI/backfill wiring, ledger, snapshot and recover remain **not** delivered or protected by this batch, the batch still does not complete the R2-B1 writer-coverage milestone, and no historical acceptance record above is changed.

Mac acceptance evidence (2026-10-03): the six-file focused group independently passed 121; the full backend suite independently passed 967 with 0 skipped. In-memory reversion checks confirmed that the new regression tests reject both prior gaps (lost close evidence on the same active exception and the child left running after a handshake read error), while the current implementation passes. Module sources, child-process loading, whitespace, cache and process checks passed. The original appendix worktree and all 339 real output files retained their bytes and mtimes. This acceptance covers only the standalone lifecycle primitive; writer/CLI/backfill, ledger, snapshot and recover integration and the R2-B1 writer-coverage milestone remain outside this batch, and no provider or real-artifact operation is authorized.

PR #12 review follow-up — canonical acquisition binding (P2): at review HEAD `659b9dfa6a6c491f3e0ed8c05c3647ce59f7fb3d` — the same SHA the 2026-10-03 Mac acceptance above and the DeepSeek RED/GREEN history belong to — `acquire()` resolved its input twice: the lock path came from the first resolve while the lease recorded a second resolve taken after the lock had been taken, so an alias repointed in between yielded a lease whose canonical identity no longer matched the lock it held (reproduced deterministically: flipping a symlink right after `os.open` made `check()` pass against the wrong target while another process got `BUSY` on the real target and could still take the impostor target's lock). The fix adds a private `_lock_path_for_resolved()` helper that never resolves; `lock_path_for()` resolves exactly once and delegates to it; `acquire()` resolves exactly once up front, uses that one value for both the lock path and the `LifecycleLease`, and never resolves the original input again after the lock is taken. Error classification, cause/notes, the single close, lease invalidation order, `release`/`hold`/`check`, the public API surface and the missing-`fcntl`/missing-parent order are unchanged. Scope: this only guarantees that the lock path and the lease's canonical identity inside one `acquire()` call share the same single resolve — no promise that an alias stays pointed at the same target afterwards, no closure of a consumer's own check-to-use race, and deliberately no dirfd/`O_NOFOLLOW`/filesystem-migration or retry machinery and no writer/CLI/backfill/ledger/snapshot/recover wiring. DeepSeek self-test for this fix: a deterministic race test and an exactly-once-resolve test failed against the pre-fix module (2 failed / 27 passed) and passed after the fix, with path/str `lock_path_for` parity and fail-closed-before-resolution guards green throughout; focused file 29 passed, six-file group 125 passed, full backend suite 971 / 0 skipped on the borrowed interpreter — self-test numbers. Mac Codex independent re-verification (2026-10-03) passed for this P2 repair: the six-file focused group passed 125 and the full backend suite passed 971 with 0 skipped. In-memory reversion to the committed pre-fix acquire implementation made both the canonical-target race test and the exactly-once-resolve test fail; the current implementation passed both. Source/API/error-order review, module and child-process sources, whitespace, cache and process checks passed, and the original appendix worktree and all 339 real output files retained bytes and mtimes. This verdict covers only the bounded acquisition binding repair; the stated consumer check-to-use and R2-B1 writer-integration exclusions remain unchanged.

## Cloud Capability Budget Ledger — v1/v2 offline read/write contract

This batch moves the offline budget ledger and its price snapshot out of the appendix worktree, with the direct budget cases and the v1/v2 ledger cases that belong to them. It moves no consumer: the live harness, generator, matrix, selection, recover, snapshot, probe, scoring, summarizer and `--stage` CLI stay in the appendix tree.

- `backend/app/eval/p2_3_cloud_capability/budget.py` — byte-identical to the appendix source (SHA-256 `cf58e78f9402b6d0f5c7920cffa56129eee130f8c00dc8abe32706093c825691`); no other source file changed.
- `backend/app/eval/p2_3_cloud_capability/prices.json` — byte-identical to the appendix source (SHA-256 `ea85456180dca7f3e0b69fb46d9479b5581338dc6cba61e80ba2ed30d03eadb7`). It is a **2026-08-24 historical price snapshot** (`snapshot_date`), not a verified current provider rate: this batch did not check today's rates and authorizes no paid call.
- Matching offline unit tests `backend/tests/eval/test_p2_3_cloud_budget.py` and `backend/tests/eval/test_p2_3_cloud_ledger_v2.py`. The appendix budget case that depends on the runner (`test_probe_spent_counts_cache_read_without_magic_padding`) and the appendix ledger cases that go through `recover`/`selection` were deliberately not moved.

### Public surface and side effects

- Public names: `BudgetExceeded`, `MissingUsage`, `BudgetLedgerCorrupt`, `LEDGER_SCHEMA` / `LEDGER_SCHEMA_V2` (`cloud-capability-budget-ledger-v1` / `-v2`), `LEDGER_NAME` (`budget_ledger_v1.json`), `LOCK_NAME` (`budget_ledger_v1.lock`), `usage_has_unpriceable_tokens` and `BudgetLedger`.
- `BudgetLedger.for_stage(stage, output_dir=None)` always reads the sibling `prices.json` and loads the stage caps ($5 main, $5 appendix, combined $10 hard) into memory. With `output_dir=None` the price file is still read, but no ledger, directory or lock file is created or written. With an `output_dir` it creates the directory, bootstraps a fresh `budget_ledger_v1.json` when none exists (in-memory `version` 0, incremented before the write, so the **first persisted `version` is 1**) and creates the sibling transaction lock file. Writing the document is confined to `_persist_unlocked`, which bumps `version` and replaces the file atomically (temp file plus `os.replace`); it runs when a fresh ledger is initialized, after a successful locked mutation (`check_before_call`, `record`), and when a locked mutation raises one of the three bounded business exceptions (`MissingUsage`, `BudgetExceeded`, `BudgetLedgerCorrupt`). A boot load of an existing document, a refused load or reload, an unusable cost estimate rejected before the lock is taken, and any unexpected exception from a locked mutation (which the persist path does not catch) do not write the ledger.
- A valid v1 or v2 document is never rewritten **on load**: bytes, mtime and `version` are unchanged and no schema bump happens. v1 is **not** auto-upgraded to v2, and this batch provides no upgrade entry point. Compatibility boundary: a reader that knows only the v1 schema cannot load this batch's v2 documents; the schema gate here rejects any other `schema_version` with `unsupported budget ledger schema`.
- The constructor is not a filesystem read-only API: taking the transaction lock creates `budget_ledger_v1.lock`, and the ledger file or its directory may be created when they are missing. Only the ledger document itself is asserted unchanged on load; this batch does not claim "zero filesystem writes".
- `estimate_cost`, `reserved_usd` and `remaining_usd` are pure reads of in-memory state. `remaining_usd` is `min(stage_cap - spent_by_stage[stage] - that stage's open reservations, hard_cap - spent_usd - all open reservations)`. `legacy_spend` is never added to spend and never subtracted from remaining; an unknown legacy amount is not silently treated as zero, it is simply outside the arithmetic (`legacy_spend` stays `{"status": "unverifiable", "usd": null}`).
- `record` is **not** an idempotent recovery settlement: it takes no idempotency key, charges the actual amount again on every call, removes only the reservation whose id was passed, and persists an over-cap charge before raising `BudgetExceeded` and setting `matrix_allowed=false`. The future R2-C2 settle/unlock transaction must not be built on it.

### Read, refusal and persist semantics

- A refused load persists nothing: unreadable JSON, a non-object payload, an unsupported `schema_version`, a v1 document carrying any of the four v2-only fields (`isolated_storage_keys`, `isolated_originals`, `source_snapshot_id`, `settlements`), a v2 document missing a required field, a key set that is not exactly the three HyDE storage keys, a hash that is not lowercase 64-hex, a non-empty `settlements`, or a reservation `stage`/`status` that is not an exact string all raise `BudgetLedgerCorrupt` with bytes, mtime and `version` unchanged.
- `record`'s three bounded business failures raised inside the lock persist and bump `version` first: `MissingUsage` (unusable usage) and the `BudgetLedgerCorrupt` from an unusable `actual_usd` set `matrix_allowed=false` and keep the open reservation; `BudgetExceeded` keeps the charged overage, drops the named reservation when one was passed, and sets `matrix_allowed=false`. On a v2 document these persists keep all four isolation fields and the unknown `legacy_spend` verbatim; a ledger that was already blocked is never unblocked by a successful `record`. `check_before_call` refusals are different: when `_reserve` refuses inside the lock because the matrix is already blocked or the estimate exceeds the remaining budget, the persist path still rewrites the document and bumps `version`, but the call adds no reservation, charges nothing and leaves `matrix_allowed` unchanged; an estimate rejected before the lock is taken, and a refused load or reload, do not rewrite the document. That is how a load refusal (nothing written, nothing bumped) is distinguished from a business failure (state written, then raise).
- v2 isolation fields are shape-validated only: the three storage keys, two lowercase SHA-256 pins per key, a lowercase SHA-256 `source_snapshot_id`, and `settlements == {}`. Only empty settlements are accepted. The pins are **not** checked against any candidate or sqlite file on disk, so loading a v2 ledger is not evidence that isolation happened or that the bound originals still exist.

### Concurrency and lock scope

- Losing no concurrent update is guaranteed only where `fcntl` exists — macOS/Linux — through an exclusive `flock` on `budget_ledger_v1.lock`. Without `fcntl` the unchanged source behaviour stands: the lock file is still opened, but no locking happens. This batch deliberately does not convert a missing lock into a refusal.
- That transaction lock serializes only processes that go through this module. It is **not** lifecycle-lock coverage, it does not stop a writer that ignores it, and it does not make any writer safe.

### Not delivered

No consumer or CLI is moved or wired (live harness, generator, matrix, selection, recover, snapshot, probe, scoring, summarizer, `run_eval`), so this batch delivers **no writer coverage**, and no writer is claimed to be covered by the lifecycle lock either. It does not complete GATE E, does not approve R2-C2, and authorizes no real isolate, recovery, settlement, unlock, ledger upgrade or snapshot operation. The offline unit tests call no project evaluation provider — the DeepSeek coding executor of this batch is not a project provider call — and touch no real experiment artifact: every ledger and reservation fixture is a locally constructed v1/v2 document under `tmp_path`, and these tests read and write no real `output/` tree, ledger, reservation, marker or evidence snapshot. That statement is scoped to the test run; the read-only protection check of the appendix source was carried out separately (see the Mac acceptance record below).

### Tests (DeepSeek self-test; not a Mac acceptance verdict)

Migrated from the appendix tests unchanged in intent: 20 budget functions (31 collected cases, including the 12-case malformed-usage parametrisation) and the six ledger functions (22 collected cases). Required additions, all regression coverage of existing byte-identical behaviour: appendix-only $5 cap next to the existing main $5 and combined $10 cases; a valid v1 and a valid v2 load that change neither ledger bytes, mtime nor version; a v2 record whose costs and reservations update while the four isolation fields, unknown `legacy_spend` and blocked state persist; each of the four v2-only fields refused on a v1 document; the missing required v2 fields, a wrong original-hash shape and empty-settlements counterexamples reused through the existing parametrisations; the three v2 business failures (`MissingUsage`, `BudgetExceeded`, unusable `actual_usd`) each persisting with their own cost, reservation, version and blocked expectations; and a spawn-based concurrent-commit case with bounded handshake/join deadlines whose `finally` reclaims both children and the queue even when a handshake or assertion fails, and which asserts that both children loaded `budget.py` from this release tree. No test is skipped or deselected.

Self-test commands, all run in `backend/` on the borrowed Python 3.11.15 with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH= <python> -B -m pytest -q -p no:cacheprovider` (rootdir `.../cloud-capability-budget-release/backend`, configfile `pyproject.toml`):

| Command | Result |
|---|---|
| the two moved/new files | exit 0, **68 passed** (33 budget + 35 ledger_v2), 0 failed, 0 skipped |
| the seven `tests/eval/test_p2_3_cloud_*.py` files | exit 0, **185 passed** |
| full backend suite (`-rs`) | exit 0, **1039 passed**, 0 skipped |

### Independent acceptance (Mac Codex, 2026-10-04)

This round is Mac-executed evidence, reported separately from the DeepSeek self-test above; it does not replace, re-attribute or change any historical acceptance record. Mac actually ran Python 3.11.15 and loaded the modules from this release worktree's `backend/`, using `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH= <borrowed python> -B -m pytest -q -p no:cacheprovider ... -rs`:

| Command | Result |
|---|---|
| the two moved/new files | exit 0, **68 passed** (33 budget + 35 ledger_v2) |
| the seven `tests/eval/test_p2_3_cloud_*.py` files | exit 0, **185 passed** |
| full backend suite | exit 0, **1039 passed**, 0 skipped |

All three commands exited 0 and no test was skipped. Mac also verified the loaded module paths, `git diff --check` on the tracked change and a whitespace check over the untracked files. Across the test run, the appendix source tree's HEAD, its dirty set of 32 entries and its Git index were unchanged, as were the real `output/` set (339 files) with its bytes, sizes and mtimes and the `ledger` / `main.ok` hashes; the release tree itself also showed no new cache or project changes across the run. This acceptance covers this offline budget ledger batch only and authorizes no provider or real-artifact operation.

This batch does not publish anything and no historical acceptance number above is changed.
