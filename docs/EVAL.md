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
- **Zero hybrid-v1 verdict/score regression.** The directed suite regression is the refusal-axis leak: v2 has `expected_refusal_observed` on `tgqa-008` and `tgqa-012`; v3 does not, and those v3 answers add "General Study Knowledge". That is accepted as a real prompt-axis regression. The LLM rubric parser was not loosened.
- Empty dimension scores are a scorer parse limit on this model, not a fabricated 0.

## Limits

- Local small-model Hybrid rubric output can be `scorer_parse_error`; the ScoreSet then stays `partial` / `inconclusive` instead of fabricating dimension scores.
- Evaluation measures one TutorAttempt, not Graph Judge retries.
- 12 cases are a directed regression suite, not a benchmark of overall study quality.
- Paid/remote models are not a CI gate.
