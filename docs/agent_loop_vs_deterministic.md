# Did I build a real agent? An empirical answer.

> A response to `learn-claude-code`'s "agency = model + minimal harness" thesis,
> with 396 runs across 4 Ollama models × 2 architectures.

## The thesis I was responding to

`learn-claude-code` (the public repo at github.com/jasonkneen/learn-claude-code, README at `LEARN-CLAUDE-CODE-README-zh.md`) makes a clean claim:

> "Agency comes from the model. The harness should be minimal. A well-trained model plus a `while(tool_use)` loop is enough."

The author backs this up with s01-s04: a 30-line Python loop that drives Claude to use bash, file tools, and a subagent. It works. It's elegant. It's the antidote to overweight frameworks.

I bought 70% of this argument. The 30% I was suspicious about: **what happens when the model isn't Claude?** What happens at the 4B-8B local Ollama tier where you actually have to live without thousand-dollar inference budgets?

So I built it. Twice.

## What I built

I'm refactoring an HKBU class project into a portfolio-grade exam-coach agent called **Study Coach** (FastAPI + LangGraph + Vue 3). The Plan node — the part that takes "make me a study plan on HyDE" and produces dated milestones — was already shipped in P2.1-⑤ as a **deterministic state machine**: extract topic → resolve goal → retrieve PDF chunks → LLM milestones JSON → persist via `update_study_plan`. Single LLM call, fixed control flow.

For P2.2, I added a **parallel agent-loop variant** at `app/agent/planner_agent.py`. Same LangGraph node contract. Same SSE shape. Same judge wraparound. Only difference: the model drives the planning via `bind_tools` over 5 LLM-visible tool wrappers (`retriever_search` / `get_existing_plan` / `update_study_plan` / `generate_mindmap` / `compute_progress`), exit on `not response.tool_calls`, `max_iter=10`.

The two paths share *everything* else — same memory_hydrator, same judge_node, same memory_writer. A `mode-aware dispatcher` in `plan_node` routes between them based on a `x-planner-mode` HTTP header.

**This is the fair A/B I wanted.** Same Pydantic schemas. Same database. Same retrieval. Same judge. The ONLY thing that varies is whether a Python state machine or an LLM drives the planning step.

## What I measured

4 models × 2 modes × 14 queries × 3 runs + appendix = **396 runs**. About 5 hours sequential on a 16GB Apple Silicon Mac. Models: `gemma3:4b` (no `tools` capability per Ollama manifest), `qwen3.5:4b` (thinking + tools), `qwen2.5:7b` (instruction-tuned + tools, no thinking), `gemma4:e4b` (thinking + tools + multimodal). Dual judges: `qwen2.5:7b` local + `MiniMax-M2.7` cloud.

Full tables at `docs/EVAL.md`. Here's what survived sanity-checking.

## The findings that hold

### Finding 1: minimal harness can't rescue a model that wasn't trained for tools

`gemma3:4b agent_loop`: **42/42 runs fail in 100ms** with Ollama API 400 `does not support tools`. The model manifest explicitly lacks tools capability; the API rejects the request before the model runs.

The s01-style `while(tool_use)` loop has nothing to loop over — there are no tool calls because there's no tool call API. Minimal harness here doesn't degrade gracefully into "model produces less-good output" — it degrades into "no output at all". Our `_format_degrade_output(reason="llm_call_failed")` catches the exception and emits a user-visible disclaimer, but that's harness handling failure, not agency emerging.

**Implication for the thesis**: agency isn't *just* "model + minimal harness". It's "**model trained for tool use** + minimal harness". The training matters.

### Finding 2: tool schemas are a form of harness that learn-claude-code's framing doesn't count

The `Milestone` Pydantic schema is 4 lines:
```python
class Milestone(BaseModel):
    title: str
    due_at: str | None = None
    done: bool = False
    topic: str | None = None
```

I ran `qwen3.5:4b` deterministic with `reasoning=False` (forced thinking-off for speed): persistence rate collapsed to **10%**. The model emits garbled JSON when not allowed to think.

Same model, same `reasoning=False`, but in agent_loop mode: persistence jumped to **86%**. Why? Because `update_study_plan`'s tool wrapper validates milestones with `Milestone.model_validate(m)`. When validation fails, the wrapper returns `{"error": "invalid milestone at 0.title: Field required"}` as a `ToolMessage`. The model sees the error and retries with valid args.

**The schema is doing the structural reasoning the model isn't doing internally.** It's a 4-line schema in code, but it's not "minimal harness" — it's a *type-driven correction loop* that the s01 example deliberately doesn't have.

Same effect on `gemma4:e4b`: 43% deterministic persistence → 94% agent_loop persistence with `reasoning=False`. The cleanest external evidence I have that **tool schemas substitute for in-model reasoning**.

### Finding 3: agent_loop is sometimes worse than deterministic, and the data tells you when

`qwen2.5:7b`: an instruction-tuned 7B with tools capability but no native thinking. By the local judge, agent_loop and deterministic tied (both 0.770). By the cloud judge, deterministic beat agent_loop (0.584 vs 0.469).

But the operational data tells the story: deterministic is **3× faster** (7s vs 21s), **2.25× more persistent** (90% vs 40%), and emits **12% tool error rate**. The agent loop runs to completion (42/42 `natural_stop`) but the multi-step coordination introduces failure modes the deterministic path bypasses by construction.

**For this model tier, "agent" is the wrong abstraction.** The deterministic state machine fits better. The harness here isn't minimal — it's an active drag.

### Finding 4: the cleanest agency comes from 8B + thinking + tools

`gemma4:e4b agent_loop`: 72/72 `natural_stop`. 2.74 tool calls/run. 94% plan persistence. Top-tied judge scores on both local (0.856) and cloud (0.606). Mean wall time 48s — slower than deterministic 25s, but quality consistently higher.

This is where the s01-style thesis actually holds. Tools work. Thinking is on by default (we tested both). The model coordinates, the loop dispatches, the judge approves. The harness here genuinely is minimal — the model is the agent.

This tier is what `learn-claude-code` was implicitly assuming. Below 8B without thinking, it doesn't hold.

## Where the thesis lands

`learn-claude-code`'s "agency = model + minimal harness" survives at the 8B+ thinking-and-tools tier. Below it, the data refutes the strong form of the claim:

- **At 4B without tools**: no harness fits, period
- **At 7B older instruction-tuned + tools**: deterministic beats agent loop on objective metrics
- **At 4B+ thinking + tools but `reasoning=False`**: tool schemas substitute for thinking — this is "harness as structural reasoning", not "harness as minimal"
- **At 8B+ thinking + tools + reasoning on**: thesis holds

The unstated assumption in s01-s04 is that you're using Claude — which is the upper-right corner of every dimension (Anthropic-trained for tools, has thinking, has reasoning enabled, has 70B+ scale). When you reach for the local tier, you discover that *all four* of these capabilities matter, and missing any one of them means the harness has to compensate or fail.

**My answer**: the thesis is conditional. It's a true claim *given the model conditions*, and the model conditions are precisely what most of us don't have at home.

## What I would do next

- **P2.3**: same ablation on the Quiz path (different task, more constrained tool surface). I expect the gemma3-tier collapse to repeat, but the persistence rescue effect may not — quiz tools are simpler.
- **A non-thinking cloud judge** for cross-validation: this run used MiniMax-M2.7 (thinking model) and qwen2.5:7b (no thinking) as dual judges. Adding GPT-4o-mini (no thinking) would split the thinking-vs-non-thinking judge bias from the local-vs-cloud bias.
- **A mixed-mode dispatcher** that picks deterministic vs agent_loop based on the planner model's known capabilities (e.g., `gemma3:4b` auto-routes to deterministic). Avoids the gemma3-tier degrade entirely without losing the agent_loop option for capable models.

I'm not done. But I have the data the HKBU class report and `learn-claude-code` both lack: a head-to-head A/B at the local Ollama tier with statistical depth and dual judges, on the same task and same retrieval.

## Reproduce

```bash
git clone <study-coach repo>
cd study-coach/backend
uv sync
ollama pull gemma3:4b qwen3.5:4b qwen2.5:7b gemma4:e4b
uv run pytest -q           # 181 tests pass
export MINIMAX_API_KEY=sk-...   # optional, skips cloud judge if absent
uv run python -m app.eval.p2_2_agent_ablation.run_eval \
  --queries app/eval/p2_2_agent_ablation/queries.json \
  --output /tmp/results.jsonl \
  --runs 3 \
  --thinking-appendix
uv run python -m app.eval.p2_2_agent_ablation.summarize /tmp/results.jsonl
```

Total wall time on 16GB Apple Silicon: ~5 hours. Cost: ~$3 MiniMax API for 396 cloud judgments.

---

*Repo for this work: [Study Coach](#) (refactor of HKBU class project).
Repo this argues with: [learn-claude-code](https://github.com/jasonkneen/learn-claude-code) — full credit to the author for the cleanest articulation of the "minimal harness" thesis I've seen.*
