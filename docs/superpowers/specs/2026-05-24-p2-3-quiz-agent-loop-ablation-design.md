# P2.3 Quiz Agent Loop Ablation — Design Spec

> Brainstormed 2026-05-24 via `superpowers:brainstorming`.
> **Goal**: build an LLM tool-calling agent-loop variant of the Quiz GENERATE path, run head-to-head against P2.1-④ deterministic baseline, produce empirical data answering: does the agent_loop pattern's value transfer from Plan (P2.2) to Quiz?
> **Upstream**: P2.2 (Planner agent_loop shipped, 181 tests passing, 396-run matrix, EVAL.md + blog published).
> **Portfolio narrative**: complete the 3-way comparison promised in `docs/agent_loop_vs_deterministic.md`'s `What I would do next`. Test 3 specific predictions from that blog on a new task with a different schema strictness profile.

---

## 1. Decisions Locked (Brainstorming Q&A)

### Q1 — Agent loop scope: GENERATE-only

**Locked: agent_loop covers GENERATE turn only; GRADE turn stays deterministic in BOTH modes.**

Rationale:
- Deterministic `grade_quiz_answer` has no LLM variation (rule-based string match against stored answer + mastery delta + optional record_mistake). An agent_loop on GRADE would be the LLM dispatching one tool call then stopping — nothing to ablate.
- P2.1-④g already established judge skips GRADE turns (deterministic, nothing to LLM-judge). P2.3 mirrors that boundary.
- Keeps A/B clean: the **only** variable is GENERATE-turn architecture.

Routing implementation: `graph.py:quiz_node` checks `state.active_quiz_question_id` BEFORE consulting `configurable.quiz_mode`. If present (GRADE turn) → always deterministic. If absent (GENERATE turn) → mode-aware dispatch.

### Q2 — Tool surface: 2-tool minimal

**Locked: `retriever_search` + `persist_quiz_question` only.**

| Tool | LLM-visible args | Closure-injected context | Returns (JSON string) |
|---|---|---|---|
| `retriever_search` | `query: str, top_k: int = 5` | `retriever` | `[{"chunk_id","content","page"}, ...]` |
| `persist_quiz_question` | `topic: str, prompt: str, options: list[str], answer: Literal["A","B","C","D"], explanation: str` | `user_id, topic_repo, question_repo, goal_repo` | `{"question_id","topic_id","persisted":true}` or `{"error": "..."}` |

Rationale:
- **Fair A/B**: deterministic `quiz_master.py` uses 1 main LLM call (the generation prompt inside `generate_quiz`). agent_loop also does 1 main LLM call (the dispatcher loop, with LLM emitting question text in message content + calling `persist_quiz_question`). Both modes = single LLM-as-generator pattern. If agent_loop used `generate_quiz` as a tool, LLM would dispatch → tool internally calls LLM = 2 calls per turn = unfair comparison.
- **Mirror P2.2 architectural framing**: LLM is the creator, tool is the persistence layer (P2.2's `update_study_plan` is structurally identical).
- **Direct test of blog prediction 2 ("persistence rescue may not manifest on Quiz")**: Pydantic `QuizQuestionPersist` schema enforces `len(options) == 4`, `answer` Literal, options prefix validation, non-empty prompt/explanation — **strictly more constraints than P2.2's `Milestone` model**. If rescue effect still holds, the schema-rescue mechanism generalizes; if it changes shape, that's the finding.
- Adaptive features (mastery-aware difficulty, mistake-bank avoidance) are P3 frontend territory — out of scope for ablation.

### Q3 — Matrix shape: mirror P2.2 footprint

**Locked: 4 models × 2 modes × (10 single + 2 multi-turn) × 3 runs + gemma4:e4b thinking-ON appendix ≈ 396 records.**

| Models | Modes | Queries | Runs | Appendix |
|---|---|---|---|---|
| gemma3:4b, qwen3.5:4b, qwen2.5:7b, gemma4:e4b | deterministic, agent_loop | 10 single-turn GENERATE + 2 multi-turn (GENERATE→"A"→GRADE) | 3 per cell | gemma4:e4b thinking-ON × 2 modes × 10 single × 3 = 60 records |

**Topics overlap with P2.2** by design (HyDE / BM25 / reranking / chunking / eval / judge / embeddings / hybrid). Lets future writeup do side-by-side "same model on Plan vs Quiz" task-generalization comparison.

Multi-turn handling: turn 2 fixed reply `"A"`. Sometimes correct, sometimes wrong (depending on what LLM generated as `answer`). GRADE turn is identical code path in both modes (deterministic short-circuit), so turn 2's correctness does NOT affect the ablation comparison — it only verifies the dispatcher routing is correct.

Budget: ~5h wall time on 16GB Apple Silicon Mac, ~$3 MiniMax-M2.7 cloud judge cost (+ ~$0.5 appendix). Final record count = 336 main + 60 appendix = 396, matching P2.2 footprint exactly. Detailed breakdown in §6.1.

### Q4 — Judge: dual same as P2.2

**Locked: qwen2.5:7b local + MiniMax-M2.7 cloud, rubric → `QUIZ_DIMENSIONS` (existing from P2.1-④d).**

- Local judge: `qwen2.5:7b` with `app/agent/prompts/judge_quiz.txt` (5 dims: question_quality / option_plausibility / answer_correctness / explanation_clarity / difficulty_calibration)
- Cloud judge: `MiniMax-M2.7` same rubric template

Rationale:
- **One variable at a time**: P2.3 changes the task (Plan→Quiz). Holding judge setup constant maximizes comparability with P2.2 findings.
- Blog's promised "GPT-4o-mini cross-validation" is deferred to a **separate** P2.4 judge-bias ablation (or P3 appendix). Mixing task change + judge change makes any finding direction ambiguous.

Caveat carried over: MiniMax-M2.7 thinking content lands in `message.content`, not `reasoning_content`. Existing greedy regex in `judges.py` parses correctly. No change needed.

### Q5 — Thinking appendix: include

**Locked: gemma4:e4b thinking-ON appendix runs included (60 extra records, single-turn only, both modes).**

Mirrors P2.2's appendix structure exactly (60 records). Tests whether thinking-on shifts schema rescue magnitude on Quiz task specifically (P2.2 found thinking-off + tools schema = rescue mechanism on Plan; Quiz schema is stricter, so the interaction with thinking may differ).

Multi-turn queries excluded from appendix to keep the thinking-on/off comparison focused on single-turn GENERATE quality (avoids confounding by GRADE-turn noise).

Cost addition: ~45 min wall time, ~$0.5 cloud judge cost.

### Approach choice

**Approach A**: Mirror P2.2 exactly — hand-written while-loop + new `quiz_master_agent.py` parallel to `quiz_master.py`; mode-aware dispatcher in `graph.py:quiz_node`; fork eval harness `p2_2_agent_ablation/` → `p2_3_quiz_ablation/`.

Rejected alternatives:
- **Approach B** (refactor `quiz_master.py` to be mode-aware internally) — breaks byte-identical baseline → fair A/B compromised
- **Approach C** (use `langgraph.prebuilt.create_react_agent`) — abstracts away the structural details (tool schema strictness, exit conditions, iteration budget) we want to test

---

## 2. Architecture & Graph Topology

```
START
  → memory_hydrator                                  (unchanged)
  → router (state-aware: active_quiz_question_id → "quiz")
  → {tutor → judge | quiz_node | plan_node → judge}
              ↓
       quiz_node = state-aware + mode-aware dispatcher (NEW logic)
              │
              ├─ if state.active_quiz_question_id → ALWAYS deterministic GRADE
              │
              └─ else (GENERATE turn) → read configurable.quiz_mode
                                          │
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
                quiz_master_deterministic         quiz_master_agent_loop
                (P2.1-④ baseline,                  (new in P2.3-①)
                 byte-identical)
              ↓
       → judge (GENERATE only — quiz_action="grade" still skips)
       → memory_writer → END                       (unchanged)
```

### 2.1 Why state-check before mode-check

Quiz's GRADE-vs-GENERATE distinction is **structural** (different control flow path, different state preconditions). Mode is **experimental** (which architecture handles GENERATE). State-check first ensures GRADE turn is always deterministic regardless of mode — matches the agent_loop scope decision and prevents the agent_loop module from ever needing to know about GRADE.

### 2.2 Files touched

| File | Change | Lines (est.) |
|---|---|---|
| `app/agent/quiz_master_agent.py` | NEW — mirror of `planner_agent.py` shape | ~400 |
| `app/agent/agent_trace.py` | NEW — targeted refactor: extract `AgentTrace` + `IterationRecord` + `ToolCallRecord` from `planner_agent.py` for shared use | ~150 |
| `app/agent/planner_agent.py` | EDIT — `from app.agent.agent_trace import AgentTrace, ...` (no logic change) | ~5 |
| `app/agent/quiz_master.py` | BYTE-IDENTICAL (fair A/B) | 0 |
| `app/agent/graph.py:quiz_node` | EDIT — add state-check + mode-aware dispatch (~15 lines) | ~15 |
| `app/agent/state.py` | NO CHANGE — `agent_trace` field already exists from P2.2 | 0 |
| `app/api/deps.py` | EDIT — add `get_quiz_mode` (reads `x-quiz-mode`) + `get_quiz_master_agent` factory | ~25 |
| `app/api/routes.py` | EDIT — chat() signature + 2 new Depends + 2 new configurable keys | ~10 |
| `app/agent/tools/schemas.py` | EDIT — add `QuizQuestionPersist` Pydantic model | ~15 |
| `app/eval/p2_3_quiz_ablation/` | NEW package — fork of `p2_2_agent_ablation/` | ~550 |

**Boundary discipline**: deterministic `quiz_master.py` and all P2.2 production code (`planner.py`, `planner_agent.py` core logic) byte-identical. Only `planner_agent.py` import line changes (refactor-induced).

### 2.3 Why refactor `AgentTrace` now

CLAUDE.md: "Where existing code has problems that affect the work (e.g., a file that's grown too large, unclear boundaries, tangled responsibilities), include targeted improvements as part of the design."

`AgentTrace` was inlined in `planner_agent.py` because P2.2 had a single caller. P2.3 introduces the second caller. Duplicating the dataclass = code smell; cross-module reach into `planner_agent.py` for shared instrumentation = unclear boundary. **Moving to `app/agent/agent_trace.py` is the minimum change that keeps both files clean.**

Backward compatibility: `planner_agent.py` import-only diff. AgentTrace public API unchanged. P2.2 tests pass byte-identical.

---

## 3. Tool Wrappers & Schemas

### 3.1 New Pydantic schema

```python
# backend/app/agent/tools/schemas.py — append

class QuizQuestionPersist(BaseModel):
    """Schema for the persist_quiz_question agent tool.

    Strictly more constrained than Milestone (P2.2):
    - options: list of EXACTLY 4 strings, each prefixed "A) "/"B) "/"C) "/"D) "
    - answer: Literal["A","B","C","D"]
    - prompt and explanation: non-empty strings
    """
    topic: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def _check_option_prefixes(cls, v: list[str]) -> list[str]:
        expected_prefixes = ["A) ", "B) ", "C) ", "D) "]
        for i, (opt, prefix) in enumerate(zip(v, expected_prefixes)):
            if not opt.startswith(prefix):
                raise ValueError(f"option[{i}] must start with '{prefix}', got: {opt!r}")
        return v
```

### 3.2 Tool wrappers (closure-factory pattern)

```python
# backend/app/agent/quiz_master_agent.py

def _make_quiz_tools(
    *,
    user_id: str,
    retriever,
    topic_repo: TopicRepository,
    question_repo: QuestionRepository,
    goal_repo: GoalRepository,
) -> list:
    """Per-request tool set. user_id/repos never appear in LLM-visible schema."""

    @tool
    def retriever_search(query: str, top_k: int = 5) -> str:
        """Search the user's PDF corpus for chunks relevant to a quiz topic.
        Call BEFORE drafting the question to ground in real source material.
        Returns JSON list: [{"chunk_id","content","page"}, ...]."""
        if retriever is None:
            return "[]"
        chunks = retriever.search(query, top_k=top_k) or []
        return json.dumps(chunks, ensure_ascii=False)

    @tool
    def persist_quiz_question(
        topic: str,
        prompt: str,
        options: list[str],
        answer: str,
        explanation: str,
    ) -> str:
        """Save the multiple-choice quiz question to the database.
        Call AFTER you've written:
        - a clear topic name
        - a question prompt
        - exactly 4 options, each prefixed "A) ", "B) ", "C) ", "D) "
        - the correct answer letter (A/B/C/D)
        - a 1-2 sentence explanation of why the answer is correct

        Returns JSON {"question_id","topic_id","persisted":true} on success
        or {"error": "..."} on schema validation failure (retry with valid args).
        """
        try:
            validated = QuizQuestionPersist(
                topic=topic, prompt=prompt, options=options,
                answer=answer, explanation=explanation,
            )
        except ValidationError as exc:
            err = exc.errors()[0] if exc.errors() else {"loc": [], "msg": str(exc)}
            loc = ".".join(str(x) for x in err.get("loc", []))
            return json.dumps({"error": f"invalid at {loc}: {err.get('msg', '')}"})

        active = goal_repo.list_active_for_user(user_id)
        goal = active[0] if active else goal_repo.create(
            user_id=user_id, title="Default Study Goal"
        )
        topic_row = (
            topic_repo.get_by_name(goal_id=goal.id, name=validated.topic)
            or topic_repo.create(goal_id=goal.id, name=validated.topic)
        )
        question = question_repo.create(
            topic_id=topic_row.id,
            prompt=validated.prompt,
            options_json=list(validated.options),
            answer=validated.answer,
            explanation=validated.explanation,
        )
        return json.dumps({
            "question_id": question.id,
            "topic_id": topic_row.id,
            "persisted": True,
        }, ensure_ascii=False)

    return [retriever_search, persist_quiz_question]
```

### 3.3 Agent system prompt

```python
_AGENT_SYSTEM_PROMPT = """You are a study coach quiz generator.

The user wants a multiple-choice quiz question on a topic.

Your job:
1. Read the user's message to extract the topic.
2. Call `retriever_search` with the topic to ground in the user's PDF source material.
3. Write a single multiple-choice question:
   - One clear, unambiguous question prompt
   - Exactly 4 options, each prefixed "A) ", "B) ", "C) ", "D) "
   - Distractors should be plausible (not obviously wrong)
   - One correct answer letter
   - A 1-2 sentence explanation grounded in the retrieved source chunks
4. Call `persist_quiz_question` with the question, options, answer, and explanation.
5. After persistence succeeds, write a short markdown reply to the user showing the question. Do NOT call more tools.

Today is {today}. Difficulty: medium. Ground strictly in retrieved chunks; do not invent facts."""

# cloud-adapt: cloud BYOK models can use a terser prompt (3-line bullet form)
# cloud-adapt: cloud models with stronger reasoning may not need step-by-step instruction
```

### 3.4 Budget & exit

- `max_iter = 6` — smaller than P2.2's 10 since quiz has narrower tool surface (1 search + 1 persist + summary = 3 iterations expected; budget 2× for safety)
- Exit on `not response.tool_calls`
- Budget exhausted → degrade with `"⚠️ Quiz agent exceeded reasoning budget. Try a different topic."`

---

## 4. Agent Loop Control Flow

Mirror of `planner_agent.py:build_planner_agent` shape. Key deltas vs P2.2:

### 4.1 Factory signature

```python
def build_quiz_master_agent(
    *,
    llm,
    topic_repo: TopicRepository,
    question_repo: QuestionRepository,
    goal_repo: GoalRepository,
    retriever=None,
    now_fn: Callable[[], datetime] = datetime.utcnow,
    max_iter: int = 6,
    system_prompt: str = _AGENT_SYSTEM_PROMPT,
):
    """Same shape as build_planner_agent; smaller tool surface, different repos.
    Returns async LangGraph node callable."""
```

### 4.2 Loop body — same skeleton as P2.2

Reused verbatim from `planner_agent.py:build_planner_agent`:
- `try/except llm.ainvoke` → degrade as `"llm_call_failed"`
- `if not tool_calls: trace.exit_reason = "natural_stop"; return _format_final_output(...)`
- Iterate tool_calls → `_safe_invoke_tool(...)` → append `ToolMessage`
- Loop exhausted → `_format_degrade_output(reason="budget_exhausted")`

### 4.3 quiz_action inference

```python
def _infer_quiz_action(trace: AgentTrace) -> Literal["generate", "grade"]:
    """Agent only runs on GENERATE turns (state.active_quiz_question_id absent
    at entry per dispatcher contract). Always returns 'generate'.

    Exists for symmetry with P2.2's _infer_plan_action and for memory_writer
    contract uniformity — judge_node reads quiz_action to decide rubric vs skip.
    """
    return "generate"
```

Always returns `"generate"` because agent_loop never sees GRADE turns (dispatcher guards). This keeps the contract with `judge_node` identical to the deterministic path (which sets `quiz_action="generate"` on GENERATE turns).

### 4.4 Final output (SSE-compatible, mirrors P2.2)

```python
def _format_final_output(writer, trace, last_response, question_id):
    final_text = _extract_text(last_response.content)
    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": final_text})
    return {
        "messages": [AIMessage(content=final_text)],
        "citations": [],
        "active_quiz_question_id": question_id,    # extracted from persist tool output
        "quiz_action": "generate",
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
    }
```

`active_quiz_question_id` is extracted from the most recent successful `persist_quiz_question` tool call's output JSON. If LLM never called persist successfully (budget exhausted, validation loop failure), `active_quiz_question_id=None` → next user turn will go through GENERATE path again (no broken GRADE state).

### 4.5 AgentTrace — reused via refactor

`AgentTrace`, `IterationRecord`, `ToolCallRecord` move from `planner_agent.py` to `app/agent/agent_trace.py`. Public API unchanged. P2.2 imports updated to:
```python
from app.agent.agent_trace import AgentTrace, IterationRecord, ToolCallRecord
```

Two new helper methods added for quiz consumption:
```python
def last_persisted_question_id(self) -> str | None:
    """Return the question_id from the most recent successful persist_quiz_question call."""
    for tc in reversed(self.tool_calls):
        if tc.name == "persist_quiz_question" and not tc.error:
            try:
                return json.loads(tc.output).get("question_id")
            except (json.JSONDecodeError, AttributeError):
                return None
    return None
```

`last_persisted_plan_id` (P2.2) and `last_persisted_question_id` (P2.3) are parallel methods. Could refactor to single `last_persisted_id(tool_name, key)` later — defer until 3rd consumer appears (YAGNI).

---

## 5. State / deps.py / routes.py Wiring

### 5.1 CoachState — no schema change

`agent_trace: NotRequired[dict]` already exists (P2.2). `active_quiz_question_id`, `quiz_action`, `messages`, `citations`, `last_context` all reused.

### 5.2 deps.py additions

```python
# backend/app/api/deps.py — append

def get_quiz_mode(
    x_quiz_mode: Annotated[str | None, Header()] = None,
) -> Literal["deterministic", "agent_loop"]:
    """Read x-quiz-mode header. Default = deterministic. Unknown → deterministic."""
    if x_quiz_mode == "agent_loop":
        return "agent_loop"
    return "deterministic"


def get_quiz_master_agent(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: cloud BYOK can raise max_iter from 6 to 12 here
    from app.agent.quiz_master_agent import build_quiz_master_agent
    from app.db.repositories import (
        GoalRepository, QuestionRepository, TopicRepository,
    )
    return build_quiz_master_agent(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever,
    )
```

### 5.3 routes.py — chat() signature extension

Add 2 Depends parameters:
```python
quiz_master_agent: Annotated[object, Depends(get_quiz_master_agent)],
quiz_mode: Annotated[str, Depends(get_quiz_mode)],
```

And 2 configurable keys:
```python
"quiz_master_agent": quiz_master_agent,
"quiz_mode": quiz_mode,
```

### 5.4 graph.py — quiz_node dispatcher rewrite

```python
async def quiz_node(state, config) -> dict:
    configurable = (config or {}).get("configurable", {}) or {}

    # State-aware: GRADE turn always deterministic regardless of mode
    if state.get("active_quiz_question_id"):
        quiz_master = configurable.get("quiz_master")
        if quiz_master is None:
            return quiz_stub_node(state)
        return await quiz_master(state)

    # GENERATE turn: mode-aware dispatch
    mode = configurable.get("quiz_mode", "deterministic")
    if mode == "agent_loop":
        agent = configurable.get("quiz_master_agent")
        if agent is None:
            return quiz_stub_node(state)
        return await agent(state)
    quiz_master = configurable.get("quiz_master")
    if quiz_master is None:
        return quiz_stub_node(state)
    return await quiz_master(state)
```

### 5.5 SSE contract

Same as P2.2: `citations: [] → token: <markdown quiz> → done`. Frontend zero change. Eval harness uses `await graph.ainvoke(...)` to bypass SSE.

### 5.6 Configuration contract (delta from P2.2)

| Source | Key | Type | Purpose | Status |
|---|---|---|---|---|
| HTTP header | `x-quiz-mode` | `deterministic\|agent_loop` | switch quiz mode | **NEW** |
| HTTP header | `x-planner-mode` | (P2.2) | unchanged | existing |
| HTTP header | `x-judge-model` | (P2.1-②) | unchanged | existing |
| state field | `agent_trace` | `dict?` | now written by both Planner agent and Quiz agent | reused |

---

## 6. Eval Harness Design (P2.3-② phase)

### 6.1 Matrix orchestration

```
Main matrix:
  10 single-turn GENERATE × 4 models × 2 modes × 3 runs       = 240 records
   2 multi-turn (GENERATE+GRADE)× 2 turns × 4 models × 2 modes × 3 runs = 96 records
  ─────────────────────────────────────────────────────────────────
  Main subtotal                                                 336 records

Appendix (gemma4:e4b thinking-ON, single-turn only, both modes):
  10 single × 1 model × 2 modes × 3 runs                      =  60 records

Total ≈ 396 records (matches P2.2 footprint exactly).
```

Per-cell n (single-turn only): 30 records per (model, mode). Multi-turn adds 6 GENERATE records per (model, mode) cell. Statistical power matches P2.2's ±0.05 detection threshold.

Multi-turn queries excluded from appendix to keep the thinking-on/off comparison focused on single-turn GENERATE quality (avoids confounding by GRADE-turn noise).

Multi-turn records (turn_idx=1, GRADE) captured for completeness — `exit_reason="deterministic"` in both modes confirms dispatcher routing. Only turn_idx=0 contributes to ablation analysis.

Budget: ~5h wall time on 16GB Apple Silicon Mac, ~$3 MiniMax-M2.7 cloud judge cost (+ ~$0.5 appendix).

### 6.2 Module layout — fork of p2_2

```
backend/app/eval/p2_3_quiz_ablation/
├── __init__.py
├── matrix.py             # RunSpec + expand_matrix — DELTA: mode field → quiz_mode; thinking field same
├── single_run.py         # extract from quiz state: active_quiz_question_id, quiz_action
│                         # DELTA: persisted=bool(active_quiz_question_id) — same shape as plan_id check
├── judges.py             # qwen2.5:7b local + MiniMax-M2.7 cloud
│                         # DELTA: rubric_path="judge_quiz.txt", dimensions=QUIZ_DIMENSIONS
├── run_eval.py           # CLI: --queries --output --runs --thinking-appendix
│                         # DELTA: header is x-quiz-mode, ChatOllama(reasoning=spec.thinking) reused
├── queries.json          # NEW — 10 single GENERATE + 2 multi (GENERATE→"A"→GRADE)
├── summarize.py          # auto markdown table generator — DELTA: column labels
└── output/results.jsonl  # append-only data file
```

### 6.3 Per-run record schema (delta from P2.2)

```json
{
  "run_id": "uuid",
  "timestamp": "ISO-8601",
  "model": "gemma4:e4b",
  "mode": "agent_loop",
  "query_id": "quiz_hyde",
  "turn_idx": 0,
  "run_idx": 0,
  "operational": {
    "wall_time_s": 8.42,
    "iterations": 3,
    "tool_calls": [
      {"name": "retriever_search", "count": 1},
      {"name": "persist_quiz_question", "count": 1}
    ],
    "tool_call_count": 2,
    "tool_errors": 0,
    "input_tokens": 1842,
    "output_tokens": 287,
    "exit_reason": "natural_stop"
  },
  "output": {
    "quiz_action": "generate",
    "question_persisted": 1,
    "question_id": "<uuid>",
    "final_text_excerpt": "first 500 chars"
  },
  "judge_local": {"score": 0.82, "weak_dims": ["difficulty_calibration"], "reasoning": "..."},
  "judge_cloud": {"score": 0.68, "weak_dims": [], "reasoning": "...", "model": "MiniMax-M2.7"}
}
```

Differences from P2.2 record:
- `output.quiz_action` replaces `output.plan_action`
- `output.question_persisted` (0/1) replaces `output.milestones_persisted` (count)
- `output.question_id` replaces nothing (P2.2 had `milestones_json` array; quiz is single question)
- `turn_idx`: 0 for single-turn or 1st of multi-turn GENERATE, 1 for 2nd turn (GRADE — only recorded for multi-turn queries)

GRADE-turn records (turn_idx=1) have nearly empty `agent_trace` (no LLM call, `exit_reason="deterministic"`). Captured for completeness — analysis filters `turn_idx=0` for mode comparison.

### 6.4 Queries.json structure

```json
{
  "single_turn_quiz": [
    {"id": "quiz_hyde", "message": "quiz me on HyDE"},
    {"id": "quiz_bm25", "message": "测一下 BM25"},
    {"id": "quiz_reranking", "message": "quiz me on reranking"},
    {"id": "quiz_chunking", "message": "考一下 chunking strategies"},
    {"id": "quiz_eval", "message": "quiz me on retrieval evaluation"},
    {"id": "quiz_judge", "message": "测一下 LLM-as-judge"},
    {"id": "quiz_embeddings", "message": "quiz me on embedding models"},
    {"id": "quiz_hybrid", "message": "quiz me on hybrid retrieval"},
    {"id": "quiz_hyde_zh", "message": "测我一下 HyDE"},
    {"id": "quiz_rrf", "message": "quiz me on reciprocal rank fusion"}
  ],
  "multi_turn_grade": [
    {
      "id": "quiz_hyde_then_grade",
      "messages": ["quiz me on HyDE", "A"]
    },
    {
      "id": "quiz_bm25_then_grade",
      "messages": ["quiz me on BM25", "A"]
    }
  ]
}
```

Topic overlap with P2.2 (HyDE / BM25 / reranking / chunking / eval / judge / embeddings / hybrid) is intentional — enables future cross-task analysis ("same model on Plan vs Quiz for HyDE").

### 6.5 Execution strategy — same as P2.2

- Sequential (Ollama local single-instance)
- Resumable (skip rows already in `results.jsonl` by `run_id`)
- Failure not retried (failure IS data — `gemma3:4b agent_loop` failure is expected)
- Wall time est: ~5h on 16GB Mac
- Cost: ~$3 MiniMax-M2.7 API for 396 cloud judgments

### 6.6 Metrics dimensions — same as P2.2 + quiz-specific

| Dimension | Source | Notes |
|---|---|---|
| latency | `operational.wall_time_s` | same |
| tool calling correctness | `operational.tool_calls[].name` sequence | NEW: expected pattern is `[retriever_search → persist_quiz_question]` (2 calls) |
| robustness | `operational.exit_reason` distribution | same |
| token cost | `input + output tokens` | same — agent_loop only |
| question persistence | `output.question_persisted` boolean | NEW: 0/1 per record (vs P2.2's milestone count) |
| question quality | `judge_local.score + judge_cloud.score` (avg) | same with QUIZ_DIMENSIONS |
| judge agreement | `|judge_local - judge_cloud|` | same |

---

## 7. Testing Strategy

| Layer | File | Coverage | Count |
|---|---|---|---|
| A. Tool unit | `tests/agent/test_quiz_master_agent_tools.py` | `retriever_search` + `persist_quiz_question` × schema validation, closure injection, JSON return | 4 |
| B. AgentTrace refactor | `tests/agent/test_agent_trace.py` | (existing P2.2 tests pass byte-identical) + 1 new `last_persisted_question_id` test | 1 |
| C. Loop (stub LLM) | `tests/agent/test_quiz_master_agent_loop.py` | natural_stop / budget_exhausted / llm_error / tool_error self-correct / persist with valid schema / persist with invalid schema (retry) | 6 |
| D. quiz_action inference | covered by C | always `"generate"` | 0 (asserted in loop tests) |
| E. Graph e2e (stub LLM) | `tests/agent/test_graph_quiz_agent_e2e.py` | mode switching / state-aware (GRADE always deterministic) / agent_trace lands in state | 3 |
| F. Routes integration | `tests/api/test_routes_quiz_agent.py` | x-quiz-mode header routing / SSE contract identical / multi-turn coherence | 3 |
| G. Eval harness | `tests/eval/test_p2_3_harness.py` | matrix expansion / record schema / resumable | 3 |

**Target: 181 baseline + 20 = 201 tests passing.**

P2.2 AgentTrace tests (3 existing) move to test_agent_trace.py and pass unchanged — proves the refactor is non-breaking.

---

## 8. Cut-by-Cut Plan Skeleton

writing-plans skill will detail each cut; this is the skeleton:

- **P2.3-①a** AgentTrace refactor extraction → `app/agent/agent_trace.py` + planner_agent.py import update + P2.2 test pass-through (1 new test, +0 net change to public API, ~half day)
- **P2.3-①b** Tool wrappers + `QuizQuestionPersist` schema + closure factory (A 4 tests, ~1 day)
- **P2.3-①c** Loop body + `_infer_quiz_action` + final output formatting (C 6 tests, ~2 days)
- **P2.3-①d** Graph wiring: state-aware + mode-aware dispatcher in `quiz_node` (E 3 tests, ~1 day)
- **P2.3-①e** deps.py + routes.py production wiring (F 3 tests, ~1 day)
- **P2.3-①f** Real-Ollama smoke test 4 models × 2 modes one happy-path GENERATE each + 1 multi-turn (manual, ~half day)
- **P2.3-②a** Eval harness fork from p2_2 (G 3 tests, ~1 day)
- **P2.3-②b** Run full matrix ~396 records, dump results.jsonl (~5h wall time + ~1 day analysis)
- **P2.3-③** EVAL append-section + blog "Quiz task: does persistence rescue replicate?" (~2 days)

**Total estimate: 9-11 days, ~1.5-2 weeks** (matches user budget commitment).

---

## 9. Verification Criteria

- [ ] 201 tests passing (181 baseline + 20 P2.3)
- [ ] P2.2's 3 AgentTrace tests pass byte-identical after refactor (proves non-breaking move)
- [ ] 4 models × 2 modes pass real Ollama smoke test (P2.3-①f happy-path each)
- [ ] Eval matrix ~396 runs complete, `results.jsonl` persisted with 0 harness_error
- [ ] `docs/EVAL.md` gets new section "P2.3 Quiz Ablation" with table + finding ("does schema-rescue effect replicate, weaken, or strengthen on Quiz?")
- [ ] Blog continuation `docs/quiz_ablation_followup.md` (or appended to existing blog) explicitly answers the 3 predictions from `docs/agent_loop_vs_deterministic.md`
- [ ] memory `project_study_coach_refactor.md` updated with P2.3 segment
- [ ] ROADMAP.md updated (move P2.3 from "candidate" to "shipped")
- [ ] cloud-adapt hooks grep-able in new files

---

## 10. Out of Scope

- LLM tool-calling for Quiz GRADE turn (Quiz GRADE is purely deterministic; ablating it has no LLM variable)
- Adaptive features (mastery-aware difficulty, mistake-bank avoidance) — P3 product feature, not ablation
- GPT-4o-mini cloud judge cross-validation — P2.4 separate judge-bias ablation
- Frontend Quiz adaptive view — P3 polish
- Tool surface beyond 2 (no `get_topic_mastery` / `get_recent_mistakes`) — see Q2 rationale
- ReAct-style prebuilt agents — would abstract the structural details we're testing
- Multi-agent (Quiz-master orchestrator + Question-generator subagent etc.) — P3
- Token cost data on deterministic path — same gap as P2.2, deferred to future instrumentation
- Statistical power upgrade (more queries / runs) — P2.2 noted ±0.05 limit at n=36-72; P2.3 matches that constraint

---

## 11. Cloud-Adapt Hooks (grep anchors)

```python
# cloud-adapt: cloud BYOK provider can raise max_iter from 6 to 12 here (quiz_master_agent factory)
# cloud-adapt: tool descriptions can be terser for cloud models; small Ollama models need explicit "when to use"
# cloud-adapt: cloud judge swap via x-judge-model header is already supported (P2.1-②)
# cloud-adapt: production deploy redact agent_trace.tool_calls[].output to prevent PII leak
# cloud-adapt: temperature 0.7 for local — cloud BYOK may default to 0.3 for tighter quiz quality
# cloud-adapt: think mechanism via ChatOllama(reasoning=False) — cloud providers ignore this kwarg cleanly
```

---

## 12. Predictions to Test (from `docs/agent_loop_vs_deterministic.md`)

| # | Prediction | Source quote | Likely outcome | What surprises would mean |
|---|---|---|---|---|
| **P1** | gemma3-tier collapse repeats | "I expect the gemma3-tier collapse to repeat" | TRUE (model manifest unchanged) | If gemma3 somehow succeeds on Quiz: Ollama released tools support for gemma3 mid-experiment |
| **P2** | Persistence rescue effect may NOT manifest | "the persistence rescue effect may not — quiz tools are simpler" | UNCERTAIN. Quiz schema is actually MORE strict than Plan's (Literal answer, len-4 options, prefix validation). Hypothesis: rescue effect manifests **MORE strongly**, not less. | If rescue effect weaker on Quiz: schema strictness is not the rescue mechanism; something else (maybe Milestone's optional fields gave LLM more "slop space" to recover) | 
| **P3** | "Quiz tools are simpler" framing | implied throughout blog's "more constrained tool surface" | FALSE on schema dimension (Quiz schema MORE strict), TRUE on count (2 vs 5). The blog's "simpler" needs disambiguation in P2.3 EVAL. | The very act of defining "simpler" precisely is a finding — schema-as-harness is a different axis than tool-count-as-harness |

**P2.3 EVAL writeup must address all 3** — that's how the blog's "What I would do next" gets closed out.

---

## 13. References

### Internal

- P2.2 spec: `docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md`
- P2.2 plan: `docs/superpowers/plans/2026-05-23-p2-2-agent-loop-ablation.md`
- P2.2 report: `docs/EVAL.md`
- P2.2 blog: `docs/agent_loop_vs_deterministic.md`
- P2.1-④ Quiz chain spec sections — reused QUIZ_DIMENSIONS rubric, judge_quiz.txt prompt
- ARCHITECTURE.md §3 (tool table — Quiz tools listed)
- ROADMAP.md P2.1-④ + P2.2 blocks

### External reference repo (carried from P2.2)

- `learn-claude-code` at `/Users/lianghaozhe/learn-claude-code/`
- LangChain docs: `bind_tools`, `@tool` decorator, ChatOllama
- Ollama tool calling: https://ollama.com/blog/tool-support

### Memory entries

- `project_study_coach_refactor` — will be updated with P2.3 segment
- `feedback_cloud_model_adaptation_hooks` — mark not implement (still applies)
- `skills_usage_by_phase` — subagent-driven-development applies to P2.3-①a through ①e, manual cuts for ①f / ②b / ③
