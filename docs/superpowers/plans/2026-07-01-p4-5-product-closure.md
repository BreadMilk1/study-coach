# P4.5 Product Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the Chat-first portfolio demo path so visible Chat quiz questions are persisted, gradeable, recoverable, and backed by Debug / Agent Run evidence.

**Architecture:** Keep the current FastAPI + LangGraph + Vue architecture. Make the backend own the quiz consistency invariant: agent-loop quiz generation may show an answerable MCQ only after `persist_quiz_question` succeeds; otherwise it degrades with a non-answerable message. Use the existing `assistant_artifacts.v1` envelope and TracePanel rather than adding tables or a new Agent Runs page.

**Tech Stack:** Python 3.11, FastAPI, LangGraph custom SSE events, Pydantic v2 validators, SQLAlchemy repositories, Vue 3, Pinia, TypeScript, `uv`, `pnpm`.

**Spec:** `docs/superpowers/specs/2026-07-01-p4-5-product-closure-design.md`

**Git rule:** Project rules override generic plan templates. Do not commit unless the user explicitly approves. The commit checkpoints below are review points only.

---

## File Structure

Planned modified files:

- `backend/tests/api/test_routes_quiz_agent.py`
  Add judge dependency override so route-level dispatcher tests never call live Ollama.

- `backend/app/agent/tools/schemas.py`
  Add narrow normalization for `QuizQuestionPersist.answer` and `QuizQuestionPersist.options`.

- `backend/tests/agent/test_quiz_master_agent_tools.py`
  Add regression tests for near-miss `persist_quiz_question` args.

- `backend/app/agent/quiz_master_agent.py`
  Enforce strong consistency: no persisted question id means degraded non-answerable output.

- `backend/tests/agent/test_quiz_master_agent_loop.py`
  Add regression test for final MCQ text after failed persistence.

- `backend/tests/api/test_routes_graph_stream.py`
  Add route-level Chat-first agent-loop generate then deterministic grade regression.

- `frontend/src/components/TracePanel.vue`
  Optional polish only if current display is insufficient after backend change. Existing panel already shows `exit_reason`, `tool_errors`, tool calls, and output previews.

- `docs/DEMO.md`, `docs/ROADMAP.md`, `README.md`, `docs/ARCHITECTURE.md`
  Final docs sync after implementation is verified. README / ARCHITECTURE should not claim P4.5 shipped until tests and manual demo pass.

No new files are required for implementation.

---

## Task 0: Execution Gate

**Purpose:** Prevent implementation from mixing with unreviewed docs or unrelated workspace state.

**Files:**
- Read: `AGENTS.md`
- Read: `docs/superpowers/specs/2026-07-01-p4-5-product-closure-design.md`
- No writes.

- [ ] **Step 1: Check branch and status**

Run:

```bash
git status --short --branch
git log --oneline --decorate --max-count=5
```

Expected before code work:

```text
## main...origin/main [ahead 2]
```

There may be uncommitted P4.5 planning docs. Do not proceed to code until the user confirms whether to keep those docs uncommitted, commit them, or continue with diff-based review.

- [ ] **Step 2: Confirm execution mode**

Ask the user:

```text
P4.5 implementation is ready to start.

Choose one:
1. Commit the P4.5 planning docs first, then implement code.
2. Leave planning docs uncommitted and implement in the current workspace.

Which mode?
```

- [ ] **Step 3: Baseline verification**

Run:

```bash
cd backend
uv run pytest tests/api/test_routes_quiz_agent.py -q
uv run pytest tests/agent/test_quiz_master_agent_tools.py tests/agent/test_quiz_master_agent_loop.py -q
cd ../frontend
pnpm build
```

Expected:

```text
3 passed
...
frontend build succeeds
```

If `tests/api/test_routes_quiz_agent.py` fails because Ollama is not running, continue to Task 1. That is the known test isolation gap.

---

## Task 1: Isolate Quiz Route Tests From Live Ollama

**Purpose:** Make route-level quiz dispatcher tests verify route behavior without requiring live judge model calls.

**Files:**
- Modify: `backend/tests/api/test_routes_quiz_agent.py`

- [ ] **Step 1: Write the failing-environment reproduction**

Run with Ollama stopped, or with `OLLAMA_HOST` pointed at an unused port:

```bash
cd backend
OLLAMA_HOST=http://127.0.0.1:59999 uv run pytest tests/api/test_routes_quiz_agent.py -q
```

Expected before the fix:

```text
FAILED ... httpx.ConnectError: All connection attempts failed
```

The failure path should go through `judge_node -> judge_response -> ChatOllama`.

- [ ] **Step 2: Add the dependency override**

Modify the imports in `backend/tests/api/test_routes_quiz_agent.py`:

```python
from app.api.deps import (
    get_judge_dependencies,
    get_quiz_master,
    get_quiz_master_agent,
)
```

Add the judge override inside `client_with_stubs`, after the quiz overrides:

```python
    app.dependency_overrides[get_quiz_master] = lambda: quiz_master_stub
    app.dependency_overrides[get_quiz_master_agent] = lambda: agent_stub
    app.dependency_overrides[get_judge_dependencies] = lambda: {
        "llm": None,
        "same_model": False,
    }
```

`judge_node` treats `judge_llm is None` as pass-through, so this keeps the test focused on route dispatch and checkpointer state.

- [ ] **Step 3: Verify isolated quiz route tests**

Run:

```bash
cd backend
OLLAMA_HOST=http://127.0.0.1:59999 uv run pytest tests/api/test_routes_quiz_agent.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 4: Commit checkpoint**

Do not run unless the user explicitly approves committing.

```bash
git add backend/tests/api/test_routes_quiz_agent.py
git commit -m "test: isolate quiz route tests from live judge model"
```

---

## Task 2: Normalize Narrow Quiz Tool Near-Misses

**Purpose:** Let `persist_quiz_question` accept common local-model formatting near-misses while preserving strict validation after normalization.

**Files:**
- Modify: `backend/app/agent/tools/schemas.py`
- Modify: `backend/tests/agent/test_quiz_master_agent_tools.py`

- [ ] **Step 1: Add failing tests for answer and option normalization**

Append to `backend/tests/agent/test_quiz_master_agent_tools.py`:

```python
def test_persist_quiz_question_normalizes_answer_suffix_and_missing_option_prefixes(session):
    user = UserRepository(session).get_or_create("fp-quiz-normalize-1")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    out = tool.invoke({
        "topic": "Prompt Engineering",
        "prompt": "What is prompt engineering in its simplest form?",
        "options": [
            "Crafting prompts to guide an LLM's output toward a specific outcome.",
            "Training large language models on massive datasets.",
            "Designing new transformer architectures.",
            "Selecting GPU hardware for AI computations.",
        ],
        "answer": "A)",
        "explanation": "Prompt engineering is the practice of crafting prompts to guide model output.",
    })

    parsed = json.loads(out)
    assert parsed["persisted"] is True
    q = question_repo.get_by_id(parsed["question_id"])
    assert q is not None
    assert q.answer == "A"
    assert q.options_json == [
        "A) Crafting prompts to guide an LLM's output toward a specific outcome.",
        "B) Training large language models on massive datasets.",
        "C) Designing new transformer architectures.",
        "D) Selecting GPU hardware for AI computations.",
    ]


def test_persist_quiz_question_rejects_wrong_explicit_option_label(session):
    user = UserRepository(session).get_or_create("fp-quiz-normalize-2")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    tools = _make_quiz_tools(
        user_id=user.id, retriever=None,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    tool = next(t for t in tools if t.name == "persist_quiz_question")

    out = tool.invoke({
        "topic": "RRF",
        "prompt": "Which option describes RRF?",
        "options": ["B) Wrong explicit label", "B) ok", "C) ok", "D) ok"],
        "answer": "A)",
        "explanation": "RRF combines ranked lists with reciprocal rank scores.",
    })

    parsed = json.loads(out)
    assert "error" in parsed
    assert "option[0]" in parsed["error"] or "A) " in parsed["error"]
```

Update the existing invalid-prefix test to use a wrong explicit label rather than a lower-case near-miss:

```python
        "options": ["B) wrong explicit label", "B) ok", "C) ok", "D) ok"],
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd backend
uv run pytest tests/agent/test_quiz_master_agent_tools.py -q
```

Expected before implementation:

```text
FAILED ... "error" in parsed
```

The first new test should fail because missing prefixes / `A)` answer are currently rejected.

- [ ] **Step 3: Implement normalization helpers**

Modify `backend/app/agent/tools/schemas.py`.

Add constants and helpers above `QuizQuestionPersist`:

```python
_OPTION_PREFIXES = ["A) ", "B) ", "C) ", "D) "]
_ANSWER_LETTERS = {"A", "B", "C", "D"}


def _normalize_answer_value(value) -> str:
    text = str(value or "").strip().upper()
    if text in _ANSWER_LETTERS:
        return text
    if len(text) >= 2 and text[0] in _ANSWER_LETTERS and text[1] in {")", ".", ":", " "}:
        return text[0]
    return text


def _normalize_option_value(value, *, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        return text

    expected = _OPTION_PREFIXES[index]
    expected_letter = expected[0]

    if text.startswith(expected):
        return text

    if len(text) >= 2 and text[0].upper() in _ANSWER_LETTERS and text[1] in {")", ".", ":"}:
        actual_letter = text[0].upper()
        body = text[2:].strip()
        if actual_letter != expected_letter:
            raise ValueError(
                f"option[{index}] must start with {expected!r}, got explicit {actual_letter})"
            )
        if not body:
            raise ValueError(f"option[{index}] must include text after {expected!r}")
        return f"{expected}{body}"

    return f"{expected}{text}"
```

Update `QuizQuestionPersist` validators:

```python
class QuizQuestionPersist(BaseModel):
    topic: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=1)

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, v):
        return _normalize_answer_value(v)

    @field_validator("options", mode="before")
    @classmethod
    def _normalize_options(cls, v):
        if not isinstance(v, list):
            return v
        return [_normalize_option_value(opt, index=i) for i, opt in enumerate(v)]

    @field_validator("options")
    @classmethod
    def _check_option_prefixes(cls, v: list[str]) -> list[str]:
        for i, (opt, prefix) in enumerate(zip(v, _OPTION_PREFIXES)):
            if not opt.startswith(prefix):
                raise ValueError(f"option[{i}] must start with {prefix!r}, got: {opt!r}")
            if not opt[len(prefix):].strip():
                raise ValueError(f"option[{i}] must include text after {prefix!r}")
        return v
```

- [ ] **Step 4: Verify tool tests**

Run:

```bash
cd backend
uv run pytest tests/agent/test_quiz_master_agent_tools.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit checkpoint**

Do not run unless the user explicitly approves committing.

```bash
git add backend/app/agent/tools/schemas.py backend/tests/agent/test_quiz_master_agent_tools.py
git commit -m "fix: normalize quiz tool near-miss formatting"
```

---

## Task 3: Enforce Agent-Loop Quiz Strong Consistency

**Purpose:** Prevent Chat from showing an answerable MCQ when `persist_quiz_question` failed.

**Files:**
- Modify: `backend/app/agent/quiz_master_agent.py`
- Modify: `backend/tests/agent/test_quiz_master_agent_loop.py`

- [ ] **Step 1: Add a failing regression test**

Append to `backend/tests/agent/test_quiz_master_agent_loop.py`:

```python
async def test_final_quiz_text_without_successful_persist_degrades_instead(session):
    user = UserRepository(session).get_or_create("fp-loop-no-persist")
    goal_repo = GoalRepository(session)
    topic_repo = TopicRepository(session)
    question_repo = QuestionRepository(session)

    invalid_persist_args = {
        "topic": "RRF",
        "prompt": "What is RRF?",
        "options": ["A) Only one option"],
        "answer": "A",
        "explanation": "RRF combines rankings.",
    }
    llm = ScriptedLLM([
        _ai(tool_calls=[{"name": "persist_quiz_question",
                         "args": invalid_persist_args, "id": "tc-bad"}]),
        _ai(content=(
            "Here is your quiz question:\n\n"
            "What is RRF?\n\n"
            "A) Reciprocal Rank Fusion\n"
            "B) Random Ranking Filter\n"
            "C) Recursive Retrieval Format\n"
            "D) Ranked Result File\n\n"
            "Answer: A"
        )),
    ])

    agent = build_quiz_master_agent(
        llm=llm,
        topic_repo=topic_repo, question_repo=question_repo, goal_repo=goal_repo,
    )
    result = await agent({
        "messages": [HumanMessage(content="quiz me on RRF")],
        "user_id": user.id,
    })

    content = result["messages"][0].content
    assert result["degraded"] is True
    assert result.get("active_quiz_question_id") is None
    assert result["agent_trace"]["tool_errors"] == 1
    assert result["agent_trace"]["exit_reason"] == "quiz_persist_failed"
    assert "couldn't save a gradeable quiz question" in content.lower()
    assert "A) Reciprocal Rank Fusion" not in content
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
cd backend
uv run pytest tests/agent/test_quiz_master_agent_loop.py::test_final_quiz_text_without_successful_persist_degrades_instead -q
```

Expected before implementation:

```text
FAILED ... assert result["degraded"] is True
```

Current behavior returns the model's final MCQ text even though persistence failed.

- [ ] **Step 3: Add quiz-specific degrade message and helper**

Modify `backend/app/agent/quiz_master_agent.py`.

Add constant near the other degrade messages:

```python
_QUIZ_PERSIST_FAILED_MSG = (
    "⚠️ I couldn't save a gradeable quiz question. Please try again, "
    "or switch Quiz mode to deterministic for a faster fallback."
)
```

Add helper after `_emit_agent_run`:

```python
def _format_unpersisted_quiz_output(writer, trace: AgentTrace) -> dict:
    trace.exit_reason = "quiz_persist_failed"
    text = _QUIZ_PERSIST_FAILED_MSG
    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": text})
    _emit_agent_run(writer, trace)
    return {
        "messages": [AIMessage(content=text)],
        "citations": [],
        "active_quiz_question_id": None,
        "quiz_action": _infer_quiz_action(trace),
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
        "degraded": True,
    }
```

- [ ] **Step 4: Gate final output on persistence**

Update `_format_final_output`:

```python
def _format_final_output(writer, trace: AgentTrace, last_response) -> dict:
    persisted_question_id = trace.last_persisted_question_id()
    if persisted_question_id is None:
        return _format_unpersisted_quiz_output(writer, trace)

    final_text = getattr(last_response, "content", "") or ""
    if not isinstance(final_text, str):
        final_text = "".join(
            (b.get("text", "") if isinstance(b, dict) else str(b))
            for b in final_text
        )

    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": final_text})
    _emit_agent_run(writer, trace)

    return {
        "messages": [AIMessage(content=final_text)],
        "citations": [],
        "active_quiz_question_id": persisted_question_id,
        "quiz_action": _infer_quiz_action(trace),
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),
    }
```

- [ ] **Step 5: Verify loop tests**

Run:

```bash
cd backend
uv run pytest tests/agent/test_quiz_master_agent_loop.py -q
```

Expected:

```text
7 passed
```

- [ ] **Step 6: Commit checkpoint**

Do not run unless the user explicitly approves committing.

```bash
git add backend/app/agent/quiz_master_agent.py backend/tests/agent/test_quiz_master_agent_loop.py
git commit -m "fix: require persisted quiz before showing agent MCQ"
```

---

## Task 4: Add Route-Level Chat Generate Then Grade Regression

**Purpose:** Prove the Chat-first demo path works through `/api/chat` in `agent_loop` mode: generate a persisted MCQ, then grade an `A/B/C/D` reply in the same session.

**Files:**
- Modify: `backend/tests/api/test_routes_graph_stream.py`

- [ ] **Step 1: Add route-level scripted quiz LLM test**

Append this test near the existing chat persistence / route tests in `backend/tests/api/test_routes_graph_stream.py`:

```python
def test_chat_quiz_agent_loop_generates_persisted_question_then_grades(client, app):
    from app.api.deps import get_llm

    class ScriptedQuizAgentLLM:
        def __init__(self):
            self.responses = [
                AIMessage(content="", tool_calls=[{
                    "name": "retriever_search",
                    "args": {"query": "Prompt Engineering"},
                    "id": "q1",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "persist_quiz_question",
                    "args": {
                        "topic": "Prompt Engineering",
                        "prompt": "What is prompt engineering in its simplest form?",
                        "options": [
                            "Crafting prompts to guide an LLM's output toward a specific outcome.",
                            "Training large language models on massive datasets.",
                            "Designing new transformer architectures.",
                            "Selecting GPU hardware for AI computations.",
                        ],
                        "answer": "A)",
                        "explanation": "Prompt engineering is the practice of crafting prompts to guide model output.",
                    },
                    "id": "q2",
                }]),
                AIMessage(content=(
                    "Here is your quiz question on Prompt Engineering.\n\n"
                    "What is prompt engineering in its simplest form?\n\n"
                    "A) Crafting prompts to guide an LLM's output toward a specific outcome.\n"
                    "B) Training large language models on massive datasets.\n"
                    "C) Designing new transformer architectures.\n"
                    "D) Selecting GPU hardware for AI computations.\n\n"
                    "Reply with A, B, C, or D."
                )),
            ]

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages, **_kwargs):
            if not self.responses:
                raise AssertionError("ScriptedQuizAgentLLM exhausted")
            return self.responses.pop(0)

        async def astream(self, _messages, **_kwargs):
            if False:
                yield None

    app.dependency_overrides[get_llm] = lambda: ScriptedQuizAgentLLM()
    session_id = "p4-5-chat-quiz-grade"

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "quiz me on Prompt Engineering", "session_id": session_id},
        headers={**_HEADERS, "x-quiz-mode": "agent_loop"},
    ) as resp:
        assert resp.status_code == 200
        turn1 = _read_sse_events(resp)

    joined1 = "".join(e["text"] for e in turn1 if e["type"] == "token")
    assert "What is prompt engineering" in joined1
    agent_run = next(e for e in turn1 if e["type"] == "agent_run")
    assert agent_run["run"]["node"] == "quiz"
    assert agent_run["run"]["exit_reason"] == "natural_stop"
    assert agent_run["run"]["tool_errors"] == 0

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "A", "session_id": session_id},
        headers={**_HEADERS, "x-quiz-mode": "agent_loop"},
    ) as resp:
        assert resp.status_code == 200
        turn2 = _read_sse_events(resp)

    joined2 = "".join(e["text"] for e in turn2 if e["type"] == "token")
    assert "Correct" in joined2

    messages_resp = client.get(
        f"/api/chat/sessions/{session_id}/messages",
        headers=_HEADERS,
    )
    assert messages_resp.status_code == 200
    messages = messages_resp.json()["messages"]
    quiz_assistant = next(
        m for m in messages
        if m["role"] == "assistant" and "What is prompt engineering" in m["content"]
    )
    assert quiz_assistant["agent_run"]["node"] == "quiz"
    assert quiz_assistant["agent_run"]["tool_call_breakdown"]["persist_quiz_question"] == 1
```

- [ ] **Step 2: Run route test and verify failure before Task 2 / 3**

If Task 2 and 3 are not implemented yet, this test should fail because `answer: "A)"` or missing option prefixes are rejected, then final quiz text is degraded.

Run:

```bash
cd backend
uv run pytest tests/api/test_routes_graph_stream.py::test_chat_quiz_agent_loop_generates_persisted_question_then_grades -q
```

Expected after Tasks 2 and 3 are complete:

```text
1 passed
```

- [ ] **Step 3: Run related route tests**

Run:

```bash
cd backend
uv run pytest tests/api/test_routes_graph_stream.py tests/api/test_routes_quiz_agent.py -q
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 4: Commit checkpoint**

Do not run unless the user explicitly approves committing.

```bash
git add backend/tests/api/test_routes_graph_stream.py
git commit -m "test: cover chat quiz agent generate and grade"
```

---

## Task 5: Verify Debug Evidence Surface

**Purpose:** Confirm the existing Debug panel already satisfies P4.5 evidence requirements, and make only minimal UI changes if verification shows a gap.

**Files:**
- Read: `frontend/src/components/TracePanel.vue`
- Read: `frontend/src/stores/chat.ts`
- Optional modify: `frontend/src/components/TracePanel.vue`

- [ ] **Step 1: Inspect current display contract**

Confirm `TracePanel.vue` renders:

```vue
<div>exit={{ latestAgentRun.exit_reason }}</div>
<div>errors={{ latestAgentRun.tool_errors }}</div>
<span :class="tc.error ? 'text-warning' : 'text-primary'">{{ tc.name }}</span>
<span class="text-fg-dim"> out={{ tc.output_preview }}</span>
```

If these lines are present, no frontend code is required for the core P4.5 Debug evidence.

- [ ] **Step 2: Optional polish only if needed**

If degraded runs are hard to spot during manual demo, add one warning line under the Agent Run grid:

```vue
<div v-if="latestAgentRun.tool_errors || latestAgentRun.exit_reason !== 'natural_stop'"
     class="mt-2 text-warning text-xs">
  run requires attention
</div>
```

Do not add a new page, filter UI, or trace inspector.

- [ ] **Step 3: Verify frontend build**

Run:

```bash
cd frontend
pnpm build
```

Expected:

```text
✓ built
```

The existing Vite large chunk warning is acceptable.

- [ ] **Step 4: Commit checkpoint**

Only if Step 2 changed frontend code and the user explicitly approves committing:

```bash
git add frontend/src/components/TracePanel.vue
git commit -m "fix: highlight agent run failures in debug panel"
```

If Step 2 was unnecessary, skip this checkpoint.

---

## Task 6: Final Docs Sync and Verification

**Purpose:** Update public docs only after implementation has actually passed automated and manual checks.

**Files:**
- Modify: `docs/DEMO.md`
- Modify: `docs/ROADMAP.md`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Update `docs/DEMO.md` after manual verification**

In `docs/DEMO.md`, change the opening note from target language to verified language:

```markdown
> P4.5 verified guide. This is the reviewer-facing demo route for Study Coach.
```

Add a short "Verified Commands" block with the actual commands that passed:

````markdown
## Verified Commands

```bash
cd backend
uv run pytest -q

cd ../frontend
pnpm build
```
````

- [ ] **Step 2: Mark ROADMAP P4.5 complete**

In `docs/ROADMAP.md`, update the P4.5 checklist from planned to done only for completed items:

```markdown
### P4.5 — Product Closure for Portfolio Demo (Done, 2026-07-01)
- [x] Scope locked in `docs/superpowers/specs/2026-07-01-p4-5-product-closure-design.md`
- [x] Reviewer demo guide verified in `docs/DEMO.md`
- [x] Quiz route test isolation: backend suite should not require live Ollama
- [x] Chat quiz strong consistency: visible MCQ must be persisted and gradeable
- [x] `persist_quiz_question` tolerates narrow format near-misses (`A)` answers, missing option prefixes) before strict validation
- [x] Failed quiz persistence degrades without showing an answerable MCQ
- [x] Debug Mode shows recoverable, redacted evidence of persist success/failure after refresh
- [x] Final verification: backend tests, frontend build, Chat-first manual demo path
```

Update Current Snapshot only after full verification:

```markdown
- **Recent (2026-07-01)**: P4.5 Product Closure shipped — Chat-first demo path stabilized, Chat quiz MCQs are persisted before display, failed quiz persistence degrades safely, and backend tests no longer depend on live Ollama.
```

- [ ] **Step 3: Update README portfolio path**

In `README.md`, add `docs/DEMO.md` to the portfolio review path or demo section:

```markdown
For a stable reviewer walkthrough, follow `docs/DEMO.md`.
```

Do not claim full harness / durable memory / multi-agent orchestration.

- [ ] **Step 4: Update ARCHITECTURE failure boundary**

In `docs/ARCHITECTURE.md`, add one sentence near the Quiz / Tool Registry / Streaming sections:

```markdown
Quiz strong consistency: Chat displays an answerable MCQ only after the backend has persisted the question and set `active_quiz_question_id`; failed agent-loop persistence degrades to a non-answerable message while preserving Agent Run evidence.
```

- [ ] **Step 5: Run final verification**

Run:

```bash
cd backend
uv run pytest -q

cd ../frontend
pnpm build

cd ..
git diff --check
```

Expected:

```text
252+ backend tests passed
✓ built
no diff --check output
```

- [ ] **Step 6: Manual Chat-first demo**

With backend and frontend running:

1. Upload a user-owned PDF in Library.
2. Enable Debug Mode.
3. Ask `What is prompt engineering?` in Chat.
4. Ask `quiz me on Prompt Engineering.` in Chat with `x-quiz-mode: agent_loop`.
5. Answer `A`.
6. Refresh browser.
7. Confirm:
   - chat history restored,
   - citations restored,
   - quiz assistant message has Agent Run evidence,
   - tool output preview is redacted,
   - no unpersisted MCQ can be answered.

- [ ] **Step 7: Commit checkpoint**

Do not run unless the user explicitly approves committing.

```bash
git add README.md docs/ARCHITECTURE.md docs/DEMO.md docs/ROADMAP.md
git commit -m "docs: mark p4.5 product closure verified"
```

---

## Spec Coverage Check

- Test isolation hardening: Task 1.
- Strong consistency quiz closure: Task 3 and Task 4.
- Tool-layer tolerance: Task 2.
- Debug evidence: Task 5 and Task 6 manual demo.
- Demo guide and roadmap sync: Task 6.
- Non-goals preserved: no new tables, no Agent Runs page, no durable memory redesign, no full Chroma filtering, no multi-agent orchestration.

## Execution Options

Plan complete. Two execution options:

1. **Subagent-Driven (recommended when available)** - one focused subagent per task, with main-agent review between tasks. Current desktop tool list does not expose a dedicated Task/subagent tool, so this may require a local `codex exec` substitute or inline fallback.
2. **Inline Execution** - execute tasks in this session using the plan above, with checkpoints after each task.

Do not start implementation until the user chooses an execution mode.
