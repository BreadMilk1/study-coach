# P2.2 Agent Loop Ablation — Design Spec

> Brainstormed 2026-05-22 via `superpowers:brainstorming`.
> **Goal**: build an LLM tool-calling agent-loop variant of Planner, run head-to-head against Cut ⑤ deterministic baseline, produce empirical data on whether agent loops are viable on local Ollama small models.
> **Upstream**: P2.1-⑤ (deterministic Planner shipped, 157 tests passing).
> **Portfolio narrative**: respond to `Learn Claude Code` README's "true agent = while(tool_use)" thesis with real data — local-Ollama tier vs reasoning-model tier × deterministic vs agent_loop. Neither HKBU class report nor learn-claude-code has this comparison.

---

## 1. Decisions Locked (Brainstorming Q&A)

### Q1a — Model selection

**Locked: 4-model 2×2 matrix**

|  | **~4B 小档** | **~7-8B 中档** |
|---|---|---|
| **旧式 instruction-tuned** | `gemma3:4b` (4B, no tools/thinking flag) | `qwen2.5:7b` (7B, tools ✓, no thinking) |
| **新式 reasoning + tools** | `qwen3.5:4b` (4.7B, tools ✓ + thinking ✓) | `gemma4:e4b` (8B, tools ✓ + thinking ✓ + multimodal) |

Rationale: 横轴控制参数量、纵轴控制"是否原生为 agent 而训"。同列对比 = "新一代有提升吗"，同行对比 = "参数量给多少帮助"，对角对比 = "4B 新一代 vs 7B 旧一代谁赢" — 这才是 portfolio 最有信息量的数据点。

**Excluded with note**: `deepseek-r1:8b` 和 `deepseek-r1-0528-qwen-8b` —— Ollama manifest 显示 `Capabilities: completion` only，没有 tools flag。即使 LangChain `bind_tools()` 能塞，模型也未被训练按 tool API 格式 emit。EVAL.md 中作为 "negative finding" 提一句。

### Q1b — Thinking on/off

**Locked: Main experiment thinking-OFF; appendix runs gemma4:e4b thinking-ON 一轮对比**

- 主 matrix (288 runs) 全部 `think=False`（或对应模型的禁用方式），公平 base
- 附录 (72 runs, gemma4:e4b only thinking on/off): 看 reasoning + token 翻倍是否换来 correctness 上升
- 共 360 runs

具体禁用 thinking 的参数名（`extra_body={"think": False}` vs `model_kwargs={"think": False}` vs system prompt `/no_think` 前缀 vs 后处理剥 `<think>` 标签）在 Cut P2.2-①f smoke test 时验证；spec 不预先死磕，作为实现细节。

### Q2 — Tool set scope

**Locked: 5 核心工具，无 done() / 无 todo()**

| Tool | LLM-visible args | Closure-injected context | Returns (JSON string) |
|---|---|---|---|
| `retriever_search` | `query: str, top_k: int = 5` | `retriever` | `[{"chunk_id","content","page"}, ...]` |
| `get_existing_plan` | (none) | `user_id, plan_repo, goal_repo` | `{"plan_id","milestones","updated_at"}` or `"null"` |
| `update_study_plan` | `milestones: list[Milestone]` | `user_id, goal_repo, plan_repo` | `{"plan_id","milestones_count","updated_at"}` |
| `generate_mindmap` | `topic: str, milestones: list[Milestone]` | `llm` | `{"mermaid_src","markdown_outline"}` |
| `compute_progress` | (none) | `user_id, plan_repo, goal_repo, mastery_scores, recent_mistakes, now_fn` | `{"done_count","total_count","overdue","weak_topics","recent_mistake_count"}` |

Rationale: minimal Anthropic-style set. Exit = `stop_reason != "tool_use"` (no `done()` sentinel). No internal todo (model decides tool order in messages, that's the experiment subject). `done()` would be GOFAI thinking — "model needs a button". Budget = `max_iter=10` as safety net.

`user_id` 永远不进 LLM args schema —— identity 不是 behavior input。

### Q3 — Plan quality scoring

**Locked: Dual judge**

- **Local judge**: `qwen2.5:7b` 复用 `PLAN_DIMENSIONS` 5-维 rubric (from P2.1-⑤)
- **Cloud judge**: BYOK GPT-4o-mini (or Claude Haiku) via existing `x-judge-model` header pattern

两个 judge 同时跑，分数一致 = 高信价；不一致 = 在 EVAL.md 中讨论分歧 (例如 self-preference bias on qwen-judge-rating-qwen2.5:7b cells).

Cost: 360 runs × ~2k tokens × GPT-4o-mini pricing ≈ $5-10。可接受。

### Q4 — Feature flag entry point

**Locked: HTTP header `x-planner-mode: deterministic|agent_loop`**

与 `x-judge-model` 同协议。`deps.py` 加 `get_planner_mode` factory 读 header。Default = `deterministic`（current production 行为不变）。Unknown values → fallback deterministic（defensive）。

Eval script 可在不重启服务的情况下逐请求切换 mode。

### Q5 — Spec detail level

**Locked: Phase-differentiated**

- **P2.2-①（实现）**: P2.1-⑤ 同详细度 — 完整代码块 + cut-by-cut TDD steps
- **P2.2-②（eval）**: harness 架构 + record schema + 参数变量表，跑脚本作为细节实现
- **P2.2-③（writeup）**: 仅核心问题 + narrative 走向，不写文章本体

### Approach choice

**Approach A**: 手写 while-loop + 嵌入 LangGraph 节点（new `planner_agent.py` 与 `planner.py` 并存）。

Rejected: Approach B (`langgraph.prebuilt.create_react_agent`) — 抽象掉实验关心的细节; Approach C (绕开 LangGraph 完全独立) — 失去公平对照基础设施。

---

## 2. Architecture & Graph Topology

```
START
  → memory_hydrator                                  （不变）
  → router (state-aware)                             （不变）
  → {tutor → judge | quiz → judge | plan_node → judge}
                                  ↓
                          plan_node = mode-aware dispatcher
                                  │
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
        planner_deterministic           planner_agent_loop
        (Cut ⑤e)                         (new in P2.2-①)
  → memory_writer → END                            （不变）
```

**关键变化点（仅 plan_node 内部 + deps.py + routes.py）**:

1. `plan_node` 升级为 mode-aware dispatcher（约 10 行）— 根据 `config.configurable.planner_mode` 选择 deterministic 或 agent_loop
2. 新模块 `app/agent/planner_agent.py`（~300 行）—— 含工具 wrappers + while-loop + 工厂
3. `deps.py` 加 `get_planner_agent` 和 `get_planner_mode`
4. `routes.py` 多读 `x-planner-mode` header + 注入 `planner_agent` 和 `planner_mode` 到 config

**LangGraph 拓扑外观不变** —— memory_hydrator/judge/memory_writer 全复用。Judge 看到 `plan_action: "generate"|"check_in"` 跟 deterministic 完全一致 → multi-rubric routing + check-in 短路 自动工作。

### 2.1 为什么 plan_node 内部 dispatch 而不是两个独立 node

- 公平性：两种模式共享 memory hydrator / judge / writer 周边基础设施
- 维护：未来切换默认 mode 只改一行
- LangGraph 拓扑稳定，已有 14 graph tests 不动

---

## 3. Tool Wrappers & LangChain ↔ Ollama Compatibility

### 3.1 Closure-factory pattern

```python
# backend/app/agent/planner_agent.py
from langchain_core.tools import tool

def _make_planner_tools(
    *,
    user_id: str,
    llm,
    retriever,
    plan_repo: PlanRepository,
    goal_repo: GoalRepository,
    mastery_scores: dict[str, float],
    recent_mistakes: list[str],
    now_fn: Callable[[], datetime] = datetime.utcnow,
) -> list:
    """Build a per-request tool set with the user/session context baked in.
    The agent loop receives this list via `llm.bind_tools(tools)`; the model
    sees only the public args, never the closure variables."""

    @tool
    def retriever_search(query: str, top_k: int = 5) -> str:
        """Search the user's uploaded PDF corpus for chunks relevant to a topic.
        Use this BEFORE drafting a study plan to ground milestones in real source material.
        Returns JSON list of chunks: [{"chunk_id","content","page"}, ...]."""
        chunks = retriever.search(query, top_k=top_k) if retriever else []
        return json.dumps(chunks, ensure_ascii=False)

    @tool
    def get_existing_plan() -> str:
        """Return the user's currently active study plan, if any.
        Use this on CHECK-IN turns to see what plan exists before adjusting it.
        Returns JSON: {"plan_id","milestones","updated_at"} or "null"."""
        active = goal_repo.list_active_for_user(user_id)
        if not active:
            return "null"
        plan = plan_repo.get_by_goal(active[0].id)
        if not plan:
            return "null"
        return json.dumps({
            "plan_id": plan.id,
            "milestones": plan.milestones_json,
            "updated_at": plan.updated_at.isoformat(),
        }, ensure_ascii=False)

    @tool
    def update_study_plan(milestones: list) -> str:
        """Persist a list of milestones as the user's study plan (upsert).
        Each milestone: {title:str, due_at:str|null, done:bool, topic:str|null}.
        Use this AFTER you've decided on the final milestone list. Returns confirmation JSON."""
        active = goal_repo.list_active_for_user(user_id)
        goal = active[0] if active else goal_repo.create(
            user_id=user_id, title=_DEFAULT_GOAL_TITLE,
        )
        validated = [Milestone.model_validate(m) for m in milestones]
        out = update_study_plan_fn(
            goal_id=goal.id, milestones=validated, plan_repo=plan_repo,
        )
        return json.dumps({
            "plan_id": out.plan_id,
            "milestones_count": len(validated),
            "updated_at": out.updated_at.isoformat(),
        }, ensure_ascii=False)

    @tool
    async def generate_mindmap(topic: str, milestones: list) -> str:
        """Generate a mermaid mindmap + markdown outline for a study plan.
        Call this ONLY if the user explicitly asks for a mindmap/脑图/思维导图.
        Returns JSON: {"mermaid_src","markdown_outline"}."""
        validated = [Milestone.model_validate(m) for m in milestones]
        out = await generate_mindmap_fn(topic=topic, milestones=validated, llm=llm)
        return json.dumps({
            "mermaid_src": out.mermaid_src,
            "markdown_outline": out.markdown_outline,
        }, ensure_ascii=False)

    @tool
    def compute_progress() -> str:
        """Compute deterministic progress summary for the user's active plan.
        Use this on CHECK-IN turns to see what's done/overdue before adjusting.
        Returns JSON: {"done_count","total_count","overdue","weak_topics","recent_mistake_count"}."""
        active = goal_repo.list_active_for_user(user_id)
        if not active:
            return json.dumps({"error": "No active goal"})
        plan = plan_repo.get_by_goal(active[0].id)
        if not plan:
            return json.dumps({"error": "No active plan"})
        progress = compute_progress_fn(plan, mastery_scores, recent_mistakes, now=now_fn())
        return json.dumps({
            "done_count": progress.done_count,
            "total_count": progress.total_count,
            "overdue": [m.get("title") for m in progress.overdue],
            "weak_topics": progress.weak_topics,
            "recent_mistake_count": progress.recent_mistake_count,
        }, ensure_ascii=False)

    return [retriever_search, get_existing_plan, update_study_plan,
            generate_mindmap, compute_progress]
```

### 3.2 LangChain ↔ Ollama binding

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=model_name,
    base_url=ollama_base_url,
    temperature=0.7,
    # cloud-adapt: thinking models (qwen3.5:4b / gemma4:e4b) take an extra param
    # via model_kwargs or extra_body. Exact key verified in Cut P2.2-①f smoke test.
)
llm_with_tools = llm.bind_tools(tools)
response = await llm_with_tools.ainvoke(messages)
# response.tool_calls = [{"name": "...", "args": {...}, "id": "..."}]
```

### 3.3 Agent system prompt

```python
_AGENT_SYSTEM_PROMPT = """You are a study coach planner agent.

User wants either a new study plan or a check-in on an existing plan.

Your job:
1. Read the user's message to understand what topic they care about.
2. If you don't know what plan (if any) they have, call `get_existing_plan` first.
3. If they want a NEW plan or explicit re-plan keywords (帮我做 / make a plan / 重做):
   - Call `retriever_search` with the topic to ground in their materials.
   - Call `update_study_plan` with 3-7 specific, dated milestones.
4. If they want a CHECK-IN (existing plan + 进度 / check in / 调整 / etc):
   - Call `compute_progress` to see what's done/overdue.
   - Call `update_study_plan` with the adjusted milestone list.
5. If they mention mindmap / 脑图 / mind map / 思维导图: call `generate_mindmap`.
6. When you're done, write a short markdown summary for the user with the milestones (and mindmap if generated). Do NOT call more tools.

Today is {today}. Be concise. Call tools to act, prose to summarize."""
```

### 3.4 Budget & exit

- `max_iter = 10` —— learn-claude-code s04 subagent uses 30; we halve since planner is single-purpose
- Exit on `not response.tool_calls`
- Budget exhausted → degrade with disclaimer, no plan persistence

---

## 4. Agent Loop Control Flow + Instrumentation

### 4.1 Main loop skeleton

```python
async def planner_agent_loop(state: CoachState, *, llm, tools, system_prompt,
                              max_iter: int = 10) -> dict:
    writer = _safe_writer()
    user_msg = _last_human_msg(state)
    today = datetime.utcnow().date().isoformat()

    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt.format(today=today)),
        HumanMessage(content=user_msg),
    ]
    trace = AgentTrace(t_start=time.monotonic())

    for iteration in range(max_iter):
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as exc:
            trace.record_llm_error(str(exc))
            return _degrade_output(writer, state, trace, reason="llm_call_failed")

        messages.append(response)
        trace.record_iteration(response, iteration)

        if not response.tool_calls:
            return _format_final_output(writer, state, trace, response)

        for tc in response.tool_calls:
            output = await _safe_invoke_tool(tool_map, tc, trace)
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

    trace.record_budget_exhaustion(max_iter)
    return _degrade_output(writer, state, trace, reason="budget_exhausted")
```

### 4.2 Exit conditions

| Trigger | Handling | User-visible | trace.exit_reason |
|---|---|---|---|
| `not response.tool_calls` | format final output | summary text (one-shot emit) | `"natural_stop"` |
| `iteration == max_iter` | degrade w/ disclaimer | `"⚠️ Agent exceeded reasoning budget (10 turns). The last partial plan was not persisted."` | `"budget_exhausted"` |
| LLM ainvoke 抛异常 | degrade | `"⚠️ Could not reach the planner model. Please try again."` | `"llm_call_failed"` |
| Tool 抛异常 | **不 degrade**，错误塞回 ToolMessage 让 model 自纠 | (loop 继续) | recorded per-tool in `trace.tool_calls[i].error` |

### 4.3 Tool error self-correction

```python
async def _safe_invoke_tool(tool_map, tc, trace):
    name, args = tc["name"], tc["args"]
    handler = tool_map.get(name)
    if handler is None:
        msg = f"Error: unknown tool '{name}'. Available: {sorted(tool_map.keys())}"
        trace.record_tool_call(name, args, msg, error=True)
        return msg
    try:
        output = await handler.ainvoke(args)
    except (ValidationError, ValueError, TypeError) as exc:
        output = f"Error calling {name}: {exc}. Please check arg types and retry."
        trace.record_tool_call(name, args, output, error=True)
        return output
    except Exception as exc:
        output = f"Error: {exc}"
        trace.record_tool_call(name, args, output, error=True)
        return output
    trace.record_tool_call(name, args, output, error=False)
    return output
```

Tool errors are recoverable (model can retry); LLM/network errors are not (degrade).

### 4.4 plan_action inference

```python
def _infer_plan_action(trace: AgentTrace) -> Literal["generate", "check_in"]:
    """Agent doesn't use a done() sentinel by design. Infer from trace."""
    called = trace.tool_names_called()
    if "get_existing_plan" in called and trace.get_existing_plan_returned_nonnull():
        return "check_in"
    return "generate"
```

`plan_action` is graph-internal routing signal for judge_node. Inferring from trace is deterministic post-processing — doesn't break experimental fairness.

### 4.5 Final output (SSE-compatible)

```python
def _format_final_output(writer, state, trace, last_response):
    plan_action = _infer_plan_action(trace)
    plan_id = trace.last_persisted_plan_id()
    last_text = _extract_text(last_response.content)

    # Same SSE contract as deterministic: citations → token (one-shot) → done
    writer({"type": "citations", "citations": []})
    writer({"type": "token", "text": last_text})

    return {
        "messages": [AIMessage(content=last_text)],   # only final summary added
        "citations": [],
        "active_plan_id": plan_id,
        "plan_action": plan_action,
        "last_context": trace.aggregated_retriever_context(),
        "agent_trace": trace.serialize(),              # NEW state field for eval
    }
```

**Key design choice**: Only final summary AIMessage joins `state["messages"]`. Intermediate reasoning / tool calls stay in agent's transient loop scope. Future turns see clean user-visible conversation history.

### 4.6 AgentTrace dataclass

```python
@dataclass
class AgentTrace:
    t_start: float
    iterations: list[IterationRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    exit_reason: str = "in_flight"
    llm_error: str | None = None

    def record_iteration(self, response, idx: int): ...
    def record_tool_call(self, name, args, output, error): ...
    def record_llm_error(self, exc: str): ...
    def record_budget_exhaustion(self, max_iter: int): ...

    def serialize(self) -> dict:
        return {
            "total_iterations": len(self.iterations),
            "total_tool_calls": len(self.tool_calls),
            "tool_call_breakdown": Counter(tc.name for tc in self.tool_calls),
            "tool_errors": sum(1 for tc in self.tool_calls if tc.error),
            "input_tokens": sum(it.input_tokens for it in self.iterations),
            "output_tokens": sum(it.output_tokens for it in self.iterations),
            "wall_time_s": time.monotonic() - self.t_start,
            "exit_reason": self.exit_reason,
            "llm_error": self.llm_error,
        }
```

This is the **命脉** for P2.2-② eval — every per-run record dumps `serialize()` into results.jsonl.

---

## 5. State / deps.py / routes.py Wiring + SSE Contract

### 5.1 CoachState additions

```python
# backend/app/agent/state.py — append to CoachState
class CoachState(TypedDict):
    # ... existing fields unchanged ...

    # P2.2-① — agent loop instrumentation. Populated only when
    # planner_mode=="agent_loop"; deterministic path doesn't write this.
    agent_trace: NotRequired[dict]
```

仅一个新字段。`active_plan_id` / `plan_action` / `mastery_scores` / `last_context` 全部复用。两种模式 schema 一致 → judge/memory_writer 不需任何 if-else。

### 5.2 deps.py additions

```python
# backend/app/api/deps.py — append

def get_planner_mode(
    x_planner_mode: Annotated[str | None, Header()] = None,
) -> Literal["deterministic", "agent_loop"]:
    """Read x-planner-mode header. Default = deterministic. Unknown → deterministic."""
    if x_planner_mode == "agent_loop":
        return "agent_loop"
    return "deterministic"


def get_planner_agent(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: when provider=cloud, max_iter can be raised to 20-30 here.
    from app.agent.planner_agent import build_planner_agent
    from app.db.repositories import (
        GoalRepository, MasteryRepository, MistakeRepository, PlanRepository,
    )
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
    )
```

### 5.3 routes.py — chat signature extension

```python
from .deps import (
    # ... existing imports ...
    get_planner_agent,
    get_planner_mode,
)

@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_user_id)],
    graph: Annotated[object, Depends(get_graph)],
    judge: Annotated[dict, Depends(get_judge_dependencies)],
    quiz_master: Annotated[object, Depends(get_quiz_master)],
    planner: Annotated[object, Depends(get_planner)],
    planner_agent: Annotated[object, Depends(get_planner_agent)],         # NEW
    planner_mode: Annotated[str, Depends(get_planner_mode)],              # NEW
    memory_hydrator: Annotated[object, Depends(get_memory_hydrator)],
    memory_writer: Annotated[object, Depends(get_memory_writer)],
):
    thread_id = body.session_id or user_id

    async def event_stream():
        input_state = {
            "messages": [HumanMessage(content=body.message)],
            "user_id": user_id,
        }
        config = {
            "configurable": {
                "thread_id": thread_id,
                "judge_llm": judge["llm"],
                "quiz_master": quiz_master,
                "planner": planner,
                "planner_agent": planner_agent,                           # NEW
                "planner_mode": planner_mode,                             # NEW
                "memory_hydrator": memory_hydrator,
                "memory_writer": memory_writer,
            }
        }
        # ... rest of stream logic unchanged ...
```

### 5.4 graph.py — plan_node mode-aware dispatch

```python
# backend/app/agent/graph.py — replace existing plan_node closure

async def plan_node(state: CoachState, config) -> dict:
    configurable = (config or {}).get("configurable", {}) or {}
    mode = configurable.get("planner_mode", "deterministic")
    if mode == "agent_loop":
        agent = configurable.get("planner_agent")
        if agent is None:
            return plan_stub_node(state)
        return await agent(state)
    planner = configurable.get("planner")
    if planner is None:
        return plan_stub_node(state)
    return await planner(state)
```

3 lines of mode-branching, everything else unchanged. This is the **only graph-layer change** in P2.2.

### 5.5 SSE contract

```
data: {"type": "citations", "citations": []}\n\n
data: {"type": "token", "text": "<markdown plan + optional mermaid>"}\n\n
data: {"type": "done"}\n\n
```

**Two modes produce identical SSE shape.** Frontend zero-change. Eval harness bypasses SSE (uses `await graph.ainvoke(...)` to get full state including `agent_trace`).

### 5.6 Configuration contract (full map)

| Source | Key | Type | Purpose |
|---|---|---|---|
| HTTP header | `x-planner-mode` | `deterministic\|agent_loop` | switch mode (NEW) |
| HTTP header | `x-judge-model` | `model_name?` | switch judge model (existing, P2.1-②) |
| HTTP header | `x-model` / `x-provider` / `x-api-key` / `x-base-url` | BYOK | switch planner model itself (existing) |
| HTTP body | `session_id` | `str?` | thread_id reuse (existing) |
| state field | `agent_trace` | `dict?` | eval-only instrumentation output (NEW) |

---

## 6. Eval Harness Design (P2.2-② phase)

### 6.1 Matrix orchestration

```
4 models × 2 modes × 12 queries × 3 runs = 288 main experiment runs
+ 1 model (gemma4) × 2 thinking states × 12 queries × 3 = 72 appendix runs
= 360 runs total
```

`query_set` reuses `tests/fixtures/retrieval_eval_queries.json` (12 HKBU queries) + 4 custom multi-turn plan scenarios (GENERATE→CHECK-IN→edit→mindmap).

### 6.2 Module layout

```
backend/app/eval/p2_2_agent_ablation/
├── run_eval.py           # top-level CLI: --model --mode --runs --output
├── matrix.py             # matrix expansion + task scheduling
├── single_run.py         # one run: graph.ainvoke + extract agent_trace + auto metrics
├── judges.py             # dual judge: qwen2.5:7b local + cloud BYOK GPT-4o-mini
└── output/results.jsonl  # append-only data file
```

Bypasses LangGraph topology entry from FastAPI; directly `await graph.ainvoke(input_state, config=...)`. Each run appends one JSON line to `output/results.jsonl`.

### 6.3 Per-run record schema

```json
{
  "run_id": "uuid",
  "timestamp": "ISO-8601",
  "model": "qwen3.5:4b",
  "mode": "agent_loop",
  "query_id": "hyde_basic",
  "run_idx": 0,
  "operational": {
    "wall_time_s": 4.23,
    "iterations": 4,
    "tool_calls": [{"name":"retriever_search","success":true,"error":null}, ...],
    "tool_call_count": 4,
    "tool_errors": 0,
    "input_tokens": 1843,
    "output_tokens": 412,
    "exit_reason": "natural_stop"
  },
  "output": {
    "plan_action": "generate",
    "milestones_persisted": 5,
    "milestones_json": [...],
    "final_text_excerpt": "first 500 chars"
  },
  "judge_local": {"score": 0.78, "weak_dims": ["milestone_granularity"], "reasoning": "..."},
  "judge_cloud": {"score": 0.82, "weak_dims": [], "reasoning": "...", "model": "gpt-4o-mini"}
}
```

### 6.4 Execution strategy

- **Sequential** (no concurrency) — Ollama 本地单实例，并发 = GPU 争抢
- **Resumable** — `run_eval.py` 启动时读 `results.jsonl`，跳过已完成 `run_id`
- **Failure not retried** — single run fails → mark `error` row, next run; failure 本身是数据
- **Wall time estimate** — single run avg 5-15s, total 360 runs ≈ 1-2 hours local

### 6.5 Metrics dimensions

| Dimension | Source | Objectivity |
|---|---|---|
| latency | `operational.wall_time_s` | fully objective |
| tool calling correctness | `operational.tool_calls[].name` 序列 vs expected pattern | objective (rule-based) |
| robustness | `operational.exit_reason` distribution | objective |
| token cost | `input_tokens + output_tokens` | objective |
| plan quality | `judge_local.score` + `judge_cloud.score` average | subjective (bias disclosed) |
| judge agreement | `|judge_local.score - judge_cloud.score|` distribution | meta-metric |

---

## 7. Testing Strategy

| Layer | File | Coverage | Count |
|---|---|---|---|
| A. Tool unit | `tests/agent/test_planner_agent_tools.py` | each @tool wrapper × 1 test: schema + closure injection + JSON return | 5 |
| B. AgentTrace | `tests/agent/test_agent_trace.py` | record_iteration / record_tool_call / serialize / budget_exhausted | 3 |
| C. Loop (stub LLM) | `tests/agent/test_planner_agent_loop.py` | natural_stop / budget_exhausted / llm_error / tool_error self-correct | 5 |
| D. plan_action inference | same as C | generate inference / check_in inference / no-tools fallback | 3 |
| E. Graph e2e (stub LLM) | `tests/agent/test_graph_plan_agent_e2e.py` | mode switching / agent_trace lands in state / judge reuses | 3 |
| F. Routes integration | `tests/api/test_routes_plan_agent.py` | x-planner-mode header routing / SSE contract identical | 2 |
| G. Eval harness | `tests/eval/test_p2_2_harness.py` | matrix expansion / record schema / resumable | 3 |

**Target: 157 baseline + 24 = 181 tests passing**.

---

## 8. Cut-by-Cut Plan Skeleton

writing-plans skill will detail each cut; this is the skeleton:

- **P2.2-①a** Tool wrappers + closure factory + schemas (A 5 tests, ~1-2 days)
- **P2.2-①b** AgentTrace dataclass (B 3 tests, ~half day)
- **P2.2-①c** agent_loop body + plan_action inference + error handling (C+D 8 tests, ~2-3 days)
- **P2.2-①d** Graph wiring + state extension + mode dispatch (E 3 tests, ~1 day)
- **P2.2-①e** deps.py + routes.py production wiring (F 2 tests, ~1 day)
- **P2.2-①f** Real-Ollama smoke test 4 models × 2 modes one happy-path run each (manual, ~1 day)
- **P2.2-②a** Eval harness skeleton (G 3 tests, ~1-2 days)
- **P2.2-②b** Run full matrix 360 runs, dump results.jsonl (~1-2 hours wall time, plus 1 day analysis)
- **P2.2-③** EVAL.md integration + blog post "Did I build a real agent? An empirical answer" (~2 days)

**Total estimate: 10-13 days, ~2 weeks**.

---

## 9. Verification Criteria

- [ ] 181 tests passing (157 baseline + 24 P2.2)
- [ ] 4 models × 2 modes pass real Ollama smoke test (P2.2-①f happy path each)
- [ ] Eval matrix 360 runs complete, `results.jsonl` persisted
- [ ] `docs/EVAL.md` contains ≥ 1 data table + ≥ 1 conclusion paragraph ("which mode wins on which dimension")
- [ ] Blog post `docs/agent_loop_vs_deterministic.md` publishable, explicitly responds to learn-claude-code's thesis
- [ ] memory `project_study_coach_refactor.md` updated with P2.2 results
- [ ] ROADMAP.md updated
- [ ] cloud-adapt hooks all grep-able

---

## 10. Out of Scope

- Frontend Plan Timeline view (P3 polish)
- LLM tool-calling for Quiz path (P2.3; do Planner first)
- Persistent checkpointer upgrade (P3)
- Thinking-ON in main experiment (only appendix on gemma4:e4b)
- deepseek-r1 series (Ollama manifest no `tools` capability — drop with note in EVAL)
- ReAct-style prebuilt agents (`langgraph.prebuilt.create_react_agent`) — would abstract away the very thing we're studying
- Multi-agent (Planner-orchestrator + Quiz-subagent etc.) — P3
- mistral-nemo:12b (skipped to keep matrix at 4 models; revisit if 4-model data is inconclusive)

---

## 11. Cloud-Adapt Hooks (grep anchors)

```python
# cloud-adapt: thinking models can enable_thinking=True; eval main axis OFF + appendix ON
# cloud-adapt: max_iter=10 for local small models; cloud BYOK can raise to 20-30
# cloud-adapt: production deploy strip agent_trace from response (size concern)
# cloud-adapt: x-judge-model header supports dual-judge routing already
# cloud-adapt: BYOK cloud models can default mindmap_default=True via factory kwarg
# cloud-adapt: tool descriptions can be terser for cloud models; small models need explicit "when to use"
```

---

## 12. References

### Internal

- P2.1-⑤ spec: `docs/superpowers/specs/2026-05-22-p2-1-5-plan-chain-design.md`
- P2.1-⑤ plan: `docs/superpowers/plans/2026-05-22-p2-1-5-plan-chain.md`
- ARCHITECTURE.md §1 + §3 (tool table)
- ROADMAP.md P2.1-⑤ block

### External reference repo

- `learn-claude-code` at `/Users/lianghaozhe/learn-claude-code/`
  - s01_agent_loop.py — base while-loop template
  - s02_tool_use.py — dispatch map pattern
  - s04_subagent.py — context isolation pattern (used as max_iter=30 reference)
  - docs/zh/s01-the-agent-loop.md, s02-tool-use.md
  - Repo's central thesis (LEARN-CLAUDE-CODE-README-zh.md): "Agency 来自模型 + harness 让 agency 落地"
- LangChain docs: `bind_tools`, `@tool` decorator, ChatOllama
- Ollama tool calling: https://ollama.com/blog/tool-support

### Memory entries

- `feedback_cloud_model_adaptation_hooks` (2026-05-22) — must mark not implement
- `feedback_scaffold_write_readfirst` (Phase 1 lesson) — relevant if scaffolding any new directory
- `skills_usage_by_phase` (P2 skill activation rules)
- `project_study_coach_refactor` — full project context; will be updated with P2.2 segment
