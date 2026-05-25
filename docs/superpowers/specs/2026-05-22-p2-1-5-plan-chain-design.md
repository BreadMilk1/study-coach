# P2.1-⑤ Plan Chain — Design Spec

> Brainstormed 2026-05-22 via `superpowers:brainstorming`.
> Upstream: P2.0 + P2.1-①/②/③/④ (含 ④a-④i). Baseline: 124 backend tests passing.
> Target: Planner + Reviewer 双节点一次交付（ARCHITECTURE.md §1 原图），2 new tools
> (`update_study_plan` + `generate_mindmap`).

---

## 1. Decisions Locked (Brainstorming Q&A)

| # | Question | Choice |
|---|---|---|
| Q1 | Planner 核心行为 | **B. 生成 + 自适应调整** — GENERATE + CHECK-IN 双路径 |
| Q2 | `generate_mindmap` 触发 | **C. Planner 内部关键词可选** — 默认快路径，关键词显式触发 |
| Q3 | Reviewer 职责 | **C. 双职责** — GENERATE 审质量（plan rubric）、CHECK-IN 审进度（短路 pass） |
| Q4 | 多轮交互 | **A. 设 `active_plan_id`** — state-aware router 复用 P2.1-④ 模式 |

**Architectural approach chosen**: Approach 2 — Planner + Reviewer 两个独立节点，Reviewer
复用 `judge_node` 的 multi-rubric 路由（不另起 `plan_reviewer_node`）。

---

## 2. Architecture & Graph Topology

```
START
  → memory_hydrator
  → router (state-aware: active_plan_id 存在 → intent=plan)
  → {tutor → judge(tutor rubric) | quiz → judge(quiz rubric) | planner → judge(plan rubric)}
                                                                              ↓
                                                                       memory_writer
                                                                              ↓
                                                                            END
```

变化点 vs P2.1-④ 现有拓扑：

1. **新节点 `planner_node`** 替换 `plan_stub_node`（同 ④e replace quiz_stub 模式）。
2. **`judge_node` multi-rubric 路由扩第三种 rubric**: `intent == "plan" AND plan_action == "generate"` → `load_plan_rubric()` + `PLAN_DIMENSIONS`。
3. **`judge_node` 跳判扩展**: `intent == "plan" AND plan_action == "check_in"` → 短路 pass，复用 ④g grade-skip 逻辑分支。
4. **router state-aware 再扩一条**: `active_plan_id` 存在 → 强制 `intent="plan"`（同 ④e `active_quiz_question_id` 模式）。
5. **图编译入口 `build_graph(..., checkpointer=None)`** 签名不变。`planner` 通过
   `RunnableConfig.configurable` 注入（同 quiz_master / memory_hydrator / memory_writer 模式）。

### 2.1 为什么 Reviewer = `judge_node` 而不是新节点

- `judge_node` 已识别 `state.intent` + `state.quiz_action`，加 `state.plan_action` 是最自然扩展。
- ARCHITECTURE.md §1 原图把 Judge Guard 描述为"统一守卫"，复用契合该精神。
- plan rubric 4-5 维可控，judge.py 仍能容纳；未来 plan rubric 维度膨胀再 fork 出 `plan_reviewer_node`。
- 唯一例外 CHECK-IN 走确定性进度总结、不调 LLM judge（确定性输出无 LLM 可判内容，复用 ④g grade-skip 判例）。

---

## 3. Planner Internal State Machine & Tools

### 3.1 伪代码

```python
async def planner_node(state, config):
    writer = _safe_writer()
    user_msg = last_human(state)
    user_id = state.get("user_id")

    plan_action = decide(state, user_msg)            # "generate" | "check_in"

    if plan_action == "generate":
        topic     = extract_topic(user_msg)
        active    = goal_repo.list_active_for_user(user_id)
        goal      = active[0] if active else goal_repo.create(
                        user_id=user_id, title=_DEFAULT_GOAL_TITLE)
        chunks    = retriever.search(topic, top_k=5) if retriever else []
        milestones_raw = await llm.ainvoke(planner_prompt(
            topic, chunks, state.get("mastery_scores", {}), goal.exam_date
        ))
        milestones = parse_milestones(milestones_raw)         # pydantic Milestone[]
        if not milestones:
            return _err(writer, f"Couldn't draft a plan on '{topic}'. Try a clearer goal.")
        plan = update_study_plan(goal_id=goal.id,
                                 milestones=milestones,
                                 plan_repo=plan_repo)

        mindmap = None
        if _has_mindmap_keyword(user_msg):                    # cloud-adapt: provider==cloud → 默认 True
            try:
                mindmap = await generate_mindmap(topic=topic,
                                                 milestones=milestones,
                                                 llm=llm)
            except Exception:
                mindmap = None                                # tolerant: 失败仅丢 mindmap

        text = format_plan_output(plan, mindmap)
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": text})
        return {
            "messages": [AIMessage(content=text)],
            "citations": [],
            "active_plan_id": plan.id,
            "plan_action": "generate",
            "last_context": format_chunks(chunks),
        }

    else:  # check_in
        active = goal_repo.list_active_for_user(user_id)
        plan   = plan_repo.get_by_goal(active[0].id) if active else None
        if plan is None:                                       # 用户删了 goal/plan 又来 check-in
            return await _fallback_to_generate(state, config)
        goal = active[0]
        progress = compute_progress(
            plan, state.get("mastery_scores", {}), state.get("recent_mistakes", []),
        )
        adjusted_raw = await llm.ainvoke(check_in_prompt(plan.milestones_json, progress))
        adjusted = parse_milestones(adjusted_raw)
        if adjusted:
            plan = update_study_plan(goal_id=goal.id,
                                     milestones=adjusted,
                                     plan_repo=plan_repo)
            schema_skip_note = ""
        else:
            schema_skip_note = "\n\n⚠️ Auto-adjust skipped: model output didn't match plan schema."

        text = format_check_in_output(plan, progress) + schema_skip_note
        writer({"type": "citations", "citations": []})
        writer({"type": "token", "text": text})
        return {
            "messages": [AIMessage(content=text)],
            "citations": [],
            "active_plan_id": plan.id,
            "plan_action": "check_in",
            "last_context": "",
        }
```

### 3.2 `decide(state, user_msg)` 决策表

| state 输入 | plan_action |
|---|---|
| `active_plan_id is None` | `generate` |
| `active_plan_id` 存在 + msg 含 check-in 关键词（`check in` / `进度` / `怎么样了` / `调整`） | `check_in` |
| `active_plan_id` 存在 + 其它消息（如「把第三章拆两节」） | `check_in` （默认增量） |

### 3.3 两个新 tool（`backend/app/agent/tools/plan.py`）

| Tool | 签名 | 副作用 | Stateless |
|---|---|---|---|
| `update_study_plan` | `(*, goal_id: str, milestones: list[Milestone], plan_repo: PlanRepository) -> Plan` | upsert `plans.milestones_json + updated_at`（无 plan → create；有 plan → 覆盖） | ✅ |
| `generate_mindmap` | `(*, topic: str, milestones: list[Milestone], llm) -> MindmapOut` | none（纯 LLM 调用 + 解析） | ✅ |

- milestone 结构（pydantic `Milestone` in `tools/schemas.py`）:
  `{title: str, due_at: date|None, done: bool=False, topic: str|None}`，
  与 ARCHITECTURE.md §4 `plans.milestones_json` 字段对齐。
- `generate_mindmap` 输出: `MindmapOut(mermaid_src: str, markdown_outline: str)`. tolerant:
  fenced ```mermaid → bare → 失败 → 仅 markdown outline（mermaid_src 为空字符串），不抛。

### 3.4 `_safe_writer()` 复用

- planner_node 用与 `quiz_master.py` 同名 `_safe_writer()`（试 import；不抽公共模块以免引入 yet-another-utils），允许单元测试直调而不进 graph runnable context。

---

## 4. State Extension & Data Flow

### 4.1 `CoachState` 新增字段

```python
# P2.1-⑤ Plan
active_plan_id: NotRequired[str | None]
plan_action: NotRequired[Literal["generate", "check_in"]]
```

- 全 `NotRequired`，与 ④ 的 `active_quiz_question_id` / `quiz_action` 对称命名。
- 不引入 `current_topic` / `goal_id` 等 ARCHITECTURE.md §2 表里列的字段——planner 内部解析即可，跨节点无消费者。YAGNI。
- Citation 字段空数组（Plan 路径不产 RAG citation；mindmap 引用的 chunk 来自 retriever search，但不进 citations 数组，避免误导）。

### 4.2 GENERATE 数据流

```
user msg "帮我做学习计划 on HyDE [+画脑图]"
    ↓
[router] state-aware: active_plan_id=None → route_intent(msg)="plan"
    ↓
[planner GENERATE]
  1. emit("citations", [])
  2. topic = "HyDE"
  3. goal = active_goals[0] or create_default_goal()
  4. chunks = retriever.search("HyDE", top_k=5)        # 复用 ④h RAG-grounding
  5. milestones = LLM(planner_prompt(topic, chunks, mastery, exam_date))
  6. plan = update_study_plan(goal.id, milestones, plan_repo)    # upsert
  7. if has_mindmap_keyword(msg):
       mindmap = await generate_mindmap(topic, milestones, llm)
  8. text = format_plan_output(plan, mindmap)
  9. emit("token", text)                               # 一次性 emit (同 quiz)
    ↓ {active_plan_id, plan_action:"generate"}
[judge plan rubric]
    pass → memory_writer
    weak → degrade with disclaimer → memory_writer
    ↓
[memory_writer] no-op (planner 无 mastery delta)
    ↓ END
```

### 4.3 CHECK-IN 数据流

```
user msg "进度怎么样了"  (active_plan_id 存在)
    ↓
[router] state-aware → intent="plan"
    ↓
[planner CHECK-IN]
  1. plan = plan_repo.get_by_goal(goal_id)
  2. progress = compute_progress(plan, mastery, recent_mistakes)
       → {done_count, total_count, overdue, weak_topics, recent_mistake_count}
  3. adjusted = LLM(check_in_prompt(plan.milestones, progress))
  4. parse → 成功: update_study_plan; 失败: 沿用原 plan + schema_skip_note
  5. text = format_check_in_output(plan, progress)[ + schema_skip_note ]
  6. emit("token", text)
    ↓ {plan_action:"check_in"}
[judge plan rubric] check_in 短路 pass (复用 ④g 模式)
    ↓
[memory_writer] no-op
    ↓ END
```

### 4.4 SSE 合约（前端零改）

`citations(空) → token(整段文本) → done`，与 tutor/quiz 一致。Plan 路径文本可能很长
（milestones + mermaid + outline），用 markdown，前端现有渲染器（v-md-editor 或类似）已能处理。
气泡渲染竖排问题不会出现，因为：

1. 一次性 emit 整段（同 quiz GENERATE，已在 ④h 验证 SSE 正常）。
2. 不走 LLM astream chunk 级 emit（仅 tutor 那样）。
3. `json.dumps` 标准转义不变。

### 4.5 `compute_progress` pure function

```python
def compute_progress(plan, mastery_scores, recent_mistakes, now=None) -> ProgressSummary:
    now = now or datetime.utcnow()
    milestones = plan.milestones_json or []
    done = [m for m in milestones if m.get("done")]
    overdue = [
        m for m in milestones
        if not m.get("done")
        and m.get("due_at")
        and parse_date(m["due_at"]) < now
    ]
    weak_topics = [name for name, score in mastery_scores.items() if score < 0.4]
    return ProgressSummary(
        done_count=len(done),
        total_count=len(milestones),
        overdue=overdue,
        weak_topics=weak_topics,
        recent_mistake_count=len(recent_mistakes),
    )
```

完全确定性、独立可测（pure function，无 LLM 无 DB），TDD 第一刀。喂给 check_in LLM prompt 作为 grounding context。

---

## 5. Error Handling, Degrade, Plan Rubric

### 5.1 错误处理矩阵

| 失败点 | 处理 | 用户可见 |
|---|---|---|
| `extract_topic()` 解析不到主题 | 用整句作为 topic 喂 LLM | 正常 |
| `retriever.search()` 空 | `chunks=[]`，prompt 走 ungrounded fallback（同 ④h） | 正常 |
| LLM milestones JSON 解析失败 | 3-tier tolerant 解析（fenced array → strict → 首个 `[...]` 块） → 失败 → 返回错误文案，不持久化 | 错误文案 |
| `update_study_plan` DB 写失败 | 异常上抛 → FastAPI 500 | routes 层异常文案 |
| `generate_mindmap` 失败 / mermaid 错 | tolerant → 仅 markdown outline | milestones + outline，无 mermaid |
| CHECK-IN LLM 输出破坏 schema | pydantic 验证失败 → 不写 DB，沿用原 plan + `schema_skip_note` | 进度报告正常，调整跳过 |
| `active_plan_id` 存在但 plan 已被外部删除 | `get_by_goal` 返回 None → 自动回退到 GENERATE | 首次生成体验，无报错 |

### 5.2 Judge 跳判 & 降级

```
intent=plan, plan_action="generate" → plan rubric
    score ≥ 0.6 → pass → memory_writer
    score < 0.6 → degrade 直接（不重跑：deterministic 重跑同输入会循环；
                          重生成 milestones = ablation 留 P2.2）
                  append "⚠️ Self-check note: plan scored low (...; weak on ...).
                          考虑提供 exam_date 或上传更相关材料。"

intent=plan, plan_action="check_in" → 短路 pass (judge_score=1.0)  ← ④g grade-skip
```

降级文案沿用 `_degrade_disclaimer(score, weak_dims, retry_count=0)`——当前签名已支持
`retry_count=0`，不改 judge.py。

### 5.3 Plan Rubric (`prompts/judge_plan.txt` + `PLAN_DIMENSIONS`)

5 维（同 quiz 维度数；与 tutor/quiz 完全不重叠）:

| 维度 | 1-5 评分含义（5=优秀） |
|---|---|
| `milestone_specificity` | milestone 具体可执行（"读完§3" 5；"复习数学" 2） |
| `milestone_granularity` | 粒度均匀，无超大块或碎屑（典型 3-7 个 milestone 5；单块或 >10 个 2） |
| `time_feasibility` | due_at 分布相对 exam_date 合理；无堆末日或全 None；与今天的时间差合理 |
| `topic_coverage` | 覆盖 weak_topics（mastery < 0.4）；不重复堆同主题 |
| `actionability` | 用户读完知道下一步做什么；无"加油学习"空话 |

- 阈值 0.6（= 3/5 平均，同 tutor/quiz）
- weak_dims = ≤3 分维度（同 tutor/quiz）
- prompt 加 bias-aware 指令 + 2-3 calibrated few-shot；不需 6 example
- `load_plan_rubric()` + `PLAN_DIMENSIONS` 加到 `judge.py`，与 tutor/quiz 平行
- `judge_node` rubric/dimensions 选择扩展为 `intent in {"plan","quiz"} else tutor` 三分支

---

## 6. Testing Strategy

| 层 | 文件 | 覆盖 | 数量 |
|---|---|---|---|
| A. Pure function | `tests/agent/test_compute_progress.py` | done/overdue/weak_topics/空 plan/全 done/None due_at/now 注入 | 5 |
| B. Tools | `tests/agent/test_plan_tools.py` | `update_study_plan` upsert（create/覆盖/bump updated_at）；`generate_mindmap` LLM tolerant（fenced/bare/失败 fallback） | 5 |
| C. Planner node 直调 | `tests/agent/test_planner.py` | GENERATE 含/不含 mindmap；CHECK-IN 含 schema 失败回退；`decide()` 3 行；no-user_id；plan 被外删 → GENERATE 回退 | 7 |
| D. Plan rubric | `tests/agent/test_judge_plan.py` | `load_plan_rubric` 文件；`PLAN_DIMENSIONS` 5 维互斥；好/弱 plan 评分；check-in skip 短路 | 4 |
| E. Graph e2e | `tests/agent/test_graph_plan.py` | router state-aware → intent=plan；GENERATE → judge plan → memory_writer → END；CHECK-IN → judge skip → END；degrade 路径 | 4 |
| F. Routes integration | `tests/api/test_routes_plan.py` | `/api/chat` 单轮 GENERATE SSE 三段；双轮 GENERATE→CHECK-IN（thread_id 复用） | 2 |
| G. Repository | `tests/db/test_plan_repository.py` | `PlanRepository.update_milestones`（upsert + updated_at） | 2 |

**预期总数**: 124 baseline + 29 = **~153 tests passing**。

### 6.1 Cut-by-cut TDD 骨架（writing-plans 细化）

```
Cut ⑤a  compute_progress (pure)                           — 5 tests   (1-2h)
Cut ⑤b  PlanRepository.update_milestones                  — 2 tests   (~30min)
Cut ⑤c  plan tools: update_study_plan + generate_mindmap  — 5 tests   (1-2h)
Cut ⑤d  PLAN_DIMENSIONS + load_plan_rubric + judge 扩展    — 4 tests   (1h)
Cut ⑤e  planner_node deterministic (GENERATE + CHECK-IN)  — 7 tests   (2-3h)
Cut ⑤f  graph wire + state-aware router + state 字段       — 4 tests   (1-2h)
Cut ⑤g  production wiring (deps + routes + main)          — 2 tests   (1h)
Cut ⑤h  real-Ollama E2E + memory/ROADMAP 同步              — manual    (1h)
```

每 cut 完跑 `uv run pytest -q` 全绿才进下一刀。

---

## 7. Production Wiring (Cut ⑤g)

参考 P2.1-④f 范式，对称加 plan。

| 文件 | 改动 |
|---|---|
| `app/api/deps.py` | `+ get_planner(...)` 工厂；session-scoped repo 注入；retriever 从 app.state 取 |
| `app/api/routes.py` | `config["configurable"]["planner"] = planner_callable`（同 quiz_master 模式） |
| `app/main.py` | 无改动（`InMemorySaver` 已就位） |
| `app/agent/graph.py` | `plan_stub_node` → `planner_node` delegator（同 quiz_node 模式）；fallback 仍是 stub message 让 P2.1-① 旧 test 通过 |

---

## 8. Cloud-Model Adaptation Hooks (Mark, Do Not Implement)

每个 hook 是一行 `# cloud-adapt: ...` 注释 + 此 spec 索引；不加 provider detection / feature
flag / 死代码分支。

| 位置 | 当前（local Ollama 最优） | 云端切换条件 / 改动 |
|---|---|---|
| `planner_node` mindmap 触发 | `_has_mindmap_keyword(msg)` | `provider == cloud or keyword` |
| `planner_node` GENERATE 文本 emit | 一次性整段（避免 chunk 级竖排隐患） | switch to chunk-level `astream` when latency < 3s |
| `planner_node` CHECK-IN LLM prompt | 强约束「只能 add / reorder / mark done」 | 可放开重写约束 |
| `judge_node` plan rubric | 本地 qwen2.5:7b（同 quiz） | `x-judge-model` header 已支持；零改动 |
| `judge_node` 阈值 0.6 | 保守，避免 gemma3:4b 误伤 | 可调到 0.7 |
| CHECK-IN schema 验证容忍度 | 严格（小模型易胡说） | 可放宽 |
| `compute_progress` 输入 | plan + mastery + mistakes | 可扩 chat 历史摘要 |
| `deps.py::get_planner` | 默认 mindmap_default=False | `# cloud-adapt: when provider=cloud, can pass mindmap_default=True` |

---

## 9. Verification Criteria (P2.1-⑤ Done Definition)

- [ ] `uv run pytest -q` ≥ 153 passing (124 baseline + 29 new)
- [ ] `帮我做学习计划 on HyDE` → SSE `citations(空) → token → done`；UI 看到 ≥3 milestones + (无 mindmap 关键词时)无 mermaid
- [ ] `帮我做学习计划 on HyDE 画脑图` → 上述 + 合法 mermaid + markdown outline
- [ ] 多轮: 第一轮 GENERATE 后第二轮「进度怎么样了」→ CHECK-IN 路径触发，judge skip，输出 progress 卡片
- [ ] 多轮: 第一轮 GENERATE 后第二轮「把第三章拆两节」→ CHECK-IN 路径，milestones 被 LLM 增量更新（或 schema 失败 → schema_skip_note）
- [ ] Plan path 不竖排（一次性 emit 验证）
- [ ] Production wiring: deps.py + routes.py 完整；checkpointer thread 持续 active_plan_id
- [ ] 不回归 tutor / quiz 任何既有 e2e
- [ ] memory + ROADMAP 同步更新

---

## 10. Out of Scope (Deferred)

- **LLM tool-calling Planner** (deterministic 是基线；agent loop 留 P2.2/P3 ablation)
- **多 active goal 并存**: 当前 `goal_repo.list_active_for_user(user_id)[0]` 取第一个 active；多 goal UI + 选择路由留 P3
- **Plan rubric 重生成 milestones**: deterministic 重跑会循环；重生成 = future ablation
- **Frontend Plan Timeline view**: 后端 SSE 合约不变，前端富 UI 留 P3（ROADMAP P3 "UI/UX polish"）
- **跨 session active_plan_id 持久化**: 依赖 `InMemorySaver`；持久 checkpointer 升级到 SqliteSaver 留 P3
- **退出 plan chain 机制**: state-aware router 强制 `intent=plan` 一旦 `active_plan_id` set，用户无法在同一 thread 内回到 tutor 问普通问题（症状: "What is HyDE?" 会进 CHECK-IN，被 LLM 当作 plan 调整意图）。**临时缓解**: 用户切换 thread (新 session_id) 即可。**正式 fix 留 P2.2**: 加 "exit plan" 关键词清空 `active_plan_id`，或 router 改条件覆盖（tutor 关键词强信号时退出）。在 §9 verification 里明确不验证此场景。

---

## 11. References

- ARCHITECTURE.md §1, §2, §3 (tool table rows for `update_study_plan` / `generate_mindmap`), §4 (`plans.milestones_json` schema)
- ROADMAP.md P2.1-⑤
- `app/agent/quiz_master.py` (deterministic node template)
- `app/agent/judge.py::judge_response(dimensions=)` (multi-rubric extension point)
- `app/agent/graph.py` (state-aware router pattern from ④e)
- `app/api/deps.py::get_quiz_master` (factory + production wiring template)
- Memory: `feedback_cloud_model_adaptation_hooks` (2026-05-22), `skills_usage_by_phase`, `project_study_coach_refactor`
