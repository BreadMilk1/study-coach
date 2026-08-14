# Learning Run Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task by task. Use `superpowers:test-driven-development` for every production-code change and `superpowers:verification-before-completion` before claiming completion.

**Goal:** 在不偏离 Study Coach 学习教练主题的前提下，实现一个面向技术面试官的 Tutor Grounded QA Learning Run Harness：受控执行、exact evidence、Hybrid Score、historical re-score、controlled compare 和 Run Lab UI。

**Architecture:** Production Graph 与 Eval Runner 只共享 graph-free `TutorAttemptEngine`。Registry 解析 Git 中冻结的 Task/Prompt/Corpus/Scorer definitions；Eval 使用隔离 corpus 执行一次 Tutor attempt，将 immutable CandidateArtifact 和“历史只追加、单行只终结一次”的 ScoreSets 保存到现有 SQLite 的三个 eval tables；FastAPI 用 authenticated POST fetch stream 暴露执行与 re-score，Vue 中央 store 管理 attached execution。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2、Alembic、SQLite、LangGraph/LangChain、Chroma、pytest/pytest-asyncio、Vue 3、Pinia、TypeScript、Vitest、Tailwind CSS。

---

## 执行约束

- 设计事实来源：`docs/superpowers/specs/2026-08-12-learning-run-harness-design.md`、`CONTEXT.md`、`docs/adr/0001-share-tutor-attempt-not-production-orchestration.md`。
- 当前 `study-coach/main` 是 dirty planning checkout，且在计划编写时落后 `origin/main` 1 commit；不得在该 checkout 实现业务代码。
- 实施必须从最新 `origin/main` 建立独立 worktree，并在开始每个任务前检查 `git status --short --branch`。
- 不覆盖、回退或格式化用户既有改动；README、ARCHITECTURE、EVAL、DEMO、ROADMAP 只在最终文档任务处理，并先人工合并语义。
- 不新增 dependency。若实现发现必须新增，停止当前 task，提供替代方案、风险和依赖理由，等待批准。
- 不运行真实付费/远程模型作为 CI gate；单元与集成测试使用 fake retriever、fake LLM、fake scorer。
- 除非代码块内部明确 `cd ..`，每个命令代码块都以 feature worktree repository root 为独立起点执行；不要继承上一个代码块的 cwd。
- 本计划中的 commit 命令全部是 **approval-gated**：只有用户再次明确授权才执行。否则只保留经过验证的 working tree diff。
- 每个任务必须先记录真实 RED 命令、失败测试和关键失败原因，再写最少生产代码并记录 GREEN 命令与 exit code。

## 完成定义

只有以下全部成立才可称 MVP 完成：

1. Production Graph adapter 与 Eval Runner 共用同一个 `TutorAttemptEngine` contract，Graph retry 与 Eval exactly-once 均有测试。
2. Registry 中存在冻结且 hash 可验证的 12-case suite、独立 calibration fixtures、`tutor-v2` 与 `tutor-v3`；当前生产默认仍是逐字一致的 v2。
3. Eval 只能读取隔离 corpus，不触碰 global retriever、Router、runtime Judge、Memory 或 Chat persistence。
4. CandidateArtifact 单次 finalize；historical re-score 只追加 ScoreSet，单个 ScoreSet 只 terminal finalize 一次；malformed/timeout scorer 不会 fallback Pass。
5. run、cancel、re-score、compare、single-flight、restart reconciliation、reset/auth/local-mode contracts 都有自动化测试。
6. Run Lab 三个页面可 build，并用真实生成的历史 artifacts 展示 expected-refusal 与至少一个 regression。
7. 全量 backend pytest、frontend non-watch Vitest、production build、fresh Alembic migration 全部通过；无法执行的项目明确列为未验证。

## Task 0: 建立可执行的干净边界

**Files:**

- Read only: `AGENTS.md`
- Read only: `docs/ROADMAP.md`
- Read only: `docs/ARCHITECTURE.md`
- Read only: `docs/EVAL.md`
- Read only: `design-system/MASTER.md`
- Read only: `docs/superpowers/specs/2026-08-12-learning-run-harness-design.md`
- Read only: `CONTEXT.md`
- Read only: `docs/adr/0001-share-tutor-attempt-not-production-orchestration.md`
- Carry forward: `.gitignore`
- Carry forward: `CONTEXT.md`
- Carry forward: `docs/adr/0001-share-tutor-attempt-not-production-orchestration.md`
- Carry forward: `docs/superpowers/specs/2026-08-12-learning-run-harness-design.md`
- Carry forward: `docs/superpowers/plans/2026-08-12-learning-run-harness.md`
- Local-only carry forward: `AGENTS.md`

### Step 1: 记录 planning checkout 状态

Run from repository root:

```bash
git status --short --branch
git rev-list --left-right --count main...origin/main
git worktree list --porcelain
```

Expected: 记录而不是清理 dirty changes；确认真正的 repo root 与现有 worktrees。

### Step 2: 同步远端引用并建立隔离 worktree

以下是 implementation kickoff，执行前再次取得用户授权：

```bash
git fetch origin
git worktree add .worktrees/learning-run-harness -b feat/learning-run-harness origin/main
```

Expected: 新 worktree 指向当时最新 `origin/main`，原 dirty checkout 不变。

### Step 3: 在新 worktree 重做治理检查

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

Expected: worktree clean，两个 SHA 相同；否则停止，不进入 Task 1。

### Step 4: 建立 baseline verification

```bash
cd backend
uv sync --frozen
uv run pytest
cd ../frontend
pnpm install --frozen-lockfile
pnpm exec vitest run
pnpm build
cd ..
```

Expected: 记录每条命令 exit code。若 baseline 失败，先区分环境问题与既有回归，不将其归因于 Learning Run。

### Step 5: 只迁入已批准的 planning artifacts

新 worktree 不会自动包含当前 dirty checkout 的 untracked docs。使用 `apply_patch` 将以下内容逐字添加到 feature worktree：

- `.gitignore` 中仅本 spec/plan 的两条 unignore 规则。
- `CONTEXT.md`。
- `docs/adr/0001-share-tutor-attempt-not-production-orchestration.md`。
- approved design spec。
- 本 implementation plan。
- 当前 repo-local `AGENTS.md` 作为 ignored local guard；不得 stage/commit。

迁移前后分别运行：

```bash
shasum -a 256 CONTEXT.md docs/adr/0001-share-tutor-attempt-not-production-orchestration.md docs/superpowers/specs/2026-08-12-learning-run-harness-design.md docs/superpowers/plans/2026-08-12-learning-run-harness.md
git check-ignore -v AGENTS.md
git status --short --branch
```

Expected: 四份批准文档的 hashes 与 planning checkout 相同；`AGENTS.md` 仍 ignored；除这四份 docs 与 `.gitignore` 的精确规则外没有其他 dirty 文件。不得从 planning checkout 复制 README、ARCHITECTURE、DEMO、ROADMAP 或任何业务代码。

**Stop condition:** clean worktree、latest origin/main、baseline evidence 三项缺一不可。

## Task 1: 抽取 graph-free TutorAttemptEngine，同时保护生产 Graph parity

**Files:**

- Create: `backend/app/agent/tutor_attempt.py`
- Modify: `backend/app/agent/prompt.py`
- Modify: `backend/app/agent/graph.py`
- Create: `backend/tests/agent/test_tutor_attempt.py`
- Modify: `backend/tests/agent/test_graph.py`
- Modify: `backend/tests/agent/test_graph_judge.py`
- Verify: `backend/tests/api/test_routes_graph_stream.py`

### Step 1: 写共享 engine contract 的 RED test

在 `backend/tests/agent/test_tutor_attempt.py` 写固定 fake retriever、fake token sequence 与 recording sink：

```python
@pytest.mark.asyncio
async def test_tutor_attempt_returns_exact_candidate_and_streams_tokens():
    retriever = FakeRetriever(chunks=[CHUNK_A, CHUNK_B])
    sink = RecordingSink()

    candidate = await TutorAttemptEngine().answer(
        question="What is RRF?",
        retriever=retriever,
        llm=FakeStreamingLLM(["Reciprocal ", "rank fusion [1]."]),
        prompt_template=TutorPromptTemplate(version="tutor-v2", system_instruction="frozen"),
        event_sink=sink,
        attempt_config=TutorAttemptConfig(top_k=5, retrieval_seconds=5, generation_seconds=55),
    )

    assert retriever.search_calls == [("What is RRF?", 5)]
    assert candidate.answer == "Reciprocal rank fusion [1]."
    assert candidate.evidence == [CHUNK_A, CHUNK_B]
    assert candidate.citations[0]["chunk_id"] == CHUNK_A["chunk_id"]
    assert [event["text"] for event in sink.of_type("token")] == [
        "Reciprocal ", "rank fusion [1]."
    ]
```

同时覆盖 empty retrieval、LLM exception、usage unavailable、deadline 后禁止返回 Candidate。

### Step 2: 运行 RED

```bash
cd backend && uv run pytest tests/agent/test_tutor_attempt.py -q
```

Expected RED: `ModuleNotFoundError: app.agent.tutor_attempt` 或缺少 contract types；不能接受因 fixture/import 拼写导致的假 RED。

### Step 3: 写最小 engine contract

`backend/app/agent/tutor_attempt.py` 只定义 Attempt 层职责：

```python
@dataclass(frozen=True)
class TutorCandidate:
    answer: str
    citations: list[dict]
    evidence: list[dict]
    formatted_context: str
    usage: dict[str, int] | Literal["unavailable"]
    trace: list[dict]

class TutorAttemptEngine:
    async def answer(
        self,
        *,
        question: str,
        retriever: RetrieverLike,
        llm: StreamingLLMLike,
        prompt_template: TutorPromptTemplate,
        event_sink: EventSink,
        attempt_config: TutorAttemptConfig,
    ) -> TutorCandidate:
        raise NotImplementedError
```

- retrieval 只调用一次，top-k 与 5s/55s limits 由 `TutorAttemptConfig` 传入；Prompt template 不持有 retrieval config。
- engine 发出 citations、token、trace/budget events，返回 exact evidence。
- 同步 retrieval 超时后即使底层 thread 稍后完成，也不得 finalize Candidate。
- production 使用 `TutorAttemptConfig.production_default()` 保持当前无强制 deadline 行为；evaluation 使用 Registry 解析出的 top-k 与 5s/55s limits。
- engine 不 import Router、Judge、Memory、DB repository 或 FastAPI。

### Step 4: 让 Prompt 支持显式模板而不改变 v2 文本

在 `backend/app/agent/prompt.py`：

- 保留 `SYSTEM_INSTRUCTION` 字节内容不变。
- 新增 immutable `TutorPromptTemplate` 与 `render()`。
- `build_prompt()` 继续作为 production v2 compatibility wrapper。
- retry hint 由 Graph adapter 构造 runtime-only suffix；Eval Registry template 不带 retry suffix。

增加断言：

```python
assert TutorPromptTemplate.production_v2().render(query, chunks) == build_prompt(query, chunks)
```

### Step 5: 将 Graph tutor node 改为 thin adapter

在 `backend/app/agent/graph.py` 用 engine 替换 node 内重复的 retrieve/prompt/stream 逻辑；Graph 仍负责 retry_count、`_retry_hint`、Judge 和 state mapping：

```python
candidate = await tutor_attempt_engine.answer(
    question=last_user,
    retriever=retriever,
    llm=llm,
    prompt_template=template.with_suffix(retry_hint),
    event_sink=LangGraphEventSink(writer),
    attempt_config=TutorAttemptConfig.production_default(),
)
return {
    "messages": [AIMessage(content=candidate.answer)],
    "citations": candidate.citations,
    "last_context": candidate.formatted_context,
}
```

### Step 6: 添加调用次数与 parity tests

- Judge first-pass success：Attempt 调用 1 次。
- 两次 retry：Attempt 最多调用 3 次。
- 固定 fake token 的 Graph SSE 仍发 citations/token/done，拼接答案不变。
- 不断言真实模型 token 边界。

### Step 7: 运行 GREEN 与相关回归

```bash
cd backend && uv run pytest tests/agent/test_tutor_attempt.py tests/agent/test_graph.py tests/agent/test_graph_judge.py tests/api/test_routes_graph_stream.py -q
```

Expected GREEN: 全部通过；现有 streaming contract 无变化。

### Step 8: 审查架构边界

```bash
rg -n "Router|judge_response|Memory|Repository|Session|FastAPI" backend/app/agent/tutor_attempt.py
git diff --check
```

Expected: engine 文件不依赖这些 orchestration/persistence symbols。

**Approval-gated checkpoint:**

```bash
git add backend/app/agent/tutor_attempt.py backend/app/agent/prompt.py backend/app/agent/graph.py backend/tests/agent/test_tutor_attempt.py backend/tests/agent/test_graph.py backend/tests/agent/test_graph_judge.py
git commit -m "refactor: extract tutor attempt engine"
```

## Task 2: 建立 versioned Registry、12-case definitions 与隔离 CorpusSnapshot

**Files:**

- Create: `backend/app/eval/learning_run/__init__.py`
- Create: `backend/app/eval/learning_run/contracts.py`
- Create: `backend/app/eval/learning_run/registry.py`
- Create: `backend/app/eval/learning_run/corpus.py`
- Create: `backend/app/eval/learning_run/definitions/experiment.json`
- Create: `backend/app/eval/learning_run/definitions/task_cases.json`
- Create: `backend/app/eval/learning_run/definitions/corpus.json`
- Create: `backend/app/eval/learning_run/definitions/prompts/tutor-v2.txt`
- Create: `backend/app/eval/learning_run/definitions/prompts/tutor-v3.txt`
- Create: `backend/app/eval/learning_run/definitions/scorers/hybrid-v1.json`
- Create: `backend/app/eval/learning_run/definitions/calibration/candidates.json`
- Modify: `backend/app/rag/runtime.py`
- Create: `backend/tests/eval/test_learning_run_registry.py`
- Create: `backend/tests/eval/test_learning_run_corpus.py`

### Step 1: 写 Registry hash 与 allowlist RED tests

```python
def test_registry_resolves_ids_and_rejects_client_overrides():
    resolved = TaskRegistry.load_default().resolve_run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-004",
        variant_id="tutor-v3",
        run_profile="evaluation",
    )
    assert resolved.experiment_axes == ("prompt_version",)
    assert resolved.runtime_judge is False
    assert resolved.task.case_type == "expected_refusal"

def test_registry_has_twelve_cases_and_separate_calibration_ids():
    registry = TaskRegistry.load_default()
    assert len(registry.task_cases) == 12
    assert set(registry.task_cases).isdisjoint(registry.calibration_case_ids)
```

再测试 unknown ID、duplicate ID、hash mismatch、Prompt/path/expected answer/profile override 被拒绝。

### Step 2: 运行 RED

```bash
cd backend && uv run pytest tests/eval/test_learning_run_registry.py -q
```

Expected RED: learning_run Registry 尚不存在。

### Step 3: 定义强类型 contracts 与 canonical hash

在 `contracts.py` 定义 frozen dataclasses/Pydantic models：`TaskCase`、`CorpusChunk`、`CorpusSnapshot`、`PromptDefinition`、`ScorerBundle`、`ExperimentDefinition`、`ResolvedRunDefinition`、`RunManifest`。所有 hash 使用 UTF-8、sorted-key compact JSON；同一 helper 供 write/read verification 使用。

关键约束：

- case types 只能是 `answerable | multi_evidence | expected_refusal`。
- suite 分布严格为 6/3/3。
- `experiment_axes == ("prompt_version",)`。
- `runtime_judge == False`。
- `tutor-v2` 文本必须等于 `SYSTEM_INSTRUCTION`，测试逐字断言。
- scorer calibration IDs 与 regression suite IDs 不相交。

### Step 4: 写真实 version-controlled definitions

- 12 cases 必须写完整 question、人工理由、expected behavior、required evidence set、required dimensions、critical policy。
- Corpus 只放可再分发的精简内容，stable chunk IDs、source/page、exact text、per-chunk hash、aggregate hash 完整。
- `tutor-v3` 只改变 Prompt 文本；provider/model/parameters、retrieval、corpus、budget、schema 保持一致。
- budget definition 精确冻结为 retrieval/preflight 5s、Tutor 55s、Hybrid scoring 25s、total wall 90s。
- `tutor-v3` 在真实 suite evidence 完成并人工审查前不得替换 production default。
- calibration candidates 必须包含 Pass、Fail、borderline、正确拒答、错误拒答，且不是 12-case 运行结果。
- `hybrid-v1.json` 冻结 rubric anchors、parser version、verdict policy、scorer model config。

### Step 5: 写 isolation RED tests

在 `test_learning_run_corpus.py`：

```python
def test_eval_corpus_never_uses_global_retriever():
    global_retriever = ExplodingDependency("global retriever forbidden")
    loader = CorpusSnapshotLoader(builder=FakeIsolatedBuilder())

    retriever = loader.load(snapshot=registry.corpus, global_retriever=global_retriever)

    assert retriever.search("RRF", top_k=5)
```

覆盖缺 chunk、额外 chunk、content hash mismatch、aggregate mismatch、collection name 不等于 `study_coach_chunks`。

### Step 6: 暴露可复用 retriever builder

将 `backend/app/rag/runtime.py::_build_retriever` 重命名为公开 `build_retriever_for_collection()`；production `build_default_runtime()` 继续使用它。`CorpusSnapshotLoader` 建立独立 ephemeral Chroma collection，先验证 corpus hash，再 `add_chunks()`，不得读取 `app.state.retriever`。

### Step 7: 运行 GREEN 与 RAG 回归

```bash
cd backend && uv run pytest tests/eval/test_learning_run_registry.py tests/eval/test_learning_run_corpus.py tests/rag/test_runtime.py tests/rag/test_retriever.py tests/rag/test_hybrid_retriever.py tests/rag/test_reranking_retriever.py -q
```

### Step 8: 静态检查 suite 真值

```bash
cd backend && uv run python -m app.eval.learning_run.registry --validate
git diff --check
```

Expected: 输出 12 cases、6/3/3、两个 Prompt hashes、corpus aggregate hash、calibration count；不打印 secrets 或完整 endpoint。

**Approval-gated checkpoint:**

```bash
git add backend/app/eval/learning_run backend/app/rag/runtime.py backend/tests/eval/test_learning_run_registry.py backend/tests/eval/test_learning_run_corpus.py
git commit -m "feat: add versioned learning run registry"
```

## Task 3: 增加三个 eval tables 与 immutable/append-only repositories

**Files:**

- Modify: `backend/app/db/models.py`
- Create: `backend/alembic/versions/f4c9a1d2e7b6_learning_run_eval_tables.py`
- Create: `backend/app/eval/learning_run/repositories.py`
- Modify: `backend/tests/db/test_alembic.py`
- Create: `backend/tests/eval/test_learning_run_repositories.py`

> 实施前先运行 `uv run alembic heads`；只有当前单一 head 仍为 `7a52fe598fd1` 时才使用本计划指定的 `down_revision`。如果最新 `origin/main` 已增加 migration，停止并将 `down_revision` 更新为实际单一 head；不要制造 parallel heads。

### Step 1: 写 migration 与 repository RED tests

`test_alembic.py` 先断言 fresh head 只有以下三个新 eval tables：

```python
assert {"eval_runs", "eval_score_sets", "eval_scorer_executions"} <= tables
assert "eval_suite_executions" not in tables
```

`test_learning_run_repositories.py` 先断言：

- queued → running → terminal lifecycle 合法。
- CandidateArtifact conditional finalize 恰好一次。
- terminal Run 不可改 artifact。
- Historical re-score 只 append 新 ScoreSet，不覆盖旧 ScoreSet；单个 active ScoreSet 可做受限 lifecycle transition，并且 terminal output 只 finalize 一次。
- ScorerExecution row 只 append，创建后不可改写 output/error。
- manifest/artifact/input checksum mismatch 在 read、compare、re-score 前被拒绝。

### Step 2: 运行 RED

```bash
cd backend && uv run pytest tests/db/test_alembic.py tests/eval/test_learning_run_repositories.py -q
```

Expected RED: tables/models/repositories 不存在。

### Step 3: 定义 ORM 与 migration

在 `models.py` 增加 `EvalRun`、`EvalScoreSet`、`EvalScorerExecution`；migration 建立 FK、status checks、query indexes 与唯一约束。字段完整覆盖 spec 第 11 节：

```python
class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[str]
    experiment_id: Mapped[str]
    suite_execution_id: Mapped[str | None]
    task_case_id: Mapped[str]
    task_case_version: Mapped[str]
    variant_id: Mapped[str]
    run_profile: Mapped[str]
    lifecycle: Mapped[str]
    outcome: Mapped[str | None]
    operational_error_json: Mapped[dict | None]
    manifest_json: Mapped[dict]
    manifest_hash: Mapped[str]
    candidate_artifact_json: Mapped[dict | None]
    artifact_hash: Mapped[str | None]
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
```

ScoreSet 保存 scorer bundle version、artifact input hash、status、verdict、error、aggregate、findings、timestamps；ScorerExecution 保存 scorer ID/version、status、input hash、output/error、latency、usage。

- migration revision 固定为 `f4c9a1d2e7b6`，在前置 head 仍为 `7a52fe598fd1` 时设置 `down_revision = "7a52fe598fd1"`。
- 三个 eval tables 是 local-instance global artifacts，不增加会制造虚假多用户隔离印象的 `user_id` 列；访问控制由 local-mode gate + existing-user auth 负责。
- Run lifecycle 约束为 `queued | running | finished | cancelled`，finished outcome 为 `success | system_failed | timed_out | budget_exceeded`。
- ScoreSet status 约束为 `pending | running | completed | partial | failed | cancelled`；quality verdict 为 `pass | fail | inconclusive | not_evaluated`。
- ScorerExecution status 约束为 `success | failed | skipped`。

### Step 4: 实现 bounded repositories

- 所有 JSON 写入前 canonicalize/hash。
- `finalize_candidate(run_id, expected_lifecycle="running", candidate_artifact=artifact, artifact_hash=canonical_hash(artifact))` 用 conditional UPDATE，rowcount 必须为 1。
- terminal transition 使用 compare-and-set；cancel/finalize 只能一个成功。
- ScoreSet repository 只暴露 `claim_running()` 与 conditional `finalize_once()` / `cancel_once()`；terminal 后所有 output、verdict、findings 与 error immutable。Historical re-score 必须 `create()` 新 row，绝不 update 旧 row。
- ScorerExecution repository 只暴露 append/read，不暴露 update/delete output 的方法。
- 错误只保存 stable code 与 sanitized message。
- 对外只声明 application-level append-only and checksum-verified，不使用 tamper-proof/WORM 表述。

### Step 5: 运行 GREEN

```bash
cd backend && uv run pytest tests/db/test_alembic.py tests/eval/test_learning_run_repositories.py -q
```

### Step 6: fresh-DB 与 downgrade/upgrade 验证

增加自动化 test 覆盖：upgrade head、downgrade 前一 revision、再 upgrade head；确认 FK child-first 删除和 `PRAGMA foreign_key_check` 无结果。

```bash
cd backend && uv run pytest tests/db/test_alembic.py -q
git diff --check
```

**Approval-gated checkpoint:**

```bash
git add backend/app/db/models.py backend/alembic/versions/f4c9a1d2e7b6_learning_run_eval_tables.py backend/app/eval/learning_run/repositories.py backend/tests/db/test_alembic.py backend/tests/eval/test_learning_run_repositories.py
git commit -m "feat: persist learning run artifacts"
```

## Task 4: 实现 Hybrid Scoring、校准与不可回退的 verdict policy

**Files:**

- Create: `backend/app/eval/learning_run/scoring.py`
- Modify: `backend/app/eval/learning_run/contracts.py`
- Create: `backend/tests/eval/test_learning_run_scoring.py`
- Modify: `backend/app/eval/learning_run/definitions/scorers/hybrid-v1.json`
- Modify: `backend/app/eval/learning_run/definitions/calibration/candidates.json`

### Step 1: 写 verdict truth-table RED tests

使用 table tests 明确 required dimension、hard gate 与 evaluator failure：

```python
@pytest.mark.parametrize(
    ("case_type", "scores", "hard_findings", "failed_scorers", "verdict"),
    [
        ("answerable", {"groundedness": 4, "citation_entailment": 4, "coverage": 4}, [], [], "pass"),
        ("answerable", {"groundedness": 5, "citation_entailment": 3, "coverage": 5}, [], [], "fail"),
        ("expected_refusal", {"refusal_appropriateness": 4, "unsupported_claims": 4}, [], [], "pass"),
        ("answerable", {"groundedness": 5, "citation_entailment": 5, "coverage": 5}, ["citation_invalid"], [], "fail"),
        ("answerable", {}, [], ["scorer_parse_error"], "inconclusive"),
    ],
)
def test_verdict_policy(case_type, scores, hard_findings, failed_scorers, verdict):
    result = derive_verdict(
        case_type=case_type,
        dimension_scores=scores,
        hard_findings=hard_findings,
        failed_scorers=failed_scorers,
    )
    assert result.verdict == verdict
```

另测 noncritical finding 可与 Pass 共存但必须显示；usage unavailable 保持字符串/nullable，不写 0。

### Step 2: 运行 RED

```bash
cd backend && uv run pytest tests/eval/test_learning_run_scoring.py -q
```

Expected RED: scoring module 尚不存在。

### Step 3: 实现 deterministic scorers

最小集合：

- citation presence/integrity：引用编号、chunk ID、span、evidence membership。
- retrieval-empty finding。
- expected-refusal observable finding。
- deterministic hard gate 与 critical policy。

Deterministic scorer 只读 frozen TaskCase 与 CandidateArtifact，不调用 Registry current state 重新解释历史数据。

### Step 4: 实现独立 LLM scorer parser

- 不 import 或调用 `app.agent.judge._normalise` / `judge_response`。
- rubric 输出每个 required dimension 的 1–5 score、reasoning、findings。
- missing key、非 1–5、malformed JSON、timeout、exception 都创建 failed ScorerExecution。
- `derive_score_set()` 根据实际成功 executions 生成 `completed | partial | failed` 与 `pass | fail | inconclusive | not_evaluated`。
- required dimension 每项 >=4 才可 Pass；禁止平均分覆盖单项 3。

`ScoringService` contract 必须显式产出每个 scorer 的 execution draft：

```python
async def score(
    self,
    *,
    task: TaskCase,
    candidate: CandidateArtifact,
    scorer_bundle: ScorerBundle,
    on_execution: Callable[[ScorerExecutionDraft], None],
) -> ScoreSetResultDraft:
    raise NotImplementedError
```

每个 deterministic/LLM scorer 无论 success、failed 或 skipped 都调用一次 `on_execution`；callback 由 RunService 绑定到具体 `score_set_id` 并立即 append。ScoreSet 的 aggregate/verdict 只从这些 execution drafts 派生。

### Step 5: 校准 fixtures

对独立 candidates 跑 fake/frozen scorer outputs，断言人工标签：明确 Pass、明确 Fail、borderline=Fail、正确拒答=Pass、错误拒答=Fail。测试同时断言 calibration IDs 不出现在 12-case suite。

### Step 6: 运行 GREEN 与防复用检查

```bash
cd backend && uv run pytest tests/eval/test_learning_run_scoring.py tests/agent/test_judge.py -q
rg -n "agent\.judge|judge_response|_normalise" backend/app/eval/learning_run
git diff --check
```

Expected: grep 无生产 Judge parser 复用；现有 production Judge tests 保持原状。

**Approval-gated checkpoint:**

```bash
git add backend/app/eval/learning_run/scoring.py backend/app/eval/learning_run/contracts.py backend/app/eval/learning_run/definitions/scorers/hybrid-v1.json backend/app/eval/learning_run/definitions/calibration/candidates.json backend/tests/eval/test_learning_run_scoring.py
git commit -m "feat: add calibrated hybrid scoring"
```

## Task 5: 实现 TutorRunner 与 isolated one-attempt RunService tracer bullet

**Files:**

- Create: `backend/app/eval/learning_run/service.py`
- Create: `backend/app/eval/learning_run/runner.py`
- Create: `backend/tests/eval/test_learning_run_service.py`
- Create: `backend/tests/eval/test_learning_run_runner.py`
- Modify: `backend/app/eval/learning_run/contracts.py`
- Modify: `backend/app/eval/learning_run/repositories.py`

### Step 1: 写一条完整 service RED test

第一个 tracer bullet 必须从 Registry ID 一直走到 frozen CandidateArtifact 与 ScoreSet：

```python
@pytest.mark.asyncio
async def test_run_service_executes_one_isolated_attempt_and_freezes_scores(session):
    sentinels = ForbiddenDependencies(
        router=ExplodingDependency(),
        runtime_judge=ExplodingDependency(),
        memory=ExplodingDependency(),
        chat_repository=ExplodingDependency(),
        global_retriever=ExplodingDependency(),
    )
    runner = RecordingTutorRunner(candidate=ANSWERABLE_CANDIDATE)
    service = build_service(session=session, runner=runner, sentinels=sentinels)

    result = await service.run(
        experiment_id="tutor-prompt-regression-v1",
        task_case_id="tgqa-001",
        variant_id="tutor-v3",
        run_profile="evaluation",
        connection=matching_connection(),
        events=RecordingEventSink(),
    )

    assert runner.calls == 1
    assert result.run.outcome == "success"
    assert result.run.artifact_hash == canonical_hash(result.run.candidate_artifact_json)
    assert result.score_set.artifact_input_hash == result.run.artifact_hash
```

### Step 2: 运行 RED

```bash
cd backend && uv run pytest tests/eval/test_learning_run_runner.py tests/eval/test_learning_run_service.py -q
```

Expected RED: `RunService` 不存在；test doubles 本身必须可独立构造。

### Step 3: 实现 TutorRunner exactly-once boundary

`TutorRunner` 封装 isolated corpus、resolved retrieval options 与一次 Attempt：

```python
class TutorRunner:
    async def run(
        self,
        *,
        definition: ResolvedRunDefinition,
        llm: StreamingLLMLike,
        events: EventSink,
    ) -> TutorCandidate:
        retriever = self.corpus_loader.load(definition.corpus)
        return await self.attempt_engine.answer(
            question=definition.task.question,
            retriever=retriever,
            llm=llm,
            prompt_template=definition.prompt,
            event_sink=events,
            attempt_config=definition.attempt_config,
        )
```

`test_learning_run_runner.py` 断言 corpus loader 一次、Attempt exactly once、top-k/limits 来自 resolved definition，且 global retriever sentinel 零调用。

### Step 4: 实现明确的 service collaborators

`RunService` 只编排以下已注入职责：

```python
class RunService:
    def __init__(
        self,
        *,
        registry: TaskRegistry,
        tutor_runner: TutorRunner,
        scoring_service: ScoringService,
        runs: EvalRunRepository,
        score_sets: EvalScoreSetRepository,
        scorer_executions: EvalScorerExecutionRepository,
        clock: Clock,
    ) -> None:
        self.registry = registry
        self.tutor_runner = tutor_runner
        self.scoring_service = scoring_service
        self.runs = runs
        self.score_sets = score_sets
        self.scorer_executions = scorer_executions
        self.clock = clock
```

执行顺序固定为 resolve → validate connection/config → create manifest → TutorRunner exactly once → conditional artifact finalize → create running ScoreSet → scorers append each ScorerExecution → conditional ScoreSet finalize。Registry 解析出的 provider/model/parameters 是事实来源；header 只提供匹配的连接凭证。

`RunManifest` 必须冻结 task/corpus/Prompt/provider/model/parameters/retrieval/reranker/chunking/runner/scorer/schema/budget/runtime Judge profile/code revision，以及 provider 支持时的 seed；连接 endpoint 只保存不可逆 fingerprint，不保存完整 Base URL。历史详情、compare、re-score 读取该 frozen manifest，不用 current Registry 覆盖历史事实。

### Step 5: 显式映射 operational outcomes

测试并实现：

- manifest/corpus validation → `system_failed` + 对应 stable code。
- retriever/model exception → `system_failed`。
- generation deadline → `timed_out` + `generation_timeout`。
- scoring 25s deadline → failed scorer execution；证据不足时 ScoreSet Inconclusive，绝不 Pass。
- total wall 90s deadline → `budget_exceeded`。
- normal zero retrieval → successful execution + quality finding，而不是 operational failure。
- 任一 terminal path 记录 spent budget；usage 不可得时是 `unavailable`。
- contracts 固定支持 `manifest_invalid | corpus_unavailable | corpus_mismatch | retriever_error | model_unavailable | generation_timeout | scorer_timeout | scorer_parse_error | budget_exceeded | process_interrupted | harness_internal_error`，API 只暴露 sanitized message。

### Step 6: 证明 Eval exactly-once、每个 scorer 可审计与业务隔离

sentinel tests 必须分别让 Router、runtime Judge、Memory hydrator/writer、Chat repository、global retriever 在被调用时抛异常；成功 run 证明这些依赖零调用。再写失败 scorer test，证明 scorer 失败不会重新运行 Tutor，并断言 deterministic、LLM success/failure/skipped 各有独立 ScorerExecution row；ScoreSet aggregate 可由 rows 重建。

### Step 7: 运行 GREEN

```bash
cd backend && uv run pytest tests/eval/test_learning_run_runner.py tests/eval/test_learning_run_service.py tests/agent/test_tutor_attempt.py tests/eval/test_learning_run_scoring.py tests/eval/test_learning_run_repositories.py -q
git diff --check
```

**Approval-gated checkpoint:**

```bash
git add backend/app/eval/learning_run/service.py backend/app/eval/learning_run/runner.py backend/app/eval/learning_run/contracts.py backend/app/eval/learning_run/repositories.py backend/tests/eval/test_learning_run_runner.py backend/tests/eval/test_learning_run_service.py
git commit -m "feat: execute isolated learning runs"
```

## Task 6: 暴露 local-only authenticated SSE API 与 DB-backed single-flight

**Files:**

- Create: `backend/app/api/eval_routes.py`
- Create: `backend/app/api/eval_schemas.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/eval/learning_run/repositories.py`
- Modify: `backend/app/eval/learning_run/service.py`
- Create: `backend/tests/api/test_eval_routes.py`
- Create: `backend/tests/api/test_eval_schemas.py`
- Create: `backend/tests/eval/test_learning_run_races.py`
- Create: `contracts/eval-api-v1/examples/run-stream.jsonl`
- Create: `contracts/eval-api-v1/examples/run-detail.json`
- Create: `contracts/eval-api-v1/examples/compare-controlled.json`
- Create: `contracts/eval-api-v1/examples/evaluation-busy.json`

### Step 1: 冻结 API v1 DTO/event/error schema

先在 `eval_schemas.py` 定义 Pydantic response models 与 discriminated event union，并让 `test_eval_schemas.py` 读取 repo-root examples 验证：

- `ExperimentSummary`：experiment/task family、唯一 axis、variants、case counts、run profile、budgets。
- `RunSummary`：IDs、suite execution、case/variant、lifecycle/outcome、latest ScoreSet summary、timestamps。
- `RunDetail`：summary、frozen manifest、CandidateArtifact、all historical ScoreSets/ScorerExecutions、operational error。
- `CompareResponse`：compatibility、reasons、left/right summaries、scorer bundle、nullable delta、`case | suite` scope label。
- stream union：`run_created | stage_started | stage_completed | scorer_completed | scorer_failed | run_finished | score_set_created | score_set_finished`；所有 events 带 `schema_version="eval-api-v1"`，首 event 带 entity ID。
- error detail：stable `code/message`，`evaluation_busy` 额外带 `active_entity_id/active_kind`，config mismatch 只列非敏感 field names。

Example files 是 backend/frontend 共同的 schema fixtures，不包含真实 metrics 或 secrets。Backend route tests 与 Task 10 frontend parser tests 必须读取相同 examples，禁止两边各造一套样本。

### Step 2: 写 auth/local-mode/payload RED tests

`backend/tests/api/test_eval_routes.py` 先覆盖：

- 缺 Bearer、无对应 user row → 401。
- `STUDY_COACH_LOCAL_MODE != "1"` → 403 `evaluation_disabled`。
- request body 只接受四个 ID 字段，额外 Prompt/path/corpus/scorer/expected answer/runtime Judge 字段 → 422。
- provider/model/parameters 与 Registry 不匹配 → 409 `evaluation_config_mismatch`，不得 fallback。
- API key、Authorization、完整 Base URL 不出现在 response、DB 或 captured logs。

### Step 3: 写 SSE event contract RED test

```python
def test_run_stream_emits_id_first_and_one_terminal_event(client, auth, local_mode):
    events = read_sse(client.stream(
        "POST",
        "/api/eval/runs/stream",
        headers=auth | matching_llm_headers(),
        json=RUN_REQUEST,
    ))
    assert events[0]["type"] == "run_created"
    assert events[0]["run_id"]
    assert [event["type"] for event in events].count("run_finished") == 1
```

同时断言 stage/scorer 事件顺序与 sanitized error event。

成功事件序列至少支持 `run_created`、`stage_started`、`stage_completed`、`scorer_completed | scorer_failed`、`run_finished`；同一 stream 只能出现一个 terminal event。

### Step 4: 运行 RED

```bash
cd backend && uv run pytest tests/api/test_eval_schemas.py tests/api/test_eval_routes.py -q
```

Expected RED: `/api/eval/*` 返回 404。

### Step 5: 实现统一 dependency gate 与 routes

在 `eval_routes.py` 实现 spec 中完整路径，但本 task 先让 experiments、runs、detail、run stream 可用：

```text
GET  /api/eval/experiments
GET  /api/eval/runs
GET  /api/eval/runs/{run_id}
POST /api/eval/runs/stream
```

- 所有 route 依赖 `require_existing_user` 和 `require_local_eval_mode`。
- POST schema 使用 `extra="forbid"`。
- stream 格式沿用 `data: {json}\n\n`，首事件 `run_created` 必含 `run_id`。
- API factory 通过 `app.state` 或明确 dependency overrides 注入 Registry/Service，测试不得启动真实 embedder/model。
- `main.py` include `eval_router`；现有 lifecycle middleware 继续覆盖整个 SSE lease。

### Step 6: 写 DB source-of-truth single-flight RED race

两个独立 SQLite connections 和 `threading.Barrier`/controlled hook，禁止 `sleep`：

```python
def test_run_run_claim_allows_exactly_one_writer(two_connections, claim_barrier):
    results = race_claims(
        lambda repo: repo.claim_execution(kind="run", entity_id=new_id()),
        lambda repo: repo.claim_execution(kind="run", entity_id=new_id()),
    )
    assert sorted(result.status for result in results) == ["busy", "claimed"]
```

### Step 7: 实现原子 claim

- `BEGIN IMMEDIATE` 后，在同一 transaction 查询 active Run 和 active ScoreSet，再 insert/update claim。
- Run 与 re-score 共用此检查；不得新增第四个 lock/suite table。
- 进程内 `Lock` 只可作 accelerator，DB transaction/status 是事实来源。
- busy response 固定为 `409 evaluation_busy`，返回 `active_entity_id` 和 `active_kind`。
- GET list/detail 不取写锁。

### Step 8: 运行 GREEN

```bash
cd backend && uv run pytest tests/api/test_eval_schemas.py tests/api/test_eval_routes.py tests/eval/test_learning_run_races.py tests/api/test_learning_auth.py -q
git diff --check
```

**Approval-gated checkpoint:**

```bash
git add backend/app/api/eval_routes.py backend/app/api/eval_schemas.py backend/app/api/deps.py backend/app/main.py backend/app/eval/learning_run/repositories.py backend/app/eval/learning_run/service.py backend/tests/api/test_eval_schemas.py backend/tests/api/test_eval_routes.py backend/tests/eval/test_learning_run_races.py contracts/eval-api-v1/examples
git commit -m "feat: stream authenticated learning runs"
```

## Task 7: 定义 cancel、disconnect、race 与 startup reconciliation

**Files:**

- Modify: `backend/app/api/eval_routes.py`
- Modify: `backend/app/eval/learning_run/service.py`
- Modify: `backend/app/eval/learning_run/repositories.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/api/test_eval_routes.py`
- Modify: `backend/tests/eval/test_learning_run_races.py`

### Step 1: 写 cancel/generator-close RED tests

覆盖以下不同机制，不把它们合并成一个“disconnect test”：

- `POST /api/eval/runs/{run_id}/cancel` 幂等，两次都返回同一 terminal state。
- 直接对 SSE async generator `aclose()` / 注入 `CancelledError`，run 变为 cancelled，释放 claim/lease。
- cancel 之后 Candidate 不 finalize、scorer 不继续追加。
- 已完成 run 的 cancel 不改写 finished outcome。

### Step 2: 写可控 race RED tests

在 `test_learning_run_races.py` 使用两个 connections 与 barrier/hook，参数化两种提交顺序：

- cancel / finalize。
- run / run。
- run / re-score。
- startup reconciliation / new run。
- reset / run（最终断言在 Task 9 接入 reset 后完成）。

每个 race 断言一个且仅一个 terminal state、一个 CandidateArtifact、claim 最终释放；禁止基于 wall-clock sleep。

### Step 3: 运行 RED

```bash
cd backend && uv run pytest tests/api/test_eval_routes.py tests/eval/test_learning_run_races.py -q
```

Expected RED: cancel routes/reconciliation 尚不存在，或竞态产生双 terminal。

### Step 4: 实现 cooperative cancellation

- process-local cancellation token 只负责当前 attached execution 快速检查。
- durable cancel 意图/terminal state 通过 conditional DB transition 记录。
- Runner 在 stage 边界和 token 边界检查 cancellation。
- generator cancellation 必须进入 `finally`，尝试取消未完成 run/score set 并释放 claim。
- refresh/network disconnect 不承诺重连、续跑或 replay；其终态是 cancelled 或可审计的 process interruption。

### Step 5: 启动时 reconciliation

`create_app()` 在 migration 后、服务可接收请求前：

- 将遗留 `queued/running` Run 标记 terminal，code=`process_interrupted`。
- 将遗留 `pending/running` ScoreSet 标记 failed/cancelled policy 指定的 terminal，code=`process_interrupted`。
- 不创建新 Candidate/ScoreSet，不重新运行 Tutor。

### Step 6: 运行 GREEN

```bash
cd backend && uv run pytest tests/api/test_eval_routes.py tests/eval/test_learning_run_races.py tests/api/test_routes_graph_stream.py -q
git diff --check
```

**Manual socket acceptance deferred to Task 11:** TestClient generator close 不能代替真实浏览器 refresh/network disconnect 证据。

**Approval-gated checkpoint:**

```bash
git add backend/app/api/eval_routes.py backend/app/eval/learning_run/service.py backend/app/eval/learning_run/repositories.py backend/app/main.py backend/tests/api/test_eval_routes.py backend/tests/eval/test_learning_run_races.py
git commit -m "feat: cancel and reconcile learning runs"
```

## Task 8: 增加 historical re-score、controlled compare 与 atomic suite import

**Files:**

- Create: `backend/app/eval/learning_run/compare.py`
- Create: `backend/scripts/import_learning_run_suite.py`
- Create: `backend/app/eval/learning_run/definitions/scorers/hybrid-v2.json`
- Create: `backend/app/eval/learning_run/definitions/calibration/hybrid-v2-labels.json`
- Modify: `backend/app/eval/learning_run/registry.py`
- Modify: `backend/app/eval/learning_run/service.py`
- Modify: `backend/app/api/eval_routes.py`
- Create: `backend/tests/eval/test_learning_run_compare.py`
- Create: `backend/tests/eval/test_learning_run_import.py`
- Modify: `backend/tests/eval/test_learning_run_registry.py`
- Modify: `backend/tests/eval/test_learning_run_scoring.py`
- Modify: `backend/tests/eval/test_learning_run_service.py`
- Modify: `backend/tests/api/test_eval_routes.py`

### Step 1: 写 pure compatibility RED tests

`compare.py` 输入两个 frozen manifests/artifacts/ScoreSets，输出：

- Controlled：只 `prompt_version` 不同，scorer versions 相同，允许 case delta。
- Informational：未声明 config 差异，只并排显示。
- Incompatible：task/corpus/artifact schema 无法对齐。
- scorer version 不同：不显示 score delta，并返回 `rescore_required=true`。
- single case copy 只能写 `case delta`，不得写 suite/general quality。

### Step 2: 写 re-score RED test

```python
@pytest.mark.asyncio
async def test_rescore_reads_frozen_artifact_and_never_calls_tutor(service):
    service.attempt_engine = ExplodingDependency("Tutor must not run")
    new_score_set = await service.rescore(run_id=FROZEN_RUN_ID, scorer_bundle="hybrid-v2")
    assert new_score_set.run_id == FROZEN_RUN_ID
    assert new_score_set.artifact_input_hash == FROZEN_ARTIFACT_HASH
    assert len(service.score_sets.list_for_run(FROZEN_RUN_ID)) == 2
```

并覆盖 artifact hash mismatch、cancel、first event 含 `score_set_id`、与 live Run 共享 single-flight。

`hybrid-v2` 必须是 Registry 中真实存在的 scorer bundle，而非 test-only 字符串：保持 deterministic gates、required dimensions 与 verdict threshold 不变，只版本化收紧 LLM rubric 的 atomized-claim 和 expected-refusal anchors，并升级 parser/rubric version。使用同一组独立 calibration candidates 加 `hybrid-v2-labels.json` 人工标签校准；任何 anchor 未达到人工标签时，先修 rubric/parser，不得调整 12-case expected behavior。

### Step 3: 写 suite import rollback RED tests

CLI 输入 JSONL export，在一个 transaction 中验证 Registry expected cases/variants/checksums：

- 完整有效 fixture 原子导入。
- 缺失、重复、unknown version、hash mismatch 任一出现时 rows 增量为 0。
- 同一 `suite_execution_id` 分组，但不新增 `eval_suite_executions` table。

### Step 4: 运行 RED

```bash
cd backend && uv run pytest tests/eval/test_learning_run_registry.py tests/eval/test_learning_run_scoring.py tests/eval/test_learning_run_compare.py tests/eval/test_learning_run_import.py tests/eval/test_learning_run_service.py tests/api/test_eval_routes.py -q
```

### Step 5: 实现 compare/re-score/import 与剩余 API

补齐：

```text
POST /api/eval/runs/{run_id}/cancel
POST /api/eval/runs/{run_id}/rescore/stream
POST /api/eval/score-sets/{score_set_id}/cancel
GET  /api/eval/compare?left={id}&right={id}
```

Import script 使用现有 `DATABASE_URL`/session factory，但在写入前完整解析文件并在单一 DB transaction 内验证与写入；不得写入 Registry definitions 或绕过 repository hash checks。

### Step 6: 运行 GREEN

```bash
cd backend && uv run pytest tests/eval/test_learning_run_registry.py tests/eval/test_learning_run_scoring.py tests/eval/test_learning_run_compare.py tests/eval/test_learning_run_import.py tests/eval/test_learning_run_service.py tests/api/test_eval_routes.py -q
uv run python scripts/import_learning_run_suite.py --help
cd ..
git diff --check
```

**Approval-gated checkpoint:**

```bash
git add backend/app/eval/learning_run/compare.py backend/scripts/import_learning_run_suite.py backend/app/eval/learning_run/definitions/scorers/hybrid-v2.json backend/app/eval/learning_run/definitions/calibration/hybrid-v2-labels.json backend/app/eval/learning_run/registry.py backend/app/eval/learning_run/service.py backend/app/api/eval_routes.py backend/tests/eval/test_learning_run_registry.py backend/tests/eval/test_learning_run_scoring.py backend/tests/eval/test_learning_run_compare.py backend/tests/eval/test_learning_run_import.py backend/tests/eval/test_learning_run_service.py backend/tests/api/test_eval_routes.py
git commit -m "feat: compare and rescore learning runs"
```

## Task 9: 将 Eval 正确接入 Data Summary 与 factory reset

**Files:**

- Modify: `backend/app/db/repositories.py`
- Modify: `backend/app/data_lifecycle.py`
- Modify: `backend/app/api/data_routes.py`
- Modify: `backend/tests/db/test_data_lifecycle_repository.py`
- Modify: `backend/tests/test_data_lifecycle.py`
- Modify: `backend/tests/api/test_data_routes.py`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/dataLifecycle.ts`
- Modify: `frontend/src/lib/resetClientState.test.ts`

### Step 1: 写 nested eval counts RED tests

不要把 eval counts 混进现有 learning `DataCounts` 和前端 learning-reset toast 总和。API contract 使用独立对象：

```json
{
  "users": 1,
  "documents": 0,
  "has_learning_data": false,
  "eval": {
    "runs": 2,
    "score_sets": 3,
    "scorer_executions": 9,
    "estimated_bytes": 18432
  }
}
```

Reset response 使用 `deleted` 保存原 learning counts，另用 `deleted_eval` 保存 eval counts。先测试：

- 只有 eval rows 时 `has_learning_data is False`。
- learning reset 后 eval counts/rows 不变，`deleted_eval` 全 0。
- factory reset 后 child-first 清空三表，`deleted_eval` 返回 reset 前数量。
- 前端 learning reset success toast 不把 preserved eval counts 加入删除总数。

### Step 2: 写 reset race RED tests

使用 Task 7 的两个 connections + barrier/hook，覆盖：

- run 先 claim：factory reset 得到既有 `data_operation_in_progress`。
- reset 先取得 exclusive gate：run 不创建任何 row，返回稳定 conflict。
- learning reset 期间/之后 eval artifacts 完整保留。

### Step 3: 运行 RED

```bash
cd backend && uv run pytest tests/db/test_data_lifecycle_repository.py tests/test_data_lifecycle.py tests/api/test_data_routes.py tests/eval/test_learning_run_races.py -q
cd ../frontend && pnpm exec vitest run src/lib/resetClientState.test.ts
cd ..
```

Expected RED: API schema 没有独立 eval counts，factory reset 未清 eval tables。

### Step 4: 实现 child-first reset 与独立 summary

- `DataLifecycleRepository.count_eval()` 与 `delete_eval_data()` 分开，不扩展现有 learning delete loop 的语义。
- `ResetCoordinator.summary()` 计算 `has_learning_data` 时只看原 learning counts + vectors。
- factory path 在删除 User 前先调用 eval child-first delete。
- `estimated_bytes` 只计算 eval JSON/text payload 长度并明确命名为 estimate。
- 不更改 `require_signed_user` 用于 reset retry 的现有例外；Eval routes 本身继续使用 `require_existing_user`。

### Step 5: 更新 frontend lifecycle types

`frontend/src/lib/api.ts` 只更新 summary/reset response types 与 parsing；`resetClientLearningState()` 不清 Run Lab store，因为 learning reset 明确保留 eval artifacts。factory reset 触发页面 reload，新的 Run store 在 Task 10 注册 browser-state cleanup。

### Step 6: 运行 GREEN

```bash
cd backend && uv run pytest tests/db/test_data_lifecycle_repository.py tests/test_data_lifecycle.py tests/api/test_data_routes.py tests/eval/test_learning_run_races.py -q
cd ../frontend && pnpm exec vitest run src/lib/resetClientState.test.ts src/lib/dataLifecycle.test.ts src/stores/dataLifecycle.test.ts
cd ..
git diff --check
```

**Approval-gated checkpoint:**

```bash
git add backend/app/db/repositories.py backend/app/data_lifecycle.py backend/app/api/data_routes.py backend/tests/db/test_data_lifecycle_repository.py backend/tests/test_data_lifecycle.py backend/tests/api/test_data_routes.py frontend/src/lib/api.ts frontend/src/lib/dataLifecycle.ts frontend/src/lib/resetClientState.test.ts
git commit -m "feat: include eval artifacts in factory reset"
```

## Task 10: 实现前端 authenticated fetch stream 与中央 attached-run store

**Files:**

- Create: `frontend/src/lib/evalApi.ts`
- Create: `frontend/src/lib/evalApi.test.ts`
- Create: `frontend/src/lib/evalContracts.ts`
- Create: `frontend/src/lib/evalContracts.test.ts`
- Create: `frontend/src/lib/learningRunPresentation.ts`
- Create: `frontend/src/lib/learningRunPresentation.test.ts`
- Create: `frontend/src/stores/learningRuns.ts`
- Create: `frontend/src/stores/learningRuns.test.ts`
- Modify: `frontend/src/lib/dataLifecycle.ts`
- Modify: `frontend/src/lib/resetClientState.test.ts`

### Step 1: 写 stream parser 与 token race RED tests

`evalApi.test.ts` 使用 fake fetch/ReadableStream 覆盖：

- 调用 POST 前 `await getAccessToken()`，首个请求不会因模块启动 fire-and-forget 产生 401 race。
- 精确发送 Bearer 与 Registry matching provider/model/connection headers。
- split JSON frame、同 chunk 多 frame、malformed frame、terminal frame。
- 409 `evaluation_busy` 保留 active ID/kind，不压扁成 `failed: 409`。
- payload 不包含 Prompt、path、expected answer、API key body 或 runtime Judge override。
- `evalContracts.test.ts` 读取 `contracts/eval-api-v1/examples/*`，证明 TypeScript discriminated unions 接受所有 backend-approved examples；unknown `schema_version`/event type fail closed。

### Step 2: 写 store lifecycle RED tests

```typescript
it('keeps an attached run alive across route changes', async () => {
  const harness = createStoreHarness()
  await harness.store.start(RUN_REQUEST)
  harness.route.unmount()
  expect(harness.controller.signal.aborted).toBe(false)
})

it('calls cancel endpoint before aborting the stream', async () => {
  const harness = createStoreHarness()
  await harness.store.cancelActive()
  expect(harness.calls).toEqual(['cancel-endpoint', 'abort-stream'])
})
```

另覆盖 pagehide best-effort cancel + abort、network error、terminal 后不再 mutation、旧 stream 不可写入新 run、busy response 导航到 active entity 但不假装重连 stream。

### Step 3: 运行 RED

```bash
cd frontend && pnpm exec vitest run src/lib/evalContracts.test.ts src/lib/evalApi.test.ts src/stores/learningRuns.test.ts src/lib/learningRunPresentation.test.ts
```

Expected RED: modules 不存在。

### Step 4: 实现 transport

`evalContracts.ts` 显式镜像 `eval-api-v1` DTO/event/error union；`evalApi.ts` 独立于现有 Chat stream：

```typescript
export async function streamLearningRun(
  request: LearningRunRequest,
  connection: EvalConnectionSnapshot,
  onEvent: (event: LearningRunEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const token = await getAccessToken()
  const response = await fetch('/api/eval/runs/stream', {
    method: 'POST',
    headers: evalHeaders(token, connection),
    body: JSON.stringify(request),
    signal,
  })
  // incremental `data:` frame parsing with typed errors
}
```

不要把 Eval events 混入 Chat store/API，也不要使用 EventSource。

### Step 5: 实现中央 Pinia store

store 持有 active run/score set IDs、events、status、error、`AbortController` 和 captured connection snapshot。规则：

- route component unmount 不 abort。
- explicit cancel await 幂等 endpoint 后 abort。
- pagehide 使用 authenticated `fetch(..., {keepalive: true})` best-effort cancel，再 abort；承认 browser delivery 不保证。
- refresh 后不 resume；重新读取 detail 会看到 cancelled/process_interrupted historical state。
- 30s 只暴露 `canOpenHistoricalFallback`，不自动替换当前 run。
- factory browser clear 移除仅用于导航的 active-run ID；不影响 DB artifacts。

### Step 6: 实现纯 presentation policy

`learningRunPresentation.ts` 统一：

- verdict/finding/operational error 文案。
- scorer mismatch 时隐藏 delta。
- Controlled/Informational/Incompatible badges。
- usage unavailable 与 case/suite claim 边界。

这些 pure functions 用 Vitest 测；不新增 `@vue/test-utils` dependency。

### Step 7: 运行 GREEN

```bash
cd frontend && pnpm exec vitest run src/lib/evalContracts.test.ts src/lib/evalApi.test.ts src/stores/learningRuns.test.ts src/lib/learningRunPresentation.test.ts src/lib/resetClientState.test.ts
pnpm build
cd ..
git diff --check
```

**Approval-gated checkpoint:**

```bash
git add frontend/src/lib/evalContracts.ts frontend/src/lib/evalContracts.test.ts frontend/src/lib/evalApi.ts frontend/src/lib/evalApi.test.ts frontend/src/lib/learningRunPresentation.ts frontend/src/lib/learningRunPresentation.test.ts frontend/src/stores/learningRuns.ts frontend/src/stores/learningRuns.test.ts frontend/src/lib/dataLifecycle.ts frontend/src/lib/resetClientState.test.ts
git commit -m "feat: manage attached learning runs"
```

## Task 11: 构建 Run Lab 三页面与 Evidence Console v2

**Files:**

- Create: `frontend/src/views/RunLab.vue`
- Create: `frontend/src/views/RunDetail.vue`
- Create: `frontend/src/views/RunCompare.vue`
- Create: `frontend/src/components/run-lab/RunContractBar.vue`
- Create: `frontend/src/components/run-lab/RunEvidenceTimeline.vue`
- Create: `frontend/src/components/run-lab/RunScorePanel.vue`
- Create: `frontend/src/components/run-lab/RunTraceDrawer.vue`
- Modify: `frontend/src/router.ts`
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/App.lifecycle.test.ts`
- Modify: `frontend/src/components/MobileNav.vue`
- Modify: `frontend/src/components/MobileNav.test.ts`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/zh-CN.json`
- Create: `frontend/src/views/RunLab.contract.test.ts`

### Step 1: 写 route/source/presentation RED contract

项目当前 Vitest 是 node environment 且没有 component-test dependency，因此本 task 不新增依赖。测试读取 Vue source 并结合 Task 10 pure presentation policy，断言：

- 三个 route 精确存在。
- desktop/mobile navigation 都能进入 Run Lab。
- Run detail 组件引用 Contract、Evidence、Score 与 Trace drawer。
- historical fallback 按钮文案明确 `completed historical run`，不会调用替换当前 run ID 的 helper。
- scorer mismatch/incompatible 通过 pure policy 不显示 delta。
- citation hard gate 失败时 pure policy 不可能返回 Pass。

### Step 2: 运行 RED

```bash
cd frontend && pnpm exec vitest run src/views/RunLab.contract.test.ts src/App.lifecycle.test.ts src/components/MobileNav.test.ts src/lib/learningRunPresentation.test.ts
```

Expected RED: routes/views/components 不存在。

### Step 3: 实现 `/run-lab` Experiments / Runs

页面展示：

- frozen experiment axis、12-case suite summary、regressions、inconclusive、history filters。
- 单 case live controls；完整 suite 只显示已导入结果，不提供浏览器 batch button。
- backend 返回 `evaluation_disabled` 时显示 local-mode boundary，不尝试绕过。
- 至少一个 regression 只来自真实 imported artifacts；没有数据时显示 empty state，不显示 placeholder metrics。

### Step 4: 实现 `/run-lab/runs/:runId` Evidence Console v2

保留已批准布局：

- 顶部 Contract：Task、Corpus、Prompt、Model、RunProfile、budget、compatibility。
- 左侧 Execution Evidence：stages、exact retrieved chunks、answer claims/citations。
- 右侧 Verdict、required dimension scores、Quality Findings、baseline summary。
- full trace/provenance/unavailable usage 进入 drawer。
- 桌面双栏；窄屏按 Contract → Verdict → Evidence → trace 折叠顺序，禁止横向溢出。

### Step 5: 实现 `/run-lab/compare`

- Controlled 才显示 score delta。
- Informational 显示 config difference 列表。
- Incompatible 显示原因，不渲染误导性 delta。
- case view 只称 `case delta`；suite summary 明确限定 frozen 12-case suite。

### Step 6: 接入 Run controls 与 honest fallback

- Start 使用中央 store，不由 view 持有 stream。
- Cancel 使用 store 的 endpoint-then-abort contract。
- Run detail 显示 append-only historical ScoreSets；Re-score 是次要操作，启动后与 live Run 共用 single-flight，Cancel 调用 score-set cancel endpoint。五分钟 demo 主路径展示已存在的 ScoreSets，不现场等待 re-score。
- route switch 后详情页可继续观察 store events。
- 30 秒后按钮只打开明确标识、不同 ID 的 comparable completed run；当前 run 保持自己的 terminal/history。

### Step 7: 运行 GREEN 与 build

```bash
cd frontend && pnpm exec vitest run src/lib/evalContracts.test.ts src/views/RunLab.contract.test.ts src/lib/evalApi.test.ts src/stores/learningRuns.test.ts src/lib/learningRunPresentation.test.ts src/App.lifecycle.test.ts src/components/MobileNav.test.ts
pnpm build
cd ..
git diff --check
```

### Step 8: 真实浏览器 acceptance

使用项目已批准的 browser testing workflow，保存 evidence：

- 1440px、1024px、390px viewport 无 clipping/overflow，V2 信息层级可读。
- route switch 不取消 active run。
- explicit cancel 产生 durable cancelled state。
- refresh 与 simulated network disconnect 不产生假 success；reload 后显示 cancelled 或 process_interrupted。
- 30s fallback 显示不同 historical Run ID。
- exact evidence、verdict、findings、baseline delta 无互相矛盾。

**Approval-gated checkpoint:**

```bash
git add frontend/src/views/RunLab.vue frontend/src/views/RunDetail.vue frontend/src/views/RunCompare.vue frontend/src/components/run-lab frontend/src/router.ts frontend/src/App.vue frontend/src/App.lifecycle.test.ts frontend/src/components/MobileNav.vue frontend/src/components/MobileNav.test.ts frontend/src/locales/en.json frontend/src/locales/zh-CN.json frontend/src/views/RunLab.contract.test.ts
git commit -m "feat: add learning run lab"
```

## Task 12: 生成真实 suite evidence、同步文档并执行 release gates

**Files:**

- Create ignored runtime output: `backend/app/eval/learning_run/output/tutor-prompt-regression-v1.jsonl`
- Create committed curated fixture: `backend/app/eval/learning_run/fixtures/tutor-prompt-regression-v1.jsonl`
- Create: `backend/scripts/curate_learning_run_fixture.py`
- Create: `backend/tests/eval/test_learning_run_release_contract.py`
- Modify: `.gitignore`
- Add or update: `CONTEXT.md`
- Add or update: `docs/adr/0001-share-tutor-attempt-not-production-orchestration.md`
- Add or update: `docs/superpowers/specs/2026-08-12-learning-run-harness-design.md`
- Add or update: `docs/superpowers/plans/2026-08-12-learning-run-harness.md`
- Modify: `README.md`
- Modify: `docs/EVAL.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/DEMO.md`
- Modify: `docs/ROADMAP.md`
- Modify: `AGENTS.md` only if its current implementation worktree version needs the approved context/ADR read rule

### Step 1: 写 release-contract RED test

自动化只检查可稳定验证的发布合同，不把真实 runtime metrics 写入 Git：

```python
def test_release_definitions_cover_frozen_suite_and_versions():
    registry = TaskRegistry.load_default()
    assert registry.case_type_counts == {
        "answerable": 6,
        "multi_evidence": 3,
        "expected_refusal": 3,
    }
    assert registry.experiment_axes == ("prompt_version",)
    assert registry.prompt("tutor-v2").text == SYSTEM_INSTRUCTION
    assert registry.calibration_case_ids.isdisjoint(registry.task_case_ids)
```

另用 CLI importer 将 committed curated fixture 原子导入 temporary SQLite，并断言：

- 24 Runs = 12 cases × 2 Prompt variants，且每个 manifest/artifact hash 有效。
- 3 个 expected-refusal cases 两个 variants 都存在。
- 至少一个真实 regression。
- 至少一个 frozen Candidate 同时有 `hybrid-v1` 与 `hybrid-v2` ScoreSets。
- 每个 ScoreSet 的 ScorerExecution rows 完整，可重建 verdict。
- fixture 不含 Authorization、API key、完整 Base URL、私人输入或任意非 Registry corpus。

再检查 README/EVAL/ARCHITECTURE 内部 links、CLI command 和 version constants；不做脆弱全文 snapshot 或固定测试数量断言。

### Step 2: 运行 RED

```bash
cd backend && uv run pytest tests/eval/test_learning_run_release_contract.py -q
```

Expected RED: docs/link/fixture contract 尚未全部接入。

### Step 3: 人工签核 definitions

逐案检查 12 cases 的 question、expected behavior、required evidence 与理由；逐项检查独立 calibration anchors。任何错误先修正 definitions/hash/tests，再运行 suite；不得事后为了漂亮结果改 expected behavior。

### Step 4: 执行真实 v2/v3 suite

在 local mode、固定 provider/model/config 下真实运行 12 × 2 variants，保存：

- RunManifest、Prompt/Corpus/artifact hashes。
- 24 CandidateArtifacts 和对应 ScoreSets。
- expected-refusal case evidence。
- 至少一个真实 regression；若没有 regression，诚实展示零 regression，不手工制造，但 Completion Criteria 仍不满足，需要重新评估 `tutor-v3` 作为 demo candidate。
- 同一 frozen Candidate 的 scorer-v1/v2 historical ScoreSets。

真实 raw export 写入 ignored `backend/app/eval/learning_run/output/`，不 commit。`curate_learning_run_fixture.py` 必须 fail closed：验证 24 Runs/Registry IDs/hashes、移除 allowlist 外字段、拒绝 secret patterns 与非 Registry corpus，并以 stable ordering/canonical JSONL 输出。运行：

```bash
cd backend
fixture_check_dir=$(mktemp -d)
uv run python scripts/curate_learning_run_fixture.py \
  app/eval/learning_run/output/tutor-prompt-regression-v1.jsonl \
  app/eval/learning_run/fixtures/tutor-prompt-regression-v1.jsonl
uv run python scripts/import_learning_run_suite.py \
  app/eval/learning_run/fixtures/tutor-prompt-regression-v1.jsonl \
  --database-url "sqlite:///$fixture_check_dir/learning-run-fixture-check.db"
cd ..
```

从 raw export deterministic 生成、脱敏、可再分发的 curated fixture 作为 fresh clone 的预置 demo seed 与 release-test input 进入 Git。禁止人工编辑 metrics/artifacts；temporary DB path 在实际执行时使用 `mktemp -d` 解析后的显式路径，不覆盖工作区 DB。

### Step 5: 同步规格与用户可见文档

- `.gitignore` 只 unignore 本 spec 与本 plan；继续 ignore runtime output。
- `CONTEXT.md` 与 ADR 反映最终代码事实。
- README 只声明 frozen 12-case suite 上的结果，不泛化 Tutor 总体质量。
- EVAL 记录真实 run config、hashes、日期、结果与 limits。
- ARCHITECTURE 记录 Graph/Attempt/Eval 边界、三表、single-worker contract。
- DEMO 使用五分钟路径与 historical fallback。
- DEMO 给出从 fresh clone 导入 committed curated fixture 的精确 CLI；不得要求访问被 ignore 的本机 raw output。
- ROADMAP 将已完成/延期边界准确更新；保留 background queue、multi-worker、custom corpus、Plan/Quiz eval 为 deferred。
- 不复用当前 dirty checkout 的过时数字；所有 test counts 从本次 fresh commands 取得。

### Step 6: 运行 focused release GREEN

```bash
cd backend && uv run pytest tests/eval/test_learning_run_release_contract.py tests/eval tests/api/test_eval_routes.py tests/agent/test_tutor_attempt.py tests/agent/test_graph_judge.py tests/db/test_alembic.py tests/api/test_data_routes.py -q
cd ../frontend && pnpm exec vitest run src/lib/evalContracts.test.ts src/lib/evalApi.test.ts src/stores/learningRuns.test.ts src/lib/learningRunPresentation.test.ts src/views/RunLab.contract.test.ts src/components/MobileNav.test.ts
pnpm build
cd ..
```

### Step 7: 运行完整 final gates

```bash
cd backend
uv run pytest
cd ../frontend
pnpm exec vitest run
pnpm build
cd ..
docker compose config --quiet
git diff --check
git status --short --branch
```

另运行：

```bash
cd backend
uv run alembic heads
uv run pytest tests/db/test_alembic.py -q
cd ..
```

Expected: 单一 Alembic head、fresh migration tests、完整 backend/frontend tests、build、Compose render、diff check 全绿。记录真实 totals 与 exit codes，但不将 totals 作为未来静态合同。

### Step 8: 完成 manual evidence checklist

- 浏览器 route switch、cancel、refresh、network disconnect。
- 三个 viewport 的 V2 layout。
- expected-refusal 与至少一个 regression。
- scorer-v1/v2 historical ScoreSets。
- historical fallback 使用不同 Run ID。
- API/DB/export 无 secret。
- 五分钟演示在实际环境内完成；超过时间的环节记录并修订 DEMO，而不是隐藏。

### Step 9: 独立 review

在创建 PR 前派发只读 reviewer，范围只包括：approved spec coverage、Graph parity、eval isolation、malformed scorer policy、three-table migration/reset order、race tests、API secret handling、Run Lab truthful copy。主 agent 检查 reviewer findings、完整 diff 与 final gates；reviewer 不直接修改或 commit。

**Approval-gated checkpoint:**

```bash
git add .gitignore CONTEXT.md docs/adr/0001-share-tutor-attempt-not-production-orchestration.md docs/superpowers/specs/2026-08-12-learning-run-harness-design.md docs/superpowers/plans/2026-08-12-learning-run-harness.md README.md docs/EVAL.md docs/ARCHITECTURE.md docs/DEMO.md docs/ROADMAP.md backend/app/eval/learning_run/fixtures/tutor-prompt-regression-v1.jsonl backend/scripts/curate_learning_run_fixture.py backend/tests/eval/test_learning_run_release_contract.py
git commit -m "docs: publish learning run harness evidence"
```

不要 `git add` `backend/app/eval/learning_run/output/`、SQLite、Chroma、API keys 或浏览器临时 artifacts。

## Deferred scope

本 MVP 明确不实现：

- browser batch runner、arbitrary Task/Prompt/Corpus/Scorer editor。
- custom/private student inputs 进入 eval artifacts。
- background worker、job queue、disconnect resume、event replay。
- multi-worker/multi-process coordinator 或 cloud multi-user ownership。
- production-fidelity profile、first-vs-final Judge correction analysis。
- Plan/Quiz eval migration、Langfuse/Inspect/Harbor/DSPy runtime dependency。
- filesystem blob、WORM/tamper-proof storage、`eval_suite_executions` table。
- 统计显著性或对总体学习效果的结论。

任何一个 deferred item 变成必要条件时，停止实施并先更新 spec/ADR、风险与验收标准，取得批准后再继续。
