# Learning Run Harness — Design Spec

> 日期：2026-08-12
> 状态：设计已确认；尚未授权代码实现
> 目标：让 Study Coach 在技术面试中可证明其对 Agent harness、eval、trace、provenance、budget 和 regression detection 的掌握，而不偏离 AI 学习教练主题。

## 1. 决策摘要

Study Coach 将新增一个内置、轻量的 **Learning Run Harness**。第一版只支持 `Tutor Grounded QA`，使用 12 个冻结且人工确认的 regression cases，对唯一实验轴 `prompt_version: tutor-v2 → tutor-v3` 进行受控比较。

产品采用 Hybrid Demo：预置完整 suite 结果，再现场运行一个代表性 expected-refusal case。Run Lab 展示冻结的执行契约、exact retrieved evidence、Tutor trace、Hybrid Score、Quality Findings、baseline case delta 和 historical ScoreSets。

第一版明确不建设通用 eval 平台、后台 job queue、跨 worker 调度、Prompt 编辑器、任意语料上传、生产流量采集或 Multi-Agent Coach。

## 2. 产品目标与成功信号

### 2.1 目标用户

第一优先级是技术面试官和代码 reviewer；学生端价值是间接的：系统能更可靠地检测 Tutor Prompt 变化造成的 hallucination、引用错误与拒答退化。

### 2.2 面试官应在五分钟内理解

1. 实验输入与唯一变量受控。
2. Tutor 实际使用的 evidence 可以核验。
3. 系统错误与答案质量问题被分开建模。
4. LLM evaluator 不是唯一真相，且可校准、可替换、可版本化。
5. Prompt 变化产生的改善和 regression 都会被保留，而不是只展示成功案例。

### 2.3 诚实声明

可以声明：

> Reproducible execution contract and comparable evaluation artifacts.

不可以声明：

- 模型输出逐字可复现。
- 12 cases 代表总体 Tutor 质量或学习效果。
- `evaluation` profile 等同完整生产路径。
- application-level checksum 等同 tamper-proof storage。

## 3. MVP 范围

### 3.1 包含

- 一个版本化 `Tutor Grounded QA v1` task family。
- 12-case regression suite：6 个 answerable、3 个 multi-evidence、3 个 expected-refusal cases。
- 独立 scorer calibration fixtures，不与 12-case suite 重合。
- `tutor-v2` baseline 与 `tutor-v3` candidate；当前生产 Prompt 必须先逐字冻结为 baseline，candidate 未经证据验证不得替换生产默认。
- content-addressed isolated eval corpus。
- UI 单 case live run、cancel、detail、compare 和 re-score。
- CLI/fixture 原子导入完整 suite。
- SQLite runtime artifacts 与 JSONL export。
- 一个由真实 suite export 生成、脱敏、hash-verified、可再分发并进入版本控制的 curated demo seed fixture；fresh clone 通过 CLI 原子导入它。原始 runtime DB/output 仍不进入 Git。

### 3.2 不包含

- 浏览器内编辑 Prompt、Task、Corpus 或 Scorer。
- 浏览器上传 eval PDF 或执行完整 batch。
- 任意用户问题、私人 PDF 或真实学生流量进入 eval artifacts。
- runtime Judge 的生产 fidelity 对比。
- 断线续跑、event replay、reconnect、后台 worker 或 job queue。
- 跨进程并发、云端多用户 ownership、WORM 或签名审计存储。
- Plan/Quiz eval 迁移。
- Langfuse、Inspect AI、Harbor 或 DSPy 的运行时依赖；未来可增加 export/integration。

## 4. 系统边界

```text
Version-controlled definitions
TaskCase + Prompt + Scorer + CorpusSnapshot
                    │
                    ▼
             Task / Variant Registry
                    │
                    ▼
               RunService
                    │
                    ▼
                TutorRunner
                    │
                    ▼
          TutorAttemptEngine (one attempt)
                    │
                    ▼
        CandidateArtifact + TutorRunTrace
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 Deterministic Scorers   LLM Scorers
          └─────────┬─────────┘
                    ▼
          Versioned ScoreSet(s)
                    │
                    ▼
       Run Lab / Compare / Re-score
```

Production Graph 与 Eval 共享 `TutorAttemptEngine`，但拥有不同 orchestrator：

```text
Production Graph
→ TutorAttemptEngine
→ runtime Judge / retry / MemoryWriter / Chat persistence

Evaluation TutorRunner
→ TutorAttemptEngine exactly once
→ isolated corpus / no Router / no Judge / no business writes
```

详细原因见 [ADR 0001](../../adr/0001-share-tutor-attempt-not-production-orchestration.md)。

## 5. TutorAttemptEngine 合同

概念接口合同如下；实现文件和参数命名可以匹配现有代码风格，但不得改变这些职责边界：

```text
answer(question, retriever, llm, prompt_template, event_sink, budget)
→ TutorCandidate
```

职责：

- 调用注入的 retriever 一次并保存 exact chunks。
- 使用已经解析并冻结的 Prompt template。
- 将模型 token、阶段与预算事件写入 `event_sink`。
- 返回 answer、citations、exact evidence、formatted context、usage 和 trace。

不负责：

- Router。
- runtime Judge、retry hint 或 retry budget。
- Memory hydration/write。
- Chat Session/Message persistence。
- 选择 Prompt、corpus、provider 或 expected behavior。

Production Graph adapter 必须保持现有 token、citation 和 `last_context` 契约；Judge pass 调用一次 Attempt，最多两次 retry 时总共最多三次 Attempt。Evaluation Runner 始终只调用一次。

## 6. Version-controlled Definitions

### 6.1 TaskCase

每个 case 至少包含：

- `task_case_id` 与 `task_case_version`。
- question。
- case type：`answerable | multi_evidence | expected_refusal`。
- expected behavior 与人工理由。
- required evidence chunk IDs 或允许的 evidence set。
- required scorer dimensions。
- critical finding policy。

客户端只提交 ID，不提交 expected answer、Prompt 文本或文件路径。

### 6.2 CorpusSnapshot

Snapshot 进入版本控制并包含项目有权再分发的精简 eval corpus：

- stable chunk IDs。
- exact chunk text。
- 每个 chunk 的 content hash。
- chunking、embedding、retrieval 和 reranker config versions。
- snapshot aggregate hash。

Run 仍保存本次真正提供给 Tutor 的 top-k evidence。只有 manifest 而没有原文，不足以称为可重放的 corpus snapshot。

### 6.3 Prompt 与 Scorer Registry

- 当前生产 Tutor Prompt 原样冻结为 `tutor-v2`。
- 新 candidate 为 `tutor-v3`。
- production default 在 suite evidence 被人工审查前保持 `tutor-v2`。
- scorer rubric、parser、verdict policy 和 scorer model config 都有独立 version。
- 历史 Run 读取 frozen manifest，不反查当前 Registry 重新解释配置。

## 7. Run 数据流

1. 服务端用 experiment、task、variant 和 profile IDs 解析 Registry。
2. 校验 Task、CorpusSnapshot、Prompt、Scorer 与 schema hashes。
3. 在数据库中原子 claim single-flight，并创建 Run。
4. 建立 `RunManifest`，明确 `runtime_judge=off`。
5. TutorRunner 使用 isolated corpus 调用一次 TutorAttemptEngine。
6. 冻结 CandidateArtifact：answer、citations、exact evidence、trace、budget、artifact hash。
7. deterministic scorers 执行适用的 integrity checks。
8. LLM scorers 只接收 question、Candidate 和允许使用的 evidence。
9. 每个 scorer 独立记录 `ScorerExecution`。
10. 生成新的 `ScoreSet`；聚合只用于阅读，原始分项与 findings 保留。
11. live case 与兼容 baseline case 对比。
12. historical re-score 只读取冻结 artifact，追加新 ScoreSet，不重新运行 retrieval 或 Tutor。

### 7.1 First attempt 与 production-fidelity

MVP 只保存 evaluation profile 的单次 Candidate。未来增加 `production-fidelity` 时必须同时保存 first attempt 与 runtime Judge 修正后的 final answer，不能只保留最终文本。

## 8. 状态、错误与质量

### 8.1 Run 状态

- Lifecycle：`queued | running | finished | cancelled`。
- Finished outcome：`success | system_failed | timed_out | budget_exceeded`。
- ScoreSet status：`pending | running | completed | partial | failed | cancelled`。
- Quality verdict：`pass | fail | inconclusive | not_evaluated`。

Evaluation summary 尽量从 scorer executions 派生，避免维护重复、会漂移的状态字段。

### 8.2 OperationalError

- `manifest_invalid`
- `corpus_unavailable`
- `corpus_mismatch`
- `retriever_error`
- `model_unavailable`
- `generation_timeout`
- `scorer_timeout`
- `scorer_parse_error`
- `budget_exceeded`
- `process_interrupted`
- `harness_internal_error`

每个 error 记录 stage、stable code、sanitized message、retryable 和 spent budget。不得保存 API Key、Authorization header 或完整敏感 stack。

### 8.3 QualityFinding

- `retrieval_empty`
- `citation_missing`
- `citation_invalid`
- `unsupported_claim`
- `incomplete_answer`
- `inappropriate_refusal`
- `expected_refusal_observed`

检索正常返回零结果是有效执行，不是 operational failure。其质量含义由 TaskCase expected behavior 决定。

### 8.4 Verdict Policy v1

LLM rubric 使用 1–5 anchored scale，不使用未经校准的 0.80 threshold：

- `4`：满足可发布标准。
- `3`：存在实质问题或证据不足。
- 每个 required dimension 必须独立达到 `4`，不能用平均分掩盖严重弱项。
- deterministic hard gate 或 critical finding 直接 Fail。
- scorer missing、timeout、malformed 或 exception 是 evaluation failure；若剩余证据不足则 Inconclusive，绝不 fallback Pass。
- 非关键 finding 可以与 Pass 共存，但必须显示。

Answerable cases 评估 groundedness、citation entailment 和 coverage；expected-refusal cases 评估 refusal appropriateness 与 unsupported claims。

Scorer calibration 使用与 12-case regression suite 分离的 frozen Candidate fixtures，包含明确 Pass、Fail、borderline、正确拒答与错误拒答 anchors。

## 9. Budget 与 Reproducibility

初始 Demo budget：

| Stage | Hard limit |
|---|---:|
| Corpus preflight + retrieval | 5s |
| Tutor generation | 55s |
| Hybrid scoring | 25s |
| Total wall clock | 90s |

这些数字是初始保护，不是性能承诺；实现后按实际 p95 调整。token usage 无法取得时必须写 `unavailable`，不能补 `0`。

RunManifest 冻结：task、corpus、Prompt、provider/model/parameters、retrieval/reranker/chunking、runner/scorer/schema、budget、runtime Judge profile、code revision，以及 provider 支持时的 seed。不得静默 fallback 到其他模型、corpus 或 scorer。

## 10. Controlled Comparison

每个 experiment 声明 `experiment_axes`。MVP 只允许：

```text
experiment_axes = ["prompt_version"]
```

Compatibility：

- `Controlled`：只在声明的实验轴上不同，且 task、corpus、模型、retrieval、budget、schema 等控制变量一致。
- `Informational`：存在未声明配置差异，只能并排查看。
- `Incompatible`：task、corpus 或 artifact schema 无法对齐，不进入比较。

计算 score delta 前，两边必须使用同一 scorer version；否则先对冻结 artifacts re-score。单 case 只能称 `case delta`；suite 汇总只能描述该冻结 regression suite。

## 11. Persistence

定义文件进入 Git；raw runtime DB/output 不进入 Git。唯一例外是由真实运行 deterministic 生成、经过 secret/corpus/redistribution review 的 curated demo seed fixture：它作为 fresh-clone product/test seed 进入 Git，不允许人工编辑 metrics 或伪造 artifacts。现有应用 SQLite 增加逻辑隔离的 eval bounded context：

### 11.1 `eval_runs`

结构化查询列：

- `id`
- `experiment_id`
- `suite_execution_id`（UI 单 case 为 null）
- `task_case_id` / `task_case_version`
- `variant_id`
- `run_profile`
- lifecycle / outcome / operational error
- `manifest_json` / `manifest_hash`
- `candidate_artifact_json` / `artifact_hash`
- timestamps

### 11.2 `eval_score_sets`

- `id` / `run_id`
- scorer bundle version
- artifact input hash
- status / quality verdict
- operational error code / sanitized message（用于取消、scoring timeout 与进程中断）
- aggregate scores / findings
- timestamps

### 11.3 `eval_scorer_executions`

- `id` / `score_set_id`
- scorer ID / version
- status：`success | failed | skipped`
- input hash / output
- error code / sanitized message
- latency / available usage

Compare 不持久化，由查询派生。MVP 不引入 filesystem blob；先测量 DB 增长。

### 11.4 Integrity

- queued/running Run 可以更新进度。
- CandidateArtifact 通过带条件的原子 finalize 只写入一次。
- terminal 后 artifact 视为 immutable。
- ScoreSet 与 ScorerExecution repositories 只提供 append；checksum 在读取、compare 和 re-score 前验证。
- 该保证称为 application-level append-only and checksum-verified，不称 tamper-proof。

### 11.5 Suite Import

CLI 在一个 SQLite transaction 中验证并导入完整 suite。Registry 决定 expected cases 与 variants；缺失、重复、版本或 checksum 不匹配时整批回滚。MVP 不增加 `eval_suite_executions` 表；只有未来支持 batch resume、partial import 或 UI 启动 suite 时才增加。

Repository 中的 curated demo seed 必须覆盖 12 cases × 2 Prompt variants、expected-refusal、至少一个 regression 和 historical ScoreSets；它不包含 credentials、完整 Base URL、私人输入或 Registry 之外的语料。原始 export 继续 ignored。

## 12. Reset、Privacy 与 Auth

- Eval API 使用 `require_existing_user`。
- Run Lab 只在 `STUDY_COACH_LOCAL_MODE=1` 的 single-user instance 开启；非 local-mode 默认不可用，直到未来定义 owner/admin model。
- 不伪装成多用户安全：eval tables 不增加没有真实隔离作用的 `user_id`。
- API Key 只在当前请求内存中使用，不进入 DB、trace、log 或 export。
- Manifest 保存规范化 provider/model/参数与 endpoint fingerprint，不保存完整敏感 Base URL。
- `learning` reset 保留 eval artifacts。
- `factory` reset child-first 删除 ScorerExecutions → ScoreSets → Runs，并将 eval counts/space 加入 Data Summary。
- Data Summary 单独返回 eval counts/space；这些字段不参与现有 `has_learning_data` 计算，避免 `learning` reset 因保留 eval artifacts 而无法达到空学习状态。
- active Run 会持有现有 lifecycle shared lease，因此 reset 返回当前既有的 data-operation conflict；reset recovery 语义保持不变。
- MVP 只允许预置 case 与可再分发 corpus，因此 eval artifacts 不包含任意学生输入。未来 custom case 必须进入用户数据与删除生命周期。

## 13. Services 与 API

### 13.1 Services

- `TaskRegistry`
- `CorpusSnapshotLoader`
- `TutorRunner`
- `RunService`
- `ScoringService`
- `CompareService`
- Run、ScoreSet、ScorerExecution repositories

### 13.2 API

```text
GET  /api/eval/experiments
GET  /api/eval/runs
GET  /api/eval/runs/{run_id}
POST /api/eval/runs/stream
POST /api/eval/runs/{run_id}/cancel
POST /api/eval/runs/{run_id}/rescore/stream
POST /api/eval/score-sets/{score_set_id}/cancel
GET  /api/eval/compare?left={id}&right={id}
```

Run request 只接受：

```json
{
  "experiment_id": "tutor-prompt-regression-v1",
  "task_case_id": "tgqa-004",
  "variant_id": "tutor-v3",
  "run_profile": "evaluation"
}
```

服务端 Registry 解析所有其他输入。实验定义中的 provider、model 和生成参数是受控变量的事实来源；现有 LLM headers 只提供匹配的连接配置与凭证，出现 model/provider/config mismatch 时拒绝运行，不得静默改变实验轴。服务端同时拒绝任意 Prompt、path、corpus、scorer code、expected answer 和 runtime Judge override。

沿用 authenticated POST fetch stream，不使用 EventSource。事件至少包括：

```text
run_created
stage_started
stage_completed
scorer_completed / scorer_failed
run_finished
```

首事件必须包含 `run_id`；re-score 首事件包含 `score_set_id`。

## 14. Attached Execution 与 Concurrency

MVP 不建设后台 execution registry。中央 frontend Run store 持有 stream 和 `AbortController`：

- SPA route 切换不取消 Run。
- 显式 Cancel 先调用幂等 cancel endpoint，再 abort stream。
- 页面刷新、网络断线或连接 cancellation 终止 attached execution。
- Runner 在 stage 与 token 边界检查 cancel，并确保 Candidate 不会在取消后 finalize。
- cancel/complete race 只能提交一个 terminal state。
- startup reconciliation 将遗留 running Run/ScoreSet 标记为 `process_interrupted`。
- 不承诺断线继续、重连或 stream replay。

同时只允许一个 live Run 或 re-score。数据库原子 claim 是事实来源，进程内 lock 只作当前单 worker 的快速保护：

- 第二个写请求返回 `409 evaluation_busy`、active entity ID 和 kind。
- Run 与 re-score 共享 single-flight gate。
- Runs/detail/Compare 读操作不取写锁。
- 单 worker 是显式 deployment contract；未来多 worker 必须重新设计 gate 或引入 job coordinator。

## 15. Run Lab UI

### 15.1 页面

1. `/run-lab` — Experiments / Runs：suite delta、regressions、inconclusive、history filters。
2. `/run-lab/runs/:runId` — Evidence Console v2。
3. `/run-lab/compare?left=&right=` — Controlled Compare。

### 15.2 Evidence Console v2

- 顶部 Contract：Task、Corpus、Prompt、Model、RunProfile、budget、compatibility。
- 左侧 Execution Evidence：关键 stages、exact retrieved chunks、answer claim 与 citation 对齐。
- 右侧 Verdict、分项 scores、Quality Findings 与 baseline summary。
- full trace、provenance、unavailable usage 放入二级 drawer。
- V2 视觉布局被保留，但原型占位值不属于验收数据；例如 citation hard gate 失败时不得同时显示 Pass。

### 15.3 Demo fallback

Live Run 30 秒仍未完成时，用户可以明确选择 `Open comparable completed run`。历史 Run 必须显示不同 ID；当前 Run 可取消并诚实记录，不得将历史 artifact 冒充刚完成的 live 结果。

## 16. 五分钟演示

| Time | Demo |
|---|---|
| 0:00–0:35 | 12-case suite delta、唯一 Prompt axis、至少一个 regression |
| 0:35–1:10 | 打开 expected-refusal case，核验 Task、Corpus 与 profile |
| 1:10–1:50 | 启动真实 live run；必要时切换到明确标识的历史 Run |
| 1:50–3:10 | 查看 evidence、claim/citation 与 Hybrid Score |
| 3:10–4:05 | 对比 v2/v3 case artifacts，解释 hallucination 或拒答差异 |
| 4:05–4:40 | 展示同一 Candidate 的 scorer-v1/v2 ScoreSets 并存 |
| 4:40–5:00 | 收束到 reproducibility、accountability、regression detection |

Historical re-score 按钮保留，但主路径只展示已经存在的多个 ScoreSets，不现场等待 re-score。

## 17. Testing Strategy

### 17.1 TutorAttemptEngine contract

共享合同测试使用固定 fake retriever、fake token sequence 和 event sink，验证：

- token event 语义与 answer 拼接。
- exact chunks、citations、context、trace 与 budget。
- empty retrieval 与 LLM exception。
- Graph adapter Judge pass 为一次 Attempt，最多两次 retry 为最多三次。
- Eval Runner 严格一次 Attempt。

真实模型不测试 token chunk 边界，只验证最终文本与事件语义。

### 17.2 Isolation

通过 sentinel dependencies 证明 Eval 不调用 global retriever、Router、Memory、Chat persistence 或 runtime Judge；API 拒绝客户端注入 Prompt、path、expected answer 和 profile override。

### 17.3 Scoring

- verdict truth table 与所有 anchor boundary。
- deterministic hard gate、critical/noncritical findings。
- missing、timeout、malformed、exception scorer。
- partial 与 inconclusive 派生。
- calibration fixtures 独立于 regression suite。

### 17.4 Persistence and races

- fresh DB migration/head/idempotency。
- lifecycle 与 atomic finalize。
- hash mismatch 拒绝 read/compare/re-score。
- re-score 不调用 Tutor/retriever且只追加 ScoreSet。
- suite import 缺失、重复、checksum mismatch 整批回滚。
- 两个独立 DB connections 配合 barrier/controlled hook，覆盖 run/run、run/rescore、cancel/finalize、reset/run、startup/new-run 两种提交顺序；禁止依赖 sleep。

### 17.5 SSE and frontend

- 首 event ID、event ordering、malformed chunk、terminal event。
- explicit cancel 幂等、generator `CancelledError/aclose`、cancel/complete race、gate release。
- frontend `AbortController`、await token、route switch、pagehide、stale mutation、409 attach。
- 真实 refresh/socket disconnect 留给浏览器 acceptance。

### 17.6 Compare, Reset, Auth

- Controlled/Informational/Incompatible。
- scorer version mismatch 不显示 delta。
- case delta 与 suite copy 不泛化。
- local-mode、auth、secret allowlist。
- learning reset 保留；factory reset FK 顺序删除；counts/space 正确。
- active run 与 reset 使用现有 lifecycle gate。

## 18. Verification Gates

### 18.1 Automated

- 每个 vertical slice 的 targeted backend/frontend tests。
- 最终完整 backend pytest suite。
- 最终 frontend 非 watch Vitest run。
- production frontend build。
- Alembic upgrade、head 与 fresh-DB verification。
- 现有 Graph、Chat streaming、Reset、Plan/Quiz 回归无退化。

无法运行的 gate 必须明确标记未验证，不能用 targeted tests 代替全量声明。

### 18.2 Manual evidence

- content-addressed corpus 与 Prompt hashes。
- 独立 scorer calibration fixtures 及人工标签/理由。
- 12-case 逐案 expected behavior 签核。
- 真实 tutor-v2/v3 CandidateArtifacts 与 ScoreSets。
- 从真实 export deterministic 生成的 curated demo seed 能在 fresh temporary SQLite 中原子导入并通过全部 hash checks。
- 至少一个 expected-refusal 与一个 regression。
- 真实浏览器 route switch、refresh、network disconnect、cancel。
- 五分钟 demo checklist 或录像。

### 18.3 Documentation

README、EVAL、ARCHITECTURE 人工 review，确保能力边界一致。自动化只检查 schema/version constants、fixture references、links 与示例命令存在，不做脆弱全文匹配。

## 19. Completion Criteria

MVP 只有在以下全部成立时完成：

1. targeted 与完整回归 tests 通过。
2. migration 与 production build 通过。
3. scorer calibration fixtures 与 12-case suite 相互独立。
4. suite artifacts 来自真实执行，而非手写演示结果。
5. 展示并解释 expected-refusal 与至少一个 regression。
6. Run Lab 不显示自相矛盾的 verdict、findings 或 placeholder metrics。
7. 文档只声明冻结 12-case suite 上的 v2/v3 差异。

## 20. Implementation Preconditions

- 该设计不授权当前工作区直接实现。
- 实施前必须同步并核验最新 `origin/main`，处理当前 dirty worktree 与远端差异。
- 使用隔离 worktree 和 tracer-bullet TDD slices。
- 不新增 dependency，除非实施计划单独论证并获批准。
- 任何 migration、factory reset 变化与 Prompt candidate 都必须以真实 RED → GREEN 证据推进。
