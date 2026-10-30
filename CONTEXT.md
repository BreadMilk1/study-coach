# Study Coach Domain Language

本文件固定 Study Coach 在学习辅导与 AI 行为评测中的领域语言，避免将生产期自检、实验期评分和学生学习状态混为同一概念。

## Learning Run Harness

**Learning Run**:
在固定 `TaskCase`、`CorpusSnapshot`、`RunProfile`、模型配置和预算下执行的一次可追溯 AI 学习任务。它保证执行契约可复现，不承诺模型输出逐字相同。
_Avoid_: Request, test call, agent session

**TaskCase**:
一个版本化的教育任务输入，包含问题、case 类型和人工确认的 expected behavior；它可以要求正常回答，也可以要求在证据不足时拒答。
_Avoid_: Query, prompt, test question

**CorpusSnapshot**:
评测专用、内容寻址且可校验的不可变语料版本，包含稳定 chunk ID、原文和内容 hash。
_Avoid_: Uploaded PDF, current index, document manifest

**RunManifest**:
一次 Learning Run 的冻结执行契约，记录 task、corpus、Prompt、模型、retrieval、scorer、预算、代码和 schema 版本，但不包含凭证。新运行另记录 `usage_contract="tutor-chunk-sum-v1"`；历史未声明记录不补字段、不改原 hash，也不被自动追认为相同的 usage 语义。
_Avoid_: Config, metadata, request headers

**RunProfile**:
声明一次运行采用的产品边界。`evaluation` profile 隔离生产期 Judge、Memory 和 Chat persistence；未来的 `production-fidelity` profile 才代表完整生产编排。
_Avoid_: Mode, environment

**TutorAttempt**:
Tutor 对固定问题进行的一次 retrieval、Prompt 构造与模型生成尝试。生产 Graph 可以围绕它执行 Judge retry；evaluation profile 严格只执行一次。`tutor-chunk-sum-v1` 只累加本次模型流中现有 extraction 得到的计数，保留带符号的 chunk 补偿；它不是跨 retry、Judge 或整次 turn 的用量总账，也不恢复 adapter 已丢失的字段来源。
_Avoid_: Tutor run, retry loop, graph run

**CandidateArtifact**:
TutorAttempt 完成后冻结的候选产物，包括答案、引用、实际检索证据、trace、预算使用量和校验 hash。
_Avoid_: Response, result JSON, assistant message

## Evaluation

**ScoreSet**:
一组版本化 scorer 对同一个 CandidateArtifact 生成的评估结果。Historical re-score 会追加新的 ScoreSet，不覆盖旧结果。
_Avoid_: Judge score, evaluation result

**ScorerExecution**:
一个确定版本的 scorer 对 CandidateArtifact 的单次执行记录，明确区分 success、failed、skipped、错误和可用成本信息。
_Avoid_: Judge call, score dimension

**QualityFinding**:
对答案质量的可解释发现，例如 unsupported claim、invalid citation 或 inappropriate refusal；它不等于系统执行故障。
_Avoid_: Error, exception, run failure

**OperationalError**:
Harness、模型、retriever 或 scorer 未能按执行契约完成工作时产生的运行故障。
_Avoid_: Quality failure, bad answer

**Controlled Comparison**:
两个 Runs 只在声明的 `experiment_axes` 上不同，且其余控制变量与评分版本兼容时进行的差异比较。双方均声明相同且受支持的 `usage_contract` 才能通过新的 usage 闸门；单边声明、不同、未知或非法声明降为 `informational`，不显示受控分数 delta，声明为实验轴也不能绕过。双方都未声明时保留既有质量比较行为，但不据此证明历史 usage 单位相同。当前持久化 CandidateArtifact 不含 `schema_version`；compare 将缺失值视为 `candidate-artifact-v1`，不得把该字段写进 artifact hash。
_Avoid_: A/B test, side-by-side view, improvement

**Regression Suite**:
一组冻结、人工确认 expected behavior 的定向 cases，用于检测特定 AI 行为变化；它不是总体学习效果或统计显著性的证明。Regression 可以是同一 scorer 上的 verdict/score 变差，也可以是声明轴上的 deterministic finding 变化加上语料外常识。LLM judge 解析失败必须保持 inconclusive。Run Lab 的 Regressions 计数必须使用与 release contract 相同的 hybrid-v1 pairwise 规则，不能用 fail verdict 条数代替。Fresh clone 使用 committed curated fixture 作为演示种子；本机 raw `output/` 不进 Git。
_Avoid_: Benchmark, comprehensive evaluation, golden truth

## Candidate 与 CandidateArtifact

这两个词不是同义词，分属不同阶段，不能互换：

- `Candidate`：离线 cloud capability 评测（`app/eval/p2_3_cloud_capability`）里由 `CandidateStore` 记录的模型响应对象，是**未冻结**的评测对象。它可以停留在草稿（`.partial.json`）或失败标记（`.failed.json`）状态，不进入生产存储，也不带 Run 的冻结 hash。
- `CandidateArtifact`：上文 Learning Run Harness 定义的冻结产物 —— `TutorAttempt` 完成后写入 Run、带校验 hash、可被 ScoreSet 复评；它不代表完整生产编排。
- `TutorCandidate`：`app/agent/tutor_attempt.py` 中 `TutorAttemptEngine.answer()` 返回的内存 dataclass（answer / citations / evidence / usage / trace），是冻结前的中间结果；只有被写入 Run 的 `candidate_artifact_json` 之后才成为 `CandidateArtifact`。
