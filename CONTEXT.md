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
一次 Learning Run 的冻结执行契约，记录 task、corpus、Prompt、模型、retrieval、scorer、预算、代码和 schema 版本，但不包含凭证。
_Avoid_: Config, metadata, request headers

**RunProfile**:
声明一次运行采用的产品边界。`evaluation` profile 隔离生产期 Judge、Memory 和 Chat persistence；未来的 `production-fidelity` profile 才代表完整生产编排。
_Avoid_: Mode, environment

**TutorAttempt**:
Tutor 对固定问题进行的一次 retrieval、Prompt 构造与模型生成尝试。生产 Graph 可以围绕它执行 Judge retry；evaluation profile 严格只执行一次。
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
两个 Runs 只在声明的 `experiment_axes` 上不同，且其余控制变量与评分版本兼容时进行的差异比较。
_Avoid_: A/B test, side-by-side view, improvement

**Regression Suite**:
一组冻结、人工确认 expected behavior 的定向 cases，用于检测特定 AI 行为变化；它不是总体学习效果或统计显著性的证明。Regression 可以是同一 scorer 上的 verdict/score 变差，也可以是声明轴上的 deterministic finding 变化加上语料外常识。LLM judge 解析失败必须保持 inconclusive。Fresh clone 使用 committed curated fixture 作为演示种子；本机 raw `output/` 不进 Git。
_Avoid_: Benchmark, comprehensive evaluation, golden truth

