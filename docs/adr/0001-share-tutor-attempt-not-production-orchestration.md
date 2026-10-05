# Share a Tutor Attempt, Not Production Orchestration

Study Coach 的 Production Graph 与 Learning Run Harness 共享一个 graph-free `TutorAttemptEngine`，负责单次 retrieval、版本化 Prompt 构造、token/trace event 和 Candidate 输出；Judge retry、MemoryWriter、Chat persistence 与 Router 仍由 Production Graph 拥有。Evaluation Runner 直接调用一次 TutorAttempt，并使用隔离 corpus 与 `runtime_judge=off`，因为复用完整 Graph 会让 Judge 改写被测答案并引入业务副作用，而复制一套 Eval Tutor 又会造成生产与评测逻辑漂移。

## Consequences

- Production Graph 必须通过 adapter 保持现有 token、citation、`last_context` 和 retry 行为。
- 共享 Attempt 的文本合同由 production 与 evaluation 共同继承，不能按各自方便分叉：内容的正文提取、跳过清单与固定拒绝语义在 `app.llm.content` 单点定义，两侧对非法或结构变化的内容都没有回退答案。差别只在谁来消费异常——Production Graph 让该次调用失败并截断 SSE，Evaluation Runner 用既有 `generation` / `model_unavailable` → `system_failed` 终止 Run，不冻结 CandidateArtifact/hash、不进入 scorer。任一侧都不因此放宽边界，也不新增 Judge verdict、错误码或 retry。文本边界只覆盖 LangChain message content；原始 Responses API / Google payload、provider factory、`bind_tools` 与 message history 不在其中，工具循环仍持有完整 AIMessage 与 tool_calls。
- Evaluation profile 评估的是一次基础 TutorAttempt，不等同完整生产路径。
- 未来若增加 `production-fidelity` profile，必须分别保留 first attempt 与 Judge 修正后的 final answer。
- Evaluation 使用三张 append-only 表（`eval_runs` / `eval_score_sets` / `eval_scorer_executions`）和 process-local single-worker lease；Factory reset 必须先删 executions，再删 ScoreSets，再删 Runs。
- Fresh clone 的演示种子是 committed curated fixture，不是本机 ignored raw output。

