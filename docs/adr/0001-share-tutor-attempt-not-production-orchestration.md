# Share a Tutor Attempt, Not Production Orchestration

Study Coach 的 Production Graph 与 Learning Run Harness 共享一个 graph-free `TutorAttemptEngine`，负责单次 retrieval、版本化 Prompt 构造、token/trace event 和 Candidate 输出；Judge retry、MemoryWriter、Chat persistence 与 Router 仍由 Production Graph 拥有。Evaluation Runner 直接调用一次 TutorAttempt，并使用隔离 corpus 与 `runtime_judge=off`，因为复用完整 Graph 会让 Judge 改写被测答案并引入业务副作用，而复制一套 Eval Tutor 又会造成生产与评测逻辑漂移。

## Consequences

- Production Graph 必须通过 adapter 保持现有 token、citation、`last_context` 和 retry 行为。
- Evaluation profile 评估的是一次基础 TutorAttempt，不等同完整生产路径。
- 未来若增加 `production-fidelity` profile，必须分别保留 first attempt 与 Judge 修正后的 final answer。
- Evaluation 使用三张 append-only 表（`eval_runs` / `eval_score_sets` / `eval_scorer_executions`）和 process-local single-worker lease；Factory reset 必须先删 executions，再删 ScoreSets，再删 Runs。
- Fresh clone 的演示种子是 committed curated fixture，不是本机 ignored raw output。

