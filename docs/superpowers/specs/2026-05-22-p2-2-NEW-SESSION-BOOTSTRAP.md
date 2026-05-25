# P2.2 Agent Loop Ablation — New Session Bootstrap Prompt

> **Copy the message below into the new Claude Code session.** It primes the new window with full context to continue P2.2 work without re-doing the brainstorm.

---

## Paste this into the new session

```
我要继续 Study Coach 项目的 P2.2 阶段工作 —— Agent Loop Ablation。

**项目根目录**: `/Users/lianghaozhe/Downloads/Study Compaion and JadeAI/study-coach/`

**先做这件事**：读以下两个文件，建立完整上下文。

1. **设计 spec**（self-contained，含 Q1-Q5 决策 + 完整架构 + cut 切分 + eval matrix）:
   `study-coach/docs/superpowers/specs/2026-05-22-p2-2-agent-loop-ablation-design.md`

2. **项目整体上下文**（memory 应自动加载，但请确认能看到 P2.2 段落）:
   `project_study_coach_refactor` 中关于 P2.2 的 "planned, not yet started" 段落

**当前状态（baseline）**：
- P2.1-⑤ Plan Chain 已完成，**157 个 backend tests 全绿**
- Deterministic Planner 在 `backend/app/agent/planner.py` 已上线，real-Ollama E2E 验证过
- P2.2 spec 已写好，**未开始实现**

**P2.2 目标一句话**：
在 `backend/app/agent/planner_agent.py` 写一个 LLM tool-calling agent loop 版本的 Planner（不替换 deterministic 版本，并存），跑头对头 ablation 测对比数据，产出 EVAL.md + 博客回应 `learn-claude-code` 仓库的 "agency = model + minimal harness" 立论。

**4-model matrix（已锁）**: gemma3:4b / qwen3.5:4b / qwen2.5:7b / gemma4:e4b
**Approach（已锁）**: A — 手写 while-loop + 嵌入 LangGraph plan_node mode-aware dispatch
**Feature flag（已锁）**: HTTP header `x-planner-mode: deterministic|agent_loop`
**Tools（已锁）**: 5 个 minimal set，无 done() 无 todo()，max_iter=10
**Judge（已锁）**: 双 judge — qwen2.5:7b 本地 + BYOK GPT-4o-mini 云

**关键参考代码**（learn-claude-code repo 在 `/Users/lianghaozhe/learn-claude-code/`，主要看 s01 / s02 / s04）：
- `agents/s01_agent_loop.py` —— while-loop 模板
- `agents/s02_tool_use.py` —— tool dispatch map 模式
- `agents/s04_subagent.py` —— max_iter 参考（他们用 30，我们用 10）

**关键现有代码**（study-coach 项目内）：
- `backend/app/agent/planner.py` —— deterministic 版 Planner，新 agent loop 版要 mirror 风格但**完全独立模块**
- `backend/app/agent/quiz_master.py` —— 类似的 deterministic 节点，参考工厂模式 + `_safe_writer()`
- `backend/app/agent/tools/plan.py` —— 5 个工具的纯函数实现，agent loop 用 `@tool` 装饰器封装它们
- `backend/app/agent/graph.py` —— `plan_node` 当前实现，要改成 mode-aware dispatcher
- `backend/app/api/deps.py` 和 `backend/app/api/routes.py` —— 加 `get_planner_agent` + `get_planner_mode` + chat 签名扩展

**项目纪律**（必读，CLAUDE.md 中文版）：
- **不是 git 仓库**：不要 `git init / commit / push`。CLAUDE.md 全局规则「不主動 commit」
- **不要動非本次任務的程式碼**：任何非 P2.2 文件改动前先问我
- **TDD red-green-refactor**：每 cut 先写测试再实现，每 cut 完跑 `cd backend && uv run pytest -q` 全绿才进下一刀
- **完成前调 `superpowers:verification-before-completion`**：跑命令拿 fresh evidence
- **中文回复 + 直接 + 不啰嗦**
- **`# cloud-adapt:` 注释**：每个 cloud-model 适配点 mark 一行注释，**不实现**，spec §11 有 grep 锚点清单

**下一步该做什么**：

调用 `superpowers:writing-plans` skill，针对 P2.2 spec 写出 cut-by-cut 实施 plan（spec §8 已给 skeleton：①a tools → ①b AgentTrace → ①c loop body → ①d graph wiring → ①e prod wiring → ①f real-Ollama smoke → ②a eval harness → ②b run matrix → ③ writeup）。每个 cut 含完整代码块 + TDD 步骤 + 预期 pytest 输出。

Plan 写完后我审过，再用 `superpowers:subagent-driven-development` 执行。

**重要 brainstorm 上下文（如果对设计决策有疑问可以参考）**：

读 spec §1「Decisions Locked」一节有 Q1-Q5 完整决策 + rationale。如果决策需要回溯讨论，spec 已经把"为什么这么选"写进每题的 Rationale 段了。

**已知风险点**（spec 没展开但需要 plan 阶段注意）：
1. Ollama tool calling 在 gemma3:4b 上很可能崩 —— 这就是实验本身想测的，不要因为崩了就换模型，**崩本身是数据**
2. Thinking 模型 (qwen3.5:4b / gemma4:e4b) 的 `think=False` 参数名跨版本不稳，Cut P2.2-①f smoke test 验证后再固定
3. AgentTrace 字段在 production 会很大，spec §11 标 cloud-adapt 添 redact 选项
4. P2.1-⑤ Cut ⑤i 留的 `_extract_topic` 修复是 inline 改的（不在测试里防回归），可以在 P2.2 一并加测

开始吧。读完 spec 后告诉我你的理解，确认无误后调 writing-plans。
```

---

## Why this prompt works

- **明确指路**：先读 spec + memory，建立上下文
- **状态报告**：157 tests baseline + spec done + 未开实现，无歧义
- **决策锁定**：所有 Q1-Q5 决策都点出，新窗口不会浪费 token 重新讨论
- **代码地标**：现有相关代码 + learn-claude-code 参考 templates 都指明绝对路径
- **下一步明确**：调 writing-plans → 审 plan → subagent-driven 执行
- **风险已知**：把 4 个 "implementation 阶段才发现也行" 的风险点写进 brief，省得新窗口踩
- **纪律带上**：项目纪律（不 commit / TDD / cloud-adapt 注释）一次性给

## 使用方式

1. 在当前对话**结束前**，把上面**整段灰色框内容**复制出来
2. 开新 Claude Code 会话（同一项目目录）
3. 第一条 message 粘贴该 prompt
4. 新窗口的 Claude 会读 spec + memory → 反馈理解 → 你确认 → 它调 writing-plans 写 cut-by-cut plan

新窗口任何时刻偏离方向：让它**重读 spec §1**（决策锁定列表）回到正轨。
