# Cloud Capability Appendix — Design Spec

> 日期：2026-08-23
> 状态：设计待批准。实现尚未开始。
> 目标：回答一个明确的问题——**在更高能力层上，agent-loop 相对 deterministic 的结论是否仍然成立**。
> 前置：`main@c98889b`。本文所有「已验证」条目均为实测结果，不是推断。

---

## 1. 范围

在 **P2.3 Quiz 消融矩阵**中新增一个 matched pair：

```
MiniMax-M3 × { deterministic , agent_loop }
```

### 1.1 明确不做

- **不碰 Learning Run。** 它的 `experiment_axes = ["prompt_version"]` 是冻结的；加入 cloud variant 会同时改变 provider/model，破坏 Controlled Compare 的唯一变量原则。
- **不修改既有 396 条记录。** 字节级不变。
- **不回评历史数据。** 理由见 §5.3。
- **不新增 dependency。**

---

## 2. 已验证的协议事实

以下全部为 2026-08-23 实测，非文档推断。实现者可直接依赖，但 smoke 阶段仍须复验（凭证与账号状态可能不同）。

### 2.1 MiniMax-M3 传输层

| 项 | 实测值 |
|---|---|
| Base URL | `https://api.minimaxi.com/anthropic` |
| Model ID | `MiniMax-M3` |
| 凭证 | 沿用 `MINIMAX_API_KEY` |
| Tool calling（raw） | ✅ `stop_reason: tool_use`，`enum` 约束被遵守 |
| Tool calling（LangChain） | ✅ `bind_tools` → `response.tool_calls` 正常 |
| `usage_metadata` | ✅ 完整返回 input/output/total |
| 多轮 round trip | ✅ 工具结果回传后可继续 |

走 **Anthropic 格式**是被依赖规则决定的，不是偏好：`langchain_openai` 未安装且不在 `pyproject.toml`，`langchain_anthropic>=1.4.3` 已是主依赖。受测模型必须是 LangChain `BaseChatModel`，因为 `quiz_master_agent.py` 依赖 `llm.bind_tools()` 与 `response.tool_calls`。

### 2.2 Interleaved Thinking（推翻了初稿的判断）

| 配置 | 结果 |
|---|---|
| 默认 | `content` blocks = `['text','tool_use']`，**无 thinking** |
| `thinking={"type":"enabled","budget_tokens":1024}` | blocks = `['thinking','text','tool_use']`，thinking 1260 字符 |
| 工具结果回传后的第二轮 | blocks = `['thinking','text']` ← **interleaved thinking 确实生效** |
| thinking block 是否保留在 `AIMessage` 并回传 | ✅ |

**结论：M3 的 interleaved thinking 在该端点默认关闭，须显式启用。**

初稿曾写「必须显式 `thinking: disabled` 以对齐既有 cell」——方向是反的。默认即为关闭；真正的设计问题是**要不要开**（见 §4.2）。

**agent loop 代码无需修改。** `planner_agent.py` / `quiz_master_agent.py` 的 `messages.append(response)` 附加整个 `AIMessage` 对象，已经满足 MiniMax 文档「必须完整回传 assistant 消息以保持思维链连续性」的要求。实现者**不得**为「提取纯文本」而改写这一行。

启用 thinking 时 `temperature` 须为 1（Anthropic 协议约束）。这与既有 cell 的 `temperature=0.7` 不同，必须记入 manifest 并在报告中声明。

### 2.3 凭证隔离（本机特有风险）

本机环境存在 `ANTHROPIC_AUTH_TOKEN` 与 `ANTHROPIC_BASE_URL`，它们属于**开发工具自身的会话配置**，不是 MiniMax 凭证。

实测：显式传入 `base_url` / `api_key` 时，响应模型为 `MiniMax-M3`，未误打到环境变量指向的端点。

**要求：一律显式传参，禁止依赖环境隐式解析。** 若实现依赖 ambient env，评测会产出「看起来正常但完全无效」的数据。

### 2.4 DeepSeek

| 项 | 实测值 |
|---|---|
| 可用模型 | `deepseek-v4-flash` / `deepseek-v4-pro` / `deepseek-v4-flash-vision-exp` |
| 凭证 | `DEEPSEEK_API_KEY` |
| thinking 默认 | **开启** |
| `thinking={"type":"disabled"}` | 被接受 |

关键实测（同一 judge 提示）：

| 配置 | reasoning tokens | completion 合计 | content | finish_reason |
|---|---|---|---|---|
| thinking ON, `max_tokens=200` | 200 | 200 | **空** | **`length`** |
| thinking ON, `max_tokens=4000` | 613 | 621 | `{"question_quality":2}` | `stop` |
| thinking OFF, `max_tokens=4000` | 0 | 7 | `{"question_quality":2}` | `stop` |

两点结论：

1. **空 content 是预算不足导致的截断，不是模型缺陷。** 中间那次探测证明给足预算即可正常输出。
2. 同一判定下，关闭 thinking 的 output token 是开启的 **1/89**，且分数相同（单样本，不构成质量结论）。

### 2.5 Shell 环境

`DEEPSEEK_API_KEY` 定义在 `~/.zshrc`，而 `.zshrc` 仅由**交互式** zsh 读取。非交互式 shell（多数自动化环境）取不到。

**要求：运行前 preflight 必须显式验证三把凭证可用**，不能假设它们存在。

---

## 3. 必须修改的 harness 缺陷

以下均为实测确认的既有缺陷，**在正式付费矩阵之前**必须修复。

### 3.1 路径安全（阻塞 smoke）

P2.3 的 `session_key` 是**原始字符串**而非哈希：

```python
session_key = f"{model}|{mode}|{q['id']}|r{r}"        # matrix.py:47
engine = create_engine(f"sqlite:///{parent}/eval_{spec.session_key}.db")
```

（注意：P2.2 用的是 md5 哈希，P2.3 不同。）

若 model 字符串含 `/`，`eval_anthropic/MiniMax-M3|....db` 中的 `/` 会被解释为目录分隔符，写入失败。

**要求：provider identity 与 storage identity 分离。**

- 逻辑标识：`provider` / `protocol` / `model` 三个字段
- 存储标识：filesystem-safe key 或哈希

`_run_id` 将 model 字符串纳入哈希（`_run_id(model, mode, thinking, query_id, turn_idx, run_idx)`），因此新模型自然产生新 run_id，**既有 396 条的续跑性不受影响**。实现者不得改变既有 Ollama 模型的 run_id 输入。

### 3.2 Judge fail-closed（阻塞正式运行）

现状（`judges.py`）：

```python
resp = await client.chat.completions.create(...)   # 未设 max_tokens
...
v = parsed.get(dim, 3)      # 解析失败 → 每维补 3 → score = 0.6
```

`finish_reason` 完全未被检查。截断 → content 空 → 解析失败 → 静默写入 0.6。

**要求，按重要性排序：**

1. **记录并检查 `finish_reason` / `stop_reason`，非 `stop` 一律记为 judge failure**，不得进入分数统计。这是根本要求：截断可能出于任何原因，不只是 thinking。
2. **保存 judge 原始响应、usage、模型 ID、thinking 配置。** 没有它就无法事后查证。
3. **显式设置 `max_tokens`，并对 DeepSeek 显式 `thinking: disabled`。** 这是成本与确定性选择，不是正确性要求。
4. 解析失败、缺维度、异常 → 记为 failure，**不得自动补 3**。
5. 先持久化完整 candidate，再分别执行各 judge，允许 judge-only 重试。

### 3.3 Deterministic usage 记账（阻塞成本结论）

实测：198 条 deterministic 记录**全部** `input_tokens = output_tokens = 0`；agent_loop 亦有 78 条为 0。

在修复前，本实验**不得**产出任何跨 mode 的成本结论。云端路径的 `usage_metadata` 可用（§2.1），但只覆盖一侧。

### 3.4 Session-atomic 续跑（阻塞正式运行）

`savers: dict[str, InMemorySaver]` 是进程内字典。中断后按单行 `run_id` 续跑时，`pending` 只剩 turn 1，`savers` 从空开始 → turn 1 拿到空 checkpointer → `active_quiz_question_id` 丢失 → GRADE 轮失效。

**要求：改为 session 原子续跑（整个 session 要么全跑要么全跳），或换用 durable checkpointer。**

---

## 4. 实验设计

### 4.1 计数与统计处理

来自 `queries.json`：10 个 single-turn + 2 个 multi-turn（各 2 轮），`runs=3`。

| 单位 | 每 mode | 两 mode 合计 |
|---|---|---|
| invocation records | 42 | **84** |
| 独立 session | 36 | **72** |

- primary cells：8 → **10**
- 既有 396 条**已包含** gemma4:e4b 的 thinking appendix（该模型 72 条/mode，其余 42 条/mode）。加入主 cell 后矩阵总记录数为 **480**；若同时执行 §4.2 的 M3 appendix，则为 **540**。

**统计要求：**

- **不得将 84 条视为 84 个独立样本。** multi-turn 的两轮同属一个 session，相关。按 query/session 配对或聚类。
- **`GENERATE` 与 `GRADE` 分开报告。** `graph.py:quiz_node` 中 GRADE 轮恒定走 deterministic 分支，与 mode 无关。

### 4.2 Thinking 双轨（沿用既有先例）

矩阵中已有先例：`gemma4:e4b` 主矩阵 42 条 + thinking appendix 30 条（仅 single-turn × 3 runs）= 72 条/mode。

M3 采用同一形状：

| 轨 | thinking | 记录数 | 回答什么 |
|---|---|---|---|
| 主 cell | 关闭 | 84 | 与既有 `reasoning=False` cell 对齐，隔离「能力层」单一变量 |
| appendix | 启用 | 60（30/mode，仅 single-turn） | M3 as designed 的真实 agentic 能力 |

两轨不得合并成一个数字。thinking 启用时 `temperature=1`，与主轨的 0.7 不同，必须在报告中声明为已知差异。

appendix 为可选：若 smoke 显示 thinking 启用后成本或延迟不可接受，可只跑主 cell，但须在报告中记录该决定与原因。

### 4.3 两侧必须冻结相同的

queries 与 repeats、budget、**生产 retriever**（`_build_default_retriever()`，P2.3 既有做法）、persistence、failure accounting、rubric（`QUIZ_DIMENSIONS`）、prompt。

`quiz_master.py` 与 `quiz_master_agent.py` 在本次实验中**必须字节级不变**——这是 P2.2/P2.3 已经守过的边界纪律，保证 A/B 公平。

---

## 5. Judge 配置与效度边界

### 5.1 三 judge 角色

| Judge | 角色 | 用途 |
|---|---|---|
| `qwen2.5:7b`（本地） | **legacy bridge** | 唯一在全部 10 个 cell 都存在的 judge；**跨 cell 比较只用它** |
| `deepseek-v4-pro`，显式 `thinking: disabled` | **预注册 primary subjective judge** | 新 pair 的主观质量判定 |
| `MiniMax-M2.7`（云端） | **同厂 sensitivity signal** | 与受测者同家族，**不作为独立证据** |

**分歧处理：DeepSeek 与 qwen 方向相反时，报告 `judge disagreement / inconclusive`，不得强行采用 DeepSeek。** DeepSeek 不是「中立真值」，它是一个预注册的主观判定者。

### 5.2 Construct validity：两条线必须分开

实测确认：GENERATE 轮送给 judge 的文本形如

```
📝 Quiz on HyDE:
<题干>
A) … B) … C) … D) …
Reply with A
```

**不含答案、不含解释**（P4.5 刻意的防泄漏加固）。而 quiz rubric 的五维中有 `answer_correctness` 与 `explanation_clarity`——**五分之二的维度在评一段文本里不存在的东西**。

这是 P2.3 既有缺陷，非本次引入。处理方式：

| 线 | 输入 | 可比性 | 可宣称 |
|---|---|---|---|
| **legacy comparability** | 沿用现有 payload | 与既有 8 cell 可比 | 「user-visible quiz prompt quality」，**不得**宣称评了 answer correctness |
| **structured artifact validity** | 新增 `quiz-artifact-v1`，含 question / options / answer / explanation / evidence | 与既有 cell **不可比** | 五维 rubric 的完整效度 |

**新 cell 两条线都评，分开报告，不得合成单一数字。**

### 5.3 历史 cloud judge 异常（未解释）

既有数据中存在统计异常：

| 矩阵 | `score==0.6` 且全维度弱 且 reasoning 为空 | 同型但 reasoning 非空 |
|---|---|---|
| P2.2 | 80 | 1 |
| P2.3 | 109 | 1 |

`80:1` 与 `109:1` 的比例难以用「真实的全 3 判定」解释——真实低分判定确实存在且落在 0.2（P2.2 73 条、P2.3 65 条），形态不同。该模式与 `judges.py` 解析失败回退（每维补 3 → 0.6）的签名一致。

**但成因未被证实。** 三次针对 M2.7 的机制复现探测全部正常解析，最初的假设（thinking 块内的大括号干扰贪婪正则）未能重现。原始 judge 响应从未保存、`final_text_excerpt` 又被截断，因此**无法从冻结产物回溯查证**。

**要求：**

- 新 cell **不与既有 cell 做 cloud-judge 层面的比较**；跨 cell 比较只用 §5.1 的 legacy bridge。
- 该异常必须写入 EVAL.md limitations。
- 这是 §3.2 fail-closed 的实证理由：若当初记录的是解析失败标志而非静默补 3，今天就有答案。

### 5.4 为什么不能回评历史数据

`single_run.py:110` 存档时 `final_text[:500]`；judge 在 `:82-83` 收到的是**完整** `final_text`。

因此：

- **既有各 cell 的 judge 分数彼此可比**（都基于完整文本）
- **离线重评不可行**：P2.3 有 149/396（38%）的存档卡在 500 字符上限，重评等于用更短的输入打分

准确表述是：**仅凭冻结 JSONL，无法完整、可靠地重评全部历史记录。**

**要求：新 cell 存完整文本**，使未来重评成为可能。既有 396 条维持原样。

---

## 6. Smoke test（正式运行前的门禁）

先跑 6–8 个请求，覆盖两种 mode 与 GENERATE/GRADE。

### 6.1 必须测量并回报

- tool-calling 端到端可用（`bind_tools` → `tool_calls` → 工具结果回传 → 第二轮）
- 延迟（与既有 cell 对照）
- 实际 token 与换算成本，thinking on/off 两种配置分别测
- thinking block 是否按 §2.2 预期出现与回传
- 三把凭证 preflight 通过

### 6.2 中止条件

出现以下任一，**停止，不进入完整矩阵**：

- tool-calling round trip 失败
- 主轨（thinking 关闭）响应中仍出现 thinking block → 记为 cross-matrix confound
- 成本外推超过上限
- 任一 judge preflight 失败

不得「静默继续」。

---

## 7. 凭证与成本边界

- 三把 key 只从环境变量读取：`MINIMAX_API_KEY`（M3 生成 + M2.7 judge）、`DEEPSEEK_API_KEY`（primary judge）
- **不进文件、不进 commit、不进日志、不进 JSONL artifact**
- manifest 记录规范化的 provider / protocol / model / thinking 配置与 endpoint fingerprint，**不记录完整 base URL 或 key**
- 不记录原始 thinking 内容到 artifact（可记录长度与 token 数）
- 不记录未经清洗的 provider 异常
- 成本上限由执行者在运行前设定并写入 manifest；超限即中止

---

## 8. 本实验能与不能宣称什么

### 8.1 可以宣称

> 在当前冻结语料与配置下，agent-loop 相对 deterministic 的效果，在 MiniMax-M3 这一云端能力层是否仍然成立。

### 8.2 不可以宣称

- **不得**宣称 M3 优于既有本地模型。既有实验没有 corpus fingerprint，DeepSeek 也没有评过旧数据。
- **不得**将主轨（thinking 关闭）结果当作 M3 的真实能力上限。
- **不得**用新 cell 的 cloud-judge 数字与既有 cell 比较（§5.3）。
- **不得**在 §3.3 修复前产出跨 mode 成本结论。
- **不得**将 legacy 线的分数解释为 answer correctness 或 explanation clarity 的效度（§5.2）。

---

## 9. 执行顺序

1. §3.1 路径安全 + §2.3 显式凭证 + §6.1 preflight（阻塞 smoke）
2. **Smoke test**，回报 §6.1 全部测量值 → 决定是否继续
3. §3.2 judge fail-closed、§3.3 usage 记账、§3.4 session-atomic 续跑（阻塞正式运行）
4. 主 cell 84 条
5. thinking appendix 60 条（可选，视 smoke 结果）
6. 写入 EVAL.md，含 §5.2 双线与 §5.3 异常声明

---

## 10. 完成标准

1. 全量 backend pytest、frontend Vitest、production build、CI 通过。
2. 既有 396 条 × 2 矩阵字节级未变。
3. Learning Run 的 `experiment_axes` 未变。
4. 新数据写入独立、带版本的 artifact，含 schema version、完整输出与 hash、rubric hash、模型/协议/thinking 配置、corpus fingerprint、usage 与价格快照。
5. 任何 judge failure 均可从 artifact 区分于真实低分。
6. 报告明确区分 legacy comparability 与 structured artifact validity 两条线。
7. §8.2 的任一禁止宣称均未出现在文档中。
