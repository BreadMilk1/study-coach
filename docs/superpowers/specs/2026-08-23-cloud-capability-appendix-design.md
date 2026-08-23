# Cloud Capability Appendix — Design Spec

> 日期：2026-08-23（rev 2，含 Codex review 修订）
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
- **不修改既有 396 条记录。** 字节级不变，验收基准见 §10。
- **不回评历史数据。** 理由见 §5.4。
- **不修改 production 代码。** `app/agent/judge.py`、`app/agent/tools/quiz.py`、`quiz_master.py`、`quiz_master_agent.py` 均须字节级不变。所有适配发生在 eval 包内。
- **不新增 dependency。**

---

## 2. 已验证的协议事实

以下全部为 2026-08-23 实测。实现者可直接依赖，但 §6 阶段仍须复验（凭证与账号状态可能不同）。

### 2.1 MiniMax-M3 传输层

| 项 | 实测值 |
|---|---|
| Base URL | `https://api.minimaxi.com/anthropic` |
| Model ID | `MiniMax-M3` |
| 凭证 | 沿用 `MINIMAX_API_KEY` |
| Tool calling（raw + LangChain） | ✅ `bind_tools` → `response.tool_calls` 正常 |
| `usage_metadata` | ✅ 完整返回 input/output/total |
| 多轮 round trip | ✅ 工具结果回传后可继续 |

走 **Anthropic 格式**是被依赖规则决定的，不是偏好：`langchain_openai` 未安装且不在 `pyproject.toml`，`langchain_anthropic>=1.4.3` 已是主依赖。受测模型必须是 LangChain `BaseChatModel`，因为 `quiz_master_agent.py` 依赖 `llm.bind_tools()` 与 `response.tool_calls`。

**兼容性声明的强度：** M3 经由该端点的可用性是 **account-level empirically verified**，不是官方稳定性保证。MiniMax 的 Anthropic-compatible 参考页与面向 Agent 的 M3 文档对模型列表的表述不完全一致。§6 的 protocol probe 是每次运行前的必须门禁，不得因本节已验证而跳过。

### 2.2 Interleaved Thinking

| 配置 | 实测结果 |
|---|---|
| 默认（不传 `thinking`） | blocks = `['text','tool_use']`，**无 thinking** |
| `thinking={"type":"enabled","budget_tokens":1024}` | blocks = `['thinking','text','tool_use']`，thinking 1260 字符 |
| 工具结果回传后的第二轮 | blocks = `['thinking','text']` ← **interleaved thinking 生效** |
| thinking block 保留在 `AIMessage` 并回传 | ✅ |
| `temperature=0.7` + thinking | ✅ **被接受**（blocks = `['thinking','text']`，`stop_reason=end_turn`） |

**M3 的 interleaved thinking 在该端点默认关闭，须显式启用。**

rev 1 曾写「必须显式 `thinking: disabled` 以对齐既有 cell」——方向反了。默认即为关闭；真正的设计问题是要不要开（§4.2）。

rev 1 亦曾写「启用 thinking 时 temperature 须为 1」——**实测证伪**。0.7 被接受，因此 thinking appendix 可与主轨共用 `temperature=0.7`，是**干净的单变量比较**，不存在 rev 1 声称的 confound。

**agent loop 代码无需修改。** `quiz_master_agent.py` 的 `messages.append(response)` 附加整个 `AIMessage` 对象，已满足 MiniMax 文档「必须完整回传 assistant 消息以保持思维链连续性」的要求。实现者**不得**为「提取纯文本」而改写这一行。

### 2.3 thinking 启用会破坏 deterministic 解析（关键）

`app/agent/tools/quiz.py:151`：

```python
raw = getattr(response, "content", "") or ""
```

thinking 启用时 `response.content` 是 **block list** 而非字符串，该行会取到列表，下游 JSON 解析失效。

**要求：** eval 包提供一个 provider adapter，在进入 deterministic 路径前把 block list 扁平化为文本（丢弃 thinking block，保留 text block）。**`quiz.py` 保持字节级不变。** 若实现者选择让 thinking appendix 只覆盖 agent_loop 一侧，须在报告中声明该缩减及原因。

### 2.4 完成状态的两套词汇

生成侧与评分侧走不同协议，**不能共用一个字面值判断**：

| 侧 | 协议 | 观测到的完成状态 |
|---|---|---|
| Generator（MiniMax M3） | Anthropic | `end_turn`、`tool_use`、`max_tokens` |
| Judge（DeepSeek） | OpenAI-compatible | `stop`、`length` |

**要求：** eval 包把 provider 响应规范化为三态 `completed` / `truncated` / `failed`，并保留经清洗的原始 finish 元数据。rev 1 写的「非 `stop` 一律记为 failure」是把两套词汇混为一谈，已作废。

### 2.5 凭证隔离

实测：显式传入 `base_url` / `api_key` 时，响应模型为 `MiniMax-M3`，未受同名环境变量影响。

**要求：一律显式传参，禁止依赖环境隐式解析。** 运行环境可能存在属于其他工具的同名变量；若实现依赖 ambient env，评测会产出「看起来正常但完全无效」的数据。

### 2.6 DeepSeek

| 项 | 实测值 |
|---|---|
| 可用模型 | `deepseek-v4-flash` / `deepseek-v4-pro` / `deepseek-v4-flash-vision-exp` |
| 凭证 | `DEEPSEEK_API_KEY` |
| thinking 默认 | **开启** |
| `thinking={"type":"disabled"}` | 被接受 |

同一 judge 提示下：

| 配置 | reasoning tokens | completion 合计 | content | finish_reason |
|---|---|---|---|---|
| thinking ON, `max_tokens=200` | 200 | 200 | **空** | **`length`** |
| thinking ON, `max_tokens=4000` | 613 | 621 | `{"question_quality":2}` | `stop` |
| thinking OFF, `max_tokens=4000` | 0 | 7 | `{"question_quality":2}` | `stop` |

两点结论：

1. **空 content 是预算不足导致的截断，不是模型缺陷。**
2. 关闭 thinking 的 output token 是开启的 **1/89**，分数相同（单样本，不构成质量结论）。

### 2.7 凭证与运行环境

- **两把独立 key，三个 provider role**：`MINIMAX_API_KEY` 同时服务 M3 生成与 M2.7 judge；`DEEPSEEK_API_KEY` 服务 primary judge。
- 环境变量可能只在交互式 shell 中可见。**preflight 必须显式验证三个 role 均可用**，不得假设存在。

---

## 3. 必须修改的 harness 缺陷

均为实测确认的既有缺陷。

### 3.1 路径安全

P2.3 的 `session_key` 是**原始字符串**而非哈希（P2.2 用 md5，两者不同）：

```python
session_key = f"{model}|{mode}|{q['id']}|r{r}"        # matrix.py:47
engine = create_engine(f"sqlite:///{parent}/eval_{spec.session_key}.db")
```

model 字符串含 `/` 时会被解释为目录分隔符，写入失败。

**要求：** provider identity（`provider` / `protocol` / `model` 三字段）与 storage identity（filesystem-safe key 或哈希）分离。

`_run_id(model, mode, thinking, query_id, turn_idx, run_idx)` 将 model 纳入哈希，因此新模型自然产生新 run_id，**既有 396 条的续跑性不受影响**。实现者不得改变既有 Ollama 模型的 run_id 输入。

### 3.2 两阶段持久化状态机

rev 1 把 candidate-first、judge-only retry、session-atomic resume 列为三条独立要求。**冻结为一个两阶段设计：**

**Phase 1 — Generator（session 原子）**

- 以 session 为原子单位跑完全部 turns；session 未完成即视为未开始，重跑整个 session。
- 完成后写入 **immutable CandidateArtifact**（含完整输出、非截断）。
- 这样既解决了 `savers: dict[str, InMemorySaver]` 进程内字典导致的续跑丢失 `active_quiz_question_id` 问题，**又不需要引入 durable checkpointer 依赖**。

**Phase 2 — Scoring（从 candidate 独立追加）**

- 各 ScorerExecution 从已冻结的 candidate 读取输入，独立追加，可单独重试，**不重新生成**。
- 每条 ScorerExecution 记录：规范化完成状态（§2.4）、原始响应、usage、模型 ID、thinking 配置、rubric hash。
- 解析失败、缺维度、异常、`truncated` → 记为 **failure**，不得自动补 3、不得进入分数统计。
- 显式设置 `max_tokens`；DeepSeek 显式 `thinking: disabled`（成本与确定性选择，非正确性要求）。

### 3.3 Deterministic usage 记账

实测：198 条 deterministic 记录**全部** `input_tokens = output_tokens = 0`；agent_loop 亦有 78 条为 0。

在修复前，**不得**产出任何跨 mode 的成本结论。

---

## 4. 实验设计

### 4.1 计数与统计单位

来自 `queries.json`：10 个 single-turn + 2 个 multi-turn（各 2 轮），`runs=3`。

| 单位 | 每 mode | 两 mode 合计 |
|---|---|---|
| invocation records | 42 | **84** |
| 独立 session | 36 | **72** |

- primary cells：8 → **10**
- 既有 396 条**已包含** gemma4:e4b 的 thinking appendix（该模型 72 条/mode，其余 42 条/mode）。加入主 cell 后总记录数 **480**；若同时执行 §4.2 appendix，则 **540**。

**统计要求：**

- **不得将 84 条视为 84 个独立样本。** multi-turn 两轮同属一个 session。
- **`GENERATE` 与 `GRADE` 分开报告。** `graph.py:quiz_node` 中 GRADE 恒定走 deterministic 分支，与 mode 无关。

### 4.2 预注册判定规则

**配对单位：** 36 个 `(query_id, run_idx)` GENERATE session，跨 mode 配对。

**分别报告四个指标，不合成总分：**

| 指标 | 样本 |
|---|---|
| persistence / completion rate | 全部 36 对 |
| visible quality（`legacy-visible-v1`） | 全部 GENERATE rows |
| structured quality（`quiz-artifact-v1`） | 仅两侧均成功持久化的配对，**必须同时报告保留率** |
| latency / cost | 全部（cost 须 §3.3 修复后方可报告） |

保留率是防止成功样本选择偏差的必要条件：若 agent_loop 只在容易的 case 上成功，其 structured quality 会被高估。

**判定：** 对每个指标做 cluster bootstrap（按 session 聚类）。

- 区间全为正 → 支持「结论在该能力层成立」
- 区间全为负 → 反驳
- 跨零，或 judge 方向冲突（§5.1）→ **inconclusive**

**不得**汇总出一个「总冠军」。

### 4.3 Thinking 双轨

矩阵已有先例：`gemma4:e4b` 主矩阵 42 条 + thinking appendix 30 条（仅 single-turn × 3 runs）= 72 条/mode。M3 采用同一形状：

| 轨 | thinking | 记录数 | 回答什么 |
|---|---|---|---|
| 主 cell | 显式 `disabled` | 84 | 与既有 cell 对齐，隔离「能力层」单一变量 |
| appendix | `enabled` | 60（30/mode，仅 single-turn） | M3 as designed 的真实 agentic 能力 |

- 主轨**显式请求 `thinking: disabled`**，不依赖当前默认值（默认值可能变）。
- appendix 优先使用官方 `adaptive` 模式；若采用固定 `enabled + budget_tokens`，须在报告中称为 **fixed-budget thinking appendix**，不得简称为「thinking on」。
- 两轨共用 `temperature=0.7`（§2.2 实测支持），因此是干净的单变量对比。
- appendix 需**单独批准**（§7 成本）。若 smoke 显示成本或延迟不可接受，可只跑主 cell，须记录该决定与原因。

### 4.4 冻结项

两侧必须相同：queries 与 repeats、budget、persistence、failure accounting、prompt。

**Corpus 冻结（`_build_default_retriever()` 只是复用 factory，不等于冻结语料）：**

- 运行前后记录并比对 **corpus aggregate hash、chunk count、retriever 配置**；不一致即中止。
- 使用 eval-side `RecordingRetriever` 包装，捕获两种 mode 实际检索到的 evidence，写入 artifact。

`quiz_master.py` 与 `quiz_master_agent.py` 本次实验中**字节级不变**——P2.2/P2.3 已守过的边界纪律，保证 A/B 公平。

---

## 5. Judge 配置与效度边界

### 5.1 三个 judge role

| Judge | 角色 | 用途 |
|---|---|---|
| `qwen2.5:7b`（本地） | **descriptive legacy reference** | 唯一在全部 10 个 cell 都存在；跨 cell 的**描述性**参照 |
| `deepseek-v4-pro`，显式 `thinking: disabled` | **预注册 primary subjective judge** | 新 pair 的主观质量判定 |
| `MiniMax-M2.7`（云端） | **同厂 sensitivity signal** | 与受测者同家族，**不作为独立证据** |

qwen 不称「clean bridge」：实测 P2.3 既有数据中有 **12 条** local judge 解析失败回退（`app/agent/judge.py:119` 写入明确标记字串 `Judge output parsing failed`，可确定识别；P2.2 为 0 条）。因此「judge 看到完整文本」不等于「旧分数没有 parser confound」。这 12 条须在报告中排除或单独标注。

**分歧处理：DeepSeek 与 qwen 方向相反时，报告 `judge disagreement / inconclusive`，不得强行采用 DeepSeek。** DeepSeek 不是中立真值，是一个预注册的主观判定者。分歧的判断**只在 `quiz-artifact-v1` 上有意义**——两者必须看到相同输入才可比。

### 5.2 Judge × contract 调用矩阵

rev 1 的「两条线都评」无法确定调用次数，也无法保证 judge 看到相同输入。**冻结为：**

| Contract | 输入 | Judge | 覆盖范围 |
|---|---|---|---|
| `legacy-visible-v1` | 现有 payload（题干 + 选项） | qwen2.5:7b | 全部 GENERATE rows |
| `quiz-artifact-v1` | 结构化：question / options / answer / explanation / evidence | qwen2.5:7b + deepseek-v4-pro + MiniMax-M2.7 | **仅成功持久化的** GENERATE rows |
| — | — | quality scorer 标记 `skipped` | 全部 GRADE rows（只验证 session/persistence） |

`quiz-artifact-v1` 的字段**从持久化的 Question 记录读取**，不是从模型输出文本重新解析。缺 Question 或缺 evidence → 记为 **structured artifact failure**，**不得回退到 legacy payload**。

### 5.3 Construct validity：两条线必须分开

实测确认：GENERATE 轮送给 judge 的文本形如

```
📝 Quiz on HyDE:
<题干>
A) … B) … C) … D) …
Reply with A
```

**不含答案、不含解释**（P4.5 刻意的防泄漏加固）。而 quiz rubric 五维中有 `answer_correctness` 与 `explanation_clarity`——**五分之二的维度在评一段文本里不存在的东西**。这是 P2.3 既有缺陷，非本次引入。

| 线 | 可比性 | 可宣称 |
|---|---|---|
| `legacy-visible-v1` | 与既有 8 cell 可比 | 「user-visible quiz prompt quality」，**不得**宣称评了 answer correctness |
| `quiz-artifact-v1` | 与既有 cell **不可比** | 五维 rubric 的完整效度 |

**分开报告，不得合成单一数字。**

### 5.4 历史 cloud judge 异常（未解释）

| 矩阵 | `score==0.6` 且全维度弱 且 reasoning 为空 | 同型但 reasoning 非空 |
|---|---|---|
| P2.2 | 80 | 1 |
| P2.3 | 109 | 1 |

`80:1` 与 `109:1` 难以用真实的全 3 判定解释——真实低分确实存在且落在 0.2（P2.2 73 条、P2.3 65 条），形态不同。该模式与解析失败回退（每维补 3 → 0.6）的签名一致。

**但成因未被证实。** 三次针对 M2.7 的机制复现探测全部正常解析。原始 judge 响应从未保存，因此**无法从冻结产物回溯查证**。

与 §5.1 的 12 条不同：那 12 条有明确标记字串可确定识别，这批只有统计特征。

**要求：** 新 cell **不与既有 cell 做 cloud-judge 层面的比较**；该异常写入 EVAL.md limitations；这是 §3.2 fail-closed 的实证理由。

### 5.5 为什么不能回评历史数据

`single_run.py:110` 存档 `final_text[:500]`；judge 在 `:82-83` 收到的是**完整** `final_text`。

- 既有各 cell 的 judge 分数**基于完整文本**
- **离线重评不可行**：P2.3 有 149/396（38%）存档卡在 500 字符上限

准确表述：**仅凭冻结 JSONL，无法完整、可靠地重评全部历史记录。**

**要求：新 cell 存完整文本。** 既有 396 条维持原样。

---

## 6. 执行顺序（rev 1 的顺序自相矛盾，已修正）

rev 1 把 smoke 排在 usage/fail-closed 修复之前，但 smoke 本身要求成本外推与有效 judge——逻辑冲突。正确顺序：

| 阶段 | 内容 | 门禁 |
|---|---|---|
| **1. Protocol probe** | 三个 provider role preflight；M3 tool round trip；thinking on/off 行为；DeepSeek thinking disabled。**不产生实验数据** | 任一失败即中止 |
| **2. Harness 修复** | §3.1 路径、§3.2 两阶段状态机与 strict judge、§3.3 usage、§4.4 corpus 冻结与 `RecordingRetriever` | 全量测试 + CI 通过 |
| **3. Harness smoke** | 6–8 个 **smoke cases**（须记录实际 HTTP 调用数，与 case 数不同），覆盖两种 mode 与 GENERATE/GRADE | 见下 |
| **4. 人工成本批准** | 用 smoke 实测外推主矩阵成本，人工确认 | 超 §7 上限即中止 |
| **5. 主 cell** | 84 条 | |
| **6. thinking appendix** | 60 条，**单独批准** | |
| **7. 写入 EVAL.md** | 含 §5.3 双线、§5.4 异常、§5.1 的 12 条声明 | |

### 6.1 Smoke 必须测量并回报

- tool-calling 端到端（`bind_tools` → `tool_calls` → 工具结果回传 → 第二轮）
- 延迟（与既有 cell 对照）
- 实际 token 与换算成本，thinking on/off 分别测
- 实际 HTTP 调用数（agent loop 一个 case 会产生多次调用）
- thinking block 是否按 §2.2 预期出现与回传
- corpus hash 前后一致

### 6.2 中止条件

出现以下任一，**停止，不进入完整矩阵**：

- tool-calling round trip 失败
- 主轨（显式 `thinking: disabled`）响应中仍出现 thinking block → 记为 cross-matrix confound
- 成本外推超过 §7 上限
- 任一 judge preflight 失败
- corpus hash 前后不一致

不得静默继续。

---

## 7. 凭证与成本上限

### 7.1 成本上限（写死，不交由执行者临时决定）

| 阶段 | 上限 |
|---|---|
| Protocol probe + smoke | **$1** |
| 主 cell 84 条 | **$5** |
| thinking appendix（需单独批准） | 额外 **$5** |
| **combined hard cap** | **$10** |

smoke 外推若显示任一阶段将超出对应上限，中止并重新评估，不得「先跑看看」。

### 7.2 凭证处理

- 两把 key 只从环境变量读取，服务三个 provider role（§2.7）
- **不进文件、不进 commit、不进日志、不进 artifact**
- manifest 记录**公开的**规范化 endpoint / region / provider / protocol / model / thinking 配置。base URL 不是秘密，可记录；**key 不可**
- 不记录原始 thinking 内容到 artifact（可记录长度与 token 数）
- 不记录未经清洗的 provider 异常

---

## 8. 本实验能与不能宣称什么

### 8.1 可以宣称

> 在当前冻结语料与配置下，agent-loop 相对 deterministic 的效果，在 MiniMax-M3 这一云端能力层是否仍然成立。

### 8.2 不可以宣称

- **不得**宣称 M3 优于既有本地模型。既有实验没有 corpus fingerprint，DeepSeek 也没有评过旧数据。
- **不得**将主轨（thinking disabled）结果当作 M3 的能力上限。
- **不得**用新 cell 的 cloud-judge 数字与既有 cell 比较（§5.4）。
- **不得**在 §3.3 修复前产出跨 mode 成本结论。
- **不得**将 legacy 线分数解释为 answer correctness 或 explanation clarity 的效度（§5.3）。
- **不得**把四个指标合成单一「总冠军」（§4.2）。
- **不得**将 M3 的 Anthropic-compatible 可用性表述为官方稳定保证（§2.1）。

---

## 9. 完成标准

1. 全量 backend pytest、frontend Vitest、production build、CI 通过。
2. **既有冻结产物 SHA-256 未变**（§10）。
3. Learning Run 的 `experiment_axes` 未变。
4. production 代码字节级未变（§1.1 列出的文件）。
5. 新数据写入独立、带版本的 artifact，含 schema version、完整输出与 hash、rubric hash、provider/protocol/model/thinking 配置、corpus fingerprint、实际检索 evidence、usage 与价格快照。
6. 任何 judge failure 均可从 artifact 区分于真实低分。
7. 报告明确区分 `legacy-visible-v1` 与 `quiz-artifact-v1` 两条线。
8. §8.2 的任一禁止宣称均未出现在文档中。
9. **ROADMAP.md、ARCHITECTURE.md、README.md 中的矩阵规模数字同步更新**（396 → 480 或 540；8 cells → 10）。

---

## 10. 冻结产物验收基准

运行前后比对，任一变化即为验收失败：

```
d60837707b105867574e5298cd10cf6a225414f38f31a187cb65b8540868f6ae  p2_2_agent_ablation/output/results.jsonl
2fba0df1ea6e4e540127a07fb6037e48f14d7a9b6d864594a649341e95523bfd  p2_3_quiz_ablation/output/results.jsonl
2a64bea2da11eb22603522b7a75ac2caa6c5092ab18f7a49da59191057aaea94  p2_3_quiz_ablation/output/results_no_retriever.jsonl
```

新 cell 数据写入**独立文件**，不追加到上述任一文件。
