# Study Coach

> HKBU_StudyCompanion 课程项目的 portfolio-grade 重构 —— Exam Coach Agent。
> 当前已推进到 P3 productized shell：后端 agent / eval 基础稳定，前端 7-view app 已交付。完整阶段记录见 `docs/ROADMAP.md`。

## Portfolio Thesis

`../HKBU_StudyCompanion 2/` 是一份 Prompt Engineering 课程项目：本地 PDF RAG、HyDE、CoT study plan、MCQ quiz、LLM-as-Judge，但架构本质是 **chain-of-prompts**。

`../JadeAI/` 是参考工程：BYOK header、repository pattern、DB persistence、tool-calling agent loop、长期维护型 `ARCHITECTURE.md`。

`study-coach/` 的目标是把课程项目重构成一个可放进个人简历的 **Exam Coach Agent**：旧四件套保留，但改造成 LangGraph router + tools + memory + judge + eval 的现代 agent 产品。

## Stack

- **Backend**: Python 3.11 · FastAPI · LangChain · LangGraph · Chroma · SQLAlchemy 2.x · Alembic · uv
- **Retrieval**: Dense embedding · BM25 · RRF · cross-encoder reranking (`jina-reranker-v2-base-multilingual`)
- **Frontend**: Vue 3 · TypeScript · Vite · Pinia · Tailwind 4 · vue-router · Chart.js · Mermaid
- **LLM**: dual-track —— Ollama 本地（默认）/ BYOK 云端（OpenAI · Anthropic · Gemini）via HTTP headers

## Current Capabilities

1. 上传 PDF → concat-then-split chunking → Chroma 持久化索引
2. Chat 输入问题 → hybrid retrieval + rerank → LangGraph Tutor → Judge Guard → SSE 流式回答
3. Citation 显示 source / page / span metadata
4. Planner：生成 study plan、持久化 milestones、输出 mindmap
5. QuizMaster：RAG-grounded MCQ 生成、自动 grading、错题记录、mastery 更新
6. Memory Updater：hydrator / writer 节点读写 mastery、mistakes、plan state
7. Mode dispatch：Plan / Quiz 支持 `deterministic` 与 `agent_loop` 双路径 A/B
8. P3 frontend：Overview / Chat / Plan / Quiz / Mistake Bank / Library / Settings 七页产品壳

## What This Project Demonstrates

- **From prompt pipeline to agent graph**: fixed chain-of-prompts → LangGraph router → Tutor / Planner / QuizMaster branches → Judge → Memory Writer.
- **JadeAI patterns ported to Python**: BYOK header, repository-style DB access, tool wrappers, persistence-first architecture, contract docs.
- **Agent loop is treated empirically**: P2.2 / P2.3 compare `deterministic` vs `agent_loop` across small Ollama models instead of assuming agent loop is always better.
- **Product around eval**: P3 UI exposes mode switching and alignment-safety behavior so the research result is visible in the app, not only in docs.

## Quickstart

### Prerequisites

```bash
# Backend
brew install uv

# Frontend
brew install pnpm node

# LLM (default local)
brew install ollama
ollama pull gemma3:4b nomic-embed-text
ollama serve   # leave running
```

### Run

```bash
# Terminal 1 — backend (port 8000)
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (port 5173)
cd frontend
pnpm install
pnpm dev
```

Visit <http://localhost:5173>. Library → upload a PDF → Chat / Plan / Quiz.

### Tests

```bash
cd backend
uv run pytest -q        # 214 tests, no live Ollama required for the suite

cd ../frontend
pnpm build              # typecheck + production build
```

## BYOK (cloud providers)

Open Settings, pick provider, paste your API key. The key lives only in your browser's `localStorage` and is sent per-request via `x-api-key` header. The server never persists it (same pattern as JadeAI).

Supported headers:

- `x-provider`: `ollama` / `openai` / `anthropic` / `gemini`
- `x-model`: provider-specific model id
- `x-api-key`: required for cloud providers
- `x-base-url`: optional custom endpoint
- `x-judge-model`: optional cross-model Judge Guard
- `x-planner-mode` / `x-quiz-mode`: `deterministic` / `agent_loop`

## Key Results

- **P2.0 Retrieval rebuild**: Hit@5 `0.733 → 0.933`, MRR `0.633 → 0.822` on 15 HKBU eval queries.
- **P2.2 Plan ablation**: agent loop helps thinking/tool-capable models, but adds 3-20x latency and can underperform deterministic on weaker tool-use models.
- **P2.3 Quiz ablation**: strict schemas invert the P2.2 rescue effect; agent loop has better per-success quality but lower completion rate on small models.
- **Alignment-safety finding**: agent loop with `retriever_search` can refuse off-corpus quiz generation instead of fabricating, which P3 surfaces via `EmptyCorpusBanner`.

Full reports:

- `docs/PHASE_2.0_REPORT.md`
- `docs/EVAL.md`
- `docs/agent_loop_vs_deterministic.md`
- `docs/quiz_ablation_followup.md`
- `docs/p3_frontend_productize.md`

## Layout

```
study-coach/
├── backend/
│   ├── app/
│   │   ├── main.py                         # FastAPI app factory + retriever wiring
│   │   ├── api/{routes,deps}.py            # /api/chat, documents, plans, mistakes, mastery
│   │   ├── agent/
│   │   │   ├── graph.py                    # LangGraph: memory → router → tutor/quiz/plan → judge
│   │   │   ├── planner{,_agent}.py         # deterministic + agent_loop plan paths
│   │   │   ├── quiz_master{,_agent}.py     # deterministic + agent_loop quiz paths
│   │   │   ├── judge.py                    # Tutor / Quiz / Plan rubrics
│   │   │   └── tools/                      # Pydantic tool schemas + side-effect tools
│   │   ├── rag/                            # dense, BM25/RRF hybrid, reranking retriever
│   │   ├── llm/provider.py                 # BYOK headers → LLMConfig → chat model
│   │   ├── db/{models,repositories,session}.py
│   │   ├── eval/                           # P2.2 / P2.3 ablation harnesses
│   │   └── srs/sm2.py                      # Mistake Bank scheduling
│   └── tests/                              # 214 backend tests
├── frontend/
│   └── src/
│       ├── views/                          # Overview / Chat / Plan / Quiz / Mistakes / Library / Settings
│       ├── components/                     # ModeChip, MCQCard, MindmapPanel, RadarChart, ...
│       ├── stores/                         # Pinia resource stores
│       └── lib/                            # API, parsing, fingerprint
├── design-system/MASTER.md                 # P3 visual system
└── docs/
    ├── ARCHITECTURE.md                     # contract-level spec
    ├── ROADMAP.md                          # phase history + backlog
    └── EVAL.md                             # judge + ablation reports
```

## Phase Status

- [x] **Phase 0** — `PROJECTS_OVERVIEW.md` (HKBU_StudyCompanion vs JadeAI comparison)
- [x] **Phase 0.5** — `docs/ARCHITECTURE.md` + `docs/ROADMAP.md` (contract-level specs)
- [x] **Phase 1** — Minimal closed loop: upload PDF → chat → streamed answer + citation
- [x] **Phase 2.0** — Retrieval foundation rebuild: hybrid retrieval + reranker + eval harness
- [x] **Phase 2.1** — Agent-ify: router, Judge Guard, memory schema, QuizMaster, Planner
- [x] **Phase 2.2** — Plan agent loop ablation
- [x] **Phase 2.3** — Quiz agent loop ablation
- [x] **Phase 3** — Productized frontend shell: 7 views + dashboard / quiz / plan / mistake bank
- [ ] **Phase 4** — deploy, ARCHITECTURE.md v2, demo readiness, i18n, mobile, shared plans

## Origin

This is a portfolio refactor of the **HKBU_StudyCompanion** class project (course: COMP4146/7125 Prompt Engineering). The original (Gradio + chain-of-prompts) is preserved at `../HKBU_StudyCompanion 2/`.

Reference engineering patterns are drawn from `../JadeAI/`: BYOK header pattern, repository pattern, persistent DB state, tool-calling loop, AI output hardening, and long-form architecture documentation.
