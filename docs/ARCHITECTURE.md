# Study Coach — ARCHITECTURE

> Contract-level spec. Phase 1 code = mechanical translation of this document.
> Final-form ARCHITECTURE (with rationale, ER diagrams, ADRs) ships at Phase 4.

## 1. System Overview

```
                ┌──────────────┐
   User Msg ──► │   Router     │ ─── classify intent
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┬─────────────┐
        ▼              ▼              ▼             ▼
   ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────┐
   │ Planner │   │  Tutor   │   │  Quiz   │   │ Reviewer  │
   │   sub   │   │   sub    │   │ Master  │   │   sub     │
   └────┬────┘   └────┬─────┘   └────┬────┘   └─────┬─────┘
        └─────────────┴──────┬───────┴──────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Judge Guard    │ ◄── citation + grounding
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Memory Updater  │ ─── mastery / mistake / progress
                    └────────┬────────┘
                             ▼
                       streamed output
```

Backend: FastAPI + LangChain + LangGraph + Chroma + SQLAlchemy.
Frontend: Vite + Vue 3 SPA + Pinia.
LLM: dual-track — BYOK cloud (OpenAI / Anthropic / Gemini) **or** local Ollama, switched per-request via headers.

---

## 2. LangGraph State

Single flat `TypedDict`; subgraphs share it. Optional fields use `NotRequired`.

```python
# backend/app/agent/state.py
from typing import Annotated, NotRequired, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class Citation(TypedDict):
    chunk_id: str
    source: str
    page: int
    span_start: int
    span_end: int

class CoachState(TypedDict):
    # identity
    user_id: str
    session_id: str
    # conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # goal + plan context (loaded lazily by Planner)
    goal_id: NotRequired[str]
    current_topic: NotRequired[str]
    # learning state (loaded by Memory Updater on entry, written on exit)
    mastery_scores: NotRequired[dict[str, float]]   # topic_name -> 0.0..1.0
    recent_mistakes: NotRequired[list[str]]          # mistake_ids due soon
    # output augmentation
    citations: NotRequired[list[Citation]]
    tool_call_trace: NotRequired[list[str]]
    # guardrail
    judge_score: NotRequired[float]
    retry_count: NotRequired[int]
```

| Field | Owner node(s) | Lifetime |
|---|---|---|
| `messages` | every node (append-only via reducer) | session |
| `user_id` / `session_id` | injected by API layer | session |
| `goal_id` / `current_topic` | Planner sets; Tutor/QuizMaster reads | session |
| `mastery_scores` / `recent_mistakes` | Memory Updater hydrates / writes | per-turn |
| `citations` | Tutor (via `rag_search` tool) | per-turn |
| `tool_call_trace` | every tool appends name | per-turn debug |
| `judge_score` / `retry_count` | Judge Guard | per-turn |

Checkpointer: `InMemorySaver` in dev, `SqliteSaver`/`PostgresSaver` in prod (env-controlled).

---

## 3. Tool Registry

Nine tools. Each: Pydantic input, Pydantic output, side effect noted. LLM picks which to call.

```python
# backend/app/agent/tools/schemas.py
from pydantic import BaseModel, Field

class RagSearchIn(BaseModel):
    query: str
    top_k: int = 5

class RagSearchOut(BaseModel):
    chunks: list[dict]            # {chunk_id, text, page, score}
    citations: list[Citation]
```

| Tool | Input | Output | Side effect |
|---|---|---|---|
| `rag_search` | `RagSearchIn(query, top_k)` | `RagSearchOut(chunks, citations)` | none (read Chroma) |
| `hyde_rag_search` | `RagSearchIn` | `RagSearchOut` | none; only called when prior `rag_search` score < 0.55 |
| `generate_quiz` | `QuizIn(topic, difficulty: easy/med/hard, n)` | `QuizOut(questions[])` | insert into `questions` |
| `grade_quiz_answer` | `GradeIn(question_id, user_answer)` | `GradeOut(correct, explanation, correct_answer)` | none (validation only) |
| `record_mistake` | `MistakeIn(question_id, user_answer)` | `MistakeOut(mistake_id, srs_due_at)` | insert `mistakes` (SM-2 schedule) |
| `update_mastery` | `MasteryIn(topic, delta)` | `MasteryOut(new_score)` | upsert `mastery` |
| `update_study_plan` | `PlanPatchIn(goal_id, milestones_json)` | `PlanPatchOut(updated_at)` | update `plans.milestones_json` |
| `generate_mindmap` | `MindmapIn(topic)` | `MindmapOut(mermaid_src, markdown_outline)` | none |
| `judge_response` | `JudgeIn(answer, sources)` | `JudgeOut(score: 0..1, verdict, weak_dims[])` | none; called by Judge Guard, not by LLM directly |

Convention: `grade_quiz_answer` → `record_mistake` (if wrong) → `update_mastery` is encouraged by the QuizMaster system prompt, not enforced in code.

---

## 4. Database Schema

SQLAlchemy 2.x style. SQLite dev, Postgres prod (same models, different URL).

```python
# backend/app/db/models.py
from datetime import datetime
from sqlalchemy import Boolean, ForeignKey, JSON, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)         # uuid
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)     # FingerprintJS
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Goal(Base):
    __tablename__ = "goals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200))
    exam_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")     # active/done/abandoned

class Plan(Base):
    __tablename__ = "plans"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id"))
    milestones_json: Mapped[list] = mapped_column(JSON)                   # compatibility cache
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class PlanMilestone(Base):
    __tablename__ = "plan_milestones"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"))
    topic_id: Mapped[str | None] = mapped_column(ForeignKey("topics.id"), nullable=True)
    topic_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    title: Mapped[str] = mapped_column(Text)
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    done: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(20), default="ai")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class PlanEvent(Base):
    __tablename__ = "plan_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"))
    milestone_id: Mapped[str | None] = mapped_column(ForeignKey("plan_milestones.id"), nullable=True)
    actor: Mapped[str] = mapped_column(String(20))                         # user/ai/system
    action: Mapped[str] = mapped_column(String(40))                        # created/completed/reopened/...
    before_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Topic(Base):
    __tablename__ = "topics"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id"))
    name: Mapped[str] = mapped_column(String(200))
    source_chunks: Mapped[list] = mapped_column(JSON)                     # chunk_ids list

class Mastery(Base):
    __tablename__ = "mastery"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)              # 0..1
    last_reviewed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Question(Base):
    __tablename__ = "questions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"))
    prompt: Mapped[str] = mapped_column(Text)
    options_json: Mapped[list] = mapped_column(JSON)                      # ["A...", "B...", ...]
    answer: Mapped[str] = mapped_column(String(10))
    explanation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Mistake(Base):
    __tablename__ = "mistakes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"))
    user_answer: Mapped[str] = mapped_column(String(10))
    srs_due_at: Mapped[datetime] = mapped_column(DateTime)
    srs_interval_days: Mapped[int] = mapped_column(Integer, default=1)
    srs_ease: Mapped[float] = mapped_column(Float, default=2.5)           # SM-2 ease factor

class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    role: Mapped[str] = mapped_column(String(20))                         # user/assistant/tool
    content: Mapped[str] = mapped_column(Text)
    tool_calls_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Citation(Base):
    __tablename__ = "citations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    chunk_id: Mapped[str] = mapped_column(String(64))                     # Chroma chunk id
    page: Mapped[int] = mapped_column(Integer)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    hash: Mapped[str] = mapped_column(String(64), unique=True)
    chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

Plan milestones are normalized in `plan_milestones` for stable user progression.
`plans.milestones_json` is retained as a compatibility cache for Planner/eval paths during the P4 transition.

Milestone completion is self-reported plan progress. It never directly mutates `mastery`.
Mastery remains quiz/mistake evidence, and Plan uses mastery as an input for weak-topic intervention and validation prompts.

Plan events are stored in `plan_events` and power the Recent changes panel on `/plan`.

Chunks themselves live in **Chroma** (not in SQL): collection per user, metadata `{document_id, page, position}`.

---

## 5. FastAPI Routes

```python
# backend/app/api/__init__.py — route table (annotated)
```

| Method | Path | Body / Query | Response | Stream |
|---|---|---|---|---|
| POST | `/api/chat` | `{session_id, message}` | SSE: `{type: "token"\|"tool"\|"citation"\|"done", ...}` | yes |
| POST | `/api/documents` | multipart: file | `{document_id, chunks_count}` | no |
| GET | `/api/documents` | — | `[{id, filename, chunks_count}]` | no |
| POST | `/api/goals` | `{title, exam_date}` | `{goal_id, plan_id}` (Planner auto-generates plan) | no |
| GET | `/api/goals/{id}` | — | `{goal, plan, mastery_summary}` | no |
| PATCH | `/api/plans/{id}` | `{milestones_json}` | `{updated_at}` | no |
| GET | `/api/mistakes/due` | `?limit=10` | `[{mistake_id, question, srs_due_at}]` | no |
| POST | `/api/eval/run` | `{method, mode, dataset}` | `{run_id}` (async) | no |

All AI-bearing routes (`/chat`, `/eval/run`) read **BYOK headers** (see §6) per request.

---

## 6. LLM Provider (BYOK Header Spec)

Headers (case-insensitive):

| Header | Required | Default | Notes |
|---|---|---|---|
| `x-provider` | yes | `ollama` | `openai` / `anthropic` / `google_genai` / `ollama` |
| `x-model` | yes | `gemma3:4b` | provider-specific model id |
| `x-api-key` | unless `ollama` | — | never persisted server-side |
| `x-base-url` | no | provider default | for self-hosted / Azure / proxy |
| `x-judge-model` | no | same as `x-model` | force different model for Judge Guard |

```python
# backend/app/llm/provider.py
from langchain.chat_models import init_chat_model
from fastapi import Header, HTTPException

def get_llm(
    x_provider: str = Header("ollama"),
    x_model: str = Header("gemma3:4b"),
    x_api_key: str | None = Header(None),
    x_base_url: str | None = Header(None),
):
    if x_provider != "ollama" and not x_api_key:
        raise HTTPException(401, "x-api-key required for non-ollama providers")
    return init_chat_model(
        model=x_model,
        model_provider=x_provider,
        api_key=x_api_key,
        base_url=x_base_url,
        temperature=0.3,
    )
```

Server never logs `x-api-key`. Frontend stores it in `localStorage` + Web Crypto.

---

## 7. Frontend State (Pinia)

```
stores/
├── settings.ts   # provider/model/api_key/base_url (BYOK; localStorage-persisted, Web Crypto)
├── chat.ts       # current session, streaming buffer, ordered parts (text + tool + citation)
├── goal.ts       # active goal, plan milestones, mastery summary (REST-fetched, normalized)
└── library.ts    # uploaded documents, ingest progress
```

Each store maps 1:1 to a backend resource group; no cross-store coupling. Streaming SSE chunks merge into `chat.orderedParts[]` (mirrors JadeAI's `orderedParts` pattern).

---

## Out of scope for this doc (deferred to Phase 4 final ARCHITECTURE.md)

- ER diagram
- ADR records
- Deployment topology
- Observability (tracing / metrics)
- Performance budgets
