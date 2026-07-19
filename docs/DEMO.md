# Study Coach Demo Guide

> P4.5 automated gates verified. This is the reviewer-facing demo route for Study Coach.
> Use a PDF you own. Do not copy private course PDFs into this repository.

---

## Purpose

This guide demonstrates the Chat-first Study Coach loop:

1. Upload a PDF.
2. Ask a grounded question in Chat.
3. Generate a Chat quiz.
4. Answer the quiz.
5. Refresh the frontend.
6. Confirm chat history and Debug / Agent Run evidence are restored.

P4.5 focuses on making this path stable before larger harness, durable memory, or multi-agent orchestration work.

## Verified Commands

Verified on 2026-07-02:

```bash
cd backend
uv run pytest -q

cd ../frontend
pnpm build
```

Result:

- Backend: 256 tests passed.
- Frontend: production build passed.
- Existing Vite large chunk warning is accepted for this stage.

---

## Prerequisites

- Python / `uv` environment for the backend.
- Node / `pnpm` environment for the frontend.
- Ollama running locally if using `x-provider: ollama`.
- A user-owned PDF available outside the repo.

Recommended local model for Chat-first demo:

```bash
ollama pull gemma4:e4b
```

If using another model, confirm tool-calling support in Settings or via `/api/models/tool-check`.

---

## Start Services

Terminal 1:

```bash
cd study-coach/backend
uv run uvicorn app.main:app --port 8000
```

Terminal 2:

```bash
cd study-coach/frontend
pnpm dev
```

Open:

```text
http://localhost:5173/
```

---

## Configure Settings

In the frontend Settings view:

- Provider: `ollama`
- Model: `gemma4:e4b` or another known tool-capable model
- Debug Mode: enabled
- Quiz mode: `agent_loop` for the main demo path

Use `deterministic` only as a faster fallback check.

For a judge model, use a different model when available to reduce same-model self-preference bias.

---

## Demo Path

### 1. Upload PDF

Go to Library and upload a user-owned PDF.

Expected:

- The file appears in the Library list.
- `chunks_count` is greater than 0.

### 2. Ask a Grounded Chat Question

Go to Chat and ask a source-grounded question, for example:

```text
What is prompt engineering?
```

Expected:

- The answer streams in Chat.
- Citations appear.
- Debug Mode shows router / judge trace, and agent evidence when applicable.

### 3. Ask for a Chat Quiz

In the same Chat session, ask:

```text
quiz me on Prompt Engineering.
```

P4.5 expected stable behavior:

- If a multiple-choice question is shown, it has already been persisted.
- Debug Mode shows `agent_run` evidence for the quiz path.
- If persistence fails, Study Coach shows a degraded non-answerable message instead of a quiz that cannot be graded.

### 4. Answer the Quiz

Reply with one option:

```text
A
```

Expected:

- Study Coach grades the answer.
- The active quiz question is cleared after grading.
- Mastery / mistake side effects follow the existing QuizMaster behavior.

### 5. Refresh

Refresh the browser.

Expected:

- Chat history is restored.
- Assistant citations are restored.
- Debug / Agent Run evidence for assistant turns is restored.

---

## CLI Verification

### Anonymous token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"fingerprint":"demo-user"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
```

### Empty corpus guard

Use a fresh fingerprint with no uploaded PDF:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/anonymous \
  -H "Content-Type: application/json" \
  -d '{"fingerprint":"demo-empty-corpus"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -N -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-provider: ollama" \
  -H "x-model: gemma4:e4b" \
  -d '{"message":"What is HyDE?","session_id":"demo-empty-corpus"}'
```

Expected:

- `citations` is an empty list.
- The token text asks the user to upload a PDF.
- No stale Chroma content is used.

### Chat session restore

For a CLI-created session, inspect the current session for the same bearer token:

```bash
curl -s http://localhost:8000/api/chat/sessions/current \
  -H "Authorization: Bearer $TOKEN" \
  | python -m json.tool
```

Then fetch messages:

```bash
SESSION_ID="<session_id_from_previous_command>"

curl -s "http://localhost:8000/api/chat/sessions/$SESSION_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  | python -m json.tool
```

Expected:

- User and assistant messages are present.
- Assistant citations are present when sources were used.
- Assistant `agent_run` is present for agent-loop Quiz / Planner runs.

For a browser-created session, verify restore in the browser by refreshing the Chat page. The browser stores its own anonymous/member token, so CLI requests only inspect the same session when they use the same bearer token.

---

## P4.5 Closure Gates

Before treating this as the public reviewer demo path, verify:

- [x] `cd backend && uv run pytest -q` passes without live Ollama.
- [x] `cd frontend && pnpm build` passes.
- [x] Chat quiz strong consistency holds: visible MCQ means persisted and gradeable.
- [x] Quiz GENERATE shows only the persisted prompt/options; answer and clean explanation appear only after the user replies.
- [x] Failed `persist_quiz_question` does not leave a user-answerable MCQ in Chat.
- [x] Debug Mode uses redacted Agent Run previews and restores persisted evidence after refresh.
- [x] Manual browser demo with user-owned PDFs (2026-07-20): grounded Chat answer → agent-loop Quiz → deterministic grade → refresh restore.

---

## Known Deferred Work

- Full Chroma per-user / per-document filtering.
- Agent Runs analytics page.
- Durable learning memory timeline.
- OpenTelemetry / token-cost dashboard.
- Multi-agent orchestration beyond the current LangGraph node dispatch.
