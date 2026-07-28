# Study Coach Demo Guide

> P4.5 reviewer path verified. P5 local-first data lifecycle automated and browser acceptance are complete.
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

P4.5 made this path stable before larger harness, durable memory, or multi-agent orchestration work. P5 adds an explicit single-user local-instance lifecycle around the same learning flow.

## Verified Commands

Verified on 2026-07-28:

```bash
cd backend
uv run pytest tests/db/test_data_lifecycle_repository.py -q
uv run pytest tests/rag/test_runtime.py -q
uv run pytest tests/test_data_lifecycle.py -q
uv run pytest tests/api/test_data_routes.py -q
uv run pytest tests/api/test_routes.py -q
uv run pytest tests/test_deployment_config.py -q
uv run pytest -q

cd ../frontend
pnpm test --run
pnpm build

cd ..
docker compose config
```

Result:

- Focused backend lifecycle suites: 2 + 12 + 31 + 32 + 24 tests passed.
- Focused deployment configuration suite: 12 tests passed, including build-context/order safeguards, frontend container listening, all three loopback bindings, the environment-aware Vite proxy, backend `OLLAMA_HOST=http://ollama:11434`, and all three model pre-pulls.
- Full backend: 370 tests passed.
- Frontend: 102 Vitest tests across 9 files passed; production build passed. Total automated tests: 472.
- Compose render passed with backend/frontend/Ollama host bindings `127.0.0.1:8000`, `127.0.0.1:5173`, and `127.0.0.1:11434`; backend `STUDY_COACH_LOCAL_MODE=1`, `CHROMA_PATH=/app/data/chroma`, and `OLLAMA_HOST=http://ollama:11434`; frontend proxy target `http://backend:8000`; and pre-pulls for `nomic-embed-text`, `gemma3:4b`, and `qwen2.5:7b`.
- A clean no-cache build reduced backend/frontend contexts from 384.90 MB / 263.59 MB to 47.96 kB / 4.77 kB. Runtime smoke passed for frontend `/`, direct backend `/api/health`, and frontend-proxied `/api/health` with HTTP 200.
- The first backend cold start downloads about 1.1 GB of FastEmbed model data and may temporarily return `502` through the frontend proxy; wait until `/api/health` becomes ready before continuing.
- Host-run Ollama verification passed with `gemma4:e4b` and `nomic-embed-text`: Settings connection returned Connected, embedding produced vectors, and the user-owned Topic 1 PDF indexed 49 chunks before and after a learning reset. This is host-run evidence, not a Compose model-runtime claim.
- Fly keeps `STUDY_COACH_LOCAL_MODE=0`.
- Static Google frontend runtime/UI search returned no matches; environment/config searches matched the intended local-mode and Chroma settings.
- The existing Vite warning for chunks larger than 500 kB is accepted for this stage.

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

For the P5 local-data-lifecycle acceptance path, prefer Docker Compose. It enables reset only for the local deployment and publishes the backend on loopback:

```bash
docker compose up
```

Before uploading a PDF or starting Chat, verify the Compose Ollama service has finished all model pulls:

```bash
docker compose exec ollama ollama list
```

Continue only after `nomic-embed-text`, `gemma3:4b`, and `qwen2.5:7b` appear. A successful backend `/api/health` response confirms the backend is ready, not that these Ollama models are ready.

If running the services directly, enable the same local-only boundary explicitly.

Terminal 1:

```bash
cd study-coach/backend
STUDY_COACH_LOCAL_MODE=1 \
OLLAMA_HOST=http://127.0.0.1:11434 \
CHROMA_PATH=./chroma_data \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
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

## P5 Local Data Lifecycle Acceptance — Complete

**Status (2026-07-28): passed end to end.** Learning reset, post-reset re-upload, Factory reset, cross-tab reload, Settings removal, and the single new anonymous first-run identity were verified in Chrome.

1. [x] Start Ollama, backend, and frontend through the supported local configuration.
2. [x] Import two user-owned PDFs and create Chat, Quiz, Plan, mistake, and mastery state.
3. [x] Open a new tab; verify startup blocks every other interaction.
4. [x] Choose Continue; verify all existing data remains.
5. [x] Refresh that tab; verify the gate does not reopen in the same tab session.
6. [x] Open another new tab; choose Start fresh and confirm learning reset.
7. [x] Verify Library, Chat, Quiz, Plan, milestones/events, mistakes, mastery, Chroma vectors, retriever cache, and checkpoint state are empty.
8. [x] Verify provider, model, Base URL, API key, language, and debug settings remain.
9. [x] Re-import passed with `Topic 1 - Introduction to Prompt Engineering.pdf` (49 chunks). Factory reset then restored default Settings, emptied Library and all learning counts, reloaded the second tab, and converged on one new anonymous user rather than creating one user per tab.
10. [x] No natural transient failure occurred. Injected failure and idempotent retry behavior remain covered by automated tests; no production failure-injection control was added.

During steps 3–9, explicitly check native-modal behavior: initial focus, keyboard reachability, Esc and backdrop blocking where required, and an inert background. With two tabs open, verify a learning reset requires acknowledgement in the other tab and a factory reset reloads the other tab.

Injected Chroma failure, SQLite failure, lock conflict, stale response, and idempotent retry semantics are proven by automated tests. Manual failure reproduction is optional and must use only a naturally occurring transient failure; no production failure-injection switch is permitted.

---

## Known Deferred Work

- Full Chroma per-user / per-document filtering.
- Agent Runs analytics page.
- Durable learning memory timeline.
- OpenTelemetry / token-cost dashboard.
- Multi-agent orchestration beyond the current LangGraph node dispatch.
