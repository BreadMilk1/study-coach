# Study Coach Demo Guide

> P4.5 reviewer path verified. P5 local-first data lifecycle automated remediation is current; the latest full Chrome acceptance checkpoint completed on 2026-07-31.
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

P4.5 made this path stable before larger harness, durable memory, or multi-agent orchestration work. P5 adds an explicit single-user local-instance lifecycle around the same learning flow. Portfolio screenshots for the startup gate, Settings Danger Zone, and grounded Chat live under `docs/screenshots/` (see README gallery).

## Verified Commands

Automated remediation re-verified on 2026-07-31 (current HEAD):

```bash
cd backend
uv run pytest -q

cd ../frontend
pnpm test --run
pnpm build

cd ..
docker compose config --quiet
git diff --check
```

Result:

- Full backend: 411 tests passed.
- Frontend: 132 Vitest tests across 16 files passed; production build passed. Total automated tests: 543.
- Compose render (`docker compose config --quiet`) passed.
- Existing Vite warning for chunks larger than 500 kB is accepted for this stage.
- Latest full Chrome acceptance passed on Path A on 2026-07-31 with `gemma4:e4b`, `qwen2.5:7b` judge, and `nomic-embed-text`; this is not a Compose model-runtime claim.
- The retained Fly scaffold keeps `STUDY_COACH_LOCAL_MODE=0`, but cloud deployment is deferred and was not runtime-verified.

---

## Prerequisites

- Python / `uv` environment for the backend.
- Node / `pnpm` environment for the frontend.
- A user-owned PDF available outside the repo.

This guide has two **mutually exclusive** local paths. Pick one; do not run host Ollama and Compose Ollama together — both bind `127.0.0.1:11434`.

### Path A — host-run (canonical reviewer path)

- Host Ollama on `127.0.0.1:11434`.
- Host backend and frontend (commands below).
- Pull chat + embedding models on the host:

```bash
ollama pull gemma4:e4b          # verified tool-calling model for agent-loop demo
ollama pull nomic-embed-text    # required for PDF indexing
ollama serve
```

If using another chat model, confirm tool-calling support in Settings or via `/api/models/tool-check`.

### Path B — Docker Compose (alternative)

- **Stop host `ollama serve` first** — port `11434` conflicts with the Compose `ollama` service.
- Models live in the isolated `ollama_data` volume; host-downloaded models do **not** appear in the container.
- Compose pre-pulls `nomic-embed-text`, `gemma3:4b`, and `qwen2.5:7b` only. Use those in Settings unless you pull more inside the container.

---

## Start Services

### Path A — host-run (canonical reviewer path)

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

### Path B — Docker Compose (alternative)

```bash
docker compose up
```

Before uploading a PDF or starting Chat, verify the Compose Ollama service has finished all model pulls:

```bash
docker compose exec ollama ollama list
```

Continue only after `nomic-embed-text`, `gemma3:4b`, and `qwen2.5:7b` appear. A successful backend `/api/health` response confirms the backend is ready, not that these Ollama models are ready.

For agent-loop demo with `gemma4:e4b` on Compose (not pre-pulled):

```bash
docker compose exec ollama ollama pull gemma4:e4b
```

---

## Configure Settings

### Path A — host-run

In the frontend Settings view:

- Provider: `ollama`
- Base URL: leave empty, or set `http://127.0.0.1:11434`
- Model: `gemma4:e4b`
- Planner / Quiz mode: `agent_loop`
- Debug Mode: enabled
- Save Settings, then run **Test Connection** and **Test Tool Call**

### Path B — Docker Compose

- Stop host Ollama first if it still occupies `127.0.0.1:11434`, then follow the Compose start steps above.
- If the browser previously saved a host-run Base URL, **clear Base URL and Save Settings** so the backend container uses Compose `OLLAMA_HOST=http://ollama:11434`. Do not leave Base URL as `http://127.0.0.1:11434` inside the backend container — that points at the container itself, not the Compose Ollama service.
- Provider: `ollama`
- Model for `agent_loop`: `qwen2.5:7b` (pre-pulled), or `gemma4:e4b` after `docker compose exec ollama ollama pull gemma4:e4b`
- `gemma3:4b` is only for `deterministic` mode on Compose
- Save Settings, then run **Test Connection** and **Test Tool Call**

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

The examples below assume **Path A (host-run)** with `gemma4:e4b`. On Compose, substitute a pre-pulled model (`gemma3:4b` or `qwen2.5:7b`) or pull `gemma4:e4b` inside the container first.

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

**Status (2026-07-31): Full Chrome acceptance checkpoint complete.** The run used Path A with a disposable SQLite/Chroma workspace and user-owned Topic 1 / Topic 4 PDFs. The acceptance run also exposed and fixed an Esc bypass in the native lifecycle dialogs before the checklist was repeated successfully.

1. [x] Start Ollama, backend, and frontend through the supported local configuration (Path A host-run or Path B Compose — mutually exclusive).
2. [x] Import two user-owned PDFs and create Chat, Quiz, Plan, mistake, and mastery state.
3. [x] Open a new tab; verify startup blocks every other interaction.
4. [x] Choose Continue; verify all existing data remains.
5. [x] Refresh that tab; verify the gate does not reopen in the same tab session.
6. [x] Open another new tab; choose Start fresh and confirm learning reset.
7. [x] Verify Library, Chat, Quiz, Plan, milestones/events, mistakes, mastery, Chroma vectors, retriever cache, and checkpoint state are empty.
8. [x] Verify provider, model, Base URL, API key, language, and debug settings remain.
9. [x] Re-import a user-owned PDF and confirm chunk indexing. Factory reset restores default Settings, empty Library/learning counts, reloads peer tabs, and converges on one new anonymous user. A dedicated runtime old JWT returned 200 for Factory reset and its idempotent retry, but 401 for an ordinary learning write.
10. [x] No production failure-injection control. Injected failure and idempotent retry behavior remain covered by automated tests.

During steps 3–9, explicitly check native-modal behavior: initial focus, keyboard reachability, Esc and backdrop blocking where required, and an inert background. With two tabs open, verify a learning reset requires acknowledgement in the other tab and a factory reset reloads the other tab.

Injected Chroma failure, SQLite failure, lock conflict, stale response, and idempotent retry semantics are proven by automated tests. Manual failure reproduction is optional and must use only a naturally occurring transient failure; no production failure-injection switch is permitted.

---

## Known Deferred Work

- Full Chroma per-user / per-document filtering.
- Agent Runs analytics page.
- Durable learning memory timeline.
- OpenTelemetry / token-cost dashboard.
- Multi-agent orchestration beyond the current LangGraph node dispatch.
