# P5 Local-first Data Lifecycle - Design Spec

> Brainstormed and approved 2026-07-20.
> Upstream: verified P4.5 portfolio demo closure.
> Target: make Study Coach an honest, polished local-first portfolio product before any multi-user account work.

---

## 1. Problem

P4.5 proved the main reviewer path: upload a PDF, receive a grounded answer, generate and grade a Chat quiz, then restore the session after refresh. The next trust gap is data ownership and lifecycle.

The current product mixes three incompatible ideas:

- The frontend presents Google sign-in as if account continuity were complete.
- The browser persists settings and identity in `localStorage`.
- The real learning data lives across SQLite, Chroma, and LangGraph process memory.

The Google flow is not a reliable multi-user product boundary. The frontend calls `/api/auth/google`, which switches to a separate Google user, while the backend's guest-preserving `/api/auth/upgrade` path is not used by the browser flow. Existing anonymous data can therefore appear to disappear after sign-in. Chroma also has no `user_id` or `document_id` metadata, so per-user deletion and retrieval isolation cannot be claimed safely.

At the same time, clearing browser storage alone does not clear imported PDFs, embeddings, chat history, Quiz, Plan, mistakes, or mastery. The application needs an explicit local data lifecycle that matches where data is actually stored.

---

## 2. Product Decision

P5 positions Study Coach as a **local-first, single-user portfolio application**:

- No registration is required.
- Ollama remains the preferred local model path.
- Learning data is described as belonging to this local Study Coach instance.
- Google OAuth is hidden from the product UI and is not advertised as a shipped capability.
- Existing OAuth backend code is frozen, not removed, so a future branch can either repair or replace it deliberately.

After P5 is stable, multi-user work should start from the P5 baseline in a separate `feature/multi-user-auth` branch and worktree. That future phase must solve guest-to-member upgrade, SQL ownership, Chroma metadata filtering, migration, sign-out semantics, and cross-device continuity together.

P5 must not add partial per-user behavior on top of the current global Chroma collection.

---

## 3. Goals

1. Let a user decide, once per tab session, whether to continue with existing learning data or start fresh.
2. Provide two explicit reset scopes:
   - clear learning data while preserving model and UI settings;
   - restore the complete local application to first-run state.
3. Coordinate deletion across Chroma, LangGraph state, retriever caches, SQLite, and browser storage.
4. Make reset requests idempotent and recoverable when one storage layer fails.
5. Add a small notification system that distinguishes transient success from persistent failure.
6. Remove Google OAuth from the visible product story without broad auth refactoring.
7. Leave a clean architectural boundary for later multi-user work.

The portfolio headline becomes:

> Study Coach is a privacy-conscious local AI learning workspace with an explicit, recoverable data lifecycle.

---

## 4. Non-goals

P5 does not include:

- per-user or per-document Chroma filtering;
- Google OAuth repair, account linking, or cloud sync;
- data export or backup;
- selective deletion of one document, one chat, or one goal;
- cross-device persistence;
- a durable LangGraph checkpointer migration;
- a general job queue or distributed transaction system;
- public multi-user deployment;
- replacing the existing design system or adding a UI component library.

---

## 5. Reset Scopes

Because Chroma is currently global, reset operations apply to **this Study Coach instance**, not to an implied current cloud account.

| Data | Clear learning data | Factory reset |
|---|---:|---:|
| SQL `documents` and recorded chunk counts | delete | delete |
| Chroma collection and all embeddings | delete and recreate | delete and recreate |
| chat sessions, messages, citations, assistant artifacts | delete | delete |
| Quiz questions, mistakes, mastery | delete | delete |
| goals, topics, plans, milestones, plan events | delete | delete |
| LangGraph process-lifetime checkpoint state | clear | clear |
| in-memory retriever / BM25 caches | invalidate and rebuild empty | invalidate and rebuild empty |
| current chat session id | clear | clear |
| provider, model, Base URL, API key, mode, language, debug settings | preserve | clear |
| tool-capability cache | preserve | clear |
| local fingerprint and bearer token | preserve | clear |
| backend local user rows | preserve | delete |

The startup action **Start fresh** always uses `learning` scope. Factory reset exists only in Settings > Danger Zone.

### 5.1 Browser keys

The current browser-owned keys include:

- `study-coach:fingerprint`
- `study-coach:settings`
- `study-coach:tool-capable:<model>`
- `study-coach:current-chat-session-id`

P5 adds one tab-scoped key:

- `study-coach:startup-choice-made` in `sessionStorage`

Clear-learning removes the current chat session key but preserves identity, settings, and tool-capability keys. Factory reset removes every `study-coach:*` key from both `localStorage` and `sessionStorage` after the backend confirms success.

---

## 6. API Contract

### 6.1 `GET /api/data/summary`

Purpose: decide whether the startup gate is needed and show trustworthy local counts.

Authentication:

- requires the normal bearer token;
- must use a strict current-user dependency;
- must not fall back to `default-user`.

For P5, "strict" means a valid signed bearer token is mandatory; it does not require the referenced user row to still exist. This matters when a factory reset completed but its HTTP response was lost: the old, still-valid signed token must be able to retry the same idempotent local reset.

Proposed response:

```json
{
  "reset_enabled": true,
  "has_learning_data": true,
  "documents": 2,
  "source_chunks": 110,
  "vectors": 297,
  "chat_sessions": 4,
  "messages": 18,
  "plans": 1,
  "questions": 6,
  "mistakes": 2
}
```

`source_chunks` is the sum recorded by current SQL document rows. `vectors` is the actual Chroma collection count. They may differ because the present collection contains legacy or orphaned embeddings; the API must not hide this difference.

All counts are instance-wide, matching the global Chroma reset boundary. `has_learning_data` is true when any learning table contains rows **or** Chroma still contains vectors; a stale vector-only corpus must still trigger the startup choice.

`reset_enabled` is the frontend capability contract. Summary remains readable when reset is disabled, but Start fresh and Danger Zone actions are not rendered.

### 6.2 `POST /api/data/reset`

Purpose: perform one coordinated instance reset.

Authentication uses the same strict signed-bearer rule as summary: no fallback and no user-row lookup. This deliberately permits a factory-reset retry after the users table has already been cleared but the first HTTP response was lost. After a confirmed factory success, every browser tab removes the old token. This retry policy is acceptable only behind the local-mode and loopback boundary; it is not a future public-account authorization design.

Proposed request:

```json
{
  "scope": "learning",
  "confirmation": "CLEAR_LEARNING_DATA"
}
```

or:

```json
{
  "scope": "factory",
  "confirmation": "FACTORY_RESET"
}
```

The UI requires the user to type `RESET` for factory reset, then maps that accepted interaction to the API confirmation constant. Keeping an explicit server-side confirmation value prevents an accidental empty-body request from becoming destructive.

Success response:

```json
{
  "scope": "learning",
  "status": "completed",
  "deleted": {
    "documents": 2,
    "source_chunks": 110,
    "vectors": 297,
    "chat_sessions": 4,
    "messages": 18,
    "plans": 1,
    "questions": 6,
    "mistakes": 2,
    "users": 0
  }
}
```

Stage-failure response (`HTTP 500`):

```json
{
  "detail": {
    "code": "reset_failed",
    "failed_stage": "sqlite",
    "retryable": true,
    "message": "Data reset failed. Please retry."
  }
}
```

After a destructive stage has started, other operations and the wrong reset scope receive `HTTP 409` until the original scope completes:

```json
{
  "detail": {
    "code": "reset_recovery_required",
    "required_scope": "learning",
    "message": "A previous data reset is incomplete. Retry that reset."
  }
}
```

The public error detail must remain safe and concise. Full exception text belongs in server logs, not the browser.

### 6.3 Local-mode safety boundary

An anonymous token is not a meaningful authorization boundary for a global destructive endpoint because anonymous tokens can be provisioned freely. Therefore:

- reset routes are enabled only when `STUDY_COACH_LOCAL_MODE=1`; the default is `0` (disabled);
- `GET /api/data/summary` returns `reset_enabled` so the frontend does not infer capability from deployment shape;
- supported portfolio startup sets `STUDY_COACH_LOCAL_MODE=1` and binds the backend to loopback;
- Docker Compose uses `127.0.0.1:8000:8000` for the backend host port. The container may continue listening on `0.0.0.0` internally because the host binding is the external boundary;
- when local-first mode is disabled, `/api/data/reset` returns `403` and the frontend hides Danger Zone reset actions;
- a future public deployment must not reuse this global endpoint as a substitute for ownership-aware deletion.

FastAPI dependencies remain the route-level enforcement point, matching the project's existing dependency injection pattern.

---

## 7. Reset Coordinator

Introduce one backend reset service rather than placing deletion logic in the route.

Responsibilities:

1. Validate scope and confirmation.
2. Obtain an application-level reset lock.
3. Capture pre-delete counts for the final summary.
4. Delete and recreate the Chroma collection.
5. Build an empty retriever stack against the new collection and replace the entire `app.state.retriever` reference, including the old dense collection handle and BM25 index.
6. Replace `app.state.checkpointer` with a new `InMemorySaver()`.
7. Delete relational rows in one SQLite transaction.
8. Delete users only for factory scope.
9. Return a stable summary object.
10. Release the active exclusive lease on success or failure. If failure occurs after the destructive stage begins, retain a scope-specific recovery latch until the same scope completes.

The route stays responsible for HTTP validation and status mapping. Repositories remain responsible for relational operations. The reset service coordinates them.

### 7.1 Relational deletion order

SQLite rows are deleted child-first because the current schema does not provide one database cascade that safely covers this operation:

1. `citations` -> `messages` -> `sessions`
2. `plan_events` -> `plan_milestones` -> `plans`
3. `mistakes` and `mastery`
4. `questions` -> `topics` -> `goals`
5. `documents`
6. `users` for factory scope only

The exact repository calls should follow model foreign keys rather than relying on this list blindly; the implementation plan must verify the current schema before writing the first deletion test.

### 7.2 Cross-store failure semantics

SQLite and Chroma cannot participate in one atomic transaction. P5 uses a fixed order and idempotency instead of claiming false atomicity:

1. Chroma first.
2. in-memory graph and retrieval state second.
3. SQLite transaction last.

If setup fails before destructive work starts, ordinary operations remain available. Once Chroma replacement begins, failure leaves a scope-specific recovery latch. Shared operations and the other reset scope fail with `409 reset_recovery_required`; only the required scope may retry. If Chroma succeeds and SQLite fails, that retry sees an already-empty Chroma collection and finishes SQL deletion safely. Deleting absent rows or recreating an already-reset collection must be a success condition. The latch is in-process and intentionally follows P5's one-worker boundary.

The frontend does not unlock the startup gate or clear browser identity until the backend returns `status=completed`.

### 7.3 Concurrent writes

While reset is running:

- a second reset receives `409 reset_in_progress`;
- all routes that read or write learning data, including streaming Chat, upload, Library, Plan, Quiz, mistakes, mastery, and stats, participate in one lifecycle gate;
- identity-mutating auth POST routes (`anonymous`, frozen `google`, and `upgrade`) use the same shared lease so reset cannot race with user-row creation or mutation; read-only auth config remains available;
- an already-active data operation makes reset return `409 data_operation_in_progress` instead of terminating the operation mid-stream;
- after reset begins, new data operations receive `409 reset_in_progress` until the reset completes;
- P5 local-first mode supports one backend worker. A future multi-worker deployment requires a cross-process lock and is outside this phase;
- the local single-user UI remains blocked by the modal;
- the implementation plan should use the smallest shared application-level guard compatible with existing app wiring rather than adding a job queue.

The gate should expose shared-operation and exclusive-reset leases rather than serializing every normal request behind one mutex. Its state lives on `app.state` beside the current retriever and checkpointer.

### 7.4 Chroma behavior

The current design uses a single collection. Chroma officially supports deleting a collection with `client.delete_collection(name=...)`; P5 recreates the known collection immediately so later upload paths do not depend on a process restart. While the lifecycle gate is held exclusively, it then atomically replaces the app-level retriever reference. Keeping the old retriever and merely clearing its collection is invalid because the object still owns the deleted collection handle and a populated BM25 index.

Docker currently sets `CHROMA_PERSIST_DIR`, while application startup reads `CHROMA_PATH`. P5.1 standardizes Compose on `CHROMA_PATH=/app/data/chroma` so the collection being summarized and reset is the collection on the mounted local data volume.

---

## 8. Startup Gate

Startup sequence:

1. Restore settings and provision or restore the local anonymous identity.
2. Call `/api/data/summary`.
3. If no learning data exists, enter the app directly.
4. If learning data exists and the tab has no `startup-choice-made` value, open the blocking startup gate.
5. Continue sets the tab-scoped key and unlocks the app.
6. Start fresh opens the clear-learning confirmation.
7. Successful reset clears the current chat id, sets the tab-scoped key, refreshes affected stores, shows a completion summary, and unlocks the app.
8. Failed reset keeps the gate open and exposes retry.
9. If a reload observes `reset_recovery_required`, restore the blocking reset error using the backend-provided required scope; do not offer Continue without clearing.

If summary fails, the gate shows a persistent inspection error with Retry and **Continue without clearing**. It must not offer Start fresh with unknown counts, and it must not silently fail open into the application.

The routed page is not mounted until the workspace is unlocked. Hiding only a visual overlay is insufficient because child `setup()` / `onMounted()` hooks could otherwise issue requests before the required choice.

The gate appears once per browser tab session. A normal refresh in the same tab does not ask again. A new tab asks again when learning data exists.

### 8.1 Modal behavior

Use the platform `<dialog>` element opened with `showModal()` and rendered at the document root with Vue `<Teleport to="body">`. This provides a top-layer modal and an inert background without introducing a component library.

The startup gate is a deliberate exception to normal dismissible-dialog behavior:

- backdrop clicks do not close it;
- the `cancel` event is prevented so `Esc` cannot bypass the required choice;
- focus starts on Continue and stays inside the dialog;
- both choices remain reachable by keyboard;
- the dialog closes only after Continue or a completed reset.

Settings confirmation dialogs remain cancelable with an explicit Cancel action and `Esc` until deletion begins. During deletion, actions are disabled to prevent duplicate requests.

### 8.2 Cross-tab reset propagation

The modal only blocks its own tab, so backend coordination is not sufficient for browser consistency. Add one `BroadcastChannel` named `study-coach:data-lifecycle`:

- learning reset success broadcasts a reset epoch and `scope=learning`;
- other tabs clear their current chat id, reset/refetch data-backed Pinia stores, mark their startup choice unresolved, and return to the startup gate;
- factory reset success broadcasts `scope=factory`; all tabs clear browser state and reload;
- tabs opened after the broadcast still observe the shared `localStorage` removals and the empty backend summary.

Broadcast is a state-invalidation signal, not a transport for deleted data or tokens.

---

## 9. Settings Danger Zone

Settings adds a dedicated Danger Zone below normal model settings.

### 9.1 Clear learning data

- explains exactly what will be deleted and what will be preserved;
- displays current document, chunk/vector, session, and learning counts;
- requires one explicit confirmation click;
- preserves Ollama/API and interface settings;
- clears the current chat session and refreshes all data-backed stores.

### 9.2 Factory reset

- explains that learning data, model configuration, interface settings, local identity, and backend user rows are deleted;
- requires the user to type `RESET`;
- calls the backend factory reset first;
- after success, removes every `study-coach:*` browser key and reloads;
- provisions a new anonymous identity on the next startup.

Factory reset must not clear browser state first. Doing so would discard the token needed to finish or retry the backend operation.

---

## 10. Notifications

Use three feedback surfaces with distinct responsibilities:

| Surface | Use | Persistence |
|---|---|---|
| Toast | successful save/reset and short non-blocking confirmation | about 5 seconds, manually dismissible |
| Inline status | error details, retry, connection failure, reset-stage failure | persists until resolved or dismissed intentionally |
| Modal dialog | destructive confirmation and reset progress | blocks the relevant interaction |

Implementation boundary:

- one small Pinia notification store;
- one app-level `ToastHost`;
- Lucide icons and existing semantic tokens;
- no new notification package;
- color is always paired with icon and text;
- toast host uses an appropriate live region without repeatedly announcing old items;
- reduced-motion settings remain respected.

Native `alert()` is removed from the visible Google flow when that flow is hidden. Other unrelated page-level error handling is not broadly refactored in P5.

---

## 11. OAuth Product Boundary

P5 changes the product claim, not the whole auth implementation.

In scope:

- remove Google One Tap / Google sign-in UI, its Settings lifecycle/polling code, and the GIS script tag from `frontend/index.html`;
- remove the now-unreachable frontend `googleLogin()` helper and related browser-only sign-out code;
- remove account-related copy that implies cloud continuity;
- retain anonymous provisioning because existing routes and ownership columns depend on a local user id;
- document Google OAuth as experimental/deferred.

Out of scope:

- deleting backend auth routes, Google dependencies, JWT columns, or migrations;
- wiring `/api/auth/upgrade` into the frontend;
- merging existing anonymous and Google rows;
- attempting to assign legacy Chroma vectors to users.

Future multi-user work must start with an ownership design. The current global reset API must then be disabled or replaced with an ownership-filtered service.

---

## 12. Upload Temporary-file Cleanup

The upload path currently creates `/tmp/sc_<hash>.pdf` and does not guarantee cleanup after ingestion. A hash-derived name is also unsafe for concurrent uploads of the same PDF. P5 creates a unique file per request with `tempfile.NamedTemporaryFile` or `mkstemp`, wraps its use in `try/finally`, and unlinks only the exact file created by that request whether processing succeeds or fails.

This cleanup is separate from reset semantics: reset does not scan `/tmp` using broad globs or remove files it cannot prove belong to the current operation.

---

## 13. Frontend State Refresh

After a successful learning reset, clear or refetch every store that can otherwise retain deleted server data in memory:

- documents;
- chat session and messages;
- plan;
- Quiz active state;
- mistakes;
- mastery;
- overview-derived statistics.

Prefer a small central `resetClientLearningState()` orchestration function that calls existing store reset actions. Do not create a second source of truth containing copies of all store data.

Factory reset performs browser-key removal and full reload instead of attempting to reconstruct every first-run store state manually.

---

## 14. Testing Strategy

Implementation follows TDD, with a red-green-refactor loop for each vertical slice.

### 14.1 Backend automated tests

- summary distinguishes empty and populated stores;
- summary exposes SQL chunk count and actual Chroma vector count separately;
- strict destructive dependencies reject missing/invalid identity and never use `default-user`;
- local-mode-disabled reset returns `403`;
- local mode is disabled by default and summary exposes the capability;
- learning reset deletes all learning rows and preserves users;
- factory reset deletes learning rows and users;
- Chroma collection is emptied and recreated;
- checkpoint and retrieval state are invalidated;
- deletion is idempotent;
- Chroma failure leaves SQL untouched;
- SQLite failure after Chroma deletion is retryable;
- concurrent reset returns `409`;
- active Chat/upload prevents reset, and reset prevents new data operations;
- factory reset remains idempotently retryable after users are deleted and the first response is lost;
- temporary upload files are removed on both success and failure.

### 14.2 Frontend automated tests

Add Vitest as the single new frontend dev dependency. It is Vite-native and supports TypeScript through the existing Vite pipeline.

Keep the first test boundary small and DOM-light:

- startup decision logic with injected `Storage` and API adapters;
- `sessionStorage` prevents a second prompt after refresh in the same tab;
- empty data bypasses the gate;
- Start fresh always requests learning scope;
- factory reset clears browser keys only after backend success;
- failed reset preserves browser state and exposes a retryable error;
- toast queue expiration/dismissal behavior;
- client store reset orchestration calls all owned stores.
- cross-tab learning reset invalidates stores and returns the other tab to the gate;
- cross-tab factory reset clears state and reloads;
- reset-disabled capability hides Start fresh and Danger Zone actions.

Modal focus, `Esc`, inert background, and visual placement remain browser-level acceptance checks unless a DOM environment becomes necessary. Do not add jsdom, a component-testing library, or a browser-test provider solely for P5.

### 14.3 Verification commands

```bash
cd backend
uv run pytest -q

cd ../frontend
pnpm test --run
pnpm build
```

### 14.4 Manual browser path

1. Configure Ollama and import two owned PDFs.
2. Create Chat, Quiz, Plan, mistake, and mastery data.
3. Open a new tab and confirm the startup gate blocks the app.
4. Choose Continue and confirm data remains.
5. Open another new tab, choose Start fresh, and confirm the delete scope.
6. Confirm Library, Chat, Plan, Quiz, mistakes, mastery, Chroma, and checkpoint state are empty.
7. Confirm Ollama/model/language settings remain.
8. Re-import data, run Factory reset, and confirm first-run identity/settings state.
9. Inject a Chroma or SQLite failure and confirm the gate remains, the failed stage is visible, and retry completes.
10. Confirm refresh in one already-decided tab does not reopen the gate.

---

## 15. Implementation Slices

This design stops before file-by-file implementation instructions. The implementation plan should divide work into these independently verifiable cuts:

1. **P5.0 - Product boundary:** local-first copy, OAuth UI and GIS runtime removed, README/Roadmap claims aligned, and architecture intent recorded.
2. **P5.1 - Backend data lifecycle:** strict summary API, reset coordinator, both reset scopes, local-mode/loopback guard, Docker Chroma path correction, and unique temporary-upload cleanup.
3. **P5.2 - Startup gate:** Vitest foundation, startup decision logic, modal flow, retry, and client-store refresh.
4. **P5.3 - Settings and notifications:** Danger Zone, both confirmation flows, notification store/host, summaries, and accessibility.
5. **P5.4 - Full closure:** complete automated verification, manual Demo, and final architecture/product documentation sync.

Each cut must keep the repository releasable and must not claim later cuts as shipped.

---

## 16. Risks and Rollback

### 16.1 Destructive behavior

Risk: a bug deletes data outside the intended instance.

Mitigation:

- no recursive filesystem deletion;
- use known Chroma collection APIs and repository deletes;
- exact confirmation values;
- local-mode guard;
- pre-delete counts and explicit summary;
- fixture-backed integration tests using isolated temporary databases and collections.

Rollback: revert the P5 feature commit. Deleted user data itself is intentionally unrecoverable; the UI must say so before confirmation.

### 16.2 Partial cross-store reset

Risk: Chroma succeeds while SQLite fails.

Mitigation: fixed order, explicit failed stage, idempotent retry, and no client unlock before backend completion.

Rollback: retry the same reset. P5 does not promise restore-from-backup.

### 16.3 Local-first route exposure

Risk: an anonymous remote client calls a global reset route.

Mitigation: local-mode-only registration/enforcement and loopback binding. Public hosting is outside the supported P5 product mode.

### 16.4 Future multi-user migration

Risk: local global-reset assumptions leak into the later account system.

Mitigation: isolate reset orchestration behind a service and explicitly disable/replace it when ownership-filtered deletion is introduced.

---

## 17. Documentation Sync

At design approval:

- add this spec;
- mark P5 as approved/planned in `docs/ROADMAP.md`.

After verified implementation:

- update `docs/ARCHITECTURE.md` with local-first mode, reset APIs, storage lifecycle, and OAuth boundary;
- update `docs/DEMO.md` with Continue, Start fresh, and Factory reset verification;
- record final automated and manual validation counts in ROADMAP.

P5.0 must update README product claims in the same cut that hides Google OAuth, so every intermediate cut remains truthful and releasable.

---

## 18. Official References

- [Chroma: Delete collection](https://docs.trychroma.com/reference/chroma-api/collection/delete-collection)
- [FastAPI dependencies and security](https://fastapi.tiangolo.com/reference/dependencies/)
- [Vue: Teleport](https://vuejs.org/guide/built-ins/teleport)
- [MDN: `<dialog>` element](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/dialog)
- [Vitest: Getting Started](https://vitest.dev/guide/)

These references support the chosen collection reset, dependency-enforced route boundary, top-level modal rendering, native dialog behavior, and Vite-native test runner. Project-specific behavior remains governed by this spec and the existing Study Coach architecture.
