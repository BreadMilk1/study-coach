# Owner-Scoped Production Retrieval, Not a Multi-Tenant Lifecycle

Status: Accepted, implemented and offline-verified on the
`feat/retrieval-isolation` feature branch, 2026-10-28. Not a deployment claim.

## Context

Production retrieval shares one Chroma collection and one live BM25 index. Signed
identity and the empty-Library guard do not isolate candidates when two identities
each own documents. Content-hash-only chunk IDs also let two owners uploading the
same PDF overwrite each other's source metadata. Direct chunk-read authorization
does not protect text already selected for a model prompt or reranker.

## Decision

- Bind production retrieval to the authenticated, existing user at the API
  dependency boundary, not to a model-supplied argument or checkpoint field.
  Keep the existing agent-facing `search(query, top_k)` protocol.
- Use owner-specific new chunk IDs and complete owner/document metadata in the
  shared collection. Repeated uploads by one owner remain idempotent; different
  owners uploading the same bytes do not share an upsert identity or source name.
- Apply positive ownership constraints before dense/BM25 candidate ranking and
  before fusion/reranking. Missing ownership fails closed in production. Requests
  must not mutate a shared retriever's current owner.
- Authorize new chunk reads using the ID, stored owner/document metadata and SQL
  document ownership together. Owning another copy with the same hash does not
  authorize reading a different owner's new chunk.
- Preserve ownerless legacy records without modification. They are excluded from
  production search. Strict legacy IDs may still be read through the authorized
  citation path when the caller owns the corresponding SQL document. Re-uploading
  the original PDF creates owned records and restores retrieval; document listing
  alone does not imply that legacy vectors are searchable.
- Keep isolated evaluation corpus retrieval explicit and independent. Its
  ownerless corpus, chunk IDs, frozen definitions and artifact/hash contracts do
  not acquire production ownership requirements.

## Lifecycle and Compatibility Boundary

This decision deliberately narrows the older P5 deferral: production retrieval is
hardened now, while ADR 6's instance-wide summary/reset and single-worker local
deployment remain unchanged. Reset can still clear every identity's local data.
This is not a supported multi-tenant service, per-user deletion, OAuth continuity,
or a replacement for a future multi-user lifecycle design.

No real index migration, metadata backfill, deletion, SQL schema change, historical
citation rewrite, fixture regeneration or artifact re-signing is authorized by
this change. Old vectors remain physically present and may make vector counts
larger after a re-upload. Rollback of code alone does not undo newly written
namespaced records; do not run an old unscoped retriever against a mixed index as
an isolation-preserving rollback. Any data conversion or rollback procedure needs
a separately reviewed plan.

## Verification Contract

Use temporary SQLite and real temporary Chroma with deterministic embeddings and
a stub reranker. Cover two owners with different and identical PDFs, ownerless
legacy exclusion, ranking before top-k, reranker inputs, restarted BM25 hydration,
strict new and legacy chunk-read authorization, missing owner behavior, request
scope separation, and the existing isolated Learning Run corpus boundary.

Offline tests do not establish real provider behavior or a browser/runtime smoke.
Real models, API keys and service startup require advance notification and remain
outside the initial verification path.

The reconciled backend suite passed 1,989 tests, including the 21 new retrieval
isolation cases. API isolation cases use separate temporary persistent Chroma
directories and temporary SQLite; retriever-level cases use real in-memory
Chroma. Embeddings and reranking are deterministic substitutes. A task-local
launcher denies socket/DNS calls before pytest imports, disables model downloads,
and keeps default data paths temporary. These are in-process guards, not an OS
network sandbox, and do not prove real provider, browser or deployed behavior.
