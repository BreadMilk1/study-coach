"""Owner-scoped retrieval authorization.

Production retrieval must never return another owner's chunks nor legacy
ownerless chunks.  ``RetrievalScope`` binds a single request to the caller's
owner id plus the set of document hashes that caller actually owns in SQL;
``ScopedRetriever`` is the request-level wrapper the API hands to every
Tutor/Quiz/Planner and agent consumer.

The wrapper deliberately keeps the agent-facing ``search(query, top_k)`` and
``get_chunk(chunk_id)`` protocol so the model never chooses an owner; identity
comes from ``require_existing_user`` at the request boundary and is never
cached on ``app.state`` or written into a graph checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# New ids are owner-namespaced and content-addressed. ``fullmatch`` (not
# ``match``/``$``) avoids a trailing newline being accepted. The index is
# non-negative; the page follows the processor convention (-1 or non-negative).
_V2_ID_RE = re.compile(r"v2:([^:]+):([0-9a-f]{64}):(-?\d+):(\d+)")
# Legacy ids are exactly ``<64 hex>:<page>:<index>``; anything else is not a
# compatibility candidate and is refused even when the hash is owned.
_LEGACY_ID_RE = re.compile(r"([0-9a-f]{64}):(-?\d+):(\d+)")
_HASH_RE = re.compile(r"[0-9a-f]{64}")


class UnscopedRetrieverError(RuntimeError):
    """A production retriever was handed to the scope wrapper without a scoped
    search entry point. Owner-scoped retrieval fails closed rather than falling
    back to an unscoped search."""


@dataclass(frozen=True)
class RetrievalScope:
    """Request-bound retrieval authorization for a single owner."""

    owner: str
    # document hash -> owning document id, from the caller's SQL documents.
    documents: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, str) or not self.owner or ":" in self.owner:
            raise ValueError(
                "retrieval scope owner must be a non-empty id without ':'"
            )
        # Immutable read-only view: a request can never mutate a shared scope.
        object.__setattr__(self, "documents", MappingProxyType(dict(self.documents)))

    def document_id_for(self, hash_: str) -> str | None:
        return self.documents.get(hash_)

    def allows(self, chunk: Mapping) -> bool:
        """True only for a fully consistent, SQL-owned v2 chunk of this owner.

        The chunk id version/owner/hash, the stored metadata owner/hash and the
        stored document id must all agree with the SQL scope; a missing or
        mismatched document id fails closed, and legacy/ownerless ids are never
        search-visible.
        """
        parsed = parse_v2_chunk_id(str(chunk.get("chunk_id", "")))
        if parsed is None:
            return False
        id_owner, id_hash, _page, _index = parsed
        if id_owner != self.owner:
            return False
        if chunk.get("owner") != self.owner:
            return False
        if chunk.get("hash") != id_hash:
            return False
        document_id = chunk.get("document_id")
        if not document_id or self.documents.get(id_hash) != document_id:
            return False
        return True


def build_v2_chunk_id(*, owner: str, hash_: str, page: int, index: int) -> str:
    """Owner-namespaced content-addressed chunk id (no temporary document id).

    Two owners indexing the same content hash therefore get distinct ids and
    cannot overwrite each other in a shared collection.
    """
    if not isinstance(owner, str) or not owner or ":" in owner:
        raise ValueError("chunk owner must be a non-empty id without ':'")
    if not isinstance(hash_, str) or _HASH_RE.fullmatch(hash_) is None:
        raise ValueError("chunk hash must be 64 lowercase hex characters")
    if not isinstance(page, int) or isinstance(page, bool) or page < -1:
        raise ValueError("chunk page must be -1 or a non-negative integer")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("chunk index must be a non-negative integer")
    return f"v2:{owner}:{hash_}:{page}:{index}"


def parse_v2_chunk_id(chunk_id: str) -> tuple[str, str, int, int] | None:
    match = _V2_ID_RE.fullmatch(chunk_id)
    if match is None:
        return None
    return match.group(1), match.group(2), int(match.group(3)), int(match.group(4))


def parse_legacy_chunk_id(chunk_id: str) -> tuple[str, int, int] | None:
    match = _LEGACY_ID_RE.fullmatch(chunk_id)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def authorized_get_chunk(base, chunk_id: str, *, scope: RetrievalScope) -> dict | None:
    """Unified authorized readback for citations/topic source chunk previews.

    A v2 id is readable only when the id owner, stored metadata owner, stored
    document id and hash, and the caller's SQL document ownership all agree;
    another owner indexing the same hash can never read it.  A legacy ownerless
    ``<64 hex>:<page>:<index>`` id stays readable only when the stored metadata
    carries no owner and the caller owns that hash in SQL, so old references
    survive until the user re-uploads to restore production search.
    """
    chunk = base.get_chunk(chunk_id)
    if chunk is None or chunk.get("chunk_id") != chunk_id:
        return None

    if parse_v2_chunk_id(chunk_id) is not None:
        return chunk if scope.allows(chunk) else None

    legacy = parse_legacy_chunk_id(chunk_id)
    if legacy is None:
        return None
    hash_, _page, _index = legacy
    if chunk.get("owner"):
        return None
    if hash_ not in scope.documents:
        return None
    return chunk


class ScopedRetriever:
    """Request-level retriever bound to one owner's retrieval scope."""

    def __init__(self, *, base, scope: RetrievalScope) -> None:
        self._base = base
        self._scope = scope

    @property
    def scope(self) -> RetrievalScope:
        return self._scope

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        prepared = []
        for chunk in chunks:
            parsed = parse_v2_chunk_id(str(chunk.get("chunk_id", "")))
            if parsed is None:
                raise ValueError("scoped add_chunks requires v2 chunk ids")
            id_owner, id_hash, _page, _index = parsed
            if id_owner != self._scope.owner:
                raise ValueError("chunk id owner does not match retrieval scope owner")
            if chunk.get("hash") != id_hash:
                raise ValueError("chunk id hash does not match chunk metadata hash")
            # The scope owner is authoritative and is never dropped.
            prepared.append({**chunk, "owner": self._scope.owner})
        self._base.add_chunks(prepared)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scoped = getattr(self._base, "search_scoped", None)
        if not callable(scoped):
            raise UnscopedRetrieverError(
                "production retriever does not implement search_scoped; "
                "owner-scoped retrieval fails closed"
            )
        return scoped(query, top_k=top_k, scope=self._scope)

    def get_chunk(self, chunk_id: str) -> dict | None:
        return authorized_get_chunk(self._base, chunk_id, scope=self._scope)
