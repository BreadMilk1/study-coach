from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderIdentity:
    provider: str
    protocol: str
    model: str


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cloud_run_id(
    identity: ProviderIdentity,
    mode: str,
    thinking_profile: str,
    query_id: str,
    turn_idx: int,
    run_idx: int,
) -> str:
    raw = "|".join([
        identity.provider,
        identity.protocol,
        identity.model,
        mode,
        thinking_profile,
        query_id,
        str(turn_idx),
        str(run_idx),
    ])
    return _digest(raw)


def storage_key(
    identity: ProviderIdentity,
    *,
    mode: str,
    thinking_profile: str,
    query_id: str,
    run_idx: int,
) -> str:
    raw = "|".join([
        identity.provider,
        identity.protocol,
        identity.model,
        mode,
        thinking_profile,
        query_id,
        str(run_idx),
    ])
    return "s" + _digest(raw)


DEFAULT_IDENTITY = ProviderIdentity(provider="minimax", protocol="anthropic", model="MiniMax-M3")


def recover_run_idx(row: Mapping[str, Any], *, identity: ProviderIdentity | None = None, max_runs: int = 3) -> int | None:
    if type(row.get("run_idx")) is int and int(row["run_idx"]) >= 0:
        return int(row["run_idx"])
    ident = identity or DEFAULT_IDENTITY
    mode = str(row.get("mode") or "")
    profile = str(row.get("thinking_profile") or "disabled")
    query_id = str(row.get("query_id") or "")
    turn_idx = int(row.get("turn_idx") or 0)
    run_id = str(row.get("run_id") or "")
    if run_id:
        for idx in range(max_runs):
            if cloud_run_id(ident, mode, profile, query_id, turn_idx, idx) == run_id:
                return idx
    session_key = str(row.get("session_storage_key") or "")
    if session_key:
        for idx in range(max_runs):
            if storage_key(ident, mode=mode, thinking_profile=profile, query_id=query_id, run_idx=idx) == session_key:
                return idx
    return None
