from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderIdentity:
    provider: str
    protocol: str
    model: str


_SEPARATOR = "|"


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _checked_field(value: str) -> str:
    """Reject fields that would make the joined identity ambiguous under the separator.

    The message never echoes the offending value.
    """
    if _SEPARATOR in value:
        raise ValueError("identity field contains the reserved separator")
    return value


def cloud_run_id(
    identity: ProviderIdentity,
    mode: str,
    thinking_profile: str,
    query_id: str,
    turn_idx: int,
    run_idx: int,
) -> str:
    raw = _SEPARATOR.join([
        _checked_field(identity.provider),
        _checked_field(identity.protocol),
        _checked_field(identity.model),
        _checked_field(mode),
        _checked_field(thinking_profile),
        _checked_field(query_id),
        _checked_field(str(turn_idx)),
        _checked_field(str(run_idx)),
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
    raw = _SEPARATOR.join([
        _checked_field(identity.provider),
        _checked_field(identity.protocol),
        _checked_field(identity.model),
        _checked_field(mode),
        _checked_field(thinking_profile),
        _checked_field(query_id),
        _checked_field(str(run_idx)),
    ])
    return "s" + _digest(raw)


DEFAULT_IDENTITY = ProviderIdentity(provider="minimax", protocol="anthropic", model="MiniMax-M3")


def _turn_idx_for_run_id(row: Mapping[str, Any]) -> int | None:
    """turn_idx is only needed for the run_id path; an unusable value skips that path."""
    try:
        return int(row.get("turn_idx") or 0)
    except (TypeError, ValueError, OverflowError):
        return None


def recover_run_idx(row: Mapping[str, Any], *, identity: ProviderIdentity | None = None, max_runs: int = 3) -> int | None:
    if type(row.get("run_idx")) is int and int(row["run_idx"]) >= 0:
        return int(row["run_idx"])
    ident = identity or DEFAULT_IDENTITY
    mode = str(row.get("mode") or "")
    profile = str(row.get("thinking_profile") or "disabled")
    query_id = str(row.get("query_id") or "")
    run_id = str(row.get("run_id") or "")
    if run_id:
        turn_idx = _turn_idx_for_run_id(row)
        if turn_idx is not None:
            for idx in range(max_runs):
                if cloud_run_id(ident, mode, profile, query_id, turn_idx, idx) == run_id:
                    return idx
    session_key = str(row.get("session_storage_key") or "")
    if session_key:
        for idx in range(max_runs):
            if storage_key(ident, mode=mode, thinking_profile=profile, query_id=query_id, run_idx=idx) == session_key:
                return idx
    return None
