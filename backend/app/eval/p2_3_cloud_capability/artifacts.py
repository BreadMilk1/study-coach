from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, unquote

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


SCHEMA_VERSION = "cloud-capability-candidate-v1"

_RESERVED_KEY_SUFFIXES = (".partial", ".failed")


def _checked_session_key(storage_key: str) -> str:
    """Reject storage keys that could escape the store root or collide with the
    store's own .partial/.failed artifact files. The message never echoes the key.
    """
    if type(storage_key) is not str or storage_key == "" or "\x00" in storage_key:
        raise ValueError("unsafe session storage key")
    if "/" in storage_key or "\\" in storage_key or storage_key in {".", ".."}:
        raise ValueError("unsafe session storage key")
    if storage_key.lower().endswith(_RESERVED_KEY_SUFFIXES):
        raise ValueError("unsafe session storage key")
    return storage_key


class SessionIncomplete(Exception):
    """Session did not receive the expected number of turns."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Candidate:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload, ensure_ascii=False))


def new_candidate(
    *,
    run_id: str,
    turn_idx: int,
    final_text: str,
    thinking_text: str | None = None,
    raw_response_bytes: bytes | None = None,
    api_key: str | None = None,
    failure_class: str | None = None,
    finish_status: str = "completed",
    session_storage_key: str = "",
    query_id: str = "",
    run_idx: int | None = None,
    mode: str = "deterministic",
    provider: str = "minimax",
    protocol: str = "anthropic",
    model: str = "MiniMax-M3",
    thinking_profile: str = "disabled",
    thinking_request: Mapping[str, Any] | None = None,
    corpus_snapshot_hash: str = "",
    retriever_config_hash: str = "",
    retrieval_captures: list[dict[str, Any]] | None = None,
    question_id: str = "",
    question_persisted: bool = False,
    question: Mapping[str, Any] | None = None,
    quiz_action: str | None = None,
    usage: Mapping[str, Any] | str | None = None,
    finish_raw: Any = None,
    reasoning_tokens: Any = "unavailable",
    error_type: str | None = None,
    error_message: str | None = None,
    wall_time_s: float | None = None,
    langchain_response_sha256: str | None = None,
    response_count: int = 0,
    thinking_char_length: int | None = None,
) -> Candidate:
    del api_key
    del error_message
    char_length = (
        thinking_char_length if thinking_char_length is not None else len(thinking_text or "")
    )
    present = bool(thinking_text) or char_length > 0
    cleaned_error = None
    if isinstance(error_type, str) and error_type.isidentifier():
        cleaned_error = error_type
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "session_storage_key": session_storage_key,
        "turn_idx": turn_idx,
        "query_id": query_id,
        "run_idx": run_idx,
        "mode": mode,
        "provider": provider,
        "protocol": protocol,
        "model": model,
        "thinking_profile": thinking_profile,
        "thinking_request": dict(thinking_request or {"type": "disabled"}),
        "corpus_snapshot_hash": corpus_snapshot_hash,
        "retriever_config_hash": retriever_config_hash,
        "retrieval_captures": list(retrieval_captures or []),
        "final_text": final_text,
        "question_id": question_id,
        "question_persisted": question_persisted,
        "question": dict(question) if question else None,
        "quiz_action": quiz_action,
        "usage": "unavailable" if usage is None else usage,
        "finish_status": finish_status,
        "finish_raw": finish_raw,
        "failure_class": failure_class,
        "error_type": cleaned_error,
        "reasoning_tokens": reasoning_tokens,
        "thinking": {
            "present": present,
            "char_length": char_length,
        },
        "raw_response_sha256": _sha256_bytes(raw_response_bytes or b""),
        "langchain_response_sha256": langchain_response_sha256,
        "response_count": response_count,
        "wall_time_s": wall_time_s,
    }
    return Candidate(payload)


class CandidateStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, list[Candidate]] = {}

    def _complete_path(self, storage_key: str) -> Path:
        return self.root / f"{storage_key}.json"

    def _partial_path(self, storage_key: str) -> Path:
        return self.root / f"{storage_key}.partial.json"

    def has_complete_session(self, storage_key: str) -> bool:
        _checked_session_key(storage_key)
        return self._complete_path(storage_key).exists()

    def load_session(self, storage_key: str) -> list[dict[str, Any]] | None:
        _checked_session_key(storage_key)
        path = self._complete_path(storage_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def begin_session(self, storage_key: str) -> None:
        _checked_session_key(storage_key)
        if self.has_complete_session(storage_key):
            raise ValueError("immutable")
        self._open[storage_key] = []
        self._write_partial(storage_key)

    def add_turn(self, storage_key: str, candidate: Candidate) -> None:
        _checked_session_key(storage_key)
        if self.has_complete_session(storage_key):
            raise ValueError("immutable")
        if storage_key not in self._open:
            raise ValueError(f"session not started: {storage_key}")
        payload = candidate.to_dict()
        payload["session_storage_key"] = storage_key
        self._open[storage_key].append(Candidate(payload))
        self._write_partial(storage_key)

    def commit_session(self, storage_key: str, expected_turns: int) -> None:
        _checked_session_key(storage_key)
        if self.has_complete_session(storage_key):
            raise ValueError("immutable")
        if storage_key not in self._open:
            raise ValueError(f"session not started: {storage_key}")
        turns = self._open[storage_key]
        if len(turns) != expected_turns:
            raise SessionIncomplete(
                f"{storage_key} has {len(turns)} turns, expected {expected_turns}"
            )
        complete = [item.to_dict() for item in turns]
        dest = self._complete_path(storage_key)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(dest)
        self._partial_path(storage_key).unlink(missing_ok=True)
        self._open.pop(storage_key, None)

    def discard_session(self, storage_key: str) -> None:
        _checked_session_key(storage_key)
        self._partial_path(storage_key).unlink(missing_ok=True)
        self._open.pop(storage_key, None)

    def record_failure(self, storage_key: str, candidate: Candidate) -> None:
        _checked_session_key(storage_key)
        payload = candidate.to_dict()
        payload["session_storage_key"] = storage_key
        path = self.root / f"{storage_key}.failed.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_failure(self, storage_key: str) -> dict[str, Any] | None:
        _checked_session_key(storage_key)
        path = self.root / f"{storage_key}.failed.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def discard_incomplete(self) -> None:
        for path in self.root.glob("*.partial.json"):
            path.unlink()
        self._open.clear()

    def _write_partial(self, storage_key: str) -> None:
        payload = [item.to_dict() for item in self._open.get(storage_key, [])]
        self._partial_path(storage_key).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )


_RETRYABLE_ERROR_CODES = frozenset({"missing_judge", "parse"})
_RETRYABLE_FAILURE_CLASSES = frozenset({"transport", "provider"})
_NON_RETRYABLE_ERROR_CODES = frozenset({"structured_artifact_failure"})


def encode_path_segment(value: str) -> str:
    if type(value) is not str or value == "" or value in {".", ".."}:
        raise ValueError(f"unsafe path segment: {value!r}")
    encoded = quote(value, safe="")
    if not encoded or encoded in {".", ".."}:
        raise ValueError(f"unsafe path segment: {value!r}")
    return encoded


def encode_scorer_id(scorer_id: str) -> str:
    return encode_path_segment(scorer_id)


def decode_scorer_id(encoded: str) -> str:
    return unquote(encoded)


class ScoreStore:
    """Append-only ScorerExecution store. Legacy scores/{run_id}.json is read-only."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executions_root = self.root / "executions"

    def legacy_path(self, run_id: str) -> Path:
        return self.root / f"{encode_path_segment(run_id)}.json"

    def _scorer_dir(self, run_id: str, scorer_id: str) -> Path:
        return self.executions_root / encode_path_segment(run_id) / encode_scorer_id(scorer_id)

    def load_legacy(self, run_id: str) -> list[dict[str, Any]]:
        path = self.legacy_path(run_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def load_appended(self, run_id: str, scorer_id: str) -> list[dict[str, Any]]:
        directory = self._scorer_dir(run_id, scorer_id)
        if not directory.is_dir():
            return []
        listed = sorted(directory.glob("*.json"))
        numeric_slots = [
            index
            for index, path in enumerate(listed)
            if path.stem.isascii() and path.stem.isdecimal()
        ]
        if numeric_slots:
            # Append files carrying a numeric stem must be read in integer order, so
            # 9999 precedes 10000. Only the numeric slots move; every other file keeps
            # its lexicographic position, and equal ordinals keep their original order
            # without being deduplicated.
            ranked = sorted(
                (listed[index] for index in numeric_slots),
                key=lambda path: int(path.stem),
            )
            reordered = list(listed)
            for slot, path in zip(numeric_slots, ranked):
                reordered[slot] = path
            listed = reordered
        rows: list[dict[str, Any]] = []
        for path in listed:
            if path.name.endswith(".tmp"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
        return rows

    def history(self, run_id: str, scorer_id: str) -> list[dict[str, Any]]:
        legacy = [
            item
            for item in self.load_legacy(run_id)
            if str(item.get("scorer_id") or "") == scorer_id
        ]
        return legacy + self.load_appended(run_id, scorer_id)

    def selected(self, run_id: str, scorer_id: str) -> dict[str, Any] | None:
        hist = self.history(run_id, scorer_id)
        if not hist:
            return None
        for item in hist:
            if item.get("status") == "success":
                return item
        for item in hist:
            if item.get("status") == "skipped":
                return item
        return hist[-1]

    def needs_retry(self, run_id: str, scorer_id: str) -> bool:
        current = self.selected(run_id, scorer_id)
        if current is None:
            return True
        if current.get("status") in {"success", "skipped"}:
            return False
        error_code = current.get("error_code")
        if error_code in _NON_RETRYABLE_ERROR_CODES:
            return False
        if error_code in _RETRYABLE_ERROR_CODES:
            return True
        if current.get("failure_class") in _RETRYABLE_FAILURE_CLASSES:
            return True
        return False

    def append(self, run_id: str, execution: Mapping[str, Any]) -> Path:
        if fcntl is None:
            raise RuntimeError("ScoreStore.append requires fcntl lock")
        scorer_id = str(execution.get("scorer_id") or "")
        directory = self._scorer_dir(run_id, scorer_id)
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".append.lock"
        with open(lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                seqs: list[int] = []
                for path in directory.glob("*.json"):
                    if path.name.endswith(".tmp"):
                        continue
                    if path.stem.isdigit():
                        seqs.append(int(path.stem))
                seq = (max(seqs) + 1) if seqs else 1
                dest = directory / f"{seq:04d}.json"
                tmp = directory / f"{seq:04d}.{os.getpid()}.{uuid.uuid4().hex}.json.tmp"
                tmp.write_text(
                    json.dumps(dict(execution), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(tmp, dest)
                return dest
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def known_scorer_ids(self, run_id: str) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for item in self.load_legacy(run_id):
            sid = str(item.get("scorer_id") or "")
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
        run_dir = self.executions_root / encode_path_segment(run_id)
        if run_dir.is_dir():
            for path in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                sid = decode_scorer_id(path.name)
                if sid and sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids

    def selected_executions(
        self,
        run_id: str,
        scorer_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        ids = list(scorer_ids) if scorer_ids is not None else self.known_scorer_ids(run_id)
        out: list[dict[str, Any]] = []
        for sid in ids:
            item = self.selected(run_id, sid)
            if item is not None:
                out.append(item)
        return out
