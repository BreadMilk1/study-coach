from __future__ import annotations

import json
import socket
from typing import Any, Literal, Mapping


FinishStatus = Literal["completed", "truncated", "failed"]
ThinkingTokens = int | Literal["unavailable"]


class SmokeAbort(Exception):
    """Harness smoke or matrix failed a gate; do not continue the stage."""


_TRANSPORT_TYPE_NAMES = frozenset({
    "AuthenticationError",
    "PermissionDeniedError",
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "APIStatusError",
    "InternalServerError",
    "APIError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "TimeoutException",
    "HTTPStatusError",
    "RemoteProtocolError",
})


_NETWORK_OS_ERROR_TYPES = (TimeoutError, ConnectionError, socket.gaierror, socket.herror)
_HARNESS_LOCAL_TYPES = (
    AttributeError,
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    FileExistsError,
)


def failure_class_for_exception(exc: BaseException) -> str:
    """Classify a failure as transport, harness or model.

    The whole cause/context chain is walked (cycle-guarded) before deciding, because a
    provider wrapper can hide either a local failure or a network failure. Explicit local
    evidence wins over network evidence, and network evidence wins over a bare OSError.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    names: list[str] = []
    local_evidence = False
    network_evidence = False
    os_error = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _HARNESS_LOCAL_TYPES):
            local_evidence = True
        elif isinstance(current, OSError):
            os_error = True
            if isinstance(current, _NETWORK_OS_ERROR_TYPES):
                network_evidence = True
        names.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    if any(name in _TRANSPORT_TYPE_NAMES for name in names):
        network_evidence = True
    if local_evidence:
        return "harness"
    if network_evidence:
        return "transport"
    if os_error:
        return "harness"
    return "model"


def flatten_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        block_type = (
            block.get("type") if isinstance(block, Mapping)
            else getattr(block, "type", None)
        )
        if block_type in {"thinking", "tool_use"}:
            continue
        if block_type == "text":
            text = (
                block.get("text") if isinstance(block, Mapping)
                else getattr(block, "text", "")
            )
            parts.append(text or "")
    return "".join(parts)


def normalize_finish_status(*, protocol: str, raw_stop: str | None) -> FinishStatus:
    if protocol == "anthropic":
        if raw_stop in {"end_turn", "tool_use"}:
            return "completed"
        if raw_stop == "max_tokens":
            return "truncated"
        return "failed"
    if protocol == "openai_compatible":
        if raw_stop == "stop":
            return "completed"
        if raw_stop == "length":
            return "truncated"
        return "failed"
    return "failed"


def extract_thinking_tokens(*, protocol: str, usage: Mapping[str, Any] | None) -> ThinkingTokens:
    if not isinstance(usage, Mapping):
        return "unavailable"
    if protocol == "anthropic":
        details = usage.get("output_tokens_details")
        key = "thinking_tokens"
    elif protocol == "openai_compatible":
        details = usage.get("completion_tokens_details")
        key = "reasoning_tokens"
    else:
        return "unavailable"
    if not isinstance(details, Mapping) or key not in details:
        return "unavailable"
    value = details[key]
    if type(value) is not int or value < 0:
        return "unavailable"
    return value


def extract_usage(response: Any) -> dict[str, int] | None:
    sources: list[Mapping[str, Any]] = []
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, Mapping):
        sources.append(meta)
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        for key in ("usage", "token_usage"):
            raw = response_metadata.get(key)
            if isinstance(raw, Mapping):
                sources.append(raw)
    for source in sources:
        input_tokens = source.get("input_tokens", source.get("prompt_tokens"))
        output_tokens = source.get("output_tokens", source.get("completion_tokens"))
        total_tokens = source.get("total_tokens")
        if type(input_tokens) is int and type(output_tokens) is int and input_tokens >= 0 and output_tokens >= 0:
            if type(total_tokens) is not int or total_tokens < 0:
                total_tokens = input_tokens + output_tokens
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
    return None


_SECRET_KEYS = frozenset({"api_key", "authorization", "secret", "password"})


def _is_secret_key(key: str) -> bool:
    # Header names arrive in every case style; normalise case and "-" before matching so
    # x-api-key follows the same rule as x_api_key. Retained key names are untouched.
    normalized = key.lower().replace("-", "_")
    if normalized in _SECRET_KEYS or normalized.endswith("_api_key"):
        return True
    # camelCase and acronym forms (apiKey, APIKey, serviceApiKey) collapse to one
    # separator-free suffix. A lower/upper boundary regex would miss APIKey.
    return normalized.replace("_", "").endswith("apikey")


def _jsonable_fingerprint(value: Any) -> Any:
    """Canonical value for LangChain response fingerprints. Never uses repr()."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, Mapping):
        return {
            str(k): _jsonable_fingerprint(v)
            for k, v in value.items()
            if not _is_secret_key(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_fingerprint(item) for item in value]
    payload: dict[str, Any] = {}
    for attr in ("type", "text", "thinking", "id", "name", "input", "index"):
        if hasattr(value, attr):
            payload[attr] = _jsonable_fingerprint(getattr(value, attr))
    return payload or None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def langchain_response_canonical(response: Any) -> dict[str, Any]:
    """Fingerprint payload for a LangChain AIMessage / chunk — not a raw HTTP body."""
    content = getattr(response, "content", None)
    tool_calls = getattr(response, "tool_calls", None)
    usage_metadata = getattr(response, "usage_metadata", None)
    response_metadata = getattr(response, "response_metadata", None)
    return {
        "content": _jsonable_fingerprint(content),
        "tool_calls": _jsonable_fingerprint(tool_calls) or [],
        "usage_metadata": _jsonable_fingerprint(usage_metadata) if isinstance(usage_metadata, Mapping) else None,
        "response_metadata": _jsonable_fingerprint(response_metadata) if isinstance(response_metadata, Mapping) else None,
    }


def thinking_char_length(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        block_type = (
            block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
        )
        if block_type != "thinking":
            continue
        text = (
            block.get("thinking") if isinstance(block, Mapping)
            else getattr(block, "thinking", None)
        )
        if text is None:
            text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", "")
        total += len(str(text or ""))
    return total


def finish_raw_from_response(response: Any) -> dict[str, Any] | None:
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    cleaned: dict[str, Any] = {}
    for key in ("stop_reason", "finish_reason"):
        if key in metadata and metadata[key] is not None:
            cleaned[key] = metadata[key]
    return cleaned or None


def raw_usage_from_response(response: Any) -> Mapping[str, Any] | None:
    """Prefer the full nested usage object (for thinking token dumps). Do not flatten to top-level keys only."""
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        raw = response_metadata.get("usage")
        if isinstance(raw, Mapping):
            return raw
        raw = response_metadata.get("token_usage")
        if isinstance(raw, Mapping):
            return raw
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, Mapping):
        return meta
    return None
