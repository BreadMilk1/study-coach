"""Batch A — fixed safe projection for LLM failures.

`normalize_llm_error` is the single conversion point between a raw exception
(or a legacy ``"<Category>: <detail>"`` string) and the only text allowed to
leave the agent/API boundary into traces, SSE payloads, chat persistence and
history reads.

Design constraints:

  - no SDK, provider, logger or dependency is imported here; SDK-style
    failures are matched by exception class *name*, so ``APIConnectionError``
    and friends never need to be imported
  - only type information is read from an exception: ``str`` / ``repr`` /
    ``args`` / ``request`` / ``response`` are never touched, so a
    detail-bearing exception object cannot leak through this path
  - the category text always comes from the fixed table below; arbitrary
    exception class names are never interpolated
  - ``None`` stays ``None``, and re-projecting already-safe text is idempotent
"""
from __future__ import annotations

_CONNECTION_MESSAGE = "Could not connect to the model service."
_TIMEOUT_MESSAGE = "The model request timed out."
_GENERIC_LLM_ERROR = "LLMError: Model request failed."

# Fixed category table: exact exception class name -> fixed safe sentence.
_SAFE_LLM_ERRORS: dict[str, str] = {
    "ConnectionRefusedError": f"ConnectionRefusedError: {_CONNECTION_MESSAGE}",
    "ConnectionError": f"ConnectionError: {_CONNECTION_MESSAGE}",
    "APIConnectionError": f"APIConnectionError: {_CONNECTION_MESSAGE}",
    "TimeoutError": f"TimeoutError: {_TIMEOUT_MESSAGE}",
    "APITimeoutError": f"APITimeoutError: {_TIMEOUT_MESSAGE}",
    "AuthenticationError": "AuthenticationError: Model authentication failed.",
    "PermissionDeniedError": "PermissionDeniedError: Model access was denied.",
    "RateLimitError": "RateLimitError: The model service rate limit was reached.",
}


def normalize_llm_error(value: object) -> str | None:
    """Project an exception object or legacy error string onto fixed safe text.

    - ``None`` → ``None`` (an absent field stays absent)
    - exception object → the fixed sentence for its exact class name, otherwise
      the generic LLM error; the exception itself is never stored or stringified
    - legacy ``str`` → the fixed sentence only when the text before the first
      colon is an exact whitelisted category (surrounding whitespace ignored);
      everything after the colon is discarded, never partially rewritten
    - unknown category, empty string or any other non-``None`` value → generic
    """
    if value is None:
        return None
    if isinstance(value, str):
        category, separator, _detail = value.strip().partition(":")
        if not separator:
            return _GENERIC_LLM_ERROR
        return _SAFE_LLM_ERRORS.get(category.strip(), _GENERIC_LLM_ERROR)
    return _SAFE_LLM_ERRORS.get(type(value).__name__, _GENERIC_LLM_ERROR)
