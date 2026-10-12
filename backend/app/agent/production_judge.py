"""Production strict Judge (D3, judge_policy=production_v1).

`judge_response` is the production Judge entry: strict rubric parsing with no
neutral fallback, plus a negative-completion-signal gate. The legacy
`app.agent.judge.judge_response` stays byte-for-byte unchanged for the legacy
graph policy and the frozen historical scorers.

Contract (D3-CONTRACT.md §3):

- Body text comes from the existing `app.llm.content.require_text`. The whole
  body must be a complete JSON object, or exactly one ```json / unlabeled
  JSON fence with only surrounding whitespace. Prose around the payload,
  multiple objects, multiple fences, truncated JSON, duplicate keys and
  non-object payloads are all rejected — no substring search, no JSON
  repair, no think-tag stripping.
- Every rubric dimension must exist with a real finite int/float within 1–5
  (bool and numeric strings are rejected). The range check happens before any
  float conversion, so huge integers and 1e400-style exponent overflow are
  safely rejected. No defaulting, clamping or neutral scores.
- `reasoning` defaults to a fixed empty string; when present it must be a
  str. Extra fields are never interpreted or returned.
- Negative completion signals are read from the public message fields
  (finish_reason / stop_reason / done_reason / done / tool_calls /
  invalid_tool_calls / additional_kwargs.refusal / Gemini
  prompt_feedback.block_reason) and block scoring even when the body is a
  valid high-score JSON. Field and container shapes are defended first;
  missing, unspecified or unknown states never fabricate a completion
  verdict.
- A request failure is converted to unavailable only around the single
  ainvoke call, for the approved catchable classes (ValueError, OSError and
  the fixed model-service SDK class names matched by name/MRO). Prompt
  building and program errors propagate; CancelledError propagates.
  `app.llm.errors.normalize_llm_error` is the only raw-exception projection;
  str/repr/args/request/response are never read.

Evidence limit: verified with fake models, synthetic message shapes and
temporary storage only; no real provider response was observed. A valid
score means the payload was legal — it is not provider completion evidence.
"""

import asyncio
import json
import math
import re

from langchain_core.messages import HumanMessage

from app.llm.content import require_text
from app.llm.errors import normalize_llm_error

TUTOR_DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "accuracy",
    "citation_quality",
    "accessibility",
    "example_quality",
    "learner_level_fit",
)

_WEAK_DIM_THRESHOLD = 3  # ≤ this surfaces in weak_dims
_PASS_SCORE_THRESHOLD = 0.6

# Fixed safe unavailable reasons — never contain payload fragments.
_REASON_COMPLETION_SIGNAL = (
    "Judge response signalled an incomplete or blocked completion."
)
_REASON_INVALID_PAYLOAD = "Judge response was not a valid scoring payload."
NO_JUDGE_MODEL_REASON = "Judge model is not available."

# Model-service SDK error class names matched by name/MRO; no SDK is imported.
_MODEL_SERVICE_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "APIError",
        "APIStatusError",
        "APIConnectionError",
        "APITimeoutError",
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "GoogleAPIError",
        "GoogleAPICallError",
        "ClientError",
        "ServerError",
        "ResponseError",
        # Locked langchain-google-genai 4.2.0 wraps google.genai.errors.ClientError
        # into ChatGoogleGenerativeAIError via `_handle_client_error`
        # (chat_models.py:133-145). Async judge_llm.ainvoke calls raise the
        # wrapper directly; the prior list omitted it, so the wrapper escaped
        # the ainvoke boundary and aborted the production Chat.
        "ChatGoogleGenerativeAIError",
    }
)

# finish_reason values that mean an incomplete/blocked turn: OpenAI Chat
# values plus the Gemini FinishReason enum names as projected by the locked
# langchain-google-genai adapter (response_metadata["finish_reason"]).
_NEGATIVE_FINISH_REASONS: frozenset[str] = frozenset(
    {
        "length",
        "content_filter",
        "tool_calls",
        "function_call",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "NO_IMAGE",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
    }
)

# Anthropic stop reasons (locked adapter: response_metadata["stop_reason"]).
_NEGATIVE_STOP_REASONS: frozenset[str] = frozenset(
    {"max_tokens", "tool_use", "pause_turn", "refusal"}
)

# Ollama done fields (locked adapter: response_metadata done_reason / done).
_NEGATIVE_DONE_REASONS: frozenset[str] = frozenset({"length", "load"})

# Gemini BlockedReason values from the locked SDK (unspecified/0 is not a
# block); prompt_feedback.block_reason as dumped into response_metadata.
_GEMINI_BLOCKED_REASONS: frozenset[str] = frozenset(
    {
        "SAFETY",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
        "JAILBREAK",
    }
)


def is_model_service_error(exc: BaseException) -> bool:
    """True only for the approved model-service failure classes.

    Matched strictly by exception class name along the MRO; the names are
    never interpolated into any output.
    """
    return any(
        cls.__name__ in _MODEL_SERVICE_ERROR_NAMES for cls in type(exc).__mro__
    )


class _InvalidJudgePayload(ValueError):
    """Internal marker: the judge body is not a valid scoring payload.

    The exception never leaves this module; callers map it to the fixed safe
    reason below.
    """


def _reject_duplicate_keys(pairs):
    seen: set = set()
    for key, _value in pairs:
        if key in seen:
            raise _InvalidJudgePayload
        seen.add(key)
    return dict(pairs)


def _reject_nonstandard_constant(_name: str):
    raise _InvalidJudgePayload


_FENCE_OBJECT_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_payload_text(raw: str) -> str:
    """Return the JSON body for a complete object or exactly one fence."""
    stripped = raw.strip()
    if not stripped:
        raise _InvalidJudgePayload
    if not stripped.startswith("```"):
        return stripped
    # Exactly one fence: opening ```/```json and one closing ``` only.
    if stripped.count("```") != 2:
        raise _InvalidJudgePayload
    match = _FENCE_OBJECT_RE.fullmatch(stripped)
    if match is None:
        raise _InvalidJudgePayload
    return match.group(1)


def _parse_strict_object(body: str) -> dict:
    try:
        parsed = json.loads(
            body,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except _InvalidJudgePayload:
        raise
    except RecursionError:
        # json.loads raises RecursionError on deeply nested inputs at the
        # boundary of the current Python recursion limit. The bug fix lives
        # at this single boundary: the body is malformed, not a program bug,
        # so it must map to the fixed invalid-payload marker (the same
        # unavailable reason as any other malformed body). Do not widen this
        # catch — prompt / formatter / state / writer RecursionErrors, general
        # RuntimeError, or any other exception class must keep propagating.
        raise _InvalidJudgePayload from None
    except ValueError:
        raise _InvalidJudgePayload from None
    if not isinstance(parsed, dict):
        raise _InvalidJudgePayload
    return parsed


def _validated_dimension(payload: dict, dim: str) -> float:
    if dim not in payload:
        raise _InvalidJudgePayload
    value = payload[dim]
    # bool is an int subclass; reject it before the numeric check. The range
    # comparison runs before any float conversion, so huge integers and
    # exponent overflow cannot raise or clamp.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidJudgePayload
    if isinstance(value, float) and not math.isfinite(value):
        raise _InvalidJudgePayload
    if value < 1 or value > 5:
        raise _InvalidJudgePayload
    return float(value)


def _validated_reasoning(payload: dict) -> str:
    # Distinguish "key absent" from "key present and null" — the contract
    # treats them differently:
    # - missing key  → empty string, still valid (default fallback)
    # - present + str → the verbatim value (empty string is legal)
    # - present + null / bool / int / float / list / object → malformed payload
    # The previous implementation collapsed both branches into the missing-key
    # default, silently turning `reasoning: null` into a clean pass.
    if "reasoning" not in payload:
        return ""
    reasoning = payload["reasoning"]
    if reasoning is None or not isinstance(reasoning, str):
        raise _InvalidJudgePayload
    return reasoning


def _has_negative_completion_signal(response) -> bool:
    """Defensive read of the public completion fields; shape first."""
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        finish_reason = metadata.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason in _NEGATIVE_FINISH_REASONS:
            return True
        stop_reason = metadata.get("stop_reason")
        if isinstance(stop_reason, str) and stop_reason in _NEGATIVE_STOP_REASONS:
            return True
        done_reason = metadata.get("done_reason")
        if isinstance(done_reason, str) and done_reason in _NEGATIVE_DONE_REASONS:
            return True
        if metadata.get("done") is False:
            return True
        prompt_feedback = metadata.get("prompt_feedback")
        if isinstance(prompt_feedback, dict):
            blocked = prompt_feedback.get("block_reason")
            if isinstance(blocked, str) and blocked in _GEMINI_BLOCKED_REASONS:
                return True
    tool_calls = getattr(response, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        return True
    invalid_tool_calls = getattr(response, "invalid_tool_calls", None)
    if isinstance(invalid_tool_calls, list) and invalid_tool_calls:
        return True
    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        refusal = additional_kwargs.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return True
    return False


def _valid_result(
    score: float, verdict: str, weak_dims: list[str], reasoning: str
) -> dict:
    return {
        "status": "valid",
        "score": score,
        "verdict": verdict,
        "weak_dims": weak_dims,
        "reasoning": reasoning,
    }


def _unavailable_result(reason: str) -> dict:
    return {
        "status": "unavailable",
        "score": None,
        "verdict": None,
        "reason": reason,
    }


async def judge_response(
    *,
    question: str,
    answer: str,
    context: str,
    rubric: str,
    judge_llm,
    dimensions: tuple[str, ...] | None = None,
) -> dict:
    """Run one strict production Judge call.

    Returns a `valid` scoring payload or an `unavailable` result with a fixed
    safe reason; it never invents a score, a verdict or completion evidence.
    """
    dims = dimensions or TUTOR_DIMENSIONS
    prompt = rubric.format(question=question, answer=answer, context=context)
    try:
        response = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    except asyncio.CancelledError:
        raise
    except (ValueError, OSError) as exc:
        return _unavailable_result(normalize_llm_error(exc))
    except Exception as exc:
        if not is_model_service_error(exc):
            raise
        return _unavailable_result(normalize_llm_error(exc))

    if _has_negative_completion_signal(response):
        return _unavailable_result(_REASON_COMPLETION_SIGNAL)

    try:
        raw = require_text(getattr(response, "content", ""))
        payload = _parse_strict_object(_extract_payload_text(raw))
        dim_scores = [_validated_dimension(payload, dim) for dim in dims]
        reasoning = _validated_reasoning(payload)
    except ValueError:
        # Both the strict-shape marker and require_text's fixed empty/malformed
        # body rejection map to the same fixed safe reason.
        return _unavailable_result(_REASON_INVALID_PAYLOAD)

    average = sum(dim_scores) / len(dims)
    score = round(average / 5.0, 4)
    verdict = "pass" if score >= _PASS_SCORE_THRESHOLD else "weak"
    weak_dims = [
        dim
        for dim, value in zip(dims, dim_scores)
        if value <= _WEAK_DIM_THRESHOLD
    ]
    return _valid_result(score, verdict, weak_dims, reasoning)
