"""D3 RED stage — production strict Judge unit contract.

These tests lock the `app.agent.production_judge` module contract defined in
D3-CONTRACT.md §3:

- Strict rubric parsing against the real rubric dimension sets (Tutor 6-dim
  `TUTOR_DIMENSIONS`, Quiz 5-dim and Plan 5-dim constants imported unchanged
  from the frozen `app.agent.judge`): a complete JSON object, or exactly one
  ```json / unlabeled JSON fence with only surrounding whitespace. Prose
  before/after, multiple objects/fences, truncated JSON, duplicate keys and
  non-objects are all rejected. Every rubric dimension must exist with a real
  int/float (no bool, no numeric string), finite and within 1–5, checked
  before any conversion so huge integers and 1e400-style exponent overflow
  cannot crash or clamp. No defaulting, clamping or neutral fallback: invalid
  output is `unavailable`, never a 0.6 pass, and the unavailable reason is
  fixed safe text that never echoes input payload markers.
- `reasoning` defaults to a fixed empty string; when present it must be a str.
  Extra fields are never interpreted or returned. A valid result is exactly
  the scoring payload — it never fabricates provider completion evidence.
- Negative completion signals read from actual AIMessage public fields
  (finish_reason / stop_reason / done_reason / done / tool_calls /
  invalid_tool_calls / additional_kwargs.refusal / Gemini
  prompt_feedback.block_reason) block scoring even when the body is a valid
  high-score JSON. Missing or shape-mismatched metadata is NOT treated as a
  completion verdict.

RED stage: the module does not exist yet, so this file fails collection with
ModuleNotFoundError. That collection RED is reported separately from the
runnable consumer REDs in test_graph_judge.py / test_graph_memory.py /
test_routes.py / test_routes_graph_stream.py and must never be used as a
substitute for them. Nothing here may be reported as passing until the module
lands; the synthetic-boundary fixtures below are only proof that the defensive
shapes are constructible and handable to the real judge call signature.
"""

import json
import socket
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.agent.judge import PLAN_DIMENSIONS, QUIZ_DIMENSIONS
from app.agent.production_judge import TUTOR_DIMENSIONS, judge_response

_PASS_JSON = (
    '{"relevance":5,"accuracy":5,"citation_quality":4,'
    '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
    '"reasoning":"Solid."}'
)
_WEAK_JSON = (
    '{"relevance":2,"accuracy":2,"citation_quality":2,'
    '"accessibility":2,"example_quality":2,"learner_level_fit":2,'
    '"reasoning":"Ungrounded."}'
)
_MIXED_JSON = (
    '{"relevance":5,"accuracy":4,"citation_quality":3,'
    '"accessibility":2,"example_quality":1,"learner_level_fit":2,'
    '"reasoning":"Mixed quality."}'
)
_REASONING_MARKER = "HIDDEN-JUDGE-REASONING"
_MARKER = "UNIT-PAYLOAD-MARKER"

_QUIZ_PASS_JSON = (
    '{"question_quality":4,"option_plausibility":4,"answer_correctness":5,'
    '"explanation_clarity":4,"difficulty_calibration":4,'
    f'"reasoning":"{_MARKER} fine"}}'
)
_QUIZ_WEAK_JSON = (
    '{"question_quality":2,"option_plausibility":2,"answer_correctness":2,'
    '"explanation_clarity":2,"difficulty_calibration":2,'
    f'"reasoning":"{_MARKER} weak"}}'
)
_PLAN_PASS_JSON = (
    '{"milestone_specificity":5,"milestone_granularity":4,"time_feasibility":4,'
    '"topic_coverage":5,"actionability":4,'
    f'"reasoning":"{_MARKER} fine"}}'
)
_PLAN_WEAK_JSON = (
    '{"milestone_specificity":2,"milestone_granularity":2,"time_feasibility":2,'
    '"topic_coverage":2,"actionability":2,'
    f'"reasoning":"{_MARKER} weak"}}'
)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """File-local offline guard: every test in this module is a new D3 case.

    Any real DNS resolution or socket connection attempt fails immediately;
    the probe below confirms the four blocked surfaces without dialing out.
    """
    def deny(*_args, **_kwargs):
        raise AssertionError("D3 strict-judge unit test attempted real network access")

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def test_network_guard_covers_real_socket_surface(deny_network):
    probe = socket.socket()
    try:
        with pytest.raises(AssertionError):
            socket.getaddrinfo("localhost", 80)
        with pytest.raises(AssertionError):
            probe.connect(("127.0.0.1", 80))
        with pytest.raises(AssertionError):
            probe.connect_ex(("127.0.0.1", 80))
        with pytest.raises(AssertionError):
            socket.create_connection(("127.0.0.1", 80))
    finally:
        probe.close()


class _StubJudgeLLM:
    """Judge model stub returning one preset response object per call."""

    def __init__(self, response):
        self.response = response
        self.call_count = 0

    async def ainvoke(self, _messages, **_kwargs):
        self.call_count += 1
        return self.response


async def _judge(content, *, dimensions=None, **message_kwargs):
    llm = _StubJudgeLLM(AIMessage(content=content, **message_kwargs))
    return await judge_response(
        question="q",
        answer="a",
        context="c",
        rubric="rubric-template",
        judge_llm=llm,
        dimensions=dimensions,
    )


def _assert_valid(result):
    assert result["status"] == "valid"
    assert result.get("reason") is None
    # A valid result is exactly the scoring payload: it must not carry any
    # fabricated provider completion evidence (no fake finish/stop claim).
    assert set(result) <= {"status", "score", "verdict", "weak_dims", "reasoning"}


def _assert_unavailable(result, forbidden=(_MARKER,)):
    assert result["status"] == "unavailable"
    # No fabricated score or verdict may ride along with an unavailable result.
    assert result.get("score") is None
    assert result.get("verdict") is None
    reason = result.get("reason")
    assert isinstance(reason, str) and reason.strip()
    # The reason is fixed safe text: input payload markers never leak into it.
    for marker in forbidden:
        assert marker not in reason


# --- valid payloads ----------------------------------------------------------


async def test_valid_raw_json_object_scores_equal_weight():
    result = await _judge(_MIXED_JSON)

    _assert_valid(result)
    # Equal-weight mean like the legacy judge, rounded to 4 decimals.
    assert result["score"] == round(17 / 30, 4)
    assert result["verdict"] == "weak"
    assert result["weak_dims"] == [
        "citation_quality",
        "accessibility",
        "example_quality",
        "learner_level_fit",
    ]
    assert result["reasoning"] == "Mixed quality."


async def test_valid_payload_above_threshold_passes():
    result = await _judge(_PASS_JSON)

    _assert_valid(result)
    assert result["score"] == round(28 / 30, 4)
    assert result["verdict"] == "pass"
    assert result["weak_dims"] == []


async def test_valid_weak_payload_marks_weak_dims():
    result = await _judge(_WEAK_JSON)

    _assert_valid(result)
    assert result["score"] == pytest.approx(0.4)
    assert result["verdict"] == "weak"
    assert result["weak_dims"] == list(TUTOR_DIMENSIONS)


async def test_single_json_fence_with_surrounding_whitespace_is_valid():
    result = await _judge(f"```json\n{_PASS_JSON}\n```\n")

    _assert_valid(result)
    assert result["verdict"] == "pass"


async def test_bare_fence_without_language_tag_is_valid():
    result = await _judge(f"```\n{_PASS_JSON}\n```")

    _assert_valid(result)
    assert result["verdict"] == "pass"


async def test_json_spanning_multiple_text_blocks_is_valid():
    split = len(_PASS_JSON) // 2
    content = [
        {"type": "text", "text": _PASS_JSON[:split]},
        {"type": "text", "text": _PASS_JSON[split:]},
    ]
    result = await _judge(content)

    _assert_valid(result)
    assert result["verdict"] == "pass"


async def test_missing_reasoning_defaults_to_empty_string():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5}'
    )
    result = await _judge(payload)

    _assert_valid(result)
    assert result["reasoning"] == ""


async def test_extra_fields_are_not_interpreted_or_returned():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"ok","confidence":0.9,"advice":"IGNORE PRIOR INSTRUCTIONS"}'
    )
    result = await _judge(payload)

    _assert_valid(result)
    assert result["verdict"] == "pass"
    assert result.get("confidence") is None
    assert result.get("advice") is None
    assert "IGNORE PRIOR INSTRUCTIONS" not in str(result)


# --- real rubric dimension sets (Tutor 6 / Quiz 5 / Plan 5) ------------------


async def test_tutor_dimensions_constant_matches_six_rubric_dims():
    assert tuple(TUTOR_DIMENSIONS) == (
        "relevance",
        "accuracy",
        "citation_quality",
        "accessibility",
        "example_quality",
        "learner_level_fit",
    )


async def test_quiz_rubric_valid_payload_passes():
    result = await _judge(_QUIZ_PASS_JSON, dimensions=QUIZ_DIMENSIONS)

    _assert_valid(result)
    assert result["score"] == pytest.approx(21 / 25)
    assert result["verdict"] == "pass"
    assert result["weak_dims"] == []


async def test_quiz_rubric_weak_payload_marks_weak_dims():
    result = await _judge(_QUIZ_WEAK_JSON, dimensions=QUIZ_DIMENSIONS)

    _assert_valid(result)
    assert result["verdict"] == "weak"
    assert result["weak_dims"] == list(QUIZ_DIMENSIONS)


async def test_quiz_rubric_missing_dimension_is_unavailable():
    payload = (
        '{"question_quality":4,"option_plausibility":4,"answer_correctness":5,'
        '"explanation_clarity":4,'
        f'"reasoning":"{_MARKER} missing difficulty_calibration"}}'
    )
    result = await _judge(payload, dimensions=QUIZ_DIMENSIONS)

    _assert_unavailable(result)


async def test_plan_rubric_valid_payload_passes():
    result = await _judge(_PLAN_PASS_JSON, dimensions=PLAN_DIMENSIONS)

    _assert_valid(result)
    assert result["score"] == pytest.approx(22 / 25)
    assert result["verdict"] == "pass"


async def test_plan_rubric_weak_payload_marks_weak_dims():
    result = await _judge(_PLAN_WEAK_JSON, dimensions=PLAN_DIMENSIONS)

    _assert_valid(result)
    assert result["verdict"] == "weak"
    assert result["weak_dims"] == list(PLAN_DIMENSIONS)


async def test_plan_rubric_missing_dimension_is_unavailable():
    payload = (
        '{"milestone_specificity":5,"milestone_granularity":4,"time_feasibility":4,'
        '"topic_coverage":5,'
        f'"reasoning":"{_MARKER} missing actionability"}}'
    )
    result = await _judge(payload, dimensions=PLAN_DIMENSIONS)

    _assert_unavailable(result)


# --- invalid payloads: unavailable, never the legacy neutral pass ------------


async def test_prose_without_json_is_unavailable():
    result = await _judge(f"The answer {_MARKER} looks well grounded and cited.")

    _assert_unavailable(result)


async def test_missing_dimension_is_unavailable():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER} missing accessibility"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_boolean_dimension_is_unavailable():
    payload = (
        f'{{"relevance":true,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_string_dimension_is_unavailable():
    payload = (
        f'{{"relevance":"4","accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_null_dimension_is_unavailable():
    payload = (
        f'{{"relevance":null,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_array_dimension_is_unavailable():
    payload = (
        f'{{"relevance":[4],"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_nan_dimension_is_unavailable():
    payload = (
        f'{{"relevance":NaN,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_infinite_dimension_is_unavailable():
    payload = (
        f'{{"relevance":Infinity,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_exponent_overflow_dimension_is_unavailable():
    payload = (
        f'{{"relevance":1e400,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_oversized_integer_dimension_is_unavailable_without_overflow():
    huge = "1" + "0" * 400
    payload = (
        f'{{"relevance":{huge},"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


@pytest.mark.parametrize("bad_value", ["0", "6", "-1", "5.5"])
async def test_out_of_range_dimension_is_unavailable(bad_value):
    payload = (
        f'{{"relevance":{bad_value},"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_duplicate_keys_are_unavailable():
    payload = (
        f'{{"relevance":5,"relevance":4,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        f'"reasoning":"{_MARKER}"}}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_truncated_json_is_unavailable():
    result = await _judge(_PASS_JSON[:-6])

    _assert_unavailable(result)


async def test_multiple_json_objects_are_unavailable():
    result = await _judge(f"{_PASS_JSON}\n{_WEAK_JSON}")

    _assert_unavailable(result)


async def test_multiple_fences_are_unavailable():
    result = await _judge(f"```json\n{_PASS_JSON}\n```\n```json\n{_WEAK_JSON}\n```")

    _assert_unavailable(result)


async def test_prose_around_valid_json_is_unavailable():
    result = await _judge(f"Sure! {_MARKER} {_PASS_JSON} hope that helps.")

    _assert_unavailable(result)


async def test_json_string_scalar_is_unavailable():
    body = json.dumps(_PASS_JSON)
    # The scalar case must be a real JSON string value, not broken syntax.
    assert isinstance(json.loads(body), str)
    result = await _judge(body)

    _assert_unavailable(result)


@pytest.mark.parametrize("body", ["[]", "123", "null", "true"])
async def test_json_non_object_scalar_is_unavailable(body):
    assert isinstance(json.loads(body), (list, int, float, type(None), bool))
    result = await _judge(body)

    _assert_unavailable(result)


async def test_reasoning_only_object_is_unavailable():
    result = await _judge(f'{{"reasoning":"{_MARKER} all fluff no scores"}}')

    _assert_unavailable(result)


async def test_empty_list_body_is_unavailable():
    result = await _judge([])

    _assert_unavailable(result)


async def test_non_string_reasoning_is_unavailable():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":123}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_whitespace_only_body_is_unavailable():
    result = await _judge("   \n  ")

    _assert_unavailable(result)


async def test_malformed_content_blocks_are_unavailable_without_echo():
    result = await _judge([{"type": "unknown_vendor_block", "text": _MARKER}])

    _assert_unavailable(result)


# --- negative completion signals block scoring even with a valid body -------


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        {"finish_reason": "content_filter"},
        {"finish_reason": "tool_calls"},
        {"finish_reason": "function_call"},
        {"stop_reason": "max_tokens"},
        {"stop_reason": "tool_use"},
        {"stop_reason": "pause_turn"},
        {"stop_reason": "refusal"},
        {"done_reason": "length"},
        {"done_reason": "load"},
        {"done": False},
        # Gemini adapter projects the FinishReason enum name into
        # response_metadata["finish_reason"] (locked SDK chat_models.py:1172).
        {"finish_reason": "MAX_TOKENS"},
        {"finish_reason": "SAFETY"},
        {"finish_reason": "RECITATION"},
        {"finish_reason": "LANGUAGE"},
        {"finish_reason": "OTHER"},
        {"finish_reason": "BLOCKLIST"},
        {"finish_reason": "PROHIBITED_CONTENT"},
        {"finish_reason": "SPII"},
        {"finish_reason": "MALFORMED_FUNCTION_CALL"},
        {"finish_reason": "UNEXPECTED_TOOL_CALL"},
        {"finish_reason": "IMAGE_SAFETY"},
        {"finish_reason": "IMAGE_PROHIBITED_CONTENT"},
        {"finish_reason": "NO_IMAGE"},
        {"finish_reason": "IMAGE_RECITATION"},
        {"finish_reason": "IMAGE_OTHER"},
        # prompt_feedback.block_reason: only the SDK-defined BlockedReason
        # values (unspecified/0 is covered separately below as non-blocking).
        {"prompt_feedback": {"block_reason": "SAFETY"}},
        {"prompt_feedback": {"block_reason": "OTHER"}},
        {"prompt_feedback": {"block_reason": "BLOCKLIST"}},
        {"prompt_feedback": {"block_reason": "PROHIBITED_CONTENT"}},
        {"prompt_feedback": {"block_reason": "IMAGE_SAFETY"}},
        {"prompt_feedback": {"block_reason": "MODEL_ARMOR"}},
        {"prompt_feedback": {"block_reason": "JAILBREAK"}},
    ],
)
async def test_negative_completion_signal_blocks_scoring(metadata):
    result = await _judge(_PASS_JSON, response_metadata=metadata)

    _assert_unavailable(result, forbidden=())


async def test_explicit_refusal_in_additional_kwargs_blocks_scoring():
    result = await _judge(
        _PASS_JSON, additional_kwargs={"refusal": "cannot help with that"}
    )

    _assert_unavailable(result, forbidden=("cannot help",))


async def test_nonempty_tool_calls_block_scoring():
    result = await _judge(
        _PASS_JSON,
        tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}],
    )

    _assert_unavailable(result, forbidden=())


async def test_invalid_tool_calls_block_scoring():
    result = await _judge(
        _PASS_JSON,
        invalid_tool_calls=[
            {"name": "broken", "args": "{", "id": "c2", "error": "malformed"}
        ],
    )

    _assert_unavailable(result, forbidden=())


# --- unspecified / missing metadata never fabricates completion evidence -----


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": 0},
        {"finish_reason": "FINISH_REASON_UNSPECIFIED"},
        {"finish_reason": "STOP"},
        {"stop_reason": "end_turn"},
        {"done_reason": "stop", "done": True},
        {"prompt_feedback": {"block_reason": 0}},
        {"prompt_feedback": {"block_reason": "BLOCKED_REASON_UNSPECIFIED"}},
        {"finish_reason": "UNKNOWN_99"},
    ],
)
async def test_unspecified_or_unknown_status_is_not_a_block(metadata):
    result = await _judge(_PASS_JSON, response_metadata=metadata)

    _assert_valid(result)
    assert result["verdict"] == "pass"


@pytest.mark.parametrize(
    "bad_metadata",
    [None, "weird", 123, ["not", "a", "dict"]],
)
async def test_synthetic_boundary_metadata_shapes_are_not_treated_as_signals(
    bad_metadata,
):
    """Shape-defensive contract against a synthetic model-boundary object.

    The locked SDK rejects these response_metadata shapes at real AIMessage
    construction (ValidationError), so this fixture is an explicit synthetic
    boundary object carrying the public fields the judge consumes. It proves
    the defensive rule — mismatched metadata is missing evidence, never a
    completion verdict, never a crash — not an SDK-reachable provider shape.
    """
    boundary = SimpleNamespace(
        content=_PASS_JSON,
        response_metadata=bad_metadata,
        tool_calls=[],
        invalid_tool_calls=[],
    )
    result = await judge_response(
        question="q",
        answer="a",
        context="c",
        rubric="rubric-template",
        judge_llm=_StubJudgeLLM(boundary),
        dimensions=None,
    )

    _assert_valid(result)
    assert result["verdict"] == "pass"


async def test_missing_metadata_with_valid_payload_stays_valid():
    result = await _judge(_PASS_JSON)

    _assert_valid(result)
    assert result["verdict"] == "pass"


async def test_missing_metadata_cannot_rescue_an_invalid_payload():
    result = await _judge(f"not json at all {_MARKER}")

    _assert_unavailable(result)


# --- D3 repair RED: explicit null reasoning, RecursionError, Gemini wrapper ---


async def test_explicit_null_reasoning_is_unavailable():
    """A present-but-null `reasoning` is not the same as missing.

    The contract says: missing defaults to empty string and stays valid; an
    explicit null is a malformed payload and must be unavailable.
    """
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":null}'
    )
    result = await _judge(payload)

    _assert_unavailable(result)


async def test_explicit_null_reasoning_does_not_become_pass():
    """Specifically guards against the current conflation bug.

    With the conflated semantics the unit returns
    `{"reasoning":"","score":1.0,"status":"valid","verdict":"pass"}`,
    so we assert the converse: status is unavailable, score is None,
    verdict is None, and reasoning never appears in the result.
    """
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":null}'
    )
    result = await _judge(payload)

    assert result["status"] == "unavailable"
    assert result.get("score") is None
    assert result.get("verdict") is None
    assert "reasoning" not in result


async def test_missing_reasoning_still_default_to_empty_string():
    """Missing key is the only path that yields an empty-string reasoning."""
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5}'
    )
    result = await _judge(payload)

    _assert_valid(result)
    assert result["reasoning"] == ""


async def test_nested_model_json_does_not_abort_on_recursion():
    """A finite but deeply-nested model body must not raise RecursionError.

    The fix lives in `_parse_strict_object` itself: the boundary catch is
    around the single `json.loads` call, and the RecursionError is mapped to
    the internal invalid-payload marker so the public result stays unavailable.
    """
    prefix = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"ok","ignored":'
    )
    # ~3000 chars total: a finite object with a 1500-deep nested array.
    body = prefix + "[" * 1500 + "0" + "]" * 1500 + "}"
    # The fixture is large enough to trigger the project's Python recursion
    # ceiling, but finite enough not to require lowering sys.setrecursionlimit.
    assert len(body) < 5000

    result = await _judge(body)

    _assert_unavailable(result)


async def test_locked_gemini_wrapped_client_error_becomes_unavailable():
    """The locked SDK wraps google.genai.errors.ClientError in
    ChatGoogleGenerativeAIError; the production judge must classify that
    wrapper as a model-service error without reading the raw detail, and the
    public reason must not echo the synthetic error marker.

    The SDK wrapping helper (`_handle_client_error`) is used unchanged — only
    the judge-side class-name matching is fixed. No SDK module is imported
    inside production_judge.py; this test exercises the wrapper through the
    real helper to prove the type identity, not the production code.
    """
    from langchain_google_genai.chat_models import (
        ChatGoogleGenerativeAIError,
        _handle_client_error,
    )
    from google.genai.errors import ClientError

    raw = ClientError(
        400,
        {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                   "message": "D3-SYNTHETIC-GEMINI-ERROR"}},
    )
    try:
        _handle_client_error(raw, {"model": "synthetic-gemini-model"})
    except ChatGoogleGenerativeAIError as wrapped:
        exception = wrapped

    assert type(exception.__cause__).__name__ == "ClientError"

    class BrokenJudge:
        async def ainvoke(self, _messages, **_kwargs):
            raise exception

    result = await judge_response(
        question="q",
        answer="a",
        context="c",
        rubric="rubric-template",
        judge_llm=BrokenJudge(),
    )

    _assert_unavailable(result, forbidden=("D3-SYNTHETIC-GEMINI-ERROR",))


async def test_locked_gemini_wrapper_is_matched_by_name_only():
    """The fix is a single class-name entry in the approved list.

    We exercise the matcher directly against the locked SDK's wrapper class —
    real type identity via the wrapper helper, no SDK import inside the
    production judge. A class with the same name not in the SDK must not be
    matched, so the catch stays narrow.
    """
    from langchain_google_genai.chat_models import (
        ChatGoogleGenerativeAIError,
        _handle_client_error,
    )
    from google.genai.errors import ClientError

    from app.agent.production_judge import is_model_service_error

    raw = ClientError(
        400,
        {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                   "message": "x"}},
    )
    try:
        _handle_client_error(raw, {"model": "synthetic-gemini-model"})
    except ChatGoogleGenerativeAIError as wrapped:
        exception = wrapped

    assert is_model_service_error(exception) is True

    class NotAServiceError(Exception):
        pass

    assert is_model_service_error(NotAServiceError("x")) is False
    assert is_model_service_error(RuntimeError("x")) is False
    assert is_model_service_error(ValueError("x")) is False
