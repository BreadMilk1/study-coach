import json

import pytest
from langchain_core.messages import AIMessage

from app.agent.judge import QUIZ_DIMENSIONS, judge_response, load_quiz_rubric
from app.eval.p2_3_cloud_capability.budget import (
    BudgetExceeded,
    BudgetLedgerCorrupt,
    MissingUsage,
)
from app.eval.p2_3_cloud_capability.protocol import SmokeAbort
from app.eval.p2_3_cloud_capability.scoring import (
    RecordingJudgeLLM,
    score_candidate,
    validate_quiz_judge_payload,
)


def test_empty_object_is_invalid_even_though_json_parses():
    assert validate_quiz_judge_payload("{}") is None


def test_missing_one_dimension_is_invalid():
    payload = {
        "question_quality": 5,
        "option_plausibility": 5,
        "answer_correctness": 5,
        "explanation_clarity": 5,
        "reasoning": "ok",
    }
    assert validate_quiz_judge_payload(payload) is None


def test_non_numeric_dimension_is_invalid():
    payload = {d: 4 for d in QUIZ_DIMENSIONS}
    payload["question_quality"] = "good"
    payload["reasoning"] = "ok"
    assert validate_quiz_judge_payload(payload) is None


def test_all_threes_is_valid_true_score():
    payload = {d: 3 for d in QUIZ_DIMENSIONS}
    payload["reasoning"] = "mid"
    out = validate_quiz_judge_payload(payload)
    assert out is not None
    assert out["score"] == pytest.approx(0.6)


def test_dimension_out_of_range_is_invalid():
    payload = {d: 4 for d in QUIZ_DIMENSIONS}
    payload["question_quality"] = 6
    payload["reasoning"] = "ok"
    assert validate_quiz_judge_payload(payload) is None
    payload["question_quality"] = 0
    assert validate_quiz_judge_payload(payload) is None


def test_nan_dimension_is_invalid_in_mapping_and_json():
    payload = {d: 4 for d in QUIZ_DIMENSIONS}
    payload["question_quality"] = float("nan")
    payload["reasoning"] = "ok"
    assert validate_quiz_judge_payload(payload) is None
    as_json = json.dumps({d: (float("nan") if d == "question_quality" else 4) for d in QUIZ_DIMENSIONS} | {"reasoning": "ok"})
    assert "NaN" in as_json
    assert validate_quiz_judge_payload(as_json) is None


def test_infinite_dimension_is_invalid():
    for value in (float("inf"), float("-inf")):
        payload = {d: 4 for d in QUIZ_DIMENSIONS}
        payload["difficulty_calibration"] = value
        payload["reasoning"] = "ok"
        assert validate_quiz_judge_payload(payload) is None
        as_json = json.dumps(
            {d: (value if d == "difficulty_calibration" else 4) for d in QUIZ_DIMENSIONS}
            | {"reasoning": "ok"}
        )
        assert validate_quiz_judge_payload(as_json) is None


@pytest.mark.asyncio
async def test_recording_wrapper_rejects_empty_json_despite_judge_response_point_six():
    class Inner:
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(content="{}")

    wrapper = RecordingJudgeLLM(Inner())
    result = await judge_response(
        question="q",
        answer="a",
        context="",
        rubric=load_quiz_rubric(),
        judge_llm=wrapper,
        dimensions=QUIZ_DIMENSIONS,
    )
    assert result["score"] == pytest.approx(0.6)
    assert "Judge output parsing failed" not in result["reasoning"]
    independent = validate_quiz_judge_payload(wrapper.last_text)
    assert independent is None


@pytest.mark.asyncio
async def test_missing_dimension_raw_is_not_adopted_as_high_score():
    raw = (
        '{"question_quality":5,"option_plausibility":5,'
        '"answer_correctness":5,"explanation_clarity":5,"reasoning":"x"}'
    )

    class Inner:
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(content=raw)

    wrapper = RecordingJudgeLLM(Inner())
    result = await judge_response(
        question="q", answer="a", context="",
        rubric=load_quiz_rubric(), judge_llm=wrapper, dimensions=QUIZ_DIMENSIONS,
    )
    assert result["score"] == pytest.approx(0.92)
    assert validate_quiz_judge_payload(wrapper.last_text) is None


@pytest.mark.asyncio
async def test_judge_exception_is_failure_not_crash():
    class AuthenticationError(Exception):
        pass

    class Boom:
        model = "deepseek-v4-pro"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            raise AuthenticationError("Error code: 401")

    class Ok:
        model = "MiniMax-M2.7"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            from langchain_core.messages import AIMessage

            payload = {d: 4 for d in QUIZ_DIMENSIONS}
            payload["reasoning"] = "ok"
            return AIMessage(content=json.dumps(payload))

    execs = await score_candidate(
        candidate={
            "quiz_action": "generate",
            "final_text": "📝 Quiz on HyDE:\nQ\nA) a\nB) b\nC) c\nD) d",
            "question": {
                "prompt": "Q",
                "options": ["A) a", "B) b", "C) c", "D) d"],
                "answer": "A",
                "explanation": "e",
            },
            "question_persisted": True,
            "retrieval_captures": [{"capture_status": "ok", "evidence": []}],
        },
        judges={"qwen": Boom(), "deepseek": Boom(), "m27": Ok()},
    )
    by_id = {e["scorer_id"]: e for e in execs}
    assert by_id["legacy-visible-v1/qwen2.5:7b"]["status"] == "failed"
    assert by_id["legacy-visible-v1/qwen2.5:7b"]["failure_class"] == "transport"
    assert by_id["quiz-artifact-v1/deepseek-v4-pro"]["status"] == "failed"
    assert by_id["quiz-artifact-v1/deepseek-v4-pro"]["failure_class"] == "transport"
    assert by_id["quiz-artifact-v1/MiniMax-M2.7"]["status"] == "success"


@pytest.mark.asyncio
async def test_grade_rows_are_skipped_for_quality():
    execs = await score_candidate(
        candidate={"quiz_action": "grade", "final_text": "✓ Correct"},
        judges={"qwen": lambda *_: {"score": 1}, "deepseek": None, "m27": None},
    )
    assert execs
    assert all(e["status"] == "skipped" for e in execs if "quality" in e.get("role", "quality"))


def _valid_rubric():
    payload = {d: 4 for d in QUIZ_DIMENSIONS}
    payload["reasoning"] = "ok"
    return payload


@pytest.mark.asyncio
async def test_happy_path_emits_legacy_and_three_artifact_judges():
    from langchain_core.messages import AIMessage

    valid = json.dumps(_valid_rubric())

    class Judge:
        def __init__(self, model, protocol):
            self.model = model
            self.protocol = protocol
            self.thinking_config = {"type": "disabled"}

        async def ainvoke(self, messages, **kwargs):
            return AIMessage(
                content=valid,
                usage_metadata={"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
                response_metadata={"stop_reason": "end_turn", "finish_reason": "stop"},
            )

    execs = await score_candidate(
        candidate={
            "quiz_action": "generate",
            "final_text": "📝 Quiz on HyDE:\nWhat?\nA) a\nB) b\nC) c\nD) d\nReply with A",
            "question": {
                "prompt": "What?",
                "options": ["A) a", "B) b", "C) c", "D) d"],
                "answer": "A",
                "explanation": "because",
            },
            "question_persisted": True,
            "retrieval_captures": [{"capture_status": "ok", "evidence": []}],
        },
        judges={
            "qwen": Judge("qwen2.5:7b", "anthropic"),
            "deepseek": Judge("deepseek-v4-pro", "openai_compatible"),
            "m27": Judge("MiniMax-M2.7", "openai_compatible"),
        },
    )
    legacy = [e for e in execs if e["scorer_id"] == "legacy-visible-v1/qwen2.5:7b"]
    artifact = [e for e in execs if e["scorer_id"].startswith("quiz-artifact-v1")]
    assert len(legacy) == 1 and legacy[0]["status"] == "success"
    assert len(artifact) == 3
    assert {e["scorer_id"] for e in artifact} == {
        "quiz-artifact-v1/qwen2.5:7b",
        "quiz-artifact-v1/deepseek-v4-pro",
        "quiz-artifact-v1/MiniMax-M2.7",
    }
    assert all(e["status"] == "success" for e in artifact)
    for e in artifact + legacy:
        assert e.get("output") is not None
        assert e.get("raw_response_sha256")
        assert e.get("blocks")
        assert e.get("finish_status") == "completed"
        assert e.get("usage")
        assert e.get("model")
        assert e.get("protocol")
        assert e.get("thinking") is not None
        assert e.get("rubric_hash")


def test_m27_strips_think_then_validates():
    from app.eval.p2_3_cloud_capability.scoring import prepare_judge_content

    valid = json.dumps(_valid_rubric())
    wrapped = f"<think>internal chain</think>\n\n{valid}"
    text, meta = prepare_judge_content(wrapped, allow_strip_think=True)
    assert meta["think_stripped"] is True
    assert validate_quiz_judge_payload(text) is not None


def test_m27_probe_think_wrapper_is_stripped():
    from app.eval.p2_3_cloud_capability.scoring import prepare_judge_content

    probe = (
        "<think>\nThe user asks: \"Reply with JSON object {}\". "
        "Thus final: {}.\n</think>\n\n{}"
    )
    text, meta = prepare_judge_content(probe, allow_strip_think=True)
    assert meta["think_stripped"] is True
    # empty {} remains fail-closed
    assert validate_quiz_judge_payload(text) is None
    valid = json.dumps(_valid_rubric())
    mixed = probe.rsplit("{}", 1)[0] + valid
    text2, meta2 = prepare_judge_content(mixed, allow_strip_think=True)
    assert meta2["think_stripped"] is True
    assert validate_quiz_judge_payload(text2) is not None


@pytest.mark.asyncio
async def test_missing_question_does_not_fallback_to_legacy_payload_for_artifact():
    execs = await score_candidate(
        candidate={
            "quiz_action": "generate",
            "final_text": "📝 Quiz on HyDE:\nQ\nA) ...",
            "question": None,
            "question_persisted": False,
            "retrieval_captures": [{"capture_status": "ok", "evidence": []}],
        },
        judges={"qwen": None, "deepseek": lambda p: (_ for _ in ()).throw(AssertionError("should not be called")), "m27": None},
    )
    art = [e for e in execs if e["scorer_id"].startswith("quiz-artifact-v1")]
    assert art and all(e["status"] == "failed" and e["error_code"] == "structured_artifact_failure" for e in art)
    legacy = [e for e in execs if e["scorer_id"] == "legacy-visible-v1/qwen2.5:7b"]
    assert len(legacy) == 1


@pytest.mark.asyncio
async def test_unpersisted_generate_scores_visible_payload_on_legacy_line():
    seen: list[str] = []
    valid = json.dumps(_valid_rubric())

    class Judge:
        model = "qwen2.5:7b"
        protocol = "anthropic"
        thinking_config = {"type": "disabled"}

        async def ainvoke(self, messages, **kwargs):
            seen.append(messages[0].content)
            return AIMessage(
                content=valid,
                usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                response_metadata={"stop_reason": "end_turn"},
            )

    visible = "📝 Quiz on HyDE:\nWhat is HyDE?\nA) a\nB) b\nC) c\nD) d"
    execs = await score_candidate(
        candidate={
            "quiz_action": "generate",
            "final_text": visible,
            "question": None,
            "question_persisted": False,
            "retrieval_captures": [{"capture_status": "ok", "evidence": []}],
        },
        judges={
            "qwen": Judge(),
            "deepseek": lambda *_: (_ for _ in ()).throw(AssertionError("artifact judge")),
            "m27": lambda *_: (_ for _ in ()).throw(AssertionError("artifact judge")),
        },
    )
    legacy = [e for e in execs if e["scorer_id"] == "legacy-visible-v1/qwen2.5:7b"]
    art = [e for e in execs if str(e.get("scorer_id", "")).startswith("quiz-artifact-v1")]
    assert len(legacy) == 1
    assert legacy[0]["status"] == "success"
    assert seen and visible in seen[0]
    assert "Quiz output to evaluate" in seen[0]
    assert "question_quality" in seen[0]
    assert len(art) == 3
    assert all(e["status"] == "failed" and e["error_code"] == "structured_artifact_failure" for e in art)
    assert legacy[0].get("rubric_hash")
    assert all(e.get("rubric_hash") in (None, "") for e in art)


@pytest.mark.asyncio
async def test_score_prompts_use_rubric_and_retrieval_context():
    seen: dict[str, list[str]] = {}
    valid = json.dumps(_valid_rubric())
    evidence = "HyDE stores last_aux_tokens so ExperimentRunner can see true cost."

    class Judge:
        def __init__(self, key: str):
            self.key = key
            self.model = key
            self.protocol = "openai_compatible"
            self.thinking_config = {"type": "disabled"}

        async def ainvoke(self, messages, **kwargs):
            seen.setdefault(self.key, []).append(messages[0].content)
            return AIMessage(content=valid)

    execs = await score_candidate(
        candidate={
            "quiz_action": "generate",
            "query_id": "quiz_hyde",
            "final_text": "📝 Quiz on HyDE:\nWhat is HyDE?\nA) a\nB) b",
            "question": {
                "prompt": "What is HyDE?",
                "options": ["A) a", "B) b", "C) c", "D) d"],
                "answer": "B",
                "explanation": "because retrieval",
            },
            "question_persisted": True,
            "retrieval_captures": [{
                "capture_status": "ok",
                "evidence": [{"source": "notes.pdf", "page": 2, "content": evidence}],
            }],
        },
        judges={
            "qwen": Judge("qwen"),
            "deepseek": Judge("deepseek"),
            "m27": Judge("m27"),
        },
    )
    legacy = seen["qwen"][0]
    artifact = seen["deepseek"][0]
    assert "Quiz output to evaluate" in legacy
    assert "Quiz output to evaluate" in artifact
    assert "What is HyDE?" in artifact
    assert "because retrieval" in artifact
    assert "📝 Quiz on HyDE" in legacy
    assert evidence in legacy and evidence in artifact
    assert "notes.pdf" in artifact
    assert "You are an impartial Study Coach judge" in legacy
    by_id = {e["scorer_id"]: e for e in execs}
    assert by_id["legacy-visible-v1/qwen2.5:7b"].get("rubric_hash")
    assert by_id["quiz-artifact-v1/deepseek-v4-pro"].get("rubric_hash")


@pytest.mark.asyncio
async def test_rubric_hash_only_when_rubric_applied():
    execs = await score_candidate(
        candidate={"quiz_action": "grade", "final_text": "✓"},
        judges={"qwen": None, "deepseek": None, "m27": None},
    )
    assert all(not e.get("rubric_hash") for e in execs)

    execs = await score_candidate(
        candidate={
            "quiz_action": "generate",
            "final_text": "visible",
            "question": None,
            "question_persisted": False,
            "retrieval_captures": [{"capture_status": "ok", "evidence": []}],
        },
        judges={"qwen": None, "deepseek": None, "m27": None},
    )
    legacy = [e for e in execs if e["scorer_id"].startswith("legacy-visible")][0]
    art = [e for e in execs if str(e.get("scorer_id", "")).startswith("quiz-artifact")]
    assert legacy["error_code"] == "missing_judge"
    assert not legacy.get("rubric_hash")
    assert all(e["error_code"] == "structured_artifact_failure" and not e.get("rubric_hash") for e in art)


# ---------------------------------------------------------------------------
# Coverage added when the module moved into the release worktree: abort
# propagation and scorer_ids selection semantics. The tests above are the
# unchanged appendix tests for this file.
# ---------------------------------------------------------------------------

_CANONICAL_SCORER_ORDER = [
    "legacy-visible-v1/qwen2.5:7b",
    "quiz-artifact-v1/qwen2.5:7b",
    "quiz-artifact-v1/deepseek-v4-pro",
    "quiz-artifact-v1/MiniMax-M2.7",
]

_PERSISTED_CANDIDATE = {
    "quiz_action": "generate",
    "final_text": "📝 Quiz on HyDE:\nQ\nA) a\nB) b",
    "question": {
        "prompt": "Q",
        "options": ["A) a", "B) b", "C) c", "D) d"],
        "answer": "A",
        "explanation": "because",
    },
    "question_persisted": True,
    "retrieval_captures": [{"capture_status": "ok", "evidence": []}],
}


def _recording_judges(calls):
    class Judge:
        def __init__(self, key):
            self.key = key
            self.model = key
            self.protocol = "openai_compatible"
            self.thinking_config = {"type": "disabled"}

        async def ainvoke(self, messages, **kwargs):
            calls.append(self.key)
            return AIMessage(content=json.dumps(_valid_rubric()))

    return {"qwen": Judge("qwen"), "deepseek": Judge("deepseek"), "m27": Judge("m27")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        SmokeAbort("smoke gate failed"),
        BudgetExceeded("stage cap exceeded"),
        MissingUsage("usage missing"),
        BudgetLedgerCorrupt("ledger corrupt"),
    ],
    ids=["SmokeAbort", "BudgetExceeded", "MissingUsage", "BudgetLedgerCorrupt"],
)
async def test_abort_exceptions_propagate_unchanged_and_stop_later_judges(error):
    calls: list[str] = []

    class AbortingJudge:
        model = "qwen2.5:7b"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append("qwen")
            raise error

    class LaterJudge:
        model = "later"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append("later")
            return AIMessage(content=json.dumps(_valid_rubric()))

    with pytest.raises(type(error)) as excinfo:
        await score_candidate(
            candidate=dict(_PERSISTED_CANDIDATE),
            judges={
                "qwen": AbortingJudge(),
                "deepseek": LaterJudge(),
                "m27": LaterJudge(),
            },
        )
    assert excinfo.value is error
    assert calls == ["qwen"]


@pytest.mark.asyncio
async def test_abort_from_a_later_judge_stops_the_remaining_one():
    calls: list[str] = []
    error = BudgetExceeded("mid-run cap")

    class OkJudge:
        model = "qwen2.5:7b"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append("qwen")
            return AIMessage(content=json.dumps(_valid_rubric()))

    class AbortingJudge:
        model = "deepseek-v4-pro"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append("deepseek")
            raise error

    class ThirdJudge:
        model = "MiniMax-M2.7"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append("m27")
            return AIMessage(content=json.dumps(_valid_rubric()))

    with pytest.raises(BudgetExceeded) as excinfo:
        await score_candidate(
            candidate=dict(_PERSISTED_CANDIDATE),
            judges={
                "qwen": OkJudge(),
                "deepseek": AbortingJudge(),
                "m27": ThirdJudge(),
            },
        )
    assert excinfo.value is error
    # canonical order calls the qwen judge twice (legacy row, then artifact qwen row);
    # the aborting deepseek judge stops the m27 judge from ever running
    assert calls == ["qwen", "qwen", "deepseek"]


@pytest.mark.asyncio
async def test_ordinary_judge_error_is_a_failed_row_not_an_abort():
    class Boom:
        model = "qwen2.5:7b"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            raise ValueError("judge blew up")

    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges={"qwen": Boom(), "deepseek": None, "m27": None},
    )
    legacy = [e for e in execs if e["scorer_id"] == "legacy-visible-v1/qwen2.5:7b"]
    assert len(legacy) == 1
    assert legacy[0]["status"] == "failed"
    assert legacy[0]["error_code"] == "ValueError"
    assert legacy[0]["failure_class"] == "model"


@pytest.mark.asyncio
async def test_scorer_ids_none_selects_the_full_canonical_set():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE), judges=_recording_judges(calls)
    )
    assert [e["scorer_id"] for e in execs] == _CANONICAL_SCORER_ORDER
    # four rows: the legacy visible row and the three artifact rows; the qwen
    # judge instance serves both the legacy row and the first artifact row
    assert calls == ["qwen", "qwen", "deepseek", "m27"]
    assert all(e["status"] == "success" for e in execs)


@pytest.mark.asyncio
async def test_scorer_ids_subset_runs_only_the_selected_judge():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges=_recording_judges(calls),
        scorer_ids=["quiz-artifact-v1/deepseek-v4-pro"],
    )
    assert [e["scorer_id"] for e in execs] == ["quiz-artifact-v1/deepseek-v4-pro"]
    assert calls == ["deepseek"]


@pytest.mark.asyncio
async def test_scorer_ids_input_order_does_not_change_result_order():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges=_recording_judges(calls),
        scorer_ids=[
            "quiz-artifact-v1/MiniMax-M2.7",
            "legacy-visible-v1/qwen2.5:7b",
            "quiz-artifact-v1/deepseek-v4-pro",
        ],
    )
    assert [e["scorer_id"] for e in execs] == [
        "legacy-visible-v1/qwen2.5:7b",
        "quiz-artifact-v1/deepseek-v4-pro",
        "quiz-artifact-v1/MiniMax-M2.7",
    ]
    assert calls == ["qwen", "deepseek", "m27"]


@pytest.mark.asyncio
async def test_scorer_ids_duplicates_collapse_to_one_row_and_one_call():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges=_recording_judges(calls),
        scorer_ids=["quiz-artifact-v1/MiniMax-M2.7"] * 3,
    )
    assert [e["scorer_id"] for e in execs] == ["quiz-artifact-v1/MiniMax-M2.7"]
    assert calls == ["m27"]


@pytest.mark.asyncio
async def test_empty_scorer_ids_selects_nothing_and_calls_no_judge():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges=_recording_judges(calls),
        scorer_ids=[],
    )
    assert execs == []
    assert calls == []


@pytest.mark.asyncio
async def test_unknown_scorer_ids_are_silently_ignored():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges=_recording_judges(calls),
        scorer_ids=["quiz-artifact-v1/does-not-exist"],
    )
    assert execs == []
    assert calls == []


@pytest.mark.asyncio
async def test_known_and_unknown_scorer_ids_mix_keeps_known_in_canonical_order():
    calls: list[str] = []
    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges=_recording_judges(calls),
        scorer_ids=[
            "quiz-artifact-v1/nope",
            "quiz-artifact-v1/qwen2.5:7b",
            "legacy-visible-v1/qwen2.5:7b",
            "unknown/other",
        ],
    )
    assert [e["scorer_id"] for e in execs] == [
        "legacy-visible-v1/qwen2.5:7b",
        "quiz-artifact-v1/qwen2.5:7b",
    ]
    assert calls == ["qwen", "qwen"]


# ---------------------------------------------------------------------------
# Boundary coverage added after independent review: an exact integer dimension
# far outside float range must be refused as invalid, not raise OverflowError.
# ---------------------------------------------------------------------------

_HUGE_INT_VALUES = [10**400, -(10**400)]
_HUGE_INT_IDS = ["positive", "negative"]


@pytest.mark.parametrize("value", _HUGE_INT_VALUES, ids=_HUGE_INT_IDS)
def test_huge_int_dimension_is_invalid_in_mapping(value):
    payload = _valid_rubric()
    payload["question_quality"] = value
    assert validate_quiz_judge_payload(payload) is None


@pytest.mark.parametrize("value", _HUGE_INT_VALUES, ids=_HUGE_INT_IDS)
def test_huge_int_dimension_is_invalid_in_json(value):
    payload = {
        d: (value if d == "question_quality" else 4) for d in QUIZ_DIMENSIONS
    } | {"reasoning": "ok"}
    as_json = json.dumps(payload)
    # json.dumps keeps the exact integer literal, so json.loads returns an int again
    assert str(value) in as_json
    assert validate_quiz_judge_payload(as_json) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", _HUGE_INT_VALUES, ids=_HUGE_INT_IDS)
async def test_huge_int_judge_payload_is_a_parse_failure_and_later_judges_continue(value):
    calls: list[str] = []
    huge = json.dumps(
        {d: (value if d == "question_quality" else 4) for d in QUIZ_DIMENSIONS}
        | {"reasoning": "ok"}
    )

    class HugeJudge:
        model = "qwen2.5:7b"
        protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append("qwen")
            return AIMessage(content=huge)

    class OkJudge:
        def __init__(self, key):
            self.model = key
            self.protocol = "openai_compatible"

        async def ainvoke(self, messages, **kwargs):
            calls.append(self.model)
            return AIMessage(content=json.dumps(_valid_rubric()))

    execs = await score_candidate(
        candidate=dict(_PERSISTED_CANDIDATE),
        judges={
            "qwen": HugeJudge(),
            "deepseek": OkJudge("deepseek"),
            "m27": OkJudge("m27"),
        },
    )
    # canonical order is kept and the later selected judges still run: an
    # out-of-range payload is a failed row, unlike the four abort types which
    # propagate immediately and stop the remaining judges
    assert [e["scorer_id"] for e in execs] == _CANONICAL_SCORER_ORDER
    assert calls == ["qwen", "qwen", "deepseek", "m27"]
    legacy, artifact_qwen, artifact_deepseek, artifact_m27 = execs
    assert legacy["status"] == "failed"
    assert legacy["error_code"] == "parse"
    assert legacy["failure_class"] == "model"
    assert artifact_qwen["status"] == "failed"
    assert artifact_qwen["error_code"] == "parse"
    assert artifact_qwen["failure_class"] == "model"
    assert artifact_deepseek["status"] == "success"
    assert artifact_m27["status"] == "success"
