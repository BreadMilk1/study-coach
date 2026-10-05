"""Judge utility unit tests (P2.1-② RED-1, revised for 6-dimension rubric).

`judge_response` evaluates a Tutor answer against retrieved sources using a
6-criteria LLM-as-judge rubric:

Grounding side (3):
- relevance / accuracy / citation_quality

Pedagogical side (3, split from prior single pedagogical_fit per user feedback):
- accessibility       — high score = plain language, jargon explained
- example_quality     — high score = concrete examples that aid understanding
- learner_level_fit   — high score = pitched at exam-prep student level

Splitting pedagogical_fit avoids averaging-away problems (e.g., good examples
masking heavy unexplained jargon). All dimensions weighted equally.
"""

import pytest
from langchain_core.messages import AIMessage

from app.agent.judge import (
    PLAN_DIMENSIONS,
    QUIZ_DIMENSIONS,
    judge_response,
    load_plan_rubric,
    load_quiz_rubric,
    load_tutor_rubric,
)

# Batch C publishes these as fixed, payload-free rejection text. The literals
# are duplicated on purpose: a consumer bug must stay observable even when the
# shared helper module is absent.
_MALFORMED_CONTENT_MESSAGE = "LLM response content is not a supported text shape"
_EMPTY_CONTENT_MESSAGE = "LLM response text has no non-whitespace body"
_SENTINEL = "SENTINEL-DO-NOT-LEAK"


class StubJudgeLLM:
    def __init__(self, json_payload: str):
        self.payload = json_payload
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.payload)


_PED_DIMS = ("accessibility", "example_quality", "learner_level_fit")
_ALL_DIMS = ("relevance", "accuracy", "citation_quality") + _PED_DIMS


@pytest.mark.asyncio
async def test_judge_returns_normalised_score_from_six_dimensions():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"Well grounded and well paced."}'
    )
    judge_llm = StubJudgeLLM(payload)

    result = await judge_response(
        question="What is HyDE?",
        answer="HyDE is query rewriting per [1].",
        context="[1] a.pdf p.1: HyDE rewrites queries.",
        rubric=load_tutor_rubric(),
        judge_llm=judge_llm,
    )

    # (5+5+4+4+5+5)/6/5 = 28/6/5 = 0.9333
    assert result["score"] == pytest.approx(28 / 6 / 5, abs=1e-3)
    assert result["verdict"] == "pass"
    assert result["weak_dims"] == []
    assert "well grounded" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_judge_marks_low_dimensions_as_weak_with_fine_grained_keys():
    # The user's worry: a Tutor answer with good examples (5) but heavy jargon (2)
    # under a single pedagogical_fit would average to 3.5 and hide the jargon issue.
    # With split pedagogical dims, accessibility=2 surfaces directly in weak_dims.
    payload = (
        '{"relevance":4,"accuracy":4,"citation_quality":3,'
        '"accessibility":2,"example_quality":5,"learner_level_fit":3,'
        '"reasoning":"Decent grounding but heavy jargon."}'
    )
    judge_llm = StubJudgeLLM(payload)

    result = await judge_response(
        question="Explain BM25",
        answer="BM25 is an Okapi-family probabilistic ranking with IDF-weighted TF saturation...",
        context="[1] x.pdf p.3: BM25 is a lexical retrieval method.",
        rubric=load_tutor_rubric(),
        judge_llm=judge_llm,
    )

    # (4+4+3+2+5+3)/6/5 = 21/6/5 = 0.7
    assert result["score"] == pytest.approx(0.7, abs=1e-3)
    assert result["verdict"] == "pass"  # 0.7 ≥ 0.6 threshold
    # Fine-grained weak_dims: only dims ≤ 3 surface; accessibility (2) front-and-centre
    assert set(result["weak_dims"]) == {"citation_quality", "accessibility", "learner_level_fit"}


@pytest.mark.asyncio
async def test_judge_prompt_includes_question_answer_context_and_all_six_dimensions():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":5,'
        '"accessibility":5,"example_quality":5,"learner_level_fit":5,"reasoning":"ok"}'
    )
    judge_llm = StubJudgeLLM(payload)

    await judge_response(
        question="What is HyDE?",
        answer="HyDE generates a hypothetical answer first.",
        context="[1] Topic7.pdf p.2: HyDE rewrites queries.",
        rubric=load_tutor_rubric(),
        judge_llm=judge_llm,
    )

    p = judge_llm.last_prompt
    assert p is not None
    assert "What is HyDE?" in p
    assert "HyDE generates a hypothetical answer first." in p
    assert "Topic7.pdf p.2" in p
    # All 6 dimensions must be present in the injected rubric
    for dim in _ALL_DIMS:
        assert dim in p, f"rubric missing dimension '{dim}' in prompt"


@pytest.mark.asyncio
async def test_judge_prompt_includes_bias_aware_instruction():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":5,'
        '"accessibility":5,"example_quality":5,"learner_level_fit":5,"reasoning":"ok"}'
    )
    judge_llm = StubJudgeLLM(payload)

    await judge_response(
        question="q", answer="a", context="ctx",
        rubric=load_tutor_rubric(), judge_llm=judge_llm,
    )

    p = judge_llm.last_prompt.lower()
    assert "critically" in p or "harsh" in p or "self-preference" in p, (
        "bias-aware instruction missing from judge prompt"
    )


@pytest.mark.asyncio
async def test_judge_parses_markdown_fenced_json():
    payload = (
        "```json\n"
        '{"relevance":3,"accuracy":4,"citation_quality":3,'
        '"accessibility":4,"example_quality":3,"learner_level_fit":4,'
        '"reasoning":"Mixed."}\n'
        "```"
    )
    judge_llm = StubJudgeLLM(payload)

    result = await judge_response(
        question="q", answer="a", context="ctx",
        rubric=load_tutor_rubric(), judge_llm=judge_llm,
    )

    # (3+4+3+4+3+4)/6/5 = 21/6/5 = 0.7
    assert result["score"] == pytest.approx(0.7, abs=1e-3)
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_judge_falls_back_to_neutral_score_when_json_unparseable():
    judge_llm = StubJudgeLLM("This is not JSON at all.")

    result = await judge_response(
        question="q", answer="a", context="ctx",
        rubric=load_tutor_rubric(), judge_llm=judge_llm,
    )

    # Neutral fallback: 3/5 = 0.6 (right at the threshold → still pass)
    assert result["score"] == pytest.approx(0.6, abs=1e-3)
    assert result["verdict"] == "pass"
    assert result["weak_dims"] == []
    assert "parsing" in result["reasoning"].lower() or "fallback" in result["reasoning"].lower()


# ---------------------------------------------------------------------------
# Batch C — content-block text boundary (strict extraction before JSON parse).
# ---------------------------------------------------------------------------


class ContentJudgeLLM:
    """Returns an arbitrary message-content shape for the boundary tests."""

    def __init__(self, content):
        self.content = content
        self.last_prompt: str | None = None

    async def ainvoke(self, messages, **_kwargs):
        self.last_prompt = messages[-1].content if messages else ""
        return AIMessage(content=self.content)


async def _judge(content, *, dimensions=None, rubric=None):
    return await judge_response(
        question="q",
        answer="a",
        context="ctx",
        rubric=rubric or load_tutor_rubric(),
        judge_llm=ContentJudgeLLM(content),
        dimensions=dimensions,
    )


@pytest.mark.asyncio
async def test_judge_extracts_text_blocks_split_across_multiple_blocks():
    payload = (
        '{"relevance":5,"accuracy":5,"citation_quality":4,'
        '"accessibility":4,"example_quality":5,"learner_level_fit":5,'
        '"reasoning":"Well grounded and well paced."}'
    )
    content = [
        {"type": "thinking", "thinking": _SENTINEL},
        {"type": "text", "text": payload[:30]},
        payload[30:70],
        {"type": "text", "text": payload[70:]},
    ]

    result = await _judge(content)

    assert result["score"] == pytest.approx(28 / 6 / 5, abs=1e-3)
    assert result["verdict"] == "pass"
    assert "well grounded" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_judge_ignores_text_like_fields_on_skipped_blocks():
    content = [
        {"type": "reasoning", "text": '{"relevance":1}'},
        {"type": "tool_result", "text": '{"relevance":1}'},
        {"type": "text", "text": '{"relevance":5,"accuracy":5,"citation_quality":5,'
                                 '"accessibility":5,"example_quality":5,"learner_level_fit":5,'
                                 '"reasoning":"clean"}'},
    ]

    result = await _judge(content)

    assert result["score"] == pytest.approx(1.0, abs=1e-3)
    assert result["weak_dims"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dimensions", "rubric_loader", "payload"),
    [
        (
            None,
            load_tutor_rubric,
            '{"relevance":5,"accuracy":5,"citation_quality":5,'
            '"accessibility":5,"example_quality":5,"learner_level_fit":5}',
        ),
        (
            QUIZ_DIMENSIONS,
            load_quiz_rubric,
            '{"question_quality":4,"option_plausibility":4,"answer_correctness":4,'
            '"explanation_clarity":4,"difficulty_calibration":4}',
        ),
        (
            PLAN_DIMENSIONS,
            load_plan_rubric,
            '{"milestone_specificity":4,"milestone_granularity":4,"time_feasibility":4,'
            '"topic_coverage":4,"actionability":4}',
        ),
    ],
)
async def test_judge_keeps_dimension_control_for_text_block_content(
    dimensions, rubric_loader, payload
):
    result = await _judge(
        [{"type": "text", "text": payload}],
        dimensions=dimensions,
        rubric=rubric_loader(),
    )

    assert result["verdict"] == "pass"
    assert result["score"] == pytest.approx(0.8 if dimensions else 1.0, abs=1e-3)
    assert result["weak_dims"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        [],
        "   \n\t ",
        [{"type": "thinking", "thinking": _SENTINEL}],
        [{"type": "text", "text": ""}],
        [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
    ],
)
async def test_judge_rejects_content_without_body_instead_of_neutral_fallback(content):
    with pytest.raises(ValueError) as exc_info:
        await _judge(content)

    assert str(exc_info.value) == _EMPTY_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)


@pytest.mark.asyncio
async def test_judge_rejects_unknown_block_type_without_leaking_payload():
    with pytest.raises(ValueError) as exc_info:
        await _judge([{"type": "unknown_vendor_block", "text": _SENTINEL}])

    assert str(exc_info.value) == _MALFORMED_CONTENT_MESSAGE
    assert _SENTINEL not in str(exc_info.value)
    assert "unknown_vendor_block" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_judge_keeps_neutral_fallback_for_non_empty_unparseable_text():
    """The legacy neutral fallback survives; only "no body at all" is rejected."""

    result = await _judge([{"type": "text", "text": "This is not JSON at all."}])

    assert result["score"] == pytest.approx(0.6, abs=1e-3)
    assert result["verdict"] == "pass"
