"""Quiz tool chain (ARCHITECTURE.md §3).

Four plain functions; explicit repo args (no factory). QuizMaster node creates
repos from a Session per turn and passes them in. Tool internals never touch
FastAPI/LangGraph types — keeps them reusable from CLI/eval scripts too.
"""
import json
import re
from datetime import datetime

from langchain_core.messages import HumanMessage

from app.db.repositories import (
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
)
from app.srs.sm2 import next_schedule

from .schemas import GradeOut, MasteryOut, MistakeOut, QuizOut, QuizQuestion


_QUIZ_PROMPT_BASE = """You are a quiz question generator for an exam coach.

Generate {n} multiple-choice question(s) on the topic: {topic_name}.
Difficulty: {difficulty}.
{context_section}
Output ONLY a JSON array. Each item must have:
- prompt: the question text
- options: exactly 4 strings, each prefixed with "A) " "B) " "C) " "D) "
- answer: a single letter A/B/C/D matching the correct option
- explanation: 1-2 sentences explaining why the answer is correct

Example shape:
[
  {{"prompt": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A", "explanation": "..."}}
]
"""

_GROUNDED_CONTEXT_TEMPLATE = """
GROUND your questions strictly in these source chunks. Do NOT invent facts not
supported by them; if a chunk doesn't cover an angle, don't ask about it.

{chunks}
"""


def _format_context_section(context_chunks: list[dict] | None) -> str:
    if not context_chunks:
        return ""
    chunk_lines = "\n".join(
        f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(context_chunks)
    )
    return _GROUNDED_CONTEXT_TEMPLATE.format(chunks=chunk_lines)


# Same 3-tier tolerance pattern as app/agent/judge.py (HKBU lesson):
# fenced ```json ... ``` > bare [...] > whatever parses.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_quiz_json(raw: str) -> list[dict]:
    fence = _FENCE_RE.search(raw)
    if fence:
        return json.loads(fence.group(1))
    bare = _ARRAY_RE.search(raw)
    if bare:
        return json.loads(bare.group(0))
    raise ValueError(f"could not extract JSON array from LLM output: {raw[:200]!r}")


def update_mastery(
    *,
    user_id: str,
    topic_id: str,
    delta: float,
    mastery_repo: MasteryRepository,
) -> MasteryOut:
    new_score = mastery_repo.apply_delta(
        user_id=user_id, topic_id=topic_id, delta=delta
    )
    return MasteryOut(new_score=new_score)


def record_mistake(
    *,
    user_id: str,
    question_id: str,
    user_answer: str,
    mistake_repo: MistakeRepository,
    now: datetime | None = None,
) -> MistakeOut:
    """Log a wrong quiz answer with a first-time-wrong SM-2 schedule (interval=1).

    Note: this is the "first sighting" path. Subsequent failed reviews of the
    same question should call SM-2 with the prior interval/ease (deferred to
    P2.1-④ review-mistake loop or P2.2 polish).
    """
    schedule = next_schedule(quality=0, now=now)
    mistake = mistake_repo.create(
        user_id=user_id,
        question_id=question_id,
        user_answer=user_answer,
        srs_due_at=schedule.due_at,
        srs_interval_days=schedule.interval_days,
        srs_ease=schedule.ease,
    )
    return MistakeOut(mistake_id=mistake.id, srs_due_at=schedule.due_at)


def grade_quiz_answer(
    *,
    question_id: str,
    user_answer: str,
    question_repo: QuestionRepository,
) -> GradeOut:
    question = question_repo.get_by_id(question_id)
    if question is None:
        raise ValueError(f"question {question_id} not found")
    normalized = user_answer.strip().upper()
    correct = normalized == question.answer.strip().upper()
    return GradeOut(
        correct=correct,
        explanation=question.explanation,
        correct_answer=question.answer,
    )


async def generate_quiz(
    *,
    topic_id: str,
    topic_name: str,
    difficulty: str = "medium",
    n: int = 1,
    llm,
    question_repo: QuestionRepository,
    context_chunks: list[dict] | None = None,
) -> QuizOut:
    prompt = _QUIZ_PROMPT_BASE.format(
        n=n,
        topic_name=topic_name,
        difficulty=difficulty,
        context_section=_format_context_section(context_chunks),
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = getattr(response, "content", "") or ""
    parsed = _parse_quiz_json(raw)

    questions: list[QuizQuestion] = []
    for item in parsed[:n]:
        row = question_repo.create(
            topic_id=topic_id,
            prompt=item["prompt"],
            options_json=list(item["options"]),
            answer=item["answer"],
            explanation=item["explanation"],
        )
        questions.append(
            QuizQuestion(
                id=row.id,
                prompt=row.prompt,
                options=row.options_json,
                answer=row.answer,
                explanation=row.explanation,
            )
        )
    return QuizOut(questions=questions)
