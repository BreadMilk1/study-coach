"""Pydantic I/O schemas for the Quiz tool chain (ARCHITECTURE.md §3).

Tool functions in `app/agent/tools/quiz.py` take/return these models so the
QuizMaster node (and any future LLM tool-calling agent) sees a stable
contract independent of repository internals.
"""
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# --- update_mastery ----------------------------------------------------------

class MasteryIn(BaseModel):
    user_id: str
    topic_id: str
    delta: float


class MasteryOut(BaseModel):
    new_score: float


# --- record_mistake ----------------------------------------------------------

class MistakeIn(BaseModel):
    user_id: str
    question_id: str
    user_answer: str


class MistakeOut(BaseModel):
    mistake_id: str
    srs_due_at: datetime


# --- grade_quiz_answer -------------------------------------------------------

class GradeIn(BaseModel):
    question_id: str
    user_answer: str


class GradeOut(BaseModel):
    correct: bool
    explanation: str
    correct_answer: str


# --- generate_quiz -----------------------------------------------------------

class QuizIn(BaseModel):
    topic_id: str
    topic_name: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    n: int = Field(default=1, ge=1, le=10)


class QuizQuestion(BaseModel):
    id: str
    prompt: str
    options: list[str]
    answer: str
    explanation: str


class QuizOut(BaseModel):
    questions: list[QuizQuestion]


# --- update_study_plan -------------------------------------------------------

class Milestone(BaseModel):
    id: str | None = None
    topic_id: str | None = None
    title: str
    due_at: str | None = None      # ISO date "YYYY-MM-DD" or None
    done: bool = False
    topic: str | None = None


class PlanPatchIn(BaseModel):
    goal_id: str
    milestones: list[Milestone]


class PlanPatchOut(BaseModel):
    plan_id: str
    updated_at: datetime


# --- generate_mindmap --------------------------------------------------------

class MindmapIn(BaseModel):
    topic: str
    milestones: list[Milestone]


class MindmapOut(BaseModel):
    mermaid_src: str               # "" when LLM output couldn't be parsed
    markdown_outline: str


# --- persist_quiz_question (LLM tool-calling agent, P2.3) --------------------

_OPTION_PREFIXES = ["A) ", "B) ", "C) ", "D) "]
_ANSWER_LETTERS = {"A", "B", "C", "D"}
_RETRIEVAL_METADATA_SUFFIX = re.compile(
    r'\s*,\s*\{?\s*"source"\s*:\s*"[^"]*"\s*,'
    r'\s*"page"\s*:\s*\d+\s*,'
    r'\s*"score"\s*:\s*-?\d+(?:\.\d+)?\s*\}\s*$',
    re.DOTALL,
)


def _normalize_answer_value(value) -> str:
    text = str(value or "").strip().upper()
    if text in _ANSWER_LETTERS:
        return text
    if len(text) >= 2 and text[0] in _ANSWER_LETTERS and text[1] in {")", ".", ":", " "}:
        return text[0]
    return text


def _normalize_option_value(value, *, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        return text

    expected = _OPTION_PREFIXES[index]
    expected_letter = expected[0]

    if text.startswith(expected):
        return text

    if len(text) >= 2 and text[0].upper() in _ANSWER_LETTERS and text[1] in {")", ".", ":"}:
        actual_letter = text[0].upper()
        body = text[2:].strip()
        if actual_letter != expected_letter:
            raise ValueError(
                f"option[{index}] must start with {expected!r}, got explicit {actual_letter})"
            )
        if not body:
            raise ValueError(f"option[{index}] must include text after {expected!r}")
        return f"{expected}{body}"

    return f"{expected}{text}"


class QuizQuestionPersist(BaseModel):
    """Schema for the persist_quiz_question agent tool.

    Strictly more constrained than Milestone (P2.2):
    - options: exactly 4 strings, each prefixed "A) "/"B) "/"C) "/"D) "
    - answer: Literal["A","B","C","D"]
    - prompt and explanation: non-empty strings
    - topic: non-empty string (consumer creates Goal/Topic rows if missing)
    """
    topic: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(min_length=1)

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, v):
        return _normalize_answer_value(v)

    @field_validator("explanation", mode="before")
    @classmethod
    def _remove_retrieval_metadata_suffix(cls, v):
        if not isinstance(v, str):
            return v
        text = v.strip()
        return _RETRIEVAL_METADATA_SUFFIX.sub("", text).strip()

    @field_validator("options", mode="before")
    @classmethod
    def _normalize_options(cls, v):
        if not isinstance(v, list) or len(v) != len(_OPTION_PREFIXES):
            return v
        return [_normalize_option_value(opt, index=i) for i, opt in enumerate(v)]

    @field_validator("options")
    @classmethod
    def _check_option_prefixes(cls, v: list[str]) -> list[str]:
        for i, (opt, prefix) in enumerate(zip(v, _OPTION_PREFIXES)):
            if not opt.startswith(prefix):
                raise ValueError(f"option[{i}] must start with {prefix!r}, got: {opt!r}")
            if not opt[len(prefix):].strip():
                raise ValueError(f"option[{i}] must include text after {prefix!r}")
        return v
