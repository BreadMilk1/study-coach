"""Pydantic I/O schemas for the Quiz tool chain (ARCHITECTURE.md §3).

Tool functions in `app/agent/tools/quiz.py` take/return these models so the
QuizMaster node (and any future LLM tool-calling agent) sees a stable
contract independent of repository internals.
"""
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

    @field_validator("options")
    @classmethod
    def _check_option_prefixes(cls, v: list[str]) -> list[str]:
        expected_prefixes = ["A) ", "B) ", "C) ", "D) "]
        for i, (opt, prefix) in enumerate(zip(v, expected_prefixes)):
            if not opt.startswith(prefix):
                raise ValueError(f"option[{i}] must start with {prefix!r}, got: {opt!r}")
        return v
