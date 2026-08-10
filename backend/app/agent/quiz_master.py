"""QuizMaster — deterministic Quiz orchestration node for P2.1-④.

State machine:
  - state.active_quiz_question_id present → GRADE: grade_quiz_answer →
    if correct: update_mastery (+delta); if wrong: record_mistake +
    update_mastery (-delta). Clears active_quiz_question_id.
  - absent → GENERATE: extract topic from user message, resolve goal +
    topic (auto-create both if missing), call generate_quiz, persist
    question, set active_quiz_question_id, return formatted prompt.

Why deterministic (not LLM tool-calling) for now: Ollama function calling
is wobbly on small models; this baseline lets us measure agent-loop value
against it in a later P2.2/P3 ablation (portfolio narrative).
"""
import re
from datetime import datetime
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
)

from .state import CoachState
from .tools.quiz import (
    generate_quiz,
    grade_quiz_answer,
    record_mistake,
    update_mastery,
)


_TOPIC_PATTERNS = [
    re.compile(r"quiz me on (.+)", re.IGNORECASE),
    re.compile(r"quiz me about (.+)", re.IGNORECASE),
    re.compile(r"测我一下\s*(.+)"),
    re.compile(r"测一下\s*(.+)"),
    re.compile(r"考一下\s*(.+)"),
    re.compile(r"quiz\s+(.+)", re.IGNORECASE),
]
_DEFAULT_GOAL_TITLE = "Default Study Goal"


def _safe_writer():
    """Stream writer that no-ops when invoked outside a LangGraph runnable.

    Lets unit tests exercise the node by calling it directly without spinning up
    a StateGraph; production paths (graph.astream / ainvoke) get the real writer.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _extract_topic(text: str) -> str:
    for pattern in _TOPIC_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().rstrip("?!?.,。！")
    return text.strip()


def build_quiz_master(
    *,
    llm,
    topic_repo: TopicRepository,
    question_repo: QuestionRepository,
    mistake_repo: MistakeRepository,
    mastery_repo: MasteryRepository,
    goal_repo: GoalRepository,
    retriever=None,
    now_fn: Callable[[], datetime] = datetime.utcnow,
    correct_delta: float = 0.1,
    wrong_delta: float = -0.1,
):
    async def quiz_master_node(state: CoachState) -> dict:
        writer = _safe_writer()
        user_id = state.get("user_id")
        active_q_id = state.get("active_quiz_question_id")
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_msg = user_msgs[-1].content if user_msgs else ""

        # ---- GRADE path ----
        if active_q_id and user_id:
            grade = grade_quiz_answer(
                question_id=active_q_id,
                user_answer=user_msg,
                question_repo=question_repo,
            )
            question = question_repo.get_by_id(active_q_id)
            topic_id = question.topic_id if question else None

            if grade.correct:
                if topic_id:
                    update_mastery(
                        user_id=user_id,
                        topic_id=topic_id,
                        delta=correct_delta,
                        mastery_repo=mastery_repo,
                    )
                feedback = f"✓ Correct! {grade.explanation}"
            else:
                record_mistake(
                    user_id=user_id,
                    question_id=active_q_id,
                    user_answer=user_msg,
                    mistake_repo=mistake_repo,
                    now=now_fn(),
                )
                if topic_id:
                    update_mastery(
                        user_id=user_id,
                        topic_id=topic_id,
                        delta=wrong_delta,
                        mastery_repo=mastery_repo,
                    )
                feedback = (
                    f"✗ Incorrect. Correct answer: {grade.correct_answer}. "
                    f"{grade.explanation}"
                )

            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": feedback})
            return {
                "messages": [AIMessage(content=feedback)],
                "citations": [],
                "active_quiz_question_id": None,
                "last_context": grade.explanation,
                "quiz_action": "grade",
            }

        # ---- GENERATE path ----
        topic_name = _extract_topic(user_msg)

        if not user_id:
            err = "Sign in (provide x-fingerprint header) to start a quiz session."
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": err})
            return {"messages": [AIMessage(content=err)], "citations": []}

        active_goals = goal_repo.list_active_for_user(user_id)
        if active_goals:
            goal_id = active_goals[0].id
        else:
            goal_id = goal_repo.create(user_id=user_id, title=_DEFAULT_GOAL_TITLE).id

        topic = topic_repo.get_by_name(goal_id=goal_id, name=topic_name) \
            or topic_repo.create(goal_id=goal_id, name=topic_name)

        # Cut ④h: ground in source chunks if retriever is wired and finds any.
        # Empty retrieval → fallback to ungrounded generation (current behavior).
        context_chunks = []
        if retriever is not None:
            context_chunks = retriever.search(topic_name, top_k=5) or []
            if context_chunks:
                topic_repo.set_source_chunks(
                    topic_id=topic.id,
                    chunk_ids=[c["chunk_id"] for c in context_chunks],
                )

        quiz = await generate_quiz(
            topic_id=topic.id,
            topic_name=topic_name,
            difficulty="medium",
            n=1,
            llm=llm,
            question_repo=question_repo,
            context_chunks=context_chunks,
        )

        if not quiz.questions:
            err = f"Couldn't generate a quiz question on '{topic_name}'. Try a different topic."
            writer({"type": "citations", "citations": []})
            writer({"type": "token", "text": err})
            return {"messages": [AIMessage(content=err)], "citations": []}

        q = quiz.questions[0]
        text = (
            f"📝 Quiz on {topic_name}:\n\n"
            f"{q.prompt}\n\n"
            + "\n".join(q.options)
            + "\n\nReply with A, B, C, or D."
        )
        writer({"type": "citations", "citations": []})
        writer({"type": "quiz_question", "question_id": q.id})
        writer({"type": "token", "text": text})
        return {
            "messages": [AIMessage(content=text)],
            "citations": [],
            "active_quiz_question_id": q.id,
            "last_context": q.explanation,
            "quiz_action": "generate",
        }

    return quiz_master_node
