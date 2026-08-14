import os
from typing import Annotated, Literal

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth import decode_token
from app.db.repositories import (
    GoalRepository,
    MasteryRepository,
    MistakeRepository,
    QuestionRepository,
    TopicRepository,
    UserRepository,
)
from app.db.session import get_session, get_eval_session as _get_eval_session
from app.llm.provider import LLMConfig, parse_llm_config


async def require_signed_user(
    authorization: str | None = Header(None),
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="signed bearer token required") from None
    try:
        user_id = decode_token(authorization[len("Bearer "):]).user_id
    except (KeyError, ValueError):
        raise HTTPException(401, detail="signed bearer token required") from None
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(401, detail="signed bearer token required") from None
    return user_id


async def require_existing_user(
    user_id: Annotated[str, Depends(require_signed_user)],
    session: Annotated[Session, Depends(get_session)],
) -> str:
    """Learning routes: signed bearer whose user row still exists.

    Lifecycle summary/reset keep `require_signed_user` (row-less) so a
    factory-reset success response can be retried with the same JWT after the
    user row is gone. Ordinary learning writes must not recreate orphan data.
    """
    if UserRepository(session).get_by_id(user_id) is None:
        raise HTTPException(401, detail="user no longer exists") from None
    return user_id


async def require_local_eval_mode() -> None:
    """Allow evaluation routes only on an explicitly local instance."""
    if os.environ.get("STUDY_COACH_LOCAL_MODE") != "1":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "evaluation_disabled",
                "message": "local evaluation mode is disabled",
            },
        )


def get_eval_session():
    """Dependency alias kept under the API dependency boundary."""
    yield from _get_eval_session()


def get_user_id(
    x_fingerprint: Annotated[str, Header()],
    session: Annotated[Session, Depends(get_session)],
) -> str:
    return UserRepository(session).get_or_create(x_fingerprint).id


def get_llm_config(
    x_provider: Annotated[str | None, Header()] = None,
    x_model: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    x_base_url: Annotated[str | None, Header()] = None,
    x_judge_model: Annotated[str | None, Header()] = None,
) -> LLMConfig:
    return parse_llm_config(
        x_provider=x_provider,
        x_model=x_model,
        x_api_key=x_api_key,
        x_base_url=x_base_url,
        x_judge_model=x_judge_model,
    )


def get_retriever(request: Request):
    return request.app.state.retriever


def get_document_processor(request: Request):
    return request.app.state.document_processor


def get_checkpointer(request: Request):
    return request.app.state.checkpointer


def get_lifecycle_gate(request: Request):
    return request.app.state.data_lifecycle_gate


def get_retriever_runtime(request: Request):
    return request.app.state.retriever_runtime


def get_llm(llm_config: Annotated[LLMConfig, Depends(get_llm_config)]):
    from app.llm.provider import get_chat_model
    return get_chat_model(llm_config)


def get_judge_dependencies(
    llm_config: Annotated[LLMConfig, Depends(get_llm_config)],
) -> dict:
    """Return judge_llm + same_model flag based on x-judge-model header.

    - x-judge-model set AND distinct from x-model → distinct judge LLM, same_model=False
    - otherwise → same LLM, same_model=True (caller surfaces self-preference warning)
    """
    from app.llm.provider import get_chat_model
    judge_target = llm_config.effective_judge_model()
    same_model = judge_target == llm_config.model
    if same_model:
        return {"llm": get_chat_model(llm_config), "same_model": True}
    judge_cfg = LLMConfig(
        provider=llm_config.provider,
        model=judge_target,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        judge_model=judge_target,
    )
    return {"llm": get_chat_model(judge_cfg), "same_model": False}


def get_graph(
    retriever: Annotated[object, Depends(get_retriever)],
    llm: Annotated[object, Depends(get_llm)],
    checkpointer: Annotated[object, Depends(get_checkpointer)],
):
    from app.agent.graph import build_graph
    return build_graph(retriever=retriever, llm=llm, checkpointer=checkpointer)


def get_quiz_master(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    from app.agent.quiz_master import build_quiz_master
    return build_quiz_master(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        mistake_repo=MistakeRepository(session),
        mastery_repo=MasteryRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever,
    )


def get_memory_hydrator(session: Annotated[Session, Depends(get_session)]):
    from app.agent.memory_updater import build_memory_hydrator
    return build_memory_hydrator(
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
    )


def get_memory_writer(session: Annotated[Session, Depends(get_session)]):
    from app.agent.memory_updater import build_memory_writer
    return build_memory_writer(
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
    )


def get_planner(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: when provider=cloud, a future kwarg `mindmap_default=True` can
    # be threaded through to enable auto-mindmap without keyword.
    from app.agent.planner import build_planner
    from app.db.repositories import (
        GoalRepository,
        MasteryRepository,
        MistakeRepository,
        PlanRepository,
    )
    return build_planner(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
    )


def get_planner_mode(
    x_planner_mode: Annotated[str | None, Header()] = None,
) -> Literal["deterministic", "agent_loop"]:
    """Read x-planner-mode header. Default = deterministic. Unknown → deterministic.

    Defensive default: unknown values silently fall back so a typo on the
    client side never breaks production. The eval harness sends the header
    explicitly and never relies on default.
    """
    if x_planner_mode == "agent_loop":
        return "agent_loop"
    return "deterministic"


def get_planner_agent(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: when provider=cloud (BYOK GPT/Claude/Gemini), max_iter can safely be raised to 20-30 here; small local Ollama models cap at 10.
    from app.agent.planner_agent import build_planner_agent
    from app.db.repositories import (
        GoalRepository,
        MasteryRepository,
        MistakeRepository,
        PlanRepository,
    )
    return build_planner_agent(
        llm=llm,
        plan_repo=PlanRepository(session),
        goal_repo=GoalRepository(session),
        mastery_repo=MasteryRepository(session),
        mistake_repo=MistakeRepository(session),
        retriever=retriever,
    )


def get_quiz_mode(
    x_quiz_mode: Annotated[str | None, Header()] = None,
) -> Literal["deterministic", "agent_loop"]:
    """Read x-quiz-mode header. Default = deterministic. Unknown → deterministic."""
    if x_quiz_mode == "agent_loop":
        return "agent_loop"
    return "deterministic"


def get_quiz_master_agent(
    session: Annotated[Session, Depends(get_session)],
    llm: Annotated[object, Depends(get_llm)],
    retriever: Annotated[object, Depends(get_retriever)],
):
    # cloud-adapt: cloud BYOK can raise max_iter from 6 to 12 here
    from app.agent.quiz_master_agent import build_quiz_master_agent
    from app.db.repositories import (
        GoalRepository, QuestionRepository, TopicRepository,
    )
    return build_quiz_master_agent(
        llm=llm,
        topic_repo=TopicRepository(session),
        question_repo=QuestionRepository(session),
        goal_repo=GoalRepository(session),
        retriever=retriever,
    )
