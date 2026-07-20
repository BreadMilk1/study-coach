import logging
import os
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import (
    get_lifecycle_gate,
    get_retriever_runtime,
    require_signed_user,
)
from app.data_lifecycle import (
    DataOperationInProgress,
    ResetCoordinator,
    ResetInProgress,
    ResetStageError,
)
from app.db.repositories import DataLifecycleRepository
from app.db.session import get_session


logger = logging.getLogger(__name__)

data_router = APIRouter(prefix="/api/data")


class DataCounts(BaseModel):
    users: int
    documents: int
    source_chunks: int
    vectors: int
    chat_sessions: int
    messages: int
    citations: int
    goals: int
    topics: int
    plans: int
    plan_milestones: int
    plan_events: int
    questions: int
    mastery: int
    mistakes: int


class DataSummary(DataCounts):
    reset_enabled: bool
    has_learning_data: bool


class ResetRequest(BaseModel):
    scope: Literal["learning", "factory"]
    confirmation: str


class ResetResponse(BaseModel):
    scope: Literal["learning", "factory"]
    status: Literal["completed"]
    deleted: DataCounts


def reset_enabled() -> bool:
    return os.environ.get("STUDY_COACH_LOCAL_MODE", "0") == "1"


def get_reset_coordinator(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    gate: Annotated[object, Depends(get_lifecycle_gate)],
    runtime: Annotated[object, Depends(get_retriever_runtime)],
) -> ResetCoordinator:
    return ResetCoordinator(
        gate=gate,
        runtime=runtime,
        repository=DataLifecycleRepository(session),
        session=session,
        app_state=request.app.state,
        checkpointer_factory=InMemorySaver,
    )


@data_router.get("/summary", response_model=DataSummary)
def data_summary(
    _user_id: Annotated[str, Depends(require_signed_user)],
    coordinator: Annotated[ResetCoordinator, Depends(get_reset_coordinator)],
):
    try:
        return coordinator.summary(reset_enabled=reset_enabled())
    except ResetInProgress:
        raise HTTPException(
            409,
            detail={
                "code": "reset_in_progress",
                "message": "A data reset is already in progress.",
            },
        ) from None


@data_router.post("/reset", response_model=ResetResponse)
def reset_data(
    body: ResetRequest,
    _user_id: Annotated[str, Depends(require_signed_user)],
    coordinator: Annotated[ResetCoordinator, Depends(get_reset_coordinator)],
):
    if not reset_enabled():
        raise HTTPException(
            403,
            detail={
                "code": "reset_disabled",
                "message": "Data reset is disabled in this environment.",
            },
        ) from None
    expected_confirmation = {
        "learning": "CLEAR_LEARNING_DATA",
        "factory": "FACTORY_RESET",
    }[body.scope]
    if body.confirmation != expected_confirmation:
        raise HTTPException(
            422,
            detail={
                "code": "invalid_confirmation",
                "message": "Confirmation text does not match reset scope.",
            },
        ) from None
    try:
        return coordinator.reset(body.scope)
    except ResetInProgress:
        raise HTTPException(
            409,
            detail={
                "code": "reset_in_progress",
                "message": "A data reset is already in progress.",
            },
        ) from None
    except DataOperationInProgress:
        raise HTTPException(
            409,
            detail={
                "code": "data_operation_in_progress",
                "message": "A data operation is currently in progress.",
            },
        ) from None
    except ResetStageError as exc:
        logger.exception("Data reset failed at %s stage", exc.stage)
        raise HTTPException(
            500,
            detail={
                "code": "reset_failed",
                "failed_stage": exc.stage,
                "retryable": True,
                "message": "Data reset failed. Please retry.",
            },
        ) from None
