import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt

from app.api.data_routes import ResetRequest, reset_data
from app.api.deps import require_signed_user
from app.auth import JWT_ALGORITHM, JWT_SECRET, issue_token
from app.data_lifecycle import ResetStageError
from app.db.repositories import DataLifecycleRepository
from app.db.models import Document, EvalRun, EvalScoreSet, EvalScorerExecution, User
from app.db.session import session_scope
from app.main import create_app


class FakeRuntime:
    def __init__(self, vectors: int = 0) -> None:
        self.vectors = vectors
        self.retriever = object()

    def vector_count(self) -> int:
        return self.vectors

    def reset_empty(self):
        self.vectors = 0
        self.retriever = object()
        return self.retriever


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_COACH_TEST_MODE", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/data-api.db")
    monkeypatch.delenv("STUDY_COACH_LOCAL_MODE", raising=False)

    from app.db import session as session_module

    session_module._engine = None
    session_module._SessionLocal = None
    application = create_app()
    application.state.retriever_runtime = FakeRuntime()
    application.state.retriever = application.state.retriever_runtime.retriever
    yield application
    if session_module._engine is not None:
        session_module._engine.dispose()
    session_module._engine = None
    session_module._SessionLocal = None


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def bearer(user_id: str = "signed-user-without-a-row") -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(user_id, 'member')}"}


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/data/summary", None),
        (
            "post",
            "/api/data/reset",
            {"scope": "learning", "confirmation": "CLEAR_LEARNING_DATA"},
        ),
    ],
)
@pytest.mark.parametrize(
    "authorization",
    [None, "Basic abc", "Bearer garbage"],
)
def test_data_routes_require_a_valid_signed_bearer(
    client,
    method,
    path,
    json_body,
    authorization,
):
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.request(method, path, headers=headers, json=json_body)

    assert response.status_code == 401


def test_summary_accepts_signed_token_when_user_row_does_not_exist(client):
    response = client.get("/api/data/summary", headers=bearer())

    assert response.status_code == 200
    assert response.json()["users"] == 0
    assert response.json()["current_user_exists"] is False


def test_summary_reports_current_user_exists_for_live_identity(client):
    with session_scope() as session:
        session.add(User(id="live-user", fingerprint="fp-live"))
        session.commit()

    response = client.get("/api/data/summary", headers=bearer("live-user"))

    assert response.status_code == 200
    assert response.json()["current_user_exists"] is True
    assert response.json()["users"] == 1


def test_reset_is_disabled_by_default(client):
    response = client.post(
        "/api/data/reset",
        headers=bearer(),
        json={"scope": "learning", "confirmation": "CLEAR_LEARNING_DATA"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "code": "reset_disabled",
            "message": "Data reset is disabled in this environment.",
        }
    }


COUNT_KEYS = {
    "users",
    "documents",
    "source_chunks",
    "vectors",
    "chat_sessions",
    "messages",
    "citations",
    "goals",
    "topics",
    "plans",
    "plan_milestones",
    "plan_events",
    "questions",
    "mastery",
    "mistakes",
}
EVAL_COUNT_KEYS = {
    "runs",
    "score_sets",
    "scorer_executions",
    "estimated_bytes",
}


def test_summary_ignores_users_when_computing_learning_data(client):
    with session_scope() as session:
        session.add(User(id="user-1", fingerprint="only-user"))
        session.commit()

    response = client.get("/api/data/summary", headers=bearer("user-1"))

    assert response.status_code == 200
    assert response.json()["users"] == 1
    assert response.json()["has_learning_data"] is False
    assert response.json()["reset_enabled"] is False


def test_summary_treats_vectors_as_learning_data(client, app):
    app.state.retriever_runtime.vectors = 4

    response = client.get("/api/data/summary", headers=bearer())

    assert response.status_code == 200
    assert response.json()["vectors"] == 4
    assert response.json()["has_learning_data"] is True


def test_summary_reports_source_chunks_separately_from_vectors(client, app):
    with session_scope() as session:
        session.add(User(id="user-1", fingerprint="document-owner"))
        session.flush()
        session.add(
            Document(
                id="document-1",
                user_id="user-1",
                filename="notes.pdf",
                hash="hash-1",
                chunks_count=7,
            )
        )
        session.commit()
    app.state.retriever_runtime.vectors = 3

    response = client.get("/api/data/summary", headers=bearer("user-1"))

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == COUNT_KEYS | {
        "reset_enabled",
        "has_learning_data",
        "current_user_exists",
        "eval",
    }
    assert payload["eval"] == {
        "runs": 0,
        "score_sets": 0,
        "scorer_executions": 0,
        "estimated_bytes": 0,
    }
    assert payload["current_user_exists"] is True
    assert payload["source_chunks"] == 7
    assert payload["vectors"] == 3


def enable_reset(monkeypatch) -> None:
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")


def test_reset_rejects_unknown_scope(client, monkeypatch):
    enable_reset(monkeypatch)

    response = client.post(
        "/api/data/reset",
        headers=bearer(),
        json={"scope": "everything", "confirmation": "FACTORY_RESET"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("scope", "confirmation"),
    [
        ("learning", "FACTORY_RESET"),
        ("factory", "CLEAR_LEARNING_DATA"),
    ],
)
def test_reset_rejects_confirmation_for_the_other_scope(
    client,
    monkeypatch,
    scope,
    confirmation,
):
    enable_reset(monkeypatch)

    response = client.post(
        "/api/data/reset",
        headers=bearer(),
        json={"scope": scope, "confirmation": confirmation},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "invalid_confirmation",
            "message": "Confirmation text does not match reset scope.",
        }
    }


@pytest.mark.parametrize(
    ("scope", "confirmation"),
    [
        ("learning", "CLEAR_LEARNING_DATA"),
        ("factory", "FACTORY_RESET"),
    ],
)
def test_reset_completes_with_every_deleted_count_key(
    client,
    app,
    monkeypatch,
    scope,
    confirmation,
):
    enable_reset(monkeypatch)
    with session_scope() as session:
        session.add(User(id="user-1", fingerprint="reset-owner"))
        session.commit()
    app.state.retriever_runtime.vectors = 2

    response = client.post(
        "/api/data/reset",
        headers=bearer("user-1"),
        json={"scope": scope, "confirmation": confirmation},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == scope
    assert payload["status"] == "completed"
    assert set(payload["deleted"]) == COUNT_KEYS
    assert payload["deleted"]["vectors"] == 2
    assert payload["deleted"]["users"] == (0 if scope == "learning" else 1)
    assert set(payload["deleted_eval"]) == EVAL_COUNT_KEYS
    assert payload["deleted_eval"] == {
        "runs": 0,
        "score_sets": 0,
        "scorer_executions": 0,
        "estimated_bytes": 0,
    }


def _seed_eval_rows(run_id: str = "eval-run-1") -> None:
    with session_scope() as session:
        session.add(
            EvalRun(
                id=run_id,
                experiment_id="tutor-prompt-regression-v1",
                task_case_id="tgqa-001",
                task_case_version="1",
                variant_id="tutor-v2",
                run_profile="evaluation",
                lifecycle="finished",
                outcome="success",
                manifest_json={"task_case_id": "tgqa-001"},
                manifest_hash="a" * 64,
                candidate_artifact_json={"answer": "x"},
                artifact_hash="b" * 64,
            )
        )
        session.add(
            EvalScoreSet(
                id=f"{run_id}-score",
                run_id=run_id,
                scorer_id="hybrid",
                scorer_version="hybrid-v1",
                scorer_snapshot_json={"scorer_id": "hybrid", "version": "hybrid-v1"},
                scorer_definition_hash="c" * 64,
                artifact_input_hash="b" * 64,
                status="completed",
                quality_verdict="pass",
                aggregate_scores_json={"groundedness": 4},
            )
        )
        session.add(
            EvalScorerExecution(
                id=f"{run_id}-exec",
                score_set_id=f"{run_id}-score",
                scorer_id="retrieval-integrity",
                scorer_version="v1",
                status="success",
                input_hash="b" * 64,
                output_json={"result": True},
            )
        )
        session.commit()


def test_summary_eval_only_does_not_count_as_learning_data(client):
    _seed_eval_rows()

    response = client.get("/api/data/summary", headers=bearer())

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_learning_data"] is False
    assert payload["eval"]["runs"] == 1
    assert payload["eval"]["score_sets"] == 1
    assert payload["eval"]["scorer_executions"] == 1
    assert payload["eval"]["estimated_bytes"] > 0


def test_learning_reset_preserves_eval_rows_and_reports_empty_deleted_eval(
    client, monkeypatch
):
    enable_reset(monkeypatch)
    _seed_eval_rows()

    response = client.post(
        "/api/data/reset",
        headers=bearer(),
        json={"scope": "learning", "confirmation": "CLEAR_LEARNING_DATA"},
    )

    assert response.status_code == 200
    assert response.json()["deleted_eval"] == {
        "runs": 0,
        "score_sets": 0,
        "scorer_executions": 0,
        "estimated_bytes": 0,
    }
    summary = client.get("/api/data/summary", headers=bearer()).json()
    assert summary["eval"]["runs"] == 1
    assert summary["eval"]["score_sets"] == 1
    assert summary["eval"]["scorer_executions"] == 1


def test_active_eval_lease_rejects_factory_reset_and_keeps_eval_rows(
    client, app, monkeypatch
):
    enable_reset(monkeypatch)
    _seed_eval_rows()

    with app.state.data_lifecycle_gate.shared_operation():
        response = client.post(
            "/api/data/reset",
            headers=bearer(),
            json={"scope": "factory", "confirmation": "FACTORY_RESET"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "data_operation_in_progress"
    summary = client.get("/api/data/summary", headers=bearer()).json()
    assert summary["eval"]["runs"] == 1
    assert summary["eval"]["score_sets"] == 1
    assert summary["eval"]["scorer_executions"] == 1


def test_factory_reset_clears_eval_and_returns_pre_reset_deleted_eval(
    client, monkeypatch
):
    enable_reset(monkeypatch)
    _seed_eval_rows()
    before = client.get("/api/data/summary", headers=bearer()).json()["eval"]

    response = client.post(
        "/api/data/reset",
        headers=bearer(),
        json={"scope": "factory", "confirmation": "FACTORY_RESET"},
    )

    assert response.status_code == 200
    assert response.json()["deleted_eval"] == before
    after = client.get("/api/data/summary", headers=bearer()).json()["eval"]
    assert after == {
        "runs": 0,
        "score_sets": 0,
        "scorer_executions": 0,
        "estimated_bytes": 0,
    }


def test_reset_accepts_signed_token_when_user_row_does_not_exist(client, monkeypatch):
    enable_reset(monkeypatch)

    response = client.post(
        "/api/data/reset",
        headers=bearer("never-persisted"),
        json={"scope": "learning", "confirmation": "CLEAR_LEARNING_DATA"},
    )

    assert response.status_code == 200


def reset_payload(scope: str = "learning") -> dict[str, str]:
    return {
        "scope": scope,
        "confirmation": (
            "CLEAR_LEARNING_DATA" if scope == "learning" else "FACTORY_RESET"
        ),
    }


def test_reset_reports_active_data_operation_conflict(client, app, monkeypatch):
    enable_reset(monkeypatch)

    with app.state.data_lifecycle_gate.shared_operation():
        response = client.post(
            "/api/data/reset",
            headers=bearer(),
            json=reset_payload(),
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "data_operation_in_progress"


def test_reset_reports_active_reset_conflict(client, app, monkeypatch):
    enable_reset(monkeypatch)

    with app.state.data_lifecycle_gate.exclusive_reset():
        response = client.post(
            "/api/data/reset",
            headers=bearer(),
            json=reset_payload(),
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "reset_in_progress"


def test_summary_reports_active_reset_conflict(client, app):
    with app.state.data_lifecycle_gate.exclusive_reset():
        response = client.get("/api/data/summary", headers=bearer())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "reset_in_progress"


def test_incomplete_reset_only_allows_same_scope_recovery(client, app, monkeypatch):
    enable_reset(monkeypatch)
    app.state.data_lifecycle_gate.mark_recovery_required("learning")

    summary = client.get("/api/data/summary", headers=bearer())
    wrong_scope = client.post(
        "/api/data/reset",
        headers=bearer(),
        json={"scope": "factory", "confirmation": "FACTORY_RESET"},
    )

    expected_detail = {
        "code": "reset_recovery_required",
        "required_scope": "learning",
        "message": "A previous data reset is incomplete. Retry that reset.",
    }
    assert summary.status_code == 409
    assert summary.json()["detail"] == expected_detail
    assert wrong_scope.status_code == 409
    assert wrong_scope.json()["detail"] == expected_detail

    retry = client.post(
        "/api/data/reset",
        headers=bearer(),
        json=reset_payload(),
    )
    assert retry.status_code == 200
    assert client.get("/api/data/summary", headers=bearer()).status_code == 200


@pytest.mark.parametrize("stage", ["chroma", "sqlite"])
def test_reset_stage_failure_has_stable_safe_payload(
    client,
    app,
    monkeypatch,
    stage,
):
    enable_reset(monkeypatch)
    sensitive_error = f"secret {stage} path and credentials"

    if stage == "chroma":
        def fail_reset_empty():
            raise RuntimeError(sensitive_error)

        monkeypatch.setattr(app.state.retriever_runtime, "reset_empty", fail_reset_empty)
    else:
        def fail_delete(_repository, *, include_users):
            raise RuntimeError(sensitive_error)

        monkeypatch.setattr(DataLifecycleRepository, "delete_learning_data", fail_delete)

    response = client.post(
        "/api/data/reset",
        headers=bearer(),
        json=reset_payload(),
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "reset_failed",
            "failed_stage": stage,
            "retryable": True,
            "message": "Data reset failed. Please retry.",
        }
    }
    assert sensitive_error not in response.text


@pytest.mark.parametrize("user_id", [None, "", "   ", 0, [], {}])
def test_signed_token_with_malformed_user_claim_is_rejected(client, user_id):
    token = jose_jwt.encode(
        {"user_id": user_id, "tier": "member"},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )

    response = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_signed_token_without_required_user_claim_is_rejected(client):
    token = jose_jwt.encode({"tier": "member"}, JWT_SECRET, algorithm=JWT_ALGORITHM)

    response = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_bearer_error_suppresses_decode_exception_context():
    with pytest.raises(HTTPException) as caught:
        await require_signed_user("Bearer garbage")

    assert caught.value.status_code == 401
    assert caught.value.__suppress_context__ is True


def test_reset_error_suppresses_stage_exception_context(monkeypatch):
    monkeypatch.setenv("STUDY_COACH_LOCAL_MODE", "1")

    class FailingCoordinator:
        def reset(self, _scope):
            try:
                raise RuntimeError("secret sqlite path")
            except RuntimeError as exc:
                raise ResetStageError("sqlite") from exc

    with pytest.raises(HTTPException) as caught:
        reset_data(
            ResetRequest(scope="learning", confirmation="CLEAR_LEARNING_DATA"),
            "signed-user",
            FailingCoordinator(),
        )

    assert caught.value.status_code == 500
    assert caught.value.__suppress_context__ is True
    assert "secret sqlite path" not in str(caught.value.detail)


def test_factory_reset_can_retry_with_same_token_after_user_is_deleted(
    client,
    monkeypatch,
):
    enable_reset(monkeypatch)
    with session_scope() as session:
        session.add(User(id="user-1", fingerprint="factory-owner"))
        session.commit()
    headers = bearer("user-1")

    first_response = client.post(
        "/api/data/reset",
        headers=headers,
        json=reset_payload("factory"),
    )
    retry_response = client.post(
        "/api/data/reset",
        headers=headers,
        json=reset_payload("factory"),
    )

    assert first_response.status_code == 200
    assert first_response.json()["deleted"]["users"] == 1
    assert retry_response.status_code == 200
    assert retry_response.json()["status"] == "completed"
    assert retry_response.json()["deleted"]["users"] == 0


def test_factory_reset_old_token_cannot_recreate_learning_data(
    client,
    app,
    monkeypatch,
):
    """Old JWT may retry factory reset, but must not write orphan learning rows."""
    enable_reset(monkeypatch)
    with session_scope() as session:
        session.add(User(id="u-old", fingerprint="fp-old"))
        session.commit()
    headers = bearer("u-old")

    reset_response = client.post(
        "/api/data/reset",
        headers=headers,
        json=reset_payload("factory"),
    )
    assert reset_response.status_code == 200

    # Response-lost retry must remain available for the same signed token.
    retry = client.post(
        "/api/data/reset",
        headers=headers,
        json=reset_payload("factory"),
    )
    assert retry.status_code == 200

    goal_response = client.post(
        "/api/goals",
        headers=headers,
        json={"title": "orphan after factory"},
    )
    assert goal_response.status_code in {401, 409}

    summary = client.get("/api/data/summary", headers=headers)
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["users"] == 0
    assert payload["goals"] == 0
    assert payload["has_learning_data"] is False
    assert payload["current_user_exists"] is False
