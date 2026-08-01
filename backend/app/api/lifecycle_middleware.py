"""Pure ASGI middleware that acquires the shared data-lifecycle lease before
any request body is read, and holds it until the response completes.

FastAPI resolves `UploadFile` / multipart parsing before request-scoped
dependencies, so a Depends-based lease is too late for slow uploads.
BaseHTTPMiddleware is avoided because it can release before streaming bodies
finish.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.data_lifecycle import ResetInProgress, ResetRecoveryRequired


def requires_shared_lease(scope: Scope) -> bool:
    if scope.get("type") != "http":
        return False
    path = scope.get("path") or ""
    method = (scope.get("method") or "GET").upper()
    if path in {
        "/api/auth/anonymous",
        "/api/auth/google",
        "/api/auth/upgrade",
    } and method == "POST":
        return True
    if path.startswith("/api/data") or path.startswith("/api/auth"):
        return False
    if path == "/api/health" or path.startswith("/api/models"):
        return False
    return path.startswith("/api/")


def _conflict_response(exc: ResetInProgress) -> JSONResponse:
    if isinstance(exc, ResetRecoveryRequired):
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": "reset_recovery_required",
                    "required_scope": exc.required_scope,
                    "message": (
                        "A previous data reset is incomplete. Retry that reset."
                    ),
                }
            },
        )
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "code": "reset_in_progress",
                "message": "Data reset is in progress.",
            }
        },
    )


class DataLifecycleLeaseMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not requires_shared_lease(scope):
            await self.app(scope, receive, send)
            return

        app_obj = scope.get("app")
        gate = getattr(getattr(app_obj, "state", None), "data_lifecycle_gate", None)
        if gate is None:
            await self.app(scope, receive, send)
            return

        try:
            lease = gate.shared_operation()
            lease.__enter__()
        except ResetInProgress as exc:
            await _conflict_response(exc)(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            lease.__exit__(None, None, None)
