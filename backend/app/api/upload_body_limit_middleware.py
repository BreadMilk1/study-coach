"""Pure ASGI middleware: reject oversized upload bodies before multipart parse.

Starlette/FastAPI spool multipart parts (including UploadFile) while reading
the request body. Endpoint-level size checks therefore run too late. This
middleware sits outside DataLifecycleLeaseMiddleware and inside CORS so that:

- Content-Length over the request limit returns 413 without acquiring a lease
- streaming receives that exceed the limit abort before the full body is taken
- CORS still wraps the 413 response
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api import upload_limits


class UploadBodyTooLarge(BaseException):
    """Streamed body exceeded the upload request cap.

    Inherits BaseException (not Exception) so FastAPI/Starlette exception
    handlers cannot convert it into a generic 400/500 before this middleware
    returns a stable 413. Lifecycle middleware ``finally`` still runs.
    """


def _is_document_upload(scope: Scope) -> bool:
    if scope.get("type") != "http":
        return False
    if (scope.get("method") or "").upper() != "POST":
        return False
    return (scope.get("path") or "") == "/api/documents"


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == b"content-length":
            try:
                return int(value.decode("latin-1"))
            except ValueError:
                return None
    return None


def _payload_too_large_response(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": {
                "code": "payload_too_large",
                "message": f"Upload request body must be at most {limit} bytes.",
            }
        },
    )


class UploadBodyLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def _limit(self) -> int:
        # Module attribute lookup so tests can monkeypatch MAX_UPLOAD_REQUEST_BYTES.
        return upload_limits.MAX_UPLOAD_REQUEST_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_document_upload(scope):
            await self.app(scope, receive, send)
            return

        limit = self._limit()
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _payload_too_large_response(limit)(scope, receive, send)
            return

        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"") or b""
                total += len(body)
                if total > limit:
                    raise UploadBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except UploadBodyTooLarge:
            await _payload_too_large_response(limit)(scope, receive, send)
