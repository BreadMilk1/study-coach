from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import (
    ANONYMOUS_TOKEN_TTL_SECONDS,
    issue_token,
    verify_google_credential,
)
from app.db.repositories import UserRepository
from app.db.session import get_session

auth_router = APIRouter(prefix="/api/auth")


class GoogleLoginIn(BaseModel):
    credential: str


class AnonymousLoginIn(BaseModel):
    fingerprint: str


class GuestUpgradeIn(BaseModel):
    credential: str
    fingerprint: str


class AuthOut(BaseModel):
    access_token: str
    user_id: str
    tier: str


@auth_router.post("/google", response_model=AuthOut)
def login_google(body: GoogleLoginIn, session: Annotated[Session, Depends(get_session)]):
    try:
        payload = verify_google_credential(body.credential)
    except ValueError as e:
        raise HTTPException(401, detail=str(e))
    google_id = payload["sub"]
    email = payload.get("email", "")
    user = UserRepository(session).get_or_create_google(google_id=google_id, email=email)
    token = issue_token(user.id, "member")
    return AuthOut(access_token=token, user_id=user.id, tier="member")


@auth_router.post("/anonymous", response_model=AuthOut)
def login_anonymous(body: AnonymousLoginIn, session: Annotated[Session, Depends(get_session)]):
    user = UserRepository(session).get_or_create(body.fingerprint)
    token = issue_token(user.id, "guest", ANONYMOUS_TOKEN_TTL_SECONDS)
    return AuthOut(access_token=token, user_id=user.id, tier="guest")


@auth_router.post("/upgrade", response_model=AuthOut)
def upgrade_guest(body: GuestUpgradeIn, session: Annotated[Session, Depends(get_session)]):
    try:
        payload = verify_google_credential(body.credential)
    except ValueError as e:
        raise HTTPException(401, detail=str(e))
    google_id = payload["sub"]
    email = payload.get("email", "")
    user = UserRepository(session).upgrade_anonymous(
        fingerprint=body.fingerprint, google_id=google_id, email=email
    )
    token = issue_token(user.id, "member")
    return AuthOut(access_token=token, user_id=user.id, tier="member")
