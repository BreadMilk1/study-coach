import os
from dataclasses import dataclass

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from jose import jwt as jose_jwt
from jose.exceptions import JWTError

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 days
ANONYMOUS_TOKEN_TTL_SECONDS = 3600  # 1 hour

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")


@dataclass
class AuthClaims:
    user_id: str
    tier: str  # "member" | "guest"


def verify_google_credential(credential: str) -> dict:
    """Verify a Google Identity Services ID token. Returns the parsed payload
    or raises ValueError on failure."""
    if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID not configured on server")
    return id_token.verify_oauth2_token(
        credential,
        google_requests.Request(),
        GOOGLE_CLIENT_ID,
    )


def issue_token(user_id: str, tier: str, ttl: int = ACCESS_TOKEN_TTL_SECONDS) -> str:
    payload = {"user_id": user_id, "tier": tier}
    return jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> AuthClaims:
    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return AuthClaims(user_id=payload["user_id"], tier=payload.get("tier", "guest"))
    except JWTError:
        raise ValueError("invalid or expired token")
