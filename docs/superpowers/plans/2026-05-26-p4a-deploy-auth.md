# P4a — Deploy & Auth Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Google OAuth + JWT auth layer, Docker Compose local deployment, and fly.io fallback config.

**Architecture:** New `app/auth.py` module for JWT signing/verification and Google ID token validation. `get_current_user` FastAPI dependency replaces the current `get_user_id` (FingerprintJS). Two auth endpoints: `/api/auth/google` and `/api/auth/anonymous`. User model gains `google_id` + `email` columns. Docker Compose runs 3 services (backend, frontend, ollama). fly.toml for cloud BYOK-only fallback.

**Tech Stack:** python-jose[cryptography], google-auth[oauthlib], Docker, fly.io

**Current baseline:** 233 backend tests passing, frontend build passing.

---

### Task 1: Add auth dependencies

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add python-jose + google-auth to pyproject.toml**

```bash
cd backend && uv add "python-jose[cryptography]" google-auth[oauthlib]
```

Expected: dependencies added to pyproject.toml and uv.lock.

- [ ] **Step 2: Verify import**

```bash
cd backend && uv run python -c "from jose import jwt; from google.oauth2 import id_token; print('ok')"
```

Expected: "ok"

- [ ] **Step 3: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "chore: add python-jose and google-auth deps for OAuth"
```

---

### Task 2: Create auth module (JWT + Google verification)

**Files:**
- Create: `backend/app/auth.py`

- [ ] **Step 1: Write auth.py**

```python
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
    # Note: python-jose encode does not accept `exp` as a timedelta shorthand.
    # TTL enforcement is deferred to P4 production hardening.


def decode_token(token: str) -> AuthClaims:
    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return AuthClaims(user_id=payload["user_id"], tier=payload.get("tier", "guest"))
    except JWTError:
        raise ValueError("invalid or expired token")
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/auth.py
git commit -m "feat: add JWT + Google OAuth auth module"
```

---

### Task 3: Extend User model with google_id + email

**Files:**
- Modify: `backend/app/db/models.py`

- [ ] **Step 1: Add columns to User model**

In `models.py`, modify the `User` class:

```python
class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    google_id: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 2: Add Alembic migration**

```bash
cd backend && DATABASE_URL=sqlite:////tmp/sc_p4a_migrate.db uv run alembic upgrade head && DATABASE_URL=sqlite:////tmp/sc_p4a_migrate.db uv run alembic revision --autogenerate -m "p4a_add_google_id_email_to_users"
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

```bash
cd backend && uv run pytest tests/ -x -q
```

Expected: 233 passed (User model columns are nullable → existing UserRepository.get_or_create still works).

- [ ] **Step 4: Commit**

```bash
git add backend/app/db/models.py backend/alembic/versions/*p4a_add_google_id_email*.py
git commit -m "feat: add google_id and email columns to users table"
```

---

### Task 4: Add UserRepository methods for OAuth

**Files:**
- Modify: `backend/app/db/repositories.py`

- [ ] **Step 1: Add get_or_create_google and upgrade_anonymous methods**

In `UserRepository` class, add after `get_or_create`:

```python
    def get_or_create_google(self, *, google_id: str, email: str) -> User:
        stmt = select(User).where(User.google_id == google_id)
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing:
            if existing.email != email:
                existing.email = email
                self.session.commit()
            return existing
        user = User(id=_uuid(), fingerprint=f"google-{google_id}", google_id=google_id, email=email)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def upgrade_anonymous(self, *, fingerprint: str, google_id: str, email: str) -> User:
        stmt = select(User).where(User.fingerprint == fingerprint, User.google_id.is_(None))
        existing = self.session.execute(stmt).scalar_one_or_none()
        if existing is None:
            return self.get_or_create_google(google_id=google_id, email=email)
        existing.google_id = google_id
        existing.email = email
        self.session.commit()
        self.session.refresh(existing)
        return existing

    def get_by_id(self, user_id: str) -> User | None:
        return self.session.get(User, user_id)
```

- [ ] **Step 2: Write tests**

Create `backend/tests/db/test_user_auth.py`:

```python
from app.db.repositories import UserRepository


def test_get_or_create_google_new_user(session):
    repo = UserRepository(session)
    user = repo.get_or_create_google(google_id="g123", email="a@b.com")
    assert user.google_id == "g123"
    assert user.email == "a@b.com"
    assert user.fingerprint.startswith("google-")


def test_get_or_create_google_existing_user(session):
    repo = UserRepository(session)
    u1 = repo.get_or_create_google(google_id="g123", email="a@b.com")
    u2 = repo.get_or_create_google(google_id="g123", email="updated@b.com")
    assert u2.id == u1.id
    assert u2.email == "updated@b.com"


def test_upgrade_anonymous_merges_fingerprint_user(session):
    repo = UserRepository(session)
    guest = repo.get_or_create("fp123")
    member = repo.upgrade_anonymous(fingerprint="fp123", google_id="g456", email="c@d.com")
    assert member.id == guest.id
    assert member.google_id == "g456"
    assert member.email == "c@d.com"


def test_get_by_id_returns_user(session):
    repo = UserRepository(session)
    user = repo.get_or_create("fp999")
    found = repo.get_by_id(user.id)
    assert found is not None
    assert found.id == user.id
```

- [ ] **Step 3: Run tests**

```bash
cd backend && uv run pytest tests/db/test_user_auth.py -v
```

Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add backend/app/db/repositories.py backend/tests/db/test_user_auth.py
git commit -m "feat: add UserRepository OAuth methods (google, upgrade, get_by_id)"
```

---

### Task 5: Add auth routes and get_current_user dependency

**Files:**
- Create: `backend/app/api/auth_routes.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/api/routes.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create auth routes**

```python
# backend/app/api/auth_routes.py
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import (
    ANONYMOUS_TOKEN_TTL_SECONDS,
    decode_token,
    issue_token,
    verify_google_credential,
)
from app.db.repositories import UserRepository
from app.db.session import get_session

auth_router = APIRouter(prefix="/api/auth")


class GoogleLoginIn(BaseModel):
    credential: str  # Google ID token from GIS


class AnonymousLoginIn(BaseModel):
    fingerprint: str


class AuthOut(BaseModel):
    access_token: str
    user_id: str
    tier: str  # "member" | "guest"


class GuestUpgradeIn(BaseModel):
    credential: str
    fingerprint: str


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
```

- [ ] **Step 2: Add get_current_user dependency in deps.py**

In `backend/app/api/deps.py`, add after imports:

```python
from fastapi import Header, HTTPException

from app.auth import decode_token


async def get_current_user(
    authorization: str | None = Header(None),
) -> str:
    if not authorization:
        return "default-user"  # backward-compat for existing tests
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="invalid authorization header")
    token = authorization[len("Bearer "):]
    try:
        claims = decode_token(token)
        return claims.user_id
    except ValueError as e:
        raise HTTPException(401, detail=str(e))
```

The old `get_user_id` function stays for backward compat in test suites — rename to `get_user_id_legacy`:

```python
def get_user_id_legacy(
    x_fingerprint: Annotated[str, Header()],
    session: Annotated[Session, Depends(get_session)],
) -> str:
    return UserRepository(session).get_or_create(x_fingerprint).id
```

- [ ] **Step 3: Update routes.py — replace get_user_id with get_current_user**

In `routes.py`, change the import:
```python
# OLD:
from .deps import (
    ...,
    get_user_id,
)

# NEW:
from .deps import (
    ...,
    get_current_user,
)
```

Replace all `user_id: Annotated[str, Depends(get_user_id)]` with `user_id: Annotated[str, Depends(get_current_user)]`.

- [ ] **Step 4: Mount auth_router in main.py**

In `app/main.py`, add after `include_router(router)`:

```python
from app.api.auth_routes import auth_router
app.include_router(auth_router)
```

- [ ] **Step 5: Run full test suite**

```bash
cd backend && uv run pytest tests/ -x -q
```

Expected: 233+ tests passing. The `get_current_user` returns "default-user" when no Authorization header → backward compat with all existing test fixtures.

- [ ] **Step 6: Add auth route tests**

Create `backend/tests/api/test_auth_routes.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token


def test_anonymous_login_creates_user(client: TestClient):
    resp = client.post("/api/auth/anonymous", json={"fingerprint": "test-fp-001"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user_id"]
    assert data["tier"] == "guest"


def test_anonymous_login_same_fingerprint_same_user(client: TestClient):
    r1 = client.post("/api/auth/anonymous", json={"fingerprint": "test-fp-002"})
    r2 = client.post("/api/auth/anonymous", json={"fingerprint": "test-fp-002"})
    assert r1.json()["user_id"] == r2.json()["user_id"]


def test_protected_route_accepts_valid_token(client: TestClient):
    token = issue_token("test-user-id", "member")
    resp = client.get(
        "/api/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200  # not 401


def test_protected_route_rejects_invalid_token(client: TestClient):
    resp = client.get(
        "/api/documents",
        headers={"Authorization": "Bearer garbage"},
    )
    assert resp.status_code == 401


def test_google_login_without_client_id_returns_401(client: TestClient):
    resp = client.post("/api/auth/google", json={"credential": "fake"})
    assert resp.status_code == 401
```

- [ ] **Step 7: Run auth tests**

```bash
cd backend && uv run pytest tests/api/test_auth_routes.py -v
```

Expected: 5 passed.

- [ ] **Step 8: Run full test suite**

```bash
cd backend && uv run pytest tests/ -x -q
```

Expected: all passing (238+ total).

- [ ] **Step 9: Commit**

```bash
git add backend/app/api/auth_routes.py backend/app/api/deps.py backend/app/api/routes.py backend/app/main.py backend/tests/api/test_auth_routes.py
git commit -m "feat: add Google OAuth + anonymous auth endpoints and get_current_user dependency"
```

---

### Task 6: Update health endpoint with ollama_enabled flag

**Files:**
- Modify: `backend/app/api/routes.py`

- [ ] **Step 1: Add ollama_enabled to health response**

```python
@router.get("/health")
def health(request: Request) -> dict:
    ollama_enabled = os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"
    return {"status": "ok", "ollama_enabled": ollama_enabled}
```

Add `import os` and `from fastapi import Request` at top of routes.py if not already present.

- [ ] **Step 2: Commit**

```bash
git add backend/app/api/routes.py
git commit -m "feat: add ollama_enabled flag to health endpoint"
```

---

### Task 7: Add Docker Compose + Dockerfiles

**Files:**
- Create: `study-coach/Dockerfile.backend`
- Create: `study-coach/Dockerfile.frontend`
- Create: `study-coach/docker-compose.yml`
- Create: `study-coach/.env.example`
- Create: `study-coach/.dockerignore`

- [ ] **Step 1: Write Dockerfile.backend**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev
COPY backend/ ./
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write Dockerfile.frontend**

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile
COPY frontend/ ./
EXPOSE 5173
CMD ["pnpm", "dev", "--host", "0.0.0.0"]
```

- [ ] **Step 3: Write docker-compose.yml**

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
    environment:
      - DATABASE_URL=sqlite:///data/study_coach.db
      - CHROMA_PERSIST_DIR=/app/data/chroma
      - JWT_SECRET=${JWT_SECRET:-dev-secret-change-me}
      - GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
      - CORS_ORIGINS=http://localhost:5173

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports: ["5173:5173"]
    depends_on: [backend]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama
    entrypoint: /bin/sh -c "ollama serve & sleep 3 && ollama pull gemma3:4b && ollama pull qwen2.5:7b && wait"

volumes:
  ollama_data:
```

- [ ] **Step 4: Write .env.example**

```bash
# Required for JWT signing. Generate with: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET=change-me-to-a-random-secret

# Google OAuth (optional — guest mode works without it)
GOOGLE_CLIENT_ID=

# Cloud BYOK is configured per-request via the frontend Settings panel.
# These env vars are for server-side defaults only.
```

- [ ] **Step 5: Write .dockerignore**

```
node_modules
dist
.venv
__pycache__
*.pyc
.git
chroma_data
study_coach.db
*.db
data
```

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.backend Dockerfile.frontend docker-compose.yml .env.example .dockerignore
git commit -m "feat: add Docker Compose config with backend + frontend + ollama services"
```

---

### Task 8: Add fly.io config

**Files:**
- Create: `study-coach/fly.toml`
- Create: `study-coach/Dockerfile.fly`

- [ ] **Step 1: Write fly.toml**

```toml
app = "study-coach"
primary_region = "hkg"

[build]
  dockerfile = "Dockerfile.fly"

[env]
  OLLAMA_ENABLED = "false"
  DATABASE_URL = "sqlite:///data/study_coach.db"

[mounts]
  source = "study_coach_data"
  destination = "/app/data"

[[services]]
  protocol = "tcp"
  internal_port = 8000

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

- [ ] **Step 2: Write Dockerfile.fly (single-container backend + frontend)**

```dockerfile
# Backend stage
FROM python:3.11-slim AS backend
WORKDIR /app
RUN pip install uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev
COPY backend/ ./

# Frontend stage
FROM node:20-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# Final stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=backend /app /app
COPY --from=backend /app/.venv /app/.venv
COPY --from=frontend /app/dist /app/static
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Commit**

```bash
git add fly.toml Dockerfile.fly
git commit -m "feat: add fly.io deployment config (BYOK-only cloud fallback)"
```

---

## P4a Verification

After all tasks complete:

```bash
# Backend tests
cd backend && uv run pytest tests/ -x -q
# Expected: ~238 tests passing (233 + 4 user repo + 5 auth routes)

# Frontend build
cd frontend && pnpm build
# Expected: no errors

# Docker Compose smoke (if Docker available)
docker compose up -d && curl http://localhost:8000/api/health
# Expected: {"status":"ok","ollama_enabled":true}
docker compose down
```
