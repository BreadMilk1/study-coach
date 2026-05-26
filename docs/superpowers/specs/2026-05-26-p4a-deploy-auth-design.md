# P4a — Deploy & Auth Hardening Design

> 2026-05-26 brainstorm. P4a = OAuth + JWT + Docker Compose + fly.io fallback.

## 1. OAuth: Google Sign-In → JWT

### Flow

```
Frontend                                    Backend
────────                                    ───────
1. Google Identity Services (GIS)
   → user consents → ID token (JWT)

2. POST /api/auth/google                    → verify Google ID token (google-auth)
   Body: { credential: "eyJ..." }           → User.upsert(fingerprint → google_id)
                                            → issue app JWT { user_id, exp=7d }
                                            → return { access_token, user }

3. All subsequent requests:
   Authorization: Bearer <access_token>
   → dependency get_current_user() extracts user_id
```

### Anonymous tier (FingerprintJS → guest)

```
POST /api/auth/anonymous
Body: { fingerprint: "abc123" }
→ upsert User(fingerprint="abc123", google_id=None)
→ issue app JWT { user_id, exp=1h }
→ return { access_token, user, tier: "guest" }
```

### Guest → Google upgrade

When a guest user later signs in with Google:
- `get_current_user()` + incoming Google email → find existing `User` by fingerprint, update `google_id` + `email`
- All guest data (goals, plans, mastery, mistakes) preserved — owned by same `user_id`

### Dependencies

- `google-auth[oauthlib]` — verify Google ID token server-side
- `python-jose[cryptography]` — sign/verify app JWT (HS256, `JWT_SECRET` env var)
- `google-auth` frontend: Google Identity Services script tag (no npm dep needed for basic flow)

### Routes

| Method | Path | Auth | Response |
|--------|------|------|----------|
| POST | `/api/auth/google` | none | `{ access_token, user, tier: "member" }` |
| POST | `/api/auth/anonymous` | none | `{ access_token, user, tier: "guest" }` |

### Migration of existing routes

Current: `user_id: str = Header("default-user")` in `routes.py`

New: 
```python
async def get_current_user(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_session),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401)
    token = authorization.split(" ", 1)[1]
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    user = await user_repo.get(payload["user_id"])
    if not user:
        raise HTTPException(401)
    return user
```

`GET /api/models/tool-check` and `GET /api/models/ping` remain auth-free.

## 2. Docker Compose (primary demo target)

```yaml
# study-coach/docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
    environment:
      - DATABASE_URL=sqlite:///data/study_coach.db
      - CHROMA_PERSIST_DIR=/app/data/chroma
      - JWT_SECRET=${JWT_SECRET:-dev-secret-change-me}
      - GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}

  frontend:
    build: ./frontend
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

### Dockerfile (backend)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY . .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile (frontend)

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
EXPOSE 5173
CMD ["pnpm", "dev", "--host", "0.0.0.0"]
```

### .env.example

```bash
# Required for JWT signing
JWT_SECRET=generate-a-random-secret-here

# Google OAuth (optional — guest mode works without it)
GOOGLE_CLIENT_ID=

# Cloud BYOK (optional)
# x-provider / x-model / x-api-key set per-request via frontend Settings
# These env vars are for server-side defaults only
```

## 3. fly.io fallback (cloud demo link)

```toml
# fly.toml
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
```

- No ollama service on fly — pure BYOK cloud path
- Frontend Settings hides Ollama option when `OLLAMA_ENABLED=false` (read from `GET /api/health` which returns `{ ollama_enabled: bool }`)
- SQLite data on fly volume (sufficient for portfolio demo traffic)
- `Dockerfile.fly` = backend + frontend in single container (nginx reverse proxy, or `uvicorn --proxy-headers`)

## 4. Cloud-adapt hooks

- `# cloud-adapt`: JWT TTL — local dev 7d is fine; cloud may want 1h + refresh token
- `# cloud-adapt`: Google OAuth client ID validation — cloud needs real Google API verification; local dev can accept mock tokens in test mode

## 5. Verification gates

- 3 auth unit tests: google login, anonymous login, guest→google upgrade
- 2 route tests: protected route rejects missing token, accepts valid token
- Docker Compose: `docker compose up` → frontend + backend + ollama all healthy
- fly.io: `fly deploy` → public URL responds to health check
