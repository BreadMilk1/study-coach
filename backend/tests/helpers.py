from sqlalchemy.orm import Session

from app.db.models import User


def ensure_user(
    session: Session,
    user_id: str,
    *,
    fingerprint: str | None = None,
) -> User:
    user = session.get(User, user_id)
    if user is None:
        user = User(id=user_id, fingerprint=fingerprint or f"test-{user_id}")
        session.add(user)
        session.commit()
    return user
