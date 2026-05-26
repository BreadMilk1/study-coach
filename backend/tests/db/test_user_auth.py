import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.repositories import UserRepository


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


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
