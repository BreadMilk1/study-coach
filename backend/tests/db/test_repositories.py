import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base
from app.db.repositories import DocumentRepository, UserRepository


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_user_repo_creates_and_finds_by_fingerprint(session):
    repo = UserRepository(session)

    user = repo.get_or_create("fp-abc")
    found = repo.get_or_create("fp-abc")

    assert user.id == found.id
    assert user.fingerprint == "fp-abc"


def test_user_repo_creates_distinct_users_for_distinct_fingerprints(session):
    repo = UserRepository(session)
    a = repo.get_or_create("fp-a")
    b = repo.get_or_create("fp-b")
    assert a.id != b.id


def test_document_repo_lists_by_user(session):
    users = UserRepository(session)
    docs = DocumentRepository(session)
    user = users.get_or_create("fp-a")

    d1 = docs.create(user_id=user.id, filename="lec1.pdf", hash_="h1", chunks_count=12)
    d2 = docs.create(user_id=user.id, filename="lec2.pdf", hash_="h2", chunks_count=8)

    listed = docs.list_for_user(user.id)
    assert {d.id for d in listed} == {d1.id, d2.id}


def test_document_repo_skips_duplicate_hashes_for_same_user(session):
    users = UserRepository(session)
    docs = DocumentRepository(session)
    user = users.get_or_create("fp-a")

    first = docs.create(user_id=user.id, filename="lec.pdf", hash_="h1", chunks_count=12)
    again = docs.create(user_id=user.id, filename="lec.pdf", hash_="h1", chunks_count=12)

    assert first.id == again.id   # dedup by content hash
    assert len(docs.list_for_user(user.id)) == 1
