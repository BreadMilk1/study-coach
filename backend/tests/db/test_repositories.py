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


def test_user_repo_get_or_create_recovers_from_concurrent_insert_race(session, monkeypatch):
    """Force commit → IntegrityError → rollback → re-query winner branch."""
    from sqlalchemy.exc import IntegrityError

    from app.db.models import User

    # Winner already committed by a concurrent provisioner.
    winner = User(id="winner-id", fingerprint="fp-race")
    session.add(winner)
    session.commit()

    repo = UserRepository(session)
    real_execute = session.execute
    real_commit = session.commit
    real_rollback = session.rollback
    phase = {"selects": 0, "commits": 0, "rollbacks": 0}

    class EmptyResult:
        def scalar_one_or_none(self):
            return None

    def execute_wrapper(statement, *args, **kwargs):
        phase["selects"] += 1
        # First fingerprint lookup races ahead of the winner commit visibility.
        if phase["selects"] == 1:
            return EmptyResult()
        return real_execute(statement, *args, **kwargs)

    def commit_wrapper():
        phase["commits"] += 1
        if phase["commits"] == 1:
            raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed: users.fingerprint"))
        return real_commit()

    def rollback_wrapper():
        phase["rollbacks"] += 1
        return real_rollback()

    monkeypatch.setattr(session, "execute", execute_wrapper)
    monkeypatch.setattr(session, "commit", commit_wrapper)
    monkeypatch.setattr(session, "rollback", rollback_wrapper)

    recovered = repo.get_or_create("fp-race")

    assert recovered.id == "winner-id"
    assert recovered.fingerprint == "fp-race"
    assert phase["commits"] == 1
    assert phase["rollbacks"] == 1
    assert phase["selects"] >= 2


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


def test_document_repo_create_recovers_from_concurrent_insert_race(session, monkeypatch):
    """Force commit → IntegrityError → rollback → re-query winning document."""
    from sqlalchemy.exc import IntegrityError

    from app.db.models import Document, User

    user = User(id="user-race", fingerprint="fp-doc-race")
    session.add(user)
    session.commit()

    winner = Document(
        id="winner-doc",
        user_id="user-race",
        filename="a.pdf",
        hash="same-hash",
        chunks_count=2,
    )
    session.add(winner)
    session.commit()

    docs = DocumentRepository(session)
    real_execute = session.execute
    real_commit = session.commit
    real_rollback = session.rollback
    phase = {"selects": 0, "commits": 0, "rollbacks": 0}

    class EmptyResult:
        def scalar_one_or_none(self):
            return None

    def execute_wrapper(statement, *args, **kwargs):
        phase["selects"] += 1
        if phase["selects"] == 1:
            return EmptyResult()
        return real_execute(statement, *args, **kwargs)

    def commit_wrapper():
        phase["commits"] += 1
        if phase["commits"] == 1:
            raise IntegrityError(
                "INSERT",
                {},
                Exception("UNIQUE constraint failed: uq_user_doc_hash"),
            )
        return real_commit()

    def rollback_wrapper():
        phase["rollbacks"] += 1
        return real_rollback()

    monkeypatch.setattr(session, "execute", execute_wrapper)
    monkeypatch.setattr(session, "commit", commit_wrapper)
    monkeypatch.setattr(session, "rollback", rollback_wrapper)

    recovered = docs.create(
        user_id="user-race",
        filename="b.pdf",
        hash_="same-hash",
        chunks_count=2,
    )

    assert recovered.id == "winner-doc"
    assert recovered.filename == "a.pdf"
    assert phase["commits"] == 1
    assert phase["rollbacks"] == 1
    assert phase["selects"] >= 2
