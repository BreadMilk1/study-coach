"""Production SQLite connections must enforce foreign keys."""

from sqlalchemy import text

from app.db.session import make_engine


def test_make_engine_enables_foreign_keys_on_file_sqlite(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/fk-file.db")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_make_engine_enables_foreign_keys_on_memory_sqlite():
    engine = make_engine("sqlite:///:memory:")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_make_engine_enables_foreign_keys_on_sqlite_pysqlite_memory():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    try:
        assert engine.dialect.name == "sqlite"
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_make_engine_enables_foreign_keys_on_every_new_connection(tmp_path):
    engine = make_engine(f"sqlite+pysqlite:///{tmp_path}/fk-reconnect.db")
    try:
        with engine.connect() as first:
            assert first.execute(text("PRAGMA foreign_keys")).scalar() == 1
        with engine.connect() as second:
            assert second.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        engine.dispose()


def test_sqlite_url_detection_uses_sqlalchemy_backend_name():
    from app.db import session as session_module

    assert session_module._is_sqlite_url("sqlite:///:memory:") is True
    assert session_module._is_sqlite_url("sqlite+pysqlite:///:memory:") is True
    assert session_module._is_sqlite_url("postgresql+psycopg://u:p@localhost/db") is False
