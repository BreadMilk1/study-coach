"""Alembic migration environment.

URL resolution priority:
1. Explicit `set_main_option("sqlalchemy.url", ...)` (tests, programmatic callers).
2. `DATABASE_URL` env var (dev/prod CLI usage).
3. Default `sqlite:///./study_coach.db` (matches `app.db.session.get_database_url`).
"""
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.db.models import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    explicit = config.get_main_option("sqlalchemy.url") or ""
    if explicit and not explicit.startswith("driver://"):
        return explicit
    return os.environ.get("DATABASE_URL", "sqlite:///./study_coach.db")


def run_migrations_offline() -> None:
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    config.set_main_option("sqlalchemy.url", _resolve_url())
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
