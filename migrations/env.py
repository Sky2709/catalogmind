"""Alembic environment.

The database URL comes from `Settings`, never from alembic.ini. Config lives in exactly
one place, so `make migrate` can never target a different database than the app does.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.models.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Injected here rather than written into alembic.ini, which would mean a second copy of
# the credentials that can drift from the app's.
config.set_main_option("sqlalchemy.url", get_settings().postgres_dsn)

target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    """Keep autogenerate focused on our own tables."""
    return not (type_ == "table" and name in {"alembic_version"})


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting. Useful for review or a DBA handoff."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Without these two, autogenerate silently misses column type changes and
        # default changes - the migrations look clean while the schema drifts.
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
