import os
from uuid import uuid4

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from hmph.conversation import Conversation
from hmph.database import Database


async def test_bootstrap_refuses_a_missing_schema_instead_of_writing_to_fallback(conversation):
    import pytest

    from hmph.contracts import DomainError

    database = Database(conversation.db.url, schema="missing_" + uuid4().hex)

    @event.listens_for(database.engine.sync_engine, "connect")
    def configure_search_path(connection, record):
        cursor = connection.cursor()
        cursor.execute(f'SET search_path TO "{database.schema}", "{conversation.db.schema}"')
        cursor.close()

    try:
        service = Conversation(database, digest_key=b"test-only-digest-key")
        with pytest.raises(DomainError, match="database_migration_required"):
            await service.bootstrap()
    finally:
        await database.close()


async def test_empty_schema_is_migrated_even_if_search_path_contains_another_version_table():
    url = os.getenv(
        "HMPH_TEST_DATABASE_URL", "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:54329/hmph"
    )
    destination, fallback = "test_" + uuid4().hex, "test_" + uuid4().hex
    admin = create_async_engine(url)
    database = Database(url, schema=destination)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{destination}"'))
        await connection.execute(text(f'CREATE SCHEMA "{fallback}"'))
        await connection.execute(
            text(f'CREATE TABLE "{fallback}".alembic_version (version_num varchar(32) PRIMARY KEY)')
        )
        await connection.execute(
            text(f"INSERT INTO \"{fallback}\".alembic_version VALUES ('0002')")
        )

    @event.listens_for(database.engine.sync_engine, "connect")
    def configure_search_path(connection, record):
        cursor = connection.cursor()
        cursor.execute(f'SET search_path TO "{destination}", "{fallback}"')
        cursor.close()

    try:
        await database.migrate()
        service = Conversation(database, digest_key=b"test-only-digest-key")
        identity = await service.bootstrap()
        assert await service.chats(identity.owner_id) == []
    finally:
        await database.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{destination}" CASCADE'))
            await connection.execute(text(f'DROP SCHEMA "{fallback}" CASCADE'))
        await admin.dispose()
