import os
from uuid import uuid4

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hmph.conversation import Conversation
from hmph.database import Database


@pytest_asyncio.fixture
async def conversation():
    url = os.getenv(
        "HMPH_TEST_DATABASE_URL",
        "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:54329/hmph",
    )
    schema = "test_" + uuid4().hex
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    db = Database(url, schema=schema)
    try:
        await db.migrate()
        service = Conversation(db, digest_key=b"test-only-request-digest-key")
        yield service
    finally:
        await db.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
