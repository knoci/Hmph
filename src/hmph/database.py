import re
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


class Database:
    def __init__(self, url: str, *, schema: str = "public"):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", schema):
            raise ValueError("Invalid database schema")
        if not url.startswith("postgresql+asyncpg://"):
            raise ValueError("Hmph requires PostgreSQL with asyncpg")
        self.url = url
        self.schema = schema
        self.engine = create_async_engine(
            url,
            pool_pre_ping=True,
            connect_args={"server_settings": {"search_path": f'"{schema}",public'}},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def migrate(self):
        def upgrade(connection):
            config = Config()
            config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

        async with self.engine.begin() as connection:
            await connection.run_sync(upgrade)

    async def close(self):
        await self.engine.dispose()
