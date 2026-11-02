import re
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
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

    @staticmethod
    def _migration_config():
        config = Config()
        config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
        return config

    async def is_current(self):
        """Check this schema's version, never a search_path fallback's version."""
        expected = set(ScriptDirectory.from_config(self._migration_config()).get_heads())
        async with self.sessions() as session:
            table = f'"{self.schema}".alembic_version'
            if await session.scalar(text("SELECT to_regclass(:table)"), {"table": table}) is None:
                return False
            versions = set(await session.scalars(text(f"SELECT version_num FROM {table}")))
            return versions == expected

    async def migrate(self):
        def upgrade(connection):
            config = self._migration_config()
            config.attributes["connection"] = connection
            config.attributes["schema"] = self.schema
            command.upgrade(config, "head")

        async with self.engine.begin() as connection:
            await connection.run_sync(upgrade)

    async def close(self):
        await self.engine.dispose()
