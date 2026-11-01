from alembic import context

from hmph.models import Base

connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("Run migrations with `hmph migrate`.")
context.configure(
    connection=connection,
    target_metadata=Base.metadata,
    compare_type=True,
    version_table_schema=context.config.attributes["schema"],
)
with context.begin_transaction():
    context.run_migrations()
