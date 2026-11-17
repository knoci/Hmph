"""Keep the selected TiMem snapshot while its replacement waits for selection."""

from alembic import op
from sqlalchemy import text

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None

INDEX = "one_ready_memory_build"


def upgrade():
    op.drop_index(INDEX, table_name="memory_builds")
    op.create_index(
        INDEX,
        "memory_builds",
        ["pair_id", "backend"],
        unique=True,
        postgresql_where=text("status = 'ready' AND backend <> 'mem0_timem'"),
    )


def downgrade():
    if op.get_bind().scalar(
        text(
            "SELECT EXISTS (SELECT 1 FROM memory_builds "
            "WHERE status = 'ready' AND backend = 'mem0_timem' "
            "GROUP BY pair_id HAVING count(*) > 1)"
        )
    ):
        raise RuntimeError("Select the replacement TiMem snapshot before downgrading")
    op.drop_index(INDEX, table_name="memory_builds")
    op.create_index(
        INDEX,
        "memory_builds",
        ["pair_id", "backend"],
        unique=True,
        postgresql_where=text("status = 'ready'"),
    )
