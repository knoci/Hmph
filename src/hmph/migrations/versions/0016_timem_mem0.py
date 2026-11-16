"""Allow a jointly published mem0 and TiMem index."""

from alembic import op
from sqlalchemy import text

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("memory_build_backend", "memory_builds", type_="check")
    op.create_check_constraint(
        "memory_build_backend", "memory_builds", "backend IN ('mem0','graphiti','mem0_timem')"
    )


def downgrade():
    if op.get_bind().scalar(
        text("SELECT EXISTS (SELECT 1 FROM memory_builds WHERE backend = 'mem0_timem')")
    ):
        raise RuntimeError("Combined index history must not be discarded")
    op.drop_constraint("memory_build_backend", "memory_builds", type_="check")
    op.create_check_constraint(
        "memory_build_backend", "memory_builds", "backend IN ('mem0','graphiti')"
    )
