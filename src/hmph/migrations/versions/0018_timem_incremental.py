"""Record the published snapshot a TiMem writer extends."""

import sqlalchemy as sa
from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("memory_builds", sa.Column("base_build_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_memory_builds_base_build_id_memory_builds",
        "memory_builds", "memory_builds", ["base_build_id"], ["id"],
    )


def downgrade():
    op.drop_constraint(
        "fk_memory_builds_base_build_id_memory_builds", "memory_builds", type_="foreignkey"
    )
    op.drop_column("memory_builds", "base_build_id")
