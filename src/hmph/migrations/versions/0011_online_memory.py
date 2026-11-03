"""Explicit build selection and bounded product read capabilities; no model calls."""

import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "pairs",
        sa.Column("memory_config_revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column("pairs", sa.Column("selected_memory_build", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "pair_selected_memory_build", "pairs", "memory_builds", ["selected_memory_build"], ["id"]
    )
    op.add_column(
        "turns", sa.Column("memory_context", sa.dialects.postgresql.JSONB(), nullable=True)
    )
    op.create_table(
        "memory_reads",
        sa.Column("turn_id", sa.Uuid(), sa.ForeignKey("turns.id"), primary_key=True),
        sa.Column("build_id", sa.Uuid(), sa.ForeignKey("memory_builds.id"), nullable=False),
        sa.Column("space_id", sa.Uuid(), sa.ForeignKey("memory_spaces.id"), nullable=False),
        sa.Column("config_revision", sa.Integer(), nullable=False),
        sa.Column("cutoff", sa.BigInteger(), nullable=False),
        sa.Column("job_fence", sa.Integer(), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("invalidated", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_memory_reads_build_id", "memory_reads", ["build_id"])


def downgrade():
    op.drop_table("memory_reads")
    op.drop_column("turns", "memory_context")
    op.drop_constraint("pair_selected_memory_build", "pairs", type_="foreignkey")
    op.drop_column("pairs", "selected_memory_build")
    op.drop_column("pairs", "memory_config_revision")
