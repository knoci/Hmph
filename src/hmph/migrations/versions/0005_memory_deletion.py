"""Opaque receipts for atomic memory deletion."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "memory_deletions",
        sa.Column("id", UUID(), primary_key=True),
        sa.Column("pair_id", UUID(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("memory_id", UUID(), sa.ForeignKey("memory_entries.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("memories", sa.Integer(), nullable=False),
        sa.Column("messages", sa.Integer(), nullable=False),
        sa.Column("turns", sa.Integer(), nullable=False),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade():
    op.drop_table("memory_deletions")
