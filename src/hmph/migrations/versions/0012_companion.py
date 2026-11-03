"""Source-backed companion repairs and actual published mentions; no model calls."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("turns", sa.Column("companion_plan", JSONB(), nullable=True))
    op.create_table(
        "companion_repairs",
        sa.Column("source_message_id", sa.Uuid(), sa.ForeignKey("messages.id"), primary_key=True),
        sa.Column("kind", sa.String(40), primary_key=True),
        sa.Column("pair_id", sa.Uuid(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column(
            "source_memory_id", sa.Uuid(), sa.ForeignKey("memory_entries.id"), nullable=False
        ),
        sa.Column("source_revision", sa.Integer(), nullable=False),
        sa.Column("policy_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
    )
    op.create_index("ix_companion_repairs_pair_id", "companion_repairs", ["pair_id"])
    op.create_table(
        "companion_mentions",
        sa.Column("turn_id", sa.Uuid(), sa.ForeignKey("turns.id"), primary_key=True),
        sa.Column("source_message_id", sa.Uuid(), sa.ForeignKey("messages.id"), primary_key=True),
        sa.Column("source_revision", sa.Integer(), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
    )


def downgrade():
    op.drop_table("companion_mentions")
    op.drop_table("companion_repairs")
    op.drop_column("turns", "companion_plan")
