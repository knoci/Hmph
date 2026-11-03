"""Map isolated native episode IDs to authoritative messages without copying text."""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "memory_native_sources",
        sa.Column("space_id", sa.Uuid(), sa.ForeignKey("memory_spaces.id"), primary_key=True),
        sa.Column("native_id", sa.String(100), primary_key=True),
        sa.Column("message_id", sa.Uuid(), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("message_revision", sa.Integer(), nullable=False),
        sa.Column("source_order", sa.BigInteger(), nullable=False),
    )
    op.create_index("ix_memory_native_sources_message_id", "memory_native_sources", ["message_id"])


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT EXISTS(SELECT 1 FROM memory_native_sources)")):
        raise RuntimeError("Clean native spaces before downgrade")
    op.drop_table("memory_native_sources")
