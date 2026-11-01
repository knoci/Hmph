"""Track which source messages and memory versions informed each generated turn."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "turn_source_uses",
        sa.Column("turn_id", UUID(), sa.ForeignKey("turns.id"), primary_key=True),
        sa.Column("message_id", UUID(), sa.ForeignKey("messages.id"), primary_key=True),
        sa.Column("message_revision", sa.Integer(), nullable=False),
    )
    op.create_index("ix_turn_source_uses_message_id", "turn_source_uses", ["message_id"])
    op.create_table(
        "turn_memory_uses",
        sa.Column("turn_id", UUID(), sa.ForeignKey("turns.id"), primary_key=True),
        sa.Column("memory_id", UUID(), sa.ForeignKey("memory_entries.id"), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
    )
    op.create_index("ix_turn_memory_uses_memory_id", "turn_memory_uses", ["memory_id"])


def downgrade():
    op.drop_table("turn_memory_uses")
    op.drop_table("turn_source_uses")
