"""Grounded short-term state, continuous watermarks and fenced maintenance."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "memory_deletions",
        sa.Column("dialogue_states", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "dialogue_states",
        sa.Column("conversation_id", UUID(), sa.ForeignKey("conversations.id"), primary_key=True),
        sa.Column("pair_id", UUID(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("memory_epoch", sa.Integer(), nullable=False),
        sa.Column("coverage_from_seq", sa.BigInteger(), nullable=False),
        sa.Column("processed_through_seq", sa.BigInteger(), nullable=False),
        sa.Column("data", JSONB(), nullable=False),
    )
    op.create_index("ix_dialogue_states_pair_id", "dialogue_states", ["pair_id"])
    op.create_table(
        "dialogue_sources",
        sa.Column(
            "conversation_id",
            UUID(),
            sa.ForeignKey("dialogue_states.conversation_id"),
            primary_key=True,
        ),
        sa.Column("message_id", UUID(), sa.ForeignKey("messages.id"), primary_key=True),
        sa.Column("message_revision", sa.Integer(), nullable=False),
    )
    op.create_index("ix_dialogue_sources_message_id", "dialogue_sources", ["message_id"])
    op.create_table(
        "dialogue_tasks",
        sa.Column(
            "conversation_id",
            UUID(),
            sa.ForeignKey("dialogue_states.conversation_id"),
            primary_key=True,
        ),
        sa.Column("requested_through_seq", sa.BigInteger(), nullable=False),
        sa.Column("model_mode", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("fence", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(100)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(60)),
        sa.Column("usage", JSONB()),
        sa.CheckConstraint(
            "status IN ('pending','leased','idle','failed')", name="dialogue_task_status"
        ),
    )


def downgrade():
    for table in ("dialogue_tasks", "dialogue_sources", "dialogue_states"):
        op.drop_table(table)
    op.drop_column("memory_deletions", "dialogue_states")
