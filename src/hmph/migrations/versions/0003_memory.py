"""Sourced, versioned memory and a durable extraction queue."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "pairs", sa.Column("next_memory_seq", sa.BigInteger(), nullable=False, server_default="1")
    )
    op.add_column(
        "messages", sa.Column("deleted", sa.Boolean(), nullable=False, server_default="false")
    )
    op.add_column(
        "turns", sa.Column("content_redacted", sa.Boolean(), nullable=False, server_default="false")
    )
    op.create_table(
        "memory_entries",
        sa.Column("id", UUID(), primary_key=True),
        sa.Column("pair_id", UUID(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("predicate", sa.String(60), nullable=False),
        sa.Column("scope", sa.String(20), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_order", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
    )
    op.create_index("ix_memory_entries_pair_id", "memory_entries", ["pair_id"])
    op.create_table(
        "memory_versions",
        sa.Column("memory_id", UUID(), sa.ForeignKey("memory_entries.id"), primary_key=True),
        sa.Column("revision", sa.Integer(), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("time_text", sa.String(100)),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "memory_sources",
        sa.Column("memory_id", UUID(), primary_key=True),
        sa.Column("revision", sa.Integer(), primary_key=True),
        sa.Column("message_id", UUID(), sa.ForeignKey("messages.id"), primary_key=True),
        sa.Column("message_revision", sa.Integer(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["memory_id", "revision"], ["memory_versions.memory_id", "memory_versions.revision"]
        ),
    )
    op.create_index("ix_memory_sources_message_id", "memory_sources", ["message_id"])
    op.create_table(
        "memory_tasks",
        sa.Column("id", UUID(), primary_key=True),
        sa.Column("pair_id", UUID(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("source_message_id", UUID(), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("source_revision", sa.Integer(), nullable=False),
        sa.Column("source_order", sa.BigInteger(), nullable=False),
        sa.Column("memory_epoch", sa.Integer(), nullable=False),
        sa.Column("model_mode", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("fence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("worker_id", sa.String(100)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(60)),
        sa.Column("usage", JSONB()),
        sa.UniqueConstraint("source_message_id", "source_revision"),
        sa.CheckConstraint(
            "status IN ('pending','leased','done','failed')", name="memory_task_status"
        ),
    )
    op.create_index("ix_memory_tasks_pair_id", "memory_tasks", ["pair_id"])
    op.create_index(
        "one_memory_lease_per_pair",
        "memory_tasks",
        ["pair_id"],
        unique=True,
        postgresql_where=sa.text("status = 'leased'"),
    )


def downgrade():
    for name in ("memory_tasks", "memory_sources", "memory_versions", "memory_entries"):
        op.drop_table(name)
    op.drop_column("turns", "content_redacted")
    op.drop_column("messages", "deleted")
    op.drop_column("pairs", "next_memory_seq")
