"""Durable conversations, per-pair serialization and generation jobs."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "owners",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
    )
    op.create_table(
        "characters",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("slug", sa.String(80), nullable=False, unique=True),
        sa.Column("persona_version", sa.Integer(), nullable=False),
    )
    op.create_table(
        "pairs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("owners.id"), nullable=False),
        sa.Column("character_id", sa.Uuid(), sa.ForeignKey("characters.id"), nullable=False),
        sa.Column("preferences", postgresql.JSONB(), nullable=False),
        sa.Column("policy_revision", sa.Integer(), nullable=False),
        sa.Column("memory_epoch", sa.Integer(), nullable=False),
        sa.UniqueConstraint("owner_id", "character_id"),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("pair_id", sa.Uuid(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("next_source_seq", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("id", "pair_id"),
    )
    op.create_index("ix_conversations_pair_id", "conversations", ["pair_id"])
    op.create_table(
        "turns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("pair_id", sa.Uuid(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("source_seq", sa.BigInteger(), nullable=False),
        sa.Column("next_event_id", sa.Integer(), nullable=False),
        sa.Column("published_segments", sa.Integer(), nullable=False),
        sa.Column("policy_revision", sa.Integer(), nullable=False),
        sa.Column("memory_epoch", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_code", sa.String(60)),
        sa.ForeignKeyConstraint(
            ["conversation_id", "pair_id"], ["conversations.id", "conversations.pair_id"]
        ),
        sa.CheckConstraint(
            "status IN ('accepted','preparing','generating','completed',"
            "'cancelled','failed','invalidated')",
            name="turn_status",
        ),
    )
    op.create_index("ix_turns_conversation_id", "turns", ["conversation_id"])
    op.create_index(
        "one_active_turn_per_pair",
        "turns",
        ["pair_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('accepted','preparing','generating')"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("conversation_id", sa.Uuid(), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("turn_id", sa.Uuid(), sa.ForeignKey("turns.id"), nullable=False),
        sa.Column("source_seq", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.UniqueConstraint("turn_id", "role"),
        sa.UniqueConstraint("conversation_id", "source_seq"),
        sa.CheckConstraint("role IN ('user','assistant')", name="message_role"),
    )
    op.create_table(
        "stream_events",
        sa.Column("turn_id", sa.Uuid(), sa.ForeignKey("turns.id"), primary_key=True),
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("type", sa.String(40), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("segment_index", sa.Integer()),
        sa.UniqueConstraint("turn_id", "segment_index"),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("turn_id", sa.Uuid(), sa.ForeignKey("turns.id"), nullable=False, unique=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("fence", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(100)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('pending','leased','done')", name="job_status"),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_table(
        "write_requests",
        sa.Column("owner_id", sa.Uuid(), sa.ForeignKey("owners.id"), primary_key=True),
        sa.Column("operation", sa.String(40), primary_key=True),
        sa.Column("request_id", sa.String(128), primary_key=True),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("result_id", sa.Uuid(), nullable=False),
    )


def downgrade():
    for table in (
        "write_requests",
        "jobs",
        "stream_events",
        "messages",
        "turns",
        "conversations",
        "pairs",
        "characters",
        "owners",
    ):
        op.drop_table(table)
