"""Source-backed shared threads; interaction rhythm derives from published turns."""

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "shared_threads",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("pair_id", sa.Uuid(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("topic", sa.String(120), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_message_id", sa.Uuid(), sa.ForeignKey("messages.id"), nullable=False),
        sa.CheckConstraint(
            "status IN ('open','paused','resolved','dropped')", name="shared_thread_status"
        ),
        sa.UniqueConstraint("pair_id", "topic"),
    )
    op.create_index("ix_shared_threads_pair_id", "shared_threads", ["pair_id"])
    op.create_table(
        "shared_thread_sources",
        sa.Column(
            "thread_id",
            sa.Uuid(),
            sa.ForeignKey("shared_threads.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("message_id", sa.Uuid(), sa.ForeignKey("messages.id"), primary_key=True),
    )


def downgrade():
    op.drop_table("shared_thread_sources")
    op.drop_table("shared_threads")
