"""Allow a deletion receipt to target an original user message without a memory entry."""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("memory_deletions", "memory_id", nullable=True)
    op.add_column(
        "memory_deletions",
        sa.Column("source_message_id", sa.Uuid(), sa.ForeignKey("messages.id"), nullable=True),
    )
    op.create_check_constraint(
        "deletion_one_target",
        "memory_deletions",
        "(memory_id IS NOT NULL) <> (source_message_id IS NOT NULL)",
    )


def downgrade():
    # Dropping source receipts would discard the user's deletion history.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM memory_deletions "
        "WHERE source_message_id IS NOT NULL) THEN "
        "RAISE EXCEPTION 'Source deletion receipts prevent downgrade'; END IF; END $$"
    )
    op.drop_constraint("deletion_one_target", "memory_deletions", type_="check")
    op.drop_column("memory_deletions", "source_message_id")
    op.alter_column("memory_deletions", "memory_id", nullable=False)
