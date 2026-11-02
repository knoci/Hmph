"""Durable identity and stop proof for supervised native writers."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("memory_spaces", sa.Column("execution", JSONB()))
    op.add_column("memory_spaces", sa.Column("stop_proof", sa.String(60)))


def downgrade():
    if op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM memory_spaces "
            "WHERE execution IS NOT NULL AND status != 'cleaned')"
        )
    ):
        raise RuntimeError("Clean supervised framework spaces before downgrade")
    op.drop_column("memory_spaces", "stop_proof")
    op.drop_column("memory_spaces", "execution")
