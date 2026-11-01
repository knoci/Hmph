"""Separate fake and local DeepSeek jobs, and retain only sanitized usage metadata."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "turns", sa.Column("model_mode", sa.String(20), server_default="fake", nullable=False)
    )
    op.add_column("turns", sa.Column("usage", JSONB(), nullable=True))


def downgrade():
    op.drop_column("turns", "usage")
    op.drop_column("turns", "model_mode")
