"""Preserve source timezone and referenced-time precision; never invent old dates."""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    # NULL means unknown for legacy sources. No source zone is guessed on upgrade.
    op.add_column("turns", sa.Column("source_timezone", sa.String(80)))
    op.add_column("memory_versions", sa.Column("reference_start", sa.DateTime(timezone=True)))
    op.add_column("memory_versions", sa.Column("reference_end", sa.DateTime(timezone=True)))
    op.add_column(
        "memory_versions",
        sa.Column("time_precision", sa.String(20), nullable=False, server_default="unknown"),
    )
    op.add_column(
        "memory_versions",
        sa.Column("time_basis", sa.String(30), nullable=False, server_default="unresolved"),
    )
    op.add_column("memory_versions", sa.Column("source_timezone", sa.String(80)))
    op.create_check_constraint(
        "reference_interval",
        "memory_versions",
        "(reference_start IS NULL AND reference_end IS NULL) OR "
        "(reference_start IS NOT NULL AND reference_end IS NOT NULL "
        "AND reference_start < reference_end)",
    )


def downgrade():
    op.drop_constraint("reference_interval", "memory_versions", type_="check")
    for column in (
        "reference_start",
        "reference_end",
        "time_precision",
        "time_basis",
        "source_timezone",
    ):
        op.drop_column("memory_versions", column)
    op.drop_column("turns", "source_timezone")
