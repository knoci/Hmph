"""Opt-in shadow maintenance with durable per-attempt call reservations."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "pairs",
        sa.Column(
            "memory_maintenance",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("memory_builds", sa.Column("maintenance_revision", sa.Integer(), nullable=True))
    op.add_column("memory_builds", sa.Column("call_limit", sa.Integer(), nullable=True))
    op.create_check_constraint(
        "maintenance_revision_positive",
        "memory_builds",
        "maintenance_revision IS NULL OR maintenance_revision > 0",
    )
    op.create_check_constraint(
        "maintenance_call_limit",
        "memory_builds",
        "call_limit IS NULL OR (call_limit >= 1 AND call_limit <= 128)",
    )


def downgrade():
    used = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM pairs WHERE memory_maintenance <> '{}'::jsonb) "
                "OR EXISTS (SELECT 1 FROM memory_builds WHERE maintenance_revision IS NOT NULL)"
            )
        )
        .scalar()
    )
    if used:
        raise RuntimeError("Maintenance policy/reservation history must not be discarded")
    op.drop_constraint("maintenance_call_limit", "memory_builds", type_="check")
    op.drop_constraint("maintenance_revision_positive", "memory_builds", type_="check")
    op.drop_column("memory_builds", "call_limit")
    op.drop_column("memory_builds", "maintenance_revision")
    op.drop_column("pairs", "memory_maintenance")
