"""Shadow framework builds, versioned source manifests and per-attempt spaces."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "pairs",
        sa.Column("next_framework_seq", sa.BigInteger(), nullable=False, server_default="1"),
    )
    op.create_table(
        "memory_builds",
        sa.Column("id", UUID(), primary_key=True),
        sa.Column("pair_id", UUID(), sa.ForeignKey("pairs.id"), nullable=False),
        sa.Column("generation", sa.BigInteger(), nullable=False),
        sa.Column("memory_epoch", sa.Integer(), nullable=False),
        sa.Column("backend", sa.String(20), nullable=False),
        sa.Column("backend_revision", sa.String(40), nullable=False),
        sa.Column("model", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("watermark", sa.BigInteger(), nullable=False),
        sa.Column("source_count", sa.Integer(), nullable=False),
        sa.Column("snapshot_digest", sa.String(64), nullable=False),
        sa.Column("fence", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(100)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(60)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("ready_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("pair_id", "generation"),
        sa.CheckConstraint("backend IN ('mem0','graphiti')", name="memory_build_backend"),
        sa.CheckConstraint(
            "status IN ('queued','building','ready','retired','failed')", name="memory_build_status"
        ),
    )
    op.create_index("ix_memory_builds_pair_id", "memory_builds", ["pair_id"])
    op.create_index(
        "one_ready_memory_build",
        "memory_builds",
        ["pair_id", "backend"],
        unique=True,
        postgresql_where=sa.text("status = 'ready'"),
    )
    op.create_table(
        "memory_build_sources",
        sa.Column("build_id", UUID(), sa.ForeignKey("memory_builds.id"), primary_key=True),
        sa.Column("message_id", UUID(), sa.ForeignKey("messages.id"), primary_key=True),
        sa.Column("message_revision", sa.Integer(), nullable=False),
        sa.Column("source_order", sa.BigInteger(), nullable=False),
    )
    op.create_table(
        "memory_spaces",
        sa.Column("id", UUID(), primary_key=True),
        sa.Column("build_id", UUID(), sa.ForeignKey("memory_builds.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("fence", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(100), nullable=False),
        sa.Column("quiesced", sa.Boolean(), nullable=False),
        sa.Column("cleanup_fence", sa.Integer(), nullable=False),
        sa.Column("cleanup_attempts", sa.Integer(), nullable=False),
        sa.Column("cleanup_worker_id", sa.String(100)),
        sa.Column("cleanup_until", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(60)),
        sa.Column("usage", JSONB()),
        sa.Column("cleaned_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("build_id", "fence"),
        sa.CheckConstraint(
            "status IN ('writing','ready','retired','cleaning','cleaned')",
            name="memory_space_status",
        ),
    )
    op.create_index("ix_memory_spaces_build_id", "memory_spaces", ["build_id"])
    op.add_column(
        "memory_deletions",
        sa.Column("framework_spaces", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("memory_deletions", "completed_at", nullable=True)
    op.create_table(
        "memory_deletion_spaces",
        sa.Column("deletion_id", UUID(), sa.ForeignKey("memory_deletions.id"), primary_key=True),
        sa.Column("space_id", UUID(), sa.ForeignKey("memory_spaces.id"), primary_key=True),
    )
    op.create_index("ix_memory_deletion_spaces_space_id", "memory_deletion_spaces", ["space_id"])


def downgrade():
    # Preserve the only durable map to external data until every space is cleaned.
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT EXISTS(SELECT 1 FROM memory_spaces WHERE status != 'cleaned') "
                "OR EXISTS(SELECT 1 FROM memory_deletions WHERE completed_at IS NULL)"
            )
        )
        .scalar()
    ):
        raise RuntimeError("Retire and clean all framework spaces before downgrade")
    op.drop_table("memory_deletion_spaces")
    op.drop_column("memory_deletions", "framework_spaces")
    op.alter_column("memory_deletions", "completed_at", nullable=False)
    for table in ("memory_spaces", "memory_build_sources", "memory_builds"):
        op.drop_table(table)
    op.drop_column("pairs", "next_framework_seq")
