"""add delivered project status

Revision ID: d1f8e3a22c10
Revises: b3e7f2a11c09
Create Date: 2026-07-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'd1f8e3a22c10'
down_revision: Union[str, None] = 'b3e7f2a11c09'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'delivered'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values; downgrade is a no-op
    pass
