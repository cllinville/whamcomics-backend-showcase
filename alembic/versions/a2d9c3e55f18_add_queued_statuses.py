"""add queued and queued_book statuses

Revision ID: a2d9c3e55f18
Revises: f4a4f1434161
Create Date: 2026-06-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'a2d9c3e55f18'
down_revision: Union[str, None] = 'f4a4f1434161'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TYPE is non-transactional in Postgres — must run outside a transaction block
    op.execute("COMMIT")
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'queued'")
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'queued_book'")
    op.execute("BEGIN")


def downgrade() -> None:
    # Postgres does not support removing enum values — downgrade leaves them in place
    pass
