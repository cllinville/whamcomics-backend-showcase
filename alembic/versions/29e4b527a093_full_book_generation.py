"""full_book_generation

Revision ID: 29e4b527a093
Revises: c3f7a1e44b02
Create Date: 2026-05-09 09:20:37.223183

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '29e4b527a093'
down_revision: Union[str, None] = 'c3f7a1e44b02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TYPE is non-transactional in Postgres — must run outside a transaction block
    op.execute("COMMIT")
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'generating_book'")
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'book_ready'")
    op.execute("ALTER TYPE projectstatus ADD VALUE IF NOT EXISTS 'book_failed'")
    op.execute("BEGIN")

    op.add_column('projects', sa.Column('customer_email', sa.String(), nullable=True))
    op.add_column('projects', sa.Column('book_pdf_key', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'book_pdf_key')
    op.drop_column('projects', 'customer_email')
    # Postgres does not support removing enum values — downgrade leaves them in place
