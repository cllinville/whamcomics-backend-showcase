"""add_print_pdf_key

Revision ID: 17ac1ba99234
Revises: 29e4b527a093
Create Date: 2026-05-14 18:37:59.844648

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '17ac1ba99234'
down_revision: Union[str, None] = '29e4b527a093'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('book_print_pdf_key', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'book_print_pdf_key')
