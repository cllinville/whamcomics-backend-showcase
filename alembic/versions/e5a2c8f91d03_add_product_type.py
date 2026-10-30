"""add_product_type

Revision ID: e5a2c8f91d03
Revises: 17ac1ba99234
Create Date: 2026-05-14 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5a2c8f91d03'
down_revision: Union[str, None] = '17ac1ba99234'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('product_type', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'product_type')
