"""add_character_personalization

Revision ID: b1c4e9d22f01
Revises: ac39fae813dd
Create Date: 2026-05-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b1c4e9d22f01'
down_revision: Union[str, None] = 'ac39fae813dd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('characters', sa.Column('personalization', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('characters', 'personalization')
