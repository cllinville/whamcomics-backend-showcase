"""add_project_genre

Revision ID: c3f7a1e44b02
Revises: b1c4e9d22f01
Create Date: 2026-05-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c3f7a1e44b02'
down_revision: Union[str, None] = 'b1c4e9d22f01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('genre', sa.String(), nullable=False, server_default='adventure'))


def downgrade() -> None:
    op.drop_column('projects', 'genre')
