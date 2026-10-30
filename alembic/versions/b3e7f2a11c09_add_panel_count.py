"""add panel_count to projects

Revision ID: b3e7f2a11c09
Revises: a2d9c3e55f18
Create Date: 2026-06-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b3e7f2a11c09'
down_revision: Union[str, None] = 'a2d9c3e55f18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('panel_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'panel_count')
