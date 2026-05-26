"""p4a_add_google_id_email_to_users

Revision ID: 7a52fe598fd1
Revises: 8b7d2c4f9a31
Create Date: 2026-05-26 23:38:38.682917

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a52fe598fd1'
down_revision: Union[str, Sequence[str], None] = '8b7d2c4f9a31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("google_id", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("email", sa.String(length=255), nullable=True))
        batch_op.create_unique_constraint("uq_users_google_id", ["google_id"])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("uq_users_google_id", type_="unique")
        batch_op.drop_column("email")
        batch_op.drop_column("google_id")
