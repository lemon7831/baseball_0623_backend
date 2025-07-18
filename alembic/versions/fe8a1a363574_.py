"""empty message

Revision ID: fe8a1a363574
Revises: f999d63c2202
Create Date: 2025-07-02 15:28:12.373643

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fe8a1a363574'
down_revision: Union[str, Sequence[str], None] = 'f999d63c2202'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_pitch_features_id'), table_name='pitch_features')
    op.drop_index(op.f('ix_pitch_features_pitcher_name'), table_name='pitch_features')
    op.drop_index(op.f('ix_pitch_features_video_filename'), table_name='pitch_features')
    op.drop_table('pitch_features')
    op.drop_index(op.f('ix_pitch_profiles_id'), table_name='pitch_profiles')
    op.drop_index(op.f('ix_pitch_profiles_model_name'), table_name='pitch_profiles')
    op.drop_table('pitch_profiles')
    op.add_column('pitch_analyses', sa.Column('release_frame_url', sa.String(), nullable=True))
    op.add_column('pitch_analyses', sa.Column('landing_frame_url', sa.String(), nullable=True))
    op.add_column('pitch_analyses', sa.Column('shoulder_frame_url', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('pitch_analyses', 'shoulder_frame_url')
    op.drop_column('pitch_analyses', 'landing_frame_url')
    op.drop_column('pitch_analyses', 'release_frame_url')
    op.create_table('pitch_profiles',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('model_name', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('method', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('profile_data', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=False),
    sa.Column('source_feature_count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pitch_profiles_pkey'))
    )
    op.create_index(op.f('ix_pitch_profiles_model_name'), 'pitch_profiles', ['model_name'], unique=True)
    op.create_index(op.f('ix_pitch_profiles_id'), 'pitch_profiles', ['id'], unique=False)
    op.create_table('pitch_features',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('pitcher_name', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('pitch_type', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('video_filename', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('description', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('source_csv', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('analyzed_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('trunk_flexion_excursion', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('pelvis_obliquity_at_fc', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('trunk_rotation_at_br', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('shoulder_abduction_at_br', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('trunk_flexion_at_br', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('trunk_lateral_flexion_at_hs', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('release_frame', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('landing_frame', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('shoulder_frame', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('total_frames', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pitch_features_pkey'))
    )
    op.create_index(op.f('ix_pitch_features_video_filename'), 'pitch_features', ['video_filename'], unique=False)
    op.create_index(op.f('ix_pitch_features_pitcher_name'), 'pitch_features', ['pitcher_name'], unique=False)
    op.create_index(op.f('ix_pitch_features_id'), 'pitch_features', ['id'], unique=False)
    # ### end Alembic commands ###
