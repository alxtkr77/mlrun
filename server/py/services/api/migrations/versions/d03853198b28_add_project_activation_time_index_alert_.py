# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""add_project_activation_time_index_alert_activation

Revision ID: d03853198b28
Revises: d7db206fe4ac
Create Date: 2024-11-20 17:27:56.139748

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "d03853198b28"
down_revision = "d7db206fe4ac"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index("ix_alert_activation_activation_time", table_name="alert_activations")
    op.create_index(
        "ix_alert_activation_project_activation_time",
        "alert_activations",
        ["project", "activation_time"],
        unique=False,
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        "ix_alert_activation_project_activation_time", table_name="alert_activations"
    )
    op.create_index(
        "ix_alert_activation_activation_time",
        "alert_activations",
        ["activation_time"],
        unique=False,
    )
    # ### end Alembic commands ###
