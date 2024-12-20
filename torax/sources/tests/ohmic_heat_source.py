# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ohmic_heat_source."""
from absl.testing import absltest
from torax.sources import ohmic_heat_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources.tests import test_lib


class OhmicHeatSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for OhmicHeatSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=ohmic_heat_source.OhmicHeatSource,
        runtime_params_class=ohmic_heat_source.OhmicRuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.FORMULA_BASED,
        ],
        links_back=True,
    )


if __name__ == '__main__':
  absltest.main()
