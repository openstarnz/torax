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

"""Tests for generic_current_source."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import generic_current_source
from torax.sources import source as source_lib
from torax.sources.tests import test_lib


class GenericCurrentSourceTest(test_lib.SourceTestCase):
  """Tests for GenericCurrentSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=generic_current_source.GenericCurrentSource,
        runtime_params_class=generic_current_source.RuntimeParams,
        source_name=generic_current_source.GenericCurrentSource.SOURCE_NAME,
        model_func=generic_current_source.calculate_generic_current,
    )

  def test_profile_is_on_cell_grid(self):
    """Tests that the profile is given on the cell grid."""
    geo = geometry.build_circular_geometry()
    source_builder = self._source_class_builder()
    source = source_builder()
    self.assertEqual(
        source.output_shape_getter(geo),
        source_lib.ProfileType.CELL.get_profile_shape(geo),
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    dynamic_runtime_params_slice = runtime_params_slice.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        sources={
            generic_current_source.GenericCurrentSource.SOURCE_NAME: (
                source_builder.runtime_params
            ),
        },
        torax_mesh=geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params={
            generic_current_source.GenericCurrentSource.SOURCE_NAME: (
                source_builder.runtime_params
            ),
        },
        torax_mesh=geo.torax_mesh,
    )
    self.assertEqual(
        source.get_value(
            static_slice,
            dynamic_runtime_params_slice,
            geo,
            core_profiles=None,
        ).shape,
        source_lib.ProfileType.CELL.get_profile_shape(geo),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='psi_profile_yields_profile',
          affected_core_profile=source_lib.AffectedCoreProfile.PSI,
          expected_profile=np.array([1.5, 2.5]),
      ),
      dict(
          testcase_name='unaffected_profile_yields_zeros',
          affected_core_profile=source_lib.AffectedCoreProfile.TEMP_ION,
          expected_profile=np.array([0.0, 0.0]),
      ),
  )
  def test_get_source_profile_for_affected_core_profile_with(
      self,
      affected_core_profile: source_lib.AffectedCoreProfile,
      expected_profile: chex.Array,
  ):
    source_builder = self._source_class_builder()
    source = source_builder()

    # Build a face profile with 3 values on a 2-cell grid.
    geo = geometry.build_circular_geometry(n_rho=2)
    cell_profile = np.array([1.5, 2.5])

    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            cell_profile,
            affected_core_profile.value,
            geo,
        ),
        expected_profile,
    )


if __name__ == '__main__':
  absltest.main()
