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

"""Tests for electron_cyclotron_source."""

from absl.testing import absltest
import chex
import jax.numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import electron_cyclotron_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax.stepper import runtime_params as stepper_runtime_params


class ElectronCyclotronSourceTest(test_lib.SourceTestCase):
  """Tests for ElectronCyclotronSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=electron_cyclotron_source.ElectronCyclotronSource,
        runtime_params_class=electron_cyclotron_source.RuntimeParams,
        source_name=electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME,
        model_func=electron_cyclotron_source.calc_heating_and_current,
    )

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    source_builder = self._source_class_builder()
    if not source_lib.is_source_builder(source_builder):
      raise TypeError(f"{type(self)} has a bad _source_class_builder")
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {self._source_name: source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources[self._source_name]
    source_builder.runtime_params.mode = runtime_params_lib.Mode.MODEL_BASED
    self.assertIsInstance(source, source_lib.Source)
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
            stepper=stepper_runtime_params.RuntimeParams(),
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    value = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    # ElectronCyclotronSource provides TEMP_EL and PSI
    chex.assert_rank(value, 2)
    # ElectronCyclotronSource default model_func provides sane default values
    if jnp.any(jnp.isnan(value)):
      raise AssertionError(f"Source value contains NaNs: {value}")

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    geo = geometry.build_circular_geometry()
    source = self._source_class()
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    fake_profile = jnp.stack((jnp.ones(cell), 2 * jnp.ones(cell)))
    # Check TEMP_EL and PSI are modified
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.TEMP_EL.value,
            geo,
        ),
        jnp.ones(cell),
    )
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.PSI.value,
            geo,
        ),
        2 * jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.NE.value,
            geo,
        ),
        jnp.zeros(cell),
    )


if __name__ == "__main__":
  absltest.main()
