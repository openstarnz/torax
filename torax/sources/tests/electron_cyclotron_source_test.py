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
from absl.testing import absltest
import jax.numpy as jnp
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
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
    geo = circular_geometry.build_circular_geometry()
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
        calculated_source_profiles=None,
    )
    # ElectronCyclotronSource provides TEMP_EL and PSI
    self.assertLen(value, 2)
    # ElectronCyclotronSource default model_func provides sane default values
    self.assertFalse(jnp.any(jnp.isnan(value[0])))
    self.assertFalse(jnp.any(jnp.isnan(value[1])))


if __name__ == "__main__":
  absltest.main()
