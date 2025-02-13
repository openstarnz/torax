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
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles
from torax.stepper import runtime_params as stepper_runtime_params_lib


class SourceModelsTest(parameterized.TestCase):

  def test_computing_source_profiles_works_with_all_defaults(self):
    """Tests that you can compute source profiles with all defaults."""
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    geo = circular_geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    stepper_params = stepper_runtime_params_lib.RuntimeParams()
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            source_runtime_params=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
            stepper=stepper_params,
        )
    )
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        source_models,
        explicit=True,
    )
    source_profile_builders.build_source_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        source_models,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
    )

  def test_computing_standard_source_profiles_for_single_affected_core_profile(
      self,
  ):
    geo = circular_geometry.build_circular_geometry()

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
    class TestSource(source.Source):

      @property
      def source_name(self) -> str:
        return 'foo'

      @property
      def affected_core_profiles(
          self,
      ) -> tuple[source.AffectedCoreProfile, ...]:
        return (source.AffectedCoreProfile.PSI,)

    test_source = TestSource(
        model_func=lambda *args: (jnp.ones(geo.rho.shape),)
    )
    source_models = mock.create_autospec(source_models_lib.SourceModels)
    source_models.standard_sources = {'foo': test_source}
    test_source_runtime_params = source_runtime_params.StaticRuntimeParams(
        mode=1, is_explicit=True
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={'foo': test_source_runtime_params},
        torax_mesh=geo.torax_mesh,
    )
    dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources={
            'foo': source_runtime_params.DynamicRuntimeParams(
                prescribed_values=jnp.ones(geo.rho.shape)
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
        qei=source_profiles.QeiInfo.zeros(geo),
    )
    source_profile_builders.build_standard_source_profiles(
        static_runtime_params_slice=static_params,
        dynamic_runtime_params_slice=dynamic_params,
        geo=geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        explicit=True,
        calculated_source_profiles=profiles,
    )
    psi_profiles = profiles.psi
    self.assertLen(psi_profiles, 1)
    self.assertIn('foo', psi_profiles)
    np.testing.assert_equal(psi_profiles['foo'].shape, geo.rho.shape)

  def test_computing_standard_source_profiles_for_multiple_affected_core_profile(
      self,
  ):
    geo = circular_geometry.build_circular_geometry()

    @dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
    class TestSource(source.Source):

      @property
      def source_name(self) -> str:
        return 'foo'

      @property
      def affected_core_profiles(
          self,
      ) -> tuple[source.AffectedCoreProfile, ...]:
        return (
            source.AffectedCoreProfile.TEMP_ION,
            source.AffectedCoreProfile.TEMP_EL,
        )

    test_source = TestSource(
        model_func=lambda *args: (jnp.ones_like(geo.rho),) * 2
    )
    source_models = mock.create_autospec(source_models_lib.SourceModels)
    source_models.standard_sources = {'foo': test_source}
    test_source_runtime_params = source_runtime_params.StaticRuntimeParams(
        mode=1, is_explicit=True
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={'foo': test_source_runtime_params},
        torax_mesh=geo.torax_mesh,
    )
    dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources={
            'foo': source_runtime_params.DynamicRuntimeParams(
                prescribed_values=jnp.ones(geo.rho.shape)
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
        qei=source_profiles.QeiInfo.zeros(geo),
    )
    source_profile_builders.build_standard_source_profiles(
        static_runtime_params_slice=static_params,
        dynamic_runtime_params_slice=dynamic_params,
        geo=geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        explicit=True,
        calculated_source_profiles=profiles,
    )

    # Check that a single profile is returned for each affected core profile.
    # These profiles should be the same shape as the geo.rho.
    ion_profiles = profiles.temp_ion
    self.assertLen(ion_profiles, 1)
    self.assertIn('foo', ion_profiles)
    np.testing.assert_equal(ion_profiles['foo'].shape, geo.rho.shape)

    el_profiles = profiles.temp_el
    self.assertLen(el_profiles, 1)
    self.assertIn('foo', el_profiles)
    np.testing.assert_equal(el_profiles['foo'].shape, geo.rho.shape)


if __name__ == '__main__':
  absltest.main()
