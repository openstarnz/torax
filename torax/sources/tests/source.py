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

"""Tests for source_lib.py."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib


def get_zero_profile(
    profile_type: source_lib.ProfileType,
    geo: geometry.Geometry,
) -> jax.Array:
  """Returns a source profile with all zeros."""
  return jnp.zeros(profile_type.get_profile_shape(geo))


@dataclasses.dataclass(frozen=True, eq=True)
class PsiTestSource(source_lib.Source):

  @property
  def affected_core_profiles(self):
    return (source_lib.AffectedCoreProfile.PSI,)

  @property
  def source_name(self) -> str:
    return 'foo'


PsiTestSourceBuilder = source_lib.make_source_builder(PsiTestSource)


@dataclasses.dataclass(frozen=True, eq=True)
class IonElTestSource(source_lib.Source):
  """Test source that affects ion and electron profiles."""

  @property
  def source_name(self) -> str:
    return 'foo'

  @property
  def affected_core_profiles(self):
    return (
        source_lib.AffectedCoreProfile.TEMP_ION,
        source_lib.AffectedCoreProfile.TEMP_EL,
    )


IonElTestSourceBuilder = source_lib.make_source_builder(IonElTestSource)


class SourceTest(parameterized.TestCase):
  """Tests for the base class Source."""

  def test_is_source_builder(self):
    """Test that is_source_builder works correctly."""

    @dataclasses.dataclass
    class MySource:
      pass

    @dataclasses.dataclass
    class NoRuntimeParams:

      def __call__(self) -> MySource:
        return MySource()

    obj_without_runtime_params = NoRuntimeParams()

    self.assertFalse(source_lib.is_source_builder(obj_without_runtime_params))

    @dataclasses.dataclass
    class NotCallable:
      runtime_params: int

    obj_without_call = NotCallable(runtime_params=1)

    self.assertFalse(source_lib.is_source_builder(obj_without_call))

    @dataclasses.dataclass
    class ValidBuilder:
      runtime_params: int

      def __call__(self) -> MySource:
        return MySource()

    valid_builder = ValidBuilder(runtime_params=1)
    source_lib.is_source_builder(valid_builder, raise_if_false=True)

  def test_source_builder_type_checking(self):
    """Tests that source builders check types on construction."""

    class NotDataclass:
      pass

    with self.assertRaises(TypeError):
      source_lib.make_source_builder(
          NotDataclass,
          links_back=False,
          runtime_params_type=int,
      )()

    @dataclasses.dataclass
    class NotFrozen:
      my_field: int

    with self.assertRaises(TypeError):
      source_lib.make_source_builder(
          NotFrozen,
          links_back=False,
          runtime_params_type=int,
      )()

    @dataclasses.dataclass(frozen=True)
    class NotEq:
      my_field: int

    with self.assertRaises(TypeError):
      source_lib.make_source_builder(
          NotEq,
          links_back=False,
          runtime_params_type=int,
      )()

    @dataclasses.dataclass(frozen=True, eq=True)
    class MySource:
      my_field: int
      model_func: source_lib.SourceProfileFunction | None = None

    # pylint doesn't realize this is a class
    MySourceBuilder = source_lib.make_source_builder(  # pylint: disable=invalid-name
        MySource,
        links_back=False,
        runtime_params_type=int,
    )

    valid = MySourceBuilder(my_field=1, runtime_params=2)
    valid()  # Check that it allows constructing the source too

    with self.assertRaises(TypeError):
      MySourceBuilder(my_field='hello', runtime_params=2)
    with self.assertRaises(TypeError):
      MySourceBuilder(my_field=1, runtime_params={})

  def test_zero_profile_works_by_default(self):
    """The default source impl should support profiles with all zeros."""
    source_builder = PsiTestSourceBuilder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'foo': source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources['foo']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
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
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(
        profile,
        get_zero_profile(source_lib.ProfileType.CELL, geo),
    )

  @parameterized.parameters(
      (runtime_params_lib.Mode.ZERO, np.array([0, 0, 0, 0])),
      (runtime_params_lib.Mode.MODEL_BASED, np.array([2, 2, 2, 2])),
      (runtime_params_lib.Mode.PRESCRIBED, np.array([3, 3, 3, 3])),
  )
  def test_correct_mode_called(self, mode, expected_profile):
    """The correct mode should be called."""
    source_builder = source_lib.make_source_builder(
        test_lib.TestSource,
        model_func=lambda _0, _1, _2, _3, _4, _5: jnp.ones(
            source_lib.ProfileType.CELL.get_profile_shape(geo)
        ) * 2,
    )()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'foo': source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources['foo']
    source_runtime_params = source_models_builder.runtime_params
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry(n_rho=4)
    source_runtime_params['foo'] = dataclasses.replace(
        source_models_builder.runtime_params['foo'],
        mode=mode,
    )
    source_runtime_params['foo'].prescribed_values = 3
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(
        profile,
        expected_profile,
    )

  def test_defaults_output_zeros(self):
    """The default model and formula implementations should output zeros."""
    source_builder = test_lib.TestSourceBuilder()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'foo': source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources['foo']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
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
    with self.subTest('model_based'):
      static_slice = runtime_params_slice.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          source_runtime_params={
              'foo': dataclasses.replace(
                  source_builder.runtime_params,
                  mode=runtime_params_lib.Mode.MODEL_BASED,
              )
          },
          torax_mesh=geo.torax_mesh,
      )
      with self.assertRaises(ValueError):
        source.get_value(
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            static_runtime_params_slice=static_slice,
            geo=geo,
            core_profiles=core_profiles,
        )
    with self.subTest('prescribed'):
      static_slice = runtime_params_slice.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          source_runtime_params={
              'foo': dataclasses.replace(
                  source_builder.runtime_params,
                  mode=runtime_params_lib.Mode.PRESCRIBED,
              )
          },
          torax_mesh=geo.torax_mesh,
      )
      profile = source.get_value(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          static_runtime_params_slice=static_slice,
          geo=geo,
          core_profiles=core_profiles,
      )
      np.testing.assert_allclose(
          profile,
          get_zero_profile(source_lib.ProfileType.CELL, geo),
      )

  def test_overriding_model(self):
    """The user-specified model should override the default model."""
    geo = geometry.build_circular_geometry()
    output_shape = source_lib.ProfileType.CELL.get_profile_shape(geo)
    expected_output = jnp.ones(output_shape)
    source_builder = source_lib.make_source_builder(
        IonElTestSource,
        model_func=lambda _0, _1, _2, _3, _4, _5: expected_output,
    )()
    source_builder.runtime_params.mode = runtime_params_lib.Mode.MODEL_BASED
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'foo': source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources['foo']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
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
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(profile, expected_output)

  def test_overriding_prescribed_values(self):
    """Providing prescribed values results in the correct profile."""
    geo = geometry.build_circular_geometry()
    output_shape = source_lib.ProfileType.CELL.get_profile_shape(geo)
    # Define the expected output
    expected_output = jnp.ones(output_shape)
    # Create the source
    source_builder = IonElTestSourceBuilder()
    # Prescribe the source output to something that should be equal to the
    # expected output after interpolation
    source_builder.runtime_params.mode = runtime_params_lib.Mode.PRESCRIBED
    source_builder.runtime_params.prescribed_values = {0: {0: 1, 1: 1}}
    # Build the test scenario
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {'foo': source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources['foo']
    runtime_params = general_runtime_params.GeneralRuntimeParams()
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
    profile = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    np.testing.assert_allclose(profile, expected_output)

  def test_retrieving_profile_for_affected_state(self):
    """Grabbing the correct profile works for all mesh state attributes."""
    output_shape = (2, 4)  # Some arbitrary shape.

    @dataclasses.dataclass(frozen=True)
    class TestSource(source_lib.Source):
      output_shape_getter = lambda _0: output_shape

      @property
      def source_name(self) -> str:
        return 'foo'

      @property
      def affected_core_profiles(self):
        return (
            source_lib.AffectedCoreProfile.PSI,
            source_lib.AffectedCoreProfile.NE,
        )

    profile = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])  # from get_value()
    source = TestSource(
        model_func=lambda _0, _1, _2, _3, _4, _5: profile,
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    psi_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.PSI.value, geo
    )
    np.testing.assert_allclose(psi_profile, [1, 2, 3, 4])
    ne_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.NE.value, geo
    )
    np.testing.assert_allclose(ne_profile, [5, 6, 7, 8])
    temp_ion_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.TEMP_ION.value, geo
    )
    np.testing.assert_allclose(temp_ion_profile, [0, 0, 0, 0])
    temp_el_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.TEMP_EL.value, geo
    )
    np.testing.assert_allclose(temp_el_profile, [0, 0, 0, 0])


class SingleProfileSourceTest(parameterized.TestCase):
  """Tests for SingleProfileSource."""

  def test_retrieving_profile_for_affected_state(self):
    """Grabbing the correct profile works for all mesh state attributes."""
    profile = jnp.asarray([1, 2, 3, 4])  # from get_value()

    source = test_lib.TestSource(
        model_func=lambda _0, _1, _2, _3, _4, _5: profile,
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    psi_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.PSI.value, geo
    )
    np.testing.assert_allclose(psi_profile, [0, 0, 0, 0])
    ne_profile = source.get_source_profile_for_affected_core_profile(
        profile, source_lib.AffectedCoreProfile.NE.value, geo
    )
    np.testing.assert_allclose(ne_profile, [1, 2, 3, 4])


if __name__ == '__main__':
  absltest.main()
