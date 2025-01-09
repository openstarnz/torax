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
"""Surrogate model for ion-cyclotron resonance heating (ICRH) model."""
from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
from typing import Any, ClassVar, Final, Sequence

import chex
import flax.linen as nn
import jax
from jax import numpy as jnp
from jax.scipy import integrate
import jaxtyping as jt
import numpy as np
from torax import array_typing
from torax import interpolated_param
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models
from typing_extensions import override

# Internal import.


# Environment variable for the TORIC NN model. Used if the model path
# is not set in the config.
_MODEL_PATH_ENV_VAR: Final[str] = 'TORIC_NN_MODEL_PATH'
# If no path is set in either the config or the environment variable, use
# this path.
_DEFAULT_MODEL_PATH = '~/toric_surrogate/TORIC_MLP_v1/toricnn.json'
_TORIC_GRID_SIZE = 297
_HELIUM3_ID = 'He3'
_TRITIUM_SECOND_HARMONIC_ID = '2T'
_ELECTRON_ID = 'e'


def _get_default_model_path() -> str:
  return os.environ.get(_MODEL_PATH_ENV_VAR, _DEFAULT_MODEL_PATH)


def _from_json(json_file) -> dict[str, Any]:
  """Load the model config and weights from a JSON file."""
  with open(json_file) as file_:
    model_dict = json.load(file_)
  return model_dict


# pylint: disable=invalid-name
# Many of the variables below are named to match the physics quantities
# as defined by the TORIC ICRF solver, so we keep their naming for consistency.
@chex.dataclass(frozen=True)
class ToricNNInputs:
  """Inputs to the ToricNN model."""

  # ICRF wave frequency in MHz, training range = 119 to 121.
  frequency: array_typing.ScalarFloat
  # Volume average temperature in keV, training range = 1.5 to 8.5.
  volume_average_temperature: array_typing.ScalarFloat
  # Volume average density in 10^20 m^-3, training range = 1.1 to 5.1.
  volume_average_density: array_typing.ScalarFloat
  # He3 minority concentration relative to the electron density in %,
  # training range = 1 to 5.
  minority_concentration: array_typing.ScalarFloat
  # Distance from last closed flux surface (LCFS) to the inner wall in m,
  # training range = 0 to 0.03.
  gap_inner: array_typing.ScalarFloat
  # Distance from LCFS to the outer midplane limiter in m,
  # training range = 0 to 0.05.
  gap_outer: array_typing.ScalarFloat
  # Vertical position of magnetic axis in m, training range = -0.05 to 0.05.
  z0: array_typing.ScalarFloat
  # Temperature profile peaking factor, training range = 2 to 3.
  temperature_peaking_factor: array_typing.ScalarFloat
  # Density profile peaking factor, training range = 1.15 to 1.65.
  density_peaking_factor: array_typing.ScalarFloat
  # Toroidal magnetic field on axis in T, training range = 11.8 to 12.5.
  B0: array_typing.ScalarFloat


@chex.dataclass(frozen=True)
class ToricNNOutputs:
  """Outputs from the ToricNN model."""

  # Power deposition on helium-3 in MW/m^3/MW_{abs}.
  power_deposition_He3: array_typing.ArrayFloat
  # Power deposition on tritium (second harmonic) in MW/m^3/MW_{abs}.
  power_deposition_2T: array_typing.ArrayFloat
  # Power deposition on electrons in MW/m^3/MW_{abs}.
  power_deposition_e: array_typing.ArrayFloat


class ToricNNWrapper:
  """Wrapper for the Toric NN model.

  This wrapper is currently for a SPARC-specific ion cyclotron resosonanc
  heating scheme.

  TODO(b/378072116): Make the wrapper more general to work with other ICRH
  schemes and surrogate models.

  This wrapper is the main interface for interacting with the Toric NN model.
  for making predictions of heating power deposition profiles given
  `ToricNNInputs`.

  The wrapper constructs 3 separate instances of the `_ToricNN` class, one for
  each simulated output (Helium-3, 2nd-harmonic tritium and electrons).
  """

  def __init__(self, path: str | None = None):
    if path is None:
      path = _get_default_model_path()
    logging.info('Loading ToricNN model from %s', path)
    model_config = _from_json(path)
    self.model_config = model_config

    self._params = {}
    self._power_deposition_network = self._load_network()
    self._power_deposition_He3_params = self._load_params(_HELIUM3_ID)
    self._power_deposition_2T_params = self._load_params(
        _TRITIUM_SECOND_HARMONIC_ID
    )
    self._power_deposition_e_params = self._load_params(_ELECTRON_ID)
    logging.info('Loaded ToricNN model from %s', path)

  def _load_network(self) -> _ToricNN:
    return _ToricNN(
        hidden_sizes=self.model_config['hidden_sizes'],
        pca_coeffs=self.model_config['pca_coeffs'],
        input_dim=self.model_config['input_dim'],
        radial_nodes=self.model_config['radial_nodes'],
    )

  def _load_params(self, network_name: str) -> dict[str, Any]:
    """Load a ToricNN network and its parameters."""
    params = {}
    params['params'] = self.model_config[f'{network_name}']
    for i in range(len(self.model_config['hidden_sizes']) + 1):
      params['params'][f'Dense_{i}']['kernel'] = np.array(
          self.model_config[f'{network_name}'][f'Dense_{i}']['kernel']
      )
      params['params'][f'Dense_{i}']['bias'] = np.array(
          self.model_config[f'{network_name}'][f'Dense_{i}']['bias']
      )
    params['params']['pca_components'] = np.array(
        self.model_config[f'{network_name}']['pca_components']
    )
    params['params']['pca_mean'] = np.array(
        self.model_config[f'{network_name}']['pca_mean']
    )
    params['params']['scaler_mean'] = np.array(
        self.model_config[f'{network_name}']['scaler_mean']
    )
    params['params']['scaler_scale'] = np.array(
        self.model_config[f'{network_name}']['scaler_scale']
    )
    return params

  def predict(self, inputs: ToricNNInputs) -> ToricNNOutputs:
    """Make a prediction given the inputs."""
    inputs = jnp.array([
        inputs.frequency,
        inputs.volume_average_temperature,
        inputs.volume_average_density,
        inputs.minority_concentration,
        inputs.gap_inner,
        inputs.gap_outer,
        inputs.z0,
        inputs.temperature_peaking_factor,
        inputs.density_peaking_factor,
        inputs.B0,
    ])
    outputs_He3 = self._power_deposition_network.apply(
        self._power_deposition_He3_params, inputs
    )
    outputs_2T = self._power_deposition_network.apply(
        self._power_deposition_2T_params, inputs
    )
    outputs_e = self._power_deposition_network.apply(
        self._power_deposition_e_params, inputs
    )
    return ToricNNOutputs(
        power_deposition_He3=outputs_He3,
        power_deposition_2T=outputs_2T,
        power_deposition_e=outputs_e,
    )


# pylint: enable=invalid-name


class _ToricNN(nn.Module):
  """Surrogate heating model trained on TORIC ICRF solver simulation.

  This model takes input parameters from the `ToricNNInputs` class and outputs
  power deposition profiles for helium-3, tritium (second harmonic) and
  electrons on a radial grid.

  This Flax module is not intended to be used directly but rather through the
  `ToricNNWrapper` class.

  The modelling approach is described in:
  https://iopscience.iop.org/article/10.1088/1741-4326/ad645d/pdf. The model
  is trained on regression outputs from the TORIC ICRF solver. PCA is applied
  to the outputs of the solver to reduce the dimensionality of the model.

  The structure of the model consistents of:
  - Scaling and normalisation of the input parameters.
  - An MLP transforming the scaled inputs.
  - A projection back to true values using the PCA coefficients.
  """

  # Hidden layer sizes for the MLP.
  hidden_sizes: Sequence[int]
  # Number of PCA coefficients used by ToricNN.
  pca_coeffs: int
  # Input dimensionality of the ToricNN model.
  input_dim: int
  # Number of radial nodes in output of the ToricNN model.
  radial_nodes: int

  def setup(self):
    """Setup the parameters of the ToricNN model."""
    self.scaler_mean = self.param(
        'scaler_mean',
        jax.random.normal,
        (self.input_dim,),
    )
    self.scaler_scale = self.param(
        'scaler_scale',
        jax.random.normal,
        (self.input_dim,),
    )
    self.pca_components = self.param(
        'pca_components',
        jax.random.normal,
        (
            self.pca_coeffs,
            self.radial_nodes,
        ),
    )
    self.pca_mean = self.param(
        'pca_mean',
        jax.random.normal,
        (self.radial_nodes,),
    )

  @nn.compact
  def __call__(
      self,
      x: jt.Float32[jt.Array, 'B* {self.input_dim}'],
  ) -> jt.Float32[jt.Array, 'B* {self.radial_nodes}']:
    """Run a forward pass of the ToricNN model."""
    # Scale and normalise inputs.
    x = (x - self.scaler_mean) / self.scaler_scale

    # MLP.
    for hidden_size in self.hidden_sizes:
      x = nn.Dense(
          hidden_size,
      )(x)
      x = nn.relu(x)
    x = nn.Dense(
        self.pca_coeffs,
    )(x)

    x = x @ self.pca_components + self.pca_mean  # Project back to true values.
    x = x * (x > 0)  # Eliminate non-physical values for power deposition.
    return x


# pylint: disable=invalid-name
# Several variable names below follow physics notation matching so don't adhere
# to the lint guide.
@dataclasses.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the ion cyclotron resonance source."""

  # Inner radial location of first wall at plasma midplane level [m].
  wall_inner: float = 1.24
  # Outer radial location of first wall at plasma midplane level [m].
  wall_outer: float = 2.43
  # ICRF wave frequency [Hz].
  frequency: runtime_params_lib.TimeInterpolatedInput = 120e6
  # He3 minority concentration relative to the electron density in %.
  minority_concentration: runtime_params_lib.TimeInterpolatedInput = 3.0
  # Total heating power [W].
  Ptot: runtime_params_lib.TimeInterpolatedInput = 120e6
  # Mode of the source.
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @override
  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> 'RuntimeParamsProvider':
    kwargs = self.get_provider_kwargs(torax_mesh)
    return RuntimeParamsProvider(**kwargs)


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams
  frequency: interpolated_param.InterpolatedVarSingleAxis
  minority_concentration: interpolated_param.InterpolatedVarSingleAxis
  Ptot: interpolated_param.InterpolatedVarSingleAxis

  @override
  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  frequency: array_typing.ScalarFloat
  minority_concentration: array_typing.ScalarFloat
  Ptot: array_typing.ScalarFloat
  wall_inner: float
  wall_outer: float


def _helium3_tail_temperature(
    power_deposition_he3: jax.Array,
    core_profiles: state.CoreProfiles,
    minority_concentration: float,
    Ptot: float,
) -> jax.Array:
  """Use a "Stix distribution" to estimate the tail temperature of He3."""
  helium3_mass = 3.016
  helium3_charge = 2
  helium3_fraction = minority_concentration / 100  # Min conc provided in %.
  absorbed_power_density = power_deposition_he3 * Ptot
  # Use a "Stix distribution" [Stix, Nuc. Fus. 1975] to model the non-thermal
  # He3 distribution based on an analytic solution to the FP equation.
  epsilon = (
      0.24
      * jnp.sqrt(core_profiles.temp_el.value)
      * helium3_mass
      * absorbed_power_density
  ) / (core_profiles.ne.value**2 * helium3_charge**2 * helium3_fraction)
  return core_profiles.temp_el.value * (1 + epsilon)


def icrh_model_func(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_source_models: source_models.SourceModels | None,
    toric_nn: ToricNNWrapper,
) -> jax.Array:
  """Compute ion/electron heat source terms."""
  del (
      unused_source_models,
      static_runtime_params_slice,
  )  # Unused.
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

  # Construct inputs for ToricNN.
  volume = integrate.trapezoid(geo.vpr_face, geo.rho_face_norm)
  volume_average_temperature = (
      integrate.trapezoid(
          core_profiles.temp_el.face_value() * geo.vpr_face,
          geo.rho_face_norm,
      )
      / volume
  )
  volume_average_density = (
      integrate.trapezoid(
          core_profiles.ne.face_value() * geo.vpr_face,
          geo.rho_face_norm,
      )
      / volume
  )
  # Peaking factors are core w.r.t volume averages.
  temperature_peaking_factor = (
      core_profiles.temp_el.value[0] / volume_average_temperature
  )
  density_peaking_factor = core_profiles.ne.value[0] / volume_average_density
  Router = geo.Rmaj + geo.Rmin
  Rinner = geo.Rmaj - geo.Rmin
  # Assumption: inner and outer gaps are not functions of z0.
  # This is a good assumption for the inner gap but perhaps less good for the
  # outer gap where there is significant curvature to the outer limiter.
  gap_inner = Rinner - dynamic_source_runtime_params.wall_inner
  gap_outer = dynamic_source_runtime_params.wall_outer - Router
  toric_inputs = ToricNNInputs(
      frequency=dynamic_source_runtime_params.frequency,
      volume_average_temperature=volume_average_temperature,
      volume_average_density=volume_average_density,
      minority_concentration=dynamic_source_runtime_params.minority_concentration,
      gap_inner=gap_inner,
      gap_outer=gap_outer,
      z0=geo.z_magnetic_axis,
      temperature_peaking_factor=temperature_peaking_factor,
      density_peaking_factor=density_peaking_factor,
      B0=geo.B0,
  )

  toric_nn_outputs = toric_nn.predict(toric_inputs)
  toric_grid = jnp.linspace(0.0, 1.0, _TORIC_GRID_SIZE)

  # Ideally total ICRH power should equal one but normalise if not.
  power_deposition_he3 = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_He3,
  )
  power_deposition_e = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_e,
  )
  power_deposition_2T = jnp.interp(
      geo.torax_mesh.cell_centers,
      toric_grid,
      toric_nn_outputs.power_deposition_2T,
  )
  power_deposition_all = (
      power_deposition_2T + power_deposition_e + power_deposition_he3
  )
  # An implicit integration is being done here (using the trapezoid rule)
  total_power_deposition = jnp.sum(
      power_deposition_all * geo.vpr * geo.drho_norm
  )
  power_deposition_he3 /= total_power_deposition
  power_deposition_e /= total_power_deposition
  power_deposition_2T /= total_power_deposition

  # For helium-3 we use a "Stix distribution" to model the non-thermal He3 tail.
  helium3_birth_energy = _helium3_tail_temperature(
      power_deposition_he3,
      core_profiles,
      dynamic_source_runtime_params.minority_concentration,
      dynamic_source_runtime_params.Ptot / 1e6,  # required in MW.
  )
  helium3_mass = 3.016
  frac_ion_heating = physics.fast_ion_fractional_heating_formula(
      helium3_birth_energy,
      core_profiles.temp_el.value,
      helium3_mass,
  )
  source_ion = (
      power_deposition_he3
      * frac_ion_heating
      * dynamic_source_runtime_params.Ptot
  )
  source_el = (
      power_deposition_he3
      * (1 - frac_ion_heating)
      * dynamic_source_runtime_params.Ptot
  )

  # Assume that all the power from the electron power profile goes to electrons.
  source_el += power_deposition_e * dynamic_source_runtime_params.Ptot

  # Assume that all the power from the tritium power profile goes to ions.
  source_ion += power_deposition_2T * dynamic_source_runtime_params.Ptot

  return jnp.stack([source_ion, source_el])
# pylint: enable=invalid-name


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class IonCyclotronSource(source.Source):
  """Ion cyclotron source with surrogate model."""

  SOURCE_NAME: ClassVar[str] = 'ion_cyclotron_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'icrh_model_func'

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    )

  @property
  def output_shape_getter(self) -> source.SourceOutputShapeFunction:
    return source.get_ion_el_output_shape


@dataclasses.dataclass(kw_only=True, frozen=False)
class IonCyclotronSourceBuilder:
  """Builder for the IonCyclotronSource."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )
  links_back: bool = False
  model_func: source.SourceProfileFunction | None = None

  def __post_init__(self):
    if self.model_func is None:
      self.model_func = functools.partial(
          icrh_model_func,
          toric_nn=ToricNNWrapper(),
      )

  def __call__(
      self,
  ) -> IonCyclotronSource:

    return IonCyclotronSource(
        model_func=self.model_func,
    )
