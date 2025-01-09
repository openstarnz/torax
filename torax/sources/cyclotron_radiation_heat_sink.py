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

# Many variables throughout this function are capitalized based on physics
# notational conventions rather than on Google Python style
# pylint: disable=invalid-name

"""Cyclotron radiation heat sink for electron heat equation.."""

import dataclasses
from typing import ClassVar

import chex
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models as source_models_lib


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the cyclotron radiation heat sink, updating the parent class."""

  # The wall reflection coefficient is a machine-dependent dimensionless
  # parameter corresponding to the fraction of cyclotron radiation reflected
  # off the wall and back into the plasma where it is re-absorbed.
  # The default value is a typical value.
  wall_reflection_coeff: float = 0.9

  # The beta parameter is used in the parameterized function for the
  # temperature fit. beta_min, beta_max, and beta_grid_size are used for a
  # grid search to find the best fit.
  beta_min: float = 0.5
  beta_max: float = 8.0
  beta_grid_size: int = 32
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))

  def build_static_params(self) -> 'StaticRuntimeParams':
    return StaticRuntimeParams(
        mode=self.mode.value,
        is_explicit=self.is_explicit,
        beta_min=self.beta_min,
        beta_max=self.beta_max,
        beta_grid_size=self.beta_grid_size,
    )


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(
        **self.get_dynamic_params_kwargs(t, StaticRuntimeParams)
    )


@chex.dataclass(frozen=True)
class StaticRuntimeParams(runtime_params_lib.StaticRuntimeParams):
  beta_min: float
  beta_max: float
  beta_grid_size: int


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  wall_reflection_coeff: array_typing.ScalarFloat


def _alpha_closed_form(
    *,
    beta: array_typing.ScalarFloat,
    rho_norm: array_typing.ArrayFloat,
    profile_data: array_typing.ArrayFloat,
    profile_edge_value: array_typing.ScalarFloat,
) -> array_typing.ScalarFloat:
  """Returns analytical closed form of alpha for parameterized profiles.

  See cyclotron_radiation_albajar for more details.

  We find alpha for the best fit for either the ne_data or te_data to the
  parameterized functions:

  ne = ne_data[0]*(1 - rhonorm**2)**alpha
  te = (te_data[0] - te_edge)*(1 - rhonorm**beta)**alpha + te_edge

  Where ne_edge = 0 above. The parameterizations are from the Albajar paper.

  The fit is from the magnetic axis to rhonorm=0.9, to avoid pedestal effects.

  We analytically solve for alpha by taking the derivative of the loss function
  with respect to alpha.

  Defining:
  T = (profile_data - profile_edge_data)/(profile_data[0] - profile_edge_data)

  The loss function is:
  loss = sum((log(T) -  alpha * log(1 - rhonorm**beta)) ** 2) / 2

  The solution is provided by setting d loss / d alpha = 0.

  Args:
    beta: The beta parameter to use in the parameterized functions. For the
      density fit, this is always 2.
    rho_norm: Normalized toroidal flux coordinate.
    profile_data: The profile data to be fit (i.e. density or temperature).
    profile_edge_value: The value of profile data at the edge to use. This can't
      be taken from profile_data since there are different assumptions for the
      temperature and density fits.

  Returns:
    The alpha parameter being fit for either the density or temperature fits.
  """
  # To avoid dealing with slicing of non-concrete values, we do a masking trick
  # where we replace the values of ne_data and rhonorm above 0.9 with values
  # that will not contribute to the sums in the numerator and denominator.

  mask = rho_norm < 0.9
  profile_data_norm = (profile_data - profile_edge_value) / (
      profile_data[0] - profile_edge_value
  )
  sliced_profile_data_norm = jnp.where(mask, profile_data_norm, 1.0)
  sliced_rhonorm = jnp.where(mask, rho_norm, 0.0)

  num = jnp.sum(
      (jnp.log(sliced_profile_data_norm) * jnp.log(1 - sliced_rhonorm**beta))
  )
  den = jnp.sum(jnp.log(1 - sliced_rhonorm**beta) ** 2)

  alpha_n = num / den
  return alpha_n


def _loss_for_beta_t(
    beta_t: array_typing.ScalarFloat,
    rho_norm: array_typing.ArrayFloat,
    te_data: array_typing.ArrayFloat,
) -> array_typing.ScalarFloat:
  """Returns the loss function for the temperature fit for a given beta_t.

  The fit is from the magnetic axis to rhonorm=0.9, to avoid pedestal effects.
  alpha_t is calculated with an analytical closed form solution.

  Args:
    beta_t: The beta parameter to use in the parameterized functions.
    rho_norm: Normalized toroidal flux coordinate.
    te_data: The temperature data to be fit, assumed to be on the face grid.

  Returns:
    The loss function for the temperature fit for a given beta_t.
  """
  alpha_t = _alpha_closed_form(
      profile_data=te_data,
      profile_edge_value=te_data[-1],
      rho_norm=rho_norm,
      beta=beta_t,
  )
  return _te_loss_fn(
      alpha_t=alpha_t,
      beta_t=beta_t,
      rho_norm=rho_norm,
      te_data=te_data,
  )


def _te_pred_fn(
    *,
    alpha_t: array_typing.ScalarFloat,
    beta_t: array_typing.ScalarFloat,
    rho_norm: array_typing.ArrayFloat,
    te_data: array_typing.ArrayFloat,
) -> array_typing.ArrayFloat:
  return (te_data[0] - te_data[-1]) * (
      (1 - rho_norm**beta_t)
  ) ** alpha_t + te_data[-1]


def _te_loss_fn(
    *,
    alpha_t: array_typing.ScalarFloat,
    beta_t: array_typing.ScalarFloat,
    rho_norm: array_typing.ArrayFloat,
    te_data: array_typing.ArrayFloat,
) -> array_typing.ScalarFloat:
  """Returns the loss function for the temperature fit.

  The fit is from the magnetic axis to rhonorm=0.9, to avoid pedestal effects.
  The pedestal will not extend to rhonorm=0.9. The choice of this value is from
  the Artaud paper.

  Args:
    alpha_t: The alpha parameter to use in the parameterized functions.
    beta_t: The beta parameter to use in the parameterized functions.
    rho_norm: Normalized toroidal flux coordinate.
    te_data: The temperature data to be fit, assumed to be on the face grid.

  Returns:
    The loss function for the temperature fit.
  """
  te_pred = _te_pred_fn(
      alpha_t=alpha_t,
      beta_t=beta_t,
      rho_norm=rho_norm,
      te_data=te_data,
  )
  mask = rho_norm < 0.9
  sliced_diff = jnp.where(mask, te_pred - te_data, 0.0)
  return jnp.sum(sliced_diff**2) / 2


def _solve_alpha_t_beta_t_grid_search(
    *,
    rho_norm: array_typing.ArrayFloat,
    te_data: array_typing.ArrayFloat,
    beta_scan_parameters: tuple[float, float, int],
) -> tuple[array_typing.ScalarFloat, array_typing.ScalarFloat]:
  """Returns the alpha and beta parameters that minimize the temperature loss function.

  Grid search is used for computational efficiency.

  Args:
    rho_norm: Normalized toroidal flux coordinate.
    te_data: The temperature data to be fit, assumed to be on the face grid.
    beta_scan_parameters: A tuple of (beta_min, beta_max, beta_grid_size)
      parameters for the grid search.

  Returns:
    The alpha and beta parameters that minimize the temperature loss function.
  """
  beta_t_trials = jnp.linspace(
      beta_scan_parameters[0],
      beta_scan_parameters[1],
      beta_scan_parameters[2],
  )

  losses = jax.vmap(_loss_for_beta_t, in_axes=(0, None, None))(
      beta_t_trials,
      rho_norm,
      te_data,
  )
  min_index = jnp.argmin(jnp.array(losses))
  best_beta_t = beta_t_trials[min_index]
  best_alpha_t = _alpha_closed_form(
      beta=best_beta_t,
      rho_norm=rho_norm,
      profile_data=te_data,
      profile_edge_value=te_data[-1],
  )
  return best_alpha_t, best_beta_t


def cyclotron_radiation_albajar(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> array_typing.ArrayFloat:
  """Calculates the cyclotron radiation heat sink contribution to the electron heat equation.

  Total cyclotron radiation is from:
  F. Albajar et al 2001 Nucl. Fusion 41 665
  https://doi.org/10.1088/0029-5515/41/6/301

  Radial profile of the cyclotron radiation is from:
  J.F. Artaud et al 2018 Nucl. Fusion 58 105001
  https://doi.org/10.1088/1741-4326/aad5b1

  The alpha_n, alpha_T and beta_T parameters are calculated from best fits
  of the following electron density and temperature parameterization to the
  actual plasma data.

  ne = ne(0)*(1 - rhonorm**2)**alpha_n
  Te = (Te(0)-Te(a))*(1 - rhonorm**beta_T)**alpha_T + Te(a)

  Where we take a=0.9 in rhonorm space, and perform the best fit between
  0<rhonorm<0.9, to avoid pedestal effects.

  Args:
    static_runtime_params_slice: A slice of static runtime parameters.
    dynamic_runtime_params_slice: A slice of dynamic runtime parameters.
    geo: The geometry object.
    source_name: The name of the source.
    core_profiles: The core profiles object.
    source_models: Collections of source models.

  Returns:
    The cyclotron radiation heat sink contribution to the electron heat
    equation.
  """
  del (source_models,)
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  static_source_runtime_params = static_runtime_params_slice.sources[
      source_name
  ]

  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  assert isinstance(static_source_runtime_params, StaticRuntimeParams)

  # Notation conventions based on the Albajar and Artaud papers
  # pylint: disable=invalid-name

  ne20_face = core_profiles.ne.face_value() * core_profiles.nref / 1e20
  ne20_cell = core_profiles.ne.value * core_profiles.nref / 1e20

  # Dimensionless optical thickness parameter, on-axis:
  # Simplified form of omega_pe**2 / (c * omega_ce) where omega_pe is the
  # plasma frequency and omega_ce is the cyclotron frequency.
  p_a_0 = 6.04e3 * geo.Rmin * ne20_face[0] / geo.B0

  # Dimensionless correction term for aspect ratio (equation 15 in Albajar)
  G = 0.93 * (1 + 0.85 * jnp.exp(-0.82 * geo.Rmaj / geo.Rmin))

  # Calculate profile fit parameters
  alpha_n = _alpha_closed_form(
      beta=2.0,
      rho_norm=static_runtime_params_slice.torax_mesh.face_centers,
      profile_data=ne20_face,
      profile_edge_value=0.0,
  )
  beta_scan_parameters = (
      static_source_runtime_params.beta_min,
      static_source_runtime_params.beta_max,
      static_source_runtime_params.beta_grid_size,
  )
  alpha_t, beta_t = _solve_alpha_t_beta_t_grid_search(
      rho_norm=static_runtime_params_slice.torax_mesh.face_centers,
      te_data=core_profiles.temp_el.face_value(),
      beta_scan_parameters=beta_scan_parameters,
  )

  # The "profile factor" (equation 13 in Albajar)
  K = (
      (alpha_n + 3.87 * alpha_t + 1.46) ** -0.79
      * (1.98 + alpha_t) ** 1.36
      * beta_t**2.14
      * (beta_t**1.53 + 1.87 * alpha_t - 0.16) ** -1.33
  )

  # Calculate power loss in [W]
  P_cycl_total = (
      3.84e-2
      * jnp.sqrt(1 - dynamic_source_runtime_params.wall_reflection_coeff)
      * geo.Rmaj
      * geo.Rmin**1.38
      * geo.elongation_face[-1] ** 0.79
      * geo.B0**2.62
      * ne20_face[0] ** 0.38
      * core_profiles.temp_el.face_value()[0]
      * (16 + core_profiles.temp_el.face_value()[0]) ** 2.61
      * (1 + 0.12 * core_profiles.temp_el.face_value()[0] / p_a_0**0.41)
      ** -1.51
      * K
      * G
  )

  # Calculate the radial profile on the cell grid,
  # according to the Artaud formula (A.45)
  Q_cycl_shape = (
      geo.Rmaj
      * geo.elongation**0.79
      * (geo.F / geo.Rmaj) ** 2.62
      * ne20_cell**0.38
      * core_profiles.temp_el.value**3.61
  )

  # Scale the profile shape to match the total integrated power loss
  denom = math_utils.cell_integration(Q_cycl_shape * geo.vpr, geo)
  rescaling_factor = P_cycl_total / denom
  Q_cycl = Q_cycl_shape * rescaling_factor

  return -Q_cycl


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CyclotronRadiationHeatSink(source.Source):
  """Cyclotron radiation heat sink for electron heat equation."""

  SOURCE_NAME: ClassVar[str] = 'cyclotron_radiation_heat_sink'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'cyclotron_radiation_albajar'
  model_func: source.SourceProfileFunction = cyclotron_radiation_albajar

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL,)
