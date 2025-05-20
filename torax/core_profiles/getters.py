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

"""Functions for getting updated CellVariable objects for CoreProfiles."""
import functools

import jax
from jax import numpy as jnp
from torax import array_typing
from torax import jax_utils
from torax.config import numerics
from torax.config import profile_conditions
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.physics import charge_states
from torax.physics import formulas

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def get_updated_ion_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed ion temperature profiles."""
  return _updated_temperature(
      geo.drho_norm,
      dynamic_profile_conditions.Ti,
      dynamic_profile_conditions.Ti_bound_left,
      dynamic_profile_conditions.Ti_bound_left_is_grad,
      dynamic_profile_conditions.Ti_bound_right,
      dynamic_profile_conditions.Ti_bound_right_is_grad,
      'Ti',
  )


def get_updated_electron_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron temperature profiles."""
  return _updated_temperature(
    geo.drho_norm,
    dynamic_profile_conditions.Te,
    dynamic_profile_conditions.Te_bound_left,
    dynamic_profile_conditions.Te_bound_left_is_grad,
    dynamic_profile_conditions.Te_bound_right,
    dynamic_profile_conditions.Te_bound_right_is_grad,
    'Te',
  )


def _updated_temperature(
    drho_norm: array_typing.ScalarFloat,
    value: array_typing.ArrayFloat,
    bound_left: array_typing.ScalarFloat,
    bound_left_is_grad: bool,
    bound_right: array_typing.ScalarFloat,
    bound_right_is_grad: bool,
    name: str,
) -> cell_variable.CellVariable:
  """Helper method for getting updated temperature profiles."""
  left_constraint = _ensure_value_boundary_is_positive(bound_left, bound_left_is_grad, name)
  right_constraint = _ensure_value_boundary_is_positive(bound_right, bound_right_is_grad, name)
  return cell_variable.CellVariable(
      value=value,
      dr=drho_norm,
      left_face_constraint=left_constraint,
      left_face_constraint_is_grad=bound_left_is_grad,
      right_face_constraint=right_constraint,
      right_face_constraint_is_grad=bound_right_is_grad,
  )


def _ensure_value_boundary_is_positive(bound: jax.Array, bound_is_grad: bool, name: str) -> jax.Array:
  return jax_utils.error_if(
      bound,
      (jnp.min(bound) <= 0) & (~bound_is_grad),
      name,
  )


def get_updated_electron_density(
    dynamic_numerics: numerics.DynamicNumerics,
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron density profiles."""

  nGW = (
      dynamic_profile_conditions.Ip_tot
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_numerics.nref
  )
  ne_value = jnp.where(
      dynamic_profile_conditions.ne_is_fGW,
      dynamic_profile_conditions.ne * nGW,
      dynamic_profile_conditions.ne,
  )
  ne_bound_left = jnp.where(
      dynamic_profile_conditions.ne_bound_left_is_fGW,
      dynamic_profile_conditions.ne_bound_left * nGW,
      dynamic_profile_conditions.ne_bound_left,
  )
  ne_bound_right = jnp.where(
      dynamic_profile_conditions.ne_bound_right_is_fGW,
      dynamic_profile_conditions.ne_bound_right * nGW,
      dynamic_profile_conditions.ne_bound_right,
  )

  if dynamic_profile_conditions.normalize_to_nbar:
    face_left = jnp.where(dynamic_profile_conditions.ne_bound_left_is_grad, ne_value[0], ne_bound_left)
    face_right = jnp.where(dynamic_profile_conditions.ne_bound_right_is_grad, ne_value[-1], ne_bound_right)
    face_inner = (ne_value[..., :-1] + ne_value[..., 1:]) / 2.0
    ne_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    # Find normalization factor such that desired line-averaged ne is set.
    # Line-averaged electron density (nbar) is poorly defined. In general, the
    # definition is machine-dependent and even shot-dependent since it depends
    # on the usage of a specific interferometry chord. Furthermore, even if we
    # knew the specific chord used, its calculation would depend on magnetic
    # geometry information beyond what is available in StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        dynamic_profile_conditions.ne_is_fGW,
        dynamic_profile_conditions.nbar * nGW,
        dynamic_profile_conditions.nbar,
    )

    left_is_absolute = dynamic_profile_conditions.ne_bound_left_is_absolute
    right_is_absolute = dynamic_profile_conditions.ne_bound_right_is_absolute

    start_idx = 1 if left_is_absolute else 0
    end_idx = -1 if right_is_absolute else None

    inner_nbar = _trapz(ne_face[start_idx:end_idx], geo.Rout_face[start_idx:end_idx]) / Rmin_out

    numerator_change = 0
    denominator_change = 0

    if left_is_absolute:
      numerator_change += ne_face[0] * (geo.Rout_face[1] - geo.Rout_face[0])
      denominator_change += ne_face[1] * (geo.Rout_face[1] - geo.Rout_face[0])
    if right_is_absolute:
      numerator_change += ne_face[-1] * (geo.Rout_face[-1] - geo.Rout_face[-2])
      denominator_change += ne_face[-2] * (geo.Rout_face[-1] - geo.Rout_face[-2])

    C = ((target_nbar- 0.5 * numerator_change / Rmin_out) /
         (inner_nbar + 0.5 * denominator_change / Rmin_out))

    if not left_is_absolute:
      ne_bound_left *= C
    if not right_is_absolute:
      ne_bound_right *= C
  else:
    C = 1

  ne_value = C * ne_value

  # TODO: Add tests to check that the left and right boundaries are correct in the new cases.

  if dynamic_profile_conditions.normalize_to_nbar:
    # Verify that the line integrated value is correct
    face_left = jnp.where(dynamic_profile_conditions.ne_bound_left_is_grad, ne_value[0], ne_bound_left)
    face_right = jnp.where(dynamic_profile_conditions.ne_bound_right_is_grad, ne_value[-1], ne_bound_right)
    face_inner = (ne_value[..., :-1] + ne_value[..., 1:]) / 2.0
    ne_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
    actual_nbar = _trapz(ne_face, geo.Rout_face) / Rmin_out
    target_nbar = jnp.where(
      dynamic_profile_conditions.ne_is_fGW,
      dynamic_profile_conditions.nbar * nGW,
      dynamic_profile_conditions.nbar,
    )
    diff = actual_nbar - target_nbar
    ne_value = jax_utils.error_if(ne_value, diff > 1e-6, 'nbar mismatch')

  ne = cell_variable.CellVariable(
      value=ne_value,
      dr=geo.drho_norm,
      left_face_constraint=ne_bound_left,
      left_face_constraint_is_grad=dynamic_profile_conditions.ne_bound_left_is_grad,
      right_face_constraint=ne_bound_right,
      right_face_constraint_is_grad=dynamic_profile_conditions.ne_bound_right_is_grad,
  )
  return ne


# jitted since also used outside the stepper
@functools.partial(
    jax_utils.jit, static_argnames=['static_runtime_params_slice']
)
def get_ion_density_and_charge_states(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    ne: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
) -> tuple[
    cell_variable.CellVariable,
    cell_variable.CellVariable,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
]:
  """Updated ion densities based on state.

  Main ion and impurities are each treated as a single effective ion, but could
  be comparised of multiple species within an IonMixture. The main ion and
  impurity densities are calculated depending on the Zeff constraint,
  quasineutrality, and the average impurity charge state which may be
  temperature dependent.

  Zeff = (Zi**2 * ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni*Zi = ne

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry of the tokamak.
    ne: Electron density profile [nref].
    temp_el: Electron temperature profile [keV].

  Returns:
    ni: Ion density profile [nref].
    nimp: Impurity density profile [nref].
    Zi: Average charge state of main ion on cell grid [amu].
      Typically just the average of the atomic numbers since these are normally
      low Z ions and can be assumed to be fully ionized.
    Zi_face: Average charge state of main ion on face grid [amu].
    Zimp: Average charge state of impurities on cell grid [amu].
    Zimp_face: Average charge state of impurities on face grid [amu].
  """

  Zi, Zi_face, Zimp, Zimp_face = _get_charge_states(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      temp_el,
  )

  Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff
  Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face

  dilution_factor = formulas.calculate_main_ion_dilution_factor(Zi, Zimp, Zeff)
  dilution_factor_inner_edge = formulas.calculate_main_ion_dilution_factor(
      Zi_face[0], Zimp_face[0], Zeff_face[0]
  )
  dilution_factor_outer_edge = formulas.calculate_main_ion_dilution_factor(
      Zi_face[-1], Zimp_face[-1], Zeff_face[-1]
  )

  # Assume that Zeff varies slowly across the plasma, so that the gradient of
  # ni and nimp are simply proportional to the gradient of ne
  ni = cell_variable.CellVariable(
      value=ne.value * dilution_factor,
      dr=geo.drho_norm,
      left_face_constraint=ne.left_face_constraint * dilution_factor_inner_edge,
      left_face_constraint_is_grad=ne.left_face_constraint_is_grad,
      right_face_constraint=ne.right_face_constraint * dilution_factor_outer_edge,
      right_face_constraint_is_grad=ne.right_face_constraint_is_grad,
  )

  nimp = cell_variable.CellVariable(
      value=(ne.value - ni.value * Zi) / Zimp,
      dr=geo.drho_norm,
      left_face_constraint=(
          ne.left_face_constraint - ni.left_face_constraint * dilution_factor_inner_edge
      ) / Zimp_face[0],
      left_face_constraint_is_grad=ne.left_face_constraint_is_grad,
      right_face_constraint=(
          ne.right_face_constraint - ni.right_face_constraint * Zi_face[-1]
      ) / Zimp_face[-1],
      right_face_constraint_is_grad=ne.right_face_constraint_is_grad,
  )
  return ni, nimp, Zi, Zi_face, Zimp, Zimp_face


def _get_charge_states(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    temp_el: cell_variable.CellVariable,
) -> tuple[
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
]:
  """Updated charge states based on IonMixtures and electron temperature."""
  Zi = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      Te=temp_el.value,
  )
  Zi_face = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      Te=temp_el.face_value(),
  )

  Zimp = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=temp_el.value,
  )
  Zimp_face = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=temp_el.face_value(),
  )

  return Zi, Zi_face, Zimp, Zimp_face
