from jax import numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.transport_model import transport_model
from torax.fvm.calc_coeffs import calc_eta, calc_d, calc_particle_flux, calc_temp_flux
from torax.constants import CONSTANTS
from torax.transport_model import runtime_params as runtime_params_lib
from torax.physics import collisions
import chex


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  b_n: float
  c_n: float
  b_te: float
  c_te: float
  b_ti: float
  c_ti: float


class CriticalGradientDipoleModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_outputs: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    assert isinstance(dynamic_runtime_params_slice.transport, DynamicRuntimeParams)

    nref = dynamic_runtime_params_slice.numerics.nref

    # Classical
    ne_classical_flux, Te_classical_flux, Ti_classical_flux = self._calculate_classical_flux(core_profiles, geo, nref)
    ne_classical_flux = jnp.nan_to_num(ne_classical_flux, nan=0.0)
    Te_classical_flux = jnp.nan_to_num(Te_classical_flux, nan=0.0)
    Ti_classical_flux = jnp.nan_to_num(Ti_classical_flux, nan=0.0)

    # Turbulent
    ne_turbulent_flux, Te_turbulent_flux, Ti_turbulent_flux = self._calculate_turbulent_flux(
        core_profiles, geo, dynamic_runtime_params_slice.transport
    )

    # Convert proper flux into transport coefficients
    ne_flux = ne_classical_flux + ne_turbulent_flux
    temp_el_flux = Te_classical_flux + Te_turbulent_flux
    temp_ion_flux = Ti_classical_flux + Ti_turbulent_flux
    coeffs = self._convert_flux_to_coeffs(ne_flux, temp_el_flux, temp_ion_flux, nref, core_profiles, geo)

    actual_ne_flux = calc_particle_flux(
      geo, core_profiles.ne, coeffs.v_face_el, coeffs.d_face_el
    ) * nref
    actual_temp_el_flux = calc_temp_flux(
      geo, core_profiles.ne, core_profiles.temp_el, coeffs.v_face_el, coeffs.d_face_el, coeffs.chi_face_el
    ) * nref * CONSTANTS.keV2J
    actual_temp_ion_flux = calc_temp_flux(
      geo, core_profiles.ne, core_profiles.temp_el, coeffs.v_face_el, coeffs.d_face_el, coeffs.chi_face_ion
    ) * nref * CONSTANTS.keV2J

    return coeffs

  def _calculate_classical_flux(self, core_profiles: state.CoreProfiles, geo: geometry.Geometry, nref):
    n = core_profiles.ne.face_value() * nref
    Te = core_profiles.temp_el.face_value() * CONSTANTS.keV2J
    Ti = core_profiles.temp_ion.face_value() * CONSTANTS.keV2J
    Te_grad = core_profiles.temp_el.face_grad() * CONSTANTS.keV2J
    Ti_grad = core_profiles.temp_ion.face_grad() * CONSTANTS.keV2J
    e = CONSTANTS.qe
    psi_grad = core_profiles.psi.face_grad()
    g4 = geo.g4_face
    p_grad = (
        core_profiles.ne.face_grad() * Te
        + core_profiles.ne.face_value() * Te_grad
        + core_profiles.ni.face_grad() * Ti
        + core_profiles.ni.face_value() * Ti_grad
    ) * nref

    coulomb_log = collisions._calculate_lambda_ei(Te / CONSTANTS.keV2J, n)
    prefactor = 12 * jnp.pi**(3/2) / jnp.sqrt(2.0) * CONSTANTS.epsilon0**2 / n / e**4 / coulomb_log
    tau_ee = prefactor * jnp.sqrt(CONSTANTS.me) * Te**(3/2)
    tau_ii = prefactor * jnp.sqrt(CONSTANTS.mp) * Ti**(3/2)

    tau_e = tau_ee
    tau_i = jnp.sqrt(2.0) * tau_ii

    el_heat_flux = 4 * jnp.pi**2 * CONSTANTS.me * Te / tau_e / e**2 / psi_grad**2 * (3/2*p_grad - 4.66 * n * Te_grad) * g4
    ion_heat_flux = -8 * jnp.pi**2 * n * Ti * CONSTANTS.mp / tau_i / e**2 / psi_grad**2 * Ti_grad * g4
    el_particle_flux = -2 * jnp.pi * CONSTANTS.me / tau_e / e**2 * (p_grad - 3/2 * n * Te_grad) / psi_grad**2 * g4

    return el_particle_flux, el_heat_flux, ion_heat_flux

  def _calculate_turbulent_flux(self, core_profiles: state.CoreProfiles, geo: geometry.Geometry, transport: DynamicRuntimeParams):
    eta_el = calc_eta(core_profiles.temp_el, core_profiles.ne)
    eta_el = jnp.nan_to_num(eta_el, nan=eta_el[1])
    eta_ion = calc_eta(core_profiles.temp_ion, core_profiles.ni)
    eta_ion = jnp.nan_to_num(eta_ion, nan=eta_ion[1])
    d = calc_d(geo, core_profiles)
    d = jnp.nan_to_num(d, nan=d[2])

    delta_eta_el, delta_d_el = self._calc_deltas(eta_el, d)
    delta_eta_ion, delta_d_ion = self._calc_deltas(eta_ion, d)

    ne_turbulent_flux = self._critical_model(delta_d_el, delta_eta_el, transport.b_n, transport.c_n, eta_el, True)
    Te_turbulent_flux = self._critical_model(delta_d_el, delta_eta_el, transport.b_te, transport.c_te, eta_el, False)
    Ti_turbulent_flux = self._critical_model(delta_d_ion, delta_eta_ion, transport.b_ti, transport.c_ti, eta_ion, False)

    return ne_turbulent_flux, Te_turbulent_flux, Ti_turbulent_flux

  def _calc_deltas(self, eta, d):
    return self._calc_deltas_from_lines(eta, d, [[2, 1/3], [-1.5, 5/3+2/3*1.5], [0, 0.44]], [2/3, -2/3*(0.44-8/3)])

  def _closest_point_on_line(self, x_0, y_0, a, b, c):
    return (b * (b * x_0 - a * y_0) - a * c) / (a ** 2 + b ** 2), (a * (-b * x_0 + a * y_0) - b * c) / (a ** 2 + b ** 2)

  def _calc_deltas_from_lines(self, x, y, lines, bounds):
    best_dist = jnp.full_like(x, jnp.inf)
    best_x = jnp.zeros_like(x)
    best_y = jnp.zeros_like(y)
    for i, line in enumerate(lines):
      x0, y0 = self._closest_point_on_line(x, y, line[0], -1, line[1])
      x0 = jnp.where(y < y0, x, x0)
      y0 = jnp.where(y < y0, y, y0)
      mask = jnp.full_like(x, True, dtype=bool)
      if i > 0:
        mask &= x0 > bounds[i-1]
      if i < len(lines) - 1:
        mask &= x0 < bounds[i]
      dist2 = (x-x0)**2 + (y-y0)**2
      mask &= dist2 < best_dist
      best_dist = jnp.where(mask, dist2, best_dist)
      best_x = jnp.where(mask, x0, best_x)
      best_y = jnp.where(mask, y0, best_y)
    for i, bound in enumerate(bounds):
      x0 = bound
      y0 = lines[i][0]*bound + lines[i][1]
      dist2 = (x-x0)**2 + (y-y0)**2
      mask = dist2 < best_dist
      best_dist = jnp.where(mask, dist2, best_dist)
      best_x = jnp.where(mask, x0, best_x)
      best_y = jnp.where(mask, y0, best_y)
    return jnp.abs(best_x-x), jnp.abs(best_y-y)

  def _critical_model(self, delta_d, delta_eta, b, c, eta, is_particle):
    is_particle = 1.0 if is_particle else -1.0
    return b * (delta_d**2 + delta_eta**2) ** (c/2) * jnp.sign(eta - 2/3) * is_particle

  def _convert_flux_to_coeffs(self, ne_flux, Te_flux, Ti_flux, nref, core_profiles: state.CoreProfiles, geo: geometry.Geometry) -> state.CoreTransport:
    d_face_el = -ne_flux / nref / core_profiles.ne.face_grad() / geo.g1_over_vpr_face
    chi_face_el = -Te_flux / nref / CONSTANTS.keV2J / core_profiles.temp_el.face_grad() / core_profiles.ne.face_value() / geo.g1_over_vpr_face
    chi_face_ion = -Ti_flux / nref / CONSTANTS.keV2J / core_profiles.temp_ion.face_grad() / core_profiles.ne.face_value() / geo.g1_over_vpr_face
    d_face_el = jnp.nan_to_num(d_face_el, nan=0.0)
    chi_face_el = jnp.nan_to_num(chi_face_el, nan=0.0)
    chi_face_ion = jnp.nan_to_num(chi_face_ion, nan=0.0)
    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=jnp.zeros_like(geo.rho_face_norm),
    )

  def __hash__(self):
    return hash('CriticalGradientDipoleModel')

  def __eq__(self, other):
    return isinstance(other, CriticalGradientDipoleModel)
