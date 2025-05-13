from jax import numpy as jnp
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.transport_model import transport_model
from torax.fvm.calc_coeffs import calc_eta, calc_d, calc_particle_flux, calc_temp_flux
from torax.constants import CONSTANTS


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
    nref = dynamic_runtime_params_slice.numerics.nref

    # Classical
    ne_classical_flux, Te_classical_flux, Ti_classical_flux = self._calculate_classical_flux(geo)

    # Turbulent
    ne_turbulent_flux, Te_turbulent_flux, Ti_turbulent_flux = self._calculate_turbulent_flux(geo)

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

    print('Total expected fluxes:')
    print('ne flux:', ne_flux)
    print('Te flux:', temp_el_flux)
    print('Ti flux:', temp_ion_flux)
    print('Actual fluxes:')
    print('ne flux:', actual_ne_flux)
    print('Te flux:', actual_temp_el_flux)
    print('Ti flux:', actual_temp_ion_flux)

    return coeffs

  def _calculate_classical_flux(self, geo: geometry.Geometry):
    return jnp.zeros_like(geo.rho_face_norm), jnp.zeros_like(geo.rho_face_norm), jnp.zeros_like(geo.rho_face_norm)

  def _calculate_turbulent_flux(self, geo: geometry.Geometry):
    eta_el = None
    eta_ion = None
    d = None

    delta_eta_el, delta_d_el = self._calc_deltas(eta_el, d)
    delta_eta_ion, delta_d_ion = self._calc_deltas(eta_ion, d)

    b_ne = 1.0
    c_ne = 1.0
    b_Te = 1.0
    c_Te = 1.0
    b_Ti = 1.0
    c_Ti = 1.0

    ne_turbulent_flux = self._critical_model(delta_d_el, delta_eta_el, b_ne, c_ne)
    Te_turbulent_flux = self._critical_model(delta_d_el, delta_eta_el, b_Te, c_Te)
    Ti_turbulent_flux = self._critical_model(delta_d_ion, delta_eta_ion, b_Ti, c_Ti)

    return ne_turbulent_flux, Te_turbulent_flux, Ti_turbulent_flux

  def _calc_deltas(self, eta, d):
    pass  # TODO

  def _critical_model(self, delta_d, delta_eta, b, c):
    return b * (delta_d**2 + delta_eta**2) ** (c/2)

  def _convert_flux_to_coeffs(self, ne_flux, Te_flux, Ti_flux, nref, core_profiles: state.CoreProfiles, geo: geometry.Geometry) -> state.CoreTransport:
    d_face_el = -ne_flux / nref / core_profiles.ne.face_grad() / geo.g1_over_vpr_face
    chi_face_el = -Te_flux / nref / CONSTANTS.keV2J / core_profiles.temp_el.face_grad() / core_profiles.ne.face_value() / geo.g1_over_vpr_face
    chi_face_ion = -Ti_flux / nref / CONSTANTS.keV2J / core_profiles.temp_ion.face_grad() / core_profiles.ne.face_value() / geo.g1_over_vpr_face
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
