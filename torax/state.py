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

"""Classes defining the TORAX state that evolves over time."""
import enum
from typing import Optional

from absl import logging
import chex
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import jax_utils
from torax.config import config_args
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import source_profiles
import typing_extensions


@chex.dataclass(frozen=True)
class Currents:
  """Dataclass to group currents and related variables (e.g. conductivity).

  Not all fields are actually used by the library. For example,
  j_bootstrap and I_bootstrap are updated during the sim loop,
  but not read from. These fields are an output of the library
  that may be interesting for the end user to plot, etc.
  """

  jtot: array_typing.ArrayFloat
  jtot_face: array_typing.ArrayFloat
  johm: array_typing.ArrayFloat
  external_current_source: array_typing.ArrayFloat
  j_bootstrap: array_typing.ArrayFloat
  j_bootstrap_face: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  # Using physics notation naming convention
  I_bootstrap: array_typing.ScalarFloat  # [A]
  Ip_profile_face: array_typing.ArrayFloat  # [A]
  sigma: array_typing.ArrayFloat
  jtot_hires: Optional[array_typing.ArrayFloat] = None

  @property
  def Ip_total(self) -> array_typing.ScalarFloat:
    """Returns the total plasma current [A]."""
    return self.Ip_profile_face[..., -1]

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> "Currents":
    """Returns a Currents with all zeros."""
    return cls(
        jtot=jnp.zeros(geo.rho_face.shape),
        jtot_face=jnp.zeros(geo.rho_face.shape),
        johm=jnp.zeros(geo.rho_face.shape),
        external_current_source=jnp.zeros(geo.rho_face.shape),
        j_bootstrap=jnp.zeros(geo.rho_face.shape),
        j_bootstrap_face=jnp.zeros(geo.rho_face.shape),
        I_bootstrap=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Ip_profile_face=jnp.zeros(geo.rho_face.shape),
        sigma=jnp.zeros(geo.rho_face.shape),
        jtot_hires=jnp.zeros(geo.rho_face.shape),
    )


@chex.dataclass(frozen=True, eq=False)
class CoreProfiles:
  """Dataclass for holding the evolving core plasma profiles.

  This dataclass is inspired by the IMAS `core_profiles` IDS.

  Many of the profiles in this class are evolved by the PDE system in TORAX, and
  therefore are stored as CellVariables. Other profiles are computed outside the
  internal PDE system, and are simple JAX arrays.

  Attributes:
      temp_ion: Ion temperature [keV].
      temp_el: Electron temperature [keV].
      psi: Poloidal flux [Wb].
      psidot: Time derivative of poloidal flux (loop voltage) [V].
      ne: Electron density [nref m^-3].
      ni: Main ion density [nref m^-3].
      nimp: Impurity density [nref m^-3].
      currents: Instance of the Currents dataclass.
      q_face: Safety factor.
      s_face: Magnetic shear.
      nref: Reference density [m^-3].
      vloop_lcfs: Loop voltage at LCFS (V).
      Zi: Main ion charge on cell grid [dimensionless].
      Zi_face: Main ion charge on face grid [dimensionless].
      Ai: Main ion mass [amu].
      Zimp: Impurity charge on cell grid [dimensionless].
      Zimp_face: Impurity charge on face grid [dimensionless].
      Aimp: Impurity mass [amu].
  """

  temp_ion: cell_variable.CellVariable
  temp_el: cell_variable.CellVariable
  psi: cell_variable.CellVariable
  psidot: cell_variable.CellVariable
  ne: cell_variable.CellVariable
  ni: cell_variable.CellVariable
  nimp: cell_variable.CellVariable
  currents: Currents
  q_face: array_typing.ArrayFloat
  s_face: array_typing.ArrayFloat
  nref: array_typing.ScalarFloat
  vloop_lcfs: array_typing.ScalarFloat
  # pylint: disable=invalid-name
  Zi: array_typing.ArrayFloat
  Zi_face: array_typing.ArrayFloat
  Ai: array_typing.ScalarFloat
  Zimp: array_typing.ArrayFloat
  Zimp_face: array_typing.ArrayFloat
  Aimp: array_typing.ScalarFloat
  # pylint: enable=invalid-name

  def quasineutrality_satisfied(self) -> bool:
    """Checks if quasineutrality is satisfied."""
    return jnp.allclose(
        self.ni.value * self.Zi + self.nimp.value * self.Zimp,
        self.ne.value,
    ).item()

  def negative_temperature_or_density(self) -> bool:
    """Checks if any temperature or density is negative."""
    profiles_to_check = (
        self.temp_ion,
        self.temp_el,
        self.ne,
        self.ni,
        self.nimp,
    )
    return any(
        [jnp.any(jnp.less(x, 0.0)) for x in jax.tree.leaves(profiles_to_check)]
    )

  def index(self, i: int) -> typing_extensions.Self:
    """If the CoreProfiles is a history, returns the i-th CoreProfiles."""
    idx = lambda x: x[i]
    state = jax.tree_util.tree_map(idx, self)
    # These variables track whether they are histories, so when we collapse down
    # to a single state we need to explicitly clear the history flag.
    history_vars = ["temp_ion", "temp_el", "psi", "psidot", "ne", "ni"]
    history_replace = {"history": None}
    replace_dict = {var: history_replace for var in history_vars}
    state = config_args.recursive_replace(state, **replace_dict)
    return state

  def sanity_check(self):
    for field in CoreProfiles.__dataclass_fields__:
      value = getattr(self, field)
      if hasattr(value, "sanity_check"):
        value.sanity_check()

  def __str__(self) -> str:
    return f"""
      CoreProfiles(
        temp_ion={self.temp_ion},
        temp_el={self.temp_el},
        psi={self.psi},
        ne={self.ne},
        nimp={self.nimp},
        ni={self.ni},
      )
    """


@chex.dataclass
class CoreTransport:
  """Coefficients for the plasma transport.

  These coefficients are computed by TORAX transport models. See the
  transport_model/ folder for more info.

  NOTE: The naming of this class is inspired by the IMAS `core_transport` IDS,
  but its schema is not a 1:1 mapping to that IDS.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid.
    chi_face_el: Electron heat conductivity, on the face grid.
    d_face_el: Diffusivity of electron density, on the face grid.
    v_face_el: Convection strength of electron density, on the face grid.
    chi_face_el_bohm: (Optional) Bohm contribution for electron heat
      conductivity.
    chi_face_el_gyrobohm: (Optional) GyroBohm contribution for electron heat
      conductivity.
    chi_face_ion_bohm: (Optional) Bohm contribution for ion heat conductivity.
    chi_face_ion_gyrobohm: (Optional) GyroBohm contribution for ion heat
      conductivity.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: Optional[jax.Array] = None
  chi_face_el_gyrobohm: Optional[jax.Array] = None
  chi_face_ion_bohm: Optional[jax.Array] = None
  chi_face_ion_gyrobohm: Optional[jax.Array] = None

  def __post_init__(self):
    # Use the array size of chi_face_el as a reference.
    if self.chi_face_el_bohm is None:
      self.chi_face_el_bohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_el_gyrobohm is None:
      self.chi_face_el_gyrobohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_ion_bohm is None:
      self.chi_face_ion_bohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_ion_gyrobohm is None:
      self.chi_face_ion_gyrobohm = jnp.zeros_like(self.chi_face_el)

  def chi_max(
      self,
      geo: geometry.Geometry,
  ) -> jax.Array:
    """Calculates the maximum value of chi.

    Args:
      geo: Geometry of the torus.

    Returns:
      chi_max: Maximum value of chi.
    """
    return jnp.maximum(
        jnp.max(self.chi_face_ion * geo.g1_over_vpr2_face),
        jnp.max(self.chi_face_el * geo.g1_over_vpr2_face),
    )

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a CoreTransport with all zeros. Useful for initializing."""
    shape = geo.rho_face.shape
    return cls(
        chi_face_ion=jnp.zeros(shape),
        chi_face_el=jnp.zeros(shape),
        d_face_el=jnp.zeros(shape),
        v_face_el=jnp.zeros(shape),
        chi_face_el_bohm=jnp.zeros(shape),
        chi_face_el_gyrobohm=jnp.zeros(shape),
        chi_face_ion_bohm=jnp.zeros(shape),
        chi_face_ion_gyrobohm=jnp.zeros(shape),
    )


@chex.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
  """Collection of outputs calculated after each simulation step.

  These variables are not used internally, but are useful as outputs or
  intermediate observations for overarching workflows.

  Attributes:
    pressure_thermal_ion_face: Ion thermal pressure on the face grid [Pa]
    pressure_thermal_el_face: Electron thermal pressure on the face grid [Pa]
    pressure_thermal_tot_face: Total thermal pressure on the face grid [Pa]
    pprime_face: Derivative of total pressure with respect to poloidal flux on
      the face grid [Pa/Wb]
    W_thermal_ion: Ion thermal stored energy [J]
    W_thermal_el: Electron thermal stored energy [J]
    W_thermal_tot: Total thermal stored energy [J]
    tauE: Thermal energy confinement time [s]
    H89P: L-mode confinement quality factor with respect to the ITER89P scaling
      law derived from the ITER L-mode confinement database
    H98: H-mode confinement quality factor with respect to the ITER98y2 scaling
      law derived from the ITER H-mode confinement database
    H97L: L-mode confinement quality factor with respect to the ITER97L scaling
      law derived from the ITER L-mode confinement database
    H20: H-mode confinement quality factor with respect to the ITER20 scaling
      law derived from the updated (2020) ITER H-mode confinement database
    FFprime_face: FF' on the face grid, where F is the toroidal flux function
    psi_norm_face: Normalized poloidal flux on the face grid [Wb]
    psi_face: Poloidal flux on the face grid [Wb]
    P_sol_ion: Total ion heating power exiting the plasma with all sources:
      auxiliary heating + ion-electron exchange + fusion [W]
    P_sol_el: Total electron heating power exiting the plasma with all sources
      and sinks: auxiliary heating + ion-electron exchange + Ohmic + fusion +
      radiation sinks [W]
    P_sol_tot: Total heating power exiting the plasma with all sources and sinks
    P_external_ion: Total external ion heating power: auxiliary heating + Ohmic
      [W]
    P_external_el: Total external electron heating power: auxiliary heating +
      Ohmic [W]
    P_external_tot: Total external heating power: auxiliary heating + Ohmic [W]
    P_external_injected: Total external injected power before absorption [W]
    P_ei_exchange_ion: Electron-ion heat exchange power to ions [W]
    P_ei_exchange_el: Electron-ion heat exchange power to electrons [W]
    P_generic_ion: Total generic_ion_el_heat_source power to ions [W]
    P_generic_el: Total generic_ion_el_heat_source power to electrons [W]
    P_generic_tot: Total generic_ion_el_heat power [W]
    P_alpha_ion: Total fusion power to ions [W]
    P_alpha_el: Total fusion power to electrons [W]
    P_alpha_tot: Total fusion power to plasma [W]
    P_ohmic: Ohmic heating power to electrons [W]
    P_brems: Bremsstrahlung electron heat sink [W]
    P_cycl: Cyclotron radiation electron heat sink [W]
    P_ecrh: Total electron cyclotron source power [W]
    P_rad: Impurity radiation heat sink [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_generic: Total generic source current [A]
    Q_fusion: Fusion power gain
    P_icrh_el: Ion cyclotron resonance heating to electrons [W]
    P_icrh_ion: Ion cyclotron resonance heating to ions [W]
    P_icrh_tot: Total ion cyclotron resonance heating power [W]
    P_LH_hi_dens: H-mode transition power for high density branch [W]
    P_LH_min: Minimum H-mode transition power for at ne_min_P_LH [W]
    P_LH: H-mode transition power from maximum of P_LH_hi_dens and P_LH_min [W]
    ne_min_P_LH: Density corresponding to the P_LH_min [nref]
    E_cumulative_fusion: Total cumulative fusion energy [J]
    E_cumulative_external: Total external injected energy (Ohmic + auxiliary
      heating) [J]
    te_volume_avg: Volume average electron temperature [keV]
    ti_volume_avg: Volume average ion temperature [keV]
    ne_volume_avg: Volume average electron density [nref m^-3]
    ni_volume_avg: Volume average main ion density [nref m^-3]
    ne_line_avg: Line averaged electron density [nref m^-3]
    ni_line_avg: Line averaged main ion density [nref m^-3]
    fgw_ne_volume_avg: Greenwald fraction from volume-averaged electron density
      [dimensionless]
    fgw_ne_line_avg: Greenwald fraction from line-averaged electron density
      [dimensionless]
    q95: q at 95% of the normalized poloidal flux
    Wpol: Total magnetic energy [J]
    li3: Normalized plasma internal inductance, ITER convention [dimensionless]
    dW_th_dt: Time derivative of the total stored thermal energy [W]
  """

  pressure_thermal_ion_face: array_typing.ArrayFloat
  pressure_thermal_el_face: array_typing.ArrayFloat
  pressure_thermal_tot_face: array_typing.ArrayFloat
  pprime_face: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  W_thermal_ion: array_typing.ScalarFloat
  W_thermal_el: array_typing.ScalarFloat
  W_thermal_tot: array_typing.ScalarFloat
  tauE: array_typing.ScalarFloat
  H89P: array_typing.ScalarFloat
  H98: array_typing.ScalarFloat
  H97L: array_typing.ScalarFloat
  H20: array_typing.ScalarFloat
  FFprime_face: array_typing.ArrayFloat
  psi_norm_face: array_typing.ArrayFloat
  # psi_face included in post_processed output for convenience, since the
  # CellVariable history method destroys class methods like `face_value`.
  psi_face: array_typing.ArrayFloat
  # Integrated heat sources
  P_sol_ion: array_typing.ScalarFloat  # SOL stands for "Scrape Off Layer"
  P_sol_el: array_typing.ScalarFloat
  P_sol_tot: array_typing.ScalarFloat
  P_external_ion: array_typing.ScalarFloat
  P_external_el: array_typing.ScalarFloat
  P_external_tot: array_typing.ScalarFloat
  P_external_injected: array_typing.ScalarFloat
  P_ei_exchange_ion: array_typing.ScalarFloat
  P_ei_exchange_el: array_typing.ScalarFloat
  P_generic_ion: array_typing.ScalarFloat
  P_generic_el: array_typing.ScalarFloat
  P_generic_tot: array_typing.ScalarFloat
  P_alpha_ion: array_typing.ScalarFloat
  P_alpha_el: array_typing.ScalarFloat
  P_alpha_tot: array_typing.ScalarFloat
  P_ohmic: array_typing.ScalarFloat
  P_brems: array_typing.ScalarFloat
  P_cycl: array_typing.ScalarFloat
  P_ecrh: array_typing.ScalarFloat
  P_rad: array_typing.ScalarFloat
  I_ecrh: array_typing.ScalarFloat
  I_generic: array_typing.ScalarFloat
  Q_fusion: array_typing.ScalarFloat
  P_icrh_el: array_typing.ScalarFloat
  P_icrh_ion: array_typing.ScalarFloat
  P_icrh_tot: array_typing.ScalarFloat
  P_LH_hi_dens: array_typing.ScalarFloat
  P_LH_min: array_typing.ScalarFloat
  P_LH: array_typing.ScalarFloat
  ne_min_P_LH: array_typing.ScalarFloat
  E_cumulative_fusion: array_typing.ScalarFloat
  E_cumulative_external: array_typing.ScalarFloat
  te_volume_avg: array_typing.ScalarFloat
  ti_volume_avg: array_typing.ScalarFloat
  ne_volume_avg: array_typing.ScalarFloat
  ni_volume_avg: array_typing.ScalarFloat
  ne_line_avg: array_typing.ScalarFloat
  ni_line_avg: array_typing.ScalarFloat
  fgw_ne_volume_avg: array_typing.ScalarFloat
  fgw_ne_line_avg: array_typing.ScalarFloat
  q95: array_typing.ScalarFloat
  Wpol: array_typing.ScalarFloat
  li3: array_typing.ScalarFloat
  dW_th_dt: array_typing.ScalarFloat
  # pylint: enable=invalid-name

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a PostProcessedOutputs with all zeros, used for initializing."""
    return cls(
        pressure_thermal_ion_face=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_el_face=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_tot_face=jnp.zeros(geo.rho_face.shape),
        pprime_face=jnp.zeros(geo.rho_face.shape),
        W_thermal_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        tauE=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H89P=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H98=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H97L=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H20=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        FFprime_face=jnp.zeros(geo.rho_face.shape),
        psi_norm_face=jnp.zeros(geo.rho_face.shape),
        psi_face=jnp.zeros(geo.rho_face.shape),
        P_sol_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_sol_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_sol_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_injected=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ei_exchange_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ei_exchange_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_generic_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_generic_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_generic_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ohmic=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_brems=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_cycl=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ecrh=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_rad=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_ecrh=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_generic=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Q_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_hi_dens=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ne_min_P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_cumulative_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_cumulative_external=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        te_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ti_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ne_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ni_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ne_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ni_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        fgw_ne_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        fgw_ne_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        q95=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Wpol=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        li3=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        dW_th_dt=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )

  def check_for_errors(self):
    if has_nan(self):
      return SimError.NAN_DETECTED
    else:
      return SimError.NO_ERROR


@chex.dataclass
class StepperNumericOutputs:
  """Numerical quantities related to the stepper.

  Attributes:
    outer_stepper_iterations: Number of iterations performed in the outer loop
      of the stepper.
    stepper_error_state: 0 if solver converged with fine tolerance for this step
      1 if solver did not converge for this step (was above coarse tol) 2 if
      solver converged within coarse tolerance. Allowed to pass with a warning.
      Occasional error=2 has low impact on final sim state.
    inner_solver_iterations: Total number of iterations performed in the solver
      across all iterations of the stepper.
  """

  outer_stepper_iterations: int = 0
  stepper_error_state: int = 0
  inner_solver_iterations: int = 0


@enum.unique
class SimError(enum.Enum):
  """Integer enum for sim error handling."""

  NO_ERROR = 0
  NAN_DETECTED = 1
  QUASINEUTRALITY_BROKEN = 2
  NEGATIVE_CORE_PROFILES = 3

  def log_error(self):
    match self:
      case SimError.NEGATIVE_CORE_PROFILES:
        logging.error("""
            Simulation stopped due to negative values in core profiles.
            """)
      case SimError.NAN_DETECTED:
        logging.error("""
            Simulation stopped due to NaNs in state.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.QUASINEUTRALITY_BROKEN:
        logging.error("""
            Simulation stopped due to quasineutrality being violated.
            Possible cause is bad handling of impurity species.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.NO_ERROR:
        pass
      case _:
        raise ValueError(f"Unknown SimError: {self}")


@chex.dataclass
class ToraxSimState:
  """Full simulator state.

  The simulation stepping in sim.py evolves core_profiles which includes all
  the attributes the simulation is advancing. But beyond those, there are
  additional stateful elements which may evolve on each simulation step, such
  as sources and transport.

  This class includes both core_profiles and these additional elements.

  Attributes:
    t: time coordinate.
    dt: timestep interval.
    core_profiles: Core plasma profiles at time t.
    core_transport: Core plasma transport coefficients computed at time t.
    core_sources: Profiles for all sources/sinks. These are the profiles that
      are used to calculate the coefficients for the t+dt time step. For the
      explicit sources, these are calculated at the start of the time step, so
      are the values at time t. For the implicit sources, these are the most
      recent guess for time t+dt. The profiles here are the merged version of
      the explicit and implicit profiles.
    post_processed_outputs: variables for output or intermediate observations
      for overarching workflows, calculated after each simulation step.
    geometry: Geometry at this time step used for the simulation.
    time_step_calculator_state: the state of the TimeStepper.
    stepper_numeric_outputs: Numerical quantities related to the stepper.
    sawtooth_crash: True if a sawtooth model is active and the state
      corresponds to a post-sawtooth-crash state.
  """

  t: jax.Array
  dt: jax.Array
  core_profiles: CoreProfiles
  core_transport: CoreTransport
  core_sources: source_profiles.SourceProfiles
  geometry: geometry.Geometry
  stepper_numeric_outputs: StepperNumericOutputs
  sawtooth_crash: bool = False

  def check_for_errors(self) -> SimError:
    """Checks for errors in the simulation state."""
    if self.core_profiles.negative_temperature_or_density():
      logging.info("%s", self.core_profiles)
      log_negative_profile_names(self.core_profiles)
      return SimError.NEGATIVE_CORE_PROFILES
    # If there are NaNs that occured without negative core profiles, log this
    # as a separate error.
    if has_nan(self):
      logging.info("%s", self.core_profiles)
      return SimError.NAN_DETECTED
    elif not self.core_profiles.quasineutrality_satisfied():
      return SimError.QUASINEUTRALITY_BROKEN
    else:
      return SimError.NO_ERROR


def has_nan(inputs: ToraxSimState | PostProcessedOutputs) -> bool:
  return any([jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(inputs)])


def log_negative_profile_names(inputs: CoreProfiles):
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  for path, value in path_vals:
    if jnp.any(jnp.less(value, 0.0)):
      logging.info("Found negative value in %s", jax.tree_util.keystr(path))


def check_for_errors(
    sim_state: ToraxSimState,
    post_processed_outputs: PostProcessedOutputs,
) -> SimError:
  """Checks for errors in the simulation state."""
  state_error = sim_state.check_for_errors()
  if state_error != SimError.NO_ERROR:
    return state_error
  else:
    return post_processed_outputs.check_for_errors()
