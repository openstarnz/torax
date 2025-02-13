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

"""Classes for representing a circular geometry."""


from __future__ import annotations

import chex
import numpy as np
from torax import interpolated_param
from torax.geometry import geometry
from torax.geometry import geometry_provider


# Using invalid-name because we are using the same naming convention as the
# external physics implementations
# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class CircularAnalyticalGeometry(geometry.Geometry):
  """Circular geometry type used for testing only.

  Most users should default to using the Geometry class.
  """

  elongation_hires: chex.Array


@chex.dataclass(frozen=True)
class CircularAnalyticalGeometryProvider(
    geometry_provider.TimeDependentGeometryProvider):
  """Circular geometry type used for testing only.

  Most users should default to using the GeometryProvider class.
  """

  elongation_hires: interpolated_param.InterpolatedVarSingleAxis

  def __call__(self, t: chex.Numeric) -> geometry.Geometry:
    """Returns a Geometry instance at the given time."""
    return self._get_geometry_base(t, CircularAnalyticalGeometry)


def build_circular_geometry(
    n_rho: int = 25,
    elongation_LCFS: float = 1.72,
    Rmaj: float = 6.2,
    Rmin: float = 2.0,
    B0: float = 5.3,
    hires_fac: int = 4,
    Rinner: float = 0.0,
) -> CircularAnalyticalGeometry:
  """Constructs a CircularAnalyticalGeometry.

  This is the standard entrypoint for building a circular geometry, not
  CircularAnalyticalGeometry.__init__(). chex.dataclasses do not allow
  overriding __init__ functions with different parameters than the attributes of
  the dataclass, so this builder function lives outside the class.

  Args:
    n_rho: Radial grid points (num cells)
    elongation_LCFS: Elongation at last closed flux surface. Defaults to 1.72
      for the ITER elongation, to approximately correct volume and area integral
      Jacobians.
    Rmaj: major radius (R) in meters
    Rmin: minor radius (a) in meters
    B0: Toroidal magnetic field on axis [T]
    hires_fac: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    Rinner: inner radius in meters

  Returns:
    A CircularAnalyticalGeometry instance.
  """
  # circular geometry assumption of r/Rmin = rho_norm, the normalized
  # toroidal flux coordinate.
  drho_norm = 1.0 / n_rho
  # Define mesh (Slab Uniform 1D with Jacobian = 1)
  mesh = geometry.Grid1D.construct(nx=n_rho, dx=drho_norm)
  # toroidal flux coordinate (rho) at boundary (last closed flux surface)
  rho_b = np.asarray(Rmin)
  rho_i = np.asarray(Rinner)
  delta_rho = rho_b - rho_i

  # normalized and unnormalized toroidal flux coordinate (rho)
  # on face and cell grids. See fvm documentation and paper for details on
  # face and cell grids.
  rho_face_norm = mesh.face_centers
  rho_norm = mesh.cell_centers
  rho_face = rho_face_norm * delta_rho + rho_i
  rho = rho_norm * delta_rho + rho_i

  Rmaj = np.array(Rmaj)
  B0 = np.array(B0)

  # Elongation profile.
  # Set to be a linearly increasing function from 1 to elongation_LCFS, which
  # is the elongation value at the last closed flux surface, set in config.
  elongation_FCFS = 1.0 + rho_i * (elongation_LCFS - 1) / rho_b
  elongation = elongation_FCFS + rho_norm * (elongation_LCFS - elongation_FCFS)
  elongation_face = elongation_FCFS + rho_face_norm * (elongation_LCFS - elongation_FCFS)

  # Volume in elongated circular geometry is given by:
  # V = 2*pi^2*R*rho^2*elongation
  # S = pi*rho^2*elongation

  volume = 2 * np.pi**2 * Rmaj * rho**2 * elongation
  volume_face = 2 * np.pi**2 * Rmaj * rho_face**2 * elongation_face
  area = np.pi * rho**2 * elongation
  area_face = np.pi * rho_face**2 * elongation_face

  # V' = dV/drnorm for volume integrations
  def calc_vpr(rho_, elongation_, volume_):
    return (
        4 * np.pi**2 * Rmaj * rho_ * delta_rho * elongation_
        + volume_ * (elongation_LCFS - elongation_FCFS) / elongation_
    )
  vpr = calc_vpr(rho, elongation, volume)
  vpr_face = calc_vpr(rho_face, elongation_face, volume_face)

  # S' = dS/drnorm for area integrals on cell grid
  def calc_spr(rho_, elongation_, area_):
    return (
        2 * np.pi * rho_ * elongation_ * delta_rho
        + area_ * (elongation_LCFS - elongation_FCFS) / elongation_
    )
  spr = calc_spr(rho, elongation, area)
  spr_face = calc_spr(rho_face, elongation_face, area_face)

  delta_face = np.zeros(len(rho_face))

  # Geometry variables for general geometry form of transport equations.
  # With circular geometry approximation.

  # g0: <\nabla V>
  g0 = vpr / delta_rho
  g0_face = vpr_face / delta_rho

  # g1: <(\nabla V)^2>
  g1 = vpr**2 / delta_rho**2
  g1_face = vpr_face**2 / delta_rho**2

  # g2: <(\nabla V)^2 / R^2>
  g2 = g1 / Rmaj**2
  g2_face = g1_face / Rmaj**2

  # g3: <1/R^2> (done without a elongation correction)
  # <1/R^2> =
  # 1/2pi*int_0^2pi (1/(Rmaj+r*cosx)^2)dx =
  # 1/( Rmaj^2 * (1 - (r/Rmaj)^2)^3/2 )
  g3 = 1 / (Rmaj**2 * (1 - (rho / Rmaj) ** 2) ** (3.0 / 2.0))
  g3_face = 1 / (Rmaj**2 * (1 - (rho_face / Rmaj) ** 2) ** (3.0 / 2.0))

  # simplifying assumption for now, for J=R*B/(R0*B0)
  J = np.ones(len(rho))
  J_face = np.ones(len(rho_face))
  # simplified (constant) version of the F=B*R function
  G = np.ones_like(rho)
  G_face = np.ones_like(rho_face)

  # Using an approximation where:
  # g2g3_over_rhon = 16 * pi**4 * G2 / (J * R) where:
  # G2 = vpr / (4 * pi**2) * <1/R^2>
  # This is done due to our ad-hoc elongation assumption, which leads to more
  # reasonable values for g2g3_over_rhon through the G2 definition.
  # In the future, a more rigorous analytical geometry will be developed and
  # the direct definition of g2g3_over_rhon will be used.

  g2g3_over_rhon = 4 * np.pi**2 * vpr * g3 / (J * Rmaj)
  g2g3_over_rhon_face = 4 * np.pi**2 * vpr_face * g3_face / (J_face * Rmaj)

  # High resolution versions for j (plasma current) and psi (poloidal flux)
  # manipulations. Needed if psi is initialized from plasma current, which is
  # the only option for ad-hoc circular geometry.
  rho_hires_norm = np.linspace(0, 1, n_rho * hires_fac)
  rho_hires = rho_hires_norm * delta_rho + rho_i

  Rout = Rmaj + rho
  Rout_face = Rmaj + rho_face

  Rin = Rmaj - rho
  Rin_face = Rmaj - rho_face

  # assumed elongation profile on hires grid
  elongation_hires = 1 + rho_hires_norm * (elongation_LCFS - 1)

  volume_hires = 2 * np.pi**2 * Rmaj * rho_hires**2 * elongation_hires
  area_hires = np.pi * rho_hires**2 * elongation_hires

  # V' = dV/drnorm for volume integrations on hires grid
  vpr_hires = calc_vpr(rho_hires, elongation_hires, volume_hires)
  # S' = dS/drnorm for area integrals on hires grid
  spr_hires = calc_spr(rho_hires, elongation_hires, area_hires)

  g3_hires = 1 / (Rmaj**2 * (1 - (rho_hires / Rmaj) ** 2) ** (3.0 / 2.0))
  G_hires = np.ones_like(rho_hires)
  g2g3_over_rhon_hires = 4 * np.pi**2 * vpr_hires * g3_hires / (G_hires * Rmaj)

  return CircularAnalyticalGeometry(
      # Set the standard geometry params.
      geometry_type=geometry.GeometryType.CIRCULAR,
      torax_mesh=mesh,
      rho_i=rho_i,
      rho_b=rho_b,
      Rmaj=Rmaj,
      Rmin=rho_b,
      B0=B0,
      volume=volume,
      volume_face=volume_face,
      area=area,
      area_face=area_face,
      vpr=vpr,
      vpr_face=vpr_face,
      spr=spr,
      spr_face=spr_face,
      delta_face=delta_face,
      g0=g0,
      g0_face=g0_face,
      g1=g1,
      g1_face=g1_face,
      g2=g2,
      g2_face=g2_face,
      g3=g3,
      g3_face=g3_face,
      g2g3_over_rhon=g2g3_over_rhon,
      g2g3_over_rhon_face=g2g3_over_rhon_face,
      g2g3_over_rhon_hires=g2g3_over_rhon_hires,
      G=G,
      G_face=G_face,
      G_hires=G_hires,
      Rin=Rin,
      Rin_face=Rin_face,
      Rout=Rout,
      Rout_face=Rout_face,
      # Set the circular geometry-specific params.
      elongation=elongation,
      elongation_face=elongation_face,
      volume_hires=volume_hires,
      area_hires=area_hires,
      spr_hires=spr_hires,
      rho_hires_norm=rho_hires_norm,
      rho_hires=rho_hires,
      elongation_hires=elongation_hires,
      vpr_hires=vpr_hires,
      # always initialize Phibdot as zero. It will be replaced once both geo_t
      # and geo_t_plus_dt are provided, and set to be the same for geo_t and
      # geo_t_plus_dt for each given time interval.
      Phibdot=np.asarray(0.0),
      _z_magnetic_axis=np.asarray(0.0),
  )
