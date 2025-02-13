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

"""Classes for representing the problem geometry."""


from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax.torax_pydantic import torax_pydantic


class Grid1D(torax_pydantic.BaseModelFrozen):
  """Data structure defining a 1-D grid of cells with faces.

  Construct via `construct` classmethod.

  Attributes:
    nx: Number of cells.
    dx: Distance between cell centers.
    face_centers: Coordinates of face centers.
    cell_centers: Coordinates of cell centers.
  """

  nx: pydantic.PositiveInt
  dx: pydantic.PositiveFloat
  face_centers: torax_pydantic.NumpyArray1D
  cell_centers: torax_pydantic.NumpyArray1D

  def __eq__(self, other: Grid1D) -> bool:
    return (
        self.nx == other.nx
        and self.dx == other.dx
        and np.array_equal(self.face_centers, other.face_centers)
        and np.array_equal(self.cell_centers, other.cell_centers)
    )

  def __hash__(self) -> int:
    return hash((self.nx, self.dx))

  @classmethod
  def construct(cls, nx: int, dx: float) -> Grid1D:
    """Constructs a Grid1D.

    Args:
      nx: Number of cells.
      dx: Distance between cell centers.

    Returns:
      grid: A Grid1D with the remaining fields filled in.
    """

    # Note: nx needs to be an int so that the shape `nx + 1` is not a Jax
    # tracer.

    return Grid1D(
        nx=nx,
        dx=dx,
        face_centers=np.linspace(0, nx * dx, nx + 1),
        cell_centers=np.linspace(dx * 0.5, (nx - 0.5) * dx, nx),
    )


def face_to_cell(face: chex.Array) -> chex.Array:
  """Infers cell values corresponding to a vector of face values.

  Simply a linear interpolation between face values.

  Args:
    face: An array containing face values.

  Returns:
    cell: An array containing cell values.
  """

  return 0.5 * (face[:-1] + face[1:])


@enum.unique
class GeometryType(enum.IntEnum):
  """Integer enum for geometry type.

  This type can be used within JAX expressions to access the geometry type
  without having to call isinstance.
  """

  CIRCULAR = 0
  CHEASE = 1
  FBT = 2
  EQDSK = 3


# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class Geometry:
  """Describes the magnetic geometry.

  Most users should default to using the StandardGeometry class, whether the
  source of their geometry comes from CHEASE, MEQ, etc.

  Properties work for both 1D radial arrays and 2D stacked arrays where the
  leading dimension is time.
  """

  # TODO(b/356356966): extend documentation to define what each attribute is.
  geometry_type: GeometryType
  torax_mesh: Grid1D
  rho_i: chex.Array
  rho_b: chex.Array
  Rmaj: chex.Array
  Rmin: chex.Array
  B0: chex.Array
  volume: chex.Array
  volume_face: chex.Array
  area: chex.Array
  area_face: chex.Array
  vpr: chex.Array
  vpr_face: chex.Array
  spr: chex.Array
  spr_face: chex.Array
  delta_face: chex.Array
  elongation: chex.Array
  elongation_face: chex.Array
  g0: chex.Array
  g0_face: chex.Array
  g1: chex.Array
  g1_face: chex.Array
  g2: chex.Array
  g2_face: chex.Array
  g3: chex.Array
  g3_face: chex.Array
  g2g3_over_rhon: chex.Array
  g2g3_over_rhon_face: chex.Array
  g2g3_over_rhon_hires: chex.Array
  # F is defined as R B_tor, and G is defined such that F=G R_0 B_0.
  G: chex.Array
  G_face: chex.Array
  G_hires: chex.Array
  Rin: chex.Array
  Rin_face: chex.Array
  Rout: chex.Array
  Rout_face: chex.Array
  volume_hires: chex.Array
  area_hires: chex.Array
  spr_hires: chex.Array
  rho_hires_norm: chex.Array
  rho_hires: chex.Array
  vpr_hires: chex.Array
  Phibdot: chex.Array
  _z_magnetic_axis: chex.Array | None

  def __post_init__(self):
    self.sanity_check()

  def sanity_check(self):
    # jax_utils.error_if doesn't work here because no further computation is done,
    # and the class is frozen.
    if (
        not jax_utils.is_tracer(self.Phi)
        and not jnp.allclose(self.Phi, jnp.pi * self.B0 * self.rho**2)
    ):
      raise ValueError('Phi does not match expected value.')
    if (
        not jax_utils.is_tracer(self.Phi_face)
        and not jnp.allclose(self.Phi_face, jnp.pi * self.B0 * self.rho_face**2)
    ):
      raise ValueError('Phi_face does not match expected value.')
    if (
        not jax_utils.is_tracer(self.rho_b) and not jax_utils.is_tracer(self.rho_i)
        and not self.rho_b > self.rho_i >= 0
    ):
      raise ValueError('rho_b must be greater than rho_i and both must be non-negative.')

  # In general, this assumes a non-zero value of Phi at the inner boundary.
  # This will be zero for Tokamak geometries, because the magnetic axis makes
  # up the end of the domain, but not in geometries with a finite inner boundary.
  @property
  def Phi(self) -> chex.Array:
    return np.pi * self.B0 * self.rho ** 2

  @property
  def Phi_face(self) -> chex.Array:
    return np.pi * self.B0 * self.rho_face ** 2

  @property
  def rho_norm(self) -> chex.Array:
    return self.torax_mesh.cell_centers

  @property
  def rho_face_norm(self) -> chex.Array:
    return self.torax_mesh.face_centers

  @property
  def drho_norm(self) -> chex.Array:
    return jnp.array(self.torax_mesh.dx)

  @property
  def rho_face(self) -> chex.Array:
    return self.rho_face_norm * self.delta_rho + self.rho_i

  @property
  def rho(self) -> chex.Array:
    return self.rho_norm * self.delta_rho + self.rho_i

  @property
  def rmid(self) -> chex.Array:
    return (self.Rout - self.Rin) / 2

  @property
  def rmid_face(self) -> chex.Array:
    return (self.Rout_face - self.Rin_face) / 2

  @property
  def delta_rho(self) -> chex.Array:
    return self.rho_b - self.rho_i

  @property
  def drho(self) -> chex.Array:
    return self.drho_norm * self.delta_rho

  @property
  def Phib(self) -> chex.Array:
    """Toroidal flux at boundary (LCFS)."""
    return self.Phi_face[..., -1]

  @property
  def g1_over_vpr(self) -> chex.Array:
    return self.g1 / self.vpr

  @property
  def g1_over_vpr2(self) -> chex.Array:
    return self.g1 / self.vpr**2

  @property
  def g0_over_vpr_face(self) -> jax.Array:
    # The correct value is 1/rho_b on-axis, where there will be a division by zero.
    # We don't need to worry about rho_i because this will only happen if rho_i is zero.
    return jnp.nan_to_num(self.g0_face / self.vpr_face, nan=1/self.rho_b)

  @property
  def g1_over_vpr_face(self) -> jax.Array:
    # Correct value is zero on-axis
    return jnp.nan_to_num(self.g1_face / self.vpr_face, nan=0.0)

  @property
  def g1_over_vpr2_face(self) -> jax.Array:
    # Correct value is 1/rho_b**2 on-axis
    return jnp.nan_to_num(self.g1_face / self.vpr_face ** 2, nan=1/self.rho_b**2)

  @property
  def z_magnetic_axis(self) -> chex.Numeric:
    return self._z_magnetic_axis

  @property
  def F(self) -> jax.Array:
    return self.G * self.Rmaj * self.B0

  @property
  def F_face(self):
    return self.G_face * self.Rmaj * self.B0

  @property
  def F_hires(self):
    return self.G_hires * self.Rmaj * self.B0


  def z_magnetic_axis(self) -> chex.Numeric:
    z_magnetic_axis = self._z_magnetic_axis
    if z_magnetic_axis is not None:
      return z_magnetic_axis
    else:
      raise ValueError('Geometry does not have a z magnetic axis.')


def stack_geometries(geometries: Sequence[Geometry]) -> Geometry:
  """Batch together a sequence of geometries.

  Args:
    geometries: A sequence of geometries to stack. The geometries must have the
      same mesh, geometry type.

  Returns:
    A Geometry object, where each array attribute has an additional
    leading axis (e.g. for the time dimension) compared to each Geometry in
    the input sequence.
  """
  if not geometries:
    raise ValueError('No geometries provided.')
  # Ensure that all geometries have same mesh and are of same type.
  first_geo = geometries[0]
  torax_mesh = first_geo.torax_mesh
  geometry_type = first_geo.geometry_type
  for geometry in geometries[1:]:
    if geometry.torax_mesh != torax_mesh:
      raise ValueError('All geometries must have the same mesh.')
    if geometry.geometry_type != geometry_type:
      raise ValueError('All geometries must have the same geometry type.')

  stacked_data = {}
  for field in dataclasses.fields(first_geo):
    field_name = field.name
    field_value = getattr(first_geo, field_name)
    # Stack stackable fields. Save first geo's value for non-stackable fields.
    if isinstance(field_value, chex.Array):
      field_values = [getattr(geo, field_name) for geo in geometries]
      stacked_data[field_name] = jnp.stack(field_values)
    else:
      stacked_data[field_name] = field_value
  # Create a new object with the stacked data with the same class (i.e.
  # could be child classes of Geometry)
  return first_geo.__class__(**stacked_data)
