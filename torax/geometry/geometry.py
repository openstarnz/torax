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

from collections.abc import Mapping, Sequence
import dataclasses
import enum

import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax import array_typing
from torax import jax_utils


@chex.dataclass(frozen=True)
class Grid1D:
  """Data structure defining a 1-D grid of cells with faces.

  Construct via `construct` classmethod.

  Attributes:
    nx: Number of cells.
    dx: Distance between cell centers.
    face_centers: Coordinates of face centers.
    cell_centers: Coordinates of cell centers.
  """

  nx: int
  dx: float
  face_centers: chex.Array
  cell_centers: chex.Array

  def __post_init__(self):
    jax_utils.assert_rank(self.nx, 0)
    jax_utils.assert_rank(self.dx, 0)
    jax_utils.assert_rank(self.face_centers, 1)
    jax_utils.assert_rank(self.cell_centers, 1)

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
class GeometryType(enum.Enum):
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
  """

  # TODO(b/356356966): extend documentation to define what each attribute is.
  geometry_type: int
  torax_mesh: Grid1D
  drho_norm: array_typing.ArrayFloat
  Phi: chex.Array
  Phi_face: chex.Array
  Rmaj: chex.Array
  Rmin: chex.Array
  B0: chex.Array
  volume: chex.Array
  volume_face: chex.Array
  area: chex.Array
  area_face: chex.Array
  vpr: chex.Array
  vpr_face: chex.Array
  spr_cell: chex.Array
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
  F: chex.Array
  F_face: chex.Array
  F_hires: chex.Array
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

  @property
  def rho_norm(self) -> chex.Array:
    return self.torax_mesh.cell_centers

  @property
  def rho_face_norm(self) -> chex.Array:
    return self.torax_mesh.face_centers

  @property
  def rho_face(self) -> chex.Array:
    return self.rho_face_norm * self.rho_b

  @property
  def rho(self) -> chex.Array:
    return self.rho_norm * self.rho_b

  @property
  def rmid(self) -> chex.Array:
    return (self.Rout - self.Rin) / 2

  @property
  def rmid_face(self) -> chex.Array:
    return (self.Rout_face - self.Rin_face) / 2

  @property
  def drho(self) -> chex.Array:
    return self.drho_norm * self.rho_b

  @property
  def rho_b(self) -> chex.Array:
    """Toroidal flux coordinate at boundary (LCFS)."""
    return jnp.sqrt(self.Phib / np.pi / self.B0)

  @property
  def Phib(self) -> chex.Array:
    """Toroidal flux at boundary (LCFS)."""
    return self.Phi_face[-1]

  @property
  def g1_over_vpr(self) -> chex.Array:
    return self.g1 / self.vpr

  @property
  def g1_over_vpr2(self) -> chex.Array:
    return self.g1 / self.vpr**2

  @property
  def g0_over_vpr_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.ones(1) / self.rho_b,  # correct value is 1/rho_b on-axis
        self.g0_face[1:] / self.vpr_face[1:],  # avoid div by zero on-axis
    ))

  @property
  def g1_over_vpr_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.zeros(1),  # correct value is zero on-axis
        self.g1_face[1:] / self.vpr_face[1:],  # avoid div by zero on-axis
    ))

  @property
  def g1_over_vpr2_face(self) -> jax.Array:
    return jnp.concatenate((
        jnp.ones(1) / self.rho_b**2,  # correct value is 1/rho_b**2 on-axis
        self.g1_face[1:] / self.vpr_face[1:] ** 2,  # avoid div by zero on-axis
    ))

  def z_magnetic_axis(self) -> chex.Numeric:
    z_magnetic_axis = self._z_magnetic_axis
    if z_magnetic_axis is not None:
      return z_magnetic_axis
    else:
      raise ValueError(
          'Geometry does not have a z magnetic axis.'
      )


def stack_geometries(
    geometries: Sequence[Geometry],
) -> Mapping[str, chex.Array]:
  """Batch together a sequence of geometries.

  The batched geometries are returned as a dictionary of arrays. Any fields
  that are not arrays are not included in the dictionary and all properties
  are excluded as well.

  Args:
    geometries: A sequence of geometries to stack. The geometries must have the
      same mesh, geometry type, and drho_norm.

  Returns:
    A dictionary of arrays, where each array has the same shape as the
    corresponding attribute in the input geometries, but with an additional
    leading axis (e.g. for the time dimension).
  """
  if not geometries:
    raise ValueError('No geometries provided.')
  # Stack the geometries.
  torax_mesh = geometries[0].torax_mesh
  geometry_type = geometries[0].geometry_type
  drho_norm = geometries[0].drho_norm
  for geometry in geometries[1:]:
    if geometry.torax_mesh != torax_mesh:
      raise ValueError('All geometries must have the same mesh.')
    if geometry.geometry_type != geometry_type:
      raise ValueError('All geometries must have the same geometry type.')
    if geometry.drho_norm != drho_norm:
      raise ValueError('All geometries must have the same drho_norm.')
  array_fields = [
      attr
      for attr, val in dataclasses.asdict(geometries[0]).items()
      if isinstance(val, chex.Array)
  ]
  return {
      attr: jnp.stack([getattr(geo, attr) for geo in geometries])
      for attr in array_fields
  }
