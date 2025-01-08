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

"""The CellVariable class.

A chex dataclass used to represent variables on meshes for the 1D fvm solver.
Naming conventions and API are similar to those developed in the FiPy fvm solver
[https://www.ctcms.nist.gov/fipy/]
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import chex
import jax
from jax import numpy as jnp


@chex.dataclass(frozen=True)
class CellVariable:
  """A variable representing values of the cells along the radius.

  CellVariables are hashable by id, and compare equal only to themselves.

  Attributes:
    value: A jax.Array containing the value of this variable at each cell.
    dr: Distance between cell centers.
    left_face_consx: An optional jax scalar specifying the value or
      gradient of the leftmost face.
    left_face_consx_is_grad: A boolean specifying whether the left face
      constraint is on the gradient (Neumann if True) or the value (Dirichlet if False).
    right_face_consx: Analogous to left_face_consx but for the right
      face, see left_face_consx.
    right_face_consx_is_grad: Analogous to left_face_consx_is_grad but
      for the right face, see left_face_consx_is_grad.
    history: If not none, this CellVariable is an entry in a history, or a
      complete history, made by stacking all the leaves of a CellVariable, e.g.
      as the output of jax.lax.scan. None of the methods of the class work in
      this case, the instance exists only to show a history of the evolution of
      the variable over time. This is None or non-None rather than bool so that
      if statements work even during Jax tracing.
  """

  value: jax.Array
  dr: jax.Array
  left_face_consx: jax.Array
  left_face_consx_is_grad: bool
  right_face_consx: jax.Array
  right_face_consx_is_grad: bool
  history: Optional[bool] = None

  @classmethod
  def of(cls, value: jax.Array, dr: jax.Array,
         left_face_constraint: Optional[jax.Array] = None, left_face_grad_constraint: Optional[jax.Array] = None,
         right_face_constraint: Optional[jax.Array] = None, right_face_grad_constraint: Optional[jax.Array] = None,
         ) -> CellVariable:
    if left_face_constraint is None:
      if left_face_grad_constraint is None:
        raise ValueError('Exactly one of left_face_constraint or left_face_grad_constraint must be specified')
      left_face_constraint = left_face_grad_constraint
      left_face_constraint_is_grad = True
    else:
      if left_face_grad_constraint is not None:
        raise ValueError('Exactly one of left_face_constraint or left_face_grad_constraint can be specified')
      left_face_constraint_is_grad = False

    if right_face_constraint is None:
      if right_face_grad_constraint is None:
        raise ValueError('Exactly one of right_face_constraint or right_face_grad_constraint must be specified')
      right_face_constraint = right_face_grad_constraint
      right_face_constraint_is_grad = True
    else:
      if right_face_grad_constraint is not None:
        raise ValueError('Exactly one of right_face_constraint or right_face_grad_constraint can be specified')
      right_face_constraint_is_grad = False
    return cls(value=value, dr=dr,
        left_face_consx=left_face_constraint, left_face_consx_is_grad=left_face_constraint_is_grad,
        right_face_consx=right_face_constraint, right_face_consx_is_grad=right_face_constraint_is_grad
    )

  def project(self, weights):
    assert self.history is not None

    def project(x):
      return jnp.dot(weights, x)

    return dataclasses.replace(
        self,
        value=project(self.value),
        dr=self.dr[0],
        left_face_consx=project(self.left_face_consx),
        right_face_consx=project(self.right_face_consx),
        history=None,
    )

  def __post_init__(self):
    self.sanity_check()

  def sanity_check(self):
    """Check that the CellVariable is valid.

    How is `sanity_check` different from `__post_init__`?
    - `sanity_check` is exposed to the client directly, so the client can
    explicitly check sanity without violating privacy conventions. This is
    useful for checking objects that were created e.g. using jax tree
    transformations.
    - `sanity_check` is guaranteed not to change the object, while
    `__post_init__` could in principle make changes.
    """
    # Automatically check dtypes of all numeric fields
    for name, value in self.items():
      if isinstance(value, jax.Array) and value.dtype != jnp.bool_:
        if value.dtype != jnp.float64 and jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float64, got dtype {value.dtype} for `{name}`'
          )
        if value.dtype != jnp.float32 and not jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float32, got dtype {value.dtype} for `{name}`'
          )
    if self.history is None:
      # jax compilation seems to need to make a dummy version of this class with
      # (,) passed in for the value, so unfortunately we can't include this
      # assert.
      # chex.assert_rank(self.value, 1)
      chex.assert_rank(self.dr, 0)

  def face_grad(self, x: jax.Array | None = None) -> jax.Array:
    """Returns the gradient of this value with respect to the faces.

    Implemented using forward differencing of cells. Leftmost and rightmost
    gradient entries are determined by user specify constraints, see
    CellVariable class docstring.

    Args:
      x: (optional) coordinates over which differentiation is carried out

    Returns:
      A jax.Array of shape (num_faces,) containing the gradient.
    """
    self.assert_not_history()
    if x is None:
      forward_difference = jnp.diff(self.value) / self.dr
    else:
      forward_difference = jnp.diff(self.value) / jnp.diff(x)

    def constrained_grad(
        face: Optional[jax.Array],
        grad: Optional[jax.Array],
        cell: jax.Array,
        right: bool,
    ) -> jax.Array:
      """Calculates the constrained gradient entry for an outer face.

      Args:
        face: Optional, constraint on the value of the face variable.
        grad: Optional, constraint on the gradient wrt the face variable.
          Exactly one of face and grad must be specified.
        cell: The value of the neighboring cell variable.
        right: If True, this is the rightmost face, else the leftmost.

      Returns:
        The gradient on this face variable.
      """

      if face is not None:
        if grad is not None:
          raise ValueError(
              'Cannot constraint both the value and gradient of '
              'a face variable.'
          )
        if x is None:
          dx = self.dr
        else:
          if right:
            dx = x[-1] - x[-2]
          else:
            dx = x[1] - x[0]
        return (1.0 - 2.0 * right) * (cell - face) / (0.5 * dx)
      else:
        if grad is None:
          raise ValueError('Must specify one of value or gradient.')
        return grad

    left_grad = constrained_grad(
        self.left_face_constraint,
        self.left_face_grad_constraint,
        self.value[0],
        right=False,
    )
    right_grad = constrained_grad(
        self.right_face_constraint,
        self.right_face_grad_constraint,
        self.value[-1],
        right=True,
    )

    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, forward_difference, right])

  def face_value(self) -> jax.Array:
    """Calculates values of this variable at faces.

    Returns:
      Values of the variable at faces.
    """
    self.assert_not_history()
    inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
    if not self.left_face_consx_is_grad:
      left_face = jnp.array([self.left_face_constraint])
    else:
      # When there is no constraint, leftmost face equals
      # leftmost cell
      left_face = self.value[..., 0:1]
    if not self.right_face_consx_is_grad:
      right_face = jnp.array([self.right_face_constraint])
    else:
      # Maintain right_face consistent with right_face_grad_constraint
      right_face = (
          self.value[..., -1:] + self.right_face_grad_constraint * self.dr / 2
      )
    return jnp.concatenate([left_face, inner, right_face], axis=-1)

  def grad(self) -> jax.Array:
    """Returns the gradient of this variable wrt cell centers."""

    self.assert_not_history()
    face = self.face_value()
    return jnp.diff(face) / self.dr

  def history_elem(self) -> CellVariable:
    """Return a history entry version of this CellVariable."""
    return dataclasses.replace(self, history=True)

  def assert_not_history(self):
    """Assert that the CellVariable is not a history."""
    # We must say "is not None" to avoid concretization error
    if self.history is not None:
      msg = (
          'This CellVariable instance is in "history" '
          'mode in a context that it was not expected to '
          'be. History mode is when several CellVariables '
          'have had their leaf values stacked to form a '
          'new CellVariable with an extra time axis, such as'
          'by `jax.lax.scan`. Most methods of a CellVariable '
          'do not work in history mode.'
      )
      if hasattr(self.history, 'ndim'):
        if self.history.ndim == 0 or (
            self.history.ndim == 1 and self.history.shape[0] == 1
        ):
          msg += (
              f' self.history={self.history} which probably indicates'
              ' (due to its scalar shape)'
              ' that an indexing or projection operation failed to'
              ' turn off history mode. self.history should be None for'
              ' non-history or a a vector of shape (history_length) for'
              ' history.'
          )
      raise AssertionError(msg)

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return self is other

  @property
  def left_face_constraint(self):
    return None if jnp.any(self.left_face_consx_is_grad) else self.left_face_consx

  @property
  def left_face_grad_constraint(self):
    return self.left_face_consx if jnp.any(self.left_face_consx_is_grad) else None

  @property
  def right_face_constraint(self):
    return None if jnp.any(self.right_face_consx_is_grad) else self.right_face_consx

  @property
  def right_face_grad_constraint(self):
    return self.right_face_consx if jnp.any(self.right_face_consx_is_grad) else None
