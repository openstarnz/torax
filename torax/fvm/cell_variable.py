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
import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax import array_typing
from typing import Optional
from torax import jax_utils


def _zero() -> array_typing.ScalarFloat:
  """Returns a scalar zero as a jax Array."""
  return jnp.zeros(())


@chex.dataclass(frozen=True)
class CellVariable:
  """A variable representing values of the cells along the radius.

  Attributes:
    value: A jax.Array containing the value of this variable at each cell.
    dr: Distance between cell centers.
    left_face_constraint: An optional jax scalar specifying the value of the
      leftmost face. Defaults to None, signifying no constraint. The user can
      modify this field at any time, but when face_grad is called exactly one of
      left_face_constraint and left_face_grad_constraint must be None.
    left_face_grad_constraint: An optional jax scalar specifying the (otherwise
      underdetermined) value of the leftmost face. See left_face_constraint.
    right_face_constraint: Analogous to left_face_constraint but for the right
      face, see left_face_constraint.
    right_face_grad_constraint: A jax scalar specifying the undetermined value
      of the gradient on the rightmost face variable.
  """
  value: jax.Array
  dr: array_typing.ScalarFloat
  left_face_constraint: array_typing.ScalarFloat
  left_face_constraint_is_grad: bool
  right_face_constraint: array_typing.ScalarFloat
  right_face_constraint_is_grad: bool

  @classmethod
  def of(cls, value: jax.Array, dr: jax.Array,
         left_face_value_constraint: Optional[jax.Array] = None, left_face_grad_constraint: Optional[jax.Array] = None,
         right_face_value_constraint: Optional[jax.Array] = None, right_face_grad_constraint: Optional[jax.Array] = None,
  ) -> 'CellVariable':
    if left_face_value_constraint is None:
      if left_face_grad_constraint is None:
        raise ValueError('Exactly one of left_face_value_constraint or left_face_grad_constraint must be specified')
      left_face_constraint = left_face_grad_constraint
      left_face_constraint_is_grad = True
    else:
      if left_face_grad_constraint is not None:
        raise ValueError('Exactly one of left_face_value_constraint or left_face_grad_constraint can be specified')
      left_face_constraint = left_face_value_constraint
      left_face_constraint_is_grad = False

    if right_face_value_constraint is None:
      if right_face_grad_constraint is None:
        raise ValueError('Exactly one of right_face_value_constraint or right_face_grad_constraint must be specified')
      right_face_constraint = right_face_grad_constraint
      right_face_constraint_is_grad = True
    else:
      if right_face_grad_constraint is not None:
        raise ValueError('Exactly one of right_face_value_constraint or right_face_grad_constraint can be specified')
      right_face_constraint = right_face_value_constraint
      right_face_constraint_is_grad = False
    return cls(value=value, dr=dr,
        left_face_constraint=left_face_constraint, left_face_constraint_is_grad=left_face_constraint_is_grad,
        right_face_constraint=right_face_constraint, right_face_constraint_is_grad=right_face_constraint_is_grad,
    )

  def __post_init__(self):
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
    # TODO: Modify these
    for name, value in self.items():
      if isinstance(value, jax.Array):
        if value.dtype != jnp.float64 and jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float64, got dtype {value.dtype} for `{name}`'
          )
        if value.dtype != jnp.float32 and not jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float32, got dtype {value.dtype} for `{name}`'
          )

  def _assert_unbatched(self):
    if len(self.value.shape) != 1:
      raise AssertionError(
          'CellVariable must be unbatched, but has `value` shape '
          f'{self.value.shape}. Consider using vmap to batch the function call.'
      )
    if self.dr.shape:
      raise AssertionError(
          'CellVariable must be unbatched, but has `dr` shape '
          f'{self.dr.shape}. Consider using vmap to batch the function call.'
      )

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
    self._assert_unbatched()
    if x is None:
      forward_difference = jnp.diff(self.value) / self.dr
    else:
      forward_difference = jnp.diff(self.value) / jnp.diff(x)

    def constrained_grad(
        constraint: jax.Array,
        constraint_is_grad: bool,
        cell: jax.Array,
        right: bool,
    ) -> jax.Array:
      """Calculates the constrained gradient entry for an outer face."""

      if x is None:
        dx = self.dr
      else:
        if right:
          dx = x[-1] - x[-2]
        else:
          dx = x[1] - x[0]
      unconstrained_grad = (1.0 - 2.0 * right) * (cell - constraint) / (0.5 * dx)
      return jax.lax.cond(constraint_is_grad, lambda: constraint, lambda: unconstrained_grad)

    left_grad = constrained_grad(
        self.left_face_constraint,
        self.left_face_constraint_is_grad,
        self.value[0],
        right=False,
    )
    right_grad = constrained_grad(
        self.right_face_constraint,
        self.right_face_constraint_is_grad,
        self.value[-1],
        right=True,
    )

    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, forward_difference, right])

  def face_value(self) -> jax.Array:
    """Calculates values of this variable at faces."""
    self._assert_unbatched()
    inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
    left_face = jax.lax.cond(
        self.left_face_constraint_is_grad,
        lambda: self.value[..., 0:1] - self.left_face_constraint * self.dr / 2,
        lambda: jnp.array([self.left_face_constraint]),
    )
    right_face = jax.lax.cond(
        self.right_face_constraint_is_grad,
        lambda: self.value[..., -1:] + self.right_face_constraint * self.dr / 2,
        lambda: jnp.array([self.right_face_constraint]),
    )
    return jnp.concatenate([left_face, inner, right_face], axis=-1)

  def grad(self) -> jax.Array:
    """Returns the gradient of this variable wrt cell centers."""
    face = self.face_value()
    return jnp.diff(face) / self.dr

  @property
  def left_face_value_constraint(self):
    return jax_utils.error_if(self.left_face_constraint, jnp.any(self.left_face_constraint_is_grad),
                              'left_face_constraint')

  @property
  def left_face_grad_constraint(self):
    return jax_utils.error_if(self.left_face_constraint, jnp.logical_not(jnp.any(self.left_face_constraint_is_grad)),
                              'left_face_constraint')

  @property
  def right_face_value_constraint(self):
    return jax_utils.error_if(self.right_face_constraint, jnp.any(self.right_face_constraint_is_grad),
                              'right_face_constraint')

  @property
  def right_face_grad_constraint(self):
    return jax_utils.error_if(self.right_face_constraint, jnp.logical_not(jnp.any(self.right_face_constraint_is_grad)),
                              'right_face_constraint')

  def __str__(self) -> str:
    output_string = f'CellVariable(value={self.value}'
    if self.left_face_constraint is not None:
      output_string += (
          f', left_face_constraint={self.left_face_constraint}'
      )
    if self.right_face_constraint is not None:
      output_string += (
          f', right_face_constraint={self.right_face_constraint}'
      )
    if self.left_face_grad_constraint is not None:
      output_string += (
          f', left_face_grad_constraint={self.left_face_grad_constraint}'
      )
    if self.right_face_grad_constraint is not None:
      output_string += (
          f', right_face_grad_constraint={self.right_face_grad_constraint}'
      )
    output_string += ')'
    return output_string
