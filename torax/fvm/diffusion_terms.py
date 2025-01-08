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

"""The `make_diffusion_terms` function.

Builds the diffusion terms of the discrete matrix equation.
"""

import chex
from jax import numpy as jnp
from torax import math_utils
from torax import jax_utils
from torax.fvm import cell_variable


def make_diffusion_terms(
    d_face: chex.Array, var: cell_variable.CellVariable
) -> tuple[chex.Array, chex.Array]:
  """Makes the terms of the matrix equation derived from the diffusion term.

  The diffusion term is of the form
  (partial / partial x) D partial x / partial x

  Args:
    d_face: Diffusivity coefficient on faces.
    var: CellVariable (to define geometry and boundary conditions)

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """

  # Start by using the formula for the interior rows everywhere
  denom = var.dr**2
  diag = jnp.asarray(-d_face[1:] - d_face[:-1])

  off = d_face[1:-1]
  vec = jnp.zeros_like(diag)

  if vec.shape[0] < 2:
    raise NotImplementedError(
        'We do not support the case where a single cell'
        ' is affected by both boundary conditions.'
    )

  # Boundary rows need to be special-cased.
  #
  # Check that the boundary conditions are well-posed.
  # These checks are redundant with CellVariable.__post_init__, but including
  # them here for readability because they're in important part of the logic
  # of this function.
  chex.assert_exactly_one_is_none(
      var.left_face_grad_constraint, var.left_face_constraint
  )
  chex.assert_exactly_one_is_none(
      var.right_face_grad_constraint, var.right_face_constraint
  )

  def left_dirichlet():
    # Left face Dirichlet condition
    diag_value = -2 * d_face[0] - d_face[1]
    vec_value = 2 * d_face[0] * var.left_face_constraint / denom
    return diag_value, vec_value
  def left_gradient():
    # Left face gradient condition
    diag_value = -d_face[1]
    vec_value = -d_face[0] * var.left_face_grad_constraint / var.dr
    return diag_value, vec_value
  def right_dirichlet():
    # Right face Dirichlet condition
    diag_value = -2 * d_face[-1] - d_face[-2]
    vec_value = 2 * d_face[-1] * var.right_face_constraint / denom
    return diag_value, vec_value
  def right_gradient():
    # Right face gradient constraint
    diag_value = -d_face[-2]
    vec_value = d_face[-1] * var.right_face_grad_constraint / var.dr
    return diag_value, vec_value

  diag_value, vec_value = jax_utils.py_cond(var.left_face_consx_is_grad, left_gradient, left_dirichlet)
  diag = diag.at[0].set(diag_value)
  vec = vec.at[0].set(vec_value)
  diag_value, vec_value = jax_utils.py_cond(var.right_face_consx_is_grad, right_gradient, right_dirichlet)
  diag = diag.at[-1].set(diag_value)
  vec = vec.at[-1].set(vec_value)

  # Build the matrix
  mat = math_utils.tridiag(diag, off, off) / denom
  return mat, vec
