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
import jax
from jax import numpy as jnp
from torax import math_utils
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
  # Left face
  diag = diag.at[0].set(jax.lax.cond(
    var.left_face_constraint_is_grad,
    lambda: -d_face[1],
    lambda: -2 * d_face[0] - d_face[1],
  ))
  vec = vec.at[0].set(jax.lax.cond(
    var.left_face_constraint_is_grad,
    lambda: -d_face[0] * var.left_face_constraint / var.dr,
    lambda: 2 * d_face[0] * var.left_face_constraint / denom,
  ))
  # Right face
  diag = diag.at[-1].set(jax.lax.cond(
    var.right_face_constraint_is_grad,
    lambda: -d_face[-2],
    lambda: -2 * d_face[-1] - d_face[-2],
  ))
  vec = vec.at[-1].set(jax.lax.cond(
    var.right_face_constraint_is_grad,
    lambda: d_face[-1] * var.right_face_constraint / var.dr,
    lambda: 2 * d_face[-1] * var.right_face_constraint / denom,
  ))

  # Build the matrix
  mat = math_utils.tridiag(diag, off, off) / denom
  return mat, vec
