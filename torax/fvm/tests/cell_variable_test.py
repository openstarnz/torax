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
from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from torax.fvm import cell_variable


class CellVariableTest(absltest.TestCase):

  def test_unconstrained_left_raises_an_error(self):
    with self.assertRaisesRegex(ValueError, 'left_face_constraint'):
      cell_variable.CellVariable.of(
          value=jnp.array([1.0, 2.0, 5.0, 3.0]),
          dr=jnp.array(0.1),
          left_face_value_constraint=None,
          left_face_grad_constraint=None,
          right_face_grad_constraint=jnp.array(0.0),
      )

  def test_unconstrained_right_raises_an_error(self):
    with self.assertRaisesRegex(ValueError, 'right_face_constraint'):
      cell_variable.CellVariable.of(
          value=jnp.array([1.0, 2.0, 5.0, 3.0]),
          dr=jnp.array(0.1),
          right_face_value_constraint=None,
          right_face_grad_constraint=None,
      )

  def test_overconstrained_left_raises_an_error(self):
    with self.assertRaisesRegex(ValueError, 'left_face_constraint'):
      cell_variable.CellVariable.of(
          value=jnp.array([1.0, 2.0, 5.0, 3.0]),
          dr=jnp.array(0.1),
          left_face_value_constraint=jnp.array(1.0),
          left_face_grad_constraint=jnp.array(1.0),
          right_face_grad_constraint=jnp.array(0.0),
      )

  def test_overconstrained_right_raises_an_error(self):
    with self.assertRaisesRegex(ValueError, 'right_face_constraint'):
      cell_variable.CellVariable.of(
          value=jnp.array([1.0, 2.0, 5.0, 3.0]),
          dr=jnp.array(0.1),
          left_face_grad_constraint=jnp.array(0.0),
          right_face_value_constraint=jnp.array(1.0),
          right_face_grad_constraint=jnp.array(1.0),
      )

  def test_face_grad_unconstrained_no_input(self):
    var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
    )

    grad = var.face_grad()
    np.testing.assert_array_equal(grad, jnp.array([0., 10., 30., -20., 0.]))

  def test_face_grad_unconstrained_with_input(self):
    var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
    )

    grad = var.face_grad(x=jnp.array([4.0, 1.0, 5.0, 3.0]))
    np.testing.assert_array_equal(
        grad, jnp.array([0., 1.0 / -3.0, 3.0 / 4.0, -2.0 / -2.0, 0.]))

  def test_face_grad_grad_constraint(self):
    var = cell_variable.CellVariable.of(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
        left_face_grad_constraint=jnp.array(1.0),
        right_face_grad_constraint=jnp.array(2.0),
    )
    grad = var.face_grad()
    np.testing.assert_array_equal(grad, jnp.array([1.0, 10., 30., -20., 2.0]))

  def test_face_grad_value_constraint(self):
    dr = 0.1
    var = cell_variable.CellVariable.of(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(dr),
        left_face_value_constraint=jnp.array(2.0),
        right_face_value_constraint=jnp.array(5.0),
    )
    grad = var.face_grad()
    left_grad = -1 / (0.5 * dr)
    right_grad = 2 / (0.5 * dr)
    np.testing.assert_array_equal(
        grad, jnp.array([left_grad, 10.0, 30.0, -20.0, right_grad])
    )

  def test_face_value_unconstrained(self):
    var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
    )
    value = var.face_value()
    np.testing.assert_array_equal(value, jnp.array([1.0, 1.5, 3.5, 4.0, 3.0]))

  def test_face_value_value_constrained(self):
    var = cell_variable.CellVariable.of(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
        left_face_value_constraint=jnp.array(2.0),
        right_face_value_constraint=jnp.array(5.0),
    )
    value = var.face_value()
    np.testing.assert_array_equal(value, jnp.array([2.0, 1.5, 3.5, 4.0, 5.0]))

  def test_grad_unconstrained(self):
    var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
    )
    grad = var.grad()
    np.testing.assert_array_equal(grad, jnp.array([5.0, 20.0, 5.0, -10.0]))

  def test_batched_core_profiles_raises_error_on_invalid_method_call(self):
    var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 5.0, 3.0]),
        dr=jnp.array(0.1),
    )
    batched_var: cell_variable.CellVariable = jax.tree_util.tree_map(
        lambda *ys: np.stack(ys), *[var, var],
    )
    print(batched_var)
    with self.subTest('raises error on face_grad'):
      with self.assertRaises(AssertionError):
        batched_var.face_grad()
    with self.subTest('raises error on face_value'):
      with self.assertRaises(AssertionError):
        batched_var.face_value()


if __name__ == '__main__':
  absltest.main()
