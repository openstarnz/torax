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
"""Runtime params for the stepper."""
import chex


@chex.dataclass(frozen=True)
class StaticRuntimeParams:
  """Static params for the stepper."""

  theta_imp: float
  convection_dirichlet_mode: str
  convection_neumann_mode: str
  use_pereverzev: bool
  predictor_corrector: bool


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the stepper which can be used as compiled args."""

  chi_per: float
  d_per: float
  corrector_steps: int
