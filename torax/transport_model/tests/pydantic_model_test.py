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
from absl.testing import parameterized
from torax.transport_model import bohm_gyrobohm
from torax.transport_model import constant
from torax.transport_model import critical_gradient
from torax.transport_model import pydantic_model as transport_pydantic_model
from torax.transport_model import qlknn_transport_model
from torax.transport_model import runtime_params
from torax.transport_model import transport_model as transport_model_lib


class PydanticModelTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          pydantic_model=transport_pydantic_model.QLKNNTransportModel,
          expected_dynamic_params=qlknn_transport_model.DynamicRuntimeParams,
          expected_transport_model=qlknn_transport_model.QLKNNTransportModel,
      ),
      dict(
          pydantic_model=transport_pydantic_model.ConstantTransportModel,
          expected_dynamic_params=constant.DynamicRuntimeParams,
          expected_transport_model=constant.ConstantTransportModel,
      ),
      dict(
          pydantic_model=transport_pydantic_model.CriticalGradientTransportModel,
          expected_dynamic_params=critical_gradient.DynamicRuntimeParams,
          expected_transport_model=critical_gradient.CriticalGradientTransportModel,
      ),
      dict(
          pydantic_model=transport_pydantic_model.BohmGyroBohmTransportModel,
          expected_dynamic_params=bohm_gyrobohm.DynamicRuntimeParams,
          expected_transport_model=bohm_gyrobohm.BohmGyroBohmTransportModel,
      ),
  )
  def test_build_transport_model(
      self,
      pydantic_model: type[transport_pydantic_model.TransportConfig],
      expected_dynamic_params: type[runtime_params.DynamicRuntimeParams],
      expected_transport_model: type[transport_model_lib.TransportModel],
  ):
    transport = pydantic_model()
    dynamic_runtime_params = transport.build_dynamic_params(t=0.0)
    self.assertIsInstance(dynamic_runtime_params, expected_dynamic_params)
    transport_model = transport.build_transport_model()
    self.assertIsInstance(transport_model, expected_transport_model)

  def test_build_qualikiz_transport_model(self):
    try:
      # pylint: disable=g-import-not-at-top
      from torax.transport_model import qualikiz_transport_model
      # pylint: enable=g-import-not-at-top
    except ImportError:
      self.skipTest('Qualikiz transport model is not available.')

    transport = qualikiz_transport_model.QualikizTransportModelConfig()
    dynamic_runtime_params = transport.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_runtime_params, qualikiz_transport_model.DynamicRuntimeParams
    )
    transport_model = transport.build_transport_model()
    self.assertIsInstance(
        transport_model, qualikiz_transport_model.QualikizTransportModel
    )

  def test_qlknn_defaults(self):
    """Tests that correct default values are set for QLKNN."""
    transport = transport_pydantic_model.QLKNNTransportModel()
    self.assertIsInstance(
        transport, transport_pydantic_model.QLKNNTransportModel
    )
    self.assertEqual(transport.smoothing_sigma, 0.1)
    self.assertEqual(transport.ETG_correction_factor, 1.0 / 3.0)


if __name__ == '__main__':
  absltest.main()
