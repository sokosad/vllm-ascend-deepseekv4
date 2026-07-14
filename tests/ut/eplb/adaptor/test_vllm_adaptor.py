import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.quantization.methods.w8a8_dynamic import _reshape_fused_expert_scale
from transformers import DeepseekV2Config


class TestVllmAdaptor(unittest.TestCase):
    def setUp(self):
        n_routed_experts = 256
        mock_model = MagicMock()
        mock_model.model.named_parameters.return_value = dict()
        config = DeepseekV2Config(n_routed_experts=n_routed_experts)
        config.first_k_dense_replace = 0
        config.num_hidden_layers = 2
        mock_model.config = config
        mock_model.get_expert_map.return_value = [i for i in range(n_routed_experts)]
        mock_model.get_log2phy_map.return_value = [i for i in range(n_routed_experts)]
        self.model = mock_model
        for layer_idx in range(config.num_hidden_layers):
            experts = self.model.model.layers[layer_idx].mlp.experts
            experts.local_num_experts = 2
            experts.craft_pool_enabled = False
            experts.quant_type = QuantType.W8A8

        self.mock_rank = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_rank", return_value=0).start()
        self.mock_size = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_world_size", return_value=4).start()

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_fp16(self, mock_func):
        self.model.quant_config = None
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_w8a8(self, mock_func):
        VllmEplbAdaptor(self.model)

    def test_pool_fused_scales_use_per_expert_views(self):
        adaptor = object.__new__(VllmEplbAdaptor)
        adaptor.expert_weight_names = ["fused_w1_scale_list", "fused_w2_scale_list"]
        adaptor.num_local_experts_per_layer = {0: 2}
        adaptor.param_dict = {}
        experts = SimpleNamespace(
            local_num_experts_main=1,
            fused_w1_scale_list=[torch.zeros(4096, dtype=torch.int64)],
            fused_w2_scale_list=[torch.zeros(4096, dtype=torch.int64)],
            fused_w1_scale_pool=torch.zeros(4096, dtype=torch.int64),
            fused_w2_scale_pool=torch.zeros(4096, dtype=torch.int64),
            fused_w1_scale_pool_list=[torch.ones(4096, dtype=torch.int64)],
            fused_w2_scale_pool_list=[torch.ones(4096, dtype=torch.int64)],
        )
        adaptor.expert_param_per_layer = {0: []}

        adaptor._init_pool_expert_param_for_layer(0, experts)

        self.assertEqual(adaptor.expert_param_per_layer[0][0][0].shape, torch.Size([4096]))
        self.assertEqual(adaptor.expert_param_per_layer[0][1][0].shape, torch.Size([4096]))
        self.assertIs(adaptor.expert_param_per_layer[0][1][0], experts.fused_w1_scale_pool_list[0])

    def test_fused_pool_scale_is_partitioned_before_pool_slice(self):
        scale = torch.arange(33 * 4096, dtype=torch.int64)

        reshaped = _reshape_fused_expert_scale(scale, 33)

        self.assertEqual(reshaped.shape, torch.Size([33, 4096]))
        self.assertEqual(reshaped[32:].shape, torch.Size([1, 4096]))

    def test_log2phy_update_preserves_captured_tensor(self):
        adaptor = object.__new__(VllmEplbAdaptor)
        captured_log2phy = torch.full((4, 2), -1, dtype=torch.int32)
        adaptor.log2phy_map_per_layer = {3: captured_log2phy}
        updated_log2phy = torch.tensor(
            [[0, 4], [1, -1], [2, -1], [3, -1]], dtype=torch.int32
        )

        adaptor.do_update_log2phy_map(3, updated_log2phy)

        self.assertIs(adaptor.log2phy_map_per_layer[3], captured_log2phy)
        self.assertTrue(torch.equal(captured_log2phy, updated_log2phy))

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()

if __name__ == "__main__":
    unittest.main()
