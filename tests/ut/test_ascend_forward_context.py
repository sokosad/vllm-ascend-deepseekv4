import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import AscendDeviceType


class TestCraftPoolCommSelection(unittest.TestCase):
    @staticmethod
    def _config(policy_type: int, pool_size: int):
        eplb_config = SimpleNamespace(
            expert_map_path="",
            eplb_policy_type=policy_type,
            craft_pool_size=pool_size,
            craft_pool_layer_sizes=None,
            metro_routing=False,
        )
        parallel_config = SimpleNamespace(
            eplb_config=eplb_config,
            enable_expert_parallel=True,
            world_size_across_dp=8,
            pipeline_parallel_size=1,
        )
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(moe_quantize="w8a8"),
            get_num_experts=lambda: 256,
        )
        return SimpleNamespace(
            parallel_config=parallel_config,
            model_config=model_config,
            additional_config={
                "eplb_config": {
                    "eplb_policy_type": policy_type,
                    "craft_pool_size": pool_size,
                }
            },
        )

    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    def test_policy4_pool_uses_fused_mc2_when_enabled(self, _mock_moe, mock_ep_group, _mock_device):
        mock_ep_group.return_value = MagicMock(world_size=8)
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=1024),
        ):
            result = select_moe_comm_method(32, self._config(policy_type=4, pool_size=1))

        self.assertEqual(result, MoECommType.FUSED_MC2)

    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    def test_policy4_large_prefill_uses_compact_pool_allgather(self, _mock_moe, mock_ep_group):
        mock_ep_group.return_value = MagicMock(world_size=8)
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=256),
        ):
            result = select_moe_comm_method(16384, self._config(policy_type=4, pool_size=1))

        self.assertEqual(result, MoECommType.ALLGATHER)

    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    def test_policy4_pool_stays_allgather_without_fused_mc2(self, _mock_moe, mock_ep_group):
        mock_ep_group.return_value = MagicMock(world_size=8)
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "0"}),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=1024),
        ):
            result = select_moe_comm_method(32, self._config(policy_type=4, pool_size=1))

        self.assertEqual(result, MoECommType.ALLGATHER)

    @patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    def test_policy2_selection_is_unchanged(self, _mock_moe, mock_ep_group, _mock_device):
        mock_ep_group.return_value = MagicMock(world_size=8)
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=1024),
        ):
            result = select_moe_comm_method(32, self._config(policy_type=2, pool_size=0))

        self.assertEqual(result, MoECommType.FUSED_MC2)

    @patch("vllm_ascend.ascend_forward_context.get_ep_group")
    @patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True)
    def test_metro_routing_remains_isolated_from_policy4_fused(self, _mock_moe, mock_ep_group):
        mock_ep_group.return_value = MagicMock(world_size=8)
        config = self._config(policy_type=2, pool_size=0)
        config.parallel_config.eplb_config.metro_routing = True
        config.additional_config["eplb_config"]["metro_routing"] = True
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=1024),
        ):
            result = select_moe_comm_method(32, config)

        self.assertEqual(result, MoECommType.ALLGATHER)

    @patch("vllm_ascend.platform.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_full_graph_support_matches_fused_setting(self, _mock_device):
        parallel_config = SimpleNamespace(
            enable_expert_parallel=True,
            tensor_parallel_size=4,
            data_parallel_size=2,
        )
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}):
            self.assertTrue(
                NPUPlatform._craft_pool_supports_full_graph(
                    parallel_config, SimpleNamespace(eplb_policy_type=4)
                )
            )
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "0"}):
            self.assertFalse(
                NPUPlatform._craft_pool_supports_full_graph(
                    parallel_config, SimpleNamespace(eplb_policy_type=4)
                )
            )
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}):
            self.assertFalse(
                NPUPlatform._craft_pool_supports_full_graph(
                    parallel_config, SimpleNamespace(eplb_policy_type=2)
                )
            )


if __name__ == "__main__":
    unittest.main()
