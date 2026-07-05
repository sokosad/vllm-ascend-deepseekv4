import unittest
from typing import ClassVar
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.moe_mlp import (
    _expert_count_from_weight,
    cumsum_group_list,
    unified_apply_mlp,
)
from vllm_ascend.ops.fused_moe.fused_moe import (
    _craft_graph_buffer,
    _is_craft_pool_graph_capturing,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestCraftPoolSplitHelpers(unittest.TestCase):
    def test_dynamic_weight_list_counts_list_entries_as_experts(self):
        per_expert_weights = [torch.empty(16, 8), torch.empty(16, 8), torch.empty(16, 8)]
        self.assertEqual(
            _expert_count_from_weight(per_expert_weights, name="w1", dynamic_eplb=True),
            3,
        )

    def test_wrapped_tensor_counts_tensor_expert_dimension(self):
        wrapped_weight = [torch.empty(3, 16, 8)]
        self.assertEqual(
            _expert_count_from_weight(wrapped_weight, name="w1", dynamic_eplb=False),
            3,
        )

    def test_graph_buffer_warmup_uses_fake_capture_path(self):
        class Layer:
            craft_pool_enabled = True

        layer = Layer()
        with (
            patch("vllm_ascend.ops.fused_moe.fused_moe._is_craft_pool_graph_mode", return_value=True),
            patch("vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.capturing = False
            extra_ctx.graph_capture_forward = False
            extra_ctx.graph_buffer_warmup = True

            self.assertTrue(_is_craft_pool_graph_capturing(layer))

    def test_graph_buffer_is_zero_initialized(self):
        class Layer:
            pass

        ref = torch.randn(2, 4)
        buf = _craft_graph_buffer(Layer(), "routed", ref)

        self.assertEqual(buf.shape, ref.shape)
        self.assertEqual(buf.dtype, ref.dtype)
        self.assertTrue(torch.equal(buf, torch.zeros_like(ref)))


class TestUnifiedApplyMlpRequest(unittest.TestCase):
    def test_request_unquant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_bias=torch.randn(1, 16),
                w2_bias=torch.randn(1, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=(expected, None)) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output[0] is expected)
        mock_unquant.assert_called_once()
        self.assertEqual(mock_unquant.call_args.kwargs["activation"], "silu")
        self.assertFalse(mock_unquant.call_args.kwargs["need_trans"])
        mock_quant.assert_not_called()

    def test_request_quant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(2, 1),
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_scale=[torch.randn(1)],
                w2_scale=[torch.randn(1)],
            ),
            quant=MoEQuantParams(
                quant_type=QuantType.MXFP8,
                mxfp=MoEMxfpParams(
                    act_quant_type=torch.float8_e4m3fn,
                    weight_quant_type=torch.float8_e4m3fn,
                    use_bf16=False,
                ),
            ),
            fusion=True,
            activation="silu",
            need_trans=False,
            dynamic_eplb=True,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=(expected, None)) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output[0] is expected)
        mock_quant.assert_called_once()
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertTrue(quant_kwargs["use_mxfp_quant"])
        self.assertTrue(quant_kwargs["fusion"])
        self.assertTrue(quant_kwargs["dynamic_eplb"])
        self.assertFalse(quant_kwargs["use_bf16"])
        mock_unquant.assert_not_called()

    def test_compact_craft_pool_forces_graph_safe_quant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([1, 1, 0], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(2, 1),
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.randn(3, 16, 8)],
                w2=[torch.randn(3, 8, 8)],
                w1_scale=[torch.randn(3, 1)],
                w2_scale=[torch.randn(3, 1)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            fusion=True,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
            compact_craft_pool=True,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=(expected, None)) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output[0] is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertFalse(quant_kwargs["fusion"])
        self.assertTrue(quant_kwargs["disable_triton_activation"])
        mock_unquant.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
