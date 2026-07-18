import unittest
import os
from typing import ClassVar
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.moe_mlp import (
    _expert_count_from_weight,
    cumsum_group_list,
    unified_apply_mlp,
)
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendFusedMoE,
    _copy_to_craft_graph_buffer,
    _craft_graph_buffer,
    _is_craft_pool_graph_capturing,
    _is_craft_pool_graph_mode,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.ascend_forward_context import MoECommType


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
    def test_pool_weight_mapping_ignores_synthetic_physical_ids(self):
        layer = object.__new__(AscendFusedMoE)
        torch.nn.Module.__init__(layer)
        layer._expert_map = torch.tensor([0, -1, 1], dtype=torch.int32)
        layer.craft_pool_enabled = True

        self.assertEqual(layer._map_global_expert_id_to_local_expert_id(2), 1)
        self.assertEqual(layer._map_global_expert_id_to_local_expert_id(3), -1)

    def test_non_pool_weight_mapping_preserves_vllm_bounds_check(self):
        layer = object.__new__(AscendFusedMoE)
        torch.nn.Module.__init__(layer)
        layer._expert_map = torch.tensor([0, -1, 1], dtype=torch.int32)
        layer.craft_pool_enabled = False

        with self.assertRaises(IndexError):
            layer._map_global_expert_id_to_local_expert_id(3)

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

    def test_fused_mc2_capture_executes_real_pool_moe(self):
        class Layer:
            craft_pool_enabled = True

        layer = Layer()
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch("vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.moe_comm_type = MoECommType.FUSED_MC2
            extra_ctx.capturing = True
            extra_ctx.graph_capture_forward = True
            extra_ctx.graph_buffer_warmup = False

            self.assertFalse(_is_craft_pool_graph_capturing(layer))

    def test_fused_mc2_does_not_retain_pool_graph_shadow_buffer(self):
        class Layer:
            craft_pool_enabled = True

        layer = Layer()
        value = torch.randn(2, 4)
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch("vllm_ascend.ops.fused_moe.fused_moe._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.moe_comm_type = MoECommType.FUSED_MC2

            self.assertFalse(_is_craft_pool_graph_mode(layer))
            self.assertIs(_copy_to_craft_graph_buffer(layer, "routed", value), value)
            self.assertFalse(hasattr(layer, "_craft_graph_routed_buffers"))

    def test_graph_buffer_is_zero_initialized(self):
        class Layer:
            pass

        ref = torch.randn(2, 4)
        buf = _craft_graph_buffer(Layer(), "routed", ref)

        self.assertEqual(buf.shape, ref.shape)
        self.assertEqual(buf.dtype, ref.dtype)
        self.assertTrue(torch.equal(buf, torch.zeros_like(ref)))


class TestW8A8PolicyIsolation(unittest.TestCase):
    @staticmethod
    def _method():
        method = object.__new__(AscendW8A8DynamicFusedMoEMethod)
        method.dynamic_eplb = True
        method.multistream_overlap_gate = False
        method.in_dtype = torch.float32
        method.quant_type = QuantType.W8A8
        return method

    @staticmethod
    def _base_layer():
        layer = torch.nn.Module()
        layer.swiglu_limit = 0
        layer.force_load_balance_routed_topk_ids = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int64
        )
        return layer

    def test_policy2_profile_keeps_original_router_path(self):
        method = self._method()
        layer = self._base_layer()
        layer.w13_weight_list = [torch.empty(16, 8) for _ in range(2)]
        layer.w2_weight_list = [torch.empty(8, 8) for _ in range(2)]
        layer.w13_weight_scale_fp32_list = [torch.ones(16) for _ in range(2)]
        layer.w2_weight_scale_list = [torch.ones(8) for _ in range(2)]
        router_ids = torch.tensor([[7, 6], [5, 4], [3, 2], [1, 0]])
        router_weights = torch.full((4, 2), 0.5)

        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8_dynamic.select_experts",
                return_value=(router_weights, router_ids),
            ) as mock_select,
            patch(
                "vllm_ascend.quantization.methods.w8a8_dynamic.build_force_load_balance_routing"
            ) as mock_force,
            patch("vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.moe_comm_type = MoECommType.ALLGATHER
            extra_ctx.moe_comm_method.fused_experts.return_value = torch.empty(4, 8)
            method.apply(
                layer=layer,
                x=torch.empty(4, 8),
                router_logits=torch.empty(4, 8),
                top_k=2,
                renormalize=True,
                global_num_experts=8,
                enable_force_load_balance=True,
            )

        mock_select.assert_called_once()
        mock_force.assert_not_called()
        request = extra_ctx.moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(request.topk_weights, router_weights)
        self.assertTrue(torch.equal(request.topk_ids, layer.force_load_balance_routed_topk_ids))

    def test_policy4_compact_pool_uses_graph_profile_routing(self):
        method = self._method()
        layer = self._base_layer()
        layer.local_num_experts_pool = 1
        layer.w13_weight_with_pool = torch.empty(3, 16, 8)
        layer.w2_weight_with_pool = torch.empty(3, 8, 8)
        layer.w13_weight_scale_fp32_with_pool = torch.ones(3, 16)
        layer.w2_weight_scale_with_pool = torch.ones(3, 8)
        layer.craft_expert_token_nums = None
        forced_weights = torch.full((4, 2), 0.5)
        forced_ids = layer.force_load_balance_routed_topk_ids

        with (
            patch("vllm_ascend.quantization.methods.w8a8_dynamic.select_experts") as mock_select,
            patch(
                "vllm_ascend.quantization.methods.w8a8_dynamic.build_force_load_balance_routing",
                return_value=(forced_weights, forced_ids),
            ) as mock_force,
            patch("vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.moe_comm_type = MoECommType.ALLGATHER
            extra_ctx.moe_comm_method.fused_experts.return_value = torch.empty(4, 8)
            method.apply(
                layer=layer,
                x=torch.empty(4, 8),
                router_logits=torch.empty(4, 8),
                top_k=2,
                renormalize=True,
                global_num_experts=8,
                enable_force_load_balance=True,
            )

        mock_force.assert_called_once()
        mock_select.assert_not_called()

    def test_policy4_cold_layer_uses_layer_local_fused_token_buffer(self):
        method = self._method()
        layer = self._base_layer()
        layer.local_num_experts_pool = 0
        layer.craft_pool_enabled = True
        layer.w13_weight_list = [torch.empty(16, 8) for _ in range(2)]
        layer.w2_weight_list = [torch.empty(8, 8) for _ in range(2)]
        layer.fused_w1_scale_list = [torch.ones(16) for _ in range(2)]
        layer.fused_w2_scale_list = [torch.ones(8) for _ in range(2)]
        layer.craft_expert_token_nums = torch.zeros((1, 2), dtype=torch.int32)
        router_ids = torch.tensor([[7, 6], [5, 4], [3, 2], [1, 0]])
        router_weights = torch.full((4, 2), 0.5)

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FUSED_MC2": "1"}),
            patch(
                "vllm_ascend.quantization.methods.w8a8_dynamic.select_experts",
                return_value=(router_weights, router_ids),
            ),
            patch("vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.moe_comm_type = MoECommType.FUSED_MC2
            extra_ctx.moe_comm_method.fused_experts.return_value = torch.empty(4, 8)
            method.apply(
                layer=layer,
                x=torch.empty(4, 8),
                router_logits=torch.empty(4, 8),
                top_k=2,
                renormalize=True,
                global_num_experts=8,
            )

        request = extra_ctx.moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertIs(request.expert_token_nums, layer.craft_expert_token_nums)


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
