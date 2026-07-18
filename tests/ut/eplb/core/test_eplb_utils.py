import os
import unittest
from unittest.mock import patch

# isort: off
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig

from vllm_ascend.ascend_config import EplbConfig, init_ascend_config
from vllm_ascend.eplb.core.eplb_utils import (
    generate_craft_rank_route_map,
    generate_craft_route_map,
    generate_pool_log2phy_map,
    get_craft_global_pool_size_per_rank,
    get_configured_craft_global_pool_size,
    get_configured_craft_pool_size,
    init_eplb_config,
)
from vllm_ascend.eplb.utils import _stack_moe_loads
# isort: on


class TestAscendConfig(unittest.TestCase):
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def setUp(self, mock_fix_incompatible_config):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {"dynamic_eplb": True, "num_redundant_experts": 2},
        }
        from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
        moe_parallel_config = FusedMoEParallelConfig(
            2, 0, 1, 2, 1, 1, 1, 1, 1, True, "hccl",
            enable_eplb=True)
        moe_config = FusedMoEConfig(
            num_experts=8,
            experts_per_token=8,
            hidden_dim=8192,
            intermediate_size_per_partition=5,
            num_local_experts=8,
            num_logical_experts=8,
            activation="silu",
            device="npu",
            routing_method=RoutingMethodType.Simulated,
            moe_parallel_config=moe_parallel_config,
            in_dtype=torch.float16,
        )
        moe_config.supports_eplb = True
        self.vllm_config = vllm_config
        self.moe_config = moe_config
        self.mock_npu = patch("torch.Tensor.npu", new=lambda self: self).start()
        os.environ["DYNAMIC_EPLB"] = "true"

    def test_init_eplb_config_with_eplb(self):
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([4, -1, -1, -1, 0, 1, 2, 3])
        gt_log2phy = torch.tensor([9, 1, 2, 3, 5, 6, 7, 8])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_init_eplb_config_with_eplb_withmap(self):
        _TEST_DIR = os.path.dirname(__file__)
        self.vllm_config.additional_config["eplb_config"]["expert_map_path"] = _TEST_DIR + "/expert_map.json"
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, 1, 4, -1, 2, -1, 0, 3])
        gt_log2phy = torch.tensor([2, 6, 9, 3, 7, 4, 5, 8])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertTrue(torch.equal(log2phy, gt_log2phy))
        self.assertEqual(redundant_experts, 2)

    def test_init_eplb_config_with_static_metro_routing(self):
        self.vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {
                "metro_routing": True,
                "num_redundant_experts": 2,
            },
        }
        eplb_config = init_ascend_config(self.vllm_config).eplb_config

        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)

        gt_expert_map = torch.tensor([4, -1, -1, -1, 0, 1, 2, 3])
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertEqual(redundant_experts, 2)
        self.assertEqual(log2phy.dim(), 2)
        self.assertTrue(torch.equal(log2phy[0], torch.tensor([0, 9], dtype=torch.int32)))
        self.assertTrue(torch.equal(log2phy[4], torch.tensor([4, 5], dtype=torch.int32)))

    def test_init_eplb_config_without_eplb(self):
        self.vllm_config.additional_config = {"refresh": True}
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
        self.assertIsNone(log2phy)
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertEqual(redundant_experts, 0)

    def test_stack_moe_loads_pads_variable_pool_slots(self):
        loads = [
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
        ]
        stacked = _stack_moe_loads(loads)
        self.assertTrue(torch.equal(stacked, torch.tensor([[1, 2, 0], [3, 4, 5]])))

    def test_get_configured_craft_pool_size_per_layer(self):
        self.vllm_config.additional_config["eplb_config"] = {
            "craft_pool_layer_sizes": [0, 2, 1],
        }
        eplb_config = init_ascend_config(self.vllm_config).eplb_config

        self.assertEqual(get_configured_craft_pool_size(eplb_config), 2)
        self.assertEqual(get_configured_craft_pool_size(eplb_config, 0), 0)
        self.assertEqual(get_configured_craft_pool_size(eplb_config, 1), 2)
        self.assertEqual(get_configured_craft_pool_size(eplb_config, 4), 0)

    def test_init_eplb_config_with_global_craft_pool(self):
        self.vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {
                "dynamic_eplb": True,
                "eplb_policy_type": 4,
                "craft_global_pool_size": 4,
            },
        }
        eplb_config = init_ascend_config(self.vllm_config).eplb_config

        global_map, expert_map, log2phy, redundant_experts = init_eplb_config(
            eplb_config, 0, self.moe_config
        )

        self.assertEqual(get_configured_craft_global_pool_size(eplb_config), 4)
        self.assertEqual(get_craft_global_pool_size_per_rank(eplb_config, 2), 2)
        self.assertEqual(redundant_experts, 0)
        self.assertEqual(int((expert_map >= 0).sum().item()), 4)
        self.assertEqual(global_map.shape, torch.Size([2, 8]))
        self.assertEqual(log2phy.shape, torch.Size([8, 3]))
        self.assertEqual(log2phy[4, 0].item(), 6)

    def test_global_craft_pool_rank_sharded_route_uses_physical_stride(self):
        self.vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {
                "dynamic_eplb": True,
                "eplb_policy_type": 4,
                "craft_global_pool_size": 4,
                "craft_rank_sharded_routing": True,
            },
        }
        eplb_config = init_ascend_config(self.vllm_config).eplb_config

        _, _, log2phy, _ = init_eplb_config(
            eplb_config,
            0,
            self.moe_config,
        )

        self.assertEqual(log2phy.shape, torch.Size([8]))
        self.assertEqual(log2phy[4].item(), 6)

    def test_global_craft_pool_requires_ep_divisibility(self):
        eplb_config = EplbConfig({
            "dynamic_eplb": True,
            "eplb_policy_type": 4,
            "craft_global_pool_size": 3,
        })

        with self.assertRaises(ValueError):
            get_craft_global_pool_size_per_rank(eplb_config, 2)

    def test_init_eplb_config_with_per_layer_craft_pool_size(self):
        self.vllm_config.additional_config = {
            "refresh": True,
            "eplb_config": {"craft_pool_layer_sizes": [0, 1]},
        }
        eplb_config = init_ascend_config(self.vllm_config).eplb_config

        _, expert_map0, log2phy0, redundant_experts0 = init_eplb_config(eplb_config, 0, self.moe_config)
        _, expert_map1, log2phy1, redundant_experts1 = init_eplb_config(eplb_config, 1, self.moe_config)

        self.assertEqual(redundant_experts0, 0)
        self.assertEqual(redundant_experts1, 2)
        self.assertEqual(int((expert_map0 >= 0).sum().item()), 4)
        self.assertEqual(int((expert_map1 >= 0).sum().item()), 5)
        self.assertIsNotNone(log2phy0)
        self.assertIsNotNone(log2phy1)
        self.assertEqual(log2phy0.dim(), 1)
        self.assertEqual(log2phy1.dim(), 2)

    def test_init_eplb_config_uses_logical_experts_for_craft_pool(self):
        eplb_config = EplbConfig({"craft_pool_size": 1, "num_redundant_experts": 2})
        self.moe_config.num_experts = 10
        self.moe_config.num_logical_experts = 8

        global_map, expert_map, log2phy, redundant_experts = init_eplb_config(
            eplb_config,
            0,
            self.moe_config,
        )

        self.assertEqual(redundant_experts, 2)
        self.assertEqual(global_map.shape, torch.Size([2, 8]))
        self.assertEqual(int((expert_map >= 0).sum().item()), 5)
        self.assertEqual(log2phy.shape, torch.Size([8, 3]))
        self.assertTrue(torch.all(log2phy[:, -1] <= -2))

    def test_pool_log2phy_shape_is_independent_of_replica_distribution(self):
        first_placement = torch.tensor([
            [0, 1, -1, -1],
            [-1, -1, 0, 1],
        ])
        second_placement = torch.tensor([
            [0, 1, 2, -1],
            [0, -1, -1, 1],
        ])

        first_log2phy = generate_pool_log2phy_map(first_placement)
        second_log2phy = generate_pool_log2phy_map(second_placement)

        self.assertEqual(first_log2phy.shape, torch.Size([4, 2]))
        self.assertEqual(second_log2phy.shape, torch.Size([4, 2]))

        craft_route = generate_craft_route_map(second_placement)
        self.assertEqual(craft_route.shape, torch.Size([4, 3]))
        self.assertTrue(torch.equal(craft_route[:, :-1], second_log2phy))
        self.assertTrue(torch.equal(craft_route[:, -1], torch.tensor([-3, -2, -2, -2])))

        rank0_route = generate_craft_rank_route_map(
            second_placement,
            0,
            local_slots=3,
        )
        rank1_route = generate_craft_rank_route_map(
            second_placement,
            1,
            local_slots=3,
        )
        self.assertTrue(torch.equal(rank0_route, torch.tensor([0, 1, 2, 4])))
        self.assertTrue(torch.equal(rank1_route, torch.tensor([3, 1, 2, 4])))

        three_rank_placement = torch.tensor([
            [0, 1],
            [0, -1],
            [0, 1],
        ])
        self.assertTrue(
            torch.equal(
                generate_craft_rank_route_map(
                    three_rank_placement,
                    0,
                    local_slots=2,
                ),
                torch.tensor([0, 1]),
            )
        )
        self.assertTrue(
            torch.equal(
                generate_craft_rank_route_map(
                    three_rank_placement,
                    1,
                    local_slots=2,
                ),
                torch.tensor([2, 1]),
            )
        )
        self.assertTrue(
            torch.equal(
                generate_craft_rank_route_map(
                    three_rank_placement,
                    2,
                    local_slots=2,
                ),
                torch.tensor([4, 5]),
            )
        )
