import os
import unittest
from unittest.mock import patch

# isort: off
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
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

    def test_init_eplb_config_without_eplb(self):
        self.vllm_config.additional_config = {"refresh": True}
        eplb_config = init_ascend_config(self.vllm_config).eplb_config
        _, expert_map, log2phy, redundant_experts = init_eplb_config(eplb_config, 0, self.moe_config)
        gt_expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
        self.assertIsNone(log2phy)
        self.assertTrue(torch.equal(expert_map, gt_expert_map))
        self.assertEqual(redundant_experts, 0)


class TestLog2PhyProximity(unittest.TestCase):
    """Node-aware (same-node-preferred) replica selection in generate_log2phy_map."""

    @staticmethod
    def _two_node_map():
        # 4 ranks, num_die_per_host=2 -> 2 nodes (ranks 0,1 | 2,3); 2 experts;
        # each rank holds 1 physical expert (valid_count=1).
        # expert0 replicated on rank0(node0) & rank2(node1);
        # expert1 replicated on rank1(node0) & rank3(node1).
        return [
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
        ]

    def test_prefers_same_node_replica(self):
        from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
        num_die, valid_count = 2, 1
        gmap = self._two_node_map()
        for ep in range(4):
            res = generate_log2phy_map(gmap, ep, num_die_per_host=num_die)
            for logical in range(2):
                phys = int(res[logical])
                self.assertEqual(
                    (phys // valid_count) // num_die, ep // num_die,
                    f"ep{ep} expert{logical} selected cross-node phys {phys}")

    def test_fallback_when_no_same_node_replica(self):
        from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
        # expert0 lives only on rank0 (node0); a node1 rank must fall back, not crash.
        gmap = [
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
            torch.tensor([-1, 0], dtype=torch.int32),
        ]
        res = generate_log2phy_map(gmap, 3, num_die_per_host=2)
        self.assertEqual(int(res[0]), 0)

    def test_disabled_matches_round_robin(self):
        from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
        gmap = self._two_node_map()
        # num_die_per_host=0 disables node-awareness -> original round-robin.
        for ep in range(4):
            res = generate_log2phy_map(gmap, ep, num_die_per_host=0)
            # expert0 replicas [phys0(rank0), phys2(rank2)] -> ep%2 selection
            self.assertEqual(int(res[0]), [0, 2][ep % 2])
