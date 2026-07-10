import unittest
from queue import Empty
from unittest.mock import MagicMock, patch
import torch
from vllm_ascend.eplb.eplb_updator import EplbUpdator


class TestEplbUpdatorComputeAndSetMoeLoad(unittest.TestCase):
    def setUp(self):

        # ====================== 1. Mock environment ======================
        self.rank = 0
        self.world_size = 4
        self.device = torch.device("cpu")

        # mock dist
        p1 = patch("torch.distributed.get_rank", return_value=self.rank)
        p2 = patch("torch.distributed.get_world_size", return_value=self.world_size)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

        # ====================== 2. Mock comm group ======================
        self.mock_comm_group = MagicMock()
        self.mock_comm_group.ranks = list(range(self.world_size))
        self.mock_comm_group.rank_in_group = self.rank
        self.mock_comm_group.cpu_group = "cpu_group"

        def mock_all_gather(tensor, dim):
            gathered = torch.cat([tensor for _ in range(self.world_size)], dim=dim)
            return gathered

        self.mock_comm_group.all_gather = mock_all_gather

        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group",
                   return_value=self.mock_comm_group)
        self.addCleanup(p3.stop)
        p3.start()

        # ====================== 3. Mock EplbUpdator ======================
        self.eplb_config = MagicMock()
        self.eplb_config.eplb_policy_type = 4
        self.eplb_config.expert_heat_collection_interval = 20
        self.eplb_config.algorithm_execution_interval = 5
        self.eplb_config.expert_map_path = None
        self.eplb_config.expert_map_record_path = None
        self.loader = MagicMock()
        self.eplb_process = MagicMock()
        self.process = MagicMock()
        self.eplb_process.shared_dict = {}

        self.updator = EplbUpdator(
            eplb_config=self.eplb_config,
            loader=self.loader,
            eplb_process=self.eplb_process,
            process=self.process
        )

        # ====================== 4. Mock adaptor ======================
        self.adaptor = MagicMock()
        self.adaptor.num_moe_layers = 4
        self.adaptor.num_dense_layers = 2
        self.mock_local_load = torch.randn(58, 100, 8, device=self.device)
        self.adaptor.get_rank_expert_workload.return_value = self.mock_local_load

        self.updator.set_adaptor(self.adaptor)

    def test_compute_and_set_moe_load_normal(self):
        self.updator.multi_stage = False

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (58, self.world_size, 100, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")
        self.assertEqual(moe_load.shape[1], self.world_size)

    def test_compute_and_set_moe_load_multi_stage(self):
        self.updator.multi_stage = True

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (100, 58, self.world_size, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")

    def test_select_rank_update_info_from_full_plan(self):
        self.updator.rank_in_group = 2
        full_plan = [
            {
                "send_all": [["s0"], ["s1"], ["s2"], ["s3"]],
                "recv_all": [["r0"], ["r1"], ["r2"], ["r3"]],
                "maps_all": [["m0"], ["m1"], ["m2"], ["m3"]],
                "log2phy_all": [["l0"], ["l1"], ["l2"], ["l3"]],
                "layer_id": 7,
            }
        ]

        selected = self.updator._select_rank_update_info(full_plan)

        self.assertEqual(selected, [(["s2"], ["r2"], ["m2"], ["l2"], 7)])

    def test_broadcast_update_info_uses_rank0_plan(self):
        self.updator.rank_id = 1
        self.updator.plan_src_rank = 0

        def fake_broadcast(object_list, src, group):
            self.assertEqual(src, 0)
            self.assertEqual(group, "cpu_group")
            object_list[0] = ["rank0-plan"]

        with patch("torch.distributed.broadcast_object_list", side_effect=fake_broadcast):
            update_info = self.updator._broadcast_update_info(["local-plan"])

        self.assertEqual(update_info, ["rank0-plan"])

    def test_full_rank_plan_non_source_does_not_wait_on_local_planner(self):
        self.updator.rank_id = 1
        self.updator.plan_src_rank = 0
        self.updator.cur_iterations = (
            self.updator.expert_heat_collection_interval
            + self.updator.algorithm_execution_interval
            - 1
        )
        full_plan = [
            {
                "send_all": [[], [], [], []],
                "recv_all": [[], [], [], []],
                "maps_all": [[0], [0], [0], [0]],
                "log2phy_all": [[0], [0], [0], [0]],
                "layer_id": 0,
            }
        ]

        with patch.object(self.updator, "_broadcast_update_info", return_value=full_plan):
            self.updator.forward_before()

        self.eplb_process.block_update_q.get.assert_not_called()
        self.assertEqual(len(self.updator.update_info_all), 1)

    def test_full_rank_plan_timeout_broadcasts_skip(self):
        self.updator.rank_id = 0
        self.updator.plan_src_rank = 0
        self.updator.cur_iterations = (
            self.updator.expert_heat_collection_interval
            + self.updator.algorithm_execution_interval
            - 1
        )
        self.eplb_process.block_update_q.get.side_effect = Empty

        with patch.object(self.updator, "_broadcast_update_info", return_value=None) as mock_broadcast:
            self.updator.forward_before()

        self.eplb_process.block_update_q.get.assert_called_once_with(timeout=self.updator.plan_timeout_s)
        mock_broadcast.assert_called_once_with(None)
        self.assertEqual(self.updator.update_info_all, [])

    def test_full_rank_plan_only_wakes_source_worker(self):
        self.updator.rank_id = 1
        self.updator.plan_src_rank = 0
        self.updator.wakeup_eplb_worker()
        self.eplb_process.planner_q.put.assert_not_called()

        self.updator.rank_id = 0
        self.updator.wakeup_eplb_worker()
        self.eplb_process.planner_q.put.assert_called_once_with(1)

    def test_forward_before_selects_update_info_by_index_without_pop(self):
        self.updator.cur_iterations = (
            self.updator.expert_heat_collection_interval
            + self.updator.algorithm_execution_interval
            + 1
        )
        self.updator.update_info_all = [
            ([], [], [0, 1], [[0, 1]], 0),
            ([(2, 3)], [(0, 3)], [1, 0], [[1, 0]], 1),
        ]

        self.updator.forward_before()

        self.assertEqual(len(self.updator.update_info_all), 2)
        self.assertEqual(self.updator.update_info_index, 1)
        self.loader.set_log2phy_map.assert_called_once()
        self.loader.generate_expert_d2d_transfer_task.assert_called_once()
        args = self.loader.generate_expert_d2d_transfer_task.call_args.args
        self.assertEqual(args[0], [(2, 3)])
        self.assertEqual(args[1], [(0, 3)])
        self.assertEqual(args[3], 3)
        self.loader.asyn_expert_weight_transfer.assert_called_once()

    def test_policy2_consumes_rank_local_plan_with_original_pop_order(self):
        self.updator.full_rank_plan = False
        self.updator.cur_iterations = (
            self.updator.expert_heat_collection_interval
            + self.updator.algorithm_execution_interval
        )
        self.updator.update_info_all = [
            ([(2, 3)], [(0, 3)], [1, 0], [[1, 0]], 0),
            ([], [], [0, 1], [[0, 1]], 1),
        ]

        self.updator.forward_before(skip_update=True)

        self.assertFalse(self.updator.skip_current_step)
        self.assertEqual(len(self.updator.update_info_all), 1)
        self.loader.generate_expert_d2d_transfer_task.assert_called_once()
        self.loader.asyn_expert_weight_transfer.assert_called_once()

    def test_skip_update_step_holds_iteration_without_transfer(self):
        self.updator.cur_iterations = (
            self.updator.expert_heat_collection_interval
            + self.updator.algorithm_execution_interval
        )
        self.updator.update_info_all = [
            ([(2, 3)], [(0, 3)], [1, 0], [[1, 0]], 0),
        ]

        self.updator.forward_before(skip_update=True)
        self.updator.forward_end()

        self.loader.generate_expert_d2d_transfer_task.assert_not_called()
        self.loader.update_expert_map_and_weight.assert_not_called()
        self.assertEqual(
            self.updator.cur_iterations,
            self.updator.expert_heat_collection_interval
            + self.updator.algorithm_execution_interval,
        )
        self.assertFalse(self.updator.skip_current_step)


if __name__ == '__main__':
    unittest.main()
