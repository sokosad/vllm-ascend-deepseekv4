import unittest

import numpy as np
import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker


def test_pack_update_info_returns_full_rank_plan():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 4
    worker.metro_routing = False
    worker.full_rank_plan = True

    send_info = {0: [(1, 2)]}
    recv_info = {1: [(0, 2)]}
    new_expert_map = torch.tensor(
        [
            [0, 1, -1],
            [0, -1, 1],
        ],
        dtype=torch.long,
    )

    packed = worker.pack_update_info([(send_info, recv_info, new_expert_map, 3)])

    assert len(packed) == 1
    record = packed[0]
    assert record["send_all"] == [[(1, 2)], []]
    assert record["recv_all"] == [[], [(0, 2)]]
    assert record["maps_all"] == [[0, 1, -1], [0, -1, 1]]
    assert record["layer_id"] == 3
    assert len(record["log2phy_all"]) == 2
    assert record["log2phy_all"][0] == record["log2phy_all"][1]


def test_policy4_pack_update_info_marks_unchanged_layer_as_noop():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 4
    worker.metro_routing = False
    worker.full_rank_plan = True

    new_expert_map = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    packed = worker.pack_update_info(
        [({}, {}, new_expert_map, 0)], changed_layers=[False]
    )

    assert packed == [{"noop": True, "layer_id": 0}]


def test_global_pool_restores_unchanged_layer_routes_during_changed_cycle():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 4
    worker.full_rank_plan = True
    worker.craft_global_pool_size = 2
    worker.num_local_experts = 3
    expert_map = torch.tensor([[0, 1, -1, -1], [-1, -1, 0, 1]])
    records = [({}, {}, expert_map, 0), ({}, {}, expert_map, 1)]

    packed = worker.pack_update_info(records, changed_layers=[True, False])

    assert all("noop" not in record for record in packed)
    assert "route_only" not in packed[0]
    assert packed[1]["route_only"] is True
    assert "send_all" not in packed[1]
    assert "maps_all" not in packed[1]
    assert packed[1]["layer_id"] == 1


def test_global_pool_migration_always_uses_home_rank_source():
    worker = EplbWorker.__new__(EplbWorker)
    worker.craft_global_pool_size = 3
    current = torch.tensor([[
        [-1, -1, -1, -1, 2, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, 1],
    ]])
    updated = torch.tensor([[
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 2, -1],
        [-1, -1, -1, -1, 0, 1],
    ]])

    send, recv, _, _ = next(
        worker.compose_expert_update_info_greedy(updated, current)
    )

    assert send == {2: [(1, 4)]}
    assert recv == {1: [(2, 4)]}


def test_policy2_pack_update_info_keeps_rank_local_plan():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 2
    worker.rank_id = 1
    worker.metro_routing = False
    worker.full_rank_plan = False

    send_info = {0: [(1, 2)]}
    recv_info = {1: [(0, 2)]}
    new_expert_map = torch.tensor(
        [
            [0, 1, -1],
            [0, -1, 1],
        ],
        dtype=torch.long,
    )

    packed = worker.pack_update_info([(send_info, recv_info, new_expert_map, 3)])

    assert packed == [([], [(0, 2)], [0, -1, 1], [2, 1, 3], 3)]


def test_policy2_placement_validation_uses_legacy_path_without_pool_symbols():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 2
    old_placement = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    new_placement = old_placement.clone()

    worker.check_expert_placement(old_placement, new_placement)

    assert torch.equal(new_placement, old_placement)


def test_policy4_placement_validation_accepts_padded_pool_slots():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 4
    old_placement = torch.tensor([[[0, 1, 4], [2, 3, -1]]], dtype=torch.long)
    new_placement = old_placement.clone()

    worker.check_expert_placement(old_placement, new_placement)

    assert torch.equal(new_placement, old_placement)


def test_global_pool_validation_rejects_unowned_slot():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 4
    worker.craft_global_pool_size = 2
    old_placement = torch.tensor([
        [[0, 1, 2], [2, 3, -1]],
        [[0, 1, -1], [2, 3, 0]],
    ])
    new_placement = old_placement.clone()
    new_placement[1, 1, 2] = -1

    worker.check_expert_placement(old_placement, new_placement)

    assert torch.equal(new_placement, old_placement)


def test_compute_imbalance_handles_padded_layer_table():
    deployment = torch.tensor([[[0, 1, 4], [2, 3, -1]]], dtype=torch.long)
    hotness = np.asarray([[10.0, 20.0, 30.0, 40.0, 50.0]])

    mean_imbalance, max_imbalance = EplbWorker._compute_imbalance(deployment, hotness)

    assert np.isfinite(mean_imbalance)
    assert np.isfinite(max_imbalance)
    assert mean_imbalance == max_imbalance


def test_craft_pool_utilization_reports_active_pool_slots():
    deployment = torch.tensor(
        [
            [[0, 1, 4], [2, 3, 1]],
            [[0, 1, 2], [2, 3, 0]],
        ],
        dtype=torch.long,
    )
    load_info = torch.tensor(
        [
            [[10, 20, 5], [30, 40, 0]],
            [[10, 20, 0], [30, 40, 7]],
        ],
        dtype=torch.int64,
    )

    stats = EplbWorker._craft_pool_utilization(deployment, load_info)

    assert stats is not None
    assert stats["total_slots"] == 4
    assert stats["active_slots"] == 2
    assert stats["active_ratio"] == 0.5
    assert stats["pool_tokens"] == 12.0
    assert stats["pool_token_share"] == 12.0 / 212.0
    assert stats["zero_hit_layers"] == 0
    assert stats["top_layers"] == [(1, 7, 1), (0, 5, 1)]


def test_craft_migration_cost_gate_rejects_slow_payback():
    worker = EplbWorker.__new__(EplbWorker)
    worker.shared_dict = {
        "craft_expert_cost_metadata": [
            {"transfer_bytes": 100, "compute_bytes": 100}
        ]
    }
    worker.expert_heat_collection_interval = 10
    worker.craft_migration_cost_ratio = 1.0
    worker.craft_max_payback_steps = 1
    old_placement = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    new_placement = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)
    hotness = np.asarray([[10.0, 0.0, 0.0, 0.0]])

    gated = worker._apply_craft_migration_cost_gate(
        old_placement, new_placement, hotness
    )

    assert torch.equal(gated, old_placement)


def test_craft_migration_cost_gate_accepts_fast_payback():
    worker = EplbWorker.__new__(EplbWorker)
    worker.shared_dict = {
        "craft_expert_cost_metadata": [
            {"transfer_bytes": 100, "compute_bytes": 100}
        ]
    }
    worker.expert_heat_collection_interval = 10
    worker.craft_migration_cost_ratio = 1.0
    worker.craft_max_payback_steps = 1
    old_placement = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    new_placement = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)
    hotness = np.asarray([[100.0, 0.0, 0.0, 0.0]])

    gated = worker._apply_craft_migration_cost_gate(
        old_placement, new_placement, hotness
    )

    assert torch.equal(gated, new_placement)


def load_tests(loader_obj, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(test_pack_update_info_returns_full_rank_plan))
    suite.addTest(unittest.FunctionTestCase(test_policy4_pack_update_info_marks_unchanged_layer_as_noop))
    suite.addTest(unittest.FunctionTestCase(test_global_pool_restores_unchanged_layer_routes_during_changed_cycle))
    suite.addTest(unittest.FunctionTestCase(test_global_pool_migration_always_uses_home_rank_source))
    suite.addTest(unittest.FunctionTestCase(test_policy2_pack_update_info_keeps_rank_local_plan))
    suite.addTest(unittest.FunctionTestCase(test_policy2_placement_validation_uses_legacy_path_without_pool_symbols))
    suite.addTest(unittest.FunctionTestCase(test_policy4_placement_validation_accepts_padded_pool_slots))
    suite.addTest(unittest.FunctionTestCase(test_global_pool_validation_rejects_unowned_slot))
    suite.addTest(unittest.FunctionTestCase(test_compute_imbalance_handles_padded_layer_table))
    suite.addTest(unittest.FunctionTestCase(test_craft_pool_utilization_reports_active_pool_slots))
    suite.addTest(unittest.FunctionTestCase(test_craft_migration_cost_gate_rejects_slow_payback))
    suite.addTest(unittest.FunctionTestCase(test_craft_migration_cost_gate_accepts_fast_payback))
    return suite
