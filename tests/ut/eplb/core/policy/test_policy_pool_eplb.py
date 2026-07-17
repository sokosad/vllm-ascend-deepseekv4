import unittest

import numpy as np
import torch

from vllm_ascend.eplb.core.policy.policy_abstract import DynamicConfig
from vllm_ascend.eplb.core.policy.policy_pool_eplb import PoolBalanceEplb


def test_pool_policy_skips_layers_without_pool_slots():
    policy = PoolBalanceEplb(DynamicConfig())
    current = torch.tensor([
        [[0, -1], [1, -1]],
        [[0, 1], [1, 0]],
    ])
    workload = torch.tensor([
        [[4, 0], [2, 0]],
        [[10, 1], [1, 10]],
    ])

    _, _, updated = policy.rebalance_experts(current, workload)
    updated = torch.tensor(updated)

    assert torch.equal(updated[0], current[0])
    assert torch.equal(updated[1, :, 1] >= 0, torch.tensor([True, True]))


def test_pool_policy_limits_candidate_experts_to_top_m():
    policy = PoolBalanceEplb(DynamicConfig())
    policy.candidate_top_m = 2

    candidates = policy._candidate_experts(np.array([10.0, 1.0, 9.0, 0.5]), pool_size=1, num_ranks=2)

    assert set(candidates.tolist()) == {0, 2}


def test_pool_policy_reads_craft_pool_config():
    config = DynamicConfig()
    config.craft_pool_top_m = 3
    config.craft_pool_top_m_factor = 2
    config.craft_pool_min_hotness_delta = 0.25
    config.craft_pool_min_improvement = 0.1

    policy = PoolBalanceEplb(config)

    assert policy.candidate_top_m == 3
    assert policy.candidate_factor == 2
    assert policy.min_hotness_delta == 0.25
    assert policy.min_improvement == 0.1


def test_pool_policy_accumulates_small_hotness_changes_before_recompute():
    policy = PoolBalanceEplb(DynamicConfig())
    policy.min_hotness_delta = 0.5

    assert not policy._should_skip_layer(0, np.array([100.0, 0.0]))
    assert policy._should_skip_layer(0, np.array([80.0, 20.0]))
    assert not policy._should_skip_layer(0, np.array([60.0, 40.0]))


def test_pool_policy_ignores_sampling_volume_changes():
    policy = PoolBalanceEplb(DynamicConfig())
    policy.min_hotness_delta = 0.05

    assert not policy._should_skip_layer(0, np.array([90.0, 10.0]))
    assert policy._should_skip_layer(0, np.array([9000.0, 1000.0]))
    assert not policy._should_skip_layer(0, np.array([70.0, 30.0]))


def test_pool_policy_skips_migration_when_improvement_is_too_small():
    policy = PoolBalanceEplb(DynamicConfig())
    policy.min_hotness_delta = 0.0
    policy.min_improvement = 0.99
    current = torch.tensor([[
        [0, 1, 2],
        [2, 3, 0],
    ]])
    workload = torch.tensor([[
        [1, 100, 1],
        [1, 1, 1],
    ]])

    changed, _, updated = policy.rebalance_experts(current, workload)

    assert not changed
    assert torch.equal(torch.tensor(updated), current)


def test_global_pool_assigns_each_active_slot_to_at_most_one_layer():
    config = DynamicConfig()
    config.craft_global_pool_size = 2
    config.craft_pool_min_hotness_delta = 0.0
    policy = PoolBalanceEplb(config)
    current = torch.tensor([
        [[0, 1, -1], [2, 3, -1]],
        [[0, 1, -1], [2, 3, -1]],
    ])
    workload = torch.tensor([
        [[10, 1, 0], [9, 1, 0]],
        [[100, 1, 0], [2, 1, 0]],
    ])

    changed, _, updated = policy.rebalance_experts(current, workload)
    updated = torch.tensor(updated)

    assert changed
    owners = torch.sum(updated[:, :, 2:] >= 0, dim=0)
    assert torch.all(owners <= 1)
    assert torch.sum(owners).item() == 1
    assert updated[1, 1, 2].item() == 0


def test_global_pool_hotness_gate_ignores_cold_layer_noise():
    config = DynamicConfig()
    config.craft_pool_top_m = 2
    config.craft_pool_min_hotness_delta = 0.05
    policy = PoolBalanceEplb(config)

    first = np.array([[1000.0, 500.0], [1.0, 0.0]])
    cold_layer_shift = np.array([[1000.0, 500.0], [0.0, 1.0]])

    assert policy._global_hotness_changed(first, total_slots=1)
    assert not policy._global_hotness_changed(cold_layer_shift, total_slots=1)


def test_global_pool_hotness_gate_accumulates_top_candidate_changes():
    config = DynamicConfig()
    config.craft_pool_top_m = 2
    config.craft_pool_min_hotness_delta = 0.5
    policy = PoolBalanceEplb(config)

    assert policy._global_hotness_changed(np.array([[100.0, 0.0]]), total_slots=1)
    assert not policy._global_hotness_changed(np.array([[80.0, 20.0]]), total_slots=1)
    assert policy._global_hotness_changed(np.array([[60.0, 40.0]]), total_slots=1)


def test_global_pool_imbalance_weights_layers_by_traffic():
    table = np.array(
        [
            [[0], [1]],
            [[0], [1]],
        ]
    )
    hotness = np.array(
        [
            [100.0, 0.0],
            [1.0, 1.0],
        ]
    )

    imbalance = PoolBalanceEplb._global_imbalance(table, hotness)

    assert np.isclose(imbalance, (2.0 * 100.0 + 1.0 * 2.0) / 102.0)


class TestGlobalPoolMarginalPlacement(unittest.TestCase):
    def test_prefers_marginal_balance_gain_over_raw_hotness(self):
        policy = PoolBalanceEplb(DynamicConfig())
        home = [[[0, 1], [2, 3]]]
        hotness = np.array([[80.0, 70.0, 100.0, 0.0]])

        assignments = policy._desired_global_assignments(home, hotness, pool_size=1)

        self.assertEqual(assignments[1], [(0, 1)])
        self.assertEqual(assignments[0], [])


def test_global_pool_realistic_shape_fills_slots_and_stays_stable():
    num_layers = 43
    num_ranks = 8
    num_experts = 256
    pool_size = 2
    main_size = num_experts // num_ranks

    config = DynamicConfig()
    config.craft_global_pool_size = num_ranks * pool_size
    config.craft_pool_min_hotness_delta = 0.05
    policy = PoolBalanceEplb(config)

    current = torch.full(
        (num_layers, num_ranks, main_size + pool_size),
        -1,
        dtype=torch.long,
    )
    home = torch.arange(num_experts, dtype=torch.long).view(num_ranks, main_size)
    current[:, :, :main_size] = home
    workload = torch.ones_like(current)
    workload[18, 2, 19] = 10_000
    workload[42, 7, 31] = 8_000

    changed, _, updated_list = policy.rebalance_experts(current, workload)
    updated = torch.tensor(updated_list)

    assert changed
    assert torch.equal(updated[:, :, :main_size], current[:, :, :main_size])
    assert torch.all(torch.sum(updated[:, :, main_size:] >= 0, dim=0) <= 1)
    for layer_id in range(num_layers):
        for rank_id in range(num_ranks):
            valid = updated[layer_id, rank_id]
            valid = valid[valid >= 0]
            assert valid.numel() == torch.unique(valid).numel()

    changed_again, _, stable_list = policy.rebalance_experts(updated, workload)

    assert not changed_again
    assert torch.equal(torch.tensor(stable_list), updated)


def test_global_pool_first_cycle_rejects_non_improving_layout():
    config = DynamicConfig()
    config.craft_global_pool_size = 2
    config.craft_pool_min_hotness_delta = 0.0
    config.craft_pool_min_improvement = 0.01
    policy = PoolBalanceEplb(config)
    current = torch.tensor([[
        [0, 1, -1],
        [2, 3, -1],
    ]])
    workload = torch.ones_like(current)

    changed, _, updated = policy.rebalance_experts(current, workload)

    assert not changed
    assert torch.equal(torch.tensor(updated), current)


def test_global_pool_returns_partial_assignment_when_capacity_cannot_help():
    policy = PoolBalanceEplb(DynamicConfig())
    home = [[[0], [1]]]
    hotness = np.array([[100.0, 1.0]])

    assignments = policy._desired_global_assignments(
        home,
        hotness,
        pool_size=2,
    )

    assert sum(len(rank_items) for rank_items in assignments) < 4


def test_global_pool_zero_hotness_has_no_candidates():
    policy = PoolBalanceEplb(DynamicConfig())

    assert policy._global_candidates(np.zeros((43, 256)), 16) == []
