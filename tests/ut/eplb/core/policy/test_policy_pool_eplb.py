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


def test_global_pool_assigns_each_physical_slot_to_one_layer():
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
    assert torch.all(torch.sum(updated[:, :, 2:] >= 0, dim=0) == 1)
    assert updated[1, 1, 2].item() == 0
