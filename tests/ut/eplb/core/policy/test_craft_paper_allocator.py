import numpy as np

from vllm_ascend.eplb.core.policy.craft_paper_allocator import (
    allocate_replica_budget,
    interleaved_replica_capacities,
    place_layer_experts,
    plan_craft_replication,
    replica_count_options,
)


def test_replica_budget_follows_layer_skew():
    hotness = np.ones((4, 16), dtype=np.float64)
    hotness[0, 0] = 1_000
    hotness[1, :4] = 250
    hotness[2, :2] = 300

    layer_replicas, extra_capacities, placements = plan_craft_replication(
        hotness,
        total_replicas=8,
        num_ranks=4,
    )

    assert layer_replicas.tolist() == [4, 2, 2, 0]
    assert extra_capacities.sum(axis=0).tolist() == [2, 2, 2, 2]
    for layer_id, ranks in enumerate(placements):
        assert sum(len(rank) for rank in ranks) == 16 + layer_replicas[layer_id]
        assert all(
            len(rank) == 4 + extra_capacities[layer_id, rank_id]
            for rank_id, rank in enumerate(ranks)
        )


def test_fine_grained_options_include_every_layer_budget():
    assert replica_count_options(8, fine_grained=True) == list(range(9))
    assert replica_count_options(8) == [0, 1, 2, 4, 8]


def test_fine_grained_plan_uses_non_power_of_two_layer_budget():
    hotness = np.random.default_rng(2).lognormal(0, 2, (4, 16))

    layer_replicas, extra_capacities, placements = plan_craft_replication(
        hotness,
        total_replicas=8,
        num_ranks=4,
        fine_grained=True,
    )

    assert layer_replicas.tolist() == [2, 3, 1, 2]
    assert extra_capacities.sum(axis=0).tolist() == [2, 2, 2, 2]
    for layer_id, ranks in enumerate(placements):
        assert sum(len(rank) for rank in ranks) == 16 + layer_replicas[layer_id]


def test_active_layer_replica_floor_covers_every_active_layer():
    hotness = np.ones((4, 16), dtype=np.float64)
    hotness[3] = 0
    hotness[0, 0] = 1_000

    layer_replicas, extra_capacities, _ = plan_craft_replication(
        hotness,
        total_replicas=8,
        num_ranks=4,
        fine_grained=True,
        min_replicas_per_active_layer=1,
    )

    assert np.all(layer_replicas[:3] >= 1)
    assert int(layer_replicas.sum()) == 8
    assert extra_capacities.sum(axis=0).tolist() == [2, 2, 2, 2]


def test_active_layer_replica_floor_rejects_insufficient_budget():
    hotness = np.ones((5, 16), dtype=np.float64)

    try:
        plan_craft_replication(
            hotness,
            total_replicas=4,
            num_ranks=4,
            fine_grained=True,
            min_replicas_per_active_layer=1,
        )
    except ValueError as error:
        assert "requires 5 replicas" in str(error)
    else:
        raise AssertionError("insufficient active-layer replica budget must fail")


def test_interleaved_capacities_balance_every_rank():
    capacities = interleaved_replica_capacities(
        np.asarray([4, 2, 2, 0]),
        num_ranks=4,
    )

    assert capacities.tolist() == [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    assert capacities.sum(axis=0).tolist() == [2, 2, 2, 2]


def test_deepseek_half_budget_uses_every_slot_once():
    num_layers = 43
    num_ranks = 8
    num_experts = 256
    total_replicas = 168
    hotness = np.ones((num_layers, num_experts), dtype=np.float64)
    hotness[18, 83] = 10_000
    hotness[18, 93] = 8_000
    hotness[42, 255] = 9_000

    layer_replicas, extra_capacities, placements = plan_craft_replication(
        hotness,
        total_replicas=total_replicas,
        num_ranks=num_ranks,
    )

    assert int(layer_replicas.sum()) == total_replicas
    assert extra_capacities.sum(axis=0).tolist() == [21] * num_ranks
    for layer_id, ranks in enumerate(placements):
        flattened = []
        for rank_id, rank_experts in enumerate(ranks):
            assert len(rank_experts) == 32 + extra_capacities[layer_id, rank_id]
            assert len(rank_experts) == len(set(rank_experts))
            flattened.extend(rank_experts)
        counts = np.bincount(flattened, minlength=num_experts)
        assert np.all(counts >= 1)
        assert int(counts.sum()) == num_experts + layer_replicas[layer_id]


def test_fixed_home_plan_only_uses_extra_capacity_for_replicas():
    num_layers = 4
    num_ranks = 4
    num_experts = 16
    hotness = np.ones((num_layers, num_experts), dtype=np.float64)
    hotness[0, 0] = 1_000
    hotness[1, 7] = 500
    home = [
        [
            list(range(rank_id * 4, (rank_id + 1) * 4))
            for rank_id in range(num_ranks)
        ]
        for _ in range(num_layers)
    ]

    layer_replicas, extra_capacities, placements = plan_craft_replication(
        hotness,
        total_replicas=8,
        num_ranks=num_ranks,
        home_placements=home,
    )

    assert int(layer_replicas.sum()) == 8
    assert extra_capacities.sum(axis=0).tolist() == [2, 2, 2, 2]
    for layer_id, ranks in enumerate(placements):
        for rank_id, rank_experts in enumerate(ranks):
            assert rank_experts[:4] == home[layer_id][rank_id]
            assert len(rank_experts) == 4 + extra_capacities[layer_id, rank_id]
            assert len(rank_experts) == len(set(rank_experts))


def test_top_m_adds_rank_feasibility_candidates():
    num_ranks = 4
    hotness = np.arange(16, 0, -1, dtype=np.float64)[None, :]
    home = [[
        list(range(rank_id * 4, (rank_id + 1) * 4))
        for rank_id in range(num_ranks)
    ]]

    layer_replicas, extra_capacities, placements = plan_craft_replication(
        hotness,
        total_replicas=4,
        num_ranks=num_ranks,
        home_placements=home,
        candidate_top_m=1,
    )

    assert layer_replicas.tolist() == [4]
    assert extra_capacities.tolist() == [[1, 1, 1, 1]]
    for rank_id, rank_experts in enumerate(placements[0]):
        assert rank_experts[:4] == home[0][rank_id]
        assert len(rank_experts) == 5
        assert len(rank_experts) == len(set(rank_experts))


def test_no_home_candidate_mask_controls_replication():
    assignments, _ = place_layer_experts(
        np.asarray([100.0, 10.0, 1.0, 1.0]),
        num_replicas=1,
        rank_capacities=np.asarray([3, 2]),
        candidate_mask=np.asarray([False, True, False, False]),
    )

    counts = np.bincount(
        [expert_id for rank in assignments for expert_id in rank],
        minlength=4,
    )
    assert counts.tolist() == [1, 2, 1, 1]


def test_no_home_placement_relocates_to_reach_feasible_solution():
    assignments, _ = place_layer_experts(
        np.asarray([1.001, 2.182, 0.772, 0.986, 0.286, 0.993]),
        num_replicas=2,
        rank_capacities=np.asarray([3, 3, 2]),
    )

    assert [len(rank) for rank in assignments] == [3, 3, 2]
    assert all(len(rank) == len(set(rank)) for rank in assignments)
    counts = np.bincount(
        [expert_id for rank in assignments for expert_id in rank],
        minlength=6,
    )
    assert counts.tolist() == [1, 3, 1, 1, 1, 1]


def test_budget_options_must_include_zero():
    try:
        allocate_replica_budget(
            np.asarray([[1.0]], dtype=np.float64),
            options=[1],
            total_replicas=0,
        )
    except ValueError as error:
        assert "include zero" in str(error)
    else:
        raise AssertionError("missing zero option must be rejected")


def test_fixed_home_capacity_interleaving_uses_rank_loads():
    base_loads = np.asarray(
        [
            [0.778, 2.637, 1.900, 29.470],
            [44.447, 9.914, 51.579, 6.977],
            [51.914, 30.188, 2.448, 1.482],
            [8.690, 8.392, 19.724, 2.010],
        ],
        dtype=np.float64,
    )
    hotness = np.zeros((4, 16), dtype=np.float64)
    home = []
    for layer_id in range(4):
        layer_home = []
        for rank_id in range(4):
            rank = list(range(rank_id * 4, (rank_id + 1) * 4))
            layer_home.append(rank)
            hotness[layer_id, rank[0]] = base_loads[layer_id, rank_id]
        home.append(layer_home)

    layer_replicas, extra_capacities, _ = plan_craft_replication(
        hotness,
        total_replicas=8,
        num_ranks=4,
        home_placements=home,
    )
    slot_only_capacities = interleaved_replica_capacities(
        layer_replicas,
        num_ranks=4,
    )

    def plan_score(capacities):
        score = 0.0
        for layer_id in range(4):
            _, rank_loads = place_layer_experts(
                hotness[layer_id],
                num_replicas=int(layer_replicas[layer_id]),
                rank_capacities=np.full(4, 4) + capacities[layer_id],
                home_assignments=home[layer_id],
            )
            score += rank_loads.mean() / rank_loads.max()
        return score

    assert extra_capacities.sum(axis=0).tolist() == [2, 2, 2, 2]
    assert extra_capacities.tolist() != slot_only_capacities.tolist()
    assert plan_score(extra_capacities) > plan_score(slot_only_capacities)


def test_load_aware_interleaving_rejects_realized_regression():
    rng = np.random.default_rng(7)
    hotness = rng.lognormal(0, 2, (4, 16))
    home = [
        [
            list(range(rank_id * 4, (rank_id + 1) * 4))
            for rank_id in range(4)
        ]
        for _ in range(4)
    ]

    layer_replicas, extra_capacities, _ = plan_craft_replication(
        hotness,
        total_replicas=8,
        num_ranks=4,
        home_placements=home,
    )

    assert layer_replicas.tolist() == [2, 0, 2, 4]
    assert np.array_equal(
        extra_capacities,
        interleaved_replica_capacities(layer_replicas, num_ranks=4),
    )
