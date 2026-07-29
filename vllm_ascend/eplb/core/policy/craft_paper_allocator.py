# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
"""CRAFT-style replica budgeting for dynamic fixed-home expert pools."""

from __future__ import annotations

import numpy as np


def _replica_candidate_mask(
    expert_loads: np.ndarray,
    top_m: int,
    home_assignments: list[list[int]],
) -> np.ndarray:
    if top_m <= 0 or top_m >= expert_loads.size:
        return np.ones(expert_loads.size, dtype=bool)

    candidate_mask = np.zeros(expert_loads.size, dtype=bool)
    top_indices = np.argpartition(-expert_loads, top_m - 1)[:top_m]
    candidate_mask[top_indices] = True

    owner_by_expert = np.empty(expert_loads.size, dtype=np.int64)
    for rank_id, rank in enumerate(home_assignments):
        owner_by_expert[np.asarray(rank, dtype=np.int64)] = rank_id
    hottest_expert = int(np.argmax(expert_loads))
    candidate_mask[hottest_expert] = True
    hottest_owner = owner_by_expert[hottest_expert]
    outside_hottest_owner = np.flatnonzero(owner_by_expert != hottest_owner)
    if outside_hottest_owner.size:
        outside_loads = expert_loads[outside_hottest_owner]
        candidate_mask[
            int(outside_hottest_owner[int(np.argmax(outside_loads))])
        ] = True
    return candidate_mask


def _normalize_home_placements(
    home_placements: list[list[list[int]]],
    num_layers: int,
    num_experts: int,
    num_ranks: int,
) -> list[list[list[int]]]:
    if len(home_placements) != num_layers:
        raise ValueError("CRAFT home placement must have one entry per layer.")
    normalized = []
    expected_experts = list(range(num_experts))
    for layer in home_placements:
        if len(layer) != num_ranks:
            raise ValueError("CRAFT home placement must have one row per rank.")
        normalized_layer = [
            [int(expert_id) for expert_id in rank] for rank in layer
        ]
        flattened = [
            expert_id for rank in normalized_layer for expert_id in rank
        ]
        if (
            len(flattened) != num_experts
            or sorted(flattened) != expected_experts
        ):
            raise ValueError(
                "CRAFT fixed-home placement must contain every logical expert once."
            )
        normalized.append(normalized_layer)
    return normalized


def place_layer_experts(
    expert_loads: np.ndarray,
    num_replicas: int,
    rank_capacities: np.ndarray,
    home_assignments: list[list[int]],
    candidate_mask: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    num_ranks = rank_capacities.size
    if len(home_assignments) != num_ranks:
        raise ValueError("CRAFT home placement must have one row per rank.")
    assignments = [[int(expert_id) for expert_id in rank] for rank in home_assignments]
    flattened_home = [expert_id for rank in assignments for expert_id in rank]
    if (
        len(flattened_home) != expert_loads.size
        or sorted(flattened_home) != list(range(expert_loads.size))
    ):
        raise ValueError(
            "CRAFT fixed-home placement must contain every logical expert once."
        )
    if any(
        len(assignments[rank_id]) > rank_capacities[rank_id]
        for rank_id in range(num_ranks)
    ):
        raise ValueError("CRAFT fixed-home placement exceeds rank capacity.")

    free_capacity = rank_capacities - np.asarray(
        [len(rank) for rank in assignments],
        dtype=np.int64,
    )
    if int(free_capacity.sum()) != num_replicas:
        raise ValueError("CRAFT fixed-home capacity does not match replica count.")

    rank_loads = np.zeros(num_ranks, dtype=np.float64)
    owner_mask = np.zeros(
        (expert_loads.size, num_ranks),
        dtype=bool,
    )
    home_sizes = np.fromiter(
        (len(rank) for rank in assignments),
        dtype=np.int64,
        count=num_ranks,
    )
    home_experts = np.asarray(flattened_home, dtype=np.int64)
    home_ranks = np.repeat(np.arange(num_ranks, dtype=np.int64), home_sizes)
    assigned_sizes = home_sizes.copy()
    rank_loads[:] = np.bincount(
        home_ranks,
        weights=expert_loads[home_experts],
        minlength=num_ranks,
    )
    owner_mask[home_experts, home_ranks] = True
    copy_counts = np.ones(expert_loads.size, dtype=np.int64)
    expert_ids = np.flatnonzero(candidate_mask)
    candidate_count = expert_ids.size
    rank_ids = np.arange(num_ranks, dtype=np.int64)
    expert_grid = np.broadcast_to(expert_ids, (num_ranks, candidate_count))
    rank_grid = np.broadcast_to(rank_ids[:, None], expert_grid.shape)
    load_grid = np.broadcast_to(expert_loads[expert_ids], expert_grid.shape)

    for _ in range(num_replicas):
        candidate_copy_counts = copy_counts[expert_ids]
        candidate_expert_loads = expert_loads[expert_ids]
        old_per_copy = candidate_expert_loads / candidate_copy_counts
        new_per_copy = candidate_expert_loads / (candidate_copy_counts + 1)
        reduced_loads = (
            rank_loads[None, :]
            - (old_per_copy - new_per_copy)[:, None] * owner_mask[expert_ids]
        )

        rank_has_capacity = assigned_sizes < rank_capacities
        valid = (
            rank_has_capacity[:, None]
            & ~owner_mask[expert_ids].T
            & (candidate_copy_counts < num_ranks)[None, :]
        )
        if not np.any(valid):
            raise ValueError("CRAFT cannot place a fixed-home replica.")
        target_loads = reduced_loads.T + new_per_copy[None, :]
        max_loads = np.maximum(
            np.max(reduced_loads, axis=1)[None, :],
            target_loads,
        )
        load_sums = np.sum(reduced_loads, axis=1) + new_per_copy
        load_square_sums = (
            np.sum(reduced_loads * reduced_loads, axis=1)[None, :]
            + 2.0 * reduced_loads.T * new_per_copy[None, :]
            + new_per_copy[None, :] * new_per_copy[None, :]
        )
        variances = (
            load_square_sums / num_ranks
            - (load_sums[None, :] / num_ranks) ** 2
        )
        std_loads = np.sqrt(np.maximum(variances, 0.0))
        max_loads[~valid] = np.inf
        std_loads[~valid] = np.inf
        order = np.lexsort(
            (
                rank_grid.ravel(),
                expert_grid.ravel(),
                -load_grid.ravel(),
                std_loads.ravel(),
                max_loads.ravel(),
            )
        )
        rank_id, candidate_id = np.unravel_index(
            int(order[0]),
            valid.shape,
        )
        rank_id = int(rank_id)
        candidate_id = int(candidate_id)
        expert_id = int(expert_ids[candidate_id])
        rank_loads = reduced_loads[candidate_id].copy()
        rank_loads[rank_id] += new_per_copy[candidate_id]
        assignments[rank_id].append(expert_id)
        assigned_sizes[rank_id] += 1
        owner_mask[expert_id, rank_id] = True
        copy_counts[expert_id] += 1

    return assignments, rank_loads


def _balancedness(rank_loads: np.ndarray) -> float:
    max_load = float(np.max(rank_loads)) if rank_loads.size else 0.0
    if max_load <= 0:
        return 1.0
    return float(np.mean(rank_loads)) / max_load


def _home_rank_loads(
    hotness: np.ndarray,
    home_placements: list[list[list[int]]],
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(np.sum(hotness[layer_id, rank]))
                for rank in home_placements[layer_id]
            ]
            for layer_id in range(hotness.shape[0])
        ],
        dtype=np.float64,
    )


def estimate_replication_benefits(
    hotness: np.ndarray,
    options: list[int],
    home_placements: list[list[list[int]]],
    candidate_masks: np.ndarray,
    objective: str = "balancedness",
    *,
    home_rank_loads: np.ndarray,
) -> np.ndarray:
    hotness = np.asarray(hotness, dtype=np.float64)
    if objective not in ("balancedness", "critical_path"):
        raise ValueError(
            "CRAFT replica benefit objective must be balancedness or "
            "critical_path."
        )
    num_layers = hotness.shape[0]
    benefits = np.zeros((num_layers, len(options)), dtype=np.float64)
    baseline = np.zeros(num_layers, dtype=np.float64)

    for layer_id in range(num_layers):
        rank_loads = home_rank_loads[layer_id]
        baseline[layer_id] = (
            float(np.max(rank_loads))
            if objective == "critical_path"
            else _balancedness(rank_loads)
        )

    for option_id, num_replicas in enumerate(options):
        if num_replicas == 0:
            continue
        for layer_id in range(num_layers):
            home = home_placements[layer_id]
            base_loads = home_rank_loads[layer_id]
            extra_capacities = np.zeros(base_loads.size, dtype=np.int64)
            extra_capacities[
                np.argsort(base_loads, kind="stable")[:num_replicas]
            ] = 1
            capacities = np.asarray(
                [len(rank) for rank in home],
                dtype=np.int64,
            ) + extra_capacities
            _, rank_loads = place_layer_experts(
                hotness[layer_id],
                num_replicas,
                capacities,
                home,
                candidate_masks[layer_id],
            )
            score = (
                float(np.max(rank_loads))
                if objective == "critical_path"
                else _balancedness(rank_loads)
            )
            benefits[layer_id, option_id] = max(
                0.0,
                baseline[layer_id] - score
                if objective == "critical_path"
                else score - baseline[layer_id],
            )
    return benefits


def allocate_replica_budget(
    benefits: np.ndarray,
    options: list[int],
    total_replicas: int,
) -> np.ndarray:
    benefits = np.asarray(benefits, dtype=np.float64)
    if benefits.ndim != 2:
        raise ValueError("CRAFT benefits must have shape [layers, options].")
    if not options or 0 not in options:
        raise ValueError("CRAFT replica options must include zero.")
    if any(option < 0 for option in options):
        raise ValueError("CRAFT replica options must be non-negative.")
    if total_replicas < 0:
        raise ValueError("CRAFT total replicas must be non-negative.")
    num_layers = benefits.shape[0]
    if benefits.shape[1] != len(options):
        raise ValueError("CRAFT benefit columns must match replica options.")

    dp = np.full((num_layers + 1, total_replicas + 1), -np.inf, dtype=np.float64)
    choices = np.full((num_layers + 1, total_replicas + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for layer_id in range(1, num_layers + 1):
        previous = dp[layer_id - 1]
        for option_id, num_replicas in enumerate(options):
            if num_replicas > total_replicas:
                continue
            candidates = (
                previous[: total_replicas + 1 - num_replicas]
                + benefits[layer_id - 1, option_id]
            )
            current = dp[layer_id, num_replicas:]
            improved = candidates > current + 1e-12
            current[improved] = candidates[improved]
            choices[layer_id, num_replicas:][improved] = option_id

    capacity = total_replicas
    if not np.isfinite(dp[num_layers, capacity]):
        raise ValueError(
            f"CRAFT cannot allocate the exact replica budget {total_replicas}."
        )

    allocation = np.zeros(num_layers, dtype=np.int64)
    for layer_id in range(num_layers, 0, -1):
        option_id = int(choices[layer_id, capacity])
        if option_id < 0:
            raise ValueError("CRAFT failed to reconstruct the replica allocation.")
        allocation[layer_id - 1] = options[option_id]
        capacity -= options[option_id]
    return allocation


def interleaved_replica_capacities(
    layer_replicas: np.ndarray,
    num_ranks: int,
    rank_base_loads: np.ndarray | None = None,
) -> np.ndarray:
    layer_replicas = np.asarray(layer_replicas, dtype=np.int64)
    if rank_base_loads is not None:
        rank_base_loads = np.asarray(rank_base_loads, dtype=np.float64)
        if rank_base_loads.shape != (layer_replicas.size, num_ranks):
            raise ValueError(
                "CRAFT rank base loads must have shape [layers, ranks]."
            )
    assignment = np.repeat(
        (layer_replicas // num_ranks)[:, None],
        num_ranks,
        axis=1,
    )
    rank_totals = assignment.sum(axis=0)

    for layer_id, remainder in enumerate(layer_replicas % num_ranks):
        remainder = int(remainder)
        if remainder == 0:
            continue
        if rank_base_loads is not None:
            rank_ids = np.arange(num_ranks, dtype=np.int64)
            selected = np.lexsort(
                (
                    rank_ids,
                    rank_base_loads[layer_id],
                    rank_totals,
                )
            )[:remainder]
            assignment[layer_id, selected] += 1
            rank_totals[selected] += 1
            continue
        sorted_totals = np.sort(rank_totals)
        cutoff = sorted_totals[remainder - 1]
        must_choose = np.flatnonzero(rank_totals < cutoff)
        tied = np.flatnonzero(rank_totals == cutoff)
        tied_count = remainder - must_choose.size
        if tied_count > 0:
            positions = np.linspace(0, tied.size - 1, tied_count).astype(np.int64)
            selected = np.concatenate((must_choose, tied[positions]))
        else:
            selected = must_choose
        assignment[layer_id, selected] += 1
        rank_totals[selected] += 1
    return assignment


def _place_craft_layers(
    hotness: np.ndarray,
    layer_replicas: np.ndarray,
    extra_capacities: np.ndarray,
    home_placements: list[list[list[int]]],
    candidate_masks: np.ndarray,
    reuse_placements: list[list[list[int]]] | None = None,
    reuse_capacities: np.ndarray | None = None,
) -> list[list[list[int]]]:
    placements = []
    for layer_id in range(hotness.shape[0]):
        if (
            reuse_placements is not None
            and reuse_capacities is not None
            and np.array_equal(
                extra_capacities[layer_id],
                reuse_capacities[layer_id],
            )
        ):
            placements.append(reuse_placements[layer_id])
            continue
        home = home_placements[layer_id]
        rank_capacities = np.asarray(
            [len(rank) for rank in home],
            dtype=np.int64,
        ) + extra_capacities[layer_id]
        assignments, _ = place_layer_experts(
            hotness[layer_id],
            int(layer_replicas[layer_id]),
            rank_capacities,
            home,
            candidate_masks[layer_id],
        )
        placements.append(assignments)
    return placements


def _placement_imbalance(
    hotness: np.ndarray,
    placements: list[list[list[int]]],
) -> float:
    weighted_ratio = 0.0
    total_weight = 0.0
    num_experts = hotness.shape[1]
    for layer_id, rank_assignments in enumerate(placements):
        counts = np.bincount(
            [
                expert_id
                for rank in rank_assignments
                for expert_id in rank
            ],
            minlength=num_experts,
        )
        per_copy = np.divide(
            hotness[layer_id],
            counts,
            out=np.zeros(num_experts, dtype=np.float64),
            where=counts > 0,
        )
        rank_loads = np.asarray(
            [float(np.sum(per_copy[rank])) for rank in rank_assignments],
            dtype=np.float64,
        )
        mean_load = float(np.mean(rank_loads))
        layer_weight = float(np.sum(hotness[layer_id]))
        if mean_load > 0 and layer_weight > 0:
            weighted_ratio += float(np.max(rank_loads)) / mean_load * layer_weight
            total_weight += layer_weight
    return weighted_ratio / total_weight if total_weight > 0 else 0.0


def plan_craft_replication(
    hotness: np.ndarray,
    total_replicas: int,
    num_ranks: int,
    home_placements: list[list[list[int]]],
    candidate_top_m: int = 0,
    *,
    benefit_objective: str = "balancedness",
) -> tuple[np.ndarray, np.ndarray, list[list[list[int]]]]:
    hotness = np.asarray(hotness, dtype=np.float64)
    if hotness.ndim != 2:
        raise ValueError("CRAFT hotness must have shape [layers, experts].")
    if num_ranks <= 0 or hotness.shape[1] % num_ranks != 0:
        raise ValueError("CRAFT requires experts to be divisible by the rank count.")
    if total_replicas < 0 or total_replicas % num_ranks != 0:
        raise ValueError("CRAFT total replicas must be a non-negative rank multiple.")

    options = list(range(num_ranks + 1))
    max_replicas = hotness.shape[0] * options[-1]
    if total_replicas > max_replicas:
        raise ValueError(
            f"CRAFT replica budget {total_replicas} exceeds maximum {max_replicas}; "
            "the allocator supports at most one replica per rank per layer."
        )
    home_placements = _normalize_home_placements(
        home_placements,
        hotness.shape[0],
        hotness.shape[1],
        num_ranks,
    )
    home_loads = _home_rank_loads(hotness, home_placements)
    candidate_masks = np.asarray(
        [
            _replica_candidate_mask(
                hotness[layer_id],
                candidate_top_m,
                home_placements[layer_id],
            )
            for layer_id in range(hotness.shape[0])
        ],
        dtype=bool,
    )
    benefits = estimate_replication_benefits(
        hotness,
        options,
        home_placements,
        candidate_masks,
        objective=benefit_objective,
        home_rank_loads=home_loads,
    )
    layer_replicas = allocate_replica_budget(benefits, options, total_replicas)
    slot_only_capacities = interleaved_replica_capacities(
        layer_replicas,
        num_ranks,
    )
    extra_capacities = interleaved_replica_capacities(
        layer_replicas,
        num_ranks,
        rank_base_loads=home_loads,
    )
    placements = _place_craft_layers(
        hotness,
        layer_replicas,
        extra_capacities,
        home_placements,
        candidate_masks,
    )
    if not np.array_equal(extra_capacities, slot_only_capacities):
        slot_only_placements = _place_craft_layers(
            hotness,
            layer_replicas,
            slot_only_capacities,
            home_placements,
            candidate_masks,
            reuse_placements=placements,
            reuse_capacities=extra_capacities,
        )
        slot_only_imbalance = _placement_imbalance(
            hotness,
            slot_only_placements,
        )
        load_aware_imbalance = _placement_imbalance(hotness, placements)
        relative_improvement = (
            (slot_only_imbalance - load_aware_imbalance) / slot_only_imbalance
            if slot_only_imbalance > 0
            else 0.0
        )
        if relative_improvement < 1e-3:
            extra_capacities = slot_only_capacities
            placements = slot_only_placements
    return layer_replicas, extra_capacities, placements
