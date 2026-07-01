# Copyright (c) 2024 CRAFT Contributors. All rights reserved.
"""CRAFT: Cost-aware expert Replica Allocation with layerwise variable NR.

Implements the CRAFT paper's three-step approach as a dynamic EPLB policy:

1. **Per-layer load analysis**: Aggregate expert workload to per-logical-expert
   load, estimate each layer's imbalance (max/mean ratio).
2. **Layerwise NR allocation**: High-skew layers receive more replicas, low-skew
   layers receive fewer. Uses a threshold-based approach: find the minimum NR
   per layer that achieves the improvement threshold.
3. **Balanced bin-packing**: Greedy LPT (Longest Processing Time) bin-pack to
   spread expert replicas across devices, minimizing the maximum device load.

For dynamic EPLB compatibility, layers with fewer replicas are padded with -1
to maintain a uniform slot count (max_slots). This preserves:
- Weight tensor shape (cudagraph compatible)
- Expert map format ([L, npus, n_logical] with -1 for unused)
- Migration logic (compose skips -1, empty d2d is a no-op)

Environment variables (all optional, safe defaults):
    CRAFT_LAYERWISE:     1=enable layerwise NR (default), 0=uniform NR
    CRAFT_TOPK:          Only rebalance top-k most imbalanced layers (default: 6)
    CRAFT_IMPROVE_THRESH: Accept new placement if max_load < thresh * current (default: 0.97)
    CRAFT_REAL_EVAL:     1=use polling-aware real max for gating (default: 0)
"""
import os

import numpy as np

from .policy_abstract import DynamicConfig, EplbPolicy


def _alloc_layer(load, ep_size, n_redundant, max_slots=None):
    """CRAFT single-layer bin-packing.

    Allocates n_redundant replica slots to spread expert load across ep_size
    devices. Each replica reduces the effective per-copy load of the hottest
    expert. Items are bin-packed (LPT) to minimize the maximum device load.

    Args:
        load: Per-expert load array [n_experts].
        ep_size: Number of devices (EP size).
        n_redundant: Number of redundant replica slots for this layer.
        max_slots: If provided and larger than the natural slot count, pad
            remaining slots with -1 (for layerwise variable NR). If None,
            pad with the last expert (original behavior).

    Returns:
        cards: List of [ep_size] device lists, each containing slot entries
               (expert IDs or -1 for padding).
        max_card_load: The maximum total load across all devices.
    """
    n = len(load)
    slots = (n + n_redundant) // ep_size
    if max_slots is None:
        max_slots = slots

    # Step 1: Decide replica count per expert (greedy: hottest first)
    copies = [1] * n
    eff_load = list(load)
    for _ in range(n_redundant):
        hottest = max(range(n),
                      key=lambda i: eff_load[i] if copies[i] < ep_size else -1.0)
        if load[hottest] <= 0 or copies[hottest] >= ep_size:
            break
        copies[hottest] += 1
        eff_load[hottest] = load[hottest] / copies[hottest]

    # Step 2: Build items (one per expert copy) sorted by per-copy load (desc)
    items = sorted(
        ((expert_id, load[expert_id] / copies[expert_id])
         for expert_id in range(n) for _ in range(copies[expert_id])),
        key=lambda x: -x[1],
    )

    # Step 3: Bin-pack items into ep_size cards (LPT greedy)
    cards = [[] for _ in range(ep_size)]
    card_experts = [set() for _ in range(ep_size)]
    card_load = [0.0] * ep_size
    for expert_id, per_copy_load in items:
        target = -1
        target_load = float("inf")
        for c in range(ep_size):
            if (len(cards[c]) < slots and expert_id not in card_experts[c]
                    and card_load[c] < target_load):
                target = c
                target_load = card_load[c]
        if target < 0:
            target = next((c for c in range(ep_size) if len(cards[c]) < slots), 0)
        cards[target].append(expert_id)
        card_experts[target].add(expert_id)
        card_load[target] += per_copy_load

    # Step 4: Pad to max_slots
    for c in range(ep_size):
        while len(cards[c]) < max_slots:
            if max_slots != slots:
                # Layerwise padding: mark unused slots with -1
                cards[c].append(-1)
            else:
                # Original padding: duplicate last expert
                cards[c].append(cards[c][-1] if cards[c] else 0)

    return cards, (max(card_load) if card_load else 0.0)


def _constraint_local_exchange(current_table, new_deployment):
    """Reorder experts within each card to minimize weight migration.

    The EPLB worker constraint requires that experts already resident on a card
    stay in their current slot (no weight reload needed). This function reorders
    the new placement's experts to match the current placement's slot positions
    where possible, only placing new experts in changed slots.

    -1 padding entries are kept fixed and do not participate in reordering.
    """
    for layer_id in range(len(new_deployment)):
        for card_id in range(len(new_deployment[layer_id])):
            current_slots = [int(x) for x in current_table[layer_id][card_id]]
            new_slots = [int(x) for x in new_deployment[layer_id][card_id]]
            num_slots = len(new_slots)

            result = [-1] * num_slots
            slot_used = [False] * num_slots
            unplaced = []

            # Pass 1: Place experts that exist in both current and new at their
            # current slot position (no weight reload needed)
            for i in range(num_slots):
                if new_slots[i] == -1:
                    result[i] = -1
                    slot_used[i] = True
                    continue
                matched = False
                for j in range(num_slots):
                    if (not slot_used[j] and current_slots[j] == new_slots[i]
                            and current_slots[j] != -1):
                        slot_used[j] = True
                        result[j] = new_slots[i]
                        matched = True
                        break
                if not matched:
                    unplaced.append(new_slots[i])

            # Pass 2: Fill remaining experts into unused non-padding slots
            idx = 0
            for i in range(num_slots):
                if idx >= len(unplaced):
                    break
                if not slot_used[i] and new_slots[i] != -1:
                    result[i] = unplaced[idx]
                    slot_used[i] = True
                    idx += 1

            new_deployment[layer_id][card_id] = result

    return new_deployment


def _predict_real_max(cards, placement_layer, workload_layer, n_logical, npus):
    """Simulate ep_rank % num_replicas polling to estimate the real max card load.

    The framework's generate_log2phy_map uses ep_rank % d to bind each rank to
    a replica. This means all tokens from a given rank go to the same replica.
    If per-rank load is uneven, the real max card load can be much higher than
    the ideal (load/copies) estimate.

    This function simulates the polling behavior to provide a more accurate
    max card load for the REAL_EVAL gating mechanism.
    """
    # 1. Extract per-rank per-expert load from current placement + workload
    rank_expert_load = {}
    for rank in range(npus):
        placement_row = placement_layer[rank]
        workload_row = workload_layer[rank]
        for slot_idx in range(len(placement_row)):
            expert_id = int(placement_row[slot_idx])
            if 0 <= expert_id < n_logical:
                key = (expert_id, rank)
                rank_expert_load[key] = rank_expert_load.get(key, 0.0) + float(
                    workload_row[slot_idx])

    # 2. Map new cards to replica groups (by rank, ascending = deterministic)
    replica_ranks = {}
    for rank in range(npus):
        for expert_raw in cards[rank]:
            expert_id = int(expert_raw)
            if 0 <= expert_id < n_logical:
                replica_ranks.setdefault(expert_id, [])
                if rank not in replica_ranks[expert_id]:
                    replica_ranks[expert_id].append(rank)

    # 3. Simulate ep_rank % d polling and compute per-card load
    card_load = [0.0] * npus
    for expert_id, rank_list in replica_ranks.items():
        num_replicas = len(rank_list)
        if num_replicas == 0:
            continue
        for replica_idx in range(num_replicas):
            replica_load = sum(
                rank_expert_load.get((expert_id, r), 0.0)
                for r in range(npus) if r % num_replicas == replica_idx)
            card_load[rank_list[replica_idx]] += replica_load

    return max(card_load) if card_load else 0.0


class CraftPolicy(EplbPolicy):
    """CRAFT dynamic EPLB policy with layerwise variable replica count.

    Implements the CRAFT paper's cost-aware replica allocation as a drop-in
    dynamic EPLB policy. Key features:
    - Layerwise NR: each layer gets the minimum replicas needed for balance
    - Bin-pack placement: LPT greedy to minimize max device load
    - Incremental adjustment: only rebalance top-k most imbalanced layers
    - REAL_EVAL: optional polling-aware max estimation for accurate gating
    """

    def __init__(self, config: DynamicConfig):
        super().__init__(config)
        # Incremental adjustment: only rebalance top-k most imbalanced layers
        self.topk = int(os.environ.get("CRAFT_TOPK", "6"))
        # Accept new placement if predicted max < thresh * current max
        self.improve_thresh = float(os.environ.get("CRAFT_IMPROVE_THRESH", "0.97"))
        # Layerwise variable NR (CRAFT paper core)
        self.layerwise = int(os.environ.get("CRAFT_LAYERWISE", "1"))
        # Polling-aware real max for gating (0=ideal load/copies, 1=simulated)
        self.real_eval = int(os.environ.get("CRAFT_REAL_EVAL", "0"))

    def _allocate_layer_nr(self, load, npus, max_nr, cur_layer_max):
        """Find the minimum replicas needed for this layer to meet the threshold.

        Tries NR values from 0 upward (in steps of ep_size for slot alignment).
        Returns the smallest NR where _alloc_layer achieves sufficient improvement.
        """
        if not self.layerwise:
            return max_nr

        for nr in range(0, max_nr + 1, npus):
            _, predicted_max = _alloc_layer(load.tolist(), npus, nr)
            if predicted_max < self.improve_thresh * cur_layer_max:
                return nr
        return max_nr

    def rebalance_experts(self, current_expert_table, expert_workload):
        """Compute new expert placement using CRAFT algorithm.

        Args:
            current_expert_table: [L, npus, slots] current expert placement.
            expert_workload: [L, npus, slots] measured per-slot workload.

        Returns:
            Tuple of (change_flag, layer_priority, new_placement).
        """
        workload = np.array(expert_workload)
        placement = np.array(current_expert_table)
        num_layers, npus, slots = workload.shape
        n_logical = int(len(np.unique(placement[0])))
        n_redundant = npus * slots - n_logical

        # Step 1: Per-layer load analysis
        cur_layer_max = np.array(
            [float(workload[l].sum(axis=1).max()) for l in range(num_layers)])
        layer_priority = np.argsort(cur_layer_max)[::-1].tolist()
        hot_layers = set(layer_priority[:self.topk])

        # Step 2: Compute new placement for hot layers
        new_deployment = [None] * num_layers
        cur_heat = 0.0
        new_heat = 0.0
        improved = 0

        for layer in range(num_layers):
            cur_heat += float(cur_layer_max[layer])

            if layer not in hot_layers:
                new_deployment[layer] = placement[layer].tolist()
                new_heat += float(cur_layer_max[layer])
                continue

            # Aggregate per-expert load
            expert_load = np.zeros(n_logical, dtype=float)
            placement_flat = placement[layer].reshape(-1)
            workload_flat = workload[layer].reshape(-1)
            for expert_id, w in zip(placement_flat, workload_flat):
                eid = int(expert_id)
                if 0 <= eid < n_logical:
                    expert_load[eid] += float(w)

            # Determine per-layer NR (layerwise or uniform)
            layer_nr = self._allocate_layer_nr(
                expert_load, npus, n_redundant, cur_layer_max[layer])

            # Bin-pack placement
            cards, predicted_max = _alloc_layer(
                expert_load.tolist(), npus, layer_nr,
                max_slots=slots if self.layerwise else None)

            # REAL_EVAL: use polling-aware max if enabled
            if self.real_eval:
                predicted_max = _predict_real_max(
                    cards, placement[layer], workload[layer], n_logical, npus)

            # Accept if improvement meets threshold
            if predicted_max < self.improve_thresh * cur_layer_max[layer]:
                new_deployment[layer] = cards
                new_heat += predicted_max
                improved += 1
            else:
                new_deployment[layer] = placement[layer].tolist()
                new_heat += float(cur_layer_max[layer])

        # Step 3: Decide whether to apply changes
        change = 1 if (improved > 0
                       and new_heat < self.improve_thresh * cur_heat) else 0
        if not change:
            return 0, layer_priority, np.array(current_expert_table).tolist()

        # Step 4: Constraint exchange (minimize weight migration)
        current_table = np.array(current_expert_table).tolist()
        new_deployment = _constraint_local_exchange(current_table, new_deployment)

        return change, layer_priority, new_deployment
