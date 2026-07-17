# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import numpy as np
import torch
from vllm.logger import logger

from .policy_abstract import DynamicConfig, EplbPolicy


def _get_int_config(config: DynamicConfig, attr: str, env_name: str, default: int) -> int:
    value = getattr(config, attr, None)
    if value is None:
        value = os.getenv(env_name, str(default))
    return int(value)


def _get_float_config(config: DynamicConfig, attr: str, env_name: str, default: float) -> float:
    value = getattr(config, attr, None)
    if value is None:
        value = os.getenv(env_name, str(default))
    return float(value)


class PoolBalanceEplb(EplbPolicy):
    """Dynamic EPLB policy for fixed-capacity CRAFT pool slots.

    The first `pool_start` local slots are fixed home experts. Only slots
    `[pool_start, E_local)` are replaced, so tensor shapes and main ownership
    stay stable while hot experts can rotate through the pool.
    """

    def __init__(self, config: DynamicConfig):
        self.config = config
        self.candidate_top_m = max(0, _get_int_config(config, "craft_pool_top_m", "CRAFT_POOL_TOP_M", 0))
        self.candidate_factor = max(
            1,
            _get_int_config(config, "craft_pool_top_m_factor", "CRAFT_POOL_TOPM_FACTOR", 4),
        )
        self.min_hotness_delta = max(
            0.0,
            _get_float_config(config, "craft_pool_min_hotness_delta", "CRAFT_POOL_MIN_HOTNESS_DELTA", 0.05),
        )
        self.min_improvement = max(
            0.0,
            _get_float_config(config, "craft_pool_min_improvement", "CRAFT_POOL_MIN_IMPROVEMENT", 0.01),
        )
        self.global_pool_total_size = max(
            0,
            _get_int_config(
                config,
                "craft_global_pool_size",
                "VLLM_ASCEND_CRAFT_GLOBAL_POOL_SIZE",
                0,
            ),
        )
        self._last_layer_hotness: dict[int, np.ndarray] = {}
        self._last_global_hotness: np.ndarray | None = None

    @staticmethod
    def _infer_pool_start(current_expert_table: torch.Tensor) -> int:
        _, num_ranks, _ = current_expert_table.shape
        valid_experts = current_expert_table[current_expert_table >= 0]
        if valid_experts.numel() == 0:
            return 0
        num_logical_experts = int(torch.max(valid_experts).item()) + 1
        return num_logical_experts // num_ranks

    @staticmethod
    def _expert_hotness(current_expert_table: torch.Tensor, expert_workload: torch.Tensor) -> np.ndarray:
        table = current_expert_table.detach().cpu().numpy()
        load = (
            expert_workload.detach().cpu().numpy()
            if torch.is_tensor(expert_workload)
            else np.asarray(expert_workload)
        )
        num_layers = table.shape[0]
        valid_experts = table[table >= 0]
        num_experts = int(valid_experts.max()) + 1 if valid_experts.size else 0
        hotness = np.zeros((num_layers, num_experts), dtype=np.float64)
        valid = table >= 0
        layer_ids = np.broadcast_to(
            np.arange(num_layers, dtype=np.int64)[:, None, None],
            table.shape,
        )
        np.add.at(hotness, (layer_ids[valid], table[valid]), load[valid])
        return hotness

    def _candidate_experts(self, hotness, pool_size, num_ranks):
        if pool_size <= 0 or len(hotness) == 0:
            return np.array([], dtype=np.int64)
        top_m = self.candidate_top_m
        if top_m <= 0:
            top_m = max(pool_size * num_ranks * self.candidate_factor, pool_size * num_ranks)
        top_m = min(len(hotness), top_m)
        if top_m >= len(hotness):
            candidates = np.arange(len(hotness), dtype=np.int64)
        else:
            candidates = np.argpartition(-hotness, top_m - 1)[:top_m].astype(np.int64)
        return candidates[hotness[candidates] > 0]

    @staticmethod
    def _rank_load(home, pool_sets, hotness):
        num_ranks = len(home)
        copy_counts = np.ones(len(hotness), dtype=np.float64)
        for pool in pool_sets:
            for expert_id in pool:
                copy_counts[expert_id] += 1.0
        per_copy = np.divide(hotness, copy_counts, out=np.zeros_like(hotness, dtype=np.float64), where=copy_counts > 0)
        rank_load = []
        for rank_id in range(num_ranks):
            experts = set(home[rank_id]) | set(pool_sets[rank_id])
            rank_load.append(sum(per_copy[expert_id] for expert_id in experts))
        return np.asarray(rank_load, dtype=np.float64)

    def _should_skip_layer(self, layer_id, hotness):
        if self.min_hotness_delta <= 0:
            return False
        total_hotness = float(np.sum(np.abs(hotness)))
        normalized_hotness = (
            hotness / total_hotness
            if total_hotness > 0
            else np.zeros_like(hotness, dtype=np.float64)
        )
        last_hotness = self._last_layer_hotness.get(layer_id)
        if last_hotness is None or last_hotness.shape != normalized_hotness.shape:
            self._last_layer_hotness[layer_id] = normalized_hotness.copy()
            return False
        delta = float(np.sum(np.abs(normalized_hotness - last_hotness)))
        if delta < self.min_hotness_delta:
            return True
        self._last_layer_hotness[layer_id] = normalized_hotness.copy()
        return False

    def _desired_pool_sets(self, home, hotness, pool_size):
        num_ranks = len(home)
        if pool_size <= 0:
            return [set() for _ in range(num_ranks)]

        num_experts = len(hotness)
        candidate_experts = self._candidate_experts(hotness, pool_size, num_ranks)
        if candidate_experts.size == 0:
            return [set() for _ in range(num_ranks)]
        candidate_set = set(int(expert_id) for expert_id in candidate_experts)
        extra_copies = [0] * num_experts
        for _ in range(pool_size * num_ranks):
            expert_id = max(
                candidate_experts,
                key=lambda i: hotness[i] / (1 + extra_copies[i])
                if extra_copies[i] < num_ranks - 1 else -1.0,
            )
            expert_id = int(expert_id)
            if hotness[expert_id] <= 0 or extra_copies[expert_id] >= num_ranks - 1:
                break
            extra_copies[expert_id] += 1

        per_copy = [hotness[e] / (1 + extra_copies[e]) for e in range(num_experts)]
        pool = [set() for _ in range(num_ranks)]
        card_experts = [set(card) for card in home]
        card_load = [sum(per_copy[e] for e in card) for card in home]
        items = sorted(
            ((int(expert_id), per_copy[int(expert_id)]) for expert_id in candidate_experts
             for _ in range(extra_copies[int(expert_id)])),
            key=lambda x: -x[1],
        )

        for expert_id, weight in items:
            if expert_id not in candidate_set:
                continue
            best_rank = -1
            best_load = float("inf")
            for rank_id in range(num_ranks):
                if (
                    len(pool[rank_id]) < pool_size
                    and expert_id not in card_experts[rank_id]
                    and card_load[rank_id] < best_load
                ):
                    best_rank = rank_id
                    best_load = card_load[rank_id]
            if best_rank < 0:
                continue
            pool[best_rank].add(expert_id)
            card_experts[best_rank].add(expert_id)
            card_load[best_rank] += weight

        return pool

    @staticmethod
    def _complete_pool_with_old(desired_pool, old_pool, home, pool_size):
        completed = []
        for desired, old, home_experts in zip(desired_pool, old_pool, home):
            rank_pool = set(desired)
            home_set = set(home_experts)
            for expert_id in sorted(old):
                if len(rank_pool) >= pool_size:
                    break
                if expert_id not in home_set:
                    rank_pool.add(expert_id)
            completed.append(rank_pool)
        return completed

    def _should_apply_pool_update(self, home, old_pool, desired_pool, hotness):
        if self.min_improvement <= 0:
            return True
        old_load = self._rank_load(home, old_pool, hotness)
        new_load = self._rank_load(home, desired_pool, hotness)
        old_max = float(np.max(old_load)) if old_load.size else 0.0
        new_max = float(np.max(new_load)) if new_load.size else 0.0
        if old_max <= 0:
            return new_max < old_max
        improvement = (old_max - new_max) / old_max
        return improvement >= self.min_improvement

    @staticmethod
    def _infer_layer_local_slots(layer_table, pool_start):
        valid_counts = (layer_table >= 0).sum(axis=1)
        if valid_counts.size == 0:
            return pool_start
        return max(pool_start, int(valid_counts.max()))

    @staticmethod
    def _stable_pool_slots(old_rank_slots, desired_pool, pool_start, local_slots):
        new_rank_slots = old_rank_slots.copy()
        pool_slots = list(range(pool_start, local_slots))
        desired = set(desired_pool)
        if local_slots < len(new_rank_slots):
            new_rank_slots[local_slots:] = -1

        kept = set()
        free_slots = []
        for slot_id in pool_slots:
            expert_id = int(old_rank_slots[slot_id])
            if expert_id in desired:
                kept.add(expert_id)
            else:
                free_slots.append(slot_id)

        missing = sorted(desired - kept)
        for slot_id, expert_id in zip(free_slots, missing):
            new_rank_slots[slot_id] = expert_id
        return new_rank_slots

    def _global_hotness_changed(self, hotness: np.ndarray, total_slots: int) -> bool:
        if self.min_hotness_delta <= 0:
            return True
        flat_hotness = np.abs(hotness).reshape(-1)
        total_hotness = float(flat_hotness.sum())
        normalized = (
            flat_hotness / total_hotness
            if total_hotness > 0
            else np.zeros_like(flat_hotness, dtype=np.float64)
        )
        previous = self._last_global_hotness
        if previous is None or previous.shape != normalized.shape:
            self._last_global_hotness = normalized.copy()
            return True

        top_m = self.candidate_top_m
        if top_m <= 0:
            top_m = max(total_slots * self.candidate_factor, total_slots)

        def top_indices(values):
            positive = np.flatnonzero(values > 0)
            count = min(top_m, positive.size)
            if count == 0:
                return np.empty(0, dtype=np.int64)
            if count == positive.size:
                return positive
            return positive[np.argpartition(-values[positive], count - 1)[:count]]

        candidates = np.union1d(top_indices(normalized), top_indices(previous))
        delta = float(np.sum(np.abs(normalized[candidates] - previous[candidates])))
        if delta < self.min_hotness_delta:
            return False
        self._last_global_hotness = normalized.copy()
        return True

    def _global_candidates(self, hotness: np.ndarray, total_slots: int) -> list[tuple[int, int]]:
        flat = hotness.reshape(-1)
        positive = np.flatnonzero(flat > 0)
        if positive.size == 0:
            return []
        top_m = self.candidate_top_m
        if top_m <= 0:
            top_m = max(total_slots * self.candidate_factor, total_slots)
        top_m = min(top_m, positive.size)
        selected = positive[np.argpartition(-flat[positive], top_m - 1)[:top_m]]
        selected = selected[np.argsort(-flat[selected], kind="stable")]
        num_experts = hotness.shape[1]
        return [(int(index // num_experts), int(index % num_experts)) for index in selected]

    @staticmethod
    def _layer_rank_loads(home, assignments, hotness, layer_id):
        copy_counts = np.ones(len(hotness), dtype=np.float64)
        for rank_items in assignments:
            for item_layer, expert_id in rank_items:
                if item_layer == layer_id:
                    copy_counts[expert_id] += 1.0
        per_copy = np.divide(
            hotness,
            copy_counts,
            out=np.zeros_like(hotness, dtype=np.float64),
            where=copy_counts > 0,
        )
        loads = []
        for rank_id, home_experts in enumerate(home):
            experts = set(home_experts)
            experts.update(
                expert_id
                for item_layer, expert_id in assignments[rank_id]
                if item_layer == layer_id
            )
            loads.append(sum(per_copy[expert_id] for expert_id in experts))
        return np.asarray(loads, dtype=np.float64)

    @staticmethod
    def _global_assignment_gain(
        rank_loads,
        hotness,
        copy_counts,
        hosts,
        layer_id,
        expert_id,
        rank_id,
    ):
        copies = int(copy_counts[layer_id, expert_id])
        expert_hotness = float(hotness[layer_id, expert_id])
        old_loads = rank_loads[layer_id]
        new_loads = old_loads.copy()
        decrease = expert_hotness / (copies * (copies + 1))
        for host_rank in hosts[layer_id][expert_id]:
            new_loads[host_rank] -= decrease
        new_loads[rank_id] += expert_hotness / (copies + 1)
        return float(old_loads.max() - new_loads.max()), new_loads

    def _desired_global_assignments(
        self,
        home,
        hotness,
        pool_size,
        preferred_assignments=None,
    ):
        num_layers, num_experts = hotness.shape
        num_ranks = len(home[0]) if home else 0
        if num_ranks == 0:
            return []
        total_slots = pool_size * num_ranks
        candidates = self._global_candidates(hotness, total_slots)
        assignments: list[list[tuple[int, int]]] = [[] for _ in range(num_ranks)]
        copy_counts = np.ones((num_layers, num_experts), dtype=np.int64)
        hosts = [[[] for _ in range(num_experts)] for _ in range(num_layers)]
        rank_loads = np.zeros((num_layers, num_ranks), dtype=np.float64)
        for layer_id in range(num_layers):
            for rank_id in range(num_ranks):
                for expert_id in home[layer_id][rank_id]:
                    hosts[layer_id][expert_id].append(rank_id)
                    rank_loads[layer_id, rank_id] += hotness[layer_id, expert_id]

        preferred = preferred_assignments or [[] for _ in range(num_ranks)]

        def eligible(layer_id, expert_id, rank_id):
            return (
                len(assignments[rank_id]) < pool_size
                and rank_id not in hosts[layer_id][expert_id]
                and copy_counts[layer_id, expert_id] < num_ranks
            )

        def best_choice(candidate_items):
            best = None
            for layer_id, expert_id in candidate_items:
                for rank_id in range(num_ranks):
                    if not eligible(layer_id, expert_id, rank_id):
                        continue
                    gain, new_loads = self._global_assignment_gain(
                        rank_loads,
                        hotness,
                        copy_counts,
                        hosts,
                        layer_id,
                        expert_id,
                        rank_id,
                    )
                    key = (
                        gain,
                        float(hotness[layer_id, expert_id]),
                        -len(assignments[rank_id]),
                        -layer_id,
                        -expert_id,
                        -rank_id,
                    )
                    if best is None or key > best[0]:
                        best = (key, layer_id, expert_id, rank_id, new_loads)
            return best

        for _ in range(total_slots):
            choice = best_choice(candidates)
            if choice is None or choice[0][0] <= 0:
                preferred_items = [
                    item
                    for rank_items in preferred
                    for item in rank_items
                    if item not in candidates
                ]
                preferred_choice = best_choice(preferred_items)
                if preferred_choice is not None and (
                    choice is None or preferred_choice[0] > choice[0]
                ):
                    choice = preferred_choice
            if choice is None or choice[0][0] <= 0:
                break
            _, layer_id, expert_id, rank_id, new_loads = choice
            assignments[rank_id].append((layer_id, expert_id))
            copy_counts[layer_id, expert_id] += 1
            hosts[layer_id][expert_id].append(rank_id)
            rank_loads[layer_id] = new_loads
        return assignments

    @staticmethod
    def _place_global_assignments(old_table, desired, pool_start, pool_size):
        new_table = old_table.copy()
        new_table[:, :, pool_start:] = -1
        num_layers, num_ranks, _ = old_table.shape
        for rank_id in range(num_ranks):
            old_slots: list[tuple[int, int] | None] = []
            for pool_slot in range(pool_size):
                slot_id = pool_start + pool_slot
                owners = [
                    (layer_id, int(old_table[layer_id, rank_id, slot_id]))
                    for layer_id in range(num_layers)
                    if old_table[layer_id, rank_id, slot_id] >= 0
                ]
                if len(owners) > 1:
                    raise ValueError(
                        "CRAFT global pool slot has multiple layer owners: "
                        f"rank={rank_id}, slot={pool_slot}, owners={owners}."
                    )
                old_slots.append(owners[0] if owners else None)

            desired_set = set(desired[rank_id])
            assigned = set()
            free_slots = []
            for pool_slot, old_item in enumerate(old_slots):
                if old_item is not None and old_item in desired_set:
                    layer_id, expert_id = old_item
                    new_table[layer_id, rank_id, pool_start + pool_slot] = expert_id
                    assigned.add(old_item)
                else:
                    free_slots.append(pool_slot)

            missing = [item for item in desired[rank_id] if item not in assigned]
            for pool_slot, (layer_id, expert_id) in zip(free_slots, missing):
                new_table[layer_id, rank_id, pool_start + pool_slot] = expert_id
        return new_table

    @staticmethod
    def _global_imbalance(table, hotness):
        weighted_ratio = 0.0
        total_weight = 0.0
        for layer_id in range(table.shape[0]):
            valid = table[layer_id] >= 0
            counts = np.bincount(
                table[layer_id][valid],
                minlength=hotness.shape[1],
            )
            per_copy = np.divide(
                hotness[layer_id],
                counts,
                out=np.zeros_like(hotness[layer_id], dtype=np.float64),
                where=counts > 0,
            )
            rank_loads = np.asarray(
                [per_copy[rank[rank >= 0]].sum() for rank in table[layer_id]],
                dtype=np.float64,
            )
            mean_load = float(rank_loads.mean())
            layer_weight = float(np.sum(hotness[layer_id]))
            if mean_load > 0 and layer_weight > 0:
                weighted_ratio += float(rank_loads.max()) / mean_load * layer_weight
                total_weight += layer_weight
        return weighted_ratio / total_weight if total_weight > 0 else 0.0

    def _rebalance_global_pool(self, current_expert_table, expert_workload, pool_start):
        old_table = current_expert_table.detach().cpu().numpy()
        pool_size = old_table.shape[2] - pool_start
        num_ranks = old_table.shape[1]
        if pool_size <= 0 or pool_size * num_ranks != self.global_pool_total_size:
            raise ValueError(
                "CRAFT global pool table capacity mismatch: "
                f"configured={self.global_pool_total_size}, table={pool_size * num_ranks}."
            )
        hotness = self._expert_hotness(current_expert_table, expert_workload)
        if not self._global_hotness_changed(hotness, pool_size * num_ranks):
            return False, None, current_expert_table.tolist()
        home = [
            [
                [int(expert_id) for expert_id in old_table[layer_id, rank_id, :pool_start] if expert_id >= 0]
                for rank_id in range(num_ranks)
            ]
            for layer_id in range(old_table.shape[0])
        ]
        preferred_assignments = [[] for _ in range(num_ranks)]
        for layer_id in range(old_table.shape[0]):
            for rank_id in range(num_ranks):
                preferred_assignments[rank_id].extend(
                    (layer_id, int(expert_id))
                    for expert_id in old_table[layer_id, rank_id, pool_start:]
                    if expert_id >= 0
                )
        desired = self._desired_global_assignments(
            home,
            hotness,
            pool_size,
            preferred_assignments=preferred_assignments,
        )
        new_table = self._place_global_assignments(old_table, desired, pool_start, pool_size)
        old_imbalance = self._global_imbalance(old_table, hotness)
        new_imbalance = self._global_imbalance(new_table, hotness)
        relative_improvement = (
            (old_imbalance - new_imbalance) / old_imbalance
            if old_imbalance > 0
            else 0.0
        )
        changed_slots = int(np.count_nonzero(old_table != new_table))
        changed_layers = int(
            np.count_nonzero(np.any(old_table != new_table, axis=(1, 2)))
        )
        accepted = relative_improvement >= self.min_improvement
        logger.info(
            "[CRAFT-GLOBAL-BALANCE] slots=%d changed_slots=%d changed_layers=%d "
            "current=%.4f proposed=%.4f improvement=%.4f accepted=%s",
            pool_size * num_ranks,
            changed_slots,
            changed_layers,
            old_imbalance,
            new_imbalance,
            relative_improvement,
            accepted,
        )
        if not accepted:
            return False, None, current_expert_table.tolist()
        changed = not np.array_equal(old_table, new_table)
        return changed, None, new_table.tolist()

    def rebalance_experts(self, current_expert_table, expert_workload):
        if not torch.is_tensor(current_expert_table):
            current_expert_table = torch.tensor(current_expert_table)
        pool_start = self._infer_pool_start(current_expert_table)
        if pool_start <= 0:
            return False, None, current_expert_table.tolist()
        if self.global_pool_total_size > 0:
            return self._rebalance_global_pool(current_expert_table, expert_workload, pool_start)

        hotness = self._expert_hotness(current_expert_table, expert_workload)
        old_table = current_expert_table.detach().cpu().numpy()
        new_table = old_table.copy()

        for layer_id in range(old_table.shape[0]):
            local_slots = self._infer_layer_local_slots(old_table[layer_id], pool_start)
            pool_size = local_slots - pool_start
            if pool_size <= 0:
                continue
            layer_hotness = hotness[layer_id]
            if self._should_skip_layer(layer_id, layer_hotness):
                continue
            home = [
                [int(expert_id) for expert_id in old_table[layer_id, rank_id, :pool_start] if expert_id >= 0]
                for rank_id in range(old_table.shape[1])
            ]
            old_pool = [
                set(
                    int(expert_id)
                    for expert_id in old_table[layer_id, rank_id, pool_start:local_slots]
                    if expert_id >= 0
                )
                for rank_id in range(old_table.shape[1])
            ]
            desired_pool = self._desired_pool_sets(home, layer_hotness, pool_size)
            desired_pool = self._complete_pool_with_old(desired_pool, old_pool, home, pool_size)
            if not self._should_apply_pool_update(home, old_pool, desired_pool, layer_hotness):
                continue
            for rank_id in range(old_table.shape[1]):
                new_table[layer_id, rank_id] = self._stable_pool_slots(
                    old_table[layer_id, rank_id],
                    desired_pool[rank_id],
                    pool_start,
                    local_slots,
                )

        changed = not np.array_equal(old_table, new_table)
        return changed, None, new_table.tolist()
