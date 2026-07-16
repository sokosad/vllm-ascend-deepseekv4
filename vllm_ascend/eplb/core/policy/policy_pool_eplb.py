# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import numpy as np
import torch

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
        self._last_layer_hotness: dict[int, np.ndarray] = {}

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
        for layer_id in range(num_layers):
            for rank_id in range(table.shape[1]):
                for slot_id in range(table.shape[2]):
                    expert_id = int(table[layer_id, rank_id, slot_id])
                    if expert_id >= 0:
                        hotness[layer_id, expert_id] += float(load[layer_id, rank_id, slot_id])
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

    def rebalance_experts(self, current_expert_table, expert_workload):
        if not torch.is_tensor(current_expert_table):
            current_expert_table = torch.tensor(current_expert_table)
        pool_start = self._infer_pool_start(current_expert_table)
        if pool_start <= 0:
            return False, None, current_expert_table.tolist()

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
