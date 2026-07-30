# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import numpy as np
import torch
from vllm.logger import logger

from .craft_paper_allocator import plan_craft_replication
from .policy_abstract import DynamicConfig, EplbPolicy


_GLOBAL_COOLDOWN_BREAK_MIN_DELTA = 0.20
_GLOBAL_COOLDOWN_BREAK_DELTA_FACTOR = 4.0
_GLOBAL_COOLDOWN_BREAK_CONFIRMATIONS = 2
_GLOBAL_COOLDOWN_BREAK_MIN_VOLUME_RATIO = 0.25


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


def _get_str_config(
    config: DynamicConfig,
    attr: str,
    env_name: str,
    default: str,
) -> str:
    value = getattr(config, attr, None)
    if value is None:
        value = os.getenv(env_name, default)
    return str(value)


class PoolBalanceEplb(EplbPolicy):
    """Dynamic EPLB policy for fixed-capacity CRAFT pool storage.

    Layer-local pools keep the first `pool_start` slots fixed. A global pool
    shares extra storage across layers and assigns replicas with a CRAFT-style
    per-layer budget while keeping base ownership and tensor shapes stable.
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
        self.global_planner_objective = _get_str_config(
            config,
            "craft_global_planner_objective",
            "CRAFT_GLOBAL_PLANNER_OBJECTIVE",
            "balancedness",
        ).strip().lower()
        if self.global_planner_objective not in (
            "balancedness",
            "critical_path",
        ):
            raise ValueError(
                "CRAFT global planner objective must be balancedness or "
                "critical_path."
            )
        self.global_rebalance_cooldown = max(
            0,
            _get_int_config(
                config,
                "craft_global_rebalance_cooldown",
                "CRAFT_GLOBAL_REBALANCE_COOLDOWN",
                0,
            ),
        )
        self.layer_rebalance_cooldown = max(
            0,
            _get_int_config(
                config,
                "craft_layer_rebalance_cooldown",
                "CRAFT_LAYER_REBALANCE_COOLDOWN",
                0,
            ),
        )
        self._global_rebalance_cooldown_remaining = 0
        self._layer_rebalance_cooldown_remaining: dict[int, int] = {}
        self._last_layer_hotness: dict[int, np.ndarray] = {}
        self._last_global_hotness: np.ndarray | None = None
        self._last_global_plan_hotness: np.ndarray | None = None
        self._last_global_plan_volume: float | None = None
        self._global_cooldown_shift_streak = 0

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

    def _layer_cooldown_active(self, layer_id: int) -> bool:
        remaining = self._layer_rebalance_cooldown_remaining.get(layer_id, 0)
        if remaining <= 0:
            return False
        self._layer_rebalance_cooldown_remaining[layer_id] = remaining - 1
        return True

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

    @staticmethod
    def _normalize_global_hotness(hotness: np.ndarray) -> np.ndarray:
        flat_hotness = np.abs(hotness).reshape(-1)
        total_hotness = float(flat_hotness.sum())
        return (
            flat_hotness / total_hotness
            if total_hotness > 0
            else np.zeros_like(flat_hotness, dtype=np.float64)
        )

    def _global_hotness_delta(
        self,
        normalized: np.ndarray,
        previous: np.ndarray,
        total_slots: int,
    ) -> float:
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
        return float(np.sum(np.abs(normalized[candidates] - previous[candidates])))

    def _global_hotness_changed(self, hotness: np.ndarray, total_slots: int) -> bool:
        if self.min_hotness_delta <= 0:
            return True
        normalized = self._normalize_global_hotness(hotness)
        previous = self._last_global_hotness
        if previous is None or previous.shape != normalized.shape:
            self._last_global_hotness = normalized.copy()
            return True

        delta = self._global_hotness_delta(normalized, previous, total_slots)
        if delta < self.min_hotness_delta:
            return False
        self._last_global_hotness = normalized.copy()
        return True

    def _global_rebalance_is_cooling_down(
        self,
        hotness: np.ndarray,
        total_slots: int,
    ) -> bool:
        if self._global_rebalance_cooldown_remaining <= 0:
            return False
        normalized = self._normalize_global_hotness(hotness)
        previous = self._last_global_plan_hotness
        current_volume = float(np.sum(hotness))
        reference_volume = self._last_global_plan_volume
        volume_ratio = (
            current_volume / reference_volume
            if reference_volume is not None and reference_volume > 0
            else 1.0
        )
        volume_is_representative = (
            volume_ratio >= _GLOBAL_COOLDOWN_BREAK_MIN_VOLUME_RATIO
        )
        shift_delta = 0.0
        break_threshold = min(
            1.0,
            max(
                _GLOBAL_COOLDOWN_BREAK_MIN_DELTA,
                self.min_hotness_delta * _GLOBAL_COOLDOWN_BREAK_DELTA_FACTOR,
            ),
        )
        if previous is not None and previous.shape == normalized.shape:
            shift_delta = self._global_hotness_delta(
                normalized,
                previous,
                total_slots,
            )
        if volume_is_representative and shift_delta >= break_threshold:
            self._global_cooldown_shift_streak += 1
        else:
            self._global_cooldown_shift_streak = 0
        if self._global_cooldown_shift_streak >= _GLOBAL_COOLDOWN_BREAK_CONFIRMATIONS:
            logger.info(
                "[CRAFT-GLOBAL-COOLDOWN] break=true shift_delta=%.4f "
                "threshold=%.4f volume_ratio=%.4f",
                shift_delta,
                break_threshold,
                volume_ratio,
            )
            self._global_rebalance_cooldown_remaining = 0
            self._global_cooldown_shift_streak = 0
            return False
        self._global_rebalance_cooldown_remaining -= 1
        logger.info(
            "[CRAFT-GLOBAL-COOLDOWN] skipped=true remaining=%d "
            "shift_delta=%.4f threshold=%.4f volume_ratio=%.4f "
            "representative=%s shift_streak=%d/%d",
            self._global_rebalance_cooldown_remaining,
            shift_delta,
            break_threshold,
            volume_ratio,
            volume_is_representative,
            self._global_cooldown_shift_streak,
            _GLOBAL_COOLDOWN_BREAK_CONFIRMATIONS,
        )
        return True

    @staticmethod
    def _place_global_craft_plan(
        old_table,
        placements,
        extra_capacities,
        pool_start,
        pool_size,
    ):
        num_layers, num_ranks, _ = old_table.shape
        new_table = np.full_like(old_table, -1)
        pool_slots = list(range(pool_start, pool_start + pool_size))
        assigned_pool_slots = [[[] for _ in range(num_ranks)] for _ in range(num_layers)]

        for rank_id in range(num_ranks):
            old_owners = {}
            for slot_id in pool_slots:
                owners = [
                    layer_id
                    for layer_id in range(num_layers)
                    if old_table[layer_id, rank_id, slot_id] >= 0
                ]
                if len(owners) > 1:
                    raise ValueError(
                        "CRAFT global pool slot has multiple layer owners: "
                        f"rank={rank_id}, slot={slot_id}, owners={owners}."
                    )
                if owners:
                    old_owners[slot_id] = owners[0]

            free_slots = set(pool_slots)
            for layer_id in range(num_layers):
                quota = int(extra_capacities[layer_id, rank_id])
                old_layer_slots = [
                    slot_id
                    for slot_id, owner in old_owners.items()
                    if owner == layer_id
                ]
                target = set(placements[layer_id][rank_id])
                old_layer_slots.sort(
                    key=lambda slot_id: (
                        old_table[layer_id, rank_id, slot_id] not in target,
                        slot_id,
                    )
                )
                kept = old_layer_slots[:quota]
                assigned_pool_slots[layer_id][rank_id].extend(kept)
                free_slots.difference_update(kept)

            for layer_id in range(num_layers):
                quota = int(extra_capacities[layer_id, rank_id])
                missing = quota - len(assigned_pool_slots[layer_id][rank_id])
                selected = sorted(free_slots)[:missing]
                assigned_pool_slots[layer_id][rank_id].extend(selected)
                free_slots.difference_update(selected)

            if free_slots:
                raise ValueError(
                    "CRAFT interleaved capacities did not consume every global pool slot."
                )

        for layer_id in range(num_layers):
            for rank_id in range(num_ranks):
                allowed_slots = list(range(pool_start)) + assigned_pool_slots[layer_id][rank_id]
                target_experts = list(placements[layer_id][rank_id])
                target_set = set(target_experts)
                assigned = set()
                for slot_id in allowed_slots:
                    expert_id = int(old_table[layer_id, rank_id, slot_id])
                    if expert_id >= 0 and expert_id in target_set and expert_id not in assigned:
                        new_table[layer_id, rank_id, slot_id] = expert_id
                        assigned.add(expert_id)

                missing_experts = [
                    expert_id for expert_id in target_experts if expert_id not in assigned
                ]
                free_allowed_slots = [
                    slot_id
                    for slot_id in allowed_slots
                    if new_table[layer_id, rank_id, slot_id] < 0
                ]
                if len(missing_experts) != len(free_allowed_slots):
                    raise ValueError(
                        "CRAFT placement does not match the assigned layer/rank capacity."
                    )
                for slot_id, expert_id in zip(free_allowed_slots, missing_experts):
                    new_table[layer_id, rank_id, slot_id] = expert_id
        return new_table

    @staticmethod
    def _global_imbalance(table, hotness):
        table = np.asarray(table)
        hotness = np.asarray(hotness, dtype=np.float64)
        valid = table >= 0
        layer_ids = np.broadcast_to(
            np.arange(table.shape[0], dtype=np.int64)[:, None, None],
            table.shape,
        )
        rank_ids = np.broadcast_to(
            np.arange(table.shape[1], dtype=np.int64)[None, :, None],
            table.shape,
        )
        counts = np.zeros(hotness.shape, dtype=np.int64)
        np.add.at(
            counts,
            (layer_ids[valid], table[valid]),
            1,
        )
        per_copy = np.divide(
            hotness,
            counts,
            out=np.zeros_like(hotness, dtype=np.float64),
            where=counts > 0,
        )
        rank_loads = np.zeros(table.shape[:2], dtype=np.float64)
        np.add.at(
            rank_loads,
            (layer_ids[valid], rank_ids[valid]),
            per_copy[layer_ids[valid], table[valid]],
        )
        mean_loads = np.mean(rank_loads, axis=1)
        layer_weights = np.sum(hotness, axis=1)
        active = (mean_loads > 0) & (layer_weights > 0)
        if not np.any(active):
            return 0.0
        ratios = np.max(rank_loads[active], axis=1) / mean_loads[active]
        return float(
            np.sum(ratios * layer_weights[active])
            / np.sum(layer_weights[active])
        )

    @staticmethod
    def _global_critical_path_load(table, hotness):
        table = np.asarray(table)
        hotness = np.asarray(hotness, dtype=np.float64)
        valid = table >= 0
        if not np.any(valid):
            return 0.0
        layer_ids = np.broadcast_to(
            np.arange(table.shape[0], dtype=np.int64)[:, None, None],
            table.shape,
        )
        rank_ids = np.broadcast_to(
            np.arange(table.shape[1], dtype=np.int64)[None, :, None],
            table.shape,
        )
        counts = np.zeros(hotness.shape, dtype=np.int64)
        np.add.at(counts, (layer_ids[valid], table[valid]), 1)
        per_copy = np.divide(
            hotness,
            counts,
            out=np.zeros_like(hotness, dtype=np.float64),
            where=counts > 0,
        )
        rank_loads = np.zeros(table.shape[:2], dtype=np.float64)
        np.add.at(
            rank_loads,
            (layer_ids[valid], rank_ids[valid]),
            per_copy[layer_ids[valid], table[valid]],
        )
        return float(np.sum(np.max(rank_loads, axis=1)))

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
        if self._global_rebalance_is_cooling_down(
            hotness,
            pool_size * num_ranks,
        ):
            return False, None, current_expert_table.tolist()
        if not self._global_hotness_changed(hotness, pool_size * num_ranks):
            return False, None, current_expert_table.tolist()
        home = [
            [
                [
                    int(expert_id)
                    for expert_id in old_table[layer_id, rank_id, :pool_start]
                    if expert_id >= 0
                ]
                for rank_id in range(num_ranks)
            ]
            for layer_id in range(old_table.shape[0])
        ]
        planner_top_m = self.candidate_top_m
        if planner_top_m <= 0:
            planner_top_m = max(
                num_ranks,
                int(
                    np.ceil(
                        pool_size
                        * num_ranks
                        * self.candidate_factor
                        / old_table.shape[0]
                    )
                ),
            )
        layer_replicas, extra_capacities, placements = plan_craft_replication(
            hotness,
            pool_size * num_ranks,
            num_ranks,
            home_placements=home,
            candidate_top_m=planner_top_m,
            benefit_objective=self.global_planner_objective,
        )
        new_table = self._place_global_craft_plan(
            old_table,
            placements,
            extra_capacities,
            pool_start,
            pool_size,
        )
        score_fn = (
            self._global_critical_path_load
            if self.global_planner_objective == "critical_path"
            else self._global_imbalance
        )
        old_imbalance = score_fn(old_table, hotness)
        new_imbalance = score_fn(new_table, hotness)
        relative_improvement = (
            (old_imbalance - new_imbalance) / old_imbalance
            if old_imbalance > 0
            else 0.0
        )
        accepted = relative_improvement >= self.min_improvement
        logger.info(
            "[CRAFT-GLOBAL-BALANCE] slots=%d current=%.4f proposed=%.4f "
            "improvement=%.4f accepted=%s objective=%s planner_top_m=%d "
            "layer_replicas=%s",
            pool_size * num_ranks,
            old_imbalance,
            new_imbalance,
            relative_improvement,
            accepted,
            self.global_planner_objective,
            planner_top_m,
            ",".join(str(int(value)) for value in layer_replicas),
        )
        if not accepted:
            return False, None, current_expert_table.tolist()
        self._last_global_plan_hotness = self._normalize_global_hotness(hotness)
        self._last_global_plan_volume = float(np.sum(hotness))
        self._global_cooldown_shift_streak = 0
        changed = not np.array_equal(old_table, new_table)
        if changed:
            self._global_rebalance_cooldown_remaining = (
                self.global_rebalance_cooldown
            )
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
            if self._layer_cooldown_active(layer_id):
                continue
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
            if not np.array_equal(
                old_table[layer_id],
                new_table[layer_id],
            ):
                self._layer_rebalance_cooldown_remaining[layer_id] = (
                    self.layer_rebalance_cooldown
                )

        changed = not np.array_equal(old_table, new_table)
        return changed, None, new_table.tolist()
