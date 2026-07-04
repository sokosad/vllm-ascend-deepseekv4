# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import numpy as np
import torch

from .policy_abstract import DynamicConfig, EplbPolicy


class PoolBalanceEplb(EplbPolicy):
    """Dynamic EPLB policy for fixed-capacity CRAFT pool slots.

    The first `pool_start` local slots are fixed home experts. Only slots
    `[pool_start, E_local)` are replaced, so tensor shapes and main ownership
    stay stable while hot experts can rotate through the pool.
    """

    def __init__(self, config: DynamicConfig):
        self.config = config

    @staticmethod
    def _infer_pool_start(current_expert_table: torch.Tensor) -> int:
        _, num_ranks, _ = current_expert_table.shape
        num_logical_experts = int(torch.max(current_expert_table).item()) + 1
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
        num_experts = int(table.max()) + 1
        hotness = np.zeros((num_layers, num_experts), dtype=np.float64)
        for layer_id in range(num_layers):
            for rank_id in range(table.shape[1]):
                for slot_id in range(table.shape[2]):
                    expert_id = int(table[layer_id, rank_id, slot_id])
                    if expert_id >= 0:
                        hotness[layer_id, expert_id] += float(load[layer_id, rank_id, slot_id])
        return hotness

    @staticmethod
    def _desired_pool_sets(home, hotness, pool_size):
        num_ranks = len(home)
        if pool_size <= 0:
            return [set() for _ in range(num_ranks)]

        num_experts = len(hotness)
        extra_copies = [0] * num_experts
        for _ in range(pool_size * num_ranks):
            expert_id = max(
                range(num_experts),
                key=lambda i: hotness[i] / (1 + extra_copies[i])
                if extra_copies[i] < num_ranks - 1 else -1.0,
            )
            if hotness[expert_id] <= 0 or extra_copies[expert_id] >= num_ranks - 1:
                break
            extra_copies[expert_id] += 1

        per_copy = [hotness[e] / (1 + extra_copies[e]) for e in range(num_experts)]
        pool = [set() for _ in range(num_ranks)]
        card_experts = [set(card) for card in home]
        card_load = [sum(per_copy[e] for e in card) for card in home]
        items = sorted(
            (
                (expert_id, per_copy[expert_id])
                for expert_id in range(num_experts)
                for _ in range(extra_copies[expert_id])
            ),
            key=lambda x: -x[1],
        )

        for expert_id, weight in items:
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

        for rank_id in range(num_ranks):
            while len(pool[rank_id]) < pool_size:
                candidates = [e for e in range(num_experts) if e not in card_experts[rank_id]]
                if not candidates:
                    break
                expert_id = max(candidates, key=lambda e: per_copy[e])
                pool[rank_id].add(expert_id)
                card_experts[rank_id].add(expert_id)

        return pool

    @staticmethod
    def _stable_pool_slots(old_rank_slots, desired_pool, pool_start):
        new_rank_slots = old_rank_slots.copy()
        pool_slots = list(range(pool_start, len(old_rank_slots)))
        desired = set(desired_pool)

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
        pool_size = current_expert_table.shape[2] - pool_start
        if pool_size <= 0:
            return False, None, current_expert_table.tolist()

        hotness = self._expert_hotness(current_expert_table, expert_workload)
        old_table = current_expert_table.detach().cpu().numpy()
        new_table = old_table.copy()

        for layer_id in range(old_table.shape[0]):
            home = [list(map(int, old_table[layer_id, rank_id, :pool_start])) for rank_id in range(old_table.shape[1])]
            desired_pool = self._desired_pool_sets(home, hotness[layer_id], pool_size)
            for rank_id in range(old_table.shape[1]):
                new_table[layer_id, rank_id] = self._stable_pool_slots(
                    old_table[layer_id, rank_id],
                    desired_pool[rank_id],
                    pool_start,
                )

        changed = not np.array_equal(old_table, new_table)
        return changed, None, new_table.tolist()
