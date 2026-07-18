#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from multiprocessing import Process, Queue
from queue import Full
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.eplb.core.eplb_utils import (
    generate_craft_rank_route_map,
    generate_craft_route_map,
    generate_log2phy_map,
    generate_pool_log2phy_map,
)
from vllm_ascend.eplb.core.policy.policy_factory import DynamicConfig, PolicyFactory


CRAFT_POOL_POLICY_CONFIG_FIELDS = (
    "craft_global_pool_size",
    "craft_pool_top_m",
    "craft_pool_top_m_factor",
    "craft_pool_min_hotness_delta",
    "craft_pool_min_improvement",
    "craft_global_rebalance_cooldown",
    "craft_layer_rebalance_cooldown",
)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class EplbWorker:
    def __init__(self, shared_dict, policy_type, enable_d2d: bool = True, eplb_config=None):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type, self._build_policy_config(policy_type, eplb_config))
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.rank_id = dist.get_rank()
        self.multi_stage = policy_type == 3
        self.metro_routing = _coerce_bool(getattr(eplb_config, "metro_routing", False))
        self.craft_rank_sharded_routing = _coerce_bool(
            getattr(eplb_config, "craft_rank_sharded_routing", False)
        )
        self.craft_global_pool_size = max(
            0, int(getattr(eplb_config, "craft_global_pool_size", 0) or 0)
        )
        self.full_rank_plan = policy_type == 4 or self.metro_routing
        self.expert_heat_collection_interval = max(
            1, int(getattr(eplb_config, "expert_heat_collection_interval", 1))
        )
        configured_payback_steps = int(
            getattr(eplb_config, "craft_pool_max_payback_steps", 0)
        )
        self.craft_max_payback_steps = (
            configured_payback_steps
            if configured_payback_steps > 0
            else self.expert_heat_collection_interval
        )
        self.craft_migration_cost_ratio = max(
            0.0, float(getattr(eplb_config, "craft_pool_migration_cost_ratio", 1.0))
        )

    @staticmethod
    def _build_policy_config(policy_type, eplb_config=None):
        policy_config = DynamicConfig()
        if policy_type != 4 or eplb_config is None:
            return policy_config
        for field in CRAFT_POOL_POLICY_CONFIG_FIELDS:
            if hasattr(eplb_config, field):
                setattr(policy_config, field, getattr(eplb_config, field))
        return policy_config

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D
        # Get initial expert_map
        torch.set_num_threads(1)
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self._max_local_experts(self.old_expert_maps)
                if self.craft_global_pool_size > 0:
                    num_ranks = int(self.old_expert_maps.shape[1])
                    num_logical_experts = int(self.old_expert_maps.shape[2])
                    if self.craft_global_pool_size % num_ranks != 0:
                        raise ValueError(
                            "craft_global_pool_size must be divisible by the EPLB rank count."
                        )
                    self.num_local_experts = (
                        num_logical_experts // num_ranks
                        + self.craft_global_pool_size // num_ranks
                    )
                    self.num_local_experts_main = (
                        num_logical_experts // num_ranks
                    )
            else:
                raise ValueError("Failed to get expert_maps from shared_dict.")

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            return

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        if self.policy_type == 4 and self.rank_id == 0:
            self._log_craft_pool_utilization(old_placement, load_info)
        _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)

        hotness = None
        if self.rank_id == 0 or self.policy_type == 4:
            if self.multi_stage:
                hotness = self._calculate_hotness(old_placement, load_info.sum(0))
            else:
                hotness = self._calculate_hotness(old_placement, load_info)
        if self.rank_id == 0:
            current_mean, current_max = self._compute_imbalance(old_placement, hotness)
            update_mean, update_max = self._compute_imbalance(new_placement, hotness)
            logger.info(
                "[Expert Hotness] Current: mean={:.3f}, max={:.3f}, Updated: mean={:.3f}, max={:.3f}".format(
                    current_mean, current_max, update_mean, update_max
                )
            )

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        self.check_expert_placement(old_placement, new_placement)
        if self.policy_type == 4 and hotness is not None:
            if getattr(self, "craft_global_pool_size", 0) > 0:
                new_placement = self._apply_global_craft_migration_cost_gate(
                    old_placement, new_placement, hotness
                )
            else:
                new_placement = self._apply_craft_migration_cost_gate(
                    old_placement, new_placement, hotness
                )
        new_expert_maps = self.local2global(new_placement)
        changed_layers = None
        if self.policy_type == 4:
            changed_layers = [
                not torch.equal(new_expert_maps[layer_id], self.old_expert_maps[layer_id])
                for layer_id in range(new_expert_maps.shape[0])
            ]
        if getattr(self, "craft_global_pool_size", 0) > 0:
            update_info = self.compose_expert_update_info_greedy(
                new_expert_maps,
                self.old_expert_maps,
            )
            # Validate every migration source before publishing the new map.
            update_info = list(update_info)
            packed_update_info = self.pack_update_info(
                update_info,
                changed_layers=changed_layers,
            )
            if changed_layers is None or any(changed_layers):
                self.update_expert_map(new_expert_maps)
            self.old_expert_maps = new_expert_maps
            logger.debug("EPLB Process compute complete")
            return packed_update_info

        if changed_layers is None or any(changed_layers):
            self.update_expert_map(new_expert_maps)
        update_info = self.compose_expert_update_info_greedy(
            new_expert_maps,
            self.old_expert_maps,
        )
        self.old_expert_maps = new_expert_maps
        logger.debug("EPLB Process compute complete")
        return self.pack_update_info(
            update_info,
            changed_layers=changed_layers,
        )

    @staticmethod
    def _craft_pool_utilization(deployment, load_info):
        deployment = (
            deployment.detach().cpu().numpy()
            if torch.is_tensor(deployment)
            else np.asarray(deployment)
        )
        load_info = (
            load_info.detach().cpu().numpy()
            if torch.is_tensor(load_info)
            else np.asarray(load_info)
        )
        if deployment.ndim != 3 or load_info.shape != deployment.shape:
            return None

        valid = deployment >= 0
        if not valid.any() or deployment.shape[1] <= 0:
            return None
        num_logical_experts = int(deployment[valid].max()) + 1
        pool_start = num_logical_experts // deployment.shape[1]
        if pool_start >= deployment.shape[2]:
            return None

        pool_valid = valid[:, :, pool_start:]
        total_slots = int(pool_valid.sum())
        if total_slots == 0:
            return None
        pool_load = np.where(pool_valid, load_info[:, :, pool_start:], 0)
        active = pool_valid & (load_info[:, :, pool_start:] > 0)
        layer_tokens = pool_load.sum(axis=(1, 2))
        layer_active = active.sum(axis=(1, 2))
        total_tokens = float(np.where(valid, load_info, 0).sum())
        pool_tokens = float(pool_load.sum())
        top_layers = np.argsort(-layer_tokens)[: min(5, len(layer_tokens))]
        slot_tokens = [
            (
                int(layer_id),
                int(rank_id),
                int(pool_offset),
                int(deployment[layer_id, rank_id, pool_start + pool_offset]),
                int(load_info[layer_id, rank_id, pool_start + pool_offset]),
            )
            for layer_id, rank_id, pool_offset in np.argwhere(pool_valid)
        ]
        return {
            "total_slots": total_slots,
            "active_slots": int(active.sum()),
            "active_ratio": float(active.sum()) / total_slots,
            "pool_tokens": pool_tokens,
            "pool_token_share": pool_tokens / total_tokens if total_tokens > 0 else 0.0,
            "zero_hit_layers": int(np.count_nonzero(layer_active == 0)),
            "top_layers": [
                (int(layer_id), int(layer_tokens[layer_id]), int(layer_active[layer_id]))
                for layer_id in top_layers
            ],
            "slot_tokens": slot_tokens,
        }

    def _log_craft_pool_utilization(self, deployment, load_info):
        stats = self._craft_pool_utilization(deployment, load_info)
        if stats is None:
            return
        top_layers = ",".join(
            f"{layer_id}:{tokens}/{active}"
            for layer_id, tokens, active in stats["top_layers"]
        )
        logger.info(
            "[CRAFT-SLOT] active=%d/%d ratio=%.3f pool_tokens=%.0f "
            "pool_share=%.4f zero_hit_layers=%d top_layers=%s",
            stats["active_slots"],
            stats["total_slots"],
            stats["active_ratio"],
            stats["pool_tokens"],
            stats["pool_token_share"],
            stats["zero_hit_layers"],
            top_layers,
        )
        slots_by_rank: dict[int, list[str]] = {}
        for layer_id, rank_id, pool_offset, expert_id, tokens in stats["slot_tokens"]:
            slots_by_rank.setdefault(rank_id, []).append(
                f"{layer_id}:{pool_offset}:{expert_id}:{tokens}"
            )
        for rank_id, slots in sorted(slots_by_rank.items()):
            logger.info(
                "[CRAFT-SLOT-TOKENS] rank=%d slots=%s",
                rank_id,
                ",".join(slots),
            )

    @staticmethod
    def _rank_loads(deployment, hotness):
        deployment = np.asarray(deployment)
        hotness = np.asarray(hotness, dtype=np.float64)
        valid = deployment >= 0
        if not valid.any() or hotness.size == 0:
            return np.zeros(deployment.shape[0], dtype=np.float64)
        counts = np.bincount(deployment[valid].reshape(-1), minlength=hotness.shape[0])
        unit_hotness = np.divide(
            hotness,
            counts,
            out=np.zeros_like(hotness, dtype=np.float64),
            where=counts != 0,
        )
        return np.asarray(
            [unit_hotness[rank[rank >= 0]].sum() for rank in deployment],
            dtype=np.float64,
        )

    def _apply_craft_migration_cost_gate(self, old_placement, new_placement, hotness):
        metadata = self.shared_dict.get("craft_expert_cost_metadata", None)
        if not metadata:
            logger.warning_once(
                "CRAFT expert cost metadata is unavailable; migration payback gating is disabled."
            )
            return new_placement

        old_table = np.asarray(old_placement)
        new_table = np.asarray(new_placement).copy()
        for layer_id in range(min(len(metadata), old_table.shape[0])):
            changed_slots = int(np.count_nonzero(old_table[layer_id] != new_table[layer_id]))
            if changed_slots == 0:
                continue
            transfer_bytes = int(metadata[layer_id].get("transfer_bytes", 0))
            compute_bytes = int(metadata[layer_id].get("compute_bytes", 0))
            if transfer_bytes <= 0 or compute_bytes <= 0:
                continue

            old_load = self._rank_loads(old_table[layer_id], hotness[layer_id])
            new_load = self._rank_loads(new_table[layer_id], hotness[layer_id])
            load_delta = max(0.0, float(old_load.max() - new_load.max()))
            saved_compute_bytes_per_step = (
                load_delta * compute_bytes / self.expert_heat_collection_interval
            )
            migration_bytes = changed_slots * transfer_bytes
            if saved_compute_bytes_per_step <= 0:
                payback_steps = float("inf")
            else:
                payback_steps = (
                    migration_bytes * self.craft_migration_cost_ratio
                    / saved_compute_bytes_per_step
                )

            accepted = payback_steps <= self.craft_max_payback_steps
            logger.info(
                "[CRAFT-COST] layer=%d changed_slots=%d migration_mb=%.2f "
                "load_delta=%.3f payback_steps=%.1f limit=%d accepted=%s",
                layer_id,
                changed_slots,
                migration_bytes / 1e6,
                load_delta,
                payback_steps,
                self.craft_max_payback_steps,
                accepted,
            )
            if not accepted:
                new_table[layer_id] = old_table[layer_id]

        return torch.as_tensor(
            new_table, dtype=new_placement.dtype, device=new_placement.device
        )

    def _apply_global_craft_migration_cost_gate(
        self, old_placement, new_placement, hotness
    ):
        metadata = self.shared_dict.get("craft_expert_cost_metadata", None)
        if not metadata:
            logger.warning_once(
                "CRAFT expert cost metadata is unavailable; global migration "
                "payback gating is disabled."
            )
            return new_placement

        old_table = np.asarray(old_placement)
        new_table = np.asarray(new_placement)
        if np.array_equal(old_table, new_table):
            return new_placement

        pool_start = getattr(self, "num_local_experts_main", None)
        if pool_start is None:
            num_ranks = int(old_table.shape[1])
            pool_size = self.craft_global_pool_size // num_ranks
            pool_start = old_table.shape[2] - pool_size

        migration_bytes = 0
        changed_pool_slots = 0
        for rank_id in range(old_table.shape[1]):
            for slot_id in range(pool_start, old_table.shape[2]):
                old_owners = np.flatnonzero(old_table[:, rank_id, slot_id] >= 0)
                new_owners = np.flatnonzero(new_table[:, rank_id, slot_id] >= 0)
                old_item = (
                    (
                        int(old_owners[0]),
                        int(old_table[old_owners[0], rank_id, slot_id]),
                    )
                    if old_owners.size
                    else None
                )
                new_item = (
                    (
                        int(new_owners[0]),
                        int(new_table[new_owners[0], rank_id, slot_id]),
                    )
                    if new_owners.size
                    else None
                )
                if old_item == new_item:
                    continue
                changed_pool_slots += 1
                if new_item is not None and new_item[0] < len(metadata):
                    migration_bytes += int(
                        metadata[new_item[0]].get("transfer_bytes", 0)
                    )

        saved_compute_bytes_per_step = 0.0
        load_delta = 0.0
        for layer_id in range(min(len(metadata), old_table.shape[0])):
            compute_bytes = int(metadata[layer_id].get("compute_bytes", 0))
            if compute_bytes <= 0:
                continue
            old_load = self._rank_loads(old_table[layer_id], hotness[layer_id])
            new_load = self._rank_loads(new_table[layer_id], hotness[layer_id])
            layer_load_delta = float(old_load.max() - new_load.max())
            load_delta += layer_load_delta
            saved_compute_bytes_per_step += (
                layer_load_delta
                * compute_bytes
                / self.expert_heat_collection_interval
            )

        if saved_compute_bytes_per_step <= 0:
            payback_steps = float("inf")
        else:
            payback_steps = (
                migration_bytes
                * self.craft_migration_cost_ratio
                / saved_compute_bytes_per_step
            )
        accepted = payback_steps <= self.craft_max_payback_steps
        logger.info(
            "[CRAFT-GLOBAL-COST] changed_pool_slots=%d migration_mb=%.2f "
            "load_delta=%.3f saved_compute_mb_per_step=%.3f "
            "payback_steps=%.1f limit=%d accepted=%s",
            changed_pool_slots,
            migration_bytes / 1e6,
            load_delta,
            saved_compute_bytes_per_step / 1e6,
            payback_steps,
            self.craft_max_payback_steps,
            accepted,
        )
        return new_placement if accepted else old_placement

    def check_expert_placement(self, old_placement, new_placement):
        if self.policy_type != 4:
            self._check_expert_placement_legacy(old_placement, new_placement)
            return

        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            old_valid_experts = torch.unique(old_placement[layer_id][old_placement[layer_id] >= 0])
            new_valid_experts = torch.unique(new_placement[layer_id][new_placement[layer_id] >= 0])
            if not torch.all(torch.isin(old_valid_experts, new_valid_experts)):
                logger.error(f"There exists expert not placed on any rank in layer {layer_id}")
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]
                new_valid_slots = new_placement_check >= 0
                new_valid_experts = new_placement_check[new_valid_slots]

                # check if same logical experts are placed on the same NPU
                if new_valid_experts.numel() != torch.unique(new_valid_experts).numel():
                    logger.error(
                        "Replicated experts are placed on the same NPU; expert placement on "
                        f"layer {layer_id}, rank {rank_id} is invalid"
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # Experts that remain on a rank must keep their local slots so
                # only pool replacement requires weight movement.
                invalid_movement = False
                for slot_id in torch.where(new_valid_slots)[0]:
                    expert_id = new_placement_check[slot_id]
                    old_slots = torch.where(old_placement_check == expert_id)[0]
                    if old_slots.numel() > 0 and old_slots[0].item() != slot_id.item():
                        invalid_movement = True
                        break
                if invalid_movement:
                    logger.error(
                        "There exists expert movement inside NPU; expert placement on "
                        f"layer {layer_id}, rank {rank_id} is invalid"
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

        if getattr(self, "craft_global_pool_size", 0) > 0:
            pool_start = getattr(self, "num_local_experts_main", None)
            if pool_start is None:
                pool_size = self.craft_global_pool_size // num_ranks
                pool_start = new_placement.shape[2] - pool_size
            pool_owners = torch.sum(new_placement[:, :, pool_start:] >= 0, dim=0)
            if torch.any(pool_owners > 1):
                logger.error(
                    "A CRAFT global pool slot cannot be assigned to multiple layers"
                )
                new_placement.copy_(old_placement)

    @staticmethod
    def _check_expert_placement_legacy(old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]
        for layer_id in range(num_layers):
            if torch.unique(new_placement[layer_id]).numel() < torch.unique(old_placement[layer_id]).numel():
                logger.error(f"There exists expert not placed on any rank in layer {layer_id}")
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    logger.error(
                        "Replicated experts are placed on the same NPU; expert placement on "
                        f"layer {layer_id}, rank {rank_id} is invalid"
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move], old_placement_check[expert_not_move]):
                    logger.error(
                        "There exists expert movement inside NPU; expert placement on "
                        f"layer {layer_id}, rank {rank_id} is invalid"
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    updated_expert_maps_this_layer,
                    layer_id,
                )
                continue

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where(
                (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
            )

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where(
                (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
            )

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if getattr(self, "craft_global_pool_size", 0) > 0:
                    current_owners = torch.where(
                        current_expert_maps_this_layer[:, expert_id] >= 0
                    )[0]
                    stable_owners = current_owners[
                        updated_expert_maps_this_layer[current_owners, expert_id]
                        >= 0
                    ]
                    candidate_src_rank_indices = (
                        stable_owners
                        if stable_owners.numel() > 0
                        else current_owners
                    )
                    if candidate_src_rank_indices.numel() == 0:
                        raise ValueError(
                            "Cannot migrate a CRAFT global pool expert without "
                            f"a current owner: expert_id={expert_id}."
                        )
                elif not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                # TODO: improve selection criterion of NPU sending expert_id,
                # considering intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )

    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        Compute `new_map` by calling the `rebalance_experts` method of the policy instance.
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):
        self.shared_dict["expert_maps"] = expert_maps

    @staticmethod
    def _max_local_experts(expert_maps: torch.Tensor) -> int:
        valid_slots = expert_maps[expert_maps >= 0]
        if valid_slots.numel() == 0:
            return 0
        return int(torch.max(valid_slots).item()) + 1

    def global2local(self, placement: torch.Tensor, E_local: int) -> torch.Tensor:
        L, G, _ = placement.shape
        device = placement.device
        E_local = int(E_local)

        pt_local = torch.full((L, G, E_local), fill_value=-1, dtype=torch.long, device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx].long()

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:
        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global), fill_value=-1, dtype=torch.long, device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def pack_update_info(self, update_info_generator, changed_layers=None):
        """
        Pack a list of update info records for efficient IPC.

        Dynamic EPLB workers run independently on every rank. To keep D2D
        P2P send/recv pairs consistent, each worker packs the full plan for
        all ranks; the runtime broadcasts rank 0's full plan and each rank
        selects its local slice.
        """
        if not self.full_rank_plan:
            send_all = []
            recv_all = []
            maps = []
            log2phy_all = []
            layer_ids = []
            for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
                send_all.append(send_info.get(self.rank_id, []))
                recv_all.append(recv_info.get(self.rank_id, []))
                maps.append(new_expert_map[self.rank_id].numpy().tolist())
                log2phy_map = generate_log2phy_map(new_expert_map, self.rank_id)
                log2phy_all.append(log2phy_map.numpy().tolist())
                layer_ids.append(layer_id)
            return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

        packed_update_info = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
            if changed_layers is not None and not changed_layers[layer_id]:
                packed_update_info.append({"noop": True, "layer_id": layer_id})
                continue
            num_ranks = int(new_expert_map.shape[0])
            if self.policy_type == 4:
                if getattr(self, "craft_rank_sharded_routing", False):
                    log2phy_all = [
                        generate_craft_rank_route_map(
                            new_expert_map,
                            rank_id,
                            local_slots=getattr(self, "num_local_experts", None),
                        ).numpy().tolist()
                        for rank_id in range(num_ranks)
                    ]
                else:
                    shared_log2phy_map = generate_craft_route_map(
                        new_expert_map,
                        local_slots=getattr(self, "num_local_experts", None),
                    ).numpy().tolist()
                    log2phy_all = [shared_log2phy_map for _ in range(num_ranks)]
            elif self.full_rank_plan:
                shared_log2phy_map = generate_pool_log2phy_map(new_expert_map).numpy().tolist()
                log2phy_all = [shared_log2phy_map for _ in range(num_ranks)]
            else:
                log2phy_all = [
                    generate_log2phy_map(new_expert_map, rank_id).numpy().tolist()
                    for rank_id in range(num_ranks)
                ]

            packed_update_info.append(
                {
                    "send_all": [send_info.get(rank_id, []) for rank_id in range(num_ranks)],
                    "recv_all": [recv_info.get(rank_id, []) for rank_id in range(num_ranks)],
                    "maps_all": [new_expert_map[rank_id].numpy().tolist() for rank_id in range(num_ranks)],
                    "log2phy_all": log2phy_all,
                    "layer_id": layer_id,
                }
            )

        return packed_update_info

    @staticmethod
    def _compute_imbalance(deployment_all_layer, hotness_all_layer: np.ndarray):
        imbalance_list = []
        deployment_all_layer = (
            deployment_all_layer.detach().cpu().numpy()
            if torch.is_tensor(deployment_all_layer)
            else np.asarray(deployment_all_layer)
        )
        for deployment, hotness in zip(deployment_all_layer, hotness_all_layer):
            valid = deployment >= 0
            if not valid.any() or hotness.shape[0] == 0:
                continue
            counts = np.bincount(deployment[valid].reshape(-1), minlength=hotness.shape[0])

            unit_hotness = np.divide(hotness, counts, out=np.zeros_like(hotness, dtype=float), where=counts != 0)

            stage_load = []
            for rank_deployment in deployment:
                valid_rank_experts = rank_deployment[rank_deployment >= 0]
                stage_load.append(unit_hotness[valid_rank_experts].sum() if valid_rank_experts.size else 0.0)
            stage_load = np.asarray(stage_load)
            mean_load = stage_load.mean()
            stage_par = stage_load.max() / mean_load if mean_load > 0 else 0.0
            imbalance_list.append(stage_par)

        if not imbalance_list:
            return 0.0, 0.0
        max_val = max(imbalance_list)
        mean_val = sum(imbalance_list) / len(imbalance_list)
        return mean_val, max_val

    @staticmethod
    def _calculate_hotness(deployment_all_layer, moe_load_all_layer):
        hotnesses = []
        deployment_all_layer = (
            deployment_all_layer.detach().cpu().numpy()
            if torch.is_tensor(deployment_all_layer)
            else np.asarray(deployment_all_layer)
        )
        moe_load_all_layer = (
            moe_load_all_layer.detach().cpu().numpy()
            if torch.is_tensor(moe_load_all_layer)
            else np.asarray(moe_load_all_layer)
        )
        valid = deployment_all_layer >= 0
        num_of_expert = int(deployment_all_layer[valid].max()) + 1 if valid.any() else 0
        for deployment, rank_load in zip(deployment_all_layer, moe_load_all_layer):
            hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
            valid_deployment = deployment >= 0
            deployment_flat = deployment[valid_deployment].ravel()
            rank_load_flat = rank_load[valid_deployment].ravel()
            np.add.at(hotness, deployment_flat, rank_load_flat)
            hotnesses.append(hotness)

        return np.array(hotnesses)


class EplbProcess:
    def __init__(self, shared_dict, policy_type: int = 0, enable_d2d: bool = True, eplb_config=None):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q: Queue[Any] = Queue()
        self.block_update_q: Queue[Any] = Queue(maxsize=1)

        # Create EplbWorker instance
        self.worker = EplbWorker(self.shared_dict, self.policy_type, self.enable_d2d, eplb_config)

    @staticmethod
    def _publish_update(block_update_q, packed_update_info):
        while True:
            try:
                block_update_q.put(packed_update_info, timeout=1.0)
                return
            except Full:
                continue

    def worker_process(self, planner_q, block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up,
        call do_update, then notify main process update is complete.
        """
        if self.policy_type == 3:
            from vllm_ascend.eplb.core.policy.policy_flashlb import warm_up

            warm_up()
        while True:
            try:
                planner_q.get()
            except (EOFError, OSError):
                logger.warning("EPLB planner queue closed; stopping subprocess")
                break
            try:
                packed_update_info = self.worker.do_update()
            except Exception as e:
                if self.policy_type != 4:
                    logger.warning(
                        f"[EPLB subprocess exiting due to error: {e}]",
                        exc_info=True,
                    )
                    break
                logger.warning(
                    f"[EPLB subprocess update failed; publishing no-op: {e}]",
                    exc_info=True,
                )
                packed_update_info = None
            self._publish_update(block_update_q, packed_update_info)

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)

        proc.start()
        return proc
