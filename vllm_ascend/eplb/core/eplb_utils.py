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
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove eplb utils.
import json
import os.path
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.layer import determine_expert_map


def expert_file_to_tensor(expert_map_path, layer_id):
    with open(expert_map_path) as f:
        data = json.load(f)
    physical_count = 0
    device_data = []
    if layer_id > data["moe_layer_count"]:
        raise ValueError("Invalid EPLB Table")
    if layer_id == data["moe_layer_count"]:
        logger.warning("Init expert map of mtp/eagle when using sample.")
        return None, None
    for device in data["layer_list"][layer_id]["device_list"]:
        experts = device["device_expert"]
        physical_count += len(experts)
        device_data.append(torch.tensor(experts, dtype=torch.int32))
    lengths = {placement.numel() for placement in device_data}
    global_placement = torch.stack(device_data) if len(lengths) == 1 else device_data
    return global_placement, physical_count


def expert_file_pool_metadata(expert_map_path, layer_id):
    if not expert_map_path:
        return False, None, None
    with open(expert_map_path) as f:
        data = json.load(f)
    if layer_id >= data["moe_layer_count"]:
        return False, None, None
    layer = data["layer_list"][layer_id]
    file_pool_mode = data.get("pool_mode", False) or any(
        item.get("pool_size", 0) for item in data.get("layer_list", [])
    )
    pool_mode = bool(file_pool_mode or layer.get("pool_size", 0))
    pool_start = layer.get("pool_start", data.get("pool_start"))
    pool_size = layer.get("pool_size")
    return pool_mode, pool_start, pool_size


def _coerce_pool_size(pool_size) -> int:
    return max(0, int(pool_size or 0))


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def get_configured_craft_pool_size(eplb_config, layer_id: int | None = None) -> int:
    layer_sizes = getattr(eplb_config, "craft_pool_layer_sizes", None)
    if layer_sizes is None:
        return _coerce_pool_size(getattr(eplb_config, "craft_pool_size", 0))
    if isinstance(layer_sizes, dict):
        if layer_id is None:
            return max((_coerce_pool_size(size) for size in layer_sizes.values()), default=0)
        return _coerce_pool_size(layer_sizes.get(layer_id, layer_sizes.get(str(layer_id), 0)))
    if layer_id is None:
        return max((_coerce_pool_size(size) for size in layer_sizes), default=0)
    if layer_id >= len(layer_sizes):
        return 0
    return _coerce_pool_size(layer_sizes[layer_id])


@lru_cache(maxsize=16)
def expert_file_has_pool_mode(expert_map_path):
    if not expert_map_path:
        return False
    if not (os.path.exists(expert_map_path) and os.access(expert_map_path, os.R_OK)):
        return False
    with open(expert_map_path) as f:
        data = json.load(f)
    return bool(data.get("pool_mode", False) or any(layer.get("pool_size", 0) for layer in data.get("layer_list", [])))


def generate_global_placement(n_expert, ep_size, n_redundant):
    all_experts = np.arange(n_expert)
    groups = np.array_split(all_experts, ep_size)
    for i in range(n_redundant):
        j = i % ep_size + 1
        if len(groups[-j]) == 0:
            groups[-j] = np.append(groups[-j], j)
        else:
            groups[-j] = np.append(groups[-j], (groups[-j][-1] + 1) % n_expert)
    return torch.tensor(groups, dtype=torch.int32)


def generate_pool_placement(n_expert, ep_size, pool_size):
    if pool_size <= 0:
        return generate_global_placement(n_expert, ep_size, 0)
    if n_expert % ep_size != 0:
        raise ValueError("CRAFT pool requires num_experts to be divisible by ep_size.")

    main_size = n_expert // ep_size
    if pool_size > n_expert - main_size:
        raise ValueError(
            "CRAFT pool size is too large: "
            f"pool_size={pool_size}, max_per_rank={n_expert - main_size}."
        )

    groups = []
    for rankid in range(ep_size):
        home = list(range(rankid * main_size, (rankid + 1) * main_size))
        home_set = set(home)
        pool = []
        candidate = ((rankid + 1) * main_size) % n_expert
        while len(pool) < pool_size:
            if candidate not in home_set:
                pool.append(candidate)
            candidate = (candidate + 1) % n_expert
        groups.append(home + pool)
    return torch.tensor(groups, dtype=torch.int32)


def init_eplb_config(eplb_config, layer_id, moe_config):
    expert_map_path = eplb_config.expert_map_path
    n_experts = getattr(moe_config, "num_logical_experts", moe_config.num_experts)
    ep_size = moe_config.ep_size
    global_placement = None
    craft_pool_size = get_configured_craft_pool_size(eplb_config, layer_id)
    pool_mode = get_configured_craft_pool_size(eplb_config) > 0 or expert_file_has_pool_mode(expert_map_path)
    metro_routing = _coerce_bool(getattr(eplb_config, "metro_routing", False))
    eplb_enable = (
        eplb_config.dynamic_eplb
        or pool_mode
        or (metro_routing and eplb_config.num_redundant_experts > 0)
    )
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0

    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
        return None, None, None, n_redundant

    if expert_map_path:
        if not (os.path.exists(expert_map_path) and os.access(expert_map_path, os.R_OK)):
            raise ValueError("Invalid EPLB path")
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(expert_map_path, layer_id)
        if physical_count is not None:
            n_redundant = physical_count - n_experts
            if craft_pool_size > 0 and physical_count != n_experts + craft_pool_size * ep_size:
                raise ValueError(
                    "CRAFT pool expert_map conflicts with craft_pool_size: "
                    f"physical_count={physical_count}, expected={n_experts + craft_pool_size * ep_size}."
                )
            if not moe_config.supports_eplb:
                raise ValueError("Eplb supports only w8a8_dynamic quantization.")
        else:
            eplb_enable = False
    elif not eplb_enable:
        _, expert_map, _ = determine_expert_map(ep_size, moe_config.ep_rank, n_experts)
        return None, expert_map, None, 0

    if global_placement is None:
        if craft_pool_size > 0:
            global_placement = generate_pool_placement(n_experts, ep_size, craft_pool_size)
            n_redundant = craft_pool_size * ep_size
        else:
            global_placement = generate_global_placement(n_experts, ep_size, n_redundant)

    global_expert_map = []
    for rankid in range(ep_size):
        expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        expert_map[local_placement] = torch.arange(local_placement.shape[0], dtype=torch.int32)
        global_expert_map.append(expert_map)
        if rankid == moe_config.ep_rank:
            local_expert_map = expert_map
    if eplb_enable:
        if pool_mode or metro_routing:
            log2phy = generate_pool_log2phy_map(global_expert_map).npu()
        else:
            log2phy = generate_log2phy_map(global_expert_map, moe_config.ep_rank).npu()
    else:
        log2phy = None

    return torch.stack(global_expert_map), local_expert_map, log2phy, n_redundant


def generate_log2phy_map(global_expert_map, ep_rank):
    log2phy_map = defaultdict(list)
    valid_count = torch.sum(global_expert_map[0] != -1)
    for rankid, map_per_rank in enumerate(global_expert_map):
        for idx, val in enumerate(map_per_rank):
            val = val.item()
            if val != -1:
                log2phy_map[idx].append(val + rankid * valid_count)

    for key in log2phy_map:
        num_of_duplications = len(log2phy_map[key])
        log2phy_map[key] = log2phy_map[key][ep_rank % num_of_duplications]

    log2phy_map = torch.scatter(
        torch.zeros(len(log2phy_map), dtype=torch.int32),
        0,
        torch.tensor(list(log2phy_map), dtype=torch.int64),
        torch.tensor(list(log2phy_map.values()), dtype=torch.int32),
    )

    return log2phy_map


def generate_pool_log2phy_map(global_expert_map):
    """Return logical expert id -> candidate global physical slots.

    Pool mode has multiple physical copies for some logical experts. The
    runtime picks one candidate per token before dispatch so only one rank
    computes each routed expert.
    """
    if not torch.is_tensor(global_expert_map):
        global_expert_map = torch.stack(global_expert_map)

    ep_size, num_experts = global_expert_map.shape
    local_slots = int(torch.max(global_expert_map).item()) + 1
    max_copies = ep_size
    log2phy = torch.full((num_experts, max_copies), -1, dtype=torch.int32)
    copy_index = torch.zeros(num_experts, dtype=torch.int64)

    for rankid in range(ep_size):
        for expert_id, local_slot in enumerate(global_expert_map[rankid]):
            local_slot = int(local_slot.item())
            if local_slot == -1:
                continue
            idx = int(copy_index[expert_id].item())
            log2phy[expert_id, idx] = rankid * local_slots + local_slot
            copy_index[expert_id] += 1
    if torch.any(copy_index == 0):
        raise ValueError("Pool log2phy contains a logical expert without a physical replica.")
    return log2phy


def generate_local_physical_expert_mask(local_num_experts, ep_size, ep_rank):
    """Mask global physical slots that are local to this rank."""
    num_global_physical_experts = local_num_experts * ep_size
    expert_map = torch.full((num_global_physical_experts,), -1, dtype=torch.int32)
    start = ep_rank * local_num_experts
    expert_map[start:start + local_num_experts] = torch.arange(local_num_experts, dtype=torch.int32)
    return expert_map
