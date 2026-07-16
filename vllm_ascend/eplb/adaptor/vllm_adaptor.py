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
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this adaptor.
import json
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.eplb.core.eplb_utils import generate_craft_route_map


class VllmEplbAdaptor:
    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_dense_layers = getattr(self.model.config, "first_k_dense_replace", 0)
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers

        self.expert_map_per_layer_cpu = dict()  # copy of expert map on CPU to avoid device synchronize frequently

        self.num_local_experts_per_layer = self._get_num_local_experts_per_layer()
        self.num_local_experts = max(self.num_local_experts_per_layer.values(), default=0)
        self.craft_pool_enabled = any(
            getattr(self.model.model.layers[layer_idx].mlp.experts, "craft_pool_enabled", False)
            for layer_idx in self.num_local_experts_per_layer
        )
        self.craft_global_pool_enabled = any(
            getattr(self.model.model.layers[layer_idx].mlp.experts, "craft_global_pool_enabled", False)
            for layer_idx in self.num_local_experts_per_layer
        )
        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()

        num_buffer_tensor = (
            max(
                (
                    self.model.model.layers[layer_idx].mlp.experts.local_num_experts_pool
                    for layer_idx in self.num_local_experts_per_layer
                ),
                default=0,
            )
            if self.craft_global_pool_enabled
            else self.num_local_experts
        )
        self.buffer_tensor_list: list[list[Any]] = [[] for _ in range(num_buffer_tensor)]
        self.init_buffer_tensor(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = self.model.get_log2phy_map(
                self.num_dense_layers + layer_idx
            )

    def _get_num_local_experts_per_layer(self):
        num_local_experts_per_layer = {}
        for layer_idx in range(self.num_dense_layers, self.model.config.num_hidden_layers):
            experts = self.model.model.layers[layer_idx].mlp.experts
            num_local_experts_per_layer[layer_idx] = experts.local_num_experts
        return num_local_experts_per_layer

    def init_buffer_tensor(self, num_buffer_tensor):
        for buffer_id in range(num_buffer_tensor):
            for name in self.expert_weight_names:
                complete_name = "model.layers." + str(self.num_dense_layers) + ".mlp.experts." + name
                expert_tensor = self.param_dict[complete_name][0]
                buffer_tensor = torch.empty_like(expert_tensor)
                self.buffer_tensor_list[buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        self.param_dict = dict()
        if self.model.quant_config is not None:
            quant_type = self.model.model.layers[self.num_dense_layers].mlp.experts.quant_type
            if quant_type == QuantType.W8A8:
                self.expert_weight_names = [
                    "w13_weight_list",
                    "w2_weight_list",
                    "w13_weight_scale_fp32_list",
                    "w2_weight_scale_list",
                ]
                if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                    self.expert_weight_names.append("fused_w1_scale_list")
                    self.expert_weight_names.append("fused_w2_scale_list")
            elif quant_type == QuantType.W4A8:
                self.expert_weight_names = [
                    "w13_weight_list",
                    "w2_weight_list",
                    "w13_weight_scale_list",
                    "w2_weight_scale_list",
                    "w13_scale_bias_list",
                    "w2_scale_bias_list",
                ]
            else:
                raise ValueError(f"EPLB not support {quant_type}")
        else:
            self.expert_weight_names = ["w13_weight", "w2_weight"]

        for layer_idx in range(self.num_dense_layers, self.model.config.num_hidden_layers):
            self.expert_param_per_layer[layer_idx] = list()
            experts = self.model.model.layers[layer_idx].mlp.experts
            num_local_experts = self.num_local_experts_per_layer[layer_idx]
            if getattr(experts, "craft_global_pool_enabled", False):
                self._init_global_pool_expert_param_for_layer(layer_idx, experts)
                continue
            if getattr(experts, "craft_pool_enabled", False):
                self._init_pool_expert_param_for_layer(layer_idx, experts)
                continue
            for name in self.expert_weight_names:
                param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
                param_value = getattr(experts, name)
                self.param_dict[param_key] = param_value
            for local_expert_id in range(num_local_experts):
                per_expert_param = list()
                for name in self.expert_weight_names:
                    per_expert_param.append(
                        self.param_dict["model.layers." + str(layer_idx) + ".mlp.experts." + name][local_expert_id]
                    )
                self.expert_param_per_layer[layer_idx].append(per_expert_param)

    def _init_global_pool_expert_param_for_layer(self, layer_idx, experts):
        main_size = experts.local_num_experts_main
        pool = experts.craft_global_expert_pool
        for name in self.expert_weight_names:
            param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
            self.param_dict[param_key] = getattr(experts, name)

        for local_expert_id in range(experts.local_num_experts):
            if local_expert_id < main_size:
                params = [getattr(experts, name)[local_expert_id] for name in self.expert_weight_names]
            else:
                params = pool.parameters_for_slot(
                    local_expert_id - main_size,
                    self.expert_weight_names,
                )
            self.expert_param_per_layer[layer_idx].append(params)

    def _init_pool_expert_param_for_layer(self, layer_idx, experts):
        main_size = experts.local_num_experts_main
        num_local_experts = self.num_local_experts_per_layer[layer_idx]
        pool_names = {
            "w13_weight_list": "w13_weight_pool",
            "w2_weight_list": "w2_weight_pool",
            "w13_weight_scale_fp32_list": "w13_weight_scale_fp32_pool",
            "w2_weight_scale_list": "w2_weight_scale_pool",
            "fused_w1_scale_list": "fused_w1_scale_pool_list",
            "fused_w2_scale_list": "fused_w2_scale_pool_list",
        }
        for name in self.expert_weight_names:
            param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
            self.param_dict[param_key] = getattr(experts, name)
            pool_name = pool_names.get(name)
            if pool_name is not None and hasattr(experts, pool_name):
                self.param_dict[f"{param_key}_pool"] = getattr(experts, pool_name)

        for local_expert_id in range(num_local_experts):
            per_expert_param = list()
            for name in self.expert_weight_names:
                param_key = f"model.layers.{layer_idx}.mlp.experts.{name}"
                if local_expert_id < main_size:
                    per_expert_param.append(self.param_dict[param_key][local_expert_id])
                else:
                    pool_key = f"{param_key}_pool"
                    per_expert_param.append(self.param_dict[pool_key][local_expert_id - main_size])
            self.expert_param_per_layer[layer_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        self.moe_load = self.model.get_all_moe_loads()
        return self.moe_load

    def get_expert_cost_metadata(self) -> list[dict[str, int]]:
        """Return conservative per-layer compute and migration payload sizes."""
        metadata = []
        for layer_id in range(self.num_dense_layers, self.model.config.num_hidden_layers):
            transfer_sizes = []
            compute_sizes = []
            for tensors in self.expert_param_per_layer[layer_id]:
                tensor_bytes = [tensor.numel() * tensor.element_size() for tensor in tensors]
                transfer_sizes.append(sum(tensor_bytes))
                compute_sizes.append(sum(tensor_bytes[:2]))
            metadata.append(
                {
                    "transfer_bytes": max(transfer_sizes, default=0),
                    "compute_bytes": max(compute_sizes, default=0),
                }
            )
        return metadata

    def _export_tensor_to_file(self, expert_maps, expert_map_record_path: str):
        if self.rank_id == 0:
            expert_maps_list = expert_maps.tolist()
            record: dict[str, Any] = {"moe_layer_count": len(expert_maps_list), "layer_list": []}
            if not self.craft_pool_enabled:
                num_local_experts = int(expert_maps.max().item()) + 1
                for layer_idx, layer_data in enumerate(expert_maps_list):
                    layer_record: dict[str, Any] = {
                        "layer_id": layer_idx,
                        "device_count": len(layer_data),
                        "device_list": [],
                    }
                    for device_idx, experts in enumerate(layer_data):
                        placement = [experts.index(i) for i in range(num_local_experts)]
                        layer_record["device_list"].append(
                            {"device_id": device_idx, "device_expert": placement}
                        )
                    record["layer_list"].append(layer_record)
                with open(expert_map_record_path, "w") as f:
                    json.dump(record, f, indent=4)
                return

            pool_start = len(expert_maps_list[0][0]) // len(expert_maps_list[0]) if expert_maps_list else 0
            has_pool = False

            for layer_idx, layer_data in enumerate(expert_maps_list):
                max_slot = max((slot for experts in layer_data for slot in experts), default=-1)
                num_local_experts = max_slot + 1
                pool_size = max(0, num_local_experts - pool_start)
                has_pool = has_pool or pool_size > 0
                layer_record: dict[str, Any] = {
                    "layer_id": layer_idx,
                    "device_count": len(layer_data),
                    "pool_start": pool_start,
                    "pool_size": pool_size,
                    "device_list": [],
                }

                for device_idx, experts in enumerate(layer_data):
                    placement = [-1] * num_local_experts
                    for expert_id, local_slot in enumerate(experts):
                        if local_slot >= 0:
                            placement[local_slot] = expert_id
                    if any(expert_id < 0 for expert_id in placement):
                        raise ValueError(
                            "Invalid EPLB expert map: "
                            f"layer={layer_idx}, device={device_idx}, placement={placement}."
                        )
                    device_record = {"device_id": device_idx, "device_expert": placement}
                    layer_record["device_list"].append(device_record)

                record["layer_list"].append(layer_record)

            if has_pool:
                record["pool_mode"] = True
                record["pool_start"] = pool_start

            with open(expert_map_record_path, "w") as f:
                json.dump(record, f, indent=4)

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace], self.buffer_tensor_list[buffer_tensor_id]
        ):
            expert_tensor.copy_(buffer_tensor)
            logger.debug(f"Expert tensor shape is :{expert_tensor.shape}")

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def suspend_global_pool_routes(self, moe_layer_ids=None):
        if not self.craft_global_pool_enabled:
            return
        if moe_layer_ids is None:
            layer_ids = range(self.num_dense_layers, self.model.config.num_hidden_layers)
        else:
            layer_ids = (
                self.num_dense_layers + layer_id
                for layer_id in sorted(set(moe_layer_ids))
            )
        for layer_id in layer_ids:
            experts = self.model.model.layers[layer_id].mlp.experts
            main_only_expert_map = experts.global_expert_map.clone()
            main_only_expert_map[
                main_only_expert_map >= experts.local_num_experts_main
            ] = -1
            main_only_route = generate_craft_route_map(
                main_only_expert_map,
                local_slots=experts.local_num_experts,
            )
            self.do_update_log2phy_map(layer_id, main_only_route)

    def get_global_expert_map(self):
        all_layer_global_expert_map = []
        for layer_id in range(self.num_moe_layers):
            map_cpu = self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_id] = map_cpu[self.rank_id]

        return torch.stack(all_layer_global_expert_map)
