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
import time
from enum import Enum

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group


class ExpertWeightUpdateState(Enum):
    WAITING = 0  # waiting for updated expert_map by EplbWorker
    READY = 1  # ready for d2d expert weights updating
    TRANSFERRING = 2  # d2d finished and waiting for updating expert_map into model


class D2DExpertWeightLoader:
    def __init__(self, policy_type: int | None = None):
        self.comm_op_list = None
        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self.layer_id = -1  # layer id to be updated
        self.state = ExpertWeightUpdateState.WAITING
        self.recv_expert_list = []
        self.num_layers = 0
        self.comm_group = get_dynamic_eplb_group()
        self.craft_pool_migration = policy_type == 4
        self._p2p_staging_tensors = []
        self._recv_staging_tasks = []
        self._craft_transfer_tasks = []

    def set_adator(self, eplb_adaptor):
        self.eplb_adaptor = eplb_adaptor

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info, updated_expert_map, layer_id):
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once("current d2d weight update tasks are on-going, cannot accept new weight update task")
            return

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        self._p2p_staging_tensors = []
        self._recv_staging_tasks = []
        self._craft_transfer_tasks = []
        craft_comm_ops = []
        local_rank = None
        if self.craft_pool_migration:
            rank_in_group = getattr(self.comm_group, "rank_in_group", None)
            local_rank = rank_in_group if isinstance(rank_in_group, int) else dist.get_rank()
        send_bytes = 0
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
            for param_id, src_tensor in enumerate(
                self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]
            ):
                send_tensor = self._stage_tensor_for_p2p(src_tensor) if self.craft_pool_migration else src_tensor
                op = dist.P2POp(
                    dist.isend, send_tensor, dst_rank, group=self.comm_group.device_group
                )
                if self.craft_pool_migration:
                    craft_comm_ops.append(
                        ((local_rank, dst_rank, global_expert_id_to_send, param_id), op)
                    )
                    self._craft_transfer_tasks.append(
                        ((local_rank, dst_rank, global_expert_id_to_send, param_id), dist.isend, send_tensor, dst_rank)
                    )
                else:
                    self.comm_op_list.append(op)
                if self.craft_pool_migration:
                    send_bytes += src_tensor.numel() * src_tensor.element_size()

        recv_bytes = 0
        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            for param_id, buffer_tensor in enumerate(self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]):
                recv_tensor = (
                    self._stage_recv_tensor_for_p2p(buffer_tensor)
                    if self.craft_pool_migration
                    else buffer_tensor
                )
                op = dist.P2POp(
                    dist.irecv, recv_tensor, recv_rank, group=self.comm_group.device_group
                )
                if self.craft_pool_migration:
                    craft_comm_ops.append(
                        ((recv_rank, local_rank, global_expert_id_to_recv, param_id), op)
                    )
                    self._craft_transfer_tasks.append(
                        ((recv_rank, local_rank, global_expert_id_to_recv, param_id), dist.irecv, recv_tensor, recv_rank)
                    )
                else:
                    self.comm_op_list.append(op)
                if self.craft_pool_migration:
                    recv_bytes += buffer_tensor.numel() * buffer_tensor.element_size()
            local_expert_to_replace = self.updated_expert_map[global_expert_id_to_recv].item()
            self.recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

        if self.craft_pool_migration:
            self.comm_op_list = [op for _, op in sorted(craft_comm_ops, key=lambda item: item[0])]

        if self.craft_pool_migration:
            logger.info(
                "[EPLB-MIG] layer=%d send_experts=%d recv_experts=%d MB=%.2f",
                layer_id, len(expert_send_info), len(expert_recv_info),
                (send_bytes + recv_bytes) / 1e6,
            )

        self.state = ExpertWeightUpdateState.READY

    def set_log2phy_map(self, log2phy_map):
        self.updated_log2phy_map = log2phy_map

    def asyn_expert_weight_transfer(self, reqs):
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be launched
        if self.state != ExpertWeightUpdateState.READY:
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.craft_pool_migration and self._craft_transfer_tasks:
            self._synchronize_device()
            self._transfer_craft_weights_ordered()
            self._synchronize_device()
        elif self.comm_op_list:
            reqs.extend(dist.batch_isend_irecv(self.comm_op_list))

        self.state = ExpertWeightUpdateState.TRANSFERRING

    def update_expert_map_and_weight(self, reqs):
        # Only after send/recv tasks have been launched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERRING:
            return

        # Waiting for send/recv tasks finish
        t0 = time.perf_counter() if self.craft_pool_migration else None
        for req in reqs:
            req.wait()
        if self.craft_pool_migration:
            self._synchronize_device()
            logger.info("[EPLB-MIG] layer=%d transfer wait %.1f ms",
                        self.layer_id, (time.perf_counter() - t0) * 1e3)

        if self.comm_op_list is not None:
            self.comm_op_list = None

        for buffer_tensor, recv_tensor in self._recv_staging_tasks:
            buffer_tensor.copy_(recv_tensor)
        self._recv_staging_tasks = []

        # update expert_map
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id, local_expert_to_replace, buffer_tensor_id)

        if self.layer_id == self.num_layers - 1:
            logger.info("[EPLB] finished update expert weight.")

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self._p2p_staging_tensors = []
        self._craft_transfer_tasks = []
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING

    def _transfer_craft_weights_ordered(self):
        device_group = self.comm_group.device_group
        for _, op, tensor, peer_rank in sorted(
            self._craft_transfer_tasks, key=lambda task: task[0]
        ):
            op(tensor, peer_rank, group=device_group).wait()

    @staticmethod
    def _needs_zero_offset_staging(tensor):
        if not isinstance(tensor, torch.Tensor):
            return False
        try:
            return tensor.storage_offset() != 0
        except RuntimeError:
            return True

    def _stage_tensor_for_p2p(self, tensor):
        if not self._needs_zero_offset_staging(tensor):
            return tensor

        staged_tensor = torch.empty_like(tensor)
        staged_tensor.copy_(tensor)
        self._p2p_staging_tensors.append(staged_tensor)
        return staged_tensor

    def _stage_recv_tensor_for_p2p(self, tensor):
        if not self._needs_zero_offset_staging(tensor):
            return tensor

        staged_tensor = torch.empty_like(tensor)
        self._p2p_staging_tensors.append(staged_tensor)
        self._recv_staging_tasks.append((tensor, staged_tensor))
        return staged_tensor

    @staticmethod
    def _synchronize_device():
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "synchronize"):
            npu.synchronize()
