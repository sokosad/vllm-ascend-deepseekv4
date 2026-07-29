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
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this updator.
from queue import Empty

import numpy
import torch
import torch.distributed as dist
import vllm.envs as envs
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _positive_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


class EplbUpdator:
    def __init__(self, eplb_config, loader: D2DExpertWeightLoader, eplb_process: EplbProcess, process):
        self.eplb_config = eplb_config
        self.multi_stage = eplb_config.eplb_policy_type == 3
        self.init_eplb(self.eplb_config.expert_map_path, process)
        self.eplb_loader = loader
        self.eplb_process = eplb_process
        self.shared_dict = self.eplb_process.shared_dict
        self.comm_group = get_dynamic_eplb_group()
        self.craft_pool_plan = self.eplb_config.eplb_policy_type == 4
        self.full_rank_plan = self.craft_pool_plan
        self.plan_timeout_s = _positive_float(
            getattr(self.eplb_config, "craft_plan_timeout_s", 60.0),
            60.0,
        )
        rank_in_group = getattr(self.comm_group, "rank_in_group", self.rank_id)
        self.rank_in_group = rank_in_group if isinstance(rank_in_group, int) else self.rank_id
        group_ranks = getattr(self.comm_group, "ranks", None)
        self.plan_src_rank = group_ranks[0] if isinstance(group_ranks, (list, tuple)) and group_ranks else 0

    def set_adaptor(self, adaptor: VllmEplbAdaptor):
        self.adaptor = adaptor
        self.num_moe_layers = self.adaptor.num_moe_layers
        local_load = self.adaptor.get_rank_expert_workload()
        self.world_size = dist.get_world_size()
        self.device = local_load.device
        self.eplb_loader.num_layers = self.adaptor.num_dense_layers + self.adaptor.num_moe_layers
        if self.craft_pool_plan:
            self.shared_dict["craft_expert_cost_metadata"] = self.adaptor.get_expert_cost_metadata()

    def init_eplb(self, expert_map_path, process):
        self.rank_id = dist.get_rank()
        self.num_expert_load_gather = 10
        self.periodic_load_gather = True
        self.expert_heat_collection_interval: torch.int64 = self.eplb_config.expert_heat_collection_interval
        self.expert_map_path = expert_map_path
        self.expert_map_record_path = self.eplb_config.expert_map_record_path

        try:
            if not envs.VLLM_ALLOW_EXPERT_LOAD_COLLECTING:
                self.num_expert_load_gather = self.expert_heat_collection_interval
                self.periodic_load_gather = False
        except Exception:
            self.num_expert_load_gather = self.expert_heat_collection_interval
            self.periodic_load_gather = False

        self.reqs = []
        self.update_info_all = []
        self.update_info_index = 0
        self.skip_current_step = False
        self.noop_current_step = False
        self.noop_cycle_pending = False

        self.cur_iterations: torch.int64 = 0

        self.algorithm_execution_interval: torch.int64 = self.eplb_config.algorithm_execution_interval

        self.process = process

        logger.info(f"[ModelRunner] Launched EPLB process (pid={self.process.pid})")

    def update_iteration(self):
        self.cur_iterations += 1
        if self.cur_iterations == (
            self.expert_heat_collection_interval + self.algorithm_execution_interval + self.num_moe_layers
        ):
            if self.expert_map_record_path is not None:
                self.adaptor._export_tensor_to_file(self.shared_dict["expert_maps"], self.expert_map_record_path)

            self.adaptor.model.clear_all_moe_loads()
            self.cur_iterations = 0
            return True
        return False

    def get_update_info_flag(self):
        return self.cur_iterations == (self.expert_heat_collection_interval + self.algorithm_execution_interval - 1)

    def wakeup_eplb_worker_flag(self):
        return self.cur_iterations == (self.expert_heat_collection_interval - 1)

    def update_expert_weight_flag(self):
        weight_update_counter = self.cur_iterations - (
            self.expert_heat_collection_interval + self.algorithm_execution_interval
        )
        return weight_update_counter >= 0 and weight_update_counter < self.num_moe_layers

    def current_weight_update_index(self):
        return self.cur_iterations - (
            self.expert_heat_collection_interval + self.algorithm_execution_interval
        )

    def wakeup_eplb_worker(self):
        if not self.full_rank_plan or self.rank_id == self.plan_src_rank:
            self.eplb_process.planner_q.put(1)

    def _broadcast_update_info(self, update_info_all):
        object_list = [update_info_all if self.rank_id == self.plan_src_rank else None]
        group = getattr(self.comm_group, "cpu_group", None)
        if group is None:
            group = getattr(self.comm_group, "device_group", None)
        dist.broadcast_object_list(object_list, src=self.plan_src_rank, group=group)
        return object_list[0]

    def _select_rank_update_info(self, update_info_all):
        selected_update_info = []
        for record in update_info_all:
            if isinstance(record, dict) and record.get("noop", False):
                selected_update_info.append(None)
                continue
            if not isinstance(record, dict) or "send_all" not in record:
                selected_update_info.append(record)
                continue

            rank_idx = self.rank_in_group
            selected_update_info.append(
                (
                    record["send_all"][rank_idx],
                    record["recv_all"][rank_idx],
                    record["maps_all"][rank_idx],
                    record["log2phy_all"][rank_idx],
                    record["layer_id"],
                )
            )
        return selected_update_info

    def _synchronize_skip_update(self, skip_update: bool) -> bool:
        if not self.craft_pool_plan or not self.update_expert_weight_flag():
            return False

        cpu_group = getattr(self.comm_group, "cpu_group", None)
        sync_device = "cpu" if cpu_group is not None else self.device
        group = cpu_group if cpu_group is not None else self.comm_group.device_group
        update_allowed = torch.tensor(
            [not skip_update], dtype=torch.int32, device=sync_device
        )
        dist.all_reduce(update_allowed, op=dist.ReduceOp.MIN, group=group)
        return not bool(update_allowed.item())

    def forward_before(self, skip_update: bool = False):
        # A pool migration may pair any two EP ranks. DP synchronization only
        # covers ranks at the same TP position, so all EPLB ranks must agree to
        # enter a migration step before any P2P operation is launched.
        self.skip_current_step = self._synchronize_skip_update(skip_update)
        # Batch after eplb process being triggered, get update info provided by eplb process
        if self.get_update_info_flag():
            if self.full_rank_plan:
                update_info_all = None
                if self.rank_id == self.plan_src_rank:
                    try:
                        update_info_all = self.eplb_process.block_update_q.get(timeout=self.plan_timeout_s)
                    except Empty:
                        logger.error(
                            "CRAFT EPLB planner timed out after %.1f seconds; skipping this update cycle",
                            self.plan_timeout_s,
                        )
                self.update_info_all = self._broadcast_update_info(update_info_all)
                if self.update_info_all is None:
                    self.update_info_all = []
                else:
                    self.update_info_all = self._select_rank_update_info(self.update_info_all)
                    self.noop_cycle_pending = (
                        self.craft_pool_plan
                        and len(self.update_info_all) == self.num_moe_layers
                        and all(update_info is None for update_info in self.update_info_all)
                    )
            else:
                self.update_info_all = self.eplb_process.block_update_q.get()
            self.update_info_index = 0
        if self.skip_current_step:
            return
        if self.update_expert_weight_flag():
            if self.full_rank_plan:
                self.update_info_index = self.current_weight_update_index()
                if self.update_info_index < 0 or self.update_info_index >= len(self.update_info_all):
                    logger.warning_once(
                        "EPLB update info is not ready for index %s; skipping this step",
                        self.update_info_index,
                    )
                    return
                update_info = self.update_info_all[self.update_info_index]
                if update_info is None:
                    self.noop_current_step = True
                    return
            else:
                update_info = self.update_info_all.pop(0)
            (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = update_info
            log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
            self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            updated_expert_map_this_rank = torch.from_numpy(numpy.array(updated_expert_map))
            self.eplb_loader.generate_expert_d2d_transfer_task(
                expert_send_info,
                expert_recv_info,
                updated_expert_map_this_rank,
                layer_id + self.adaptor.num_dense_layers,
            )

            # set asynchronous stream for d2d expert weight update
            self.reqs = []
            self.eplb_loader.asyn_expert_weight_transfer(self.reqs)

    def forward_end(self):
        if self.wakeup_eplb_worker_flag():
            self.compute_and_set_moe_load()
            self.wakeup_eplb_worker()

        if self.noop_cycle_pending:
            if self.expert_map_record_path is not None:
                self.adaptor._export_tensor_to_file(
                    self.shared_dict["expert_maps"], self.expert_map_record_path
                )
            self.adaptor.model.clear_all_moe_loads()
            self.cur_iterations = 0
            self.update_info_all = []
            self.noop_cycle_pending = False
            logger.info("[EPLB] skipped unchanged CRAFT update cycle.")
            return

        if self.update_expert_weight_flag() and self.skip_current_step:
            self.skip_current_step = False
            return

        if self.update_expert_weight_flag() and self.noop_current_step:
            cycle_completed = self.update_iteration()
            self.noop_current_step = False
            if cycle_completed and self.craft_pool_plan and self.rank_id == 0:
                logger.info("[EPLB] completed CRAFT update cycle.")
            return

        if (
            self.update_expert_weight_flag()
            and not self.skip_current_step
            and self.expert_map_record_path is None
        ):
            self.eplb_loader.update_expert_map_and_weight(self.reqs)

        cycle_completed = self.update_iteration()
        if cycle_completed and self.craft_pool_plan and self.rank_id == 0:
            logger.info("[EPLB] completed CRAFT update cycle.")
        self.skip_current_step = False
        self.noop_current_step = False

    def compute_and_set_moe_load(self):
        local_load = self.adaptor.get_rank_expert_workload().unsqueeze(1)
        moe_load = self.comm_group.all_gather(local_load, dim=1).cpu()

        if self.multi_stage:
            moe_load = moe_load.permute(2, 0, 1, 3)

        self.shared_dict["moe_load"] = moe_load
        logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}")

        return moe_load

    def warm_up_eplb(self):
        self.shared_dict["expert_maps"] = self.adaptor.get_global_expert_map()
        self.compute_and_set_moe_load()

        src_tensor = torch.empty((1,), device=self.device)

        comm_op_list = []

        for dst_rank in range(self.world_size):
            if dst_rank == self.rank_id:
                continue
            comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank, group=self.comm_group.device_group))

        for src_rank in range(self.world_size):
            if src_rank == self.rank_id:
                continue
            comm_op_list.append(dist.P2POp(dist.irecv, src_tensor, src_rank, group=self.comm_group.device_group))
        if comm_op_list:
            reqs = dist.batch_isend_irecv(comm_op_list)

        for req in reqs:
            req.wait()

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            logger.info("[ModelRunner] EPLB process terminated")
