from types import SimpleNamespace

import torch

from vllm_ascend.eplb.global_expert_pool import (
    bind_global_craft_expert_pool,
    clear_global_craft_expert_pools,
)


def test_global_pool_is_shared_across_layers():
    clear_global_craft_expert_pools()
    first = SimpleNamespace(weight_list=[torch.tensor([1.0])])
    second = SimpleNamespace(weight_list=[torch.tensor([2.0])])

    first_pool = bind_global_craft_expert_pool(first, 2, ["weight_list"])
    second_pool = bind_global_craft_expert_pool(second, 2, ["weight_list"])

    assert first_pool is second_pool
    assert len(first_pool.parameters["weight_list"]) == 2
    assert first.craft_global_expert_pool.parameters["weight_list"][0] is (
        second.craft_global_expert_pool.parameters["weight_list"][0]
    )
