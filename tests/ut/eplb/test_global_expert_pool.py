from types import SimpleNamespace

import torch

from vllm_ascend.eplb.global_expert_pool import (
    begin_global_craft_expert_pool_model,
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


def test_global_pool_is_isolated_between_model_builds():
    clear_global_craft_expert_pools()
    begin_global_craft_expert_pool_model()
    first = SimpleNamespace(weight_list=[torch.tensor([1.0])])
    first_pool = bind_global_craft_expert_pool(
        first,
        1,
        ["weight_list"],
    )

    begin_global_craft_expert_pool_model()
    second = SimpleNamespace(weight_list=[torch.tensor([2.0])])
    second_pool = bind_global_craft_expert_pool(
        second,
        1,
        ["weight_list"],
    )

    assert first_pool is not second_pool
    assert first_pool.parameters["weight_list"][0].item() == 1.0
    assert second_pool.parameters["weight_list"][0].item() == 2.0


def test_global_pool_rejects_inconsistent_parameter_list_lengths():
    clear_global_craft_expert_pools()
    layer = SimpleNamespace(
        w1=[torch.tensor([1.0])],
        w2=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    try:
        bind_global_craft_expert_pool(layer, 1, ["w1", "w2"])
    except ValueError as exc:
        assert "same expert count" in str(exc)
    else:
        raise AssertionError("Expected inconsistent parameter list lengths to fail")


def test_global_pool_rejects_different_main_expert_counts_between_layers():
    clear_global_craft_expert_pools()
    first = SimpleNamespace(
        weight_list=[torch.tensor([1.0])],
    )
    second = SimpleNamespace(
        weight_list=[torch.tensor([2.0]), torch.tensor([3.0])],
    )
    bind_global_craft_expert_pool(first, 1, ["weight_list"])

    try:
        bind_global_craft_expert_pool(second, 1, ["weight_list"])
    except ValueError as exc:
        assert "same main expert count" in str(exc)
    else:
        raise AssertionError("Expected different main expert counts to fail")
