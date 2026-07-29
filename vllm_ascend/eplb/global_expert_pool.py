from dataclasses import dataclass

import torch


@dataclass
class GlobalCraftExpertPool:
    model_id: int
    capacity: int
    source_expert_count: int
    parameters: dict[str, list[torch.Tensor]]

    def parameters_for_slot(self, slot_id: int, names: list[str]) -> list[torch.Tensor]:
        return [self.parameters[name][slot_id] for name in names]


_GLOBAL_POOLS: dict[tuple[int, str, int], GlobalCraftExpertPool] = {}
_ACTIVE_MODEL_ID = 0


def begin_global_craft_expert_pool_model() -> int:
    """Start an isolated pool namespace for one model build."""
    global _ACTIVE_MODEL_ID
    _GLOBAL_POOLS.clear()
    _ACTIVE_MODEL_ID += 1
    return _ACTIVE_MODEL_ID


def clear_global_craft_expert_pools() -> None:
    _GLOBAL_POOLS.clear()


def cache_global_craft_weight_lists(layer) -> None:
    pool = layer.craft_global_expert_pool.parameters
    layer.craft_global_w1 = (
        layer.w13_weight_list + pool["w13_weight_list"]
    )
    layer.craft_global_w2 = layer.w2_weight_list + pool["w2_weight_list"]
    layer.craft_global_w1_scale = (
        layer.w13_weight_scale_fp32_list
        + pool["w13_weight_scale_fp32_list"]
    )
    layer.craft_global_w2_scale = (
        layer.w2_weight_scale_list + pool["w2_weight_scale_list"]
    )
    if hasattr(layer, "fused_w1_scale_list"):
        layer.craft_global_fused_w1_scale = (
            layer.fused_w1_scale_list + pool["fused_w1_scale_list"]
        )
        layer.craft_global_fused_w2_scale = (
            layer.fused_w2_scale_list + pool["fused_w2_scale_list"]
        )


def bind_global_craft_expert_pool(
    layer,
    capacity: int,
    parameter_names: list[str],
    model_id: int | None = None,
) -> GlobalCraftExpertPool:
    if capacity <= 0:
        raise ValueError(f"Global CRAFT pool capacity must be positive, got {capacity}.")

    source_lists = {name: getattr(layer, name) for name in parameter_names}
    if any(not tensors for tensors in source_lists.values()):
        raise ValueError("Cannot initialize the global CRAFT pool from an empty expert list.")
    source_lengths = {len(tensors) for tensors in source_lists.values()}
    if len(source_lengths) != 1:
        raise ValueError(
            "Global CRAFT pool requires the same expert count for every parameter list: "
            f"lengths={sorted(source_lengths)}."
        )
    source_expert_count = source_lengths.pop()
    first_tensor = source_lists[parameter_names[0]][0]
    model_id = _ACTIVE_MODEL_ID if model_id is None else int(model_id)
    key = (model_id, str(first_tensor.device), capacity)
    pool = _GLOBAL_POOLS.get(key)
    if pool is None:
        parameters = {
            name: [tensors[slot_id % len(tensors)].clone() for slot_id in range(capacity)]
            for name, tensors in source_lists.items()
        }
        pool = GlobalCraftExpertPool(
            model_id=model_id,
            capacity=capacity,
            source_expert_count=source_expert_count,
            parameters=parameters,
        )
        _GLOBAL_POOLS[key] = pool
    else:
        if set(pool.parameters) != set(parameter_names):
            raise ValueError("Global CRAFT pool parameter layout changed between MoE layers.")
        if pool.source_expert_count != source_expert_count:
            raise ValueError(
                "Global CRAFT pool requires the same main expert count across MoE layers: "
                f"expected={pool.source_expert_count}, actual={source_expert_count}."
            )
        for name, tensors in source_lists.items():
            expected = pool.parameters[name][0]
            actual = tensors[0]
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ValueError(
                    "Global CRAFT pool requires identical expert shapes across MoE layers: "
                    f"parameter={name}, expected={tuple(expected.shape)}/{expected.dtype}, "
                    f"actual={tuple(actual.shape)}/{actual.dtype}."
                )

    layer.craft_global_expert_pool = pool
    return pool
