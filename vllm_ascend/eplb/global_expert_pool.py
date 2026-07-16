from dataclasses import dataclass

import torch


@dataclass
class GlobalCraftExpertPool:
    capacity: int
    parameters: dict[str, list[torch.Tensor]]

    def parameters_for_slot(self, slot_id: int, names: list[str]) -> list[torch.Tensor]:
        return [self.parameters[name][slot_id] for name in names]


_GLOBAL_POOLS: dict[tuple[str, int], GlobalCraftExpertPool] = {}


def clear_global_craft_expert_pools() -> None:
    _GLOBAL_POOLS.clear()


def bind_global_craft_expert_pool(
    layer,
    capacity: int,
    parameter_names: list[str],
) -> GlobalCraftExpertPool:
    if capacity <= 0:
        raise ValueError(f"Global CRAFT pool capacity must be positive, got {capacity}.")

    source_lists = {name: getattr(layer, name) for name in parameter_names}
    if any(not tensors for tensors in source_lists.values()):
        raise ValueError("Cannot initialize the global CRAFT pool from an empty expert list.")
    first_tensor = source_lists[parameter_names[0]][0]
    key = (str(first_tensor.device), capacity)
    pool = _GLOBAL_POOLS.get(key)
    if pool is None:
        parameters = {
            name: [tensors[slot_id % len(tensors)].clone() for slot_id in range(capacity)]
            for name, tensors in source_lists.items()
        }
        pool = GlobalCraftExpertPool(capacity=capacity, parameters=parameters)
        _GLOBAL_POOLS[key] = pool
    else:
        if set(pool.parameters) != set(parameter_names):
            raise ValueError("Global CRAFT pool parameter layout changed between MoE layers.")
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
