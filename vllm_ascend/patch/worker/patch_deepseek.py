from itertools import islice

import torch

from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Model,
    _get_llama_4_scaling,
)
from vllm.sequence import IntermediateTensors

import vllm_ascend.prefetch

logger = init_logger(__name__)


def forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
):
    from vllm_ascend.envs import (
        VLLM_PREFETCH,
        VLLM_PREFETCH_LOG,
        VLLM_PREFETCH_MODE,
        VLLM_PREFETCH_WEIGHT_SIZE_LIMIT,
        VLLM_PREFETCH_WEIGHTS,
    )

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    try:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling")
    except AttributeError:
        llama_4_scaling_config = None
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    layers = list(islice(self.layers, self.start_layer, self.end_layer))
    num_layers = len(layers)
    enable_prefetch = VLLM_PREFETCH

    if enable_prefetch:
        _prefetch_weights = set(
            w.strip() for w in VLLM_PREFETCH_WEIGHTS.split(",") if w.strip()
        )
        max_size = VLLM_PREFETCH_WEIGHT_SIZE_LIMIT
        if VLLM_PREFETCH_LOG:
            logger.info(
                "[prefetch] DeepSeek forward | layers=%d | weights=%s",
                num_layers,
                _prefetch_weights,
            )

    for i, layer in enumerate(layers):
        hidden_states, residual = layer(
            positions, hidden_states, residual, llama_4_scaling
        )

        if enable_prefetch:
            has_moe = hasattr(layer, "mlp") and hasattr(
                layer.mlp, "gate"
            )

            if "gate" in _prefetch_weights and has_moe:
                gate_w = layer.mlp.gate.weight
                torch.ops.vllm.prefetch_after_attn(
                    hidden_states,
                    gate_w,
                    max_size,
                    i,
                    f"layer.{i}.moe.gate",
                )

            if "next_qkv" in _prefetch_weights and i + 1 < num_layers:
                next_layer = layers[i + 1]
                next_attn = next_layer.self_attn
                qkv_w = getattr(
                    next_attn, "fused_qkv_a_proj_with_bias", None
                )
                if qkv_w is None:
                    qkv_w = getattr(next_attn, "qkv_proj", None)
                    if qkv_w is not None:
                        qkv_w = qkv_w.weight
                else:
                    qkv_w = qkv_w.weight

                if qkv_w is not None:
                    torch.ops.vllm.prefetch_after_mlp(
                        hidden_states,
                        qkv_w,
                        max_size,
                        i,
                        f"layer.{i}.next_qkv",
                    )

            torch.ops.vllm.prefetch_sync(hidden_states, i)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual,
        })

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


DeepseekV2Model.forward = forward
