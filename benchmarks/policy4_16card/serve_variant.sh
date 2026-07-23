#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

GROUP=${1:?usage: serve_variant.sh baseline|policy2|policy4}
if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required" >&2
  exit 2
fi

BASE_CONFIG='"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false,"fuse_norm_quant":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false'

case "$GROUP" in
  baseline)
    export DYNAMIC_EPLB=false
    ADDITIONAL_CONFIG="{${BASE_CONFIG}}"
    ;;
  policy2)
    export DYNAMIC_EPLB=true
    EPLB_CONFIG='"eplb_config":{"dynamic_eplb":true,"eplb_policy_type":2,"num_redundant_experts":'"$NUM_REDUNDANT"',"expert_heat_collection_interval":'"$HEAT"',"algorithm_execution_interval":'"$ALGO"'}'
    ADDITIONAL_CONFIG="{${BASE_CONFIG},${EPLB_CONFIG}}"
    ;;
  policy4)
    export DYNAMIC_EPLB=true
    EPLB_CONFIG='"eplb_config":{"dynamic_eplb":true,"eplb_policy_type":4,"num_redundant_experts":'"$NUM_REDUNDANT"',"craft_pool_size":'"$POOL_SIZE"',"expert_heat_collection_interval":'"$HEAT"',"algorithm_execution_interval":'"$ALGO"'}'
    ADDITIONAL_CONFIG="{${BASE_CONFIG},${EPLB_CONFIG}}"
    ;;
  *)
    echo "unknown group: $GROUP" >&2
    exit 2
    ;;
esac

CMD=(
  vllm serve "$MODEL_PATH"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --served-model-name "$MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --api-server-count 1
  --max-num-seqs "$MAX_NUM_SEQS"
  --no-enable-prefix-caching
  --data-parallel-size "$DP"
  --tensor-parallel-size "$TP"
  --enable-expert-parallel
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --safetensors-load-strategy prefetch
  --quantization ascend
  --port "$PORT"
  --block-size 128
  --additional-config "$ADDITIONAL_CONFIG"
  --compilation-config '{"cudagraph_mode":"'"$GRAPH_MODE"'","pass_config":{"enable_sp":false}}'
)

echo "[serve] group=$GROUP cards=$CARDS dp=$DP tp=$TP ep=$EP port=$PORT" >&2
echo "[serve] heat=$HEAT algo=$ALGO redundant=$NUM_REDUNDANT pool=$POOL_SIZE" >&2
echo "[serve] fused_mc2=1 flashcomm1=1 graph=$GRAPH_MODE" >&2
echo "[serve] additional_config=$ADDITIONAL_CONFIG" >&2

if [[ ${PRINT_ONLY:-0} == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ -n "$REPO" ]]; then
  cd "$REPO"
fi
exec "${CMD[@]}"
