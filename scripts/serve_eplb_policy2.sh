#!/usr/bin/bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/mnt/sdb/models/DeepSeek-V4-Flash-w8a8-mtp/}
CARDS=${CARDS:-0,1,2,3,4,5,6,7}
DP=${DP:-2}
TP=${TP:-4}
NUM_REDUNDANT=${NUM_REDUNDANT:-8}
HEAT=${HEAT:-60}
ALGO=${ALGO:-10}
PORT=${PORT:-8008}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024000}
MNBT=${MNBT:-16384}
MNS=${MNS:-256}
GMU=${GMU:-0.90}
FUSED_MC2=${FUSED_MC2:-0}
GRAPH_MODE=${GRAPH_MODE:-FULL_DECODE_ONLY}
LOG_FILE=${LOG_FILE:-}

if [[ -n "$LOG_FILE" ]]; then
  exec >"$LOG_FILE" 2>&1
fi

export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_VERSION=0.18.0
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2="$FUSED_MC2"
export DYNAMIC_EPLB=true
export VLLM_ENGINE_READY_TIMEOUT_S=1200

ADDITIONAL_CONFIG=$(printf '%s' \
  '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false,"fuse_norm_quant":false},' \
  '"enable_cpu_binding":"true","multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,' \
  '"eplb_config":{"dynamic_eplb":true,"eplb_policy_type":2,' \
  '"num_redundant_experts":'"$NUM_REDUNDANT"',' \
  '"expert_heat_collection_interval":'"$HEAT"',"algorithm_execution_interval":'"$ALGO"'}}')

echo "[EPLB2] cards=$CARDS redundant=$NUM_REDUNDANT heat=$HEAT algo=$ALGO fused_mc2=$FUSED_MC2 graph_mode=$GRAPH_MODE"
echo "[EPLB2] additional_config=$ADDITIONAL_CONFIG"

exec vllm serve "$MODEL_PATH" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MNBT" \
  --served-model-name dsv4 \
  --gpu-memory-utilization "$GMU" \
  --api-server-count 1 \
  --max-num-seqs "$MNS" \
  --no-enable-prefix-caching \
  --data-parallel-size "$DP" \
  --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --safetensors-load-strategy prefetch \
  --quantization ascend \
  --port "$PORT" \
  --block-size 128 \
  --additional-config "$ADDITIONAL_CONFIG" \
  --compilation-config '{"cudagraph_mode":"'"$GRAPH_MODE"'","pass_config":{"enable_sp":false}}'
