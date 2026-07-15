#!/usr/bin/env bash

# This package is intentionally fixed to the validated single-node 16-card
# topology. Runtime locations may be supplied by the entrypoint.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL_PATH:=}"
: "${MODEL_NAME:=dsv4}"
: "${RESULT_ROOT:=}"
: "${REPO:=}"

CARDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
DP=4
TP=4
EP=16
NUM_REDUNDANT=16
POOL_SIZE=1

# Standard vLLM Ascend EPLB timing parameters.
HEAT=600
ALGO=50

PORT=${PORT:-8077}
GRAPH_MODE=FULL_DECODE_ONLY
HCCL_BUFFSIZE=4096
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
START_TIMEOUT=${START_TIMEOUT:-1800}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-7200}
EPLB_WARMUP_OUTPUT_LEN=${EPLB_WARMUP_OUTPUT_LEN:-800}
MAX_WARMUP_ROUNDS=${MAX_WARMUP_ROUNDS:-2}

detect_repo() {
  if [[ -n "$REPO" ]]; then
    return
  fi

  local packaged_repo
  packaged_repo=$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || true)
  if [[ -d "$packaged_repo/vllm_ascend" ]]; then
    REPO=$packaged_repo
  elif [[ -d "$PWD/vllm_ascend" ]]; then
    REPO=$PWD
  fi
}

configure_runtime_env() {
  detect_repo
  export MODEL_PATH MODEL_NAME RESULT_ROOT REPO
  export CARDS DP TP EP NUM_REDUNDANT POOL_SIZE HEAT ALGO PORT
  export GRAPH_MODE HCCL_BUFFSIZE MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS
  export MAX_NUM_SEQS GPU_MEMORY_UTILIZATION START_TIMEOUT BENCH_TIMEOUT
  export EPLB_WARMUP_OUTPUT_LEN MAX_WARMUP_ROUNDS

  export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
  if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
  fi
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=10
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export VLLM_USE_V1=1
  export VLLM_VERSION=0.18.0
  export USE_MULTI_BLOCK_POOL=1
  export USE_MULTI_GROUPS_KV_CACHE=1
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
  export VLLM_ASCEND_ENABLE_FUSED_MC2=1
  export VLLM_ENGINE_READY_TIMEOUT_S=1200
  export HCCL_BUFFSIZE

  if [[ -n "$REPO" ]]; then
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

configure_runtime_env
