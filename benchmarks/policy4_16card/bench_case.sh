#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

CASE_NAME=${1:?usage: bench_case.sh CASE GROUP RUN RESULT_DIR}
GROUP=${2:?usage: bench_case.sh CASE GROUP RUN RESULT_DIR}
RUN=${3:?usage: bench_case.sh CASE GROUP RUN RESULT_DIR}
RESULT_DIR=${4:?usage: bench_case.sh CASE GROUP RUN RESULT_DIR}

DATASET=random
IN_LEN=16
OUT_LEN=64
PROMPTS=256
CONCURRENCY=8
REQUEST_RATE=inf
NUM_PREFIXES=0
INTERNAL_WARMUPS=1

case "$CASE_NAME" in
  decode_c8)
    IN_LEN=16; OUT_LEN=64; PROMPTS=256; CONCURRENCY=8
    ;;
  decode_c32)
    IN_LEN=16; OUT_LEN=64; PROMPTS=512; CONCURRENCY=32
    ;;
  hotspot_prefix)
    DATASET=prefix_repetition
    IN_LEN=16; OUT_LEN=128; PROMPTS=512; CONCURRENCY=32; NUM_PREFIXES=1
    ;;
  balanced_512_128)
    IN_LEN=512; OUT_LEN=128; PROMPTS=256; CONCURRENCY=16
    ;;
  prefill_2048_32)
    IN_LEN=2048; OUT_LEN=32; PROMPTS=128; CONCURRENCY=8
    ;;
  long_decode_c4)
    IN_LEN=16; OUT_LEN=1024; PROMPTS=40; CONCURRENCY=4
    ;;
  long_decode_c8)
    IN_LEN=16; OUT_LEN=1024; PROMPTS=80; CONCURRENCY=8
    ;;
  long_decode_c16)
    IN_LEN=16; OUT_LEN=512; PROMPTS=160; CONCURRENCY=16
    ;;
  hotspot_long_decode)
    DATASET=prefix_repetition
    IN_LEN=512; OUT_LEN=1024; PROMPTS=80; CONCURRENCY=8; NUM_PREFIXES=1
    ;;
  longctx_2500_1500_c50)
    IN_LEN=2500; OUT_LEN=1500; PROMPTS=500; CONCURRENCY=50; REQUEST_RATE=4
    ;;
  longctx_3500_1500_c10)
    IN_LEN=3500; OUT_LEN=1500; PROMPTS=100; CONCURRENCY=10; REQUEST_RATE=4
    ;;
  longctx_9000_1000_c50)
    IN_LEN=9000; OUT_LEN=1000; PROMPTS=500; CONCURRENCY=50; REQUEST_RATE=4
    ;;
  longctx_32000_1000_c1)
    IN_LEN=32000; OUT_LEN=1000; PROMPTS=10; CONCURRENCY=1; REQUEST_RATE=4
    ;;
  *)
    echo "unknown benchmark case: $CASE_NAME" >&2
    exit 2
    ;;
esac

if [[ ${BENCH_WARMUP_MODE:-0} == 1 ]]; then
  WARMUP_WAVES=$(((EPLB_WARMUP_OUTPUT_LEN + OUT_LEN - 1) / OUT_LEN))
  PROMPTS=$((CONCURRENCY * WARMUP_WAVES))
  INTERNAL_WARMUPS=0
fi

mkdir -p "$RESULT_DIR"
FILENAME="${GROUP}_${CASE_NAME}_run${RUN}.json"
CMD=(
  vllm bench serve
  --backend openai
  --base-url "http://127.0.0.1:$PORT"
  --endpoint /v1/completions
  --model "$MODEL_NAME"
  --served-model-name "$MODEL_NAME"
  --tokenizer "$MODEL_PATH"
  --dataset-name "$DATASET"
  --num-prompts "$PROMPTS"
  --max-concurrency "$CONCURRENCY"
  --request-rate "$REQUEST_RATE"
  --temperature 0
  --ignore-eos
  --seed 0
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,99
  --save-result
  --save-detailed
  --result-dir "$RESULT_DIR"
  --result-filename "$FILENAME"
  --label "$GROUP-$CASE_NAME-run$RUN"
  --metadata
    "group=$GROUP" "case=$CASE_NAME" "run=$RUN"
    "cards=$CARDS" "dp=$DP" "tp=$TP" "ep=$EP"
    "heat=$HEAT" "algo=$ALGO" "pool_size=$POOL_SIZE"
)

if (( INTERNAL_WARMUPS > 0 )); then
  CMD+=(--num-warmups "$INTERNAL_WARMUPS")
fi

if [[ "$DATASET" == random ]]; then
  CMD+=(
    --random-input-len "$IN_LEN"
    --random-output-len "$OUT_LEN"
    --random-range-ratio 0
  )
else
  CMD+=(
    --prefix-repetition-prefix-len "$IN_LEN"
    --prefix-repetition-suffix-len 1
    --prefix-repetition-output-len "$OUT_LEN"
    --prefix-repetition-num-prefixes "$NUM_PREFIXES"
  )
fi

printf '%q ' "${CMD[@]}"
printf '\n'
if [[ ${PRINT_ONLY:-0} == 1 ]]; then
  exit 0
fi
exec "${CMD[@]}"
