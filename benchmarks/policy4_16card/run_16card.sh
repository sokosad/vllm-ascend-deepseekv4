#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

MODEL_PATH=${MODEL_PATH:-}
MODEL_NAME=${MODEL_NAME:-dsv4}
RESULT_ROOT=${RESULT_ROOT:-}
REPO=${REPO:-}
SUITES=fixed
EXACT_CASES=""
BENCH_GROUPS="baseline policy2 policy4"
RUNS=3
DETACH=0
DRY_RUN=0
SKIP_BUSY_CHECK=0
SELF_TEST=${POLICY4_BENCH_SELF_TEST:-0}

usage() {
  cat <<'EOF'
Usage:
  bash run_16card.sh --model-path PATH [options]

Required:
  --model-path PATH       DeepSeek-V4 model and tokenizer directory.

Options:
  --repo PATH             vllm-ascend source tree. Auto-detected when this
                          package remains inside the repository; otherwise the
                          installed vllm-ascend package is used.
  --root PATH             Persistent result directory. Default:
                          $HOME/policy4_16card_<timestamp>
  --model-name NAME       Served model name (default: dsv4).
  --suites LIST           Comma-separated: fixed, advantage, longctx.
                          Default: fixed.
  --cases LIST            Exact comma-separated cases; overrides --suites.
  --groups LIST           Comma-separated subset of baseline,policy2,policy4.
  --runs N                Measured repetitions per case/group (default: 3).
  --port N                API port (default: 8077).
  --detach                Run unattended with output in <root>/runner.log.
  --dry-run               Validate the environment and print all commands.
  --skip-busy-check       Allow launch while another vLLM process is visible.
  -h, --help              Show this help.

Fixed test settings:
  16 cards, DP=4, TP=4, EP=16, FULL_DECODE_ONLY, Fused MC2, FlashComm1,
  expert_heat_collection_interval=600, algorithm_execution_interval=50,
  Policy2 num_redundant_experts=16, Policy4 craft_pool_size=1.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --model-path) MODEL_PATH=${2:?missing value for --model-path}; shift 2 ;;
    --repo) REPO=${2:?missing value for --repo}; shift 2 ;;
    --root) RESULT_ROOT=${2:?missing value for --root}; shift 2 ;;
    --model-name) MODEL_NAME=${2:?missing value for --model-name}; shift 2 ;;
    --suites) SUITES=${2:?missing value for --suites}; shift 2 ;;
    --cases) EXACT_CASES=${2:?missing value for --cases}; shift 2 ;;
    --groups) BENCH_GROUPS=${2:?missing value for --groups}; shift 2 ;;
    --runs) RUNS=${2:?missing value for --runs}; shift 2 ;;
    --port) PORT=${2:?missing value for --port}; export PORT; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-busy-check) SKIP_BUSY_CHECK=1; shift ;;
    --foreground) DETACH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "--model-path is required" >&2
  usage >&2
  exit 2
fi
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--runs must be a positive integer" >&2
  exit 2
fi

MODEL_PATH=$(cd "$MODEL_PATH" 2>/dev/null && pwd) || {
  echo "model path does not exist or is not accessible: $MODEL_PATH" >&2
  exit 2
}
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "model path is missing config.json: $MODEL_PATH" >&2
  exit 2
fi
if [[ -n "$REPO" ]]; then
  REPO=$(cd "$REPO" 2>/dev/null && pwd) || {
    echo "repository path is not accessible: $REPO" >&2
    exit 2
  }
  if [[ ! -d "$REPO/vllm_ascend" ]]; then
    echo "repository does not contain vllm_ascend/: $REPO" >&2
    exit 2
  fi
fi

if [[ -z "$RESULT_ROOT" ]]; then
  RESULT_ROOT="${HOME:-/tmp}/policy4_16card_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RESULT_ROOT"
RESULT_ROOT=$(cd "$RESULT_ROOT" && pwd)

BENCH_GROUPS=${BENCH_GROUPS//,/ }
SUITES=${SUITES//,/ }
EXACT_CASES=${EXACT_CASES//,/ }

FIXED_CASES="decode_c8 decode_c32 hotspot_prefix balanced_512_128 prefill_2048_32"
ADVANTAGE_CASES="long_decode_c4 long_decode_c8 long_decode_c16 hotspot_long_decode"
LONGCTX_CASES="longctx_2500_1500_c50 longctx_3500_1500_c10 longctx_9000_1000_c50 longctx_32000_1000_c1"

if [[ -n "$EXACT_CASES" ]]; then
  SELECTED_CASES=$EXACT_CASES
else
  SELECTED_CASES=""
  for suite in $SUITES; do
    case "$suite" in
      fixed) SELECTED_CASES+=" $FIXED_CASES" ;;
      advantage) SELECTED_CASES+=" $ADVANTAGE_CASES" ;;
      longctx) SELECTED_CASES+=" $LONGCTX_CASES" ;;
      *) echo "unknown suite: $suite" >&2; exit 2 ;;
    esac
  done
fi
SELECTED_CASES=${SELECTED_CASES# }

for group in $BENCH_GROUPS; do
  case "$group" in
    baseline|policy2|policy4) ;;
    *) echo "unknown group: $group" >&2; exit 2 ;;
  esac
done

export MODEL_PATH MODEL_NAME RESULT_ROOT REPO RUNS
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if (( DETACH == 1 )); then
  CHILD_ARGS=(
    --model-path "$MODEL_PATH"
    --root "$RESULT_ROOT"
    --model-name "$MODEL_NAME"
    --suites "$SUITES"
    --groups "$BENCH_GROUPS"
    --runs "$RUNS"
    --port "$PORT"
    --foreground
  )
  [[ -n "$REPO" ]] && CHILD_ARGS+=(--repo "$REPO")
  [[ -n "$EXACT_CASES" ]] && CHILD_ARGS+=(--cases "$EXACT_CASES")
  (( DRY_RUN == 1 )) && CHILD_ARGS+=(--dry-run)
  (( SKIP_BUSY_CHECK == 1 )) && CHILD_ARGS+=(--skip-busy-check)
  nohup bash "$SCRIPT_DIR/run_16card.sh" "${CHILD_ARGS[@]}" > "$RESULT_ROOT/runner.log" 2>&1 &
  RUNNER_PID=$!
  echo "$RUNNER_PID" > "$RESULT_ROOT/runner.pid"
  echo "started: pid=$RUNNER_PID root=$RESULT_ROOT"
  echo "follow: tail -f '$RESULT_ROOT/runner.log'"
  exit 0
fi

mkdir -p "$RESULT_ROOT"/{env,serve,bench,results,warmup_results,state,commands}
exec > >(tee -a "$RESULT_ROOT/orchestrator.log") 2>&1

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[preflight-failed] required command not found: $1"
    return 1
  }
}

preflight() {
  local failed=0
  if (( SELF_TEST == 1 )); then
    for command_name in awk bash grep seq tail tee; do
      require_command "$command_name" || failed=1
    done
    if [[ -z "$PYTHON_BIN" ]]; then
      echo "[preflight-failed] python3 or python is required"
      failed=1
    fi
    (( failed == 0 )) && echo "[preflight-ok] self-test mode"
    return "$failed"
  fi

  for command_name in awk bash curl npu-smi pgrep seq setsid tail tee timeout vllm; do
    require_command "$command_name" || failed=1
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[preflight-failed] python3 or python is required"
    failed=1
  fi

  local card_count
  card_count=$(awk -F, '{print NF}' <<< "$CARDS")
  if (( card_count != 16 || DP * TP != 16 || EP != 16 )); then
    echo "[preflight-failed] this package requires exactly 16 cards and DP*TP=16"
    failed=1
  fi

  if [[ -n "$PYTHON_BIN" ]]; then
    local npu_count
    npu_count=$("$PYTHON_BIN" -c 'import torch, torch_npu; print(torch.npu.device_count())' 2>/dev/null || echo 0)
    if ! [[ "$npu_count" =~ ^[0-9]+$ ]] || (( npu_count < 16 )); then
      echo "[preflight-failed] visible NPU count is $npu_count, expected at least 16"
      failed=1
    fi
  fi

  pgrep -af 'vllm serve|VLLM::EngineCore|VLLMWorker' > "$RESULT_ROOT/env/preexisting_vllm_processes.txt" 2>/dev/null || true
  if [[ -s "$RESULT_ROOT/env/preexisting_vllm_processes.txt" && $SKIP_BUSY_CHECK -eq 0 ]]; then
    echo "[preflight-failed] existing vLLM processes found; see env/preexisting_vllm_processes.txt"
    failed=1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[preflight-failed] port $PORT already has a healthy service"
    failed=1
  fi
  if (( failed != 0 )); then
    return 1
  fi
  echo "[preflight-ok] model=$MODEL_PATH repo=${REPO:-installed-package} cards=$CARDS"
}

snapshot_environment() {
  date -Is > "$RESULT_ROOT/state/started_at"
  cp "$SCRIPT_DIR"/*.sh "$RESULT_ROOT/commands/"
  cp "$SCRIPT_DIR/summarize_results.py" "$RESULT_ROOT/commands/"
  uname -a > "$RESULT_ROOT/env/uname.txt"
  lscpu > "$RESULT_ROOT/env/lscpu.txt" 2>&1 || true
  env | sort > "$RESULT_ROOT/env/environment.txt"
  if (( SELF_TEST == 1 )); then
    echo "not queried in self-test mode" > "$RESULT_ROOT/env/npu_before.txt"
    "$PYTHON_BIN" -c 'import sys; print(sys.version)' > "$RESULT_ROOT/env/python_versions.txt" 2>&1
  else
    npu-smi info > "$RESULT_ROOT/env/npu_before.txt" 2>&1 || true
    "$PYTHON_BIN" -c 'import sys, torch, torch_npu, vllm, vllm_ascend; print(sys.version); print("torch", torch.__version__); print("torch_npu", torch_npu.__version__); print("vllm", vllm.__version__); print("vllm_ascend", vllm_ascend.__file__)' \
      > "$RESULT_ROOT/env/python_versions.txt" 2>&1 || true
  fi
  if [[ -n "$REPO" && -d "$REPO/.git" ]]; then
    git -C "$REPO" rev-parse HEAD > "$RESULT_ROOT/env/git_head.txt" 2>&1 || true
    git -C "$REPO" log -10 --oneline > "$RESULT_ROOT/env/git_log.txt" 2>&1 || true
    git -C "$REPO" status --short > "$RESULT_ROOT/env/git_status.txt" 2>&1 || true
    git -C "$REPO" diff > "$RESULT_ROOT/env/git_diff.patch" 2>&1 || true
  fi
  cat > "$RESULT_ROOT/env/run_config.txt" <<EOF
MODEL_PATH=$MODEL_PATH
MODEL_NAME=$MODEL_NAME
REPO=${REPO:-installed-package}
CARDS=$CARDS
DP=$DP
TP=$TP
EP=$EP
GROUPS=$BENCH_GROUPS
CASES=$SELECTED_CASES
RUNS=$RUNS
HEAT=$HEAT
ALGO=$ALGO
NUM_REDUNDANT=$NUM_REDUNDANT
POOL_SIZE=$POOL_SIZE
GRAPH_MODE=$GRAPH_MODE
FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2
FLASHCOMM1=$VLLM_ASCEND_ENABLE_FLASHCOMM1
PORT=$PORT
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
MAX_NUM_SEQS=$MAX_NUM_SEQS
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
EOF
}

SERVICE_PID=""
SERVICE_PGID=""

stop_service() {
  if [[ -n "$SERVICE_PGID" ]] && kill -0 -- "-$SERVICE_PGID" 2>/dev/null; then
    kill -TERM -- "-$SERVICE_PGID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 -- "-$SERVICE_PGID" 2>/dev/null || break
      sleep 2
    done
    if kill -0 -- "-$SERVICE_PGID" 2>/dev/null; then
      kill -KILL -- "-$SERVICE_PGID" 2>/dev/null || true
    fi
  fi
  SERVICE_PID=""
  SERVICE_PGID=""
  for _ in $(seq 1 30); do
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break
    sleep 2
  done
  sleep 10
}

on_exit() {
  stop_service
  date -Is > "$RESULT_ROOT/state/stopped_at"
}

on_signal() {
  exit 130
}

trap on_exit EXIT
trap on_signal INT TERM

wait_ready() {
  local waited=0
  while (( waited < START_TIMEOUT )); do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$SERVICE_PID" ]] && ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  return 1
}

start_service() {
  local case_name=$1 group=$2
  local serve_dir="$RESULT_ROOT/serve/$case_name/$group"
  mkdir -p "$serve_dir"
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[start-failed] port $PORT is occupied"
    return 1
  fi

  echo "[service-start] case=$case_name group=$group $(date -Is)" >> "$serve_dir/server.log"
  setsid bash "$SCRIPT_DIR/serve_variant.sh" "$group" >> "$serve_dir/server.log" 2>&1 &
  SERVICE_PID=$!
  SERVICE_PGID=$SERVICE_PID
  echo "$SERVICE_PID" > "$serve_dir/service.pid"
  date -Is > "$serve_dir/started_at"

  if ! wait_ready; then
    echo failed > "$serve_dir/status"
    tail -n 100 "$serve_dir/server.log" || true
    return 1
  fi
  echo ready > "$serve_dir/status"
  date -Is > "$serve_dir/ready_at"
  curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$serve_dir/models.json" 2>&1 || true
  npu-smi info > "$serve_dir/npu_ready.txt" 2>&1 || true
}

result_file() {
  local case_name=$1 group=$2 run=$3
  echo "$RESULT_ROOT/results/$group/$case_name/${group}_${case_name}_run${run}.json"
}

case_complete() {
  local case_name=$1 group=$2
  for run in $(seq 1 "$RUNS"); do
    local result status
    result=$(result_file "$case_name" "$group" "$run")
    status="$RESULT_ROOT/bench/$case_name/$group/measured_run${run}.status"
    [[ -s "$result" && -f "$status" ]] && grep -qx 0 "$status" || return 1
  done
}

run_bench() {
  local case_name=$1 group=$2 run=$3 phase=${4:-measured}
  local result_dir bench_dir log status command
  if [[ "$phase" == measured ]]; then
    result_dir="$RESULT_ROOT/results/$group/$case_name"
  else
    result_dir="$RESULT_ROOT/warmup_results/$group/$case_name/$phase"
  fi
  bench_dir="$RESULT_ROOT/bench/$case_name/$group"
  log="$bench_dir/${phase}_run${run}.log"
  status="$bench_dir/${phase}_run${run}.status"
  command="$bench_dir/${phase}_run${run}.command"
  mkdir -p "$result_dir" "$bench_dir"

  if [[ "$phase" == measured ]]; then
    local expected
    expected=$(result_file "$case_name" "$group" "$run")
    if [[ -s "$expected" && -f "$status" ]] && grep -qx 0 "$status"; then
      echo "[skip] $group/$case_name/run$run already completed"
      return 0
    fi
  fi

  date -Is > "$bench_dir/${phase}_run${run}.started_at"
  BENCH_WARMUP_MODE=$([[ "$phase" == measured ]] && echo 0 || echo 1) \
    PRINT_ONLY=1 bash "$SCRIPT_DIR/bench_case.sh" \
      "$case_name" "$group" "$run" "$result_dir" > "$command" 2> "$log.command_error"
  local command_rc=$?
  if (( command_rc != 0 )); then
    echo "$command_rc" > "$status"
    return "$command_rc"
  fi

  BENCH_WARMUP_MODE=$([[ "$phase" == measured ]] && echo 0 || echo 1) \
    timeout "$BENCH_TIMEOUT" bash "$SCRIPT_DIR/bench_case.sh" \
      "$case_name" "$group" "$run" "$result_dir" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$status"
  date -Is > "$bench_dir/${phase}_run${run}.finished_at"
  npu-smi info > "$bench_dir/${phase}_run${run}.npu_after.txt" 2>&1 || true
  return "$rc"
}

count_eplb_cycles() {
  local log=$1 group=$2
  local migrated noop
  migrated=$(grep -Fc '[EPLB] finished update expert weight.' "$log" 2>/dev/null || true)
  noop=0
  if [[ "$group" == policy4 ]]; then
    noop=$(grep -Ec '\[EPLB\] (completed CRAFT update cycle|skipped unchanged CRAFT update cycle)\.' "$log" 2>/dev/null || true)
  fi
  echo $((migrated + noop))
}

warm_up_and_confirm_eplb() {
  local case_name=$1 group=$2
  local server_log="$RESULT_ROOT/serve/$case_name/$group/server.log"
  if [[ "$group" == baseline ]]; then
    run_bench "$case_name" "$group" 1 warmup
    return $?
  fi

  local cycles_before cycles_after
  cycles_before=$(count_eplb_cycles "$server_log" "$group")
  for round in $(seq 1 "$MAX_WARMUP_ROUNDS"); do
    echo "[warmup] group=$group case=$case_name round=$round target_decode_steps=$EPLB_WARMUP_OUTPUT_LEN"
    run_bench "$case_name" "$group" "$round" "warmup_round${round}" || true
    cycles_after=$(count_eplb_cycles "$server_log" "$group")
    if (( cycles_after > cycles_before )); then
      echo confirmed > "$RESULT_ROOT/serve/$case_name/$group/eplb_warmup_status"
      sleep 10
      return 0
    fi
  done

  echo no_completed_cycle > "$RESULT_ROOT/serve/$case_name/$group/eplb_warmup_status"
  echo "[warmup-failed] no migration or unchanged-cycle completion for $group/$case_name"
  return 1
}

dry_run_commands() {
  local output="$RESULT_ROOT/commands/dry_run_commands.txt"
  : > "$output"
  for group in $BENCH_GROUPS; do
    echo "# serve: $group" >> "$output"
    PRINT_ONLY=1 bash "$SCRIPT_DIR/serve_variant.sh" "$group" >> "$output" 2>> "$output"
  done
  for case_name in $SELECTED_CASES; do
    echo "# bench: $case_name" >> "$output"
    PRINT_ONLY=1 bash "$SCRIPT_DIR/bench_case.sh" "$case_name" baseline 1 /tmp/policy4_dry_run >> "$output"
  done
  cat "$output"
}

if ! preflight; then
  exit 3
fi
snapshot_environment

if (( DRY_RUN == 1 )); then
  dry_run_commands
  date -Is > "$RESULT_ROOT/state/dry_run_finished_at"
  touch "$RESULT_ROOT/DRY_RUN_OK"
  trap - EXIT INT TERM
  echo "[dry-run-ok] $RESULT_ROOT"
  exit 0
fi

FAILURES=0
for case_name in $SELECTED_CASES; do
  for group in $BENCH_GROUPS; do
    echo "$case_name $group" > "$RESULT_ROOT/state/current"
    if case_complete "$case_name" "$group"; then
      echo "[skip-group] $group/$case_name already complete"
      continue
    fi

    stop_service
    echo "[group-start] case=$case_name group=$group $(date -Is)"
    if ! start_service "$case_name" "$group"; then
      echo "[group-start-failed] $group/$case_name"
      FAILURES=$((FAILURES + 1))
      stop_service
      continue
    fi
    if ! warm_up_and_confirm_eplb "$case_name" "$group"; then
      FAILURES=$((FAILURES + 1))
      stop_service
      continue
    fi

    date -Is > "$RESULT_ROOT/serve/$case_name/$group/measured_started_at"
    for run in $(seq 1 "$RUNS"); do
      echo "[bench] group=$group case=$case_name run=$run"
      if ! run_bench "$case_name" "$group" "$run" measured; then
        FAILURES=$((FAILURES + 1))
      fi
    done
    date -Is > "$RESULT_ROOT/serve/$case_name/$group/measured_finished_at"
    stop_service
    "$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" "$RESULT_ROOT" || true
  done
done

stop_service
npu-smi info > "$RESULT_ROOT/env/npu_after.txt" 2>&1 || true
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" "$RESULT_ROOT" || true
date -Is > "$RESULT_ROOT/state/finished_at"
rm -f "$RESULT_ROOT/state/current"

if (( FAILURES > 0 )); then
  echo "$FAILURES" > "$RESULT_ROOT/COMPLETED_WITH_FAILURES"
  trap - EXIT INT TERM
  echo "[finished-with-failures] count=$FAILURES root=$RESULT_ROOT"
  exit 1
fi

touch "$RESULT_ROOT/DONE"
trap - EXIT INT TERM
echo "[done] $RESULT_ROOT"
