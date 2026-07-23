#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "python3 or python is required" >&2
  exit 1
fi

for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done
"$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/summarize_results.py"

TMP_ROOT=$(mktemp -d)
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

MODEL_DIR="$TMP_ROOT/model"
RESULT_DIR="$TMP_ROOT/result"
mkdir -p "$MODEL_DIR"
printf '{}\n' > "$MODEL_DIR/config.json"

if ! POLICY4_BENCH_SELF_TEST=1 bash "$SCRIPT_DIR/run_16card.sh" \
  --model-path "$MODEL_DIR" \
  --root "$RESULT_DIR" \
  --runs 1 \
  --dry-run > "$TMP_ROOT/dry_run.log" 2>&1; then
  cat "$TMP_ROOT/dry_run.log" >&2
  exit 1
fi

test -f "$RESULT_DIR/DRY_RUN_OK"
COMMANDS="$RESULT_DIR/commands/dry_run_commands.txt"
CONFIG="$RESULT_DIR/env/run_config.txt"
test -s "$COMMANDS"
test -s "$CONFIG"

grep -q '^HEAT=600$' "$CONFIG"
grep -q '^ALGO=50$' "$CONFIG"
grep -q '^CARDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15$' "$CONFIG"
grep -q '^DP=4$' "$CONFIG"
grep -q '^TP=4$' "$CONFIG"
grep -q '^EP=16$' "$CONFIG"
grep -q '^NUM_REDUNDANT=16$' "$CONFIG"
grep -q '^POOL_SIZE=1$' "$CONFIG"
grep -q '^GRAPH_MODE=FULL_DECODE_ONLY$' "$CONFIG"
grep -q '^FUSED_MC2=1$' "$CONFIG"
grep -q '^FLASHCOMM1=1$' "$CONFIG"

grep -q '# serve: baseline' "$COMMANDS"
grep -q '# serve: policy2' "$COMMANDS"
grep -q '# serve: policy4' "$COMMANDS"
grep -q 'eplb_policy_type.*2' "$COMMANDS"
grep -q 'eplb_policy_type.*4' "$COMMANDS"
grep -q 'craft_pool_size.*1' "$COMMANDS"
grep -q 'expert_heat_collection_interval.*600' "$COMMANDS"
grep -q 'algorithm_execution_interval.*50' "$COMMANDS"
grep -q 'FULL_DECODE_ONLY' "$COMMANDS"

for case_name in decode_c8 decode_c32 hotspot_prefix balanced_512_128 prefill_2048_32; do
  grep -q "# bench: $case_name" "$COMMANDS"
done

SYNTH_ROOT="$TMP_ROOT/synthetic"
mkdir -p "$SYNTH_ROOT/results/policy4/decode_c8" "$SYNTH_ROOT/serve/decode_c8/policy4"
cat > "$SYNTH_ROOT/results/policy4/decode_c8/policy4_decode_c8_run1.json" <<'EOF'
{"group":"policy4","case":"decode_c8","run":"1","completed":8,"failed":0,"output_throughput":100.0,"median_tpot_ms":10.0,"p99_itl_ms":20.0}
EOF
cat > "$SYNTH_ROOT/serve/decode_c8/policy4/server.log" <<'EOF'
[Expert Hotness] Current: mean=1, max=2, Updated: mean=1, max=1
[CRAFT-COST] layer=1 changed_slots=1 migration_mb=1 load_delta=1 payback_steps=1 limit=600 accepted=True
[EPLB] completed CRAFT update cycle.
[EPLB-MIG] layer=1 send_experts=1 recv_experts=1 MB=2.00
[EPLB-MIG] layer=1 payload transfer 3.0 ms
[EPLB-MIG] layer=1 transfer wait 4.0 ms
EOF
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_results.py" "$SYNTH_ROOT" >/dev/null
test -s "$SYNTH_ROOT/summary.csv"
test -s "$SYNTH_ROOT/REPORT.md"
grep -q 'Policy4 16-card EPLB 600/50 Report' "$SYNTH_ROOT/REPORT.md"
grep -q '100.00' "$SYNTH_ROOT/REPORT.md"

echo "validation passed: syntax, dry-run commands, fixed EPLB settings, and report generation"
