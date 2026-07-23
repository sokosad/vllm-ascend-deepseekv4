# Policy4 Portable 16-card Benchmark

This package runs a controlled single-node comparison of baseline, EPLB
Policy2, and CRAFT Policy4. It is self-contained and can either remain inside
the vllm-ascend repository or be copied as a directory to another compatible
Ascend environment.

## One-command run

Run in the foreground:

```bash
bash benchmarks/policy4_16card/run_16card.sh \
  --model-path /path/to/DeepSeek-V4-Flash-w8a8-mtp
```

Run unattended:

```bash
bash benchmarks/policy4_16card/run_16card.sh \
  --model-path /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --detach
```

The result root is printed at startup. By default it is created under
`$HOME/policy4_16card_<timestamp>`.

If this directory is copied outside the repository, the installed
`vllm-ascend` package is used. To test a particular source checkout, add
`--repo /path/to/vllm-ascend`.

## Fixed configuration

- Cards: `0-15`
- DP/TP/EP: `4/4/16`
- Graph: `FULL_DECODE_ONLY`
- Fused MC2: enabled
- FlashComm1: enabled
- Heat collection interval: `600`
- Algorithm execution interval: `50`
- Policy2 redundant experts: `16` (one extra slot per rank/layer)
- Policy4 pool size: `1` (one extra slot per rank/layer)
- Repetitions: `3` per case/group

The script does not silently disable graph capture or fused communication.

## Default matrix

The `fixed` suite is enabled by default:

- `decode_c8`: input 16, output 64, concurrency 8
- `decode_c32`: input 16, output 64, concurrency 32
- `hotspot_prefix`: one repeated prefix, output 128, concurrency 32
- `balanced_512_128`: input 512, output 128, concurrency 16
- `prefill_2048_32`: input 2048, output 32, concurrency 8

Each case/group starts a fresh service. Before measured requests, the runner
generates at least 800 decode steps. Policy2 must complete a migration cycle;
Policy4 may either complete a migration or explicitly finish an unchanged
no-op cycle. This prevents short tests from measuring only initial placement.

Additional suites are available:

```bash
# Decode-heavy candidate workloads
bash benchmarks/policy4_16card/run_16card.sh \
  --model-path /path/to/model --suites advantage

# Four long-context workloads
bash benchmarks/policy4_16card/run_16card.sh \
  --model-path /path/to/model --suites longctx

# Exact cases and a persistent resumable root
bash benchmarks/policy4_16card/run_16card.sh \
  --model-path /path/to/model \
  --cases decode_c8,long_decode_c8 \
  --root /data/policy4_run1
```

Reusing the same `--root` skips successful result/status pairs and resumes the
remaining work.

## Validate without starting a service

```bash
bash benchmarks/policy4_16card/validate_scripts.sh
```

The validator performs shell and Python syntax checks, runs the entrypoint in
self-test dry-run mode, checks all fixed topology/EPLB settings and generated
commands, and validates report generation with synthetic data. It does not
start vLLM or access an NPU.

On the target machine, a hardware-aware dry-run is also available:

```bash
bash benchmarks/policy4_16card/run_16card.sh \
  --model-path /path/to/model --dry-run
```

This additionally checks the installed runtime, 16 visible NPUs, port, and
existing vLLM processes.

## Safety and artifacts

The preflight refuses to run when another vLLM process or a service on the
configured port is visible. The runner only terminates the process group it
created; it never uses a global `pkill`. Use `--skip-busy-check` only when the
reported process is known not to use these 16 cards.

The result root contains:

- `env/`: topology, versions, source revision/diff, NPU snapshots, run config
- `serve/<case>/<group>/server.log`: complete service and EPLB logs
- `bench/<case>/<group>/`: exact commands, logs, statuses, timestamps
- `results/<group>/<case>/`: detailed measured JSON results
- `warmup_results/`: non-measured EPLB warm-up results
- `summary.csv`: one row per measured run
- `REPORT.md`: aggregate performance, migration, no-op, and payback statistics
- `state/`: current position, PID, start and finish timestamps

Compatible target environments need Bash, Python, `vllm`, `torch_npu`,
`npu-smi`, curl, and the normal Ascend runtime libraries.
