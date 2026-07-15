#!/usr/bin/env python3
"""Create dependency-free CSV and Markdown summaries for the 16-card run."""

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METRICS = (
    "output_throughput",
    "total_token_throughput",
    "request_throughput",
    "duration",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def load_results(root):
    rows = []
    for path in sorted((root / "results").glob("*/*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = {
            "group": str(data.get("group", path.parts[-3])),
            "case": str(data.get("case", path.parts[-2])),
            "run": str(data.get("run", path.stem.rsplit("run", 1)[-1])),
            "completed": int(data.get("completed", 0) or 0),
            "failed": int(data.get("failed", 0) or 0),
        }
        for metric in METRICS:
            row[metric] = float(data.get(metric, 0.0) or 0.0)
        rows.append(row)
    return rows


def migration_stats(log_path):
    stats = {
        "plans": 0,
        "weight_update_records": 0,
        "craft_cycle_records": 0,
        "noop_cycle_records": 0,
        "rank_layer_records": 0,
        "zero_rank_layer_records": 0,
        "recv_experts": 0,
        "logged_send_recv_mb": 0.0,
        "payload_ms_mean": 0.0,
        "payload_ms_max": 0.0,
        "wait_ms_mean": 0.0,
        "wait_ms_max": 0.0,
        "cost_accepted": 0,
        "cost_rejected": 0,
        "planner_timeouts": 0,
        "eplb_errors": 0,
    }
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stats

    stats["plans"] = text.count("[Expert Hotness]")
    stats["weight_update_records"] = text.count("[EPLB] finished update expert weight.")
    stats["craft_cycle_records"] = text.count("[EPLB] completed CRAFT update cycle.")
    stats["noop_cycle_records"] = text.count("[EPLB] skipped unchanged CRAFT update cycle.")
    stats["planner_timeouts"] = text.count("CRAFT EPLB planner timed out")
    stats["eplb_errors"] = text.count("EPLB subprocess exiting due to error")
    payload_times = []
    wait_times = []

    for line in text.splitlines():
        if "[CRAFT-COST]" in line:
            if "accepted=True" in line:
                stats["cost_accepted"] += 1
            elif "accepted=False" in line:
                stats["cost_rejected"] += 1
        if "[EPLB-MIG]" not in line:
            continue
        transfer = re.search(r"send_experts=(\d+) recv_experts=(\d+) MB=([0-9.]+)", line)
        if transfer:
            sent = int(transfer.group(1))
            received = int(transfer.group(2))
            stats["rank_layer_records"] += 1
            stats["recv_experts"] += received
            stats["logged_send_recv_mb"] += float(transfer.group(3))
            if sent == 0 and received == 0:
                stats["zero_rank_layer_records"] += 1
        payload = re.search(r"payload transfer ([0-9.]+) ms", line)
        if payload:
            payload_times.append(float(payload.group(1)))
        wait = re.search(r"transfer wait ([0-9.]+) ms", line)
        if wait:
            wait_times.append(float(wait.group(1)))

    if payload_times:
        stats["payload_ms_mean"] = mean(payload_times)
        stats["payload_ms_max"] = max(payload_times)
    if wait_times:
        stats["wait_ms_mean"] = mean(wait_times)
        stats["wait_ms_max"] = max(wait_times)
    return stats


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["group"])].append(row)
    result = {}
    for key, items in grouped.items():
        throughputs = [item["output_throughput"] for item in items]
        result[key] = {
            "runs": len(items),
            "completed": sum(item["completed"] for item in items),
            "failed": sum(item["failed"] for item in items),
            "output_throughput": mean(throughputs),
            "spread": (
                (max(throughputs) - min(throughputs)) / mean(throughputs) * 100.0
                if throughputs and mean(throughputs) > 0
                else 0.0
            ),
            "median_ttft_ms": mean([item["median_ttft_ms"] for item in items]),
            "median_tpot_ms": mean([item["median_tpot_ms"] for item in items]),
            "median_itl_ms": mean([item["median_itl_ms"] for item in items]),
            "p99_itl_ms": mean([item["p99_itl_ms"] for item in items]),
        }
    return result


def write_csv(root, rows):
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["group", "case", "run", "completed", "failed", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(root, rows):
    aggregated = aggregate(rows)
    cases = sorted({case for case, _ in aggregated})
    lines = [
        "# Policy4 16-card EPLB 600/50 Report",
        "",
        "Fixed topology: 16 cards, DP=4, TP=4, EP=16. Policy2 uses 16 redundant experts; Policy4 uses one pool slot per rank.",
        "",
        "## Performance",
        "",
        "| Case | Group | Runs | Success/Failed | Output tok/s | Spread | Median TTFT ms | Median TPOT ms | Median ITL ms | P99 ITL ms | P4 vs P2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        policy2_throughput = aggregated.get((case, "policy2"), {}).get("output_throughput", 0.0)
        for group in ("baseline", "policy2", "policy4"):
            item = aggregated.get((case, group))
            if not item:
                continue
            delta = ""
            if group == "policy4" and policy2_throughput > 0:
                delta = f"{(item['output_throughput'] / policy2_throughput - 1.0) * 100.0:+.2f}%"
            lines.append(
                f"| {case} | {group} | {item['runs']} | {item['completed']}/{item['failed']} | "
                f"{item['output_throughput']:.2f} | {item['spread']:.2f}% | "
                f"{item['median_ttft_ms']:.2f} | {item['median_tpot_ms']:.2f} | "
                f"{item['median_itl_ms']:.2f} | {item['p99_itl_ms']:.2f} | {delta} |"
            )

    lines.extend(
        [
            "",
            "## EPLB Activity",
            "",
            "Counts are log records; multi-rank logs can contain duplicate cycle records. Logged MB is the sum of rank send and receive values, not unique wire traffic.",
            "",
            "| Case | Group | Plans | Weight updates | CRAFT cycles | No-op cycles | Rank-layer records | Zero records | Received experts | Logged MB | Payload mean/max ms | Wait mean/max ms | Cost accept/reject | Timeout/Error |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for log_path in sorted((root / "serve").glob("*/*/server.log")):
        case = log_path.parts[-3]
        group = log_path.parts[-2]
        stats = migration_stats(log_path)
        lines.append(
            f"| {case} | {group} | {stats['plans']} | {stats['weight_update_records']} | "
            f"{stats['craft_cycle_records']} | {stats['noop_cycle_records']} | "
            f"{stats['rank_layer_records']} | {stats['zero_rank_layer_records']} | "
            f"{stats['recv_experts']} | {stats['logged_send_recv_mb']:.2f} | "
            f"{stats['payload_ms_mean']:.2f}/{stats['payload_ms_max']:.2f} | "
            f"{stats['wait_ms_mean']:.2f}/{stats['wait_ms_max']:.2f} | "
            f"{stats['cost_accepted']}/{stats['cost_rejected']} | "
            f"{stats['planner_timeouts']}/{stats['eplb_errors']} |"
        )

    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_results.py RESULT_ROOT")
    root = Path(sys.argv[1]).resolve()
    rows = load_results(root)
    write_csv(root, rows)
    write_report(root, rows)
    print(f"wrote {root / 'summary.csv'} and {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
