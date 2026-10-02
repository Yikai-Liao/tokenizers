#!/usr/bin/env python3
"""Summarize completed block comparisons; preserve failed/omitted calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/block-fused-prepare"


def row(case):
    data = json.loads((RESULTS / (case + ".jsonl")).read_text())
    environment = RESULTS / (case + ".environment.json")
    assert hashlib.sha256(environment.read_bytes()).hexdigest() == data["environment_sha256"]
    stats = data["indexed_stats"]
    assert (stats["workers"], stats["initialization_workers"], stats["layout"], stats["corpus_slot_bytes"]) == (4, 4, "parallel_u32_dict16", 4)
    assert stats["atomic_corpus"] and not stats["alias_fallback"]
    assert data["peak_vmswap_bytes"] == 0 and data["min_memavailable_bytes"] > 1 << 30
    for allocation in [stats["posting_allocations"], stats["speculative_posting_allocations"]]:
        assert allocation["arena_requested_bytes"] == allocation["arena_retired_bytes"]
        assert allocation["heap_requested_bytes"] == allocation["heap_freed_bytes"]
    return data


def main():
    summary = {"comparisons": {}, "completed_calls": [], "failures": [], "omitted": ["zh512m-fragments"], "sampling": "one run per case; timing uncertainty not estimated"}
    for language in ["en16m", "zh256m"]:
        baseline = row(language + "-plan")
        summary["completed_calls"].append(language + "-plan")
        for mode in ["fragments", "pooled"]:
            case = language + "-" + mode
            if not (RESULTS / (case + ".jsonl")).exists():
                continue
            candidate = row(case)
            assert candidate["model_sha256"] == baseline["model_sha256"]
            assert candidate["input_bytes"] == baseline["input_bytes"]
            assert candidate["vocab_size"] == baseline["vocab_size"]
            assert candidate["min_frequency"] == baseline["min_frequency"]
            assert candidate["indexed_stats"]["fused_block_batches"] > 0
            summary["completed_calls"].append(case)
            summary["comparisons"][case] = {
                "baseline": language + "-plan", "model_equal": True,
                "train_ms": candidate["train_ms"], "baseline_train_ms": baseline["train_ms"],
                "train_change_percent": 100 * (candidate["train_ms"] / baseline["train_ms"] - 1),
                "merge_change_percent": 100 * (candidate["indexed_stats"]["merge_ms"] / baseline["indexed_stats"]["merge_ms"] - 1),
                "rss_change_percent": 100 * (candidate["peak_rss_bytes"] / baseline["peak_rss_bytes"] - 1),
                "stats": candidate["indexed_stats"], "peak_rss_bytes": candidate["peak_rss_bytes"]}
    failed = json.loads((RESULTS / "zh512m-plan.jsonl").read_text())
    assert "failure" in failed
    summary["failures"].append({"case": "zh512m-plan", **failed})
    summary["actual_native_calls"] = len(summary["completed_calls"]) + len(summary["failures"])
    summary["selected_commit"] = "1959202f30673fc21e68ab2868ba40213062c409"
    summary["selected_cases"] = ["en16m-pooled", "zh256m-pooled"]
    summary["selection_reason"] = "Pooled offsets remove per-slice offset allocations; both measured merge times improve over fd300ca8. Single-run precision is not established."
    (RESULTS / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
