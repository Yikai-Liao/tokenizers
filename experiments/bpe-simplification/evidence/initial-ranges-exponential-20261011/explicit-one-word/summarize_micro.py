"""Summarize the twelve formal pairs after discarding warmup pair zero."""
import csv
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
with (root / "microbench.csv").open() as file:
    runs = list(csv.DictReader(file))
summary = []
for name in dict.fromkeys(r["case"] for r in runs):
    selected = [r for r in runs if r["case"] == name and r["pair"] != "0"]
    row = {key: selected[0][key] for key in ["case", "words", "width", "slots", "ranges"]}
    row["formal_pairs"] = 12
    row["medians_ns"] = {arm: statistics.median(float(r["ns_per_call"]) for r in selected if r["arm"] == arm)
                         for arm in ["binary", "linear", "exponential"]}
    row["paired_median_delta_percent"] = {}
    for baseline in ["binary", "linear"]:
        deltas = []
        for pair in range(1, 13):
            arms = {r["arm"]: float(r["ns_per_call"]) for r in selected if r["pair"] == str(pair)}
            deltas.append((arms["exponential"] / arms[baseline] - 1) * 100)
        row["paired_median_delta_percent"][baseline] = statistics.median(deltas)
    summary.append(row)
(root / "microbench-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
