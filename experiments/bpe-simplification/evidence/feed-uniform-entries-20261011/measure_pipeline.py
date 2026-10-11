"""Isolated public Feed/Train matrix: small warmups, two opposite-order full pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def machine_state():
    state = {}
    for name in ["loadavg", "meminfo", "vmstat", "pressure/cpu", "pressure/memory", "pressure/io"]:
        path = Path("/proc") / name
        if path.exists():
            state[name] = path.read_text()
    state["cpu_mhz"] = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("cpu MHz")]
    state["thermal"] = {str(p): p.read_text().strip() for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")}
    return state


def competitors(binaries):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            comm = (path / "comm").read_text().strip()
            exe = str((path / "exe").resolve())
            if comm in ("cargo", "rustc") or comm.startswith("bpe-bench") or exe in binaries.values():
                result.append([path.name, comm])
        except (OSError, ProcessLookupError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Comma-separated scenario names; empty selects all.")
    parser.add_argument("--extra-block", type=int, help="Explicit justified follow-up pair; 2 means A then B.")
    parser.add_argument("--reason", help="Required reason for an extra pair.")
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    builds = json.loads((evidence / "builds.json").read_text())
    binaries = {r["arm"]: r["binary"] for r in builds}
    for row in builds:
        assert row["exit_code"] == 0 and digest(row["binary"]) == row["sha256"]
    inputs = []
    for language in ["zh", "en"]:
        source = Path(f"/root/code/tokenizers-bpe-benchmarks/.bench/owner/{language}256/text.txt")
        metadata = json.loads(source.with_name("manifest.json").read_text())
        assert digest(source) == metadata["sha256"] and source.stat().st_size == metadata["bytes"]
        warmup = evidence / f"{language}-warmup.txt"
        if not warmup.exists():
            with source.open("rb") as stream, warmup.open("wb") as out:
                size = 0
                for line in stream:
                    if size + len(line) > 2**20:
                        break
                    out.write(line)
                    size += len(line)
        inputs.append(dict(language=language, path=str(source), bytes=metadata["bytes"], sha256=metadata["sha256"], lines=metadata["lines"],
                           warmup=dict(path=str(warmup), bytes=warmup.stat().st_size, sha256=digest(warmup))))
    scenarios = [dict(item, pretokenizer=kind, workers=workers,
                      scenario=f"{item['language']}-{kind}-w{workers}")
                 for workers in [1, 4] for item in inputs for kind in ["whitespace", "bytelevel_regex"]]
    manifest = dict(inputs=inputs, builds=builds, scenario_order=[c["scenario"] for c in scenarios],
                    formal_pairs=2, order={0: ["baseline", "candidate"], 1: ["candidate", "baseline"]},
                    warmups="one 1 MiB complete-line prefix per side per scenario before any full runs (16 total)",
                    profile=dict(opt_level=3, lto="fat", codegen_units=1, features="runner defaults empty; tk-train default-features=false"),
                    hash_policy="Production RandomState::default untouched; process-random iteration order; no fixed seed, sorting or diagnostics in Feed.",
                    timing="Process-internal monotonic wall and getrusage user+system CPU; lazy line reading/preprocessing/Feed, then train_vocab. Input open and Processor setup excluded equally. map->Vec completes before Feed end. Checkpoint overhead lies between Feed and Train inside pipeline.",
                    memory="Feed RSS: current /proc/self/status VmRSS. Feed HWM: cumulative process VmHWM at Feed return. Pipeline HWM before count export, model/native export; post-validation child HWM reported separately by parent.",
                    cache="SHA256 reads before matrix; small warmups; ordinary shared OS page cache, no eviction, no corpus preload in runner.",
                    validation="Every successful warmup/formal sample: full JSON-key/count TSV externally sorted LC_ALL=C and SHA256, count records; complete native model JSON and original exact vocab/ordered merges compared across arms/workers. All export/sort outside timings and pre-validation HWM.",
                    affinity={1: "0", 4: "0-3"}, environment=machine_state())
    if not (evidence / "measurement-manifest.json").exists():
        (evidence / "measurement-manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.only:
        requested = args.only.split(",")
        scenarios = [c for c in scenarios if c["scenario"] in requested]
        assert len(scenarios) == len(requested)
    if args.extra_block is not None:
        assert args.extra_block >= 2 and args.reason and args.only
        (evidence / f"extra-block-{args.extra_block}.json").write_text(json.dumps(vars(args), indent=2))
    runs_path = evidence / "pipeline-runs.json"
    records = json.loads(runs_path.read_text()) if runs_path.exists() else []
    references = {}
    for row in records:
        if row.get("valid"):
            key = (row["language"], row["pretokenizer"], row["warmup"])
            if key not in references:
                folder = evidence / "runs" / row["run"]
                references[key] = dict(counts=row["counts_sha256"], model=json.loads((folder / "model.json").read_text()), native=json.loads((folder / "model.native.json").read_text()))
    schedule = []
    if args.extra_block is None:
        for scenario in scenarios:
            schedule.extend((scenario, -1, arm) for arm in ["baseline", "candidate"])
    for scenario in scenarios:
        for block in ([args.extra_block] if args.extra_block is not None else [0, 1]):
            order = ["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]
            schedule.extend((scenario, block, arm) for arm in order)
    for scenario, block, arm in schedule:
        name = f"{scenario['scenario']}-b{block}-{arm}"
        old = next((r for r in records if r["run"] == name), None)
        if old:
            assert old["valid"], ("Existing failed sample requires explicit diagnosis before retry", old)
            continue
        competing = competitors(binaries)
        assert not competing, competing
        out = evidence / "runs" / name
        out.mkdir(parents=True)
        warmup = block == -1
        input_path = scenario["warmup"]["path"] if warmup else scenario["path"]
        job = dict(protocol_version=1, attempt_id=name, build_id=arm, input_id=scenario["scenario"], mode="pipeline",
                   input=input_path, output=str(out / "model.json"), workers=scenario["workers"], pretokenizer=scenario["pretokenizer"],
                   trainer=dict(vocab_size=50000, min_frequency=2, prefix=None, suffix=None, max_token_length=None))
        (out / "job.json").write_text(json.dumps(job, indent=2))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("BPE_", "TK_WORD_COUNTS_CACHE", "TK_WRITE_WORD_COUNTS_CACHE"))}
        env.update(LC_ALL="C", RAYON_NUM_THREADS=str(scenario["workers"]), RUST_BACKTRACE="full")
        cpus = "0" if scenario["workers"] == 1 else "0-3"
        command = ["taskset", "-c", cpus, binaries[arm], str(out / "job.json")]
        before = machine_state()
        peak_swap = peak_rss = peak_hwm = 0
        begin = time.monotonic()
        with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                try:
                    fields = {line.split(":")[0]: int(line.split()[1]) for line in Path(f"/proc/{process.pid}/status").read_text().splitlines() if line.startswith(("VmSwap:", "VmRSS:", "VmHWM:"))}
                    peak_swap = max(peak_swap, fields.get("VmSwap", 0))
                    peak_rss = max(peak_rss, fields.get("VmRSS", 0))
                    peak_hwm = max(peak_hwm, fields.get("VmHWM", 0))
                except (FileNotFoundError, ProcessLookupError):
                    pass
                time.sleep(.025)
        row = dict(run=name, scenario=scenario["scenario"], language=scenario["language"], pretokenizer=scenario["pretokenizer"], workers=scenario["workers"],
                   block=block, arm=arm, warmup=warmup, command=command, returncode=process.returncode, max_swap_kib=peak_swap,
                   sampled_process_rss_kib_including_validation=peak_rss, sampled_process_hwm_kib_including_validation=peak_hwm,
                   subprocess_wall_including_export=time.monotonic()-begin, environment={k: env[k] for k in ["LC_ALL", "RAYON_NUM_THREADS", "RUST_BACKTRACE"]},
                   machine_before=before, machine_after=machine_state(), concurrent_builds_or_benchmarks=competing, valid=False)
        if process.returncode == 0:
            row.update(json.loads((out / "stdout.log").read_text()))
            counts = Path(row["output"]["counts_path"])
            sorted_counts = out / "counts.sorted.tsv"
            subprocess.run(["sort", "-S", "64M", "-T", str(out), str(counts), "-o", str(sorted_counts)], env=env, check=True)
            row["counts_sha256"] = digest(sorted_counts)
            with sorted_counts.open("rb") as stream:
                row["count_records"] = sum(1 for _ in stream)
            row["model_sha256"] = digest(out / "model.json")
            row["native_sha256"] = digest(out / "model.native.json")
            model = json.loads((out / "model.json").read_text())
            native = json.loads((out / "model.native.json").read_text())
            key = (scenario["language"], scenario["pretokenizer"], warmup)
            if key not in references:
                references[key] = dict(counts=row["counts_sha256"], model=model, native=native)
            reference = references[key]
            row.update(counts_equal=row["counts_sha256"] == reference["counts"], model_equal=model == reference["model"], native_equal=native == reference["native"])
            metrics = row["metrics"]
            row["input_complete"] = warmup or metrics["input_lines"] == scenario["lines"]
            row["valid"] = all(row[k] for k in ["counts_equal", "model_equal", "native_equal", "input_complete"]) and row["count_records"] == metrics["feed_unique_words"] and metrics["feed_workers"] == scenario["workers"]
            # Swap is recorded rather than conflated with semantic correctness.
            row["performance_uncontaminated"] = row["valid"] and peak_swap == 0
            if row["valid"]:
                counts.unlink()
                sorted_counts.unlink()
        records.append(row)
        runs_path.write_text(json.dumps(records, indent=2))
        print(json.dumps({k: row.get(k) for k in ["scenario", "block", "arm", "valid", "max_swap_kib", "metrics"]}), flush=True)
        if not row["valid"]:
            raise RuntimeError(f"Failed or non-equivalent sample: {out}; raw logs retained")
    summarize(evidence, records)


def summarize(evidence, records):
    keys = ["feed_seconds", "feed_cpu_seconds", "feed_rss_kib", "feed_hwm_kib", "pipeline_seconds", "pipeline_cpu_seconds", "process_hwm_kib_before_validation", "train_seconds", "train_cpu_seconds"]
    summary = []
    for name in dict.fromkeys(row["scenario"] for row in records):
        rows = [r for r in records if r["scenario"] == name and not r["warmup"] and r["valid"]]
        if not rows:
            continue
        blocks = sorted({r["block"] for r in rows})
        result = dict(scenario=name, workers=rows[0]["workers"], paired_samples=len(blocks), all_zero_child_swap=all(r["max_swap_kib"] == 0 for r in rows), pairs=[])
        for arm in ["baseline", "candidate"]:
            selected = [r for r in rows if r["arm"] == arm]
            result[arm] = {key: dict(median=statistics.median(r["metrics"][key] for r in selected), min=min(r["metrics"][key] for r in selected), max=max(r["metrics"][key] for r in selected)) for key in keys}
        for block in blocks:
            a = next(r for r in rows if r["arm"] == "baseline" and r["block"] == block)
            b = next(r for r in rows if r["arm"] == "candidate" and r["block"] == block)
            result["pairs"].append(dict(block=block, metrics={key: dict(baseline=a["metrics"][key], candidate=b["metrics"][key], difference=b["metrics"][key]-a["metrics"][key], ratio=b["metrics"][key]/a["metrics"][key], percent=(b["metrics"][key]/a["metrics"][key]-1)*100) for key in keys}))
        result["paired_delta_percent"] = {key: statistics.median(pair["metrics"][key]["percent"] for pair in result["pairs"]) for key in keys}
        summary.append(result)
    (evidence / "pipeline-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
