#!/usr/bin/env python3
"""Run one budgeted, monitored perf invocation of the frozen diagnostic binary."""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/language-training-gap"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "SwapFree:")):
            mem[line.split(":")[0]] = int(line.split()[1]) * 1024
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {**mem, "pswpin": int(vm["pswpin"]), "pswpout": int(vm["pswpout"]),
            "loadavg": Path("/proc/loadavg").read_text().split()[:3]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("language", choices=("en", "zh"))
    p.add_argument("mode", choices=("stat", "record"))
    a = p.parse_args()
    case = f"{a.language}-final-{a.mode}"
    if len(list(RESULTS.glob("*-final-*.environment.json"))) >= 8:
        raise SystemExit("eight native calls already reserved/completed")
    env_path = RESULTS / (case + ".environment.json")
    if env_path.exists():
        raise SystemExit(f"refuse to overwrite {case}")
    build = ROOT / ".build/native-language-gap-dwarf32"
    manifest_path = build / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    binary = Path(manifest["binary_path"])
    if sha(binary) != manifest["binary_sha256"] or manifest["benchmark_posting_block_bits"] != 32:
        raise SystemExit("diagnostic binary/32-bit manifest gate failed")
    formal_case = "en-final-formal-clean" if a.language == "en" else "zh-final-formal"
    formal = json.loads((RESULTS / (formal_case + ".jsonl")).read_text())
    formal_env = json.loads((RESULTS / (formal_case + ".environment.json")).read_text())
    corpus = Path(formal["input"])
    # Census already rehashed each input. Avoid another input pass here.
    census = json.loads((RESULTS / "final-none32.json").read_text())["census"][a.language]
    if (str(corpus), corpus.stat().st_size, census["sha256"]) != (
            census["path"], census["input_bytes"], formal_env["corpus_sha256"]):
        raise SystemExit("census/formal input metadata gate failed")
    env = dict(os.environ)
    for key in ("HF_BPE_PREFIX", "HF_BPE_SUFFIX", "HF_BPE_ORACLE", "HF_BPE_MAX_TOKEN_LENGTH", "HF_BPE_COHORT_DISABLE"):
        env.pop(key, None)
    overrides = {"TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "4", "HF_BPE_BENCH_WORKERS": "4"}
    env.update(overrides)
    native = [str(binary), str(corpus), "none", "reference", "30000", "2"]
    if a.mode == "stat":
        command = ["perf", "stat", "--no-big-num", "--no-scale", "-x", ";", "-o", str(RESULTS / (case + ".perf-stat.csv")),
                   "-e", "cycles:u,instructions:u,cache-references:u,cache-misses:u,task-clock,context-switches,cpu-migrations,page-faults", "--", *native]
    else:
        command = ["perf", "record", "--clockid", "mono", "-F", "199", "-e", "cycles:u", "--call-graph", "dwarf,8192",
                   "--sample-cpu", "-m", "128", "-o", str(RESULTS / (case + ".perf.data")), "--", *native]
    before = host()
    if before["MemAvailable"] <= 1 << 30:
        raise SystemExit("MemAvailable gate failed before run")
    metadata = {"case": case, "mode": a.mode, "command": command, "native_command": native,
                "worktree_commit": manifest["worktree_commit"], "binary_path": str(binary), "binary_sha256": sha(binary),
                "build_manifest_path": str(manifest_path), "build_manifest_sha256": sha(manifest_path),
                "run_script_sha256": sha(Path(__file__)), "corpus_path": str(corpus), "corpus_sha256": census["sha256"],
                "corpus_bytes": census["input_bytes"], "env_overrides": overrides, "expected_model_sha256": formal["model_sha256"],
                "posting_block_bits": 32, "threads_requested": 4, "vocab_size": 30000, "min_frequency": 2,
                "host_before": before, "formal_control": formal_case,
                "event_scope": "whole native command and inherited workers; hardware user mode; task-clock includes system CPU",
                "cache_event_selection_reason": "large posting/corpus and heap-byte disparity requires aggregate cache evidence; four hardware events, no branch event"}
    env_path.write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = peak_swap = max_threads = 0
    min_available = before["MemAvailable"]
    fault = None
    observed_pids = set()
    with (RESULTS / (case + ".stdout")).open("w") as out, (RESULTS / (case + ".stderr")).open("w") as err:
        proc = subprocess.Popen(command, env=env, stdout=out, stderr=err, start_new_session=True)
        while proc.poll() is None:
            current = host()
            min_available = min(min_available, current["MemAvailable"])
            try:
                children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
                for child in children:
                    status = Path(f"/proc/{child}/status").read_text().splitlines()
                    observed_pids.add(int(child))
                    values = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in status if line.startswith(("VmRSS:", "VmSwap:"))}
                    peak_rss = max(peak_rss, values.get("VmRSS", 0))
                    peak_swap = max(peak_swap, values.get("VmSwap", 0))
                    threads = next((int(line.split()[1]) for line in status if line.startswith("Threads:")), 0)
                    max_threads = max(max_threads, threads)
            except (FileNotFoundError, ProcessLookupError):
                pass
            data_path = RESULTS / (case + ".perf.data")
            if min_available <= 1 << 30 or peak_swap or time.monotonic() - start > 60 or (data_path.exists() and data_path.stat().st_size > 128 << 20):
                fault = "resource/time/perf-data budget gate"
                os.killpg(proc.pid, 15)
                break
            time.sleep(.05)
        returncode = proc.wait()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    metadata.update(returncode=returncode, failure=fault, wall_seconds=time.monotonic() - start,
                    host_after=host(), min_memavailable_bytes=min_available,
                    peak_rss_bytes=peak_rss, peak_vmswap_bytes=peak_swap, max_process_threads=max_threads,
                    target_pids=sorted(observed_pids),
                    wrapper_child_user_seconds=usage_after.ru_utime - usage_before.ru_utime,
                    wrapper_child_system_seconds=usage_after.ru_stime - usage_before.ru_stime,
                    wrapper_child_minor_faults=usage_after.ru_minflt - usage_before.ru_minflt,
                    wrapper_child_major_faults=usage_after.ru_majflt - usage_before.ru_majflt,
                    wrapper_child_voluntary_switches=usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                    wrapper_child_involuntary_switches=usage_after.ru_nivcsw - usage_before.ru_nivcsw)
    env_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if returncode or fault:
        raise SystemExit(f"{case} failed; retained outputs: {returncode}, {fault}")
    row = json.loads((RESULTS / (case + ".stdout")).read_text())
    stderr = (RESULTS / (case + ".stderr")).read_text().splitlines()
    stats = [json.loads(line)["bench_indexed_stats"] for line in stderr if line.startswith("{") and "bench_indexed_stats" in line]
    phases = [json.loads(line) for line in stderr if line.startswith("{") and "diagnostic_phase" in line]
    if len(stats) != 1 or len(phases) != 2:
        raise SystemExit("missing native stats/phase markers")
    row["indexed_stats"] = stats[0]
    row["diagnostic_phases"] = phases
    row.update(case=case, env_metadata=str(env_path), environment_sha256=sha(env_path))
    (RESULTS / (case + ".jsonl")).write_text(json.dumps(row) + "\n")
    if row["model_sha256"] != formal["model_sha256"] or (stats[0]["workers"], stats[0]["initialization_workers"]) != (4, 4) or stats[0]["layout"] != "parallel_u32_flat32":
        raise SystemExit("model/workers/layout gate failed")
    for session in ("posting_allocations", "speculative_posting_allocations"):
        counts = stats[0][session]
        for left, right in (("arena_requested_bytes", "arena_retired_bytes"), ("heap_requested_bytes", "heap_freed_bytes"), ("heap_buffers", "heap_frees")):
            if counts[left] != counts[right]:
                raise SystemExit("allocation lifetime gate failed")
    print(json.dumps({"case": case, "train_ms": row["train_ms"], "observed_threads": max_threads,
                      "diagnostic_aa_batches": stats[0]["diagnostic_aa_batches"]}))


if __name__ == "__main__":
    main()
