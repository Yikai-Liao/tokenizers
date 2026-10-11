"""Paired, isolated public do_train measurements using frozen release runners.

python measure.py --baseline BIN --candidate BIN --bytelevel WORDS \
    --whitespace WORDS --output DIR
Round zero warms both arms; four alternating formal pairs follow by default.
"""
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


def competing_processes():
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            name = (path / "comm").read_text().strip()
            if name in ("cargo", "rustc", "bpe-bench-runner"):
                result.append((path.name, name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "bytelevel", "whitespace", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpus", default="0-3")
    args = parser.parse_args()
    assert args.rounds >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {name: str(getattr(args, name).resolve()) for name in ("baseline", "candidate")}
    inputs = {"bytelevel_regex": args.bytelevel.resolve(), "whitespace": args.whitespace.resolve()}
    manifest = dict(
        binaries={name: dict(path=path, sha256=digest(path)) for name, path in binaries.items()},
        inputs={name: dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
                for name, path in inputs.items()},
        workers=4, cpus=args.cpus, rounds=args.rounds, warmup_round=0,
        order="AB on even rounds, BA on odd rounds",
        rustc=subprocess.check_output(["rustc", "-Vv"], text=True),
        cpu=subprocess.check_output(["lscpu"], text=True),
        profile=dict(opt_level=3, lto="fat", codegen_units=1),
        scope="Public do_train, including output construction; excludes load and serialization. "
              "Process HWM includes loaded input counts. Swap sampled every 25 ms.",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    runs = []
    for kind, path in inputs.items():
        reference = None
        for block in range(args.rounds):
            for arm in (["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]):
                assert not competing_processes(), "Concurrent build/test runner"
                folder = args.output / f"{kind}-{block}-{arm}"
                folder.mkdir()
                job = dict(
                    protocol_version=1, attempt_id=folder.name, build_id=arm, input_id=kind,
                    mode="core", input=str(path), output=str(folder / "model.json"),
                    workers=4, pretokenizer=kind,
                    trainer=dict(vocab_size=50000, min_frequency=2, prefix=None, suffix=None,
                                 max_token_length=None),
                )
                job_path = folder / "job.json"
                job_path.write_text(json.dumps(job, indent=2))
                command = ["taskset", "-c", args.cpus, binaries[arm], str(job_path)]
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("TK_WORD_COUNTS_CACHE", "TK_WRITE_WORD_COUNTS_CACHE", "BPE_TRACE_"))}
                swap = 0
                with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                    process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                    while process.poll() is None:
                        try:
                            lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                            swap = max(swap, next((int(line.split()[1]) for line in lines
                                                  if line.startswith("VmSwap:")), 0))
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                        time.sleep(.025)
                assert process.returncode == 0, f"Failed sample: {folder}"
                output = json.loads((folder / "stdout.log").read_text())
                model = json.loads((folder / "model.json").read_text())
                if reference is None:
                    reference = model
                equal = model == reference
                run = dict(case=kind, block=block, arm=arm, warmup=block == 0,
                           command=command, model_equal=equal, model_sha256=digest(folder / "model.json"),
                           max_swap_kib=swap, **output)
                runs.append(run)
                (args.output / "runs.json").write_text(json.dumps(runs, indent=2))
                print(json.dumps({key: run[key] for key in ("case", "block", "arm", "metrics", "model_equal")}), flush=True)
                assert equal and swap == 0, f"Invalid sample: {folder}"
    summary = []
    for kind in inputs:
        rows = [run for run in runs if run["case"] == kind and not run["warmup"]]
        keys = ("train_seconds", "train_cpu_seconds", "process_hwm_kib_before_validation")
        row = dict(case=kind, formal_pairs=args.rounds - 1)
        for arm in binaries:
            row[arm] = {key: statistics.median(run["metrics"][key] for run in rows if run["arm"] == arm)
                        for key in keys}
        row["paired_median_delta_percent"] = {
            key: statistics.median(
                (next(run for run in rows if run["arm"] == "candidate" and run["block"] == block)["metrics"][key]
                 / next(run for run in rows if run["arm"] == "baseline" and run["block"] == block)["metrics"][key] - 1) * 100
                for block in range(1, args.rounds)) for key in keys}
        summary.append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
