"""Independent-process paired measurements against an immutable full-model oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def measure(root, manifest, arm, label, language, mode, workers, vocab, block, warmup):
    case = f"{language}-{mode}-w{workers}-v{vocab}"
    out = root / "runs" / label / case / f"b{block}-{arm}"
    out.mkdir(parents=True, exist_ok=False)
    model = out / "model.json"
    job = dict(
        protocol_version=1, attempt_id=out.name, build_id=arm,
        input_id=language + "-bytelevel", mode=mode,
        input=manifest["inputs"][language + "-bytelevel-" + mode]["path"],
        output=str(model), workers=workers, pretokenizer="bytelevel_regex",
        trainer=dict(vocab_size=vocab, min_frequency=2, prefix=None,
                     suffix=None, max_token_length=None),
    )
    job_path = out / "job.json"
    job_path.write_text(json.dumps(job, indent=2))
    binary = root / "bin" / ("baseline" if arm in ("A", "A2") else label)
    env = {k: v for k, v in os.environ.items() if not k.startswith("BPE_TRACE_")}
    command = ["taskset", "-c", f"0-{min(workers, 6)-1}", str(binary), str(job_path)]
    before = time.monotonic()
    swap = 0
    available = 2**63
    cpu_samples = []
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        while process.poll() is None:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                swap = max(swap, next((int(s.split()[1]) for s in status.splitlines()
                                      if s.startswith("VmSwap:")), 0))
                info = Path("/proc/meminfo").read_text()
                available = min(available, next(int(s.split()[1]) for s in info.splitlines()
                                                if s.startswith("MemAvailable:")))
                cpu_samples.append(Path("/proc/stat").read_text().splitlines()[0])
            except (FileNotFoundError, ProcessLookupError):
                pass
            time.sleep(0.1)
    record = dict(arm=arm, label=label, case=case, block=block, warmup=warmup,
                  returncode=process.returncode, subprocess_wall=time.monotonic()-before,
                  max_swap_kib=swap, min_available_kib=available,
                  binary_sha256=digest(binary), command=command, cpu_samples=cpu_samples)
    if process.returncode == 0:
        record.update(json.loads((out / "stdout.log").read_text()))
        reference = root / "reference" / (case + ".json")
        reference.parent.mkdir(exist_ok=True)
        if not reference.exists():
            if arm != "A":
                raise RuntimeError("Baseline must create reference first")
            reference.write_bytes(model.read_bytes())
        actual = json.loads(model.read_text())
        expected = json.loads(reference.read_text())
        record["model_equal"] = actual == expected
        record["model_sha256"] = digest(model)
        record["model_vocab_entries"] = len(actual[0])
        record["model_merges"] = len(actual[1])
        if record["model_equal"]:
            model.unlink()
    record["valid"] = process.returncode == 0 and record.get("model_equal", False) and swap == 0
    (out / "result.json").write_text(json.dumps(record, indent=2))
    with (root / (label + ".jsonl")).open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps({k: record.get(k) for k in ("case", "arm", "block", "warmup", "valid", "metrics")}), flush=True)
    if not record["valid"]:
        raise RuntimeError(f"Invalid measurement; raw record retained: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--root", type=Path, default=Path("/root/code/tokenizers-simplification-results"))
    parser.add_argument("--aa", action="store_true")
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--languages", nargs="+", default=["en", "zh"])
    parser.add_argument("--modes", nargs="+", default=["core", "pipeline"])
    parser.add_argument("--workers", nargs="+", type=int, default=[4])
    parser.add_argument("--vocabs", nargs="+", type=int, default=[50000])
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    cases = [(l, m, w, v) for l in args.languages for m in args.modes
             for w in args.workers for v in args.vocabs]
    rng = random.Random(20261009)
    for block in range(args.reps + 1):
        rng.shuffle(cases)
        for language, mode, workers, vocab in cases:
            arms = ["A", "A2" if args.aa else "B"]
            if block % 2:
                arms.reverse()
            for arm in arms:
                measure(args.root, manifest, arm, args.label, language, mode, workers,
                        vocab, block, block == 0)


if __name__ == "__main__":
    main()
