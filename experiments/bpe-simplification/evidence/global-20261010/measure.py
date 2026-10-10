"""Paired engine checks against the fixed 727fa3e6 baseline; no diagnostic build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = Path("/tmp/bpe-global-measurements")
root.mkdir(exist_ok=False)
baseline = Path("/tmp/bpe-global-baseline")
candidate = root / "candidate"
shutil.copy2("/tmp/bpe-global-release-target/release/bpe-bench-runner", candidate)
inputs = json.loads(Path("/root/code/tokenizers-simplification-results/online-initial/input-pretokenizers.json").read_text())
source = Path("/root/code/tokenizers-workspaces/bpe-global-simplification")
manifest = dict(
    baseline_commit="727fa3e67c9cf85a9505e9d70ec8eb9b6f37f72c",
    candidate_patch_sha256=digest("/tmp/bpe-global-final.patch"),
    binaries={"baseline": str(baseline), "candidate": str(candidate)},
    binary_sha256={"baseline": digest(baseline), "candidate": digest(candidate)},
    engine_sources_sha256={str(p.relative_to(source)): digest(p) for p in sorted((source / "tokenizers/tk-train/src/trainers/bpe/engine").glob("*.rs"))},
    lock_sha256=digest(source / "experiments/bpe-simplification/runner/Cargo.lock"),
    rustc=subprocess.check_output(["/root/.cargo/bin/rustc", "-Vv"], text=True),
    profile="release opt3 / fat LTO / codegen-units=1 / no-default-features",
    cases=inputs,
    scope="public do_train; preloaded frozen counts; serialized complete model checked after timing",
    design="one warmup pair and three measured pairs per case, alternating AB/BA; workers4/cpus0-3/vocab50000/min2",
    limitations="Shared VM. Screening observations; no equivalence budget or statistical speedup claim. No pipeline/feed changes or measurement.",
)
for item in inputs:
    assert digest(item["prepared_path"]) == item["prepared_sha256"]
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
records = []
for item in inputs:
    reference = None
    for block in range(4):
        for arm in (["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]):
            competitors = []
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    comm = (entry / "comm").read_text().strip()
                    if comm in ("cargo", "rustc") or comm.startswith("bpe-bench"):
                        competitors.append((int(entry.name), comm))
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if competitors:
                raise RuntimeError(f"Concurrent build/benchmark: {competitors}")
            out = root / f'{item["case"]}-b{block}-{arm}'
            out.mkdir()
            job = dict(protocol_version=1, attempt_id=out.name, build_id=arm, input_id=item["case"], mode="core", input=item["prepared_path"], output=str(out / "model.json"), workers=4, pretokenizer=item["pretokenizer"], trainer=dict(vocab_size=50000, min_frequency=2, prefix=None, suffix=None, max_token_length=None))
            (out / "job.json").write_text(json.dumps(job, indent=2))
            cmd = ["taskset", "-c", "0-3", manifest["binaries"][arm], str(out / "job.json")]
            env = {k: v for k, v in os.environ.items() if not k.startswith(("BPE_TRACE_", "TK_WORD_COUNTS_CACHE", "TK_WRITE_WORD_COUNTS_CACHE"))}
            swap = 0
            started = time.monotonic()
            with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
                process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
                while process.poll() is None:
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text()
                        swap = max(swap, next((int(s.split()[1]) for s in status.splitlines() if s.startswith("VmSwap:")), 0))
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    time.sleep(0.1)
            record = dict(case=item["case"], block=block, arm=arm, warmup=block == 0, command=cmd, returncode=process.returncode, subprocess_wall=time.monotonic()-started, max_swap_kib=swap)
            if process.returncode == 0:
                record.update(json.loads((out / "stdout.log").read_text()))
                actual = json.loads((out / "model.json").read_text())
                if reference is None:
                    assert arm == "baseline"
                    reference = actual
                record["model_equal"] = actual == reference
                record["model_sha256"] = digest(out / "model.json")
                record["valid"] = record["model_equal"] and swap == 0
            else:
                record["valid"] = False
            records.append(record)
            (root / "runs.json").write_text(json.dumps(records, indent=2))
            print(json.dumps({k: record.get(k) for k in ("case", "block", "arm", "warmup", "valid", "metrics")}), flush=True)
            if not record["valid"]:
                raise RuntimeError(f"Invalid sample; raw evidence retained: {out}")
