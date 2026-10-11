"""Compare committed and working-tree Positions with the same release executable.

python measure_positions.py --output /tmp/positions-microbench
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="bc944215")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", default="0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    relative = "tokenizers/tk-train/src/trainers/bpe/positions.rs"
    baseline = subprocess.check_output(
        ["git", "show", f"{args.baseline}:{relative}"], cwd=repo, text=True
    )
    candidate = (repo / relative).read_text()
    manifest = dict(
        baseline_commit=subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=repo, text=True
        ).strip(),
        source_sha256={
            name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in (("baseline", baseline), ("candidate", candidate))
        },
        rustc=subprocess.check_output(["rustc", "-Vv"], text=True),
        cpu=args.cpu,
        queries_per_sample=150_000,
        repetitions=7,
        order="AB on even repetitions, BA on odd repetitions",
        statistic="median nanoseconds per query; candidate / baseline ratio",
        profile="cargo release defaults, opt-level 3; both arms in one executable",
        scope="synthetic lower_bound only; no full training or input encoding",
    )
    with tempfile.TemporaryDirectory(prefix="positions-lower-bound-") as temporary:
        project = Path(temporary)
        (project / "src").mkdir()
        (project / "Cargo.toml").write_text(
            '[package]\nname = "positions-lower-bound-research"\n'
            'version = "0.0.0"\nedition = "2024"\n'
            '[dependencies]\nitertools = "=0.14.0"\n'
        )
        wrapper = (
            "\nimpl Positions { pub(super) fn seek(&self, target: u64) -> usize "
            "{ self.lower_bound(target) } }\n"
        )
        (project / "src/original.rs").write_text(baseline + wrapper)
        (project / "src/standard.rs").write_text(candidate + wrapper)
        shutil.copyfile(Path(__file__).with_name("microbench.rs"), project / "src/main.rs")
        with (args.output / "correctness.log").open("w") as log:
            subprocess.run(
                ["taskset", "-c", args.cpu, "cargo", "test", "--release", "--offline",
                 "--", "--nocapture", "compare_against_uncompressed_slice"],
                cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        subprocess.run(
            ["cargo", "build", "--release", "--offline"], cwd=project, check=True
        )
        binary = project / "target/release/positions-lower-bound-research"
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        with (args.output / "results.csv").open("w") as stream:
            subprocess.run(
                ["taskset", "-c", args.cpu, str(binary)], stdout=stream, check=True
            )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
