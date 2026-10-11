"""Extract frozen production methods and build the three-arm microbenchmark."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
for arm in ["binary", "linear", "exponential"]:
    source = (root / "source" / f"{arm}-corpus.rs").read_text()
    start = source.index("    pub(super) fn initial_ranges(")
    end = source.index("\n    pub(super) fn small_pair_domain", start)
    stub = """pub struct CorpusPlan<'a> {
    pub starts: &'a [usize],
    pub length: usize,
    pub words: usize,
}
impl CorpusPlan<'_> {
    fn word_count(&self) -> usize { self.words }
"""
    (root / "source" / f"{arm}-ranges.rs").write_text(stub + source[start:end].rstrip() + "\n}\n")
manifest = str(root / "micro" / "Cargo.toml")
subprocess.run(["cargo", "generate-lockfile", "--offline", "--manifest-path", manifest], check=True)
subprocess.run([
    "cargo", "build", "--offline", "--locked", "--release", "--manifest-path", manifest,
    "--target-dir", "/tmp/bpe-exponential-explicit-micro-target-20261011",
], check=True)
