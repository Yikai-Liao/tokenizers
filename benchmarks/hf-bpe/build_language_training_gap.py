#!/usr/bin/env python3
"""Build the isolated optimized DWARF diagnostic; leave the shared builder intact."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "build_affix_analysis.py"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"non-unique diagnostic anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def diagnostic_source(source):
    indexed = source / "tk-train/src/trainers/bpe/indexed.rs"
    text = indexed.read_text()
    fields = """
    pub diagnostic_aa_batches: usize,
    pub diagnostic_aa_visits: usize,
    pub diagnostic_aa_rewrites: usize,
    pub diagnostic_non_aa_single_batches: usize,
    pub diagnostic_effective_rewrites: usize,
    pub diagnostic_batch_sizes: Vec<usize>,
"""
    indexed.write_text(replace_once(text, "    pub batch_rounds: usize,", "    pub batch_rounds: usize," + fields))
    parallel = source / "tk-train/src/trainers/bpe/indexed/parallel.rs"
    text = parallel.read_text()
    text = replace_once(text, "    while ids.len() < trainer.vocab_size {", "    while ids.len() < trainer.vocab_size {\n        let diagnostic_visits_before = stats.posting_visits;")
    text = replace_once(text, "        stats.batch_rounds += 1;", """        stats.batch_rounds += 1;
        if stats.diagnostic_batch_sizes.len() <= rules.len() {
            stats.diagnostic_batch_sizes.resize(rules.len() + 1, 0);
        }
        stats.diagnostic_batch_sizes[rules.len()] += 1;
        if rules[0].edge.0 == rules[0].edge.1 {
            stats.diagnostic_aa_batches += 1;
            stats.diagnostic_aa_visits += stats.posting_visits - diagnostic_visits_before;
        } else if rules.len() == 1 {
            stats.diagnostic_non_aa_single_batches += 1;
        }""")
    text = replace_once(text, "            stats.fused_batches += 1;", """            stats.fused_batches += 1;
            stats.diagnostic_effective_rewrites += prepared.diagnostic_rewrites();""")
    text = replace_once(text, "            stats.plan_ms += stage.elapsed().as_secs_f64() * 1000.0;", """            stats.diagnostic_effective_rewrites += plans.len();
            if rules[0].edge.0 == rules[0].edge.1 {
                stats.diagnostic_aa_rewrites += plans.len();
            }
            stats.plan_ms += stage.elapsed().as_secs_f64() * 1000.0;""")
    parallel.write_text(text)
    fused = source / "tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs"
    text = fused.read_text()
    text += """
impl<O: Offset, const INLINE: usize> Prepared<O, INLINE> {
    pub(super) fn diagnostic_rewrites(&self) -> usize {
        self.valid.iter().flat_map(|job| job.iter()).map(|v| v.positions.len()).sum()
    }
}
"""
    fused.write_text(text)


def main():
    original = BASE.read_text()
    transformed = replace_once(original, "    diffs = []", """    diagnostic_source(source)
    runner_main = runner / 'src/main.rs'
    runner_text = runner_main.read_text()
    marker = '''#[repr(C)]
struct ClockTime { sec: i64, nsec: i64 }
unsafe extern "C" { fn clock_gettime(clock: i32, value: *mut ClockTime) -> i32; }
fn diagnostic_mark(phase: &str) {
    let mut t = ClockTime { sec: 0, nsec: 0 };
    assert_eq!(unsafe { clock_gettime(1, &mut t) }, 0);
    eprintln!("{}", json!({"diagnostic_phase":phase,"monotonic_ns":t.sec * 1_000_000_000 + t.nsec}));
}
'''
    runner_text = runner_text.replace('fn main()', marker + 'fn main()', 1)
    runner_text = runner_text.replace(' let (vocab,merges,stats)=', ' diagnostic_mark("train_start");\\n let (vocab,merges,stats)=', 1)
    runner_text = runner_text.replace(' let elapsed_ms=', ' diagnostic_mark("train_end");\\n let elapsed_ms=', 1)
    runner_main.write_text(runner_text)
    diffs = []""")
    transformed = replace_once(transformed, "    manifest = {", """    env.update(CARGO_PROFILE_RELEASE_DEBUG='2', CARGO_PROFILE_RELEASE_STRIP='none', CARGO_PROFILE_RELEASE_OPT_LEVEL='3')
    manifest = {""")
    transformed = replace_once(transformed, "        'cargo_command': command, 'build_environment_overrides': {'CARGO_TARGET_DIR': str(build/'target')},", """        'cargo_command': command, 'build_environment_overrides': {k:v for k,v in env.items() if k.startswith(('CARGO_PROFILE_RELEASE_', 'CARGO_TARGET_DIR'))},
        'diagnostic_builder_base_sha256': base_sha,
        'diagnostic_builder_transformed_sha256': transformed_sha,
        'diagnostic_extra_counters': ['aa batches/visits/rewrites', 'effective rewrites', 'batch-size histogram'],
        'diagnostic_clock': 'CLOCK_MONOTONIC; perf record --clockid mono',""")
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve()),
                 "diagnostic_source": diagnostic_source,
                 "base_sha": hashlib.sha256(original.encode()).hexdigest(),
                 "transformed_sha": hashlib.sha256(transformed.encode()).hexdigest()}
    exec(compile(transformed, str(BASE), "exec"), namespace)


if __name__ == "__main__":
    main()
