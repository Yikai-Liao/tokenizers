"""Instrument isolated diagnostic worktrees at joined coordinator boundaries.

No timer is added to the delivered engine. Linux process CPU includes all workers.
Both copies use the same guard, feature set and release profile. These diagnostic
samples locate current phase costs; they are not repeated performance rankings.
"""
from pathlib import Path
import shutil

ROOT = Path('/root/code/tokenizers-simplification-results')
TIMER = r'''
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
const NAMES: [&str; 10] = ["vocabulary", "corpus_plan", "initial_index", "materialize", "select", "prepare", "apply", "commit", "release", "model_output"];
static WALL: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
static CPU: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
static CALLS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
#[repr(C)]
struct Timespec { sec: std::os::raw::c_long, nano: std::os::raw::c_long }
unsafe extern "C" { fn clock_gettime(clock: i32, time: *mut Timespec) -> i32; }
fn cpu_ns() -> u64 {
    let mut time = Timespec { sec: 0, nano: 0 };
    // Linux CLOCK_PROCESS_CPUTIME_ID; this host-only diagnostic owns the output.
    assert_eq!(unsafe { clock_gettime(2, &mut time) }, 0);
    time.sec as u64 * 1_000_000_000 + time.nano as u64
}
pub(super) struct Stage { index: usize, wall: Instant, cpu: u64 }
pub(super) fn stage(index: usize) -> Stage { Stage { index, wall: Instant::now(), cpu: cpu_ns() } }
impl Drop for Stage {
    fn drop(&mut self) {
        let cpu = cpu_ns() - self.cpu;
        WALL[self.index].fetch_add(self.wall.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CPU[self.index].fetch_add(cpu, Ordering::Relaxed);
        CALLS[self.index].fetch_add(1, Ordering::Relaxed);
    }
}
pub(super) struct Report;
impl Drop for Report {
    fn drop(&mut self) {
        let rows: Vec<_> = NAMES.iter().enumerate().map(|(i, name)| format!(
            r#"{{"name":"{}","wall_seconds":{},"cpu_seconds":{},"calls":{}}}"#,
            name, WALL[i].load(Ordering::Relaxed) as f64 / 1e9,
            CPU[i].load(Ordering::Relaxed) as f64 / 1e9, CALLS[i].load(Ordering::Relaxed)
        )).collect();
        eprintln!("BPE_PHASES {{\"stages\":[{}]}}", rows.join(","));
    }
}
'''

def wrap(source, call, stage, all_calls=False):
    offset = 0
    found = 0
    while True:
        begin = source.find(call, offset)
        if begin < 0:
            break
        paren = source.index('(', begin)
        depth = 1
        end = paren + 1
        while depth:
            depth += (source[end] == '(') - (source[end] == ')')
            end += 1
        original = source[begin:end]
        replacement = '{ let _phase = phase_timing::stage(%s); %s }' % (stage, original)
        source = source[:begin] + replacement + source[end:]
        offset = begin + len(replacement)
        found += 1
        if not all_calls:
            break
    assert found, call
    return source

def prepare(kind):
    work = Path('/root/code/tokenizers-workspaces/bpe-phase-' + kind + '-20261009')
    engine = work / 'tokenizers/tk-train/src/trainers/bpe/engine'
    if (engine / 'phase_timing.rs').exists() or (ROOT / ('runner-phase-' + kind)).exists():
        raise RuntimeError('Expected fresh diagnostic worktrees and runner directories; refusing duplicate instrumentation')
    (engine / 'phase_timing.rs').write_text(TIMER)
    p = engine / 'mod.rs'
    source = p.read_text().replace('mod corpus;', 'mod corpus;\nmod phase_timing;', 1)
    # Report once per public engine call, including early returns and errors.
    start = source.index('pub(super) fn train(')
    body = source.index(') -> Result<ModelParts> {', start) + len(') -> Result<ModelParts> {')
    source = source[:body] + '\n    let _phase_report = phase_timing::Report;' + source[body:]
    if kind == 'main':
        calls = [('vocabulary::Vocabulary::initialize(', 0), ('corpus::CorpusPlan::build(', 1),
                 ('initial_pairs::InitialPairTable::build(', 2), ('pair_index::PairIndex::from_initial_pairs(', 2),
                 ('prepared_corpus.materialize::<S>(', 3), ('batch.select(', 4),
                 ('merge::prepare_merges_with_births(', 5), ('batch.candidates.clear(', 5),
                 ('prepared.apply(', 6), ('index.commit_merges_with_prepared(', 7), ('drop(events)', 7),
                 ('vocabulary.into_model_parts(', 9)]
        release = 'drop(index);\n        drop(corpus);\n        drop(arena);\n        execution.release_scratch();'
        assert release in source
        source = source.replace(release, '{ let _phase = phase_timing::stage(8); ' + release + ' }')
    else:
        calls = [('Vocabulary::initialize(', 0), ('CorpusPlan::build(', 1), ('PairIndex::build(', 2),
                 ('plan.materialize(', 3), ('Batch::select(', 4), ('batch.prepare(', 5),
                 ('prepared.apply(', 6), ('index.commit(', 7)]
        source = wrap(source, 'vocabulary.into_model_parts(', 9, all_calls=True)
        release = 'drop(index);\n            drop(corpus);'
        assert release in source
        source = source.replace(release, '{ let _phase = phase_timing::stage(8); ' + release + ' }')
    for call, stage in calls:
        source = wrap(source, call, stage)
    p.write_text(source)
    runner = ROOT / ('runner-phase-' + kind)
    shutil.copytree(ROOT / 'runner-baseline', runner, ignore=shutil.ignore_patterns('target'))
    manifest = runner / 'Cargo.toml'
    manifest.write_text(manifest.read_text().replace('/root/code/tokenizers-workspaces/bpe-simplification-baseline', str(work)))

if __name__ == '__main__':
    for kind in ['main', 'lean']:
        prepare(kind)
