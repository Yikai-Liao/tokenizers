"""Instrument matched phase boundaries in disposable production-grain copies.

Only one public do_train call is made; this diagnostic prints aggregate timings
once after a successful attempt. It never logs within a measured phase.
"""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BPE = Path('tokenizers/tk-train/src/trainers/bpe')
BASE = Path('/tmp/bpe-corpus-baseline-20261013')


def replace(path, old, new):
    value = path.read_text()
    assert old in value, (path, old)
    path.write_text(value.replace(old, new, 1))


def prepare(source, arm):
    dest = Path('/tmp/bpe-corpus-phase-' + arm + '-20261013')
    for directory in ['tokenizers/tk-train', 'tokenizers/tk-encode',
                      'tokenizers/tk-convert', 'tokenizers/tk-serialize',
                      'tokenizers/bitcannon']:
        shutil.copytree(source / directory, dest / directory,
                        ignore=shutil.ignore_patterns('target', '.git'), dirs_exist_ok=True)
    shutil.copytree(ROOT / 'experiments/bpe-simplification/runner',
                    dest / 'experiments/bpe-simplification/runner',
                    ignore=shutil.ignore_patterns('target', '.git'), dirs_exist_ok=True)
    expected = json.loads((HERE / ('final-manifest.json' if arm == 'baseline' else 'aligned-manifest.json')).read_text())
    hashes = expected['sources']['baseline'] if arm == 'baseline' else expected['source_sha256']
    for name, digest in hashes.items():
        assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest, name
    mod = dest / BPE / 'mod.rs'
    for phase, anchor in [(0, 'let (mut vocabulary, ids) = Vocabulary::initialize(self, words, alphabet, progress)?;'),
                          (1, 'let plan = CorpusPlan::build(words, ids, self, reuse)?;' if arm == 'baseline' else 'let plan = CorpusPlan::build(words, ids, self.vocab_size, mode, self.max_token_length)?;'),
                          (2, 'let mut index = PairIndex::build(&plan, self.min_frequency, progress)?;'),
                          (3, 'corpus = plan.materialize();')]:
        replace(mod, anchor, 'let phase_begin = std::time::Instant::now();\n        ' + anchor + f'\n        diagnostic_phase({phase}, phase_begin);')
    replace(mod, 'while vocabulary.len() < self.vocab_size {', 'let rounds_begin = std::time::Instant::now();\n            while vocabulary.len() < self.vocab_size {')
    replace(mod, '            index.retire();\n            drop(corpus);', '            diagnostic_phase(4, rounds_begin);\n            let retire_begin = std::time::Instant::now();\n            index.retire();\n            drop(corpus);\n            diagnostic_phase(5, retire_begin);')
    replace(mod, 'Ok(AttemptOutcome::Complete(vocabulary))', 'diagnostic_phases_report();\n        Ok(AttemptOutcome::Complete(vocabulary))')
    with mod.open('a') as out:
        out.write('''
static PHASE_NANOS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
static PHASE_CALLS: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
#[inline]
fn diagnostic_phase(index: usize, begin: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    PHASE_NANOS[index].fetch_add(begin.elapsed().as_nanos() as u64, Relaxed);
    PHASE_CALLS[index].fetch_add(1, Relaxed);
}
fn diagnostic_phases_report() {
    use std::sync::atomic::Ordering::Relaxed;
    let names = ["vocabulary", "plan", "initial_count", "materialize", "merge_rounds", "retire", "prepare", "apply", "commit"];
    for (index, name) in names.iter().enumerate() {
        eprintln!("BPE_PHASE {}", serde_json::json!({"phase":name, "nanoseconds":PHASE_NANOS[index].load(Relaxed), "calls":PHASE_CALLS[index].load(Relaxed)}));
    }
}
''')
    merge = dest / BPE / 'merge.rs'
    if arm == 'baseline':
        anchor = 'let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;'
        replace(merge, anchor, 'let phase_begin = std::time::Instant::now();\n        ' + anchor + '\n        super::diagnostic_phase(6, phase_begin);\n        let phase_begin = std::time::Instant::now();')
        replace(merge, 'index.commit(changes)?;', 'super::diagnostic_phase(7, phase_begin);\n        let phase_begin = std::time::Instant::now();\n        index.commit(changes)?;\n        super::diagnostic_phase(8, phase_begin);')
    else:
        preparation = dest / BPE / 'corpus/merging.rs'
        anchor = 'let chunks = self.prepare(selected)?;'
        replace(preparation, anchor, 'let phase_begin = std::time::Instant::now();\n        ' + anchor + '\n        super::super::diagnostic_phase(6, phase_begin);\n        let phase_begin = std::time::Instant::now();')
        replace(preparation, 'Ok(chunks', 'let changes = chunks')
        replace(preparation, '.collect())', '.collect();\n        super::super::diagnostic_phase(7, phase_begin);\n        Ok(changes)')
        replace(merge, 'index.commit(changes)?;', 'let phase_begin = std::time::Instant::now();\n        index.commit(changes)?;\n        super::diagnostic_phase(8, phase_begin);')
    (HERE / ('phase-' + arm + '-source.json')).write_text(json.dumps({
        'purpose': 'matched aggregate wall-clock boundaries; no per-phase logging; single-call runner',
        'source_sha256': {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (dest / BPE).rglob('*.rs')},
        'runner_sha256': {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (dest / 'experiments/bpe-simplification/runner/src').rglob('*.rs')},
    }, indent=2))
    return dest


if __name__ == '__main__':
    print(prepare(BASE, 'baseline'))
    print(prepare(ROOT, 'candidate'))
