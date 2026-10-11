from pathlib import Path
import os, subprocess, shutil, hashlib, sys

root = Path('/root/code/tokenizers-workspaces/bpe-positions-child-results-20261012')
work = Path('/tmp/bpe-selected-rules-measurement-20261011')
rel = Path('tokenizers/tk-train/src/trainers/bpe')
candidate = sys.argv[1]
assert candidate.startswith('child-')
module = Path('/tmp/bpe-child-diag.rs').read_text()
module = module.replace('pub payload_bytes: u64, pub retained_heap_bytes: u64,', 'pub payload_bytes: u64, pub retained_heap_bytes: u64,\n    pub payload_by_bin: [u64; 10], pub capacity_by_bin: [u64; 10],')
module = module.replace('self.payload_bytes += bytes as u64;', 'self.payload_bytes += bytes as u64; self.payload_by_bin[bin(len)] += bytes as u64; self.capacity_by_bin[bin(len)] += heap as u64;')
assert len(sys.argv) == 2 or sys.argv[2] == '--candidate-only'
for arm in ([candidate] if len(sys.argv) > 2 else ['baseline', candidate]):
    child = arm != 'baseline'
    source = root / 'source' / arm
    for f in (source / rel).glob('*.rs'):
        shutil.copyfile(f, work / rel / f.name)
    shutil.copytree(root / 'source/baseline/experiments/bpe-simplification/runner', work / 'experiments/bpe-simplification/runner', dirs_exist_ok=True)
    (work / rel / 'diagnostic.rs').write_text(module)
    p = work / rel / 'mod.rs'
    s = p.read_text().replace('mod corpus;', 'mod diagnostic;\nmod corpus;', 1)
    s = s.replace('        self.finalize_progress(progress, plan.word_count(), "Count pairs");', '        index.storage_report("initial");\n        self.finalize_progress(progress, plan.word_count(), "Count pairs");', 1)
    s = s.replace('            eprintln!("BPE_MERGE_SECONDS', '            index.storage_report("remaining");\n            eprintln!("BPE_MERGE_SECONDS', 1)
    s = s.replace('        let (vocab, merges) = vocabulary.into_model_parts(merges);', '        diagnostic::report();\n        let (vocab, merges) = vocabulary.into_model_parts(merges);', 1)
    for tag in ['INDEX', 'MATERIALIZE', 'DROP']:
        start = s.index(f'eprintln!("BPE_{tag}_SECONDS ')
        end = s.index(';', start) + 1
        s = s[:end] + f'\n        eprintln!("BPE_{tag}_MEMORY {{}}", std::fs::read_to_string("/proc/self/status").unwrap().lines().filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:") || line.starts_with("VmSwap:")).collect::<Vec<_>>().join(" / "));' + s[end:]
    p.write_text(s)
    p = work / rel / 'index.rs'
    s = p.read_text()
    initial = 'Positions' if child else 'SmallVec<[u64; 2]>'
    owner = 'SmallVec<[Positions; 1]>' if child else 'Vec<Positions>'
    if child:
        at = s.index('{\n', s.index('fn merge_positions(')) + 2
        s = s[:at] + '    let _cost = super::diagnostic::Span::new(true);\n' + s[at:]
    s += f'''
impl PairIndex {{
    pub(super) fn storage_report(&self, stage: &str) {{
        if stage == "initial" {{ eprintln!("BPE_LAYOUT positions={{}} initial_state={{}} owner_state={{}} candidate={{}}", std::mem::size_of::<Positions>(), std::mem::size_of::<State<{initial}>>(), std::mem::size_of::<State<{owner}>>(), std::mem::size_of::<Candidate>()); }}
        let mut totals = super::diagnostic::Storage::default();
        for candidate in self.queue.iter() {{ candidate.positions.storage(&mut totals); }}
        eprintln!("BPE_STORAGE {{}} {{}}", stage, serde_json::to_string(&totals).unwrap());
    }}
}}
'''
    p.write_text(s)
    p = work / rel / 'positions.rs'
    s = p.read_text()
    needle = '        let mut leaves = SmallVec' if child else '        fragments.retain('
    assert s.count(needle) == 1
    s = s.replace(needle, '        let _cost = super::diagnostic::Span::new(false);\n' + needle, 1)
    if child:
        needle = '            self.bytes.copy_within(old_head..data_end, new_head);'
        assert s.count(needle) == 1
        s = s.replace(needle, '            let moved = std::time::Instant::now();\n' + needle, 1)
        needle = '            self.bytes[old_head..new_head].fill(0);'
        s = s.replace(needle, needle + '\n            super::diagnostic::directory(data_end - old_head, moved.elapsed().as_nanos());', 1)
        s = s.replace('            _ => Ok(Self(Repr::Root {', '            _ => { super::diagnostic::root(leaves.len()); Ok(Self(Repr::Root {', 1).replace('len: len as u64,\n            })),', 'len: len as u64,\n            })) },', 1)
        s += '''
impl Leaf { fn storage(&self, totals: &mut super::diagnostic::Storage) { totals.leaf(self.len(), self.bytes.len(), if self.bytes.spilled() { self.bytes.capacity() } else { 0 }); } }
impl Drop for Leaf { fn drop(&mut self) { super::diagnostic::retired(self.len(), self.bytes.len(), if self.bytes.spilled() { self.bytes.capacity() } else { 0 }); } }
impl Positions { pub(super) fn storage(&self, totals: &mut super::diagnostic::Storage) { totals.containers += 1; if let Repr::Root { leaves, .. } = &self.0 { totals.roots += 1; totals.root_array_bytes += (leaves.len() * std::mem::size_of::<Leaf>()) as u64; totals.leaf_headers_in_roots += leaves.len() as u64; } for leaf in self.leaves() { leaf.storage(totals); } } }
impl ReadTask<'_> { pub(super) fn weight(&self) -> usize { self.len } }
'''
        p2 = work / rel / 'merge.rs'
        m = p2.read_text().replace('        let rule = &self.batch.rules[rank];', '        super::diagnostic::task(match &positions { Source::Slice(values) => values.len(), Source::Task(task) => task.weight() });\n        let rule = &self.batch.rules[rank];', 1)
        p2.write_text(m)
    else:
        s += '''
impl Positions { pub(super) fn storage(&self, totals: &mut super::diagnostic::Storage) { totals.containers += 1; let heap = match self { Self::Compressed(bytes) => bytes.len(), _ => 0 }; let bytes = if heap != 0 { heap } else { self.len() * 8 }; totals.leaf(self.len(), bytes, heap); } }
impl Drop for Positions { fn drop(&mut self) { let heap = match self { Self::Compressed(bytes) => bytes.len(), _ => 0 }; let bytes = if heap != 0 { heap } else { self.len() * 8 }; super::diagnostic::retired(self.len(), bytes, heap); } }
'''
    p.write_text(s)
    runner = work / 'experiments/bpe-simplification/runner/src'
    shutil.copyfile('/tmp/bpe-child-allocator.rs', runner / 'allocation_diagnostic.rs')
    p = runner / 'main.rs'
    p.write_text('mod allocation_diagnostic;\n' + p.read_text())
    p = runner / 'core.rs'
    s = p.read_text().replace('    let (vocab, merges, special) = trainer.do_train(&words)?;', '    crate::allocation_diagnostic::ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);\n    let (vocab, merges, special) = trainer.do_train(&words)?;\n    crate::allocation_diagnostic::report();', 1)
    p.write_text(s)
    target = root / 'final-diagnostic-source' / arm
    shutil.copytree(work / rel, target / rel, dirs_exist_ok=True)
    shutil.copytree(work / 'experiments/bpe-simplification/runner', target / 'experiments/bpe-simplification/runner', dirs_exist_ok=True)
    cmd = ['cargo', 'build', '--release', '--locked', '--offline', '--target-dir', '/root/code/tokenizers-build/bpe-affix-core-build-20261011']
    with (root / f'{arm}-final-diagnostic-build.log').open('w') as log:
        subprocess.run(cmd, cwd=work / 'experiments/bpe-simplification/runner', env=os.environ | {'PATH': '/root/.cargo/bin:' + os.environ['PATH'], 'CARGO_INCREMENTAL': '0'}, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = root / f'{arm}-final-diagnostic-bin'
    shutil.copy2('/root/code/tokenizers-build/bpe-affix-core-build-20261011/release/bpe-bench-runner', binary)
    print(arm, hashlib.sha256(binary.read_bytes()).hexdigest(), flush=True)
