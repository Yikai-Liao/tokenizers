from pathlib import Path
import shutil

root=Path('/root/code/tokenizers-workspaces/bpe-heap-timing')
engine=Path('tokenizers/tk-train/src/trainers/bpe/engine')
source=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
shutil.copy2(source/engine/'index.rs',root/engine/'index.rs')
p=root/engine/'index.rs'
s=p.read_text()
support='''
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering as AtomicOrdering}};
use std::time::Instant;
static TIMING: OnceLock<bool> = OnceLock::new();
static INITIAL_NS: AtomicU64 = AtomicU64::new(0);
static BEST_NS: AtomicU64 = AtomicU64::new(0);
static TAKE_NS: AtomicU64 = AtomicU64::new(0);
static PUSH_NS: AtomicU64 = AtomicU64::new(0);
static CLEANUP_NS: AtomicU64 = AtomicU64::new(0);
static INITIAL_RECORDS: AtomicU64 = AtomicU64::new(0);
static BEST_CALLS: AtomicU64 = AtomicU64::new(0);
static TAKE_CALLS: AtomicU64 = AtomicU64::new(0);
static PUSH_RECORDS: AtomicU64 = AtomicU64::new(0);
fn timing_begin() -> Option<Instant> {
    (*TIMING.get_or_init(|| std::env::var_os("BPE_HEAP_TIMING").is_some())).then(Instant::now)
}
fn timing_end(start: Option<Instant>, counter: &AtomicU64) {
    if let Some(start) = start {
        counter.fetch_add(u64::try_from(start.elapsed().as_nanos()).unwrap(), AtomicOrdering::Relaxed);
    }
}
pub(super) fn timing_report() {
    if !*TIMING.get_or_init(|| false) { return; }
    let read = |counter: &AtomicU64| counter.load(AtomicOrdering::Relaxed);
    eprintln!("BPE_HEAP_TIMING {}", serde_json::json!({
        "initial_ns":read(&INITIAL_NS), "best_ns":read(&BEST_NS),
        "take_ns":read(&TAKE_NS), "serial_birth_push_ns":read(&PUSH_NS),
        "index_cleanup_ns":read(&CLEANUP_NS),
        "initial_records":read(&INITIAL_RECORDS), "best_calls":read(&BEST_CALLS),
        "take_calls":read(&TAKE_CALLS), "birth_records":read(&PUSH_RECORDS)
    }));
}
pub(super) fn timed_drop_index(index: PairIndex<'_>) {
    let start = timing_begin();
    drop(index);
    timing_end(start, &CLEANUP_NS);
}
'''
s=s.replace('// Highest count first',support+'\n// Highest count first',1)
s=s.replace('        Ok(Self {\n            arena,', '''        let start = timing_begin();
        let records: usize = candidates.iter().map(Vec::len).sum();
        let queue = candidates.into_iter().flatten().collect();
        timing_end(start, &INITIAL_NS);
        INITIAL_RECORDS.fetch_add(u64::try_from(records).unwrap(), AtomicOrdering::Relaxed);
        Ok(Self {
            arena,''',1)
s=s.replace('            queue: candidates.into_iter().flatten().collect(),','            queue,',1)
s=s.replace('    pub(super) fn best(&mut self) -> Option<Priority> {','''    pub(super) fn best(&mut self) -> Option<Priority> {
        let start = timing_begin();
        let result = self.best_untimed();
        timing_end(start, &BEST_NS);
        BEST_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        result
    }
    fn best_untimed(&mut self) -> Option<Priority> {''',1)
s=s.replace("    pub(super) fn take(&mut self, priority: Priority) -> Candidate<'arena> {",'''    pub(super) fn take(&mut self, priority: Priority) -> Candidate<'arena> {
        let start = timing_begin();
        let result = self.take_untimed(priority);
        timing_end(start, &TAKE_NS);
        TAKE_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        result
    }
    fn take_untimed(&mut self, priority: Priority) -> Candidate<'arena> {''',1)
s=s.replace('        for candidate in births.into_iter().flatten() {','''        let start = timing_begin();
        let records: usize = births.iter().map(Vec::len).sum();
        for candidate in births.into_iter().flatten() {''',1)
s=s.replace('            self.queue.push(candidate);\n        }\n        Ok(())', '''            self.queue.push(candidate);
        }
        timing_end(start, &PUSH_NS);
        PUSH_RECORDS.fetch_add(u64::try_from(records).unwrap(), AtomicOrdering::Relaxed);
        Ok(())''',1)
p.write_text(s)
p=root/engine/'mod.rs'
s=(source/engine/'mod.rs').read_text()
s=s.replace('drop(index);', 'index::timed_drop_index(index);')
s=s.replace('                reuse = true;\n                continue;', '                reuse = true;\n                index::timed_drop_index(index);\n                continue;',1)
lines=[]
for line in s.splitlines():
    if line.strip().startswith('return Ok((vocab, merges, trainer.special_tokens.clone()))'):
        indent=line[:len(line)-len(line.lstrip())]
        lines.append(indent+'index::timing_report();')
    lines.append(line)
p.write_text('\n'.join(lines)+'\n')
