"""Instrument an expendable baseline worktree with the current BPE source.

python probe.py /tmp/bpe-interface-baseline
Diagnostics stay outside production. Build the instrumented library test, then run
interface_lifecycle_probe with BPE_INTERFACE_PROBE=<scenario>. Build the release
runner and use BPE_INTERFACE_PROBE=records to measure actual record allocations.
"""
from pathlib import Path
import shutil
import subprocess
import sys


repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
root = Path(sys.argv[1]).resolve()
assert root != repo and (root / ".git").is_file(), "Use a separate worktree"
relative = Path("tokenizers/tk-train/src/trainers/bpe")
for path in (repo / relative).glob("*.rs"):
    shutil.copy2(path, root / relative / path.name)
bpe = root / relative


def replace(name, old, new):
    path = bpe / name
    text = path.read_text()
    assert text.count(old) == 1, (name, old)
    path.write_text(text.replace(old, new))


replace("mod.rs", "mod word_counts;", "mod word_counts;\nmod record_probe;")
replace("mod.rs", "let (vocab, merges) = vocabulary.into_model_parts(merges);",
        'record_probe::event("output");\n        let (vocab, merges) = vocabulary.into_model_parts(merges);')
replace("merge.rs", "let reuse = index.reuse();",
        'super::record_probe::fail("select")?;\n        let reuse = index.reuse();')
replace("merge.rs", "let records = self\n", "super::record_probe::begin_records();\n        let records: Vec<MergeRecord> = self\n")
replace("merge.rs", "let jobs = self.prepare(corpus, limit)?;",
        'super::record_probe::finish_records(records.len(), records.capacity(), std::mem::size_of::<MergeRecord>());\n'
        '        let jobs = self.prepare(corpus, limit)?;\n        super::record_probe::event("writes");')
replace("merge.rs", "index.commit(changes)?;\n        Ok(records)",
        'index.commit(changes)?;\n        super::record_probe::event("records");\n        Ok(records)')
replace("merge.rs", "fn prepare(self, corpus: &Corpus, limit: usize) -> Result<Vec<Job>> {",
        'fn prepare(self, corpus: &Corpus, limit: usize) -> Result<Vec<Job>> {\n'
        '        super::record_probe::event("prepare");\n        super::record_probe::fail("prepare")?;')
replace("index.rs", "pub(super) fn commit(&mut self, changes: Vec<Vec<Change<Birth>>>) -> Result<()> {",
        'pub(super) fn commit(&mut self, changes: Vec<Vec<Change<Birth>>>) -> Result<()> {\n'
        '        super::record_probe::event("owners");\n        super::record_probe::fail("owner")?;')
replace("index.rs", "let mut candidates = index.queue.into_vec();",
        "let mut candidates = std::mem::take(&mut index.queue).into_vec();")
for name, ty, event in [("corpus.rs", "Corpus", "corpus"), ("index.rs", "PairIndex", "index"),
                        ("index.rs", "Candidate", "candidate")]:
    with (bpe / name).open("a") as stream:
        stream.write(f'\nimpl Drop for {ty} {{\n    fn drop(&mut self) {{ super::record_probe::event("drop-{event}"); }}\n}}\n')

(bpe / "record_probe.rs").write_text(r'''
use std::{alloc::{GlobalAlloc, Layout, System}, cell::Cell, time::Instant};
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<(usize, usize, usize)> = const { Cell::new((0, 0, 0)) };
    static START: Cell<Option<Instant>> = const { Cell::new(None) };
}
struct ProbeAllocator;
#[global_allocator]
static ALLOCATOR: ProbeAllocator = ProbeAllocator;
fn count(size: usize, realloc: bool) {
    let _ = ACTIVE.try_with(|active| {
        if active.get() {
            COUNTS.with(|counts| {
                let (calls, bytes, grows) = counts.get();
                counts.set((calls + 1, bytes + size, grows + usize::from(realloc)));
            });
        }
    });
}
unsafe impl GlobalAlloc for ProbeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size(), false);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size(), false);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size, true);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
fn setting() -> String { std::env::var("BPE_INTERFACE_PROBE").unwrap_or_default() }
pub(super) fn event(name: &str) {
    let mode = setting();
    if !mode.is_empty() && mode != "records" { eprintln!("BPE_LIFETIME {name}"); }
}
pub(super) fn fail(stage: &str) -> tk_encode::Result<()> {
    if setting() == stage { return Err(format!("injected {stage} failure").into()); }
    Ok(())
}
pub(super) fn begin_records() {
    if setting() == "records" {
        COUNTS.with(|counts| counts.set((0, 0, 0)));
        START.with(|start| start.set(Some(Instant::now())));
        ACTIVE.with(|active| active.set(true));
    }
}
pub(super) fn finish_records(len: usize, capacity: usize, size: usize) {
    ACTIVE.with(|active| active.set(false));
    if let Some(start) = START.with(|start| start.take()) {
        let nanos = start.elapsed().as_nanos();
        let (allocations, bytes, reallocations) = COUNTS.with(Cell::get);
        eprintln!("BPE_RECORDS {{\"len\":{len},\"capacity\":{capacity},\"element_bytes\":{size},\"allocations\":{allocations},\"bytes\":{bytes},\"reallocations\":{reallocations},\"nanos\":{nanos}}}");
    }
}
''')
with (bpe / "tests.rs").open("a") as stream:
    stream.write(r'''
#[test]
fn interface_lifecycle_probe() {
    let scenario = std::env::var("BPE_INTERFACE_PROBE").unwrap();
    let mut trainer = trainer();
    let words = if scenario == "restart" {
        trainer.continuing_subword_prefix = Some("a".into());
        trainer.special_tokens = vec![AddedToken::from("xy", true)];
        counts(&[("xy", 3), ("uv", 2), ("ab", 1)])
    } else {
        trainer.vocab_size = match scenario.as_str() { "zero" => 2, "normal" => 3, _ => 64 };
        let word = if matches!(scenario.as_str(), "normal" | "finished") {
            "ab".repeat(4096)
        } else {
            "ab".into()
        };
        counts(&[(&word, 1)])
    };
    let result = trainer.do_train_impl(WordCountsView::from_map(&words), Some(4));
    assert_eq!(result.is_err(), matches!(scenario.as_str(), "select" | "prepare" | "owner"));
}
''')
print(f"Instrumented {root}; production source remains untouched")
