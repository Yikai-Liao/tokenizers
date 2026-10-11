//! Standalone experimental counters; absent from production builds.
use std::{cell::RefCell, time::Instant};
#[derive(Default, Clone, serde::Serialize)]
pub(super) struct Storage {
    pub containers: u64, pub roots: u64, pub leaves: u64,
    pub histogram: [u64; 10], pub inline: [u64; 10],
    pub payload_bytes: u64, pub retained_heap_bytes: u64,
    pub root_array_bytes: u64, pub leaf_headers_in_roots: u64,
}
fn bin(len: usize) -> usize {
    match len { 0..=3 => len, 4..=127 => 4, 128..=255 => 5,
        256..=1023 => 6, 1024..=4095 => 7, 4096..=65535 => 8, _ => 9 }
}
impl Storage {
    pub fn leaf(&mut self, len: usize, bytes: usize, heap: usize) {
        self.leaves += 1; self.histogram[bin(len)] += 1;
        self.inline[bin(len)] += u64::from(heap == 0);
        self.payload_bytes += bytes as u64; self.retained_heap_bytes += heap as u64;
    }
}
#[derive(Default, Clone, serde::Serialize)]
struct Counters {
    retired: Storage, directory_growths: u64, directory_moved_bytes: u64,
    directory_move_nanos: u128, concat_calls: u64, concat_nanos: u128,
    reuse_calls: u64, reuse_nanos: u128, root_arrays: u64, root_children: u64,
    tasks: u64, task_items: u64,
}
thread_local! { static COUNTERS: RefCell<Counters> = RefCell::new(Counters::default()); }
pub(super) fn retired(len: usize, bytes: usize, heap: usize) {
    COUNTERS.with(|c| c.borrow_mut().retired.leaf(len, bytes, heap));
}
pub(super) fn directory(bytes: usize, nanos: u128) {
    COUNTERS.with(|c| { let mut c = c.borrow_mut(); c.directory_growths += 1;
        c.directory_moved_bytes += bytes as u64; c.directory_move_nanos += nanos; });
}
pub(super) fn root(children: usize) {
    COUNTERS.with(|c| { let mut c = c.borrow_mut(); c.root_arrays += 1; c.root_children += children as u64; });
}
pub(super) fn task(items: usize) {
    COUNTERS.with(|c| { let mut c = c.borrow_mut(); c.tasks += 1; c.task_items += items as u64; });
}
pub(super) struct Span { kind: bool, start: Instant }
impl Span { pub fn new(reuse: bool) -> Self { Self { kind: reuse, start: Instant::now() } } }
impl Drop for Span {
    fn drop(&mut self) {
        let nanos = self.start.elapsed().as_nanos();
        COUNTERS.with(|c| { let mut c = c.borrow_mut(); if self.kind { c.reuse_calls += 1; c.reuse_nanos += nanos; }
            else { c.concat_calls += 1; c.concat_nanos += nanos; } });
    }
}
pub(super) fn report() {
    let workers = rayon::broadcast(|_| COUNTERS.with(|c| c.borrow().clone()));
    eprintln!("BPE_DIAGNOSTICS {}", serde_json::to_string(&workers).unwrap());
}
