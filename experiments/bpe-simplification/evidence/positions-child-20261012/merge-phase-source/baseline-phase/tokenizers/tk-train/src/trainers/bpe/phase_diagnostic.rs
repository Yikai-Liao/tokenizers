use std::cell::Cell;
use std::time::Instant;
thread_local! { static TIMES: Cell<([f64; 4], [u64; 4])> = const { Cell::new(([0.; 4], [0; 4])) }; }
pub(super) struct Span { slot: usize, started: Instant }
impl Span { pub(super) fn new(slot: usize) -> Self { Self { slot, started: Instant::now() } } }
impl Drop for Span { fn drop(&mut self) { TIMES.with(|times| { let (mut sums, mut counts) = times.get(); sums[self.slot] += self.started.elapsed().as_secs_f64(); counts[self.slot] += 1; times.set((sums, counts)); }); } }
pub(super) fn report() { TIMES.with(|times| { let (sums, counts) = times.get(); eprintln!("BPE_MERGE_PHASE_SECONDS {:?}", sums); eprintln!("BPE_MERGE_PHASE_CALLS {:?}", counts); }); }
