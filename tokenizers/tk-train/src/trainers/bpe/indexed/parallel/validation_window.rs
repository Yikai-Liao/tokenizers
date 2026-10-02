//! Exact local prefixes certified while the owner's frequency ledger is frozen.
//! Workers fill them at the end of the existing commit task. Selection keeps
//! its original serial order and returns every unused candidate before updates.
use super::{Candidate, OctonaryHeap, Owner};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Clone, Copy)]
pub(super) enum SelectionMode {
    Serial,
    Cached,
    Leader,
    Bulk(usize),
}
impl SelectionMode {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Cached => "cached",
            Self::Leader => "leader",
            Self::Bulk(4) => "bulk4",
            Self::Bulk(16) => "bulk16",
            Self::Bulk(_) => "bulk",
        }
    }
    fn width(self) -> usize {
        match self {
            Self::Serial => 0,
            Self::Cached => 1,
            Self::Leader => 1,
            Self::Bulk(width) => width.max(1),
        }
    }
    fn uses_leaders(self) -> bool {
        matches!(self, Self::Leader | Self::Bulk(_))
    }
}

/// One upper bound per nonempty owner. Validate only the global winner, then
/// compare again if it fell. Only that owner changes during selection, giving
/// O(log owners) global maintenance without repeated full scans or truth reads.
#[derive(Default)]
pub(super) struct Frontier {
    leaders: OctonaryHeap<(Candidate, usize)>,
    pub(super) owner_probes: usize,
    pub(super) leader_updates: usize,
}
impl Frontier {
    pub(super) fn begin_epoch(&mut self, owners: &mut [Owner], mode: SelectionMode) {
        if !mode.uses_leaders() {
            return;
        }
        let mut leaders = std::mem::take(&mut self.leaders).into_vec();
        leaders.clear();
        for (o, ledger) in owners.iter_mut().enumerate() {
            self.owner_probes += 1;
            if let Some(candidate) = ledger.window_upper() {
                leaders.push((candidate, o));
            }
        }
        // Reuse backing storage and heapify all owner heads in O(owners).
        self.leaders = leaders.into();
    }
    pub(super) fn best(
        &mut self,
        owners: &mut [Owner],
        mode: SelectionMode,
    ) -> Option<(usize, Candidate)> {
        if mode.uses_leaders() {
            loop {
                let (upper, o) = self.leaders.peek().copied()?;
                self.owner_probes += 1;
                let exact = owners[o].window_top(mode);
                if exact == Some(upper) {
                    return Some((o, upper));
                }
                self.leader_updates += 1;
                if let Some(candidate) = exact {
                    debug_assert!(candidate <= upper);
                    *self.leaders.peek_mut().unwrap() = (candidate, o);
                } else {
                    self.leaders.pop();
                }
            }
        } else {
            self.owner_probes += owners.len();
            owners
                .iter_mut()
                .enumerate()
                .filter_map(|(o, ledger)| ledger.window_top(mode).map(|c| (o, c)))
                .max_by_key(|(_, c)| *c)
        }
    }
    pub(super) fn consume(&mut self, owners: &mut [Owner], o: usize, mode: SelectionMode) {
        owners[o].consume_top(mode);
        if mode.uses_leaders() {
            let (_, previous_owner) = *self.leaders.peek().expect("exact global leader");
            debug_assert_eq!(previous_owner, o);
            self.owner_probes += 1;
            self.leader_updates += 1;
            if let Some(next) = owners[o].window_upper() {
                *self.leaders.peek_mut().unwrap() = (next, o);
            } else {
                self.leaders.pop();
            }
        }
    }
}

#[derive(Default)]
pub(super) struct Window {
    ready: VecDeque<Candidate>,
    known: bool,
    pub(super) prefetched: usize,
    pub(super) restored: usize,
    pub(super) serial_refills: usize,
    pub(super) worker_ms: f64,
}

impl Owner {
    fn window_upper(&self) -> Option<Candidate> {
        self.window
            .ready
            .front()
            .copied()
            .or_else(|| self.heap.peek())
    }
    fn fill_window(&mut self, width: usize) {
        debug_assert!(self.window.ready.is_empty());
        for _ in 0..width {
            let Some(candidate) = self.peek_current() else {
                break;
            };
            self.heap.pop();
            self.window.ready.push_back(candidate);
        }
        self.window.known = true;
    }

    pub(super) fn prepare_window(&mut self, mode: SelectionMode) {
        debug_assert!(self.window.ready.is_empty());
        if let SelectionMode::Bulk(_) = mode {
            let begin = Instant::now();
            self.fill_window(mode.width());
            self.window.prefetched += self.window.ready.len();
            self.window.worker_ms += begin.elapsed().as_secs_f64() * 1000.0;
        }
    }

    pub(super) fn window_top(&mut self, mode: SelectionMode) -> Option<Candidate> {
        if matches!(mode, SelectionMode::Serial) {
            return self.peek_current();
        }
        if !self.window.known {
            self.window.serial_refills += 1;
            self.fill_window(mode.width());
        }
        self.window.ready.front().copied()
    }

    pub(super) fn consume_top(&mut self, mode: SelectionMode) {
        if matches!(mode, SelectionMode::Serial) {
            self.heap.pop();
        } else {
            self.window.ready.pop_front().expect("certified candidate");
            if self.window.ready.is_empty() {
                self.window.known = false;
            }
        }
    }

    pub(super) fn end_selection(&mut self) {
        self.window.restored += self.window.ready.len();
        for candidate in self.window.ready.drain(..) {
            self.heap.push(candidate);
        }
        self.window.known = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainers::bpe::indexed::parallel::{CandidateHeap, Entry, key};

    fn owners(packed: bool, shards: u32) -> Vec<Owner> {
        (0..shards)
            .map(|o| {
                let mut ledger = Owner::default();
                for i in 0..1000u32 {
                    let k = key(i, (i * 13 + o) % 1000);
                    let upper = u64::from((i * 31 + o) % 101 + 1);
                    // Missing records and stale, equal-frequency priorities coexist.
                    if i % 11 != 0 {
                        ledger.entries.insert(
                            k,
                            Entry {
                                frequency: upper / (u64::from(i % 3) + 1),
                                blocks: super::super::BlockPosting::default(),
                            },
                        );
                    }
                    ledger.heap.push(Candidate {
                        key: k,
                        frequency: upper,
                    });
                }
                ledger.heap = CandidateHeap::new(
                    std::iter::from_fn(|| ledger.heap.pop())
                        .collect::<Vec<_>>()
                        .into_iter(),
                    packed,
                );
                ledger
            })
            .collect()
    }

    fn run(mode: SelectionMode, packed: bool, shards: u32) -> Vec<(u64, u64)> {
        let mut ledgers = owners(packed, shards);
        let mut frontier = Frontier::default();
        let mut result = Vec::new();
        for epoch in 0..80 {
            for ledger in &mut ledgers {
                ledger.prepare_window(mode);
            }
            frontier.begin_epoch(&mut ledgers, mode);
            for n in 0..(epoch % 17 + 1) {
                let best = frontier.best(&mut ledgers, mode);
                let Some((o, candidate)) = best else { break };
                // Stop before consuming a candidate, as the real conflict rule does.
                if n > 0 && candidate.key % 7 == 0 {
                    break;
                }
                frontier.consume(&mut ledgers, o, mode);
                ledgers[o].entries.remove(&candidate.key).unwrap();
                result.push((candidate.key, candidate.frequency));
            }
            for ledger in &mut ledgers {
                ledger.end_selection();
                for (k, entry) in &mut ledger.entries {
                    if k % 19 == epoch % 19 {
                        entry.frequency /= 2;
                    }
                }
                ledger.entries.retain(|k, _| k % 23 != epoch % 23);
                let k = key(2000 + epoch as u32, 3000);
                ledger.entries.insert(
                    k,
                    Entry {
                        frequency: 100,
                        blocks: super::super::BlockPosting::default(),
                    },
                );
                ledger.heap.push(Candidate {
                    key: k,
                    frequency: 100,
                });
            }
        }
        result
    }

    #[test]
    fn prefixes_preserve_order_across_stops_stale_decreases_deletions_and_births() {
        for shards in [1, 4, 32, 64] {
            for packed in [false, true] {
                let expected = run(SelectionMode::Serial, packed, shards);
                for mode in [
                    SelectionMode::Cached,
                    SelectionMode::Leader,
                    SelectionMode::Bulk(4),
                    SelectionMode::Bulk(16),
                ] {
                    assert_eq!(run(mode, packed, shards), expected);
                }
            }
        }
    }
}
