//! Authoritative weighted counts and owning historical word-domain candidates.
use super::corpus::{CorpusPlan, InitialChunk, InitialEdge, WordChange};
use super::positions::Positions;
use crate::progress::ProgressBar;
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use tk_encode::Result;
use tk_encode::models::bpe::Pair;
use tk_encode::parallelism::MaybeParallelIterator;

pub(super) struct Candidate {
    pub(super) pair: Pair,
    count: u64,
    pub(super) positions: Positions,
}

impl Eq for Candidate {}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

pub(super) struct PairIndex {
    counts: AHashMap<Pair, u64>,
    queue: OctonaryHeap<Candidate>,
    minimum: u64,
}

impl PairIndex {
    pub(super) fn build(
        corpus: &CorpusPlan<'_>,
        minimum: u64,
        progress: &Option<ProgressBar>,
    ) -> Result<Self> {
        let slots = if cfg!(test) {
            16
        } else {
            corpus
                .slot_count()
                .div_ceil(super::num_workers())
                .clamp(1, 1 << 24)
        };
        let pieces = corpus
            .chunks(slots)
            .into_maybe_par_iter()
            .map(|chunk| count_chunk(corpus.token_count(), chunk, progress))
            .collect::<Result<Vec<_>>>()?;
        let mut states = AHashMap::<Pair, PairAggregate<SmallVec<[Positions; 1]>>>::new();
        for piece in pieces {
            for (pair, state) in piece {
                let total = states.entry(pair).or_default();
                super::add(&mut total.count, state.count)?;
                total.positions.push(state.positions);
            }
        }
        let mut counts = AHashMap::new();
        let mut queue = OctonaryHeap::with_capacity(states.len());
        for (pair, state) in states {
            counts.insert(pair, state.count);
            if state.count != 0 {
                let positions = Positions::concat(state.positions)?;
                queue.push(Candidate::new(pair, state.count, positions));
            }
        }
        Ok(Self {
            counts,
            queue,
            minimum,
        })
    }

    pub(super) fn take(&mut self) -> Option<Candidate> {
        loop {
            let mut top = self.queue.pop()?;
            if top.count != self.counts[&top.pair] as u64 {
                top.count = self.counts[&top.pair] as u64;
                self.queue.push(top);
                continue;
            }
            return (top.count >= self.minimum.max(1)).then_some(top);
        }
    }

    pub(super) fn commit(&mut self, changes: Vec<WordChange>) -> Result<()> {
        let mut births: AHashMap<Pair, Vec<u64>> = AHashMap::new();
        for change in changes {
            adjust_signed(
                self.counts.entry(change.pair).or_default(),
                change.weight,
                change.delta < 0,
            )?;
            if change.delta > 0 {
                births
                    .entry(change.pair)
                    .or_default()
                    .push(change.coordinate);
            }
        }
        for (pair, pos) in births.drain() {
            let count = self.counts[&pair];
            if (count as i64) > 0 {
                self.queue.push(Candidate::new(
                    pair,
                    count as u64,
                    Positions::from_unsorted(pos)?,
                ));
            }
        }
        Ok(())
    }
}

impl Candidate {
    fn new(pair: Pair, count: u64, positions: Positions) -> Self {
        // Keep the owning cohort; producers have already normalized its coordinates.
        Self {
            pair,
            count,
            positions,
        }
    }
}

fn adjust_signed(count: &mut u64, amount: u64, remove: bool) -> Result<()> {
    // Reuse keeps a signed ledger in u64 bits to preserve the queue's unsigned
    // ordering. Check each action, not its net delta: intermediate overflow matters.
    let amount = i64::try_from(amount).map_err(|_| "BPE identity-reuse adjustment exceeds i64")?;
    let delta = if remove { -amount } else { amount };
    *count = (*count as i64)
        .checked_add(delta)
        .ok_or("BPE identity-reuse count adjustment exceeds i64")? as u64;
    Ok(())
}

#[derive(Default)]
struct PairAggregate<P> {
    count: u64,
    positions: P,
}
impl PairAggregate<Positions> {
    #[inline(always)]
    fn record(&mut self, edge: InitialEdge) -> Result<()> {
        debug_assert!(edge.pair.0 != u32::MAX && edge.pair.1 != u32::MAX);
        super::add(&mut self.count, edge.weight)?;
        // Keep checked count addition before appending this ordered coordinate.
        self.positions.push(edge.coordinate)
    }
}

type InitialPiece = Vec<(Pair, PairAggregate<Positions>)>;
enum InitialPairCounts {
    Dense {
        domain: usize,
        states: Vec<PairAggregate<Positions>>,
    },
    Sparse(AHashMap<Pair, PairAggregate<Positions>>),
}

impl InitialPairCounts {
    fn new(domain: usize) -> Self {
        // The dense plane has one cell for every ordered pair: domain squared.
        // A 256-ID cutoff limits it to 65,536 cells; larger alphabets stay sparse.
        if domain <= 256 {
            Self::Dense {
                domain,
                states: (0..domain * domain)
                    .map(|_| PairAggregate::default())
                    .collect(),
            }
        } else {
            Self::Sparse(AHashMap::new())
        }
    }

    fn scan(&mut self, chunk: InitialChunk<'_, '_>) -> Result<usize> {
        // Pick the representation once; each visitor borrows only concrete storage.
        match self {
            Self::Dense { domain, states } => {
                let domain = *domain;
                let states = states.as_mut_slice();
                chunk.scan(
                    #[inline(always)]
                    |edge| {
                        // The IDs choose one row and column in this fixed dense plane.
                        states[edge.pair.0 as usize * domain + edge.pair.1 as usize].record(edge)
                    },
                )
            }
            Self::Sparse(states) => chunk.scan(
                #[inline(always)]
                |edge| states.entry(edge.pair).or_default().record(edge),
            ),
        }
    }

    fn into_entries(self) -> Vec<(Pair, PairAggregate<Positions>)> {
        // Invert the dense row-major key: quotient is the left ID, remainder
        // is the right ID. Sparse keys already carry their ordered pair.
        match self {
            Self::Dense { domain, states } => states
                .into_iter()
                .enumerate()
                .filter(|(_, state)| !state.positions.is_empty())
                .map(|(key, state)| (((key / domain) as u32, (key % domain) as u32), state))
                .collect(),
            Self::Sparse(states) => states.into_iter().collect(),
        }
    }
}
fn count_chunk(
    domain: usize,
    chunk: InitialChunk<'_, '_>,
    progress: &Option<ProgressBar>,
) -> Result<InitialPiece> {
    let mut counts = InitialPairCounts::new(domain);
    let words = counts.scan(chunk)?;
    if let Some(p) = progress {
        p.inc(words as u64);
    }
    Ok(counts.into_entries())
}
