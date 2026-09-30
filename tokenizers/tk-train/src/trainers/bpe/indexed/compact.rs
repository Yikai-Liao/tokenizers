//! Ordinary BPE: endpoint corpus, single posting owner, small heap and grouped births.
//! The fused storage uses the PR's forward compaction for short pieces; the
//! selection, canonical identities and count maintenance are shared.
use super::small_posting::SmallPosting;
use super::*;

type Entries = AHashMap<u64, Entry>;

#[inline]
fn key(pair: Pair) -> u64 {
    (u64::from(pair.0) << 32) | u64::from(pair.1)
}

#[inline]
fn pair(key: u64) -> Pair {
    ((key >> 32) as u32, key as u32)
}

struct Entry {
    frequency: u64,
    positions: SmallPosting,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct HeapItem {
    frequency: u64,
    key: u64,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.key.cmp(&self.key))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct BirthGroup {
    weight: i64,
    head: u32,
    occurrences: u32,
}

struct BirthNode {
    position: u32,
    next: u32,
}

#[derive(Default)]
struct Births {
    groups: AHashMap<u64, BirthGroup>,
    nodes: Vec<BirthNode>,
}

struct Changes<'a> {
    entries: &'a mut Entries,
    births: &'a mut Births,
    replacement: u32,
    floor: u64,
    pruned: &'a mut usize,
    allocated: &'a mut usize,
}

fn posting_bytes(entry: &Entry) -> usize {
    entry.positions.allocated_capacity() * std::mem::size_of::<u32>()
}

impl Changes<'_> {
    #[inline]
    fn remove(&mut self, edge: Pair, weight: u64) -> Result<()> {
        let k = key(edge);
        if edge.0 == self.replacement || edge.1 == self.replacement {
            // A new edge is born before it can be removed. Length-excluded
            // edges are never registered, so their negative ghost is ignored.
            if let Some(group) = self.births.groups.get_mut(&k) {
                group.weight = group
                    .weight
                    .checked_sub(weight as i64)
                    .ok_or("new pair frequency underflow")?;
            }
        } else if let Some(entry) = self.entries.get_mut(&k) {
            entry.frequency = entry
                .frequency
                .checked_sub(weight)
                .ok_or("old pair frequency underflow")?;
            if entry.frequency < self.floor {
                *self.allocated -= posting_bytes(entry);
                self.entries.remove(&k);
                *self.pruned += 1;
            }
        }
        // Missing old pairs are selected, permanently retired, or length-excluded.
        Ok(())
    }

    #[inline]
    fn add(&mut self, edge: Pair, position: u32, weight: u64) -> Result<()> {
        debug_assert!(edge.0 == self.replacement || edge.1 == self.replacement);
        let node = u32::try_from(self.births.nodes.len()).map_err(|_| "birth arena exceeds u32")?;
        if node == NONE {
            return Err("birth arena index collides with sentinel".into());
        }
        let group = self.births.groups.entry(key(edge)).or_insert(BirthGroup {
            weight: 0,
            head: NONE,
            occurrences: 0,
        });
        group.weight = group
            .weight
            .checked_add(weight as i64)
            .ok_or("new pair frequency exceeds i64")?;
        group.occurrences = group
            .occurrences
            .checked_add(1)
            .ok_or("birth posting count exceeds u32")?;
        self.births.nodes.push(BirthNode {
            position,
            next: group.head,
        });
        group.head = node;
        Ok(())
    }
}

trait Storage {
    const WORD_HANDLES: bool;
    const NAME: &'static str;
    fn initial_edges(&self, visit: impl FnMut(Pair, u32, u64) -> Result<()>) -> Result<()>;
    fn rewrite(
        &mut self,
        edge: Pair,
        replacement: u32,
        positions: &mut [u32],
        lengths: &[u32],
        max_length: usize,
        changes: &mut Changes<'_>,
        stats: &mut IndexedTrainingStats,
    ) -> Result<()>;
    fn allocated_bytes(&self) -> usize;
    fn initial_slots(&self) -> usize;
    fn slot_bytes(&self) -> usize;
}

struct Endpoints {
    corpus: Vec<u32>,
    pivots: Vec<u32>,
    weights: Vec<u64>,
}

impl Storage for Endpoints {
    const WORD_HANDLES: bool = false;
    const NAME: &'static str = "endpoints_owned";

    fn initial_edges(&self, mut visit: impl FnMut(Pair, u32, u64) -> Result<()>) -> Result<()> {
        for (&start, &weight) in self.pivots.iter().zip(&self.weights) {
            let mut p = start as usize;
            while self.corpus[p] != NONE {
                if self.corpus[p + 1] != NONE {
                    visit((self.corpus[p], self.corpus[p + 1]), p as u32, weight)?;
                }
                p += 1;
            }
        }
        Ok(())
    }

    fn rewrite(
        &mut self,
        edge: Pair,
        replacement: u32,
        positions: &mut [u32],
        lengths: &[u32],
        max_length: usize,
        changes: &mut Changes<'_>,
        stats: &mut IndexedTrainingStats,
    ) -> Result<()> {
        if edge.0 == edge.1 {
            positions.sort_unstable();
        }
        let left_len = lengths[edge.0 as usize] as usize;
        let right_len = lengths[edge.1 as usize] as usize;
        let len = left_len + right_len;
        for &position in positions.iter() {
            let p = position as usize;
            if self.corpus[p] != edge.0 {
                stats.stale_posting_visits += 1;
                continue;
            }
            let right = p + left_len;
            if right >= self.corpus.len() || self.corpus[right] != edge.1 {
                stats.stale_posting_visits += 1;
                continue;
            }
            let after = right + right_len;
            debug_assert!(after < self.corpus.len());
            let word = self.pivots.partition_point(|&start| start <= position) - 1;
            let weight = self.weights[word];
            let prior = self.corpus[p - 1];
            if prior != NONE {
                let prior_len = lengths[prior as usize] as usize;
                let before = p - prior_len;
                changes.remove((prior, edge.0), weight)?;
                if prior_len + len < max_length {
                    changes.add((prior, replacement), before as u32, weight)?;
                }
            }
            let next = self.corpus[after];
            if next != NONE {
                changes.remove((edge.1, next), weight)?;
                if len + (lengths[next as usize] as usize) < max_length {
                    changes.add((replacement, next), position, weight)?;
                }
            }
            self.corpus[p] = replacement;
            if right_len == 1 {
                self.corpus[right] = replacement;
            } else {
                self.corpus[right] = NONE;
                self.corpus[after - 1] = replacement;
            }
        }
        Ok(())
    }

    fn allocated_bytes(&self) -> usize {
        self.corpus.capacity() * 4 + self.pivots.capacity() * 4 + self.weights.capacity() * 8
    }
    fn initial_slots(&self) -> usize {
        self.corpus.len()
    }
    fn slot_bytes(&self) -> usize {
        self.corpus.capacity() * 4
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    token: u32,
    len: u32,
}

struct WordArena {
    symbols: Vec<Symbol>,
    starts: Vec<u32>,
    live: Vec<u32>,
    weights: Vec<u64>,
}

impl WordArena {
    fn from_input(input: PreparedCorpus) -> Self {
        let mut symbols = Vec::with_capacity(input.corpus.len() - input.pivots.len() - 1);
        let mut starts = Vec::with_capacity(input.pivots.len());
        let mut live = Vec::with_capacity(input.pivots.len());
        for &start in &input.pivots {
            starts.push(symbols.len() as u32);
            let mut p = start as usize;
            while input.corpus[p] != NONE {
                symbols.push(Symbol {
                    token: input.corpus[p],
                    len: 1,
                });
                p += 1;
            }
            live.push(symbols.len() as u32 - starts.last().copied().unwrap());
        }
        Self {
            symbols,
            starts,
            live,
            weights: input.weights,
        }
    }
}

impl Storage for WordArena {
    const WORD_HANDLES: bool = true;
    const NAME: &'static str = "pr_word_arena";

    fn initial_edges(&self, mut visit: impl FnMut(Pair, u32, u64) -> Result<()>) -> Result<()> {
        for (word, (&start, &live)) in self.starts.iter().zip(&self.live).enumerate() {
            for window in self.symbols[start as usize..start as usize + live as usize].windows(2) {
                visit(
                    (window[0].token, window[1].token),
                    word as u32,
                    self.weights[word],
                )?;
            }
        }
        Ok(())
    }

    fn rewrite(
        &mut self,
        edge: Pair,
        replacement: u32,
        positions: &mut [u32],
        _lengths: &[u32],
        max_length: usize,
        changes: &mut Changes<'_>,
        stats: &mut IndexedTrainingStats,
    ) -> Result<()> {
        // PR #2348's read/write pass: the last written symbol is the current
        // left neighbour, and unread symbols stay intact. No tail memmove.
        for &position in positions.iter() {
            let word = position as usize;
            let start = self.starts[word] as usize;
            let n = self.live[word] as usize;
            let run = &mut self.symbols[start..start + n];
            let weight = self.weights[word];
            let mut read = 0;
            let mut write = 0;
            while read < n {
                stats.word_scan_steps += 1;
                if run[read].token == edge.0 && read + 1 < n && run[read + 1].token == edge.1 {
                    let len = run[read].len + run[read + 1].len;
                    if write > 0 {
                        let left = run[write - 1];
                        changes.remove((left.token, edge.0), weight)?;
                        if (left.len as usize) + (len as usize) < max_length {
                            changes.add((left.token, replacement), position, weight)?;
                        }
                    }
                    if read + 2 < n {
                        let right = run[read + 2];
                        changes.remove((edge.1, right.token), weight)?;
                        if (len as usize) + (right.len as usize) < max_length {
                            changes.add((replacement, right.token), position, weight)?;
                        }
                    }
                    run[write] = Symbol {
                        token: replacement,
                        len,
                    };
                    write += 1;
                    read += 2;
                } else {
                    run[write] = run[read];
                    write += 1;
                    read += 1;
                }
            }
            self.live[word] = write as u32;
        }
        Ok(())
    }

    fn allocated_bytes(&self) -> usize {
        self.symbols.capacity() * 8
            + self.starts.capacity() * 4
            + self.live.capacity() * 4
            + self.weights.capacity() * 8
    }
    fn initial_slots(&self) -> usize {
        self.symbols.len()
    }
    fn slot_bytes(&self) -> usize {
        self.symbols.capacity() * 8
    }
}

pub(super) fn train(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    fused: bool,
) -> Result<IndexedTraining> {
    let begin = Instant::now();
    let mut ids = AHashMap::with_capacity(trainer.vocab_size);
    let mut strings = Vec::with_capacity(trainer.vocab_size);
    let progress = trainer.setup_progress();
    trainer.add_special_tokens(&mut ids, &mut strings);
    trainer.compute_alphabet(wc, &mut ids, &mut strings);
    trainer.update_progress(&progress, wc.len(), "Tokenize words");
    let mut input = PreparedCorpus::tokenize(trainer, wc, &mut ids, &mut strings, &progress)?;
    for &weight in &input.weights {
        i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")?;
    }
    let initial_symbols = input.corpus.len() - input.pivots.len() - 1;
    let lengths = std::mem::take(&mut input.lengths);
    let floor = input.monotone_floor.expect("compact requires ordinary BPE");
    // Piece geometry, independent of language. Long pieces benefit from direct
    // occurrences; short pieces avoid repeated weight searches with word handles.
    let short = fused && initial_symbols <= input.pivots.len().saturating_mul(32);
    if short {
        train_storage(
            trainer,
            WordArena::from_input(input),
            ids,
            strings,
            lengths,
            floor,
            progress,
            begin,
            initial_symbols,
        )
    } else {
        let storage = Endpoints {
            corpus: input.corpus,
            pivots: input.pivots,
            weights: input.weights,
        };
        train_storage(
            trainer,
            storage,
            ids,
            strings,
            lengths,
            floor,
            progress,
            begin,
            initial_symbols,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn train_storage<S: Storage>(
    trainer: &BpeTrainer,
    mut storage: S,
    mut ids: AHashMap<CompactString, u32>,
    mut strings: Vec<CompactString>,
    mut lengths: Vec<u32>,
    floor: u64,
    progress: Option<ProgressBar>,
    begin: Instant,
    initial_symbols: usize,
) -> Result<IndexedTraining> {
    let mut entries = Entries::new();
    let mut total = 0_i64;
    let mut initial_edges = 0;
    storage.initial_edges(|edge, position, weight| {
        initial_edges += 1;
        let signed = i64::try_from(weight).map_err(|_| "indexed BPE weight exceeds i64::MAX")?;
        total = total
            .checked_add(signed)
            .ok_or("indexed BPE weighted pair counts exceed i64::MAX")?;
        let entry = entries.entry(key(edge)).or_insert_with(|| Entry {
            frequency: 0,
            positions: SmallPosting::default(),
        });
        entry.frequency += weight; // Bounded by the checked total above.
        if !S::WORD_HANDLES || entry.positions.as_slice().last().copied() != Some(position) {
            entry.positions.push(position)?;
        }
        Ok(())
    })?;
    let before = entries.len();
    entries.retain(|_, entry| entry.frequency >= floor);
    let mut stats = IndexedTrainingStats {
        initial_symbols,
        monotone_pairs: true,
        pruned_pairs: before - entries.len(),
        layout: S::NAME,
        corpus_bytes: storage.allocated_bytes() + lengths.capacity() * 4,
        ..Default::default()
    };
    let mut allocated: usize = entries.values().map(posting_bytes).sum();
    stats.posting_bytes = allocated;
    stats.initial_corpus_bytes = stats.corpus_bytes;
    stats.initial_slots = storage.initial_slots();
    stats.initial_edges = initial_edges;
    stats.initial_pairs = entries.len();
    stats.initial_blocks = 1;
    stats.initial_slot_bytes = storage.slot_bytes();
    stats.initial_length_bytes = lengths.capacity() * 4;
    stats.initial_weight_bytes = storage.allocated_bytes() - storage.slot_bytes();
    stats.initial_posting_bytes = allocated;
    stats.initial_pair_table_bytes = super::parallel::table_bytes(entries.capacity(), 32);
    let mut queue: OctonaryHeap<HeapItem> = entries
        .iter()
        .map(|(&key, entry)| HeapItem {
            frequency: entry.frequency,
            key,
        })
        .collect();
    stats.initial_heap_bytes = queue.capacity() * 16;
    trainer.finalize_progress(&progress, initial_symbols, "Count pairs");
    stats.initialize_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let begin = Instant::now();
    let max_length = trainer.max_token_length.unwrap_or(usize::MAX);
    let mut births = Births::default();
    let mut merges = Vec::new();
    #[cfg(test)]
    let mut trace = Vec::new();
    trainer.update_progress(&progress, trainer.vocab_size, "Compute merges");
    while ids.len() < trainer.vocab_size {
        let Some(mut top) = queue.pop() else {
            break;
        };
        let Some(entry) = entries.get(&top.key) else {
            continue;
        };
        if top.frequency != entry.frequency {
            top.frequency = entry.frequency;
            queue.push(top);
            continue;
        }
        let edge = pair(top.key);
        let mut token = CompactString::with_capacity(
            strings[edge.0 as usize].len() + strings[edge.1 as usize].len(),
        );
        token.push_str(&strings[edge.0 as usize]);
        token.push_str(&strings[edge.1 as usize]);
        let replacement = if let Some(&id) = ids.get(&token) {
            stats.reused_ids += 1;
            debug_assert_eq!(
                lengths[id as usize], 0,
                "ordinary BPE cannot reactivate a string"
            );
            lengths[id as usize] = lengths[edge.0 as usize] + lengths[edge.1 as usize];
            id
        } else {
            let id = u32::try_from(strings.len()).map_err(|_| "BPE vocabulary exceeds u32")?;
            if id == NONE {
                return Err("BPE token ID collides with separator".into());
            }
            lengths.push(lengths[edge.0 as usize] + lengths[edge.1 as usize]);
            strings.push(token.clone());
            ids.insert(token, id);
            id
        };
        merges.push(edge);
        #[cfg(test)]
        trace.push((edge, top.frequency, replacement));
        let mut entry = entries.remove(&top.key).unwrap();
        allocated -= posting_bytes(&entry);
        stats.posting_visits += entry.positions.len();
        let mut pruned = stats.pruned_pairs;
        let mut changes = Changes {
            entries: &mut entries,
            births: &mut births,
            replacement,
            floor,
            pruned: &mut pruned,
            allocated: &mut allocated,
        };
        storage.rewrite(
            edge,
            replacement,
            entry.positions.as_mut_slice(),
            &lengths,
            max_length,
            &mut changes,
            &mut stats,
        )?;
        stats.pruned_pairs = pruned;
        for (k, group) in births.groups.drain() {
            if group.weight < 0 || (group.weight as u64) < floor {
                stats.pruned_pairs += 1;
                continue;
            }
            let mut positions = SmallPosting::with_capacity(group.occurrences)?;
            let mut node = group.head;
            while node != NONE {
                let birth = &births.nodes[node as usize];
                // Word visits are consecutive within each rule, so identical
                // handles are consecutive even in the reversed birth chain.
                if !S::WORD_HANDLES || positions.as_slice().last().copied() != Some(birth.position)
                {
                    positions.push(birth.position)?;
                }
                node = birth.next;
            }
            let entry = Entry {
                frequency: group.weight as u64,
                positions,
            };
            debug_assert!(!entries.contains_key(&k));
            allocated += posting_bytes(&entry);
            queue.push(HeapItem {
                key: k,
                frequency: entry.frequency,
            });
            entries.insert(k, entry);
        }
        stats.posting_bytes = stats.posting_bytes.max(allocated);
        stats.peak_birth_bytes = stats
            .peak_birth_bytes
            .max(births.nodes.capacity() * std::mem::size_of::<BirthNode>());
        births.nodes.clear();
        if let Some(p) = &progress {
            p.inc(1);
        }
        trainer.emit_json_progress("Compute merges", merges.len(), trainer.vocab_size);
    }
    trainer.finalize_progress(&progress, merges.len(), "Compute merges");
    stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;
    stats.corpus_bytes = storage.allocated_bytes() + lengths.capacity() * 4;
    Ok(IndexedTraining {
        vocab: ids.into_iter().map(|(s, id)| (s.to_string(), id)).collect(),
        merges: merges
            .into_iter()
            .map(|(a, b)| {
                (
                    strings[a as usize].to_string(),
                    strings[b as usize].to_string(),
                )
            })
            .collect(),
        special_tokens: trainer.special_tokens.clone(),
        stats,
        #[cfg(test)]
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prototype_metadata_layout() {
        assert_eq!(std::mem::size_of::<HeapItem>(), 16);
        assert_eq!(std::mem::size_of::<BirthNode>(), 8);
        assert_eq!(std::mem::size_of::<Symbol>(), 8);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<Entry>(), 24);
    }
}
