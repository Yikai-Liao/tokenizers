//! Word-local run nodes and lazy postings, ported from YTTM's worker.
//! Mechanism: f4162d846057a3118222ca04a01b84297eb8a8db, cpp/bpe.cpp.
//! Canonical adaptation: AA selection mass is (run length - 1), not length / 2.
//! The counters also retain upstream's selected-pair credits and length-filter
//! events; they are queue scores, not a reconstruction of live adjacency.
use super::Rule;
use super::pair_key;
use ahash::{AHashMap, AHashSet};
use tk_encode::Result;

#[cfg(test)]
type WorkerHook = std::sync::Arc<dyn Fn(usize, usize, usize, bool) + Send + Sync>;

const NONE: u32 = u32::MAX;
#[derive(Clone, Copy)]
struct Node {
    val: u32,
    prev: u32,
    next: u32,
    len: u32,
}
impl Node {
    fn dead() -> Self {
        Self {
            val: 0,
            prev: NONE,
            next: NONE,
            len: 0,
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct Position {
    word: u64,
    node: u64,
}
struct Word {
    nodes: Vec<Node>,
    weight: u64,
}

pub(super) struct Store {
    #[cfg(test)]
    pub(super) hook: Option<WorkerHook>,
    words: Vec<Word>,
    pub(super) max_length: Option<usize>,
    widths: Option<Vec<Vec<usize>>>,
    pub(super) counts: AHashMap<u64, i128>,
    postings: AHashMap<u64, Vec<Position>>,
    left_births: AHashSet<u32>,
    right_births: AHashSet<u32>,
    birth_words: AHashMap<u64, AHashSet<u32>>,
    track_words: bool,
    compressed: bool,
    replacement: Option<u32>,
}
impl Store {
    pub(super) fn build(words: &[(Vec<u32>, u64)], max_length: Option<usize>) -> Result<Self> {
        let mut store = Self {
            #[cfg(test)]
            hook: None,
            words: Vec::new(),
            max_length,
            widths: max_length.map(|_| Vec::new()),
            counts: AHashMap::new(),
            postings: AHashMap::new(),
            left_births: AHashSet::new(),
            right_births: AHashSet::new(),
            birth_words: AHashMap::new(),
            track_words: false,
            compressed: max_length.is_none(),
            replacement: None,
        };
        for (tokens, weight) in words {
            let weight = *weight;
            let word = u32::try_from(store.words.len())
                .map_err(|_| "YTTM local word index exceeds u32")?;
            let mut nodes: Vec<Node> = Vec::new();
            for &val in tokens {
                if let Some(last) = nodes.last_mut()
                    && last.val == val
                    && max_length.is_none()
                {
                    last.len = last
                        .len
                        .checked_add(1)
                        .ok_or("YTTM run length exceeds u32")?;
                } else {
                    let index = u32::try_from(nodes.len())
                        .map_err(|_| "YTTM local node index exceeds u32")?;
                    if index == NONE {
                        return Err("YTTM local node index reserves u32::MAX".into());
                    }
                    if let Some(last) = nodes.last_mut() {
                        last.next = index;
                    }
                    nodes.push(Node {
                        val,
                        prev: index.checked_sub(1).unwrap_or(NONE),
                        next: NONE,
                        len: 1,
                    });
                }
            }
            if let Some(widths) = &mut store.widths {
                widths.push(vec![1; nodes.len()]);
            }
            store.words.push(Word { nodes, weight });
            for node in 0..store.words[word as usize].nodes.len() {
                let node = node as u32;
                store.add_boundary(word, node)?;
                if store.node(word, node).len > 1 {
                    store.add_self(word, node)?;
                }
            }
        }
        Ok(store)
    }
    pub(super) fn enable_coverage(&mut self) -> AHashMap<u64, AHashSet<u32>> {
        self.track_words = true;
        self.postings
            .iter()
            .map(|(&key, positions)| {
                (
                    key,
                    positions
                        .iter()
                        .map(|position| position.word as u32)
                        .collect(),
                )
            })
            .collect()
    }
    // A reused ID may place equal symbols next to each other with different
    // historical spans. Expand only then, keeping YTTM's linked nodes and lazy
    // postings while reproducing Tokenizers' left-to-right occurrence updates.
    fn expand_runs(&mut self) -> Result<()> {
        for word in &mut self.words {
            let mut nodes: Vec<Node> = Vec::new();
            let mut current = if word.nodes.is_empty() { NONE } else { 0 };
            while current != NONE {
                let node = word.nodes[current as usize];
                for _ in 0..node.len {
                    let index = u32::try_from(nodes.len())
                        .map_err(|_| "YTTM local node index exceeds u32")?;
                    if index == NONE {
                        return Err("YTTM local node index reserves u32::MAX".into());
                    }
                    if let Some(last) = nodes.last_mut() {
                        last.next = index;
                    }
                    nodes.push(Node {
                        val: node.val,
                        prev: index.checked_sub(1).unwrap_or(NONE),
                        next: NONE,
                        len: 1,
                    });
                }
                current = node.next;
            }
            word.nodes = nodes;
        }
        self.postings.clear();
        for (word, value) in self.words.iter().enumerate() {
            for (node, value) in value.nodes.iter().enumerate() {
                if value.next != NONE {
                    let right = self.words[word].nodes[value.next as usize].val;
                    self.postings
                        .entry(pair_key((value.val, right)))
                        .or_default()
                        .push(Position {
                            word: word as u64,
                            node: node as u64,
                        });
                }
            }
        }
        self.compressed = false;
        Ok(())
    }
    fn record_birth(&mut self, key: u64, word: u32) {
        let (left, right) = super::key_pair(key);
        let replacement = self.replacement.expect("a birth has a replacement");
        if right == replacement {
            self.left_births.insert(left);
        } else {
            debug_assert_eq!(left, replacement);
            self.right_births.insert(right);
        }
        if self.track_words {
            self.birth_words.entry(key).or_default().insert(word);
        }
    }
    pub(super) fn word_count(&self) -> usize {
        self.words.len()
    }
    pub(super) fn release_local_storage(&mut self) {
        // C++ destroys worker-local nodes, postings and birth sets on their
        // owner thread. Count tables remain available for coordinator cleanup.
        drop(std::mem::take(&mut self.words));
        drop(self.widths.take());
        drop(std::mem::take(&mut self.postings));
        drop(std::mem::take(&mut self.left_births));
        drop(std::mem::take(&mut self.right_births));
        drop(std::mem::take(&mut self.birth_words));
    }
    fn width(&self, word: u32, node: u32) -> usize {
        self.widths
            .as_ref()
            .map_or(1, |widths| widths[word as usize][node as usize])
    }
    fn node(&self, word: u32, node: u32) -> Node {
        self.words[word as usize].nodes[node as usize]
    }
    fn set(&mut self, word: u32, node: u32, value: Node) {
        self.words[word as usize].nodes[node as usize] = value;
    }
    fn append(&mut self, word: u32, node: Node) -> Result<u32> {
        let nodes = &mut self.words[word as usize].nodes;
        let index = u32::try_from(nodes.len()).map_err(|_| "YTTM local node index exceeds u32")?;
        if index == NONE {
            return Err("YTTM local node index reserves u32::MAX".into());
        }
        nodes.push(node);
        Ok(index)
    }
    fn add(&mut self, key: u64, amount: i128) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        let value = self.counts.entry(key).or_default();
        *value = value
            .checked_add(amount)
            .ok_or("YTTM local score exceeds i128")?;
        Ok(())
    }
    fn subtract(&mut self, key: u64, amount: i128) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        let value = self.counts.entry(key).or_default();
        *value = value
            .checked_sub(amount)
            .ok_or("YTTM local score exceeds i128")?;
        Ok(())
    }

    fn boundary_key(&self, word: u32, position: u32) -> u64 {
        let node = self.node(word, position);
        pair_key((node.val, self.node(word, node.next).val))
    }
    fn remove_boundary(&mut self, word: u32, position: u32) -> Result<()> {
        self.subtract(
            self.boundary_key(word, position),
            i128::from(self.words[word as usize].weight),
        )
    }
    fn born(&self, key: u64) -> bool {
        let pair = super::key_pair(key);
        self.replacement.is_some_and(|z| pair.0 == z || pair.1 == z)
    }
    fn add_boundary(&mut self, word: u32, position: u32) -> Result<()> {
        if self.node(word, position).next == NONE {
            return Ok(());
        }
        let key = self.boundary_key(word, position);
        self.postings.entry(key).or_default().push(Position {
            word: u64::from(word),
            node: u64::from(position),
        });
        if !self.born(key)
            || self.max_length.is_none_or(|max| {
                self.width(word, position) + self.width(word, self.node(word, position).next) < max
            })
        {
            if self.born(key) {
                self.record_birth(key, word);
            }
            self.add(key, i128::from(self.words[word as usize].weight))
        } else {
            Ok(())
        }
    }
    fn add_empty_boundary(&mut self, word: u32, position: u32) {
        let key = self.boundary_key(word, position);
        self.postings.entry(key).or_default().push(Position {
            word: u64::from(word),
            node: u64::from(position),
        });
    }
    fn self_mass(&self, word: u32, position: u32) -> Result<i128> {
        i128::from(self.words[word as usize].weight)
            .checked_mul(i128::from(self.node(word, position).len - 1))
            .ok_or_else(|| "YTTM run frequency exceeds i128".into())
    }
    fn add_self(&mut self, word: u32, position: u32) -> Result<()> {
        let node = self.node(word, position);
        let key = pair_key((node.val, node.val));
        self.postings.entry(key).or_default().push(Position {
            word: u64::from(word),
            node: u64::from(position),
        });
        if !self.born(key)
            || self
                .max_length
                .is_none_or(|max| self.width(word, position) * 2 < max)
        {
            if self.born(key) {
                self.record_birth(key, word);
            }
            self.add(key, self.self_mass(word, position)?)
        } else {
            Ok(())
        }
    }
    fn decrement(&mut self, word: u32, position: u32) -> Result<()> {
        let mut node = self.node(word, position);
        debug_assert!(node.len >= 2);
        // Overlapping AA mass loses one occurrence for every removed endpoint.
        self.subtract(
            pair_key((node.val, node.val)),
            i128::from(self.words[word as usize].weight),
        )?;
        node.len -= 1;
        self.set(word, position, node);
        Ok(())
    }
    fn try_merge(&mut self, word: u32, left: u32, right: u32) -> Result<()> {
        if !self.compressed {
            return Ok(());
        }
        let mut a = self.node(word, left);
        let b = self.node(word, right);
        if a.val != b.val {
            return Ok(());
        }
        debug_assert!(a.len != 0 && b.len != 0);
        a.len = a
            .len
            .checked_add(b.len)
            .ok_or("YTTM run length exceeds u32")?;
        // Internal AA masses plus the already-added cross-node AA boundary
        // exactly equal the merged run mass; no parity compensation is needed.
        a.next = b.next;
        self.set(word, left, a);
        self.set(word, right, Node::dead());
        if a.next != NONE {
            self.words[word as usize].nodes[a.next as usize].prev = left;
            self.add_empty_boundary(word, left);
        }
        Ok(())
    }
    pub(super) fn begin_rule(&mut self, rule: &Rule, owner: usize) -> Result<Vec<Position>> {
        if rule.reserved && self.compressed {
            self.expand_runs()?;
        }
        self.left_births.clear();
        self.right_births.clear();
        self.replacement = Some(rule.replacement);
        let key = pair_key(rule.pair);
        // Own this occurrence list while appending new postings. A reused ID
        // can recreate this key; those newborn positions await the next task.
        let mut positions = self.postings.remove(&key).unwrap_or_default();
        if let Some(witness) = &rule.words {
            let mut retained = Vec::new();
            positions.retain(|position| {
                if witness[owner].contains(&(position.word as u32)) {
                    true
                } else {
                    retained.push(*position);
                    false
                }
            });
            if !retained.is_empty() {
                self.postings.insert(key, retained);
            }
        }
        if rule.pair.0 == rule.pair.1 && !self.compressed {
            positions.sort_unstable_by_key(|position| (position.word, position.node));
            positions.dedup_by_key(|position| (position.word, position.node));
        }
        Ok(positions)
    }
    pub(super) fn apply_position(&mut self, rule: &Rule, position: Position) -> Result<()> {
        let key = pair_key(rule.pair);
        let (x, y) = rule.pair;
        let z = rule.replacement;
        // All positions originate from checked u32 local node indices.
        let word = position.word as u32;
        let p1 = position.node as u32;
        let a = self.node(word, p1);
        if a.len == 0 || a.val != x {
            return Ok(());
        }
        if x == y && self.compressed {
            if a.len < 2 {
                return Ok(());
            }
            // Keep each applied occurrence's upstream credit in the selected
            // score. Combine it with the removed AA mass in one table update.
            self.subtract(
                key,
                i128::from(self.words[word as usize].weight) * i128::from(a.len - 1 - a.len / 2),
            )?;
            let p0 = a.prev;
            let p3 = a.next;
            if p0 != NONE {
                self.remove_boundary(word, p0)?;
            }
            if p3 != NONE {
                self.remove_boundary(word, p1)?;
            }
            if a.len.is_multiple_of(2) {
                self.set(
                    word,
                    p1,
                    Node {
                        val: z,
                        prev: p0,
                        next: p3,
                        len: a.len / 2,
                    },
                );
                if p0 != NONE {
                    self.add_boundary(word, p0)?;
                }
                if p3 != NONE {
                    self.add_boundary(word, p1)?;
                }
            } else {
                let p2 = self.append(
                    word,
                    Node {
                        val: x,
                        prev: p1,
                        next: p3,
                        len: 1,
                    },
                )?;
                self.set(
                    word,
                    p1,
                    Node {
                        val: z,
                        prev: p0,
                        next: p2,
                        len: a.len / 2,
                    },
                );
                if p0 != NONE {
                    self.add_boundary(word, p0)?;
                }
                self.add_boundary(word, p1)?;
                if p3 != NONE {
                    self.words[word as usize].nodes[p3 as usize].prev = p2;
                    self.add_boundary(word, p2)?;
                }
            }
            if a.len / 2 >= 2 {
                self.add_self(word, p1)?;
            }
            return Ok(());
        }
        let p2 = a.next;
        if p2 == NONE {
            return Ok(());
        }
        let b = self.node(word, p2);
        if b.len == 0 || b.val != y {
            return Ok(());
        }
        let p0 = a.prev;
        let p3 = b.next;
        // The selected boundary's removal and upstream occurrence credit
        // cancel exactly. Its score stays unchanged; update only neighbors.
        if p0 != NONE && a.len == 1 {
            self.remove_boundary(word, p0)?;
        }
        if p3 != NONE && b.len == 1 {
            self.remove_boundary(word, p2)?;
        }
        match (a.len > 1, b.len > 1) {
            (true, true) => {
                let middle = self.append(
                    word,
                    Node {
                        val: z,
                        prev: p1,
                        next: p2,
                        len: 1,
                    },
                )?;
                self.decrement(word, p1)?;
                self.decrement(word, p2)?;
                self.words[word as usize].nodes[p1 as usize].next = middle;
                self.words[word as usize].nodes[p2 as usize].prev = middle;
                self.add_boundary(word, p1)?;
                self.add_boundary(word, middle)?;
            }
            (true, false) => {
                self.set(
                    word,
                    p2,
                    Node {
                        val: z,
                        prev: p1,
                        next: p3,
                        len: 1,
                    },
                );
                self.decrement(word, p1)?;
                self.add_boundary(word, p1)?;
                if p3 != NONE {
                    self.add_boundary(word, p2)?;
                    self.try_merge(word, p2, p3)?;
                }
            }
            (false, true) => {
                self.set(
                    word,
                    p1,
                    Node {
                        val: z,
                        prev: p0,
                        next: p2,
                        len: 1,
                    },
                );
                self.decrement(word, p2)?;
                if p0 != NONE {
                    self.add_boundary(word, p0)?;
                }
                self.add_boundary(word, p1)?;
                if p0 != NONE {
                    self.try_merge(word, p0, p1)?;
                }
            }
            (false, false) => {
                if self.max_length.is_some() {
                    let width = self
                        .width(word, p1)
                        .checked_add(self.width(word, p2))
                        .ok_or("BPE symbol length overflow")?;
                    self.widths
                        .as_mut()
                        .expect("length-limited words have widths")[word as usize]
                        [p1 as usize] = width;
                }
                self.set(
                    word,
                    p1,
                    Node {
                        val: z,
                        prev: p0,
                        next: p3,
                        len: 1,
                    },
                );
                self.set(word, p2, Node::dead());
                if p3 != NONE {
                    self.words[word as usize].nodes[p3 as usize].prev = p1;
                }
                if p0 != NONE {
                    self.add_boundary(word, p0)?;
                }
                if p3 != NONE {
                    self.add_boundary(word, p1)?;
                }
                let survivor = if p0 != NONE && self.node(word, p0).val == z {
                    self.try_merge(word, p0, p1)?;
                    p0
                } else {
                    p1
                };
                if p3 != NONE {
                    self.try_merge(word, survivor, p3)?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn exchange_birth_words(&mut self, words: &mut AHashMap<u64, AHashSet<u32>>) {
        std::mem::swap(&mut self.birth_words, words);
    }
    pub(super) fn birth_scores_into(
        &self,
        left: &mut AHashMap<u32, i128>,
        right: &mut AHashMap<u32, i128>,
    ) {
        let replacement = self.replacement.expect("completed rule has a replacement");
        left.clear();
        right.clear();
        for &token in &self.left_births {
            left.insert(
                token,
                self.counts
                    .get(&pair_key((token, replacement)))
                    .copied()
                    .unwrap_or(0),
            );
        }
        for &token in &self.right_births {
            right.insert(
                token,
                self.counts
                    .get(&pair_key((replacement, token)))
                    .copied()
                    .unwrap_or(0),
            );
        }
    }
}
