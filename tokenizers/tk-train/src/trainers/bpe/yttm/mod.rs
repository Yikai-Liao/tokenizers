//! Rust port of YTTM's run lists, local postings and two-slot worker pipeline.
//! Source: VKCOM/YouTokenToMe f4162d846057a3118222ca04a01b84297eb8a8db (MIT).
//! Tokenizers compatibility: overlapping self-pair counts, canonical ties,
//! affixes and reusable vocabulary identities. Each task contains ONE rule.
mod local;
use super::{BpeTrainer, Pair};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use local::Store;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};
use tk_encode::{Result, utils::progress::ProgressBar};

fn pair_key(pair: Pair) -> u64 {
    (u64::from(pair.0) << 32) | u64::from(pair.1)
}
fn queue_score(counts: &AHashMap<u64, i128>, key: u64) -> u64 {
    // Upstream refreshes a signed count with `as u64`; length filtering can
    // produce negative counts. Preserve that cast for exact merge-list parity.
    counts.get(&key).copied().unwrap_or(0) as u64
}
fn key_pair(key: u64) -> Pair {
    ((key >> 32) as u32, key as u32)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PairPriority {
    key: u64,
    priority_count: u64,
}
impl Ord for PairPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority_count
            .cmp(&other.priority_count)
            .then_with(|| other.key.cmp(&self.key))
    }
}
impl PartialOrd for PairPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
#[derive(Clone, Debug)]
struct Rule {
    pair: Pair,
    replacement: u32,
    reserved: bool,
    words: Option<WordWitness>,
}

pub(super) fn worker_count() -> usize {
    if tk_encode::parallelism::get_parallelism() {
        tk_encode::parallelism::num_threads().max(1)
    } else {
        1
    }
}
struct Task {
    sequence: usize,
    rule: Rule,
}
struct Payload {
    owner: usize,
    sequence: usize,
    deltas: Vec<(u64, i128)>,
    births: AHashSet<u64>,
    birth_words: AHashMap<u64, AHashSet<u32>>,
}
enum Completion {
    Done(Payload),
    Failed(String),
    Witnesses(usize, AHashMap<u64, AHashSet<u32>>),
}
enum Command {
    Execute(Task),
    Witnesses,
    Stop,
}
enum Location {
    Big(usize),
    Small(usize),
    Covered,
}

fn conflicts(candidate: Pair, active: &[Rule]) -> bool {
    !active.is_empty()
        && active.iter().any(|rule| {
            rule.pair.0 == rule.pair.1
                || rule.reserved
                || candidate.0 == rule.pair.1
                || candidate.1 == rule.pair.0
        })
}

/// YTTM's sqrt split: scan high-count candidates; bucket low counts. Each
/// bucket uses canonical priority order instead of nondeterministic LIFO.
struct HighLow {
    threshold: usize,
    big: Vec<PairPriority>,
    small: Vec<OctonaryHeap<PairPriority>>,
    maximum: usize,
}
impl HighLow {
    fn new(threshold: usize, counts: &AHashMap<u64, i128>) -> Self {
        let mut queue = Self {
            threshold,
            big: Vec::new(),
            small: Vec::new(),
            maximum: 0,
        };
        for (&key, &priority_count) in counts {
            queue.push(PairPriority {
                key,
                priority_count: u64::try_from(priority_count)
                    .map_err(|_| "YTTM initial frequency exceeds u64")
                    .expect("validated initial frequency"),
            });
        }
        queue
    }
    fn push(&mut self, value: PairPriority) {
        if value.priority_count == 0 {
            return;
        }
        if value.priority_count >= self.threshold as u64 {
            self.big.push(value);
        } else {
            let index = value.priority_count as usize;
            self.small
                .resize_with((index + 1).max(self.small.len()), OctonaryHeap::new);
            self.small[index].push(value);
            self.maximum = self.maximum.max(index);
        }
    }
    fn top(
        &mut self,
        counts: &AHashMap<u64, i128>,
        active: &[Rule],
        floor: u64,
    ) -> Option<(PairPriority, Location)> {
        let mut index = 0;
        while index < self.big.len() {
            let pair = key_pair(self.big[index].key);
            // An in-flight boundary remains an upper-bound witness. Do not
            // lower it before the corresponding complete births are published.
            if !conflicts(pair, active) {
                self.big[index].priority_count = queue_score(counts, self.big[index].key);
            }
            if self.big[index].priority_count < self.threshold as u64 {
                let value = self.big.swap_remove(index);
                if value.priority_count >= floor {
                    self.push(value);
                }
            } else {
                index += 1;
            }
        }
        if let Some((index, value)) = self.big.iter().enumerate().max_by_key(|(_, value)| **value) {
            return Some((*value, Location::Big(index)));
        }
        loop {
            while self.maximum != 0
                && self
                    .small
                    .get(self.maximum)
                    .is_none_or(|bucket| bucket.is_empty())
            {
                self.maximum -= 1;
            }
            let value = self.small.get(self.maximum)?.peek().copied()?;
            if conflicts(key_pair(value.key), active) {
                return Some((value, Location::Small(self.maximum)));
            }
            let count = queue_score(counts, value.key);
            if value.priority_count == count && count >= floor {
                return Some((value, Location::Small(self.maximum)));
            }
            self.small[self.maximum].pop();
            if count >= floor {
                self.push(PairPriority {
                    key: value.key,
                    priority_count: count,
                });
                if count >= self.threshold as u64 {
                    return self.top(counts, active, floor);
                }
            }
        }
    }
    fn pop(&mut self, location: Location) {
        match location {
            Location::Big(index) => {
                self.big.swap_remove(index);
            }
            Location::Small(index) => {
                self.small[index].pop();
            }
            Location::Covered => unreachable!("plain queue has no covered head"),
        }
    }
}

type WordWitness = Arc<[AHashSet<u32>]>;
#[derive(Clone)]
struct CoveredPriority {
    priority: PairPriority,
    words: WordWitness,
}
impl PartialEq for CoveredPriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for CoveredPriority {}
impl PartialOrd for CoveredPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CoveredPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}
enum Queue {
    Plain(HighLow),
    // Reused identities require upstream's lazy, per-witness refresh order.
    // The corpus and workers continue to use the YTTM port.
    Covered(OctonaryHeap<CoveredPriority>),
}
impl Queue {
    fn new(threshold: usize, counts: &AHashMap<u64, i128>) -> Self {
        Self::Plain(HighLow::new(threshold, counts))
    }
    fn top(
        &mut self,
        counts: &AHashMap<u64, i128>,
        active: &[Rule],
        floor: u64,
    ) -> Option<(PairPriority, Location)> {
        match self {
            Self::Plain(queue) => queue.top(counts, active, 1),
            Self::Covered(queue) => loop {
                let value = queue.peek()?;
                let priority = value.priority;
                if conflicts(key_pair(priority.key), active) {
                    return Some((priority, Location::Covered));
                }
                let count = queue_score(counts, priority.key);
                if count == priority.priority_count {
                    return (count >= floor).then_some((priority, Location::Covered));
                }
                let mut value = queue.pop().expect("checked covered head");
                value.priority.priority_count = count;
                queue.push(value);
            },
        }
    }
    fn pop(&mut self, location: Location) -> Option<WordWitness> {
        match self {
            Self::Plain(queue) => {
                queue.pop(location);
                None
            }
            Self::Covered(queue) => Some(queue.pop().expect("checked covered head").words),
        }
    }
    fn covered(&self) -> bool {
        matches!(self, Self::Covered(_))
    }
    fn cover(&mut self, witnesses: &[AHashMap<u64, AHashSet<u32>>]) {
        let old = std::mem::replace(self, Self::Plain(HighLow::new(1, &AHashMap::new())));
        *self = match old {
            Self::Covered(queue) => Self::Covered(queue),
            Self::Plain(queue) => {
                let cover = |priority: PairPriority| CoveredPriority {
                    words: witnesses
                        .iter()
                        .map(|words| words.get(&priority.key).cloned().unwrap_or_default())
                        .collect::<Vec<_>>()
                        .into(),
                    priority,
                };
                Self::Covered(
                    queue
                        .big
                        .into_iter()
                        .chain(queue.small.into_iter().flatten())
                        .map(cover)
                        .collect(),
                )
            }
        };
    }
    fn push_birth(&mut self, priority: PairPriority, words: Option<Vec<AHashSet<u32>>>) {
        match self {
            Self::Plain(queue) => queue.push(priority),
            Self::Covered(queue) => queue.push(CoveredPriority {
                priority,
                words: words.expect("covered queue has word witnesses").into(),
            }),
        }
    }
}

struct Frontier {
    counts: AHashMap<u64, i128>,
    queue: Queue,
    floor: u64,
}
impl Frontier {
    fn new(stores: &[Store], floor: u64, threshold: usize) -> Result<Self> {
        let mut counts: AHashMap<u64, i128> = AHashMap::new();
        for store in stores {
            for (&key, &amount) in &store.counts {
                let value = counts.entry(key).or_default();
                *value = value
                    .checked_add(amount)
                    .ok_or("YTTM pair frequency exceeds u64")?;
            }
        }
        if counts.values().any(|&value| value > i128::from(u64::MAX)) {
            return Err("YTTM initial frequency exceeds u64".into());
        }
        let queue = Queue::new(threshold, &counts);
        Ok(Self {
            counts,
            queue,
            floor,
        })
    }
    fn publish(
        &mut self,
        deltas: AHashMap<u64, i128>,
        births: AHashSet<u64>,
        mut birth_words: AHashMap<u64, Vec<AHashSet<u32>>>,
    ) -> Result<()> {
        for (key, delta) in deltas {
            let value = self.counts.entry(key).or_default();
            *value = value
                .checked_add(delta)
                .ok_or("YTTM global count exceeds i128")?;
            if *value > i128::from(u64::MAX) {
                return Err("YTTM global frequency exceeds u64".into());
            }
            if *value == 0 {
                self.counts.remove(&key);
            }
        }
        // Tokenizers refreshes witnesses for all positive neighbor events,
        // including net-zero updates and reused identities, not just new IDs.
        // Preserve even sub-minimum positive witnesses: upstream can stop at
        // one before refreshing a later stale witness after identity reuse.
        for key in births {
            let count = queue_score(&self.counts, key);
            if count > 0 {
                self.queue.push_birth(
                    PairPriority {
                        key,
                        priority_count: count,
                    },
                    birth_words.remove(&key),
                );
            }
        }
        Ok(())
    }
}
struct Driver {
    local: Option<Store>,
    senders: Vec<mpsc::Sender<Command>>,
    receiver: mpsc::Receiver<Completion>,
    local_ready: VecDeque<Payload>,
    stop: Arc<AtomicBool>,
}
impl Drop for Driver {
    fn drop(&mut self) {
        // This guard lives INSIDE thread::scope, so shutdown runs before the
        // implicit join, including errors, identity restarts, and panics.
        self.stop.store(true, Ordering::Relaxed);
        for sender in &self.senders {
            let _ = sender.send(Command::Stop);
        }
    }
}
fn execute_task(store: &mut Store, owner: usize, task: Task, stop: &AtomicBool) -> Result<Payload> {
    #[cfg(test)]
    let observer = store.hook.clone();
    #[cfg(test)]
    if let Some(observer) = &observer {
        observer(owner, task.sequence, 0, false);
    }
    store.run(&task.rule, owner, stop)?;
    #[cfg(test)]
    if let Some(observer) = &observer {
        observer(owner, task.sequence, 0, true);
    }
    Ok(Payload {
        owner,
        sequence: task.sequence,
        deltas: store.take_changes(),
        births: store.take_births(),
        birth_words: store.take_birth_words(),
    })
}
impl Driver {
    fn dispatch(&mut self, sequence: usize, rule: Rule) -> Result<()> {
        if let Some(store) = &mut self.local {
            self.local_ready.push_back(execute_task(
                store,
                0,
                Task { sequence, rule },
                &self.stop,
            )?);
        } else {
            for sender in &self.senders {
                sender
                    .send(Command::Execute(Task {
                        sequence,
                        rule: rule.clone(),
                    }))
                    .map_err(|_| "YTTM worker task channel closed")?;
            }
        }
        Ok(())
    }
    fn enable_coverage(&mut self) -> Result<Vec<AHashMap<u64, AHashSet<u32>>>> {
        if let Some(store) = &mut self.local {
            return Ok(vec![store.enable_coverage()]);
        }
        for sender in &self.senders {
            sender
                .send(Command::Witnesses)
                .map_err(|_| "YTTM worker task channel closed")?;
        }
        let mut witnesses = vec![AHashMap::new(); self.senders.len()];
        for _ in 0..self.senders.len() {
            match self
                .receiver
                .recv()
                .map_err(|_| "YTTM worker completion channel closed")?
            {
                Completion::Witnesses(owner, words) => witnesses[owner] = words,
                Completion::Failed(error) => return Err(error.into()),
                Completion::Done(_) => return Err("YTTM coverage requires an idle pipeline".into()),
            }
        }
        Ok(witnesses)
    }
    fn receive(&mut self) -> Result<Payload> {
        if self.local.is_some() {
            return self
                .local_ready
                .pop_front()
                .ok_or_else(|| "YTTM direct owner has no completion".into());
        }
        match self
            .receiver
            .recv()
            .map_err(|_| "YTTM worker completion channel closed")?
        {
            Completion::Done(payload) => Ok(payload),
            Completion::Failed(message) => Err(message.into()),
            Completion::Witnesses(_, _) => Err("unexpected YTTM word coverage response".into()),
        }
    }
}
struct Pending {
    sequence: usize,
    rule: Rule,
    done: Vec<bool>,
    completed: usize,
    deltas: AHashMap<u64, i128>,
    births: AHashSet<u64>,
    birth_words: AHashMap<u64, Vec<AHashSet<u32>>>,
}
fn merge_payload(pending: &mut Pending, payload: Payload) -> Result<()> {
    if pending.done[payload.owner] {
        return Err("YTTM owner completed one task twice".into());
    }
    pending.done[payload.owner] = true;
    pending.completed += 1;
    pending.births.extend(payload.births);
    for (key, words) in payload.birth_words {
        let witnesses = pending
            .birth_words
            .entry(key)
            .or_insert_with(|| (0..pending.done.len()).map(|_| AHashSet::new()).collect());
        witnesses[payload.owner].extend(words);
    }
    for (key, delta) in payload.deltas {
        let value = pending.deltas.entry(key).or_default();
        *value = value
            .checked_add(delta)
            .ok_or("YTTM complete task delta exceeds i128")?;
    }
    Ok(())
}

fn coordinate(
    trainer: &BpeTrainer,
    word_to_id: &mut AHashMap<CompactString, u32>,
    id_to_word: &mut Vec<CompactString>,
    progress: &Option<ProgressBar>,
    mut frontier: Frontier,
    driver: &mut Driver,
    workers: usize,
) -> Result<Vec<(Pair, u32)>> {
    let depth = if workers == 1 { 1 } else { 2 };
    let mut sequence = 0;
    let mut pending: VecDeque<Pending> = VecDeque::new();
    let mut merges = Vec::new();
    loop {
        // Complete results enter the frontier in rule order; a fast worker may
        // already have executed the next rule, whose deltas remain private.
        while pending
            .front()
            .is_some_and(|task| task.completed == workers)
        {
            let task = pending.pop_front().expect("checked pending head");
            frontier.publish(task.deltas, task.births, task.birth_words)?;
        }
        if word_to_id.len() < trainer.vocab_size && pending.len() < depth {
            let active: Vec<_> = pending.iter().map(|task| task.rule.clone()).collect();
            if let Some((priority, location)) =
                frontier
                    .queue
                    .top(&frontier.counts, &active, frontier.floor)
            {
                let pair = key_pair(priority.key);
                if priority.priority_count >= frontier.floor && !conflicts(pair, &active) {
                    let a = &id_to_word[pair.0 as usize];
                    let mut b = id_to_word[pair.1 as usize].as_str();
                    if let Some(prefix) = &trainer.continuing_subword_prefix
                        && let Some(rest) = b.strip_prefix(prefix)
                    {
                        b = rest;
                    }
                    let token = CompactString::from(format!("{a}{b}"));
                    let existing = word_to_id.get(&token).copied();
                    // Reusing an ID can create pairs that already have queue
                    // witnesses; finish earlier rules before exposing it.
                    if existing.is_none() || active.is_empty() {
                        // Before the first identity reuse, each pair has one
                        // birth witness. Its lazy posting list retains exactly
                        // those words. Materialize coverage only at this boundary.
                        if existing.is_some() && !frontier.queue.covered() {
                            frontier.queue.cover(&driver.enable_coverage()?);
                        }
                        let words = frontier.queue.pop(location);
                        let replacement = match existing {
                            Some(id) => id,
                            None => {
                                let id = u32::try_from(id_to_word.len())
                                    .map_err(|_| "BPE vocabulary exceeds u32")?;
                                id_to_word.push(token.clone());
                                word_to_id.insert(token, id);
                                id
                            }
                        };
                        let rule = Rule {
                            pair,
                            replacement,
                            reserved: existing.is_some(),
                            words,
                        };
                        driver.dispatch(sequence, rule.clone())?;
                        pending.push_back(Pending {
                            sequence,
                            rule,
                            done: vec![false; workers],
                            completed: 0,
                            deltas: AHashMap::new(),
                            births: AHashSet::new(),
                            birth_words: AHashMap::new(),
                        });
                        sequence += 1;
                        merges.push((pair, replacement));
                        if let Some(progress) = progress {
                            progress.inc(1);
                        }
                        trainer.emit_json_progress(
                            "Compute merges",
                            merges.len(),
                            trainer.vocab_size,
                        );
                        continue;
                    }
                }
            }
        }
        if pending.is_empty() {
            break;
        }
        let payload = driver.receive()?;
        let task = pending
            .iter_mut()
            .find(|task| task.sequence == payload.sequence)
            .ok_or("YTTM completion refers to an unpublished task")?;
        merge_payload(task, payload)?;
    }
    Ok(merges)
}
pub(super) fn train(
    trainer: &BpeTrainer,
    words: Vec<Vec<u32>>,
    counts: Vec<u64>,
    word_to_id: &mut AHashMap<CompactString, u32>,
    id_to_word: &mut Vec<CompactString>,
    progress: &Option<ProgressBar>,
    workers: usize,
) -> Result<Vec<(Pair, u32)>> {
    let workers = workers.min(words.len().max(1)).max(1);
    let mass = words
        .iter()
        .zip(&counts)
        .try_fold(0_u64, |mass, (word, &weight)| {
            let len = u64::try_from(word.len()).map_err(|_| "YTTM word length exceeds u64")?;
            mass.checked_add(
                len.checked_mul(weight)
                    .ok_or("YTTM corpus mass exceeds u64")?,
            )
            .ok_or("YTTM corpus mass exceeds u64")
        })?;
    // Same sqrt split as YTTM, applied to the weighted, pre-tokenized input.
    let threshold =
        usize::try_from(mass.isqrt().max(1)).map_err(|_| "YTTM queue threshold exceeds usize")?;
    let chunk = words.len().div_ceil(workers).max(1);
    let mut shards: Vec<Vec<(Vec<u32>, u64)>> = (0..workers).map(|_| Vec::new()).collect();
    for (i, word) in words.into_iter().zip(counts).enumerate() {
        shards[(i / chunk).min(workers - 1)].push(word);
    }
    trainer.update_progress(progress, shards.iter().map(Vec::len).sum(), "Count pairs");
    let stores = std::thread::scope(|scope| {
        let handles: Vec<_> = shards
            .into_iter()
            .map(|shard| scope.spawn(move || Store::build(shard, trainer.max_token_length)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "YTTM initialization worker panicked")?
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let frontier = Frontier::new(&stores, trainer.min_frequency.max(1), threshold)?;
    trainer.finalize_progress(
        progress,
        stores.iter().map(Store::word_count).sum(),
        "Count pairs",
    );
    trainer.update_progress(progress, trainer.vocab_size, "Compute merges");
    let merges = run_owned(
        trainer, word_to_id, id_to_word, progress, stores, frontier, workers,
    )?;
    trainer.finalize_progress(progress, merges.len(), "Compute merges");
    Ok(merges)
}

fn run_owned(
    trainer: &BpeTrainer,
    word_to_id: &mut AHashMap<CompactString, u32>,
    id_to_word: &mut Vec<CompactString>,
    progress: &Option<ProgressBar>,
    stores: Vec<Store>,
    frontier: Frontier,
    workers: usize,
) -> Result<Vec<(Pair, u32)>> {
    let stop = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    if workers == 1 {
        // Same local state, queue and selection with direct execution: avoid
        // gratuitous single-core context switches in scaling measurements.
        let mut driver = Driver {
            local: stores.into_iter().next(),
            senders: Vec::new(),
            receiver,
            local_ready: VecDeque::new(),
            stop,
        };
        return coordinate(
            trainer,
            word_to_id,
            id_to_word,
            progress,
            frontier,
            &mut driver,
            workers,
        );
    }
    std::thread::scope(|scope| {
        let mut senders = Vec::new();
        for (owner, mut store) in stores.into_iter().enumerate() {
            let (task_sender, task_receiver) = mpsc::channel();
            senders.push(task_sender);
            let sender = sender.clone();
            let stop = Arc::clone(&stop);
            scope.spawn(move || {
                let outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                        while let Ok(command) = task_receiver.recv() {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            match command {
                                Command::Stop => break,
                                Command::Witnesses => {
                                    sender
                                        .send(Completion::Witnesses(owner, store.enable_coverage()))
                                        .map_err(|_| "YTTM coordinator exited")?;
                                }
                                Command::Execute(task) => {
                                    let payload = execute_task(&mut store, owner, task, &stop)?;
                                    sender
                                        .send(Completion::Done(payload))
                                        .map_err(|_| "YTTM coordinator exited")?;
                                }
                            }
                        }
                        Ok(())
                    }));
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        let _ = sender.send(Completion::Failed(error.to_string()));
                    }
                    Err(_) => {
                        let _ = sender.send(Completion::Failed("YTTM owner panicked".into()));
                    }
                }
            });
        }
        drop(sender);
        let mut driver = Driver {
            local: None,
            senders,
            receiver,
            local_ready: VecDeque::new(),
            stop,
        };
        coordinate(
            trainer,
            word_to_id,
            id_to_word,
            progress,
            frontier,
            &mut driver,
            workers,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Condvar, Mutex};

    fn fixture() -> (
        BpeTrainer,
        AHashMap<CompactString, u32>,
        Vec<CompactString>,
        Vec<Store>,
        Frontier,
    ) {
        let trainer = BpeTrainer::builder()
            .vocab_size(9)
            .show_progress(false)
            .build();
        let tokens: Vec<CompactString> = ["a", "b", "c", "d", "e", "f"]
            .into_iter()
            .map(Into::into)
            .collect();
        let ids = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let stores = vec![
            Store::build(vec![(vec![0, 1], 1000), (vec![4, 5], 800)], None).unwrap(),
            Store::build(vec![(vec![2, 3], 900)], None).unwrap(),
        ];
        let frontier = Frontier::new(&stores, 1, 61).unwrap();
        (trainer, ids, tokens, stores, frontier)
    }

    #[test]
    fn two_slot_pipeline_runs_second_rule_while_first_owner_is_blocked() {
        let (trainer, mut ids, mut tokens, mut stores, frontier) = fixture();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let slow_done = Arc::new(AtomicBool::new(false));
        for (owner, store) in stores.iter_mut().enumerate() {
            let gate = Arc::clone(&gate);
            let slow_done = Arc::clone(&slow_done);
            store.hook = Some(Arc::new(move |_, sequence, rank, done| {
                assert_eq!(rank, 0, "every task must contain one rule");
                if owner == 0 && sequence == 0 && !done {
                    let ready = gate.0.lock().unwrap();
                    let (ready, _) = gate
                        .1
                        .wait_timeout_while(ready, std::time::Duration::from_secs(5), |ready| {
                            !*ready
                        })
                        .unwrap();
                    assert!(
                        *ready,
                        "second task must be published while the first is in flight"
                    );
                } else if owner == 0 && sequence == 0 && done {
                    slow_done.store(true, Ordering::Release);
                } else if owner == 1 && sequence == 1 && done {
                    assert!(!slow_done.load(Ordering::Acquire));
                    *gate.0.lock().unwrap() = true;
                    gate.1.notify_one();
                } else if sequence == 2 {
                    assert!(
                        slow_done.load(Ordering::Acquire),
                        "third task must await a free slot"
                    );
                }
            }));
        }
        let merges =
            run_owned(&trainer, &mut ids, &mut tokens, &None, stores, frontier, 2).unwrap();
        assert_eq!(merges, [((0, 1), 6), ((2, 3), 7), ((4, 5), 8)]);
        assert!(*gate.0.lock().unwrap());
    }

    #[test]
    fn worker_panic_shuts_down_all_owners_before_scope_join() {
        let (trainer, mut ids, mut tokens, mut stores, frontier) = fixture();
        stores[0].hook = Some(Arc::new(|_, _, _, done| {
            if !done {
                panic!("injected worker panic");
            }
        }));
        let error =
            run_owned(&trainer, &mut ids, &mut tokens, &None, stores, frontier, 2).unwrap_err();
        assert!(matches!(
            error.to_string().as_str(),
            "YTTM owner panicked" | "YTTM worker task channel closed"
        ));
    }
}
