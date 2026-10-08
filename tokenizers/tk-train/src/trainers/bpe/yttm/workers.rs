//! Fixed task/result slots and cooperative count-table access, as in YTTM C++.
use super::{
    BpeTrainer, Frontier, Location, Pair, PairPriority, Rule, coordinate, local::Store, pair_key,
};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use std::sync::{
    Arc, Condvar, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tk_encode::{Result, utils::progress::ProgressBar};

#[derive(Default)]
pub(super) struct Buffers {
    pub(super) left: AHashMap<u32, i128>,
    pub(super) right: AHashMap<u32, i128>,
    pub(super) birth_words: AHashMap<u64, AHashSet<u32>>,
}
struct Task {
    sequence: usize,
    rule: Rule,
    buffers: Buffers,
}
pub(super) struct Payload {
    pub(super) owner: usize,
    pub(super) sequence: usize,
    pub(super) buffers: Buffers,
}
struct State {
    store: Store,
    tasks: [Option<Task>; 2],
}
struct Owner {
    state: Mutex<State>,
    changed: Condvar,
    // Results belong to a completed rule. Reading the older slot must not
    // wait for a worker currently updating its store for the younger rule.
    results: [Mutex<Option<Payload>>; 2],
    // Release/acquire publishes results; the mutex protects their storage.
    completed: AtomicUsize,
    // The coordinator clears this before locking all count tables. Workers
    // yield the mutex between occurrences, never during a neighbor update.
    use_counts: AtomicBool,
}
impl Owner {
    fn new(store: Store) -> Self {
        Self {
            state: Mutex::new(State {
                store,
                tasks: [None, None],
            }),
            changed: Condvar::new(),
            results: [Mutex::new(None), Mutex::new(None)],
            completed: AtomicUsize::new(0),
            use_counts: AtomicBool::new(true),
        }
    }
}
#[derive(Default)]
struct Ready {
    error: Mutex<Option<String>>,
    changed: Condvar,
}
impl Ready {
    fn notify(&self) {
        // Pair notification with the wait mutex: a completion cannot be lost
        // between the coordinator's ready scan and its condition-variable wait.
        let _guard = self.error.lock().unwrap_or_else(|error| error.into_inner());
        self.changed.notify_one();
    }
    fn fail(&self, message: String) {
        let mut error = self.error.lock().unwrap_or_else(|error| error.into_inner());
        if error.is_none() {
            *error = Some(message);
        }
        self.changed.notify_one();
    }
}

pub(super) struct Driver<'a> {
    owners: &'a [Arc<Owner>],
    ready: Arc<Ready>,
    received: Vec<usize>,
    recycled: Vec<Vec<Buffers>>,
    stop: Arc<AtomicBool>,
}
impl Drop for Driver<'_> {
    fn drop(&mut self) {
        // This guard lives inside thread::scope. Wake every owner before the
        // implicit join, including a worker paused for count-table access.
        self.stop.store(true, Ordering::Release);
        for owner in self.owners {
            let _guard = owner
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            owner.changed.notify_all();
        }
    }
}
fn finish_task(store: &mut Store, owner: usize, mut task: Task) -> Payload {
    store.birth_scores_into(&mut task.buffers.left, &mut task.buffers.right);
    store.exchange_birth_words(&mut task.buffers.birth_words);
    Payload {
        owner,
        sequence: task.sequence,
        buffers: task.buffers,
    }
}
fn work(owner: &Owner, index: usize, ready: &Ready, stop: &AtomicBool) -> Result<()> {
    let mut sequence = 0;
    let mut state = owner.state.lock().map_err(|_| "YTTM owner panicked")?;
    loop {
        state = owner
            .changed
            .wait_while(state, |state| {
                !stop.load(Ordering::Acquire)
                    && (!owner.use_counts.load(Ordering::Acquire)
                        || state.tasks[sequence % 2].is_none())
            })
            .map_err(|_| "YTTM owner panicked")?;
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut task = state.tasks[sequence % 2].take().expect("checked task slot");
        if task.sequence != sequence {
            return Err("YTTM owner task sequence mismatch".into());
        }
        // Test hooks may block an owner before its first event. They must not
        // prevent the coordinator from reading untouched local count tables.
        #[cfg(test)]
        {
            let observer = state.store.hook.clone();
            drop(state);
            if let Some(observer) = &observer {
                observer(index, sequence, 0, false);
            }
            state = owner.state.lock().map_err(|_| "YTTM owner panicked")?;
        }
        state
            .store
            .exchange_birth_words(&mut task.buffers.birth_words);
        let positions = state.store.begin_rule(&task.rule, index)?;
        for &position in &positions {
            state = owner
                .changed
                .wait_while(state, |_| {
                    !stop.load(Ordering::Acquire) && !owner.use_counts.load(Ordering::Acquire)
                })
                .map_err(|_| "YTTM owner panicked")?;
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            state.store.apply_position(&task.rule, position)?;
        }
        // C++ releases the occurrence lock before erasing the posting list,
        // then reacquires it to read the completed rule's absolute scores.
        drop(state);
        drop(positions);
        state = owner.state.lock().map_err(|_| "YTTM owner panicked")?;
        #[cfg(test)]
        {
            let observer = state.store.hook.clone();
            drop(state);
            if let Some(observer) = &observer {
                observer(index, sequence, 0, true);
            }
            state = owner.state.lock().map_err(|_| "YTTM owner panicked")?;
        }
        let payload = finish_task(&mut state.store, index, task);
        drop(state);
        let mut result = owner.results[sequence % 2]
            .lock()
            .map_err(|_| "YTTM result slot poisoned")?;
        if result.replace(payload).is_some() {
            return Err("YTTM result slot reused before consumption".into());
        }
        drop(result);
        sequence += 1;
        owner.completed.store(sequence, Ordering::Release);
        // Do not acquire Ready's wait mutex while holding the owner mutex:
        // completion notification follows publication outside the owner lock.
        ready.notify();
        state = owner.state.lock().map_err(|_| "YTTM owner panicked")?;
    }
}

/// One C++ coordinator critical section: pause, select, publish, resume.
pub(super) struct Paused<'a> {
    owners: &'a [Arc<Owner>],
    states: Vec<MutexGuard<'a, State>>,
}
impl Drop for Paused<'_> {
    fn drop(&mut self) {
        // Set every predicate under the same locks used to inspect the stores.
        // Release all stores before notification, matching the C++ lock scope.
        for owner in self.owners {
            owner.use_counts.store(true, Ordering::Release);
        }
        self.states.clear();
        for owner in self.owners {
            owner.changed.notify_one();
        }
    }
}
impl Paused<'_> {
    fn count(&self, key: u64) -> Result<i128> {
        self.states.iter().try_fold(0_i128, |count, state| {
            count
                .checked_add(state.store.counts.get(&key).copied().unwrap_or(0))
                .ok_or_else(|| "YTTM global score exceeds i128".into())
        })
    }
    pub(super) fn select(
        &self,
        frontier: &mut Frontier,
        active: &[Rule],
    ) -> Result<Option<(PairPriority, Location)>> {
        frontier.queue.top(
            &mut |key| {
                let score = self.count(key)?;
                if score > i128::from(u64::MAX) {
                    return Err("YTTM global frequency exceeds u64".into());
                }
                Ok(score as u64)
            },
            active,
            frontier.floor,
        )
    }
    pub(super) fn enable_coverage(&mut self) -> Vec<AHashMap<u64, AHashSet<u32>>> {
        self.states
            .iter_mut()
            .map(|state| state.store.enable_coverage())
            .collect()
    }
}
impl<'a> Driver<'a> {
    pub(super) fn pause(&self) -> Result<Paused<'a>> {
        for owner in self.owners {
            owner.use_counts.store(false, Ordering::Release);
        }
        let states = self
            .owners
            .iter()
            .map(|owner| owner.state.lock().map_err(|_| "YTTM owner panicked"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Paused {
            owners: self.owners,
            states,
        })
    }
    pub(super) fn refresh_scores(
        &self,
        left: &mut AHashMap<u32, i128>,
        right: &mut AHashMap<u32, i128>,
        replacement: u32,
    ) -> Result<()> {
        let paused = self.pause()?;
        for (&token, value) in left.iter_mut() {
            *value = paused.count(pair_key((token, replacement)))?;
        }
        for (&token, value) in right.iter_mut() {
            *value = paused.count(pair_key((replacement, token)))?;
        }
        Ok(())
    }
    pub(super) fn dispatch(
        &mut self,
        paused: &mut Paused<'a>,
        sequence: usize,
        rule: Rule,
    ) -> Result<()> {
        for (index, state) in paused.states.iter_mut().enumerate() {
            let slot = &mut state.tasks[sequence % 2];
            if slot.is_some() {
                return Err("YTTM task slot reused before consumption".into());
            }
            *slot = Some(Task {
                sequence,
                rule: rule.clone(),
                buffers: self.recycled[index].pop().unwrap_or_default(),
            });
        }
        Ok(())
    }
    // The C++ loop collects every ready owner for the oldest rule in one pass.
    // Its wait mutex is not held while reading completed result slots.
    pub(super) fn collect(
        &mut self,
        sequence: usize,
        mut submit: impl FnMut(&mut Payload) -> Result<()>,
    ) -> Result<bool> {
        let mut progress = false;
        for (index, owner) in self.owners.iter().enumerate() {
            if self.received[index] == sequence
                && owner.completed.load(Ordering::Acquire) > sequence
            {
                let mut payload = owner.results[sequence % 2]
                    .lock()
                    .map_err(|_| "YTTM result slot poisoned")?
                    .take()
                    .ok_or("YTTM completed task has no result")?;
                if payload.sequence != sequence || payload.owner != index {
                    return Err("YTTM completed task sequence mismatch".into());
                }
                submit(&mut payload)?;
                self.received[index] += 1;
                self.recycled[index].push(payload.buffers);
                progress = true;
            }
        }
        Ok(progress)
    }
    pub(super) fn wait(&self, sequence: usize, done: &[bool]) -> Result<()> {
        let mut error = self
            .ready
            .error
            .lock()
            .map_err(|_| "YTTM completion mutex poisoned")?;
        loop {
            if let Some(message) = &*error {
                return Err(message.clone().into());
            }
            if self.owners.iter().enumerate().any(|(index, owner)| {
                !done[index] && owner.completed.load(Ordering::Acquire) > sequence
            }) {
                return Ok(());
            }
            error = self
                .ready
                .changed
                .wait(error)
                .map_err(|_| "YTTM completion mutex poisoned")?;
        }
    }
}

enum Initial<'a> {
    Shard(&'a [(Vec<u32>, u64)]),
    #[cfg(test)]
    Store(Box<Store>),
}
struct Startup {
    owners: Mutex<Vec<Option<Arc<Owner>>>>,
    changed: Condvar,
}
struct StartupGuard<'a> {
    startup: &'a Startup,
    stop: &'a AtomicBool,
}
impl Drop for StartupGuard<'_> {
    fn drop(&mut self) {
        // Also cover initialization errors, before Driver can be constructed.
        self.stop.store(true, Ordering::Release);
        let owners = self
            .startup
            .owners
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for owner in owners.iter().flatten() {
            let _state = owner.state.lock().unwrap_or_else(|e| e.into_inner());
            owner.changed.notify_all();
        }
    }
}

pub(super) fn run_sharded(
    trainer: &BpeTrainer,
    word_to_id: &mut AHashMap<CompactString, u32>,
    id_to_word: &mut Vec<CompactString>,
    progress: &Option<ProgressBar>,
    shards: Vec<Vec<(Vec<u32>, u64)>>,
    threshold: usize,
) -> Result<Vec<(Pair, u32)>> {
    run_initial(
        trainer,
        word_to_id,
        id_to_word,
        progress,
        // Like C++, retain the source token arrays on the coordinator through
        // the scoped join. Workers borrow their contiguous partitions.
        shards
            .iter()
            .map(|shard| Initial::Shard(shard.as_slice()))
            .collect(),
        None,
        threshold,
    )
}

#[cfg(test)]
pub(super) fn run_owned(
    trainer: &BpeTrainer,
    word_to_id: &mut AHashMap<CompactString, u32>,
    id_to_word: &mut Vec<CompactString>,
    progress: &Option<ProgressBar>,
    stores: Vec<Store>,
    frontier: Frontier,
    workers: usize,
) -> Result<Vec<(Pair, u32)>> {
    assert_eq!(stores.len(), workers);
    run_initial(
        trainer,
        word_to_id,
        id_to_word,
        progress,
        stores
            .into_iter()
            .map(|store| Initial::Store(Box::new(store)))
            .collect(),
        Some(frontier),
        1,
    )
}

fn run_initial(
    trainer: &BpeTrainer,
    word_to_id: &mut AHashMap<CompactString, u32>,
    id_to_word: &mut Vec<CompactString>,
    progress: &Option<ProgressBar>,
    initial: Vec<Initial<'_>>,
    frontier: Option<Frontier>,
    threshold: usize,
) -> Result<Vec<(Pair, u32)>> {
    let workers = initial.len();
    let stop = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(Ready::default());
    let startup = Arc::new(Startup {
        owners: Mutex::new((0..workers).map(|_| None).collect()),
        changed: Condvar::new(),
    });
    std::thread::scope(|scope| {
        let _startup_guard = StartupGuard {
            startup: &startup,
            stop: &stop,
        };
        for (index, initial) in initial.into_iter().enumerate() {
            let ready = Arc::clone(&ready);
            let stop = Arc::clone(&stop);
            let startup = Arc::clone(&startup);
            scope.spawn(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let store = match initial {
                        Initial::Shard(shard) => Store::build(shard, trainer.max_token_length)?,
                        #[cfg(test)]
                        Initial::Store(store) => *store,
                    };
                    let owner = Arc::new(Owner::new(store));
                    {
                        let mut owners =
                            startup.owners.lock().map_err(|_| "YTTM startup poisoned")?;
                        owners[index] = Some(Arc::clone(&owner));
                        startup.changed.notify_one();
                    }
                    let result = work(&owner, index, &ready, &stop);
                    if stop.load(Ordering::Acquire) {
                        // The coordinator has finished reading stores. Release
                        // private allocations here, as C++'s worker scope does;
                        // leave shared counts and result tables for its cleanup.
                        let mut state = owner
                            .state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        state.store.release_local_storage();
                    }
                    result
                }));
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => ready.fail(error.to_string()),
                    Err(_) => ready.fail("YTTM owner panicked".into()),
                }
                let _owners = startup.owners.lock().unwrap_or_else(|e| e.into_inner());
                startup.changed.notify_one();
            });
        }
        let owners = {
            let mut owners = startup.owners.lock().map_err(|_| "YTTM startup poisoned")?;
            loop {
                if let Some(error) = &*ready.error.lock().map_err(|_| "YTTM startup poisoned")? {
                    return Err(error.clone().into());
                }
                if owners.iter().all(Option::is_some) {
                    break owners
                        .iter()
                        .map(|owner| Arc::clone(owner.as_ref().unwrap()))
                        .collect::<Vec<_>>();
                }
                owners = startup
                    .changed
                    .wait(owners)
                    .map_err(|_| "YTTM startup poisoned")?;
            }
        };
        let frontier = match frontier {
            Some(frontier) => frontier,
            None => {
                let states = owners
                    .iter()
                    .map(|owner| owner.state.lock().map_err(|_| "YTTM owner panicked"))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let frontier = Frontier::new(
                    states.iter().map(|state| &state.store),
                    trainer.min_frequency.max(1),
                    threshold,
                )?;
                trainer.finalize_progress(
                    progress,
                    states.iter().map(|state| state.store.word_count()).sum(),
                    "Count pairs",
                );
                trainer.update_progress(progress, trainer.vocab_size, "Compute merges");
                frontier
            }
        };
        let mut driver = Driver {
            owners: &owners,
            ready,
            received: vec![0; workers],
            recycled: (0..workers).map(|_| Vec::with_capacity(2)).collect(),
            stop: Arc::clone(&stop),
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
