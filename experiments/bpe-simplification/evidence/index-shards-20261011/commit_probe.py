"""Build isolated commit probes from frozen index/positions source.

Usage: python commit_probe.py BASELINE_ROOT CANDIDATE_ROOT OUTPUT_ROOT
Only CorpusPlan and ProgressBar (unused by this probe) are stubbed. PairIndex
routing, owner commits, heap publication and Positions encoding are unchanged.
The shards-only arm keeps the baseline Group/GroupPositions storage and appends.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tomllib

baseline, candidate, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
source = output / 'src'
source.mkdir(exist_ok=True)
base = (baseline / 'tokenizers/tk-train/src/trainers/bpe/index.rs').read_text().split('#[cfg(test)]\nmod tests')[0]
current = (candidate / 'tokenizers/tk-train/src/trainers/bpe/index.rs').read_text().split('#[cfg(test)]\nmod tests')[0]
position_source = (candidate / 'tokenizers/tk-train/src/trainers/bpe/positions.rs').read_text().split('#[cfg(test)]\nmod tests')[0]
assert position_source == (baseline / 'tokenizers/tk-train/src/trainers/bpe/positions.rs').read_text().split('#[cfg(test)]\nmod tests')[0]
legacy_group = base[base.index('/// Owner-local partial births'):base.index('// Metadata stays borrowed')]
intermediate = current.replace('// Metadata stays borrowed', legacy_group + '// Metadata stays borrowed')
intermediate = intermediate.replace('AHashMap::<(usize, Pair), State<SmallVec<[u64; 2]>>>::new()', 'AHashMap::<(usize, Pair), Group>::new()')
intermediate = intermediate.replace('groups.entry((change.bucket, change.born)).or_default()', 'groups.entry((change.bucket, change.born)).or_insert_with(|| Group::new(reuse))')
start = intermediate.index('            // Fresh jobs follow rule rank')
end = intermediate.index('\n        }\n\n        // Complete producers', start)
intermediate = intermediate[:start] + '''            match &mut group.positions {
                GroupPositions::Unordered(storage) => storage.extend(values.iter().copied()),
                GroupPositions::Ordered(storage) => {
                    if storage.is_empty() { *storage = values; }
                    else { storage.extend(values); }
                }
            }''' + intermediate[end:]
start = intermediate.index('            if reuse {\n                // Historical cohorts')
end = intermediate.index('            candidates.push(', start)
intermediate = intermediate[:start] + '''            let positions = match state.positions {
                GroupPositions::Unordered(mut values) => {
                    values.sort_unstable();
                    Positions::from_sorted(&values)?
                }
                GroupPositions::Ordered(values) => {
                    debug_assert!(!self.counts.contains_key(&pair));
                    self.counts.insert(pair, count);
                    Positions::from_sorted(&values)?
                }
            };
''' + intermediate[end:]
intermediate = intermediate.replace('positions: Positions::from_sorted(&state.positions)?,', 'positions,')
intermediate = intermediate.replace('for ((_, pair), mut state) in groups', 'for ((_, pair), state) in groups')

probe = r'''
pub(super) fn probe(case: &str, workers: usize, repetitions: usize) {
    use std::hint::black_box;
    use std::time::Instant;
    let reuse = case.starts_with("reuse");
    let length = match case { "fresh-large" | "reuse-large" => 256,
                              "fresh-spill" | "reuse-spill" => 16, _ => 2 };
    let group_count = if length == 256 { 1024 } else { 8192 };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
    let mut index = PairIndex {
        SHARDS_INIT
        queue: OctonaryHeap::new(), floor: 1, reuse,
    };
    let mut durations = Vec::new();
    let mut expected = None;
    let mut allocated = Vec::new();
    pool.install(|| {
        for round in 0..repetitions + 3 {
            for shard in &mut index.shards { COUNTS_CLEAR }
            let mut changes: Vec<Vec<_>> = (0..4).map(|_| Vec::new()).collect();
            for group in 0..group_count {
                let removed = (1_000_000 + group as u32, 1);
                let born = (2 + group as u32, 3);
                COUNTS_INSERT
                if case == "removals" {
                    changes[group % 4].push(Change { removed, born, removed_weight: 1,
                        born_weight: 0, bucket: group, positions: Birth::Partial(SmallVec::new()) });
                    continue;
                }
                if case == "complete" || (case == "mixed" && group % 2 == 0) {
                    changes[group % 4].push(Change { removed, born, removed_weight: 1,
                        born_weight: 2, bucket: group,
                        positions: Birth::Complete(Positions::from_sorted(&[group as u64 * 512, group as u64 * 512 + 1]).unwrap()) });
                    continue;
                }
                for fragment in 0..2 {
                    let rank = if reuse { 1 - fragment } else { fragment };
                    let values: SmallVec<[u64; 2]> = (0..length / 2)
                        .map(|i| group as u64 * 512 + (rank * length / 2 + i) as u64).collect();
                    changes[group % 4].push(Change { removed, born, removed_weight: 1,
                        born_weight: (length / 2) as u64, bucket: group,
                        positions: Birth::Partial(values) });
                }
            }
            crate::ALLOCS.store(0, std::sync::atomic::Ordering::Relaxed);
            crate::BYTES.store(0, std::sync::atomic::Ordering::Relaxed);
            // Allocation counting is a separate diagnostic run, to avoid timing distortion.
            crate::TRACK.store(std::env::var_os("COMMIT_PROBE_ALLOC").is_some(), std::sync::atomic::Ordering::SeqCst);
            let started = Instant::now();
            index.commit(black_box(changes)).unwrap();
            let elapsed = started.elapsed().as_nanos() as u64;
            crate::TRACK.store(false, std::sync::atomic::Ordering::SeqCst);
            let allocation = (crate::ALLOCS.load(std::sync::atomic::Ordering::Relaxed), crate::BYTES.load(std::sync::atomic::Ordering::Relaxed));
            let mut values: Vec<_> = std::mem::take(&mut index.queue).into_vec().into_iter()
                .map(|c| (c.priority.pair(), c.priority.count, c.positions.iter().collect::<Vec<_>>())).collect();
            values.sort_unstable();
            let mut hash = 0xcbf29ce484222325u64;
            let mut feed = |v: u64| { hash = (hash ^ v).wrapping_mul(0x100000001b3); };
            for (pair, count, positions) in &values {
                feed(pair.0 as u64); feed(pair.1 as u64); feed(*count);
                feed(positions.len() as u64);
                for &p in positions { feed(p); }
            }
            if let Some(previous) = expected { assert_eq!(previous, hash); }
            expected = Some(hash);
            if round >= 3 { durations.push(elapsed); allocated.push(allocation); }
        }
    });
    println!("{}", serde_json::json!({"case":case,"workers":workers,"groups":group_count,
        "positions_per_group":length,"durations_ns":durations,"checksum":expected,
        "allocations_and_requested_bytes":allocated, "layout":LAYOUT}));
}
'''
for arm, text in [('baseline', base), ('shards_only', intermediate), ('candidate', current)]:
    if arm == 'baseline':
        setup = 'shards: (0..workers).map(|_| AHashMap::new()).collect(), routes: Vec::new(),'
        clear = 'shard.clear();'
        insert = 'index.shards[owner(removed, workers)].insert(removed, 1_000_000);'
        layout = 'serde_json::json!({"group": std::mem::size_of::<Group>(), "group_positions": std::mem::size_of::<GroupPositions>(), "action":std::mem::size_of::<(usize,bool,bool)>(), "birth":std::mem::size_of::<Birth>(), "route":std::mem::size_of::<Route>(), "pair_index":std::mem::size_of::<PairIndex>()})'
    else:
        setup = 'shards: (0..workers).map(|_| Shard::default()).collect(),'
        clear = 'shard.counts.clear();'
        insert = 'index.shards[owner(removed, workers)].counts.insert(removed, 1_000_000);'
        group_type = 'Group' if arm == 'shards_only' else 'State<SmallVec<[u64;2]>>'
        layout = f'serde_json::json!({{"group":std::mem::size_of::<{group_type}>(), "action":std::mem::size_of::<(usize,bool,bool)>(), "birth":std::mem::size_of::<Birth>(), "shard":std::mem::size_of::<Shard>(), "pair_index":std::mem::size_of::<PairIndex>()}})'
    appended = probe.replace('SHARDS_INIT', setup).replace('COUNTS_CLEAR', clear).replace('COUNTS_INSERT', insert).replace('LAYOUT', layout)
    (source / f'{arm}.rs').write_text(text + appended)
(source / 'positions.rs').write_text(position_source)
(source / 'main.rs').write_text(r'''
#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub mod models { pub mod bpe { pub type Pair = (u32,u32); } }
pub mod utils { pub mod progress { pub struct ProgressBar; impl ProgressBar { pub fn inc(&self,_:u64) {} } } }
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
struct Allocator;
// SAFETY: every operation delegates unchanged pointers/layouts to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) { ALLOCS.fetch_add(1,Ordering::Relaxed); BYTES.fetch_add(layout.size() as u64,Ordering::Relaxed); }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { unsafe { System.dealloc(ptr,layout) } }
    unsafe fn realloc(&self, ptr:*mut u8,layout:Layout,new:usize)->*mut u8 {
        if TRACK.load(Ordering::Relaxed) { ALLOCS.fetch_add(1,Ordering::Relaxed); BYTES.fetch_add(new as u64,Ordering::Relaxed); }
        unsafe { System.realloc(ptr,layout,new) }
    }
}
#[global_allocator] static ALLOCATOR: Allocator = Allocator;
mod bpe {
    pub const WORD_SEPARATOR_ID:u32 = u32::MAX;
    fn add(count:&mut u64,weight:u64)->crate::Result<()> { *count=count.checked_add(weight).ok_or("BPE weighted frequency exceeds u64")?; Ok(()) }
    mod corpus {
        pub struct CorpusPlan<'a>(std::marker::PhantomData<&'a ()>);
        impl CorpusPlan<'_> {
            pub fn initial_ranges(&self,_:usize)->Vec<std::ops::Range<usize>> { unreachable!() }
            pub fn small_pair_domain(&self)->Option<usize> { unreachable!() }
            pub fn initial_edges(&self,_:std::ops::Range<usize>,_:impl FnMut(crate::models::bpe::Pair,u64,u64)->crate::Result<()>)->crate::Result<()> { unreachable!() }
        }
    }
    mod positions;
    mod merge {
        use super::positions::Positions;
        use smallvec::SmallVec;
        pub(super) struct Change<P> { pub removed: (u32,u32),pub born:(u32,u32),pub removed_weight:u64,pub born_weight:u64,pub bucket:usize,pub positions:P }
        pub(super) enum Birth { Partial(SmallVec<[u64;2]>),Complete(Positions) }
        impl Birth { pub(super) fn is_empty(&self)->bool { match self { Self::Partial(v)=>v.is_empty(),Self::Complete(v)=>v.is_empty() } } }
    }
    #[cfg(feature="baseline")] #[path="../baseline.rs"] mod index;
    #[cfg(feature="shards_only")] #[path="../shards_only.rs"] mod index;
    #[cfg(feature="candidate")] #[path="../candidate.rs"] mod index;
    pub fn run(case:&str, workers:usize, repetitions:usize) { index::probe(case,workers,repetitions); }
}
fn main() { let args:Vec<_>=std::env::args().collect(); bpe::run(&args[1],args[2].parse().unwrap(),args[3].parse().unwrap()); }
''')
(source / 'bpe').mkdir(exist_ok=True)
(source / 'bpe/positions.rs').write_text(position_source)
(output / 'Cargo.toml').write_text('''[package]
name = "index-commit-probe"
version = "0.1.0"
edition = "2024"
[features]
baseline = []
shards_only = []
candidate = []
[dependencies]
ahash = "0.8.12"
dary_heap = "0.3.8"
rayon = "1.12"
smallvec = { version = "1.16", features = ["union"] }
serde_json = "1"
itertools = "0.14"
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
''')
# Use exactly the runner's direct dependency versions, especially SmallVec.
runner_lock = tomllib.loads((candidate / 'experiments/bpe-simplification/runner/Cargo.lock').read_text())
toml = (output / 'Cargo.toml').read_text()
versions = {}
for name in ['ahash', 'dary_heap', 'rayon', 'smallvec', 'serde_json', 'itertools']:
    versions[name] = next(p['version'] for p in runner_lock['package'] if p['name'] == name)
    import re
    toml = re.sub(r'(?m)^(' + re.escape(name) + r' = (?:\{ version = )?")[^"]+"',
                  lambda m: m[1] + '=' + versions[name] + '"', toml)
(output / 'Cargo.toml').write_text(toml)
manifest = dict(baseline='c8e7348e890d45dd06aadc8464e085df8bcb1202',
    source_hashes={str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.rs')},
    profile=dict(opt_level=3,lto='fat',codegen_units=1), dependency_versions=versions)
(output / 'sources.json').write_text(json.dumps(manifest,indent=2))
for arm in ['baseline','shards_only','candidate']:
    command = ['cargo','build','--offline','--release','--manifest-path',str(output / 'Cargo.toml'),'--features',arm]
    with (output / f'{arm}-build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=log,check=True)
    import shutil
    shutil.copy2(output / 'target/release/index-commit-probe', output / arm)
