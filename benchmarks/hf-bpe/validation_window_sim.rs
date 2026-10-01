// Queue-only simulation. The builder prepends the production Candidate/Owner
// definitions and includes its real heap and validation-window modules.
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shards: usize = args[1].parse().unwrap();
    let workers: usize = args[2].parse().unwrap();
    let label = &args[3];
    let stale: u64 = args[4].parse().unwrap();
    let total: u32 = args[5].parse().unwrap();
    let target: usize = args[6].parse().unwrap();
    let mode = match label.as_str() {
        "serial" => SelectionMode::Serial,
        "cached" => SelectionMode::Cached,
        "leader" => SelectionMode::Leader,
        "bulk4" => SelectionMode::Bulk(4),
        "bulk16" => SelectionMode::Bulk(16),
        _ => panic!("invalid mode"),
    };
    assert!((1024..=16 * 1024 * 1024).contains(&total) && target < 40000);
    let initial_heads = total.div_ceil(1024);
    let first_replacement = initial_heads.max(1024) as usize + 1024;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
    let setup = Instant::now();
    let mut owners: Vec<Owner> = (0..shards).map(|_| Owner::default()).collect();
    for ledger in &mut owners {
        ledger.entries = AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19));
    }
    let mut candidates: Vec<Vec<Candidate>> = (0..shards).map(|_| Vec::new()).collect();
    for i in 0..total {
        let k = key(i >> 10, i & 1023);
        let random = u64::from(i).wrapping_mul(0x9e3779b97f4a7c15);
        let upper = 1000 + (random >> 20) % 1_000_000;
        let frequency = if random % 100 < stale {
            upper / (2 + (random >> 32) % 7)
        } else { upper };
        let o = owner(k, shards);
        if !(stale != 0 && i % 17 == 0) {
            owners[o].entries.insert(k, Entry { frequency, blocks: SmallPosting::default() });
        }
        candidates[o].push(Candidate { key: k, frequency: upper });
    }
    for (o, ledger) in owners.iter_mut().enumerate() {
        ledger.heap = CandidateHeap::new(std::mem::take(&mut candidates[o]).into_iter(), true);
    }
    let heap_bytes: usize = owners.iter().map(|o| o.heap.capacity_bytes()).sum();
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    let begin = Instant::now();
    let initial = Instant::now();
    pool.install(|| owners.par_iter_mut().for_each(|o| o.prepare_window(mode)));
    let initial_prefetch_ms = initial.elapsed().as_secs_f64() * 1000.0;
    let mut frontier = Frontier::default();
    let mut digest = Sha256::new();
    let mut consumed = 0;
    let mut batches = 0;
    let mut max_batch = 0;
    let mut select_ms = 0.0;
    let mut apply_ms = 0.0;
    while consumed < target {
        let phase = Instant::now();
        frontier.begin_epoch(&mut owners, mode);
        let mut chosen = Vec::new();
        let mut heads = ahash::AHashSet::new();
        let mut tails = ahash::AHashSet::new();
        while chosen.len() < 256 && consumed + chosen.len() < target {
            let Some((o, candidate)) = frontier.best(&mut owners, mode) else { break };
            let a = (candidate.key >> 32) as u32;
            let b = candidate.key as u32;
            if !chosen.is_empty() && (a == b || tails.contains(&a) || heads.contains(&b)) {
                break;
            }
            frontier.consume(&mut owners, o, mode);
            owners[o].entries.remove(&candidate.key).unwrap();
            heads.insert(a);
            tails.insert(b);
            digest.update(candidate.key.to_le_bytes());
            digest.update(candidate.frequency.to_le_bytes());
            chosen.push(candidate);
            if a == b { break }
        }
        for ledger in &mut owners { ledger.end_selection(); }
        select_ms += phase.elapsed().as_secs_f64() * 1000.0;
        if chosen.is_empty() { break }
        max_batch = max_batch.max(chosen.len());
        // Deterministic local changes depend on the ordered output, not shards
        // or map iteration. Each new identity creates two previously absent keys.
        let mut changes: Vec<Vec<(u64, Option<u64>)>> = (0..shards).map(|_| Vec::new()).collect();
        for (rank, candidate) in chosen.iter().enumerate() {
            let a = (candidate.key >> 32) as u32;
            let b = candidate.key as u32;
            for delta in [1, 7, 31] {
                let k = key(a % initial_heads, (b + delta) % 1024);
                changes[owner(k, shards)].push((k, None));
            }
            let replacement = (first_replacement + consumed + rank) as u32;
            for k in [key(a, replacement), key(replacement, b)] {
                changes[owner(k, shards)].push((k, Some(candidate.frequency / 2 + 1)));
            }
        }
        let phase = Instant::now();
        pool.install(|| owners.par_iter_mut().zip(changes.par_iter()).for_each(|(ledger, changes)| {
            for &(k, birth) in changes {
                if let Some(frequency) = birth {
                    assert!(ledger.entries.insert(k, Entry { frequency, blocks: SmallPosting::default() }).is_none());
                    ledger.heap.push(Candidate { key: k, frequency });
                } else if let Some(entry) = ledger.entries.get_mut(&k) {
                    entry.frequency /= 2;
                    if entry.frequency < 2 { ledger.entries.remove(&k); }
                }
            }
            ledger.prepare_window(mode);
        }));
        apply_ms += phase.elapsed().as_secs_f64() * 1000.0;
        consumed += chosen.len();
        batches += 1;
    }
    let pipeline_ms = begin.elapsed().as_secs_f64() * 1000.0;
    println!("{}", serde_json::json!({
        "shards": shards, "physical_workers": workers, "mode": label,
        "initial_stale_percent": stale, "initial_candidates": total,
        "selected": consumed, "batch_rounds": batches, "max_batch": max_batch,
        "trace_sha256": format!("{:x}", digest.finalize()),
        "setup_ms": setup_ms, "pipeline_ms": pipeline_ms,
        "initial_prefetch_ms": initial_prefetch_ms, "select_ms": select_ms, "apply_ms": apply_ms,
        "heap_bytes": heap_bytes, "entry_size": std::mem::size_of::<Entry>(),
        "owner_probes": frontier.owner_probes, "leader_updates": frontier.leader_updates,
        "truth_checks": owners.iter().map(|o| o.truth_checks).sum::<usize>(),
        "stale_corrections": owners.iter().map(|o| o.stale_corrections).sum::<usize>(),
        "prefetched": owners.iter().map(|o| o.window.prefetched).sum::<usize>(),
        "unused_restored": owners.iter().map(|o| o.window.restored).sum::<usize>(),
        "serial_refills": owners.iter().map(|o| o.window.serial_refills).sum::<usize>(),
        "worker_prefetch_ms": owners.iter().map(|o| o.window.worker_ms).sum::<f64>(),
    }));
}
