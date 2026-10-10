from pathlib import Path
import shutil

base = Path('/root/code/tokenizers-workspaces/bpe-heap-diagnostic-baseline')
candidate = Path('/root/code/tokenizers-workspaces/bpe-heap-diagnostic-candidate')
engine = Path('tokenizers/tk-train/src/trainers/bpe/engine')
shutil.copy2(Path('/root/code/tokenizers-workspaces/bpe-heap-experiment') / engine / 'index.rs', candidate / engine / 'index.rs')
for root in [base, candidate]:
    p = root / engine / 'positions.rs'
    s = p.read_text()
    s = s.replace('const RESTART:', '''use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
static LIVE_OWNED: AtomicUsize = AtomicUsize::new(0);
static PEAK_OWNED: AtomicUsize = AtomicUsize::new(0);
pub(super) fn diagnostic_totals() -> (usize, usize) {
    (LIVE_OWNED.load(AtomicOrdering::Relaxed), PEAK_OWNED.load(AtomicOrdering::Relaxed))
}
const RESTART:''', 1)
    s = s.replace('    pub(super) fn lease(&self)', '''    pub(super) fn diagnostic_buffers(&self) -> (usize, usize) {
        self.workers.iter().fold((0, 0), |(arena, scratch), worker| {
            let worker = worker.lock().unwrap_or_else(|e| e.into_inner());
            (arena + worker.bump.allocated_bytes(), scratch + worker.bytes.capacity() + worker.offsets.capacity() * std::mem::size_of::<usize>())
        })
    }
    pub(super) fn lease(&self)''', 1)
    s = s.replace('    fn pointer(&self)', '''    pub(super) fn diagnostic_owned_bytes(&self) -> usize {
        if self.is_empty() || self.count_and_flags & INLINE != 0 || self.payload.addr() & ARENA != 0 {
            return 0;
        }
        // SAFETY: the live owned descriptor points to its initialized length word.
        layout(self.len(), unsafe { self.pointer().cast::<usize>().read() }).unwrap().size()
    }
    fn pointer(&self)''', 1)
    s = s.replace('        // SAFETY: final layout reserves', '''        if !arena {
            let current = LIVE_OWNED.fetch_add(allocation.size(), AtomicOrdering::Relaxed) + allocation.size();
            PEAK_OWNED.fetch_max(current, AtomicOrdering::Relaxed);
        }
        // SAFETY: final layout reserves''', 1)
    s = s.replace('                dealloc(self.pointer(), allocation);', '                LIVE_OWNED.fetch_sub(allocation.size(), AtomicOrdering::Relaxed);\n                dealloc(self.pointer(), allocation);', 1)
    p.write_text(s)
    p = root / engine / 'index.rs'
    s = p.read_text()
    if root == candidate:
        totals = '''let mut owned = 0usize;
        let mut stale = 0usize;
        let mut stale_records = 0usize;
        for item in self.queue.iter() {
            let bytes = item.positions.diagnostic_owned_bytes();
            owned += bytes;
            if !self.reuse && !self.shards[owner(item.priority.pair(), self.shards.len())].contains_key(&item.priority.pair()) {
                stale += bytes;
                stale_records += 1;
            }
        }
        let records = self.queue.len();
        let descriptor_capacity = self.queue.capacity() * std::mem::size_of::<Candidate>();
        let map_entries: usize = self.shards.iter().map(|s| s.len()).sum();
        let map_payload_capacity: usize = self.shards.iter().map(|s| s.capacity() * std::mem::size_of::<(Pair, u64)>()).sum();'''
    else:
        totals = '''let owned: usize = self.shards.iter().flat_map(|s| s.states.values()).map(|s| s.positions.diagnostic_owned_bytes()).sum::<usize>() + self.cohorts.iter().map(|c| c.positions.diagnostic_owned_bytes()).sum::<usize>();
        let stale = 0usize;
        let stale_records = 0usize;
        let records = self.shards.iter().map(|s| s.queue.len()).sum::<usize>() + self.cohorts.len();
        let descriptor_capacity = self.shards.iter().map(|s| s.queue.capacity() * std::mem::size_of::<Priority>()).sum::<usize>() + self.cohorts.capacity() * std::mem::size_of::<Candidate>();
        let map_entries: usize = self.shards.iter().map(|s| s.states.len()).sum();
        let map_payload_capacity: usize = self.shards.iter().map(|s| s.states.capacity() * std::mem::size_of::<(Pair, State<Positions>)>()).sum();'''
    method = '''    pub(super) fn diagnostic(&self, stage: &str, merges: usize) {
        if std::env::var_os("BPE_HEAP_DIAGNOSTIC").is_none() { return; }
        TOTALS
        let (live, peak) = super::positions::diagnostic_totals();
        let (arena_bytes, scratch_bytes) = self.arena.diagnostic_buffers();
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let memory = |name: &str| -> usize { status.lines().find(|s| s.starts_with(name)).unwrap().split_whitespace().nth(1).unwrap().parse().unwrap() };
        eprintln!("BPE_HEAP_DIAGNOSTIC {}", serde_json::json!({
            "stage": stage, "merges": merges, "reuse": self.reuse,
            "owned_bytes": owned, "stale_owned_bytes": stale, "stale_records": stale_records,
            "live_owned_bytes": live, "peak_live_owned_bytes": peak,
            "queue_records": records, "queue_descriptor_capacity_bytes": descriptor_capacity,
            "map_entries": map_entries, "map_payload_capacity_bytes": map_payload_capacity,
            "arena_allocated_bytes": arena_bytes, "scratch_capacity_bytes": scratch_bytes,
            "rss_kib": memory("VmRSS:"), "hwm_kib": memory("VmHWM:")
        }));
    }
'''.replace('TOTALS', totals)
    s = s.replace("impl<'arena> PairIndex<'arena> {", "impl<'arena> PairIndex<'arena> {\n" + method, 1)
    p.write_text(s)
    p = root / engine / 'mod.rs'
    s = p.read_text().replace('            if vocabulary.len() >= trainer.vocab_size {', '            index.diagnostic("initial_index", 0);\n            if vocabulary.len() >= trainer.vocab_size {', 1)
    s = s.replace('            let work = progress.stage("Compute merges", trainer.vocab_size);', '            index.diagnostic("materialized", 0);\n            let work = progress.stage("Compute merges", trainer.vocab_size);', 1)
    s = s.replace('                index.commit(changes)?;', '                index.commit(changes)?;\n                index.diagnostic("after_commit", merges.len());', 1)
    s = s.replace('            let mut restart = false;', '            let mut restart = false;\n            let mut diagnostic_round = 0usize;\n            let diagnostic_stride = std::env::var("BPE_HEAP_DIAGNOSTIC_STRIDE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).max(1);', 1)
    s = s.replace('                index.diagnostic("after_commit", merges.len());', '                diagnostic_round += 1;\n                if diagnostic_round == 1 || diagnostic_round.is_multiple_of(diagnostic_stride) {\n                    index.diagnostic("after_commit", merges.len());\n                }', 1)
    s = s.replace('            if restart {', '            index.diagnostic("finish", merges.len());\n            if restart {', 1)
    p.write_text(s)
