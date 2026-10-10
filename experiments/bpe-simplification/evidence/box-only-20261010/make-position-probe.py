"""Instrument a disposable Box worktree; never use its binary for timing."""
import sys
from pathlib import Path
root=Path(sys.argv[1])
path=root/'tokenizers/tk-train/src/trainers/bpe/positions.rs'
s=path.read_text()
def change(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)
change('scratch: self.scratch.get_or_default().borrow_mut(),','scratch: self.scratch.get_or_default().borrow_mut(),\n            phase: 4,')
change("    }\n}\n\n/// One executing thread's", """    }
    pub(super) fn lease_at(&self, phase: usize) -> Lease<'_> {
        let mut lease = self.lease();
        lease.phase = phase;
        lease
    }
}

impl Drop for Codec {
    fn drop(&mut self) {
        let mut lists = [[0u64; 4]; 5];
        let mut positions = [[0u64; 4]; 5];
        let mut bytes = [[0u64; 4]; 5];
        let mut initial_weights = [[0u64; 3]; 4];
        for cell in self.scratch.iter_mut() {
            let scratch = cell.get_mut();
            for p in 0..5 {
                for b in 0..4 {
                    lists[p][b] += scratch.lists[p][b];
                    positions[p][b] += scratch.positions[p][b];
                    bytes[p][b] += scratch.payload_bytes[p][b];
                }
            }
            for b in 0..4 {
                for w in 0..3 {
                    initial_weights[b][w] += scratch.initial_weights[b][w];
                }
            }
        }
        eprintln!("BPE_POSITION_LENGTHS {}", serde_json::json!({
            "phases": ["initial_range", "initial_owner", "complete_birth", "owner_commit", "other"],
            "length_buckets": ["0", "1", "2", "3+"],
            "lists": lists, "positions": positions, "requested_payload_bytes": bytes,
            "initial_weight_buckets": ["0", "1", "2+"], "initial_weights": initial_weights
        }));
    }
}

/// One executing thread's""")
change('    offsets: Vec<usize>,','''    offsets: Vec<usize>,
    lists: [[u64; 4]; 5],
    positions: [[u64; 4]; 5],
    payload_bytes: [[u64; 4]; 5],
    initial_weights: [[u64; 3]; 4],''')
change("    scratch: RefMut<'codec, CodecScratch>,", "    scratch: RefMut<'codec, CodecScratch>,\n    phase: usize,")
change('/// Mutable sorted task positions,', '''impl Lease<'_> {
    pub(super) fn record_initial_weight(&mut self, count: usize, weight: u64) {
        self.scratch.initial_weights[count.min(3)][weight.min(2) as usize] += 1;
    }
}

/// Mutable sorted task positions,''')
change('        let count = input.len()?;','''        let count = input.len()?;
        let phase = lease.phase;
        let bucket = count.min(3);
        lease.scratch.lists[phase][bucket] += 1;
        lease.scratch.positions[phase][bucket] += count as u64;''')
change('        let mut bytes = Vec::with_capacity(capacity);','''        scratch.payload_bytes[phase][bucket] += capacity as u64;
        let mut bytes = Vec::with_capacity(capacity);''')
path.write_text(s)
path=root/'tokenizers/tk-train/src/trainers/bpe/index.rs'
s=path.read_text()
assert s.count('let mut lease = codec.lease();')==2
s=s.replace('let mut lease = codec.lease();','let mut lease = codec.lease_at(0);',1)
s=s.replace('let mut lease = codec.lease();','let mut lease = codec.lease_at(1);',1)
assert s.count('let lease = codec.lease();')==1
s=s.replace('let lease = codec.lease();','let lease = codec.lease_at(3);')
needle='            let positions = Positions::from_sorted(Input::Builder(&state.positions), &mut lease)?;'
assert s.count(needle)==1
s=s.replace(needle,'            lease.record_initial_weight(state.positions.len(), state.count);\n'+needle)
path.write_text(s)
path=root/'tokenizers/tk-train/src/trainers/bpe/merge.rs'
s=path.read_text()
assert s.count('let mut lease = codec.lease();')==1
path.write_text(s.replace('let mut lease = codec.lease();','let mut lease = codec.lease_at(2);'))
