"""Report actual inferred reader layout and joined output capacity; never time this build."""
import argparse
from pathlib import Path
import shutil
from prepare import BPE, replace


def instrument(source, dest):
    shutil.copytree(source, dest)
    merge = dest / BPE / 'merge.rs'
    replace(merge, '                for (matched, weight) in self.corpus.fresh_matches(&mut positions, pair) {', '''                let decoder_bytes = std::mem::size_of_val(&positions);
                let reader = self.corpus.fresh_matches(&mut positions, pair);
                static AA: std::sync::Once = std::sync::Once::new();
                static AB: std::sync::Once = std::sync::Once::new();
                let once = if pair.0 == pair.1 { &AA } else { &AB };
                once.call_once(|| eprintln!("LOOKUP_LAYOUT {}", serde_json::json!({
                    "aa":pair.0==pair.1,"decoder":decoder_bytes,"reader":std::mem::size_of_val(&reader),
                    "cache":std::mem::size_of::<super::WeightLookup>(),
                    "option_cache":std::mem::size_of::<Option<super::WeightLookup>>(),
                    "corpus":std::mem::size_of::<Corpus>(),"plan":std::mem::size_of::<super::CorpusPlan>(),
                    "prepared_chunk":std::mem::size_of::<PreparedChunk>()})));
                for (matched, weight) in reader {''')
    replace(merge, '        let chunks = self.prepare(selected)?;', '''        let begin = std::time::Instant::now();
        let chunks = self.prepare(selected)?;
        eprintln!("LOOKUP_PREPARED {}", serde_json::json!({
            "seconds":begin.elapsed().as_secs_f64(),"chunks":chunks.len(),
            "chunks_bytes":chunks.capacity()*std::mem::size_of::<PreparedChunk>(),
            "starts_heap_bytes":chunks.iter().filter(|c| c.starts.spilled())
                .map(|c| c.starts.capacity()*std::mem::size_of::<usize>()).sum::<usize>(),
            "events_bytes":chunks.iter().map(|c| c.changes.capacity()*std::mem::size_of::<PairChange<Positions>>()).sum::<usize>(),
            "position_heap_bytes":chunks.iter().flat_map(|c| &c.changes)
                .map(|c| c.positions.diagnostic_heap_bytes()).sum::<usize>()}));''')
    positions = dest / BPE / 'positions.rs'
    replace(positions, 'impl Positions {', '''impl Positions {
    pub(super) fn diagnostic_heap_bytes(&self) -> usize {
        let heap = |leaf: &Leaf| if leaf.bytes.spilled() { leaf.bytes.capacity() } else { 0 };
        match &self.0 {
            Repr::Leaf(leaf) => heap(leaf),
            Repr::Root { leaves, .. } => leaves.len()*std::mem::size_of::<Leaf>()
                + leaves.iter().map(heap).sum::<usize>(),
        }
    }
''')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('dest',type=Path)
    a=p.parse_args();instrument(a.source.resolve(),a.dest.resolve())
