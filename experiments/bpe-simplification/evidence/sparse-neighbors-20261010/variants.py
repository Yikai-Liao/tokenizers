from pathlib import Path
import shutil
work=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
root=Path('/tmp/bpe-sparse-variants'); root.mkdir(exist_ok=True)
files=['tokenizers/tk-train/src/trainers/bpe/merge.rs','tokenizers/tk-train/Cargo.toml','tokenizers/tk-train/Cargo.lock','experiments/bpe-simplification/runner/Cargo.lock']
base=root/'baseline'
for name in files:
    dest=base/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(work/name,dest)
source=(base/files[0]).read_text()
a=source.index('/// Task-local left/right neighbor aggregation')
b=source.index('impl Batch {',a)
part=source[a:b]
create=part[part.index('            let id = self.rule.replacement;'):part.index('            self.directories.touched.push')]
create=create.replace('            let index = self.changes[side].len();\n','').replace('            self.changes[side].push(Change {','            Change {').replace('            });','            }')
create='let id = self.rule.replacement;\nlet pair = self.rule.pair;\nlet rank = self.rank;\n'+create[create.index('            Change {'):].replace('self.rank','rank')
record=part[part.index('    fn record('):part.index('    fn finish(')]
finish=part[part.index('    fn finish('):]
configs={
'sparsley': ('sparsley = "=0.1.0"', 'use sparsley::SparseMap;', 'SparseMap<u32, Change<Builder>>', 'map.entry(neighbor).or_insert_with(|| { CREATE })', 'map.drain().map(|(_, change)| change)', ''),
'xsparseset': ('xsparseset = "=0.2.5"', 'use xsparseset::SparseSetVec;', 'SparseSetVec<usize, Change<Builder>>', 'let key = neighbor as usize;\nif map.get_index(key).is_none() { map.insert(key, { CREATE }); }\nmap.get_mut(key).expect("just inserted or already present")', 'std::iter::from_fn(|| map.len().checked_sub(1).and_then(|index| map.swap_remove_by_index(index)))', ''),
'bevy': ('bevy_ecs = { version = "=0.20.0", default-features = false, features = ["std"] }', 'use bevy_ecs::storage::SparseSet;', 'SparseSet<u32, Change<Builder>>', 'map.get_or_insert_with(neighbor, || { CREATE })', 'std::iter::from_fn(|| map.indices().last().copied().and_then(|key| map.remove(key)))', ''),
'cranelift': ('cranelift-entity = "=0.136.2"', 'use cranelift_entity::{EntityRef, SparseMap, SparseMapValue};', 'SparseMap<NeighborId, KeyedChange>', 'let key = NeighborId(neighbor);\nif !map.contains_key(key) { map.insert(KeyedChange { key, change: { CREATE } }); }\n&mut map.get_mut(key).expect("just inserted or already present").change', 'std::iter::from_fn(|| map.pop()).map(|value| value.change)', '''#[derive(Clone, Copy, Eq, PartialEq)]
struct NeighborId(u32);
impl EntityRef for NeighborId {
    fn new(index: usize) -> Self { Self(u32::try_from(index).expect("neighbor is a real token ID")) }
    fn index(self) -> usize { self.0 as usize }
}
struct KeyedChange { key: NeighborId, change: Change<Builder> }
impl SparseMapValue<NeighborId> for KeyedChange {
    fn key(&self) -> NeighborId { self.key }
}
'''),
'indexmap': ('', 'use indexmap::IndexMap;', 'IndexMap<u32, Change<Builder>, ahash::RandomState>', 'map.entry(neighbor).or_insert_with(|| { CREATE })', 'map.drain(..).map(|(_, change)| change)', ''),
}
for arm,(dependency,imports,maptype,group,drain,extra) in configs.items():
    dst=root/arm
    shutil.copytree(base,dst,dirs_exist_ok=True)
    block='''/// Per-task ordered neighbor groups in reusable library storage.
struct Neighbors<'a> {
    rule: &'a Rule,
    rank: usize,
    directories: &'a mut Directories,
    complete: bool,
}
/// One reusable left/right sparse map per preparation worker.
#[derive(Default)]
struct Directories { groups: [MAP; 2] }
impl Directories {
    fn reset(&mut self) { for map in &mut self.groups { map.clear(); } }
}
EXTRA
impl<'a> Neighbors<'a> {
    fn new(rule: &'a Rule, rank: usize, directories: &'a mut Directories, complete: bool) -> Self {
        Self { rule, rank, directories, complete }
    }
    fn group(&mut self, neighbor: u32, left: bool) -> &mut Change<Builder> {
        let map = &mut self.directories.groups[usize::from(!left)];
        GROUP
    }
'''.replace('MAP',maptype).replace('EXTRA',extra).replace('GROUP',group.replace('CREATE',create))
    f=finish.replace('        for mut change in self.changes.into_iter().flatten() {', '        for map in &mut self.directories.groups {\n            let start = result.len();\n            for mut change in DRAIN {'.replace('DRAIN',drain))
    reverse=arm in ('xsparseset','bevy','cranelift')
    f=f.replace('        }\n        Ok(result)', '            }\n'+('            result[start..].reverse();\n' if reverse else '')+'        }\n        Ok(result)')
    if not reverse: f=f.replace('            let start = result.len();\n','')
    s=source[:a]+block+record+f+source[b:]
    s=s.replace('use ahash::{AHashMap, AHashSet};','use ahash::{AHashMap, AHashSet};\n'+imports)
    s=s.replace('directories.reset(corpus.id_count());','directories.reset();').replace('directories.reset(self.corpus.id_count());','directories.reset();')
    (dst/files[0]).write_text(s)
    if dependency:
        p=dst/files[1];p.write_text(p.read_text().replace('[dependencies]','[dependencies]\n'+dependency))
print('Created five scratch-reusing variants, plus baseline snapshot.')
