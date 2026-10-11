"""Only fixed container declarations, freeze boundaries, and test type adaptation."""
from pathlib import Path
r=Path(__file__).resolve().parent
w=r/'worktree-boxed'
b=w/'tokenizers/tk-train/src/trainers/bpe'
def change(name, replacements):
    p=b/(name+'.rs');s=p.read_text()
    for old,new,count in replacements:
        assert s.count(old)==count,(name,old,s.count(old))
        s=s.replace(old,new)
    p.write_text(s)
change('corpus', [
 ('starts: Vec<usize>,','starts: Box<[usize]>,',2),
 ('weight_regions: Vec<(usize, u64)>,','weight_regions: Box<[(usize, u64)]>,',2),
 ('occurrence_spans: Option<Vec<AtomicUsize>>,','occurrence_spans: Option<Box<[AtomicUsize]>>,',1),
 ("words: Vec<(&'input CompactString, &'input u64)>,","words: Box<[(&'input CompactString, &'input u64)]>,",1),
 ('words.par_sort_unstable_by(|a, b| b.1.cmp(a.1));','words.par_sort_unstable_by(|a, b| b.1.cmp(a.1));\n        let words = words.into_boxed_slice();',1),
 ('            starts,\n            weight_regions,','            starts: starts.into_boxed_slice(),\n            weight_regions: weight_regions.into_boxed_slice(),',1),
 ('starts: if self.reuse { self.starts } else { Vec::new() },','starts: if self.reuse { self.starts } else { Box::default() },',1),
 ('self.occurrence_spans = Some(values);','self.occurrence_spans = Some(values.into_boxed_slice());',1),
 ('Narrow(Vec<AtomicU16>),','Narrow(Box<[AtomicU16]>),',1),
 ('Wide(Vec<AtomicU32>),','Wide(Box<[AtomicU32]>),',1),
 ('Self::Narrow((0..length).map(|_| AtomicU16::new(u16::MAX)).collect())','Self::Narrow(\n                (0..length)\n                    .map(|_| AtomicU16::new(u16::MAX))\n                    .collect::<Vec<_>>()\n                    .into_boxed_slice(),\n            )',1),
 ('.map(|_| AtomicU32::new(WORD_SEPARATOR_ID))\n                    .collect(),','.map(|_| AtomicU32::new(WORD_SEPARATOR_ID))\n                    .collect::<Vec<_>>()\n                    .into_boxed_slice(),',1),
 ('plan.words = vec![(&text, &weight); starts.len()];','plan.words = vec![(&text, &weight); starts.len()].into_boxed_slice();',1),
 ('plan.starts = starts;','plan.starts = starts.into_boxed_slice();',1),
 ('plan.starts = vec![1, 4, 8, 10];','plan.starts = vec![1, 4, 8, 10].into_boxed_slice();',1),
 ('plan.words = vec![(&text, &weight); 4];','plan.words = vec![(&text, &weight); 4].into_boxed_slice();',1),
])
change('vocabulary', [
 ('characters: Vec<u32>,','characters: Box<[u32]>,',1),
 ('decorated: Vec<[u32; 3]>,','decorated: Box<[[u32; 3]]>,',1),
 ('characters: vec![WORD_SEPARATOR_ID; 0x110000],','characters: vec![WORD_SEPARATOR_ID; 0x110000].into_boxed_slice(),',1),
 ('decorated: Vec::new(),','decorated: Box::default(),',1),
 ('            ids.decorated\n                .resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);','            let mut decorated = Vec::new();\n            decorated.resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);\n            ids.decorated = decorated.into_boxed_slice();',1),
])
change('merge', [
 ('heads: Vec<Head>,','heads: Box<[Head]>,',1),
 ('tails: Vec<bool>,','tails: Box<[bool]>,',1),
 ('let mut heads = vec![Head::Empty; id_count];','let mut heads = vec![Head::Empty; id_count].into_boxed_slice();',1),
 ('let mut tails = vec![false; id_count];','let mut tails = vec![false; id_count].into_boxed_slice();',1),
])
change('index', [
 ('shards: Vec<Shard>,','shards: Box<[Shard]>,',1),
 ('            shards,\n            queue:', '            shards: shards.into_boxed_slice(),\n            queue:',1),
])
print('converted four files')
