"""Separate safe Rust diagnostic copies; never used as timing binaries."""
from pathlib import Path
import json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/lyk/code/tokenizers');build=json.loads((r/'builds.json').read_text())
diagnostics={}
for arm,b in build['builds'].items():
    w=r/('diagnostic-'+arm)
    subprocess.run(['git','worktree','add','--detach',str(w),build['baseline_commit']],cwd=repo,check=True)
    for f in b['source_hashes']:
        if f.endswith(('.rs','Cargo.toml','Cargo.lock')):shutil.copy2(Path(b['worktree'])/f,w/f)
    folder=w/'tokenizers/tk-train/src/trainers/bpe'
    boxed=arm=='boxed'
    return_type='Box<[T]>' if boxed else 'Vec<T>'
    convert='values.into_boxed_slice()' if boxed else 'values'
    helper='''use serde_json::json;
use std::{fs,mem,time::Instant};

#[inline(never)]
pub(super) fn freeze<T>(name: &str, values: Vec<T>) -> RETURN {
    let len=values.len(); let capacity=values.capacity(); let old=values.as_ptr() as usize;
    let begin=Instant::now();
    let result=CONVERT;
    let seconds=begin.elapsed().as_secs_f64();
    eprintln!("BPE_STORAGE {}",json!({"event":"freeze","name":name,"len":len,"capacity":capacity,
        "element_bytes":size_of::<T>(),"potential_spare_bytes":(capacity-len)*size_of::<T>(),
        "old_pointer":old,"new_pointer":result.as_ptr() as usize,"pointer_changed":old!=result.as_ptr() as usize,
        "seconds":seconds,"container_bytes":mem::size_of_val(&result),"boxed":BOXED}));
    result
}

pub(super) fn phase(name: &str) {
    let status=fs::read_to_string("/proc/self/status").unwrap();
    let read=|key: &str| status.lines().find(|s|s.starts_with(key)).unwrap().split_whitespace().nth(1).unwrap().parse::<usize>().unwrap();
    eprintln!("BPE_STORAGE {}",json!({"event":"phase","name":name,"rss_kib":read("VmRSS:"),"hwm_kib":read("VmHWM:")}));
}

pub(super) fn growth<T>(name: &str, old: usize, values: &Vec<T>) {
    if old!=values.capacity() {eprintln!("BPE_STORAGE {}",json!({"event":"capacity_growth","name":name,"old_capacity":old,"capacity":values.capacity(),"len":values.len(),"element_bytes":size_of::<T>()}));}
}

pub(super) fn sizes(values: &[(&str,usize)]) {
    eprintln!("BPE_STORAGE {}",json!({"event":"sizes","values":values}));
}
'''.replace('RETURN',return_type).replace('CONVERT',convert).replace('BOXED',str(boxed).lower())
    (folder/'storage_diagnostics.rs').write_text(helper)
    p=folder/'mod.rs';s=p.read_text().replace('mod corpus;','mod storage_diagnostics;\nmod corpus;')
    s=s.replace('let plan = CorpusPlan::build(words, ids, self, reuse)?;', 'storage_diagnostics::phase("before-build");\n        let plan = CorpusPlan::build(words, ids, self, reuse)?;\n        storage_diagnostics::phase("after-build-freeze");')
    s=s.replace('corpus = plan.materialize();','corpus = plan.materialize();\n            storage_diagnostics::phase("after-materialize-and-temporary-drop");')
    s=s.replace('            drop(index);\n            drop(corpus);','            storage_diagnostics::phase("training-complete-before-drop");\n            drop(index);\n            drop(corpus);\n            storage_diagnostics::phase("after-index-corpus-drop");')
    s=s.replace('let (vocab, merges) = vocabulary.into_model_parts(merges);','let (vocab, merges) = vocabulary.into_model_parts(merges);\n        storage_diagnostics::phase("after-output-model");');p.write_text(s)
    p=folder/'corpus.rs';s=p.read_text()
    s=s.replace('                weight_regions.push((length, weight));','                let old_capacity=weight_regions.capacity();\n                weight_regions.push((length, weight));\n                super::storage_diagnostics::growth("weight_regions",old_capacity,&weight_regions);')
    if boxed:
        s=s.replace('let words = words.into_boxed_slice();','let words = super::storage_diagnostics::freeze("words", words);')
        s=s.replace('starts: starts.into_boxed_slice(),','starts: super::storage_diagnostics::freeze("starts", starts),')
        s=s.replace('weight_regions: weight_regions.into_boxed_slice(),','weight_regions: super::storage_diagnostics::freeze("weight_regions", weight_regions),')
        s=s.replace('Some(values.into_boxed_slice())','Some(super::storage_diagnostics::freeze("occurrence_spans", values))')
        s=s.replace('''Self::Narrow(
                (0..length)''','''Self::Narrow(super::storage_diagnostics::freeze("slots_narrow",
                (0..length)''').replace('''Self::Wide(
                (0..length)''','''Self::Wide(super::storage_diagnostics::freeze("slots_wide",
                (0..length)''')
        s=s.replace('''.collect::<Vec<_>>()
                    .into_boxed_slice(),
            )''','''.collect::<Vec<_>>()
            ))''')
    else:
        s=s.replace('words.par_sort_unstable_by(|a, b| b.1.cmp(a.1));','words.par_sort_unstable_by(|a, b| b.1.cmp(a.1));\n        let words=super::storage_diagnostics::freeze("words", words);')
        s=s.replace('            starts,\n            weight_regions,','            starts: super::storage_diagnostics::freeze("starts",starts),\n            weight_regions: super::storage_diagnostics::freeze("weight_regions",weight_regions),')
        s=s.replace('Some(values);','Some(super::storage_diagnostics::freeze("occurrence_spans",values));')
        s=s.replace('Self::Narrow((0..length).map(|_| AtomicU16::new(u16::MAX)).collect())','Self::Narrow(super::storage_diagnostics::freeze("slots_narrow",(0..length).map(|_| AtomicU16::new(u16::MAX)).collect()))')
        s=s.replace('Self::Wide(\n                (0..length)','Self::Wide(super::storage_diagnostics::freeze("slots_wide",\n                (0..length)').replace('.map(|_| AtomicU32::new(WORD_SEPARATOR_ID))\n                    .collect(),\n            )','.map(|_| AtomicU32::new(WORD_SEPARATOR_ID))\n                    .collect(),\n            ))')
    s=s.replace('        let tokens = Slots::new(self.length, self.slot_bound);','        super::storage_diagnostics::sizes(&[("Corpus",std::mem::size_of::<Corpus>()),("CorpusPlan",std::mem::size_of::<CorpusPlan>()),("Slots",std::mem::size_of::<Slots>())]);\n        let tokens = Slots::new(self.length, self.slot_bound);')
    s=s.replace('        self.occurrence_spans = Some(', '        super::storage_diagnostics::phase("before-alias-occurrences-freeze");\n        self.occurrence_spans = Some(')
    p.write_text(s)
    p=folder/'vocabulary.rs';s=p.read_text()
    s=s.replace('characters: vec![WORD_SEPARATOR_ID; 0x110000]'+('.into_boxed_slice()' if boxed else '')+',','characters: super::storage_diagnostics::freeze("characters",vec![WORD_SEPARATOR_ID; 0x110000]),')
    if boxed:s=s.replace('ids.decorated = decorated.into_boxed_slice();','ids.decorated = super::storage_diagnostics::freeze("decorated",decorated);')
    else:s=s.replace('            ids.decorated\n                .resize(self.tokens.len(), [WORD_SEPARATOR_ID; 3]);','            let mut decorated=Vec::new();\n            decorated.resize(self.tokens.len(),[WORD_SEPARATOR_ID;3]);\n            ids.decorated=super::storage_diagnostics::freeze("decorated",decorated);')
    s=s.replace('        // 1. Map existing character IDs', '        super::storage_diagnostics::sizes(&[("InitialTokenIds",std::mem::size_of::<InitialTokenIds>())]);\n        // 1. Map existing character IDs');p.write_text(s)
    p=folder/'merge.rs';s=p.read_text()
    for name,value in [('heads','Head::Empty'),('tails','false')]:
        s=s.replace(f'let mut {name} = vec![{value}; id_count]'+('.into_boxed_slice()' if boxed else '')+';',f'let mut {name} = super::storage_diagnostics::freeze("{name}",vec![{value}; id_count]);')
    s=s.replace('        // Resolve shared endpoints once', '        super::storage_diagnostics::sizes(&[("SelectedRules",std::mem::size_of::<SelectedRules>())]);\n        // Resolve shared endpoints once');p.write_text(s)
    p=folder/'index.rs';s=p.read_text().replace('            shards'+(': shards.into_boxed_slice()' if boxed else '')+',','            shards: super::storage_diagnostics::freeze("shards",shards),')
    s=s.replace('        // 1. Count and freeze', '        super::storage_diagnostics::sizes(&[("PairIndex",std::mem::size_of::<PairIndex>())]);\n        // 1. Count and freeze');p.write_text(s)
    diagnostics[arm]={'worktree':str(w),'binary':str(r/('diagnostic-'+arm+'-runner'))}
(r/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
print('isolated diagnostic copies prepared')
