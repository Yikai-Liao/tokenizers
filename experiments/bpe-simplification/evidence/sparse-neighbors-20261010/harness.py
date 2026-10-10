from pathlib import Path
import shutil
variants=Path('/tmp/bpe-sparse-variants');h=Path('/tmp/bpe-sparse-harness');(h/'src').mkdir(parents=True,exist_ok=True)
work=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
shutil.copyfile(work/'tokenizers/tk-train/src/trainers/bpe/positions.rs',h/'src/positions.rs')
(h/'Cargo.toml').write_text('''[package]
name="bpe-sparse-adapter-check"
version="0.0.0"
edition="2024"
[dependencies]
thread_local="1.1.10"
smallvec={version="1.16",features=["union"]}
itertools="0.14"
rayon="1.10"
ahash="0.8.12"
indexmap="2.14"
sparsley="=0.1.0"
xsparseset="=0.2.5"
bevy_ecs={version="=0.20.0",default-features=false,features=["std"]}
cranelift-entity="=0.136.2"
''')
lib='''#![allow(dead_code, unused_imports)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub mod models { pub mod bpe { pub type Pair=(u32,u32); } }
mod positions;
struct BpeTrainer;
const WORD_SEPARATOR_ID:u32=u32::MAX;
fn add(count:&mut u64, amount:u64)->Result<()> { *count=count.checked_add(amount).ok_or("BPE weighted frequency exceeds u64")?; Ok(()) }
mod corpus { pub struct Corpus; pub struct Match; }
mod vocabulary { pub struct Vocabulary; }
mod index { pub struct Candidate { pub positions:super::positions::Positions } pub struct PairIndex<'a>(std::marker::PhantomData<&'a ()>); }
'''
common='''
struct Writes;
#[test]
fn neighbor_order_reuse_floor_and_failed_task_cleanup() {
    let rule=Rule{pair:(1,2),replacement:3,candidate:Candidate{positions:Positions::default()}};
    let codec=Codec::new(1);
    let mut directories=Directories::default();
    for complete in [false,true] {
        RESET
        let mut neighbors=Neighbors::new(&rule,4,&mut directories,complete);
        for (left,id,position,weight) in [(true,8,10,3),(true,6,11,2),(true,8,12,4),(false,8,13,5),(false,3,14,1)] {
            neighbors.record(left,id,id,position,weight,true).unwrap();
        }
        let output=neighbors.finish(3,&codec).unwrap();
        assert_eq!(output.iter().map(|c|c.removed).collect::<Vec<_>>(),[(8,1),(6,1),(2,8),(2,3)]);
        assert_eq!(output.iter().map(|c|c.removed_weight).collect::<Vec<_>>(),[7,2,5,1]);
        assert_eq!(output.iter().map(|c|c.bucket).collect::<Vec<_>>(),[8,8,9,8]);
        assert_eq!(output.iter().map(|c|c.born_weight).collect::<Vec<_>>(),if complete {vec![7,0,5,0]} else {vec![7,2,5,1]});
        let values:Vec<Vec<u64>>=output.iter().map(|c|match &c.positions {Birth::Partial(b)=>b.iter().collect(),Birth::Complete(p)=>p.iter().collect()}).collect();
        assert_eq!(values,if complete {vec![vec![10,12],vec![],vec![13],vec![]]} else {vec![vec![10,12],vec![11],vec![13],vec![14]]});
    }
    // An overflow can abort preparation before finish; the next reset must drop
    // its payloads and start every reused key with an empty signed ledger.
    RESET
    {
        let mut failed=Neighbors::new(&rule,0,&mut directories,false);
        failed.record(true,8,8,0,u64::MAX,true).unwrap();
        assert!(failed.record(true,8,8,1,1,true).is_err());
    }
    RESET
    let mut next=Neighbors::new(&rule,0,&mut directories,false);
    next.record(true,8,8,2,1,true).unwrap();
    let output=next.finish(1,&codec).unwrap();
    assert_eq!(output.len(),1);
    assert_eq!(output[0].removed_weight,1);
    assert_eq!(output[0].born_weight,1);
}
pub(crate) fn allocation_probe(domain:usize, groups:usize)->(usize,usize,usize,usize) {
    let origin=crate::allocated();
    let rule=Rule{pair:(1,2),replacement:3,candidate:Candidate{positions:Positions::default()}};
    let codec=Codec::new(1);
    let mut directories=Directories::default();
    RESET_DOMAIN
    let mut neighbors=Neighbors::new(&rule,0,&mut directories,false);
    for side in [true,false] {
        for offset in 0..groups { neighbors.group((domain-1-offset) as u32,side).removed_weight=1; }
    }
    let live=crate::allocated()-origin;
    let output=neighbors.finish(1,&codec).unwrap();
    let published=crate::allocated()-origin;
    drop(output);
    let retained=crate::allocated()-origin;
    drop(directories);drop(codec);
    let released=crate::allocated()-origin;
    (live,published,retained,released)
}
'''
for arm in ['baseline','sparsley','xsparseset','bevy','cranelift','indexmap','cranelift-selected','indexmap-selected']:
    source=(variants/arm/'tokenizers/tk-train/src/trainers/bpe/merge.rs').read_text()
    header=source[:source.index('impl Batch {')]
    reset='directories.reset(64);' if arm=='baseline' else 'directories.reset();'
    reset_domain='directories.reset(domain);' if arm=='baseline' else 'directories.reset();'
    selected=arm.endswith('-selected')
    case=common.replace('Directories','NeighborGroups') if selected else common
    if selected: reset=reset_domain=''
    name=arm.replace('-','_')
    (h/f'src/{name}.rs').write_text(header+case.replace('RESET_DOMAIN',reset_domain).replace('RESET',reset))
    lib+=f'mod {name};\n'
lib+='''
use std::sync::atomic::{AtomicUsize,Ordering};
static LIVE:AtomicUsize=AtomicUsize::new(0);
struct CountAlloc;
unsafe impl std::alloc::GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self,layout:std::alloc::Layout)->*mut u8 { let p=unsafe{std::alloc::System.alloc(layout)};if !p.is_null(){LIVE.fetch_add(layout.size(),Ordering::Relaxed);} p }
    unsafe fn dealloc(&self,p:*mut u8,layout:std::alloc::Layout) { LIVE.fetch_sub(layout.size(),Ordering::Relaxed);unsafe{std::alloc::System.dealloc(p,layout)} }
    unsafe fn realloc(&self,p:*mut u8,layout:std::alloc::Layout,size:usize)->*mut u8 { let q=unsafe{std::alloc::System.realloc(p,layout,size)};if !q.is_null(){LIVE.fetch_add(size,Ordering::Relaxed);LIVE.fetch_sub(layout.size(),Ordering::Relaxed);} q }
}
#[global_allocator]static ALLOC:CountAlloc=CountAlloc;
fn allocated()->usize {LIVE.load(Ordering::Relaxed)}
pub fn probe() {
    let _ = indexmap::allocation_probe(64,1); // Prewarm ahash global seed before counting container allocations.
    println!("arm,domain,groups_per_side,live_bytes,published_bytes,retained_bytes,unreleased_bytes");
'''
for arm in ['baseline','sparsley','xsparseset','bevy','cranelift','indexmap','cranelift-selected','indexmap-selected']:
    name=arm.replace('-','_')
    lib+=f'    for (domain,groups) in [(50000,24),(1000000,24),(50000,10000)] {{let (a,b,c,d)={name}::allocation_probe(domain,groups);println!("{arm},{{domain}},{{groups}},{{a}},{{b}},{{c}},{{d}}");}}\n'
lib+='}\n'
(h/'src/lib.rs').write_text(lib)
(h/'src/main.rs').write_text('fn main(){bpe_sparse_adapter_check::probe();}\n')
print(h)
