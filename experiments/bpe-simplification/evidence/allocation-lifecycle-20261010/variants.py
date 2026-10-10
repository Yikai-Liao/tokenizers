from pathlib import Path
import subprocess, shutil, sys

work = Path('/root/code/tokenizers-workspaces/bpe-heap-experiment')
root = Path('/tmp/bpe-allocation-variants')
prefix = 'tokenizers/tk-train/src/trainers/bpe/'

diagnostic = r'''// Experimental instrumentation; never included in production source.
use std::{cell::{Cell,RefCell}, sync::atomic::{AtomicUsize, Ordering}, time::Instant};
const N: usize = 12;
const NAMES: [&str; N] = ["vocabulary", "plan", "initial", "materialize", "select", "prepare", "apply", "commit", "drop_index", "drop_corpus", "drop_codec", "model"];
static PHASE: AtomicUsize = AtomicUsize::new(0);
static CUTOFF: AtomicUsize = AtomicUsize::new(256);
#[derive(Clone, Copy, Default, serde::Serialize)]
pub(super) struct Counts {
    heap_allocations: u64, heap_bytes: u64, small_heap_allocations: u64,
    heap_frees: u64, freed_bytes: u64, arena_allocations: u64, arena_bytes: u64,
    arena_retires: u64, inline_lists: u64,
}
const ZERO: Counts = Counts { heap_allocations:0, heap_bytes:0, small_heap_allocations:0, heap_frees:0, freed_bytes:0, arena_allocations:0, arena_bytes:0, arena_retires:0, inline_lists:0 };
thread_local! {
    static COUNTS: RefCell<[Counts; N]> = const { RefCell::new([ZERO; N]) };
    static TIMES: Cell<[(f64,f64); N]> = const { Cell::new([(0.0,0.0);N]) };
}
pub(super) fn reset() {
    rayon::broadcast(|_| { COUNTS.with(|c| *c.borrow_mut()=[ZERO;N]); TIMES.set([(0.0,0.0);N]); });
    PHASE.store(0, Ordering::Relaxed);
}
pub(super) fn cutoff(value: usize) { CUTOFF.store(value, Ordering::Relaxed); }
pub(super) fn allocation(bytes: usize, arena: bool) {
    COUNTS.with(|c| { let mut rows=c.borrow_mut(); let row=&mut rows[PHASE.load(Ordering::Relaxed)];
        if arena { row.arena_allocations+=1; row.arena_bytes+=bytes as u64; }
        else { row.heap_allocations+=1; row.heap_bytes+=bytes as u64; row.small_heap_allocations+=u64::from(bytes<=CUTOFF.load(Ordering::Relaxed)); }
    });
}
pub(super) fn retirement(bytes: usize, arena: bool) {
    COUNTS.with(|c| { let mut rows=c.borrow_mut(); let row=&mut rows[PHASE.load(Ordering::Relaxed)];
        if arena {row.arena_retires+=1;} else {row.heap_frees+=1;row.freed_bytes+=bytes as u64;}
    });
}
pub(super) fn inline() {
    COUNTS.with(|c| {c.borrow_mut()[PHASE.load(Ordering::Relaxed)].inline_lists+=1;});
}
#[repr(C)] struct Timespec { seconds:i64, nanos:i64 }
unsafe extern "C" {fn clock_gettime(clock:i32, value:*mut Timespec)->i32;}
fn cpu() -> f64 {
    let mut value=Timespec{seconds:0,nanos:0};
    // SAFETY: Linux x86_64 timespec matches this repr(C) layout; the output pointer is valid.
    assert_eq!(unsafe {clock_gettime(2,&mut value)},0);
    value.seconds as f64+value.nanos as f64*1e-9
}
pub(super) struct Span {id:usize, wall:Instant, cpu:f64}
impl Span {
    pub(super) fn new(id:usize)->Self {PHASE.store(id,Ordering::Relaxed); Self{id,wall:Instant::now(),cpu:cpu()}}
}
impl Drop for Span {
    fn drop(&mut self) {let wall=self.wall.elapsed().as_secs_f64();let cpu=cpu()-self.cpu;
        TIMES.with(|t| {let mut times=t.get();times[self.id].0+=wall;times[self.id].1+=cpu;t.set(times);});
    }
}
pub(super) fn report(storage:(usize,usize)) {
    if std::env::var_os("BPE_ALLOCATION_DIAGNOSTIC").is_none() {return;}
    let workers=rayon::broadcast(|_|COUNTS.with(|c| *c.borrow()));
    let counts: Vec<_>=(0..N).map(|id| {
        let mut sum=ZERO;
        for worker in &workers {let c=worker[id];
            sum.heap_allocations+=c.heap_allocations;sum.heap_bytes+=c.heap_bytes;sum.small_heap_allocations+=c.small_heap_allocations;
            sum.heap_frees+=c.heap_frees;sum.freed_bytes+=c.freed_bytes;sum.arena_allocations+=c.arena_allocations;
            sum.arena_bytes+=c.arena_bytes;sum.arena_retires+=c.arena_retires;sum.inline_lists+=c.inline_lists;
        } sum
    }).collect();
    let times=TIMES.get();
    eprintln!("ALLOCATION_DIAGNOSTIC {}",serde_json::json!({"phases":NAMES,"times":times,"counts":counts,"small_cutoff":CUTOFF.load(Ordering::Relaxed),"retained_bump_bytes":storage.0,"retained_scratch_capacity_bytes":storage.1}));
}
'''

for arm, commit in [('arena','7de4e068'),('arena-off','7de4e068'),('box-mutex','1396e736'),('box-tls','1396e736')]:
    if len(sys.argv)>1 and arm not in sys.argv[1:]: continue
    target=root/arm
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--',prefix],cwd=work,text=True).splitlines()
    names+=['tokenizers/tk-train/Cargo.toml','tokenizers/tk-train/Cargo.lock','experiments/bpe-simplification/runner/Cargo.lock']
    for name in names:
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('Cargo.lock'):
            p.write_bytes((Path('/tmp/bpe-sparse-variants/baseline')/name).read_bytes())
        else:
            p.write_bytes(subprocess.check_output(['git','show',commit+':'+name],cwd=work))
    folder=target/prefix
    p=folder/'positions.rs';s=p.read_text()
    if arm=='arena-off':
        s=s.replace('cutoff: ((items as u128 / 256).isqrt() as usize).max(256),','cutoff: { let _ = items; 0 },')
    if arm=='box-mutex':
        a=s.index('use std::');b=s.index('use tk_encode::Result;',a)
        s=s[:a]+'use std::{ops::Range, sync::{Mutex, MutexGuard}};\n'+s[b:]
        a=s.index('pub(super) struct Codec');b=s.index('/// One executing thread',a)
        s=s[:a]+'''pub(super) struct Codec { workers: Vec<Mutex<Worker>> }
impl Codec {
    pub(super) fn new(workers: usize) -> Self { Self {workers:(0..workers).map(|_|Mutex::default()).collect()} }
    pub(super) fn lease(&self)->Lease<'_> {
        Lease {cursor:self.workers[rayon::current_thread_index().unwrap_or(0)%self.workers.len()].lock().unwrap_or_else(|e|e.into_inner())}
    }
}

'''+s[b:]
        s=s.replace("RefMut<'codec, Worker>","MutexGuard<'codec, Worker>")
    old=arm.startswith('arena')
    # Counts do not time individual allocations. All variants use the same TLS bookkeeping.
    if old:
        s=s.replace('return Ok(Self {\n                count_and_flags:', 'super::diagnostic::inline();\n            return Ok(Self {\n                count_and_flags:',1)
        s=s.replace('if pointer.is_null() {','super::diagnostic::allocation(allocation.size(), arena);\n        if pointer.is_null() {',1)
        s=s.replace('dealloc(self.pointer(), allocation);','super::diagnostic::retirement(allocation.size(), false);\n                dealloc(self.pointer(), allocation);\n            } else {\n                super::diagnostic::retirement(0, true);',1)
        storage='''impl Arena {
    pub(super) fn diagnostic_storage(&self)->(usize,usize) {
        self.workers.iter().map(|w| {let w=w.lock().unwrap();(w.bump.allocated_bytes(), w.bytes.capacity()+w.offsets.capacity()*std::mem::size_of::<usize>())}).fold((0,0),|a,b|(a.0+b.0,a.1+b.1))
    }
}
'''
    else:
        for count in [1,2]:s=s.replace(f'if count == {count} {{',f'if count == {count} {{\n            super::diagnostic::inline();',1)
        s=s.replace('Ok(Self::Compressed(bytes.into_boxed_slice()))','super::diagnostic::allocation(capacity, false);\n        Ok(Self::Compressed(bytes.into_boxed_slice()))',1)
        s+='''\nimpl Drop for Positions {
    fn drop(&mut self) { if let Self::Compressed(bytes)=self {super::diagnostic::retirement(bytes.len(),false);} }
}
'''
        if arm=='box-tls':
            storage='''impl Codec {
    pub(super) fn diagnostic_storage(&mut self)->(usize,usize) {
        (0,self.workers.iter_mut().map(|w| {let w=w.get_mut();w.bytes.capacity()+w.offsets.capacity()*std::mem::size_of::<usize>()}).sum())
    }
}
'''
        else:
            storage='''impl Codec {
    pub(super) fn diagnostic_storage(&self)->(usize,usize) {
        (0,self.workers.iter().map(|w| {let w=w.lock().unwrap();w.bytes.capacity()+w.offsets.capacity()*std::mem::size_of::<usize>()}).sum())
    }
}
'''
    s+='\n'+storage;p.write_text(s)
    p=folder/'corpus.rs';s=p.read_text();s=s.replace('let slot_bound = vocabulary.len().max(trainer.vocab_size);','super::diagnostic::cutoff(((length as u128 / 256).isqrt() as usize).max(256));\n        let slot_bound = vocabulary.len().max(trainer.vocab_size);',1);p.write_text(s)
    (folder/'diagnostic.rs').write_text(diagnostic)
    p=folder/'mod.rs';s=p.read_text().replace('mod corpus;','mod diagnostic;\nmod corpus;',1)
    start=s.index('// 1. Resolve the vocabulary');s=s[:start]+s[start:].replace('let mut vocabulary = Vocabulary::initialize(trainer, words, workers, alphabet)?;','diagnostic::reset();\n    let mut vocabulary = {let _phase=diagnostic::Span::new(0); Vocabulary::initialize(trainer, words, workers, alphabet)?};',1)
    s=s.replace('let plan = CorpusPlan::build(words, &mut vocabulary, trainer, reuse, progress)?;','let plan = {let _phase=diagnostic::Span::new(1); CorpusPlan::build(words, &mut vocabulary, trainer, reuse, progress)?};',1)
    s=s.replace('let codec = Codec::new(workers);','let mut codec = Codec::new(workers);',1).replace('let arena = Arena::new(workers, plan.items());','let mut arena = Arena::new(workers, plan.items());',1)
    # Scope the multiline build call without modifying its arguments.
    a=s.index('let mut index = PairIndex::build(');b=s.index(')?;',a)+3
    block=s[a:b];s=s[:a]+block.replace('= PairIndex::build(','= {let _phase=diagnostic::Span::new(2); PairIndex::build(',1).removesuffix(')?;')+')?};'+s[b:]
    s=s.replace('let mut corpus = plan.materialize();','let mut corpus = {let _phase=diagnostic::Span::new(3); plan.materialize()};',1)
    s=s.replace('match Batch::select(trainer, &mut vocabulary, &mut corpus, &mut index)? {','match {let _phase=diagnostic::Span::new(4); Batch::select(trainer, &mut vocabulary, &mut corpus, &mut index)?} {',1)
    a=s.index('let prepared = batch.prepare(');b=s.index(')?;',a)+3
    block=s[a:b];s=s[:a]+block.replace('= batch.prepare(','= {let _phase=diagnostic::Span::new(5); batch.prepare(',1).removesuffix(')?;')+')?};'+s[b:]
    s=s.replace('let changes = prepared.apply(&corpus);','let changes = {let _phase=diagnostic::Span::new(6); prepared.apply(&corpus)};',1)
    s=s.replace('index.commit(changes)?;','{let _phase=diagnostic::Span::new(7); index.commit(changes)?;}',1)
    owner='arena' if old else 'codec'
    ending=f'''{{let _phase=diagnostic::Span::new(8); drop(index);}}
    {{let _phase=diagnostic::Span::new(9); drop(corpus);}}
    let storage = {owner}.diagnostic_storage();
    {{let _phase=diagnostic::Span::new(10); drop({owner});}}'''
    expected=f'drop(index);\n    drop(corpus);\n    drop({owner});';assert expected in s,arm;s=s.replace(expected,ending,1)
    s=s.replace('let (vocab, merges) = vocabulary.into_model_parts(merges);','let (vocab, merges) = {let _phase=diagnostic::Span::new(11); vocabulary.into_model_parts(merges)};\n    diagnostic::report(storage);',1)
    p.write_text(s)
print('Prepared four instrumented controls')
