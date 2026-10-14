from pathlib import Path
import shutil,re
r=Path(__file__).resolve().parent
allocator=r'''
use std::sync::atomic::{AtomicUsize, Ordering};
static DIAG_ALLOC_PHASE: AtomicUsize = AtomicUsize::new(12);
static DIAG_ALLOC_NEXT_THREAD: AtomicUsize = AtomicUsize::new(0);
thread_local! { static DIAG_ALLOC_THREAD: std::cell::Cell<usize> = const {std::cell::Cell::new(usize::MAX)}; }
#[repr(align(64))]
struct DiagnosticAllocationCounter([std::sync::atomic::AtomicU64; 4]);
static DIAG_ALLOC: [[DiagnosticAllocationCounter; 12]; 64] = [const { [const { DiagnosticAllocationCounter([const {std::sync::atomic::AtomicU64::new(0)};4]) };12] };64];
struct DiagnosticAllocator;
#[global_allocator]
static DIAGNOSTIC_ALLOCATOR: DiagnosticAllocator = DiagnosticAllocator;
fn diagnostic_allocation(calls: u64, bytes: u64, reallocs: u64, freed: u64) {
    let phase = DIAG_ALLOC_PHASE.load(Ordering::Relaxed);
    if phase >= 12 {return;}
    DIAG_ALLOC_THREAD.with(|cell| {
        let mut thread = cell.get();
        if thread == usize::MAX {thread=DIAG_ALLOC_NEXT_THREAD.fetch_add(1,Ordering::Relaxed);cell.set(thread);}
        if thread >= 64 {return;}
        let counters = &DIAG_ALLOC[thread][phase].0;
        for (counter,value) in counters.iter().zip([calls,bytes,reallocs,freed]) {
            if value != 0 {counter.fetch_add(value,Ordering::Relaxed);}
        }
    });
}
unsafe impl std::alloc::GlobalAlloc for DiagnosticAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p=unsafe {std::alloc::System.alloc(layout)};
        if !p.is_null() {diagnostic_allocation(1,layout.size() as u64,0,0);}
        p
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p=unsafe {std::alloc::System.alloc_zeroed(layout)};
        if !p.is_null() {diagnostic_allocation(1,layout.size() as u64,0,0);}
        p
    }
    unsafe fn dealloc(&self,p:*mut u8,layout:std::alloc::Layout) {
        diagnostic_allocation(0,0,0,layout.size() as u64);
        unsafe {std::alloc::System.dealloc(p,layout)};
    }
    unsafe fn realloc(&self,p:*mut u8,layout:std::alloc::Layout,size:usize)->*mut u8 {
        let q=unsafe {std::alloc::System.realloc(p,layout,size)};
        if !q.is_null() {diagnostic_allocation(0,size as u64,1,layout.size() as u64);}
        q
    }
}
fn diag_start(index:usize)->std::time::Instant {
    DIAG_ALLOC_PHASE.store(index,Ordering::Relaxed);
    std::time::Instant::now()
}
fn diagnostic_allocation_report() {
    DIAG_ALLOC_PHASE.store(12,Ordering::Relaxed);
    let names=["vocabulary","plan","initial_count","materialize","merge_rounds","retire","prepare","apply","commit","select","alphabet","output"];
    for (phase,name) in names.iter().enumerate() {
        let mut totals=[0u64;4];let mut threads=Vec::new();
        for (i,thread) in DIAG_ALLOC.iter().enumerate() {
            let values=thread[phase].0.each_ref().map(|v|v.load(Ordering::Relaxed));
            for j in 0..4 {totals[j]+=values[j];}
            if values.iter().any(|&v|v!=0) {threads.push(serde_json::json!({"thread":i,"values":values}));}
        }
        eprintln!("BPE_ALLOC {}",serde_json::json!({"phase":name,"alloc_calls":totals[0],"requested_bytes":totals[1],"realloc_calls":totals[2],"freed_bytes":totals[3],"threads":threads}));
    }
}
'''
for arm in ['main','current']:
 name=arm+'-alloc';src=r/f'source-{name}';runner=r/f'runner-{name}'
 shutil.copytree(r/f'source-{arm}-phases',src)
 shutil.copytree(r/f'runner-{arm}-phases',runner)
 p=runner/'Cargo.toml';p.write_text(p.read_text().replace(f'source-{arm}-phases',f'source-{name}'))
 p=src/'tokenizers/tk-train/src/trainers/bpe'/('engine/mod.rs' if arm=='main' else 'mod.rs')
 s=p.read_text();old='let diag_begin = std::time::Instant::now();'
 while old in s:
  start=s.index(old);m=re.search(r'diag\((\d+), diag_begin\)',s[start:]);assert m
  s=s[:start]+s[start:].replace(old,f'let diag_begin = diag_start({m.group(1)});',1)
 for var,index in [('rounds_begin',4),('retire_begin',5),('output_begin',11),('select_begin',9)]:
  s=s.replace(f'let {var} = std::time::Instant::now();',f'let {var} = diag_start({index});')
 s=s.replace('fn diag(index: usize, begin: std::time::Instant) {','fn diag(index: usize, begin: std::time::Instant) {\n DIAG_ALLOC_PHASE.store(12, std::sync::atomic::Ordering::Relaxed);')
 s=s.replace('fn diag_report() {','fn diag_report() {\n diagnostic_allocation_report();')
 p.write_text(s+allocator)
 if arm=='current':
  p=src/'tokenizers/tk-train/src/trainers/bpe/merge.rs';s=p.read_text()
  for var,index in [('prep_begin',6),('apply_begin',7)]:s=s.replace(f'let {var} = std::time::Instant::now();',f'let {var} = super::super::diag_start({index});')
  p.write_text(s)
print('allocation source trees prepared')
