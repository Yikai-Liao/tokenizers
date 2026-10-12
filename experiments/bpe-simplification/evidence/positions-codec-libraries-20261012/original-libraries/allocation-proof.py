from pathlib import Path
import subprocess,os
r=Path(__file__).resolve().parent;c=r/'probe';p=c/'src/bin/allocation-proof.rs';p.parent.mkdir(exist_ok=True);base=(c/'src/main.rs').read_text();base=base[:base.index('fn run<P: List>')];base=base.replace('#[path="','#[path="../')
base+='''
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
static ALLOC:AtomicUsize=AtomicUsize::new(0); static REALLOC:AtomicUsize=AtomicUsize::new(0); static FREE:AtomicUsize=AtomicUsize::new(0);
struct Count;
unsafe impl GlobalAlloc for Count {
 unsafe fn alloc(&self,l:Layout)->*mut u8 { ALLOC.fetch_add(1,Relaxed); unsafe { System.alloc(l) } }
 unsafe fn alloc_zeroed(&self,l:Layout)->*mut u8 { ALLOC.fetch_add(1,Relaxed); unsafe { System.alloc_zeroed(l) } }
 unsafe fn realloc(&self,p:*mut u8,l:Layout,n:usize)->*mut u8 { REALLOC.fetch_add(1,Relaxed); unsafe { System.realloc(p,l,n) } }
 unsafe fn dealloc(&self,p:*mut u8,l:Layout) { FREE.fetch_add(1,Relaxed); unsafe { System.dealloc(p,l) } }
}
#[global_allocator] static COUNT:Count=Count;
fn count<P:List>(values:&[u64])->(usize,usize,usize) {
 let before=(ALLOC.load(Relaxed),REALLOC.load(Relaxed),FREE.load(Relaxed));
 let mut list=P::default();for &v in values {list.push(v);}black_box(&list);drop(list);
 (ALLOC.load(Relaxed)-before.0,REALLOC.load(Relaxed)-before.1,FREE.load(Relaxed)-before.2)
}
fn main() {
 for kind in 0..4 {
  for len in [0,1,2,3,8,127,128,129,255,256,257,512,8193] {
   let values:Vec<u64>=(0..len).map(|n|match kind {0=>0,1=>n as u64/3,2=>n as u64*(u64::MAX/8193),_=>u64::MAX}).collect();
   let hand=count::<hand::Positions>(&values);
   let gimli=count::<gimli::Positions>(&values);let slice=count::<fmt_slice::Positions>(&values);let array=count::<fmt_array::Positions>(&values);
   assert_eq!(hand,gimli);assert_eq!(hand,slice);assert_eq!(hand,array);
   println!("kind={kind} len={len} hand=gimli=fmt_slice=fmt_array alloc/realloc/free={hand:?}");
  }
 }
}
''';p.write_text(base)
env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0'}
subprocess.run(['cargo','run','--release','--offline','--manifest-path',str(c/'Cargo.toml'),'--bin','allocation-proof','--target-dir',str(r/'probe-target')],env=env,check=True)
