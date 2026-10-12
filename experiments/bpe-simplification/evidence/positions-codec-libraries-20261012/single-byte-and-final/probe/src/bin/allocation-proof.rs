#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path="../hand/positions.rs"] mod hand;
#[path="../fmt_zero_push/positions.rs"] mod fmt_zero_push;
#[path="../fmt_small/positions.rs"] mod fmt_small;
use std::{hint::black_box, time::Instant};
trait List: Default {
    fn push(&mut self,value:u64);
    fn sum(&self)->u64;
    fn seek_sum(&self,value:u64)->u64;
    fn storage(&self)->(usize,usize,bool);
}
macro_rules! list {
    ($module:ident) => {
        impl List for $module::Positions {
            #[inline(always)] fn push(&mut self,value:u64) { self.push(value).unwrap(); }
            #[inline(always)] fn sum(&self)->u64 { self.iter().fold(0u64,u64::wrapping_add) }
            #[inline(always)] fn seek_sum(&self,value:u64)->u64 { self.iter_from_value(value).fold(0u64,u64::wrapping_add) }
            fn storage(&self)->(usize,usize,bool) { self.storage() }
        }
    }
}
list!(hand);list!(fmt_zero_push);list!(fmt_small);
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
   assert_eq!(hand,count::<fmt_zero_push::Positions>(&values),"fmt_zero_push len={len} kind={kind}");
   assert_eq!(hand,count::<fmt_small::Positions>(&values),"fmt_small len={len} kind={kind}");
   println!("kind={kind} len={len} all candidates alloc/realloc/free={hand:?}");
  }
 }
}
