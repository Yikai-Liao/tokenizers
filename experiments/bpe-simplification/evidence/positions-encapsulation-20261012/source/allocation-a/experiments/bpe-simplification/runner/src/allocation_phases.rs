//! Diagnostic only. All memory operations still use the System allocator.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
pub static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static FREE_SMALL: [AtomicU64; 65] = [const { AtomicU64::new(0) }; 65];
static SNAPSHOTS: [[AtomicU64; 70]; 6] = [const { [const { AtomicU64::new(0) }; 70] }; 6];
static TOTALS: [[AtomicU64; 70]; 6] = [const { [const { AtomicU64::new(0) }; 70] }; 6];
struct Counters;
#[global_allocator] static GLOBAL: Counters = Counters;
// SAFETY: pointers, sizes and alignment are forwarded unchanged to System.
unsafe impl GlobalAlloc for Counters {
 unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
  let ptr=unsafe {System.alloc(layout)};
  if ACTIVE.load(Relaxed)&&!ptr.is_null(){COUNTS[0].fetch_add(1,Relaxed);COUNTS[3].fetch_add(layout.size() as u64,Relaxed);}ptr
 }
 unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
  let ptr=unsafe {System.alloc_zeroed(layout)};
  if ACTIVE.load(Relaxed)&&!ptr.is_null(){COUNTS[0].fetch_add(1,Relaxed);COUNTS[3].fetch_add(layout.size() as u64,Relaxed);}ptr
 }
 unsafe fn dealloc(&self,ptr:*mut u8,layout:Layout){
  if ACTIVE.load(Relaxed){COUNTS[1].fetch_add(1,Relaxed);COUNTS[4].fetch_add(layout.size() as u64,Relaxed);FREE_SMALL[layout.size().min(64)].fetch_add(1,Relaxed);}
  unsafe {System.dealloc(ptr,layout)};
 }
 unsafe fn realloc(&self,ptr:*mut u8,layout:Layout,size:usize)->*mut u8 {
  let new=unsafe {System.realloc(ptr,layout,size)};
  if ACTIVE.load(Relaxed)&&!new.is_null(){COUNTS[2].fetch_add(1,Relaxed);}new
 }
}
fn count(index:usize)->u64 {if index<5 {COUNTS[index].load(Relaxed)}else {FREE_SMALL[index-5].load(Relaxed)}}
#[unsafe(no_mangle)]
pub extern "C" fn bpe_allocation_phase(phase:u8,start:bool){
 let phase=phase as usize;
 for index in 0..70 {
  let value=count(index);
  if start {SNAPSHOTS[phase][index].store(value,Relaxed);}else{TOTALS[phase][index].fetch_add(value-SNAPSHOTS[phase][index].load(Relaxed),Relaxed);}
 }
}
pub fn report(){
 ACTIVE.store(false,Relaxed);
 let phases=["initial_count","initial_owner","prepare","apply","commit","drop"];
 for (phase,name) in phases.iter().enumerate(){
  let values:Vec<_>=TOTALS[phase].iter().map(|x|x.load(Relaxed)).collect();
  eprintln!("BPE_ALLOCATION_PHASE {}",serde_json::json!({"phase":name,"allocations":values[0],"deallocations":values[1],"reallocations":values[2],"allocated_bytes":values[3],"deallocated_bytes":values[4],"dealloc_size_histogram_64_is_ge64":&values[5..]}));
 }
 eprintln!("BPE_ALLOCATIONS {}",serde_json::json!({"allocations":COUNTS[0].load(Relaxed),"deallocations":COUNTS[1].load(Relaxed),"reallocations":COUNTS[2].load(Relaxed)}));
}
