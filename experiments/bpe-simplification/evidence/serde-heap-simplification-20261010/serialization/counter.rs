#![allow(dead_code)]
use std::{alloc::{GlobalAlloc, Layout, System}, io::{self, Write}, hint::black_box, sync::atomic::{AtomicBool, AtomicUsize, Ordering}, time::Instant};
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use serde::Serialize;
#[path = "/tmp/bpe-serde-peekmut-old-word-counts.rs"] mod old;
#[path = "/tmp/bpe-serde-peekmut-new-word-counts.rs"] mod new;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOC: AtomicUsize = AtomicUsize::new(0);
static FREE: AtomicUsize = AtomicUsize::new(0);
static REALLOC: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct Allocator;
// SAFETY: Every operation delegates the original pointer and layout to System;
// counters observe requests and never affect allocation or ownership.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) { ALLOC.fetch_add(1,Ordering::Relaxed); BYTES.fetch_add(layout.size(),Ordering::Relaxed); }
        // SAFETY: The caller supplies GlobalAlloc's valid layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) { ALLOC.fetch_add(1,Ordering::Relaxed); BYTES.fetch_add(layout.size(),Ordering::Relaxed); }
        // SAFETY: The caller supplies GlobalAlloc's valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.load(Ordering::Relaxed) { FREE.fetch_add(1,Ordering::Relaxed); }
        // SAFETY: The caller supplies the live allocation and its layout.
        unsafe { System.dealloc(ptr,layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) { REALLOC.fetch_add(1,Ordering::Relaxed); BYTES.fetch_add(size,Ordering::Relaxed); }
        // SAFETY: The caller supplies GlobalAlloc's valid realloc arguments.
        unsafe { System.realloc(ptr,layout,size) }
    }
}
#[global_allocator] static GLOBAL: Allocator = Allocator;
#[derive(Default)] struct Counter { bytes: usize }
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let length=black_box(bytes).len(); self.bytes+=length; Ok(length)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
fn cpu() -> f64 {
    let mut t=libc::timespec {tv_sec:0,tv_nsec:0};
    // SAFETY: A valid writable timespec is passed for the supported process clock.
    assert_eq!(unsafe {libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID,&mut t)},0);
    t.tv_sec as f64 + t.tv_nsec as f64 / 1e9
}
fn allocation_counts<T: Serialize>(value: &T, size: usize) -> serde_json::Value {
    ALLOC.store(0,Ordering::Relaxed); FREE.store(0,Ordering::Relaxed); REALLOC.store(0,Ordering::Relaxed); BYTES.store(0,Ordering::Relaxed);
    let mut writer=Counter::default(); TRACK.store(true,Ordering::Relaxed);
    let result=serde_json::to_writer(&mut writer,black_box(value)); TRACK.store(false,Ordering::Relaxed);
    result.unwrap(); assert_eq!(writer.bytes,size);
    serde_json::json!({"allocations":ALLOC.load(Ordering::Relaxed),"deallocations":FREE.load(Ordering::Relaxed),"reallocations":REALLOC.load(Ordering::Relaxed),"requested_bytes":BYTES.load(Ordering::Relaxed)})
}
fn time<T: Serialize>(value: &T, size: usize, repetitions: usize) -> serde_json::Value {
    let start_cpu=cpu(); let start=Instant::now(); let mut writer=Counter::default();
    for _ in 0..repetitions { serde_json::to_writer(&mut writer,black_box(value)).unwrap(); }
    let seconds=start.elapsed().as_secs_f64(); let cpu_seconds=cpu()-start_cpu;
    assert_eq!(writer.bytes,size*repetitions); black_box(writer.bytes);
    serde_json::json!({"seconds":seconds,"cpu_seconds":cpu_seconds,"repetitions":repetitions,"bytes":writer.bytes})
}
fn run<T: Serialize,U: Serialize>(case: &str, before: &T, after: &U) -> serde_json::Value {
    let a=serde_json::to_vec(before).unwrap(); let b=serde_json::to_vec(after).unwrap(); assert_eq!(a,b,"JSON bytes for {case}"); let size=a.len(); drop((a,b));
    let before_alloc=allocation_counts(before,size); let after_alloc=allocation_counts(after,size);
    time(before,size,2); time(after,size,2);
    let mut samples=Vec::new();
    for block in 0..6 {
        let (baseline,candidate)=if block%2==0 {(time(before,size,10),time(after,size,10))} else {let c=time(after,size,10);(time(before,size,10),c)};
        samples.push(serde_json::json!({"block":block,"order":if block%2==0 {"AB"} else {"BA"},"baseline":baseline,"candidate":candidate}));
    }
    serde_json::json!({"case":case,"unique_entries":262144,"json_bytes":size,"byte_equal":true,"baseline_allocations":before_alloc,"candidate_allocations":after_alloc,"samples":samples})
}
fn main() {
    let entries: Vec<(CompactString,u64)>=(0..262144).map(|i|(format!("词{i:08}").into(),(i%17+1) as u64)).collect();
    let before_entries=old::WordCounts::from_entries(entries.clone()); let after_entries=new::WordCounts::from_entries(entries.clone());
    let mut map=AHashMap::with_capacity_and_hasher(entries.len(),RandomState::with_seeds(11,13,17,19));map.extend(entries);
    let before_map=old::WordCounts::from_map(map.clone());let after_map=new::WordCounts::from_map(map);
    println!("{}",serde_json::json!({"scope":"Actual WordCounts Serialize sources, JSON to nonallocating byte-counting writer; construction and validation excluded","profile":"opt-level=3 lto=fat codegen-units=1","samples":[run("Entries",&before_entries,&after_entries),run("Map",&before_map,&after_map)]}));
}
