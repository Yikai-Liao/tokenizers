#![allow(dead_code)]
use std::{alloc::{GlobalAlloc, Layout, System}, io::{self, Write}, hint::black_box, sync::atomic::{AtomicBool, AtomicUsize, Ordering}, time::Instant};
#[path = "/tmp/bpe-cleanup-positions.rs"] mod positions;
use positions::{Codec, Input, Positions};
use itertools::Itertools;
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

struct Directory { starts: Vec<usize>, length: usize }
impl Directory {
    // Exact resident validation and word lookup from Corpus; token storage is
    // unnecessary for this collection-only measurement.
    fn resident(&self, coordinate: u64) -> usize {
        let position = usize::try_from(coordinate).expect("corpus coordinate fits resident indexing");
        assert!(position < self.length, "coordinate belongs to the resident corpus");
        position
    }
    fn word(&self, position: usize) -> usize {
        self.starts.partition_point(|&start| start <= position) - 1
    }
}
fn collect(positions: &Positions, corpus: &Directory, early: bool) -> Vec<usize> {
    if early {
        positions.iter().map(|p| corpus.word(corpus.resident(p))).dedup().collect()
    } else {
        let mut words: Vec<_> = positions.iter().map(|p| corpus.word(corpus.resident(p))).collect();
        words.dedup(); words
    }
}
fn measure(positions: &Positions, corpus: &Directory, early: bool, reps: usize) -> serde_json::Value {
    let start_cpu=cpu(); let start=Instant::now(); let mut count=0;
    for _ in 0..reps { let words=collect(black_box(positions),black_box(corpus),early);count+=words.len();black_box(words); }
    serde_json::json!({"cpu_seconds":cpu()-start_cpu,"seconds":start.elapsed().as_secs_f64(),"repetitions":reps,"total_words":count})
}
fn allocations(positions: &Positions, corpus: &Directory, early: bool) -> serde_json::Value {
    ALLOC.store(0,Ordering::Relaxed);REALLOC.store(0,Ordering::Relaxed);BYTES.store(0,Ordering::Relaxed);
    TRACK.store(true,Ordering::Relaxed); let words=collect(positions,corpus,early);TRACK.store(false,Ordering::Relaxed);
    serde_json::json!({"allocations":ALLOC.load(Ordering::Relaxed),"reallocations":REALLOC.load(Ordering::Relaxed),"requested_bytes_including_reallocations":BYTES.load(Ordering::Relaxed),"retained_capacity_bytes":words.capacity()*std::mem::size_of::<usize>(),"unique_words":words.len()})
}
fn run(case: &str, per_word: usize) -> serde_json::Value {
    let n=262144; let coords: Vec<_>=(0..n).map(|p|p as u64).collect();
    let corpus=Directory{starts:if per_word==0 { (0..n).filter(|p|p%33!=32).collect() } else { (0..n).step_by(per_word).collect() },length:n};
    let codec=Codec::new(1);let positions=Positions::from_sorted(Input::Slice(&coords),&mut codec.lease()).unwrap();drop(coords);
    assert_eq!(collect(&positions,&corpus,false),collect(&positions,&corpus,true));
    let a=allocations(&positions,&corpus,false);let b=allocations(&positions,&corpus,true);
    measure(&positions,&corpus,false,10);measure(&positions,&corpus,true,10);
    let mut samples=Vec::new();
    for block in 0..6 {
        let (baseline,candidate)=if block%2==0 {(measure(&positions,&corpus,false,50),measure(&positions,&corpus,true,50))} else {let c=measure(&positions,&corpus,true,50);(measure(&positions,&corpus,false,50),c)};
        samples.push(serde_json::json!({"block":block,"order":if block%2==0 {"AB"} else {"BA"},"baseline":baseline,"candidate":candidate}));
    }
    serde_json::json!({"case":case,"positions":n,"per_word":per_word,"same_ordered_output":true,"baseline_allocation":a,"candidate_allocation":b,"samples":samples})
}
fn main() {
    println!("{}",serde_json::json!({"scope":"Actual Positions codec and iterator; exact Corpus resident checks / word partition_point copied, no token plane needed. Collection only, not whole reuse training; setup and validation excluded.","cases":[run("no duplicates",1),run("one duplicate per 33 positions",0),run("heavy duplicates",64)]}));
}
