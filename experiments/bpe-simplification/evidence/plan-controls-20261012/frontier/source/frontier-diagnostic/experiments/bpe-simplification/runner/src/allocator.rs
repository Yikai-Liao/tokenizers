//! Experimental System allocator counters, used in diagnostic binaries only.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
pub static ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
fn added(bytes: u64) { let live = LIVE.fetch_add(bytes, Relaxed) + bytes; if ACTIVE.load(Relaxed) { PEAK.fetch_max(live, Relaxed); } }
pub fn start() { PEAK.store(LIVE.load(Relaxed), Relaxed); ACTIVE.store(true, Relaxed); }
static ALLOC: AtomicU64 = AtomicU64::new(0);
static FREE: AtomicU64 = AtomicU64::new(0);
static REALLOC: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static FREE_BYTES: AtomicU64 = AtomicU64::new(0);
struct Counters;
#[global_allocator] static GLOBAL: Counters = Counters;
// SAFETY: every operation forwards the caller's valid allocation/layout contract
// unchanged to System; counters neither inspect nor modify allocated memory.
unsafe impl GlobalAlloc for Counters {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() { added(layout.size() as u64); }
        if ACTIVE.load(Relaxed) && !ptr.is_null() { ALLOC.fetch_add(1, Relaxed); ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed); }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
        if ACTIVE.load(Relaxed) { FREE.fetch_add(1, Relaxed); FREE_BYTES.fetch_add(layout.size() as u64, Relaxed); }
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            if size >= layout.size() { added((size - layout.size()) as u64); }
            else { LIVE.fetch_sub((layout.size()-size) as u64, Relaxed); }
        }
        if ACTIVE.load(Relaxed) && !new.is_null() { REALLOC.fetch_add(1, Relaxed); ALLOC_BYTES.fetch_add(size as u64, Relaxed); FREE_BYTES.fetch_add(layout.size() as u64, Relaxed); }
        new
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() { added(layout.size() as u64); }
        if ACTIVE.load(Relaxed) && !ptr.is_null() { ALLOC.fetch_add(1, Relaxed); ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed); }
        ptr
    }
}
pub fn report() {
    ACTIVE.store(false, Relaxed);
    eprintln!("BPE_ALLOCATIONS {}", serde_json::json!({"live_requested_bytes_at_report":LIVE.load(Relaxed),"peak_live_requested_bytes_during_training":PEAK.load(Relaxed),"allocations":ALLOC.load(Relaxed),"reallocations":REALLOC.load(Relaxed),"frees":FREE.load(Relaxed),"requested_bytes":ALLOC_BYTES.load(Relaxed),"freed_bytes":FREE_BYTES.load(Relaxed)}));
}
