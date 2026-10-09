//! Opt-in, Linux-only diagnostics for the isolated BPE profiling experiment.
//! Events stay in thread-local memory; the runner writes them after timed training.
#[cfg(feature = "bpe-perfetto")]
mod enabled {
    use serde::Serialize;
    use std::{
        cell::RefCell,
        sync::{
            Mutex, OnceLock,
            atomic::{AtomicU64, Ordering},
        },
    };

    static ROUND: AtomicU64 = AtomicU64::new(0);
    static BUFFERS: Mutex<Vec<(u32, Vec<Event>)>> = Mutex::new(Vec::new());
    static LEVEL: OnceLock<u8> = OnceLock::new();
    pub fn level() -> u8 {
        *LEVEL.get_or_init(|| {
            std::env::var("BPE_TRACE_LEVEL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        })
    }
    pub fn next_round() {
        ROUND.fetch_add(1, Ordering::Relaxed);
    }
    pub fn now() -> u64 {
        let mut ts = std::mem::MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: successful clock_gettime initializes the writable timespec.
        assert_eq!(
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, ts.as_mut_ptr()) },
            0
        );
        // SAFETY: the successful call above initialized both fields.
        let ts = unsafe { ts.assume_init() };
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
    #[derive(Serialize)]
    struct Event {
        name: &'static str,
        ts: u64,
        dur: u64,
        round: u64,
        fields: [u64; 6],
    }
    struct Buffer {
        tid: u32,
        events: Vec<Event>,
    }
    impl Buffer {
        fn new() -> Self {
            // SAFETY: gettid takes no arguments and has no memory preconditions.
            Self {
                tid: unsafe { libc::gettid() } as u32,
                events: Vec::new(),
            }
        }
    }
    impl Drop for Buffer {
        fn drop(&mut self) {
            if !self.events.is_empty() {
                BUFFERS
                    .lock()
                    .unwrap()
                    .push((self.tid, std::mem::take(&mut self.events)));
            }
        }
    }
    thread_local! { static LOCAL: RefCell<Buffer> = RefCell::new(Buffer::new()); }
    pub struct Span {
        name: &'static str,
        start: u64,
        round: u64,
        pub fields: [u64; 6],
    }
    impl Span {
        #[inline]
        pub fn new(name: &'static str, min_level: u8, fields: [u64; 6]) -> Self {
            Self {
                name,
                start: if level() >= min_level { now() } else { 0 },
                round: ROUND.load(Ordering::Relaxed),
                fields,
            }
        }
    }
    impl Drop for Span {
        fn drop(&mut self) {
            if self.start != 0 {
                let end = now();
                LOCAL.with(|local| {
                    local.borrow_mut().events.push(Event {
                        name: self.name,
                        ts: self.start,
                        dur: end - self.start,
                        round: self.round,
                        fields: self.fields,
                    })
                });
            }
        }
    }
    pub fn collect() {
        LOCAL.with(|local| {
            let mut local = local.borrow_mut();
            if !local.events.is_empty() {
                BUFFERS
                    .lock()
                    .unwrap()
                    .push((local.tid, std::mem::take(&mut local.events)));
            }
        });
    }
    pub fn flush(path: &std::path::Path) -> std::io::Result<()> {
        collect();
        let data = std::mem::take(&mut *BUFFERS.lock().unwrap());
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        serde_json::to_writer(&mut file, &(std::process::id(), data))?;
        std::io::Write::flush(&mut file)
    }
}
#[cfg(feature = "bpe-perfetto")]
pub use enabled::*;
#[cfg(not(feature = "bpe-perfetto"))]
mod disabled {
    pub struct Span {
        pub fields: [u64; 6],
    }
    impl Span {
        #[inline(always)]
        pub fn new(_: &'static str, _: u8, fields: [u64; 6]) -> Self {
            Self { fields }
        }
    }
    #[inline(always)]
    pub fn next_round() {}
}
#[cfg(not(feature = "bpe-perfetto"))]
pub(crate) use disabled::*;
