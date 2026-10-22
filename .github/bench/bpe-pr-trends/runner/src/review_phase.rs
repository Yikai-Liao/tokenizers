
//! Benchmark-copy-only phase boundaries. This file is never committed to the PR series.
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) struct Phase(&'static str);

pub(crate) fn event(name: &str, action: &str) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },0);
    let peak = unsafe { usage.assume_init() }.ru_maxrss;
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let field = |name: &str| status.lines().find(|s| s.starts_with(name))
        .and_then(|s|s.split_whitespace().nth(1)).unwrap().parse::<u64>().unwrap();
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    eprintln!("BPE_REVIEW_PHASE {{\"phase\":\"{name}\",\"action\":\"{action}\",\"unix_ns\":{ns},\"rss_kib\":{},\"hwm_kib\":{},\"swap_kib\":{},\"ru_maxrss_kib\":{peak}}}",
        field("VmRSS:"),field("VmHWM:"),field("VmSwap:"));
}

impl Phase {
    pub(crate) fn new(name: &'static str) -> Self {
        event(name, "begin");
        Self(name)
    }
}

impl Drop for Phase {
    fn drop(&mut self) { event(self.0, "end"); }
}
