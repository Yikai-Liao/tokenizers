#[path = "source/binary-ranges.rs"]
mod binary;
#[path = "source/linear-ranges.rs"]
mod linear;
#[path = "source/exponential-ranges.rs"]
mod exponential;

use std::{hint::black_box, ops::Range, time::Instant};

fn timed(iterations: usize, call: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations { call(); }
    start.elapsed().as_secs_f64()
}

fn calibrate(call: &mut impl FnMut()) -> usize {
    let mut iterations = 1;
    loop {
        let elapsed = timed(iterations, call);
        if elapsed >= 0.02 {
            return ((iterations as f64 * 0.08 / elapsed) as usize).max(1);
        }
        iterations *= 2;
    }
}

fn compare(name: &str, n: usize, width: usize, slots: usize) {
    let mut length = 1usize;
    let starts: Vec<_> = (0..n).map(|i| {
        let start = length;
        length += width + i % 3;
        start
    }).collect();
    let binary = binary::CorpusPlan { starts: &starts, length, words: n };
    let linear = linear::CorpusPlan { starts: &starts, length, words: n };
    let exponential = exponential::CorpusPlan { starts: &starts, length, words: n };
    let ranges = binary.initial_ranges(slots);
    assert_eq!(ranges, linear.initial_ranges(slots));
    assert_eq!(ranges, exponential.initial_ranges(slots));
    let mut binary_call = || { black_box(black_box(&binary).initial_ranges(black_box(slots))); };
    let mut linear_call = || { black_box(black_box(&linear).initial_ranges(black_box(slots))); };
    let mut exponential_call = || { black_box(black_box(&exponential).initial_ranges(black_box(slots))); };
    // Calibrate each arm separately so even sub-microsecond calls get timed for
    // roughly 80 ms. Every arm uses the same shared layout and allocation scope.
    let iterations = [calibrate(&mut binary_call), calibrate(&mut linear_call), calibrate(&mut exponential_call)];
    let orders = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [1, 0, 2], [0, 2, 1]];
    for pair in 0..13 {
        for arm in orders[pair % orders.len()] {
            let elapsed = match arm {
                0 => timed(iterations[arm], &mut binary_call),
                1 => timed(iterations[arm], &mut linear_call),
                _ => timed(iterations[arm], &mut exponential_call),
            };
            println!("{name},{n},{width},{slots},{},{pair},{},{},{:.4}",
                ranges.len(), ["binary", "linear", "exponential"][arm], iterations[arm],
                elapsed * 1e9 / iterations[arm] as f64);
        }
    }
}

fn check(starts: &[usize], length: usize, slots: usize, expected: Option<Vec<Range<usize>>>) {
    let n = starts.len();
    let binary = binary::CorpusPlan { starts, length, words: n };
    let linear = linear::CorpusPlan { starts, length, words: n };
    let exponential = exponential::CorpusPlan { starts, length, words: n };
    let result = exponential.initial_ranges(slots);
    assert_eq!(result, binary.initial_ranges(slots));
    assert_eq!(result, linear.initial_ranges(slots));
    if let Some(expected) = expected { assert_eq!(result, expected); }
}

fn correctness() {
    for n in [0, 1, 2, 3, 7, 8, 9, 17, 127, 128, 129, 257] {
        for width in [1, 16, 65, (1 << 24) + 1] {
            let starts: Vec<_> = (0..n).map(|i| 1 + i * width).collect();
            for slots in [0, 1, 2, 16, 17, 32, 64, 65, 1 << 24, usize::MAX] {
                check(&starts, 1 + n * width, slots, None);
            }
        }
    }
    for slots in [0, 1, 3, 5, 8, usize::MAX] {
        check(&[usize::MAX - 8, usize::MAX - 3], usize::MAX, slots, None);
    }
    check(&[1, 4, 8, 10], 16, 7, Some(vec![0..2, 2..3, 3..4]));
    check(&[1, 4, 8, 10], 16, 9, Some(vec![0..3, 3..4]));
}

fn main() {
    correctness();
    if std::env::args().nth(1).as_deref() == Some("check") { return; }
    println!("case,words,width,slots,ranges,pair,arm,iterations,ns_per_call");
    for (name, n, width, slots) in [
        ("empty", 0, 16, 1 << 24),
        ("single-word", 1, 16, 1 << 24),
        ("all-fit-100k", 100_000, 16, 1 << 24),
        ("large-segments-2m", 2_000_000, 16, 1 << 24),
        ("one-word-per-segment-100k", 100_000, 16, 32),
        ("few-words-per-segment-100k", 100_000, 16, 64),
        ("medium-budget-100k", 100_000, 16, 4096),
        ("test-budget-100k", 100_000, 16, 16),
        ("oversized-100k", 100_000, (1 << 24) + 1, 1 << 24),
        ("zero-budget-100k", 100_000, 16, 0),
    ] { compare(name, n, width, slots); }
}
