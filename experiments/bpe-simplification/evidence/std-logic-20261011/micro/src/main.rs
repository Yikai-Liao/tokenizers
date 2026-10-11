#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
mod bitmap_original;
mod bitmap_standard;
mod positions_original;
mod positions_standard;
mod ranges_original;
mod ranges_standard;
use std::{hint::black_box, time::Instant};

// Only the keys are used by the extracted observed_characters function.
#[derive(Clone, Copy)]
struct WordCountsView<'a> { words: &'a [String] }
impl<'a> WordCountsView<'a> {
    fn keys(self) -> impl Iterator<Item = &'a str> {
        self.words.iter().map(String::as_str)
    }
}

fn timed(iterations: usize, call: &mut impl FnMut() -> u64) -> f64 {
    let begin = Instant::now();
    for _ in 0..iterations { black_box(call()); }
    begin.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

fn compare(name: &str, workers: usize, iterations: usize,
    mut original: impl FnMut() -> u64, mut standard: impl FnMut() -> u64) {
    black_box(original());
    black_box(standard());
    for rep in 0..7 {
        for arm in if rep % 2 == 0 { [false, true] } else { [true, false] } {
            let ns = if arm { timed(iterations, &mut standard) }
                     else { timed(iterations, &mut original) };
            println!("{name},{workers},{rep},{},{iterations},{ns:.4}",
                if arm { "standard" } else { "original" });
        }
    }
}

fn layout(n: usize, width: usize) -> (Vec<usize>, usize) {
    let mut starts = Vec::with_capacity(n);
    let mut length = 1;
    for i in 0..n {
        starts.push(length);
        length += width + i % 3;
    }
    (starts, length)
}

fn ranges() {
    for (name, n, width, slots, iterations) in [
        ("large-segments", 2_000_000, 16, 1 << 24, 64),
        ("small-segments", 100_000, 16, 32, 32),
        ("oversized-words", 100_000, (1 << 24) + 1, 1 << 24, 32),
    ] {
        let (starts, length) = layout(n, width);
        let original = ranges_original::CorpusPlan::new(starts.clone(), length);
        let standard = ranges_standard::CorpusPlan::new(starts, length);
        assert_eq!(original.initial_ranges(slots), standard.initial_ranges(slots));
        compare(&format!("ranges/{name}"), 1, iterations,
            || black_box(black_box(&original).initial_ranges(black_box(slots))).len() as u64,
            || black_box(black_box(&standard).initial_ranges(black_box(slots))).len() as u64);
    }
}

fn bitmap() {
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
        for (name, n, iterations) in [("empty", 0, 128), ("small", 64, 64), ("large", 500_000, 3)] {
            let words: Vec<_> = (0..n).map(|i| format!("中文{i}é🙂𐀀")).collect();
            let view = WordCountsView { words: &words };
            pool.install(|| {
                assert_eq!(bitmap_original::observe(view, workers), bitmap_standard::observe(view, workers));
                compare(&format!("bitmap/{name}"), workers, iterations,
                    || black_box(bitmap_original::observe(black_box(view), workers)).len() as u64,
                    || black_box(bitmap_standard::observe(black_box(view), workers)).len() as u64);
            });
        }
    }
}

fn positions() {
    for length in [2, 128, 129, 4096, 65_536, 1_048_576] {
        for duplicates in [false, true] {
            let values: Vec<_> = (0..length).map(|i| if duplicates { (i / 200) as u64 }
                else { (u64::MAX - length as u64) + i as u64 }).collect();
            let original = positions_original::Positions::from_sorted(&values).unwrap();
            let standard = positions_standard::Positions::from_sorted(&values).unwrap();
            assert!(original.iter().eq(standard.iter()));
            let iterations = (4_000_000 / length).clamp(4, 10_000);
            compare(&format!("positions/{length}/{}", if duplicates {"duplicates"} else {"distinct"}),
                1, iterations,
                || black_box(&original).iter().fold(0u64, u64::wrapping_add),
                || black_box(&standard).iter().fold(0u64, u64::wrapping_add));
        }
    }
}

fn main() {
    println!("case,workers,rep,arm,iterations,ns_per_call");
    match std::env::args().nth(1).as_deref() {
        Some("ranges") => ranges(), Some("bitmap") => bitmap(), Some("positions") => positions(),
        _ => panic!("expected ranges, bitmap or positions"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_ranges_preserve_whole_word_boundaries() {
        for n in [0, 1, 2, 3, 17, 128, 257] {
            for width in [1, 16, 65, 4096, (1 << 24) + 1] {
                let (starts, length) = layout(n, width);
                let original = ranges_original::CorpusPlan::new(starts.clone(), length);
                let standard = ranges_standard::CorpusPlan::new(starts, length);
                for slots in [0, 1, 2, 16, 17, 32, 64, 65, 4096, 1 << 24, usize::MAX] {
                    assert_eq!(original.initial_ranges(slots), standard.initial_ranges(slots));
                }
            }
        }
    }

    #[test]
    fn extracted_bitmaps_preserve_full_domain_and_empty_output() {
        for n in [0, 1, 2, 3, 17, 128] {
            let words: Vec<_> = (0..n).map(|i| format!("字符{i}é🙂𐀀")).collect();
            let view = WordCountsView { words: &words };
            for workers in [1, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
                pool.install(|| {
                    let result = bitmap_standard::observe(view, workers);
                    assert_eq!(result.len(), 0x110000);
                    assert_eq!(bitmap_original::observe(view, workers), result);
                });
            }
        }
    }
}
