#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
mod original;
mod standard;
use std::hint::black_box;
use std::time::Instant;

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn compare_against_uncompressed_slice() {
    let mut state = 723426;
    let mut queries = 0;
    for len in (0..=520).chain([1024, 4096, 65536]) {
        for mode in 0..5 {
            let values: Vec<u64> = (0..len)
                .map(|i| match mode {
                    0 => 7,
                    1 => (i / 257) as u64,
                    2 => (i as u64) * 7,
                    3 => (i as u64) * (u64::MAX / len.max(1) as u64),
                    _ => u64::MAX - (len - i) as u64,
                })
                .collect();
            let before = original::Positions::from_sorted(&values).unwrap();
            let after = standard::Positions::from_sorted(&values).unwrap();
            let mut targets = vec![0, 1, 6, 7, 8, u64::MAX];
            for &v in values.iter().step_by((len / 256).max(1)) {
                targets.extend([v.saturating_sub(1), v, v.saturating_add(1)]);
            }
            for _ in 0..64 {
                targets.push(next(&mut state));
            }
            for target in targets {
                let expected = values.partition_point(|&v| v < target);
                assert_eq!(
                    before.seek(target),
                    expected,
                    "original len={len} mode={mode} target={target}"
                );
                assert_eq!(
                    after.seek(target),
                    expected,
                    "standard len={len} mode={mode} target={target}"
                );
                queries += 1;
            }
        }
    }
    println!("Correctness: {queries} queries");
}

fn main() {
    const N: usize = 150_000;
    println!("len,mode,original_ns,standard_ns,ratio");
    for len in [2, 128, 129, 4096, 65536, 1_048_576] {
        for mode in ["distinct", "duplicates"] {
            let values: Vec<_> = (0..len)
                .map(|i| {
                    if mode == "distinct" {
                        i as u64 * 7
                    } else {
                        (i / 257) as u64
                    }
                })
                .collect();
            let before = original::Positions::from_sorted(&values).unwrap();
            let after = standard::Positions::from_sorted(&values).unwrap();
            let max = values.last().copied().unwrap() + 2;
            let mut state = 1234567;
            let queries: Vec<_> = (0..N).map(|_| next(&mut state) % max).collect();
            let mut times = [Vec::new(), Vec::new()];
            for repetition in 0..7 {
                for variant in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = Instant::now();
                    if variant == 0 {
                        for &q in &queries {
                            black_box(black_box(&before).seek(black_box(q)));
                        }
                    } else {
                        for &q in &queries {
                            black_box(black_box(&after).seek(black_box(q)));
                        }
                    }
                    times[variant].push(start.elapsed().as_nanos() as f64 / N as f64);
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "{len},{mode},{:.2},{:.2},{:.3}",
                times[0][3],
                times[1][3],
                times[1][3] / times[0][3]
            );
        }
    }
}
