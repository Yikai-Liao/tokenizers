//! A local kernel comparison, not a corpus training or scalability benchmark.
#[allow(dead_code)]
#[path = "../../tokenizers/tk-train/src/trainers/bpe/indexed/aa_parity.rs"]
mod aa;
use std::{hint::black_box, time::Instant};

#[derive(Clone, Copy)]
struct Plan {
    position: usize,
    _rank: usize,
}

fn measure<const NEAR: usize>(chunks: &[Vec<Plan>]) -> f64 {
    let begin = Instant::now();
    let mut digest = 0_usize;
    for pass in 0..128 {
        for j in 0..chunks.len() {
            let chunk = black_box(&chunks[(j * 73 + pass) % chunks.len()]);
            let summary =
                aa::summarize_hybrid::<NEAR>(chunk.len(), |i| chunk[i].position, 1).unwrap();
            digest ^= black_box(summary.last + usize::from(summary.trailing_odd));
        }
    }
    black_box(digest);
    begin.elapsed().as_secs_f64() * 1e9 / (128 * chunks.len()) as f64
}

fn main() {
    println!("pattern,binary_ns,linear_ns,hybrid8_ns,hybrid16_ns");
    for pattern in ["one_run", "short_tail", "long_tail", "mixed_tail"] {
        let chunks: Vec<Vec<Plan>> = (0..512)
            .map(|b| {
                let trailing = match pattern {
                    "one_run" => 4096,
                    "short_tail" => 1 + b % 4,
                    "long_tail" => 2048 + b % 1024,
                    _ => 1 + (b * 193) % 4095,
                };
                (0..4096)
                    .map(|i| Plan {
                        position: b * 10000 + i + usize::from(i >= 4096 - trailing) * 7,
                        _rank: 0,
                    })
                    .collect()
            })
            .collect();
        for chunk in &chunks {
            let expected = aa::summarize_hybrid::<0>(chunk.len(), |i| chunk[i].position, 1);
            assert_eq!(
                expected,
                aa::summarize_hybrid::<8>(chunk.len(), |i| chunk[i].position, 1)
            );
            assert_eq!(
                expected,
                aa::summarize_hybrid::<4096>(chunk.len(), |i| chunk[i].position, 1)
            );
        }
        // Warm the same working set. Rotating the order limits a simple linear
        // prefetch advantage; 16-byte Plan layout matches the actual trainer.
        let _ = measure::<8>(&chunks);
        println!(
            "{pattern},{:.2},{:.2},{:.2},{:.2}",
            measure::<0>(&chunks),
            measure::<4096>(&chunks),
            measure::<8>(&chunks),
            measure::<16>(&chunks)
        );
    }
}
