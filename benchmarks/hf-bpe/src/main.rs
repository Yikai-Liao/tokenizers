//! Both backends share the same feed, parameters, build and process boundaries.
use hf_front_end::pre_tokenizers::byte_level::ByteLevel;
use hf_front_end::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};
#[cfg(feature = "indexed")]
use tk_train::IndexedParallelConfig;
use tk_train::{BpeTrainer, Trainer};

struct Lines<R>(R);
impl<R: BufRead> Iterator for Lines<R> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        let mut line = String::new();
        match self.0.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => Some(line),
            Err(e) => panic!("read line: {e}"),
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() == 6,
        "usage: hf-bpe-indexed-bench INPUT SPLIT BACKEND VOCAB MIN_FREQ"
    );
    let (input, split, backend) = (&args[1], &args[2], &args[3]);
    assert!(matches!(
        split.as_str(),
        "none" | "whitespace_split" | "bytelevel"
    ));
    let mut trainer = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(args[4].parse()?)
        .min_frequency(args[5].parse()?)
        .build();
    let reader = BufReader::new(File::open(input)?);
    let begin = Instant::now();
    let bytelevel = ByteLevel::new(false, true, true);
    trainer.feed(Lines(reader), |s| {
        if split == "bytelevel" {
            let mut text = PreTokenizedString::from(s);
            bytelevel.pre_tokenize(&mut text)?;
            return Ok(text
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(part, _, _)| part.to_owned())
                .collect());
        }
        Ok(if split == "none" {
            vec![s.to_owned()]
        } else {
            s.split_whitespace().map(str::to_owned).collect()
        })
    })?;
    let feed_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let (vocab, merges, stats) = match backend.as_str() {
        "reference" => {
            let (v, m, _) = trainer.train_vocab()?;
            (v, m, None::<serde_json::Value>)
        }
        #[cfg(feature = "indexed")]
        "indexed" | "fused" => {
            let out = if backend == "fused" {
                trainer.train_vocab_fused()?
            } else {
                trainer.train_vocab_indexed()?
            };
            (
                out.vocab,
                out.merges,
                Some(serde_json::to_value(out.stats)?),
            )
        }
        #[cfg(feature = "indexed")]
        "parallel1" | "parallel4" | "dict16" | "narrow16" | "narrow32par1" | "narrow32par4"
        | "atomic32par4" | "count1par4" => {
            let out = trainer.train_vocab_indexed_parallel(IndexedParallelConfig {
                workers: if matches!(backend.as_str(), "parallel1" | "narrow32par1") {
                    1
                } else {
                    4
                },
                initialization_workers: if backend == "count1par4" {
                    Some(1)
                } else {
                    None
                },
                posting_block_bits: if matches!(backend.as_str(), "dict16" | "narrow16") {
                    16
                } else {
                    32
                },
                narrow_corpus: matches!(
                    backend.as_str(),
                    "narrow16" | "narrow32par1" | "narrow32par4" | "atomic32par4"
                ),
                atomic_corpus: backend == "atomic32par4",
                batch_size: 256,
            })?;
            (
                out.vocab,
                out.merges,
                Some(serde_json::to_value(out.stats)?),
            )
        }
        _ => panic!("unknown backend"),
    };
    let elapsed_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let mut digest = Sha256::new();
    let mut entries: Vec<_> = vocab.iter().collect();
    entries.sort_by_key(|(_, id)| **id);
    for (s, id) in entries {
        digest.update(id.to_le_bytes());
        digest.update((s.len() as u32).to_le_bytes());
        digest.update(s.as_bytes());
    }
    for (a, b) in &merges {
        digest.update(2_u32.to_le_bytes());
        for s in [a, b] {
            digest.update((s.len() as u32).to_le_bytes());
            digest.update(s.as_bytes());
        }
    }
    let peak_rss: u64 = std::fs::read_to_string("/proc/self/status")?
        .lines()
        .find(|s| s.starts_with("VmHWM:"))
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()?;
    println!(
        "{}",
        json!({"backend":backend, "input":input, "input_bytes":std::fs::metadata(input)?.len(),
        "split":split, "vocab_size":trainer.vocab_size, "min_frequency":trainer.min_frequency,
        "elapsed_ms":elapsed_ms, "feed_ms":feed_ms, "train_ms":elapsed_ms-feed_ms,
        "unique_words":trainer.get_word_count(), "actual_vocab":vocab.len(), "actual_merges":merges.len(),
        "maxrss_kib":peak_rss, "model_sha256":format!("{:x}", digest.finalize()), "indexed_stats":stats })
    );
    Ok(())
}
