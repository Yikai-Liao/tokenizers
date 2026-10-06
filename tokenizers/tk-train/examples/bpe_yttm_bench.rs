//! The same public-API runner can be copied into the pinned upstream checkout.
use std::{collections::BTreeMap, env, fs, time::Instant};
use tk_train::{BpeTrainer, Trainer};

fn main() -> tk_encode::Result<()> {
    let args: Vec<_> = env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: bpe_yttm_bench INPUT VOCAB_SIZE WORKERS [MODEL_JSON]".into());
    }
    let size: usize = args[2].parse()?;
    let workers: usize = args[3].parse()?;
    if workers == 0 {
        return Err("WORKERS must be at least 1".into());
    }
    tk_encode::parallelism::set_num_threads(workers);
    tk_encode::parallelism::set_parallelism(workers > 1);
    let start = Instant::now();
    let corpus = fs::read_to_string(&args[1])?;
    let bytes = corpus.len();
    let read_seconds = start.elapsed().as_secs_f64();
    let mut trainer = BpeTrainer::builder()
        .vocab_size(size)
        .show_progress(false)
        .build();
    let feed_start = Instant::now();
    trainer.feed(corpus.lines(), |line| {
        Ok(line.split_whitespace().map(str::to_owned).collect())
    })?;
    let feed_seconds = feed_start.elapsed().as_secs_f64();
    drop(corpus);
    let train_start = Instant::now();
    let (vocab, merges, _) = trainer.train_vocab()?;
    let train_seconds = train_start.elapsed().as_secs_f64();
    let total_seconds = start.elapsed().as_secs_f64();
    let peak_rss_kib = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<u64>().ok())
        });
    println!(
        "{}",
        serde_json::json!({
            "bytes": bytes, "workers": workers, "unique_words": trainer.get_word_count(),
            "vocab_size": vocab.len(), "merges": merges.len(),
            "read_seconds": read_seconds, "feed_seconds": feed_seconds,
            "train_seconds": train_seconds, "total_seconds": total_seconds,
            "peak_rss_kib": peak_rss_kib,
        })
    );
    if let Some(path) = args.get(4) {
        let vocab: BTreeMap<_, _> = vocab.into_iter().collect();
        fs::write(
            path,
            serde_json::to_vec(&serde_json::json!({"vocab": vocab, "merges": merges}))?,
        )?;
    }
    Ok(())
}
