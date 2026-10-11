use crate::{
    adapters::tk_train_v1::{Processor, trainer},
    model_output,
    protocol::{Error, Job, cpu_seconds, hwm},
};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};
use tk_train::Trainer;
pub fn run(job: &Job) -> Result<Value, Error> {
    let mut trainer = trainer(job);
    let processor = Processor::new(&job.pretokenizer)?;
    let input = BufReader::new(File::open(&job.input)?);
    let begin = Instant::now();
    let cpu = cpu_seconds()?;
    let faults_before = crate::protocol::faults()?;
    // Lazy line reading, public preprocessing and aggregation all belong to feed.
    trainer.feed(
        input.lines().map(|line| line.expect("input read failed")),
        |line| processor.process(line),
    )?;
    let feed_seconds = begin.elapsed().as_secs_f64();
    let feed_cpu_seconds = cpu_seconds()? - cpu;
    let feed_hwm_kib = hwm()?;
    let feed_rss_kib = crate::protocol::rss()?;
    let feed_faults = crate::protocol::faults()?;
    let feed_workers = tk_encode::parallelism::current_num_threads();
    let feed_unique_words = trainer.get_word_count();
    if std::env::var_os("BPE_FEED_COST_PROBE").is_some() {
        return Ok(json!({"metrics": {"feed_seconds": feed_seconds, "feed_cpu_seconds": feed_cpu_seconds,
            "feed_hwm_kib": feed_hwm_kib, "feed_rss_kib": feed_rss_kib,
            "feed_unique_words": feed_unique_words, "feed_workers": feed_workers},
            "diagnostic": "feed-only capacity probe; no training/count export/model validation"}));
    }
    let train_begin = Instant::now();
    let train_cpu = cpu_seconds()?;
    let (vocab, merges, special) = trainer.train_vocab()?;
    let train_seconds = train_begin.elapsed().as_secs_f64();
    let train_cpu_seconds = cpu_seconds()? - train_cpu;
    let pipeline_seconds = begin.elapsed().as_secs_f64();
    let pipeline_cpu_seconds = cpu_seconds()? - cpu;
    let process_hwm_kib_before_validation = hwm()?;
    let train_rss_kib = crate::protocol::rss()?;
    let pipeline_faults = crate::protocol::faults()?;
    // Export after the timed interval and all pre-validation memory snapshots.
    // JSON-encoded keys keep records one-line; external sort yields a canonical
    // count hash without allocating a second dictionary or copying all keys.
    let counts_path = std::path::Path::new(&job.output).with_extension("counts.tsv");
    let feed_unique_utf8_bytes =
        crate::feed_stats::export(&trainer, feed_unique_words, &counts_path)?;
    model_output::write(&job.output, &vocab, &merges)?;
    Ok(
        json!({"metrics":{"feed_seconds":feed_seconds,"feed_cpu_seconds":feed_cpu_seconds,"feed_workers":feed_workers,
        "feed_hwm_kib":feed_hwm_kib,"feed_rss_kib":feed_rss_kib,"train_rss_kib":train_rss_kib,
        "feed_minor_faults":feed_faults.0-faults_before.0,"feed_major_faults":feed_faults.1-faults_before.1,
        "pipeline_minor_faults":pipeline_faults.0-faults_before.0,"pipeline_major_faults":pipeline_faults.1-faults_before.1,
        "train_seconds":train_seconds,
        "pipeline_seconds":pipeline_seconds,"pipeline_cpu_seconds":pipeline_cpu_seconds,
        "train_cpu_seconds":train_cpu_seconds,"process_hwm_kib_before_validation":process_hwm_kib_before_validation,
        "feed_unique_words":feed_unique_words,"feed_unique_utf8_bytes":feed_unique_utf8_bytes},
        "output":{"counts_path":counts_path,"model_path":job.output,"actual_vocab":vocab.len(),"actual_merges":merges.len(),"special_tokens":special.len()},
        "timing_boundary":"public-feed-and-train-before-serialization"}),
    )
}
