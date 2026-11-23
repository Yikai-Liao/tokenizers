use crate::{adapters::{trainer, Processor}, model_output, protocol::{cpu_seconds, Error, Job}};
use serde_json::json;
use std::{fs::File, io::{BufRead, BufReader}, time::Instant};
use tk_train::Trainer;

pub fn run(job: &Job) -> Result<serde_json::Value, Error> {
    let mut trainer = trainer(job);
    let processor = Processor::new(&job.pretokenizer)?;
    let input = BufReader::new(File::open(&job.input)?);
    let begin = Instant::now();
    let cpu = cpu_seconds()?;
    trainer.feed(input.lines().map(|line| line.expect("input read failed")), |line| processor.process(line))?;
    let feed_seconds = begin.elapsed().as_secs_f64();
    let feed_cpu_seconds = cpu_seconds()? - cpu;
    let train_begin = Instant::now();
    let train_cpu = cpu_seconds()?;
    let (vocab, merges, special) = trainer.train_vocab()?;
    let train_seconds = train_begin.elapsed().as_secs_f64();
    let train_cpu_seconds = cpu_seconds()? - train_cpu;
    model_output::write(&job.output, &vocab, &merges)?;
    Ok(json!({
        "metrics": {
            "feed_seconds": feed_seconds,
            "feed_cpu_seconds": feed_cpu_seconds,
            "train_seconds": train_seconds,
            "train_cpu_seconds": train_cpu_seconds,
            "pipeline_seconds": begin.elapsed().as_secs_f64(),
            "pipeline_cpu_seconds": cpu_seconds()? - cpu
        },
        "output": {"actual_vocab": vocab.len(), "actual_merges": merges.len(), "special_tokens": special.len()}
    }))
}
