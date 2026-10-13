use crate::{
    adapters::tk_train_v1::{Processor, trainer},
    model_output,
    protocol::{Error, Job, PreparedWords, cpu_seconds, hwm},
};
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

pub fn prepare(job: &Job) -> Result<Value, Error> {
    let begin = Instant::now();
    let cpu = cpu_seconds()?;
    let processor = Processor::new(&job.pretokenizer)?;
    let mut counts = AHashMap::<CompactString, u64>::new();
    for line in BufReader::new(File::open(&job.input)?).lines() {
        for word in processor.process(&line?)? {
            *counts.entry(word.into()).or_default() += 1;
        }
    }
    // A fixed lexical reconstruction order gives every arm the same caller map.
    let mut ordered_words: Vec<_> = counts.into_iter().collect();
    ordered_words.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let data = PreparedWords {
        schema_version: 1,
        pretokenizer: job.pretokenizer.clone(),
        ordered_words,
        hash_seeds: [11, 13, 17, 19],
    };
    let preprocessing_seconds = begin.elapsed().as_secs_f64();
    let preprocessing_cpu_seconds = cpu_seconds()? - cpu;
    let mut writer = BufWriter::new(File::create(&job.output)?);
    serde_json::to_writer(&mut writer, &data)?;
    writer.flush()?;
    Ok(json!({"unique_words":data.ordered_words.len(),
        "preprocessing_seconds":preprocessing_seconds,
        "preprocessing_cpu_seconds":preprocessing_cpu_seconds}))
}
pub fn run(job: &Job) -> Result<Value, Error> {
    let load = Instant::now();
    let prepared: PreparedWords = serde_json::from_reader(BufReader::new(File::open(&job.input)?))?;
    if prepared.schema_version != 1
        || prepared.pretokenizer != job.pretokenizer
        || prepared.hash_seeds != [11, 13, 17, 19]
    {
        return Err("prepared input protocol mismatch".into());
    }
    let mut words = AHashMap::with_hasher(RandomState::with_seeds(11, 13, 17, 19));
    let mut previous: Option<CompactString> = None;
    for (word, count) in prepared.ordered_words {
        if previous.as_ref().is_some_and(|p| p >= &word) {
            return Err("prepared words not strictly ordered".into());
        }
        previous = Some(word.clone());
        words.insert(word, count);
    }
    let load_seconds = load.elapsed().as_secs_f64();
    let trainer = trainer(job);
    if job.train_repeats == 0 {
        return Err("train_repeats must be positive".into());
    }
    let mut calls = Vec::with_capacity(job.train_repeats);
    let mut exports = Vec::with_capacity(job.train_repeats);
    let mut output_counts = (0, 0, 0);
    let mut process_hwm_kib_before_validation = 0;
    for repeat in 0..job.train_repeats {
        let begin = Instant::now();
        let cpu = cpu_seconds()?;
        let (vocab, merges, special) = trainer.do_train(&words)?;
        calls.push((begin.elapsed().as_secs_f64(), cpu_seconds()? - cpu));
        process_hwm_kib_before_validation = hwm()?;
        let export = Instant::now();
        let output = if repeat + 1 == job.train_repeats {
            job.output.clone()
        } else {
            format!("{}.repeat-{repeat}", job.output)
        };
        model_output::write(&output, &vocab, &merges)?;
        exports.push(export.elapsed().as_secs_f64());
        output_counts = (vocab.len(), merges.len(), special.len());
    }
    let train_seconds = calls.iter().map(|call| call.0).sum::<f64>() / calls.len() as f64;
    let train_cpu_seconds = calls.iter().map(|call| call.1).sum::<f64>() / calls.len() as f64;
    let export_seconds = exports.iter().sum::<f64>() / exports.len() as f64;
    Ok(
        json!({"metrics":{"load_seconds":load_seconds,"train_seconds":train_seconds,
        "train_cpu_seconds":train_cpu_seconds,"process_hwm_kib_before_validation":process_hwm_kib_before_validation,
        "export_seconds":export_seconds,"training_calls":calls},
        "output":{"model_path":job.output,"actual_vocab":output_counts.0,"actual_merges":output_counts.1,"special_tokens":output_counts.2},
        "timing_boundary":"public-do-train-before-serialization"}),
    )
}
