//! Native complete model export, after all timing and pre-validation snapshots.
use crate::protocol::{Error, Job};
use tk_encode::{
    models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab},
    pipeline::{PipelineModel, PipelinePreTokenizer, PipelineTokenizer},
};
pub fn write(
    path: &std::path::Path,
    job: &Job,
    vocab: &Vocab,
    merges: &Merges,
) -> Result<(), Error> {
    let model = PipelineBPE::from_config(BpeConfig {
        vocab: vocab.clone(),
        merges: merges.clone(),
        byte_level: job.pretokenizer == "bytelevel_regex",
        continuing_subword_prefix: job.trainer.prefix.clone(),
        end_of_word_suffix: job.trainer.suffix.clone(),
        ..Default::default()
    })?;
    let tokenizer = PipelineTokenizer::from_parts(
        Default::default(),
        vec![],
        PipelinePreTokenizer::None,
        PipelineModel::BPE(model),
        Default::default(),
        None,
        Default::default(),
        None,
        None,
    );
    std::fs::write(path, tk_serialize::to_json(&tokenizer)?)?;
    Ok(())
}
