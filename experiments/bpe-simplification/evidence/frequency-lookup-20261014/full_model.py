"""Add post-timing full native-model validation to the shared experiment runner."""
from pathlib import Path
import sys
from prepare import replace


def apply(root):
    runner = root / 'experiments/bpe-simplification/runner/src'
    replace(runner/'core.rs', 'model_output::write(&job.output, &vocab, &merges)?;',
            'model_output::write_full(&job.output, &vocab, &merges, &trainer, &special)?;')
    with (runner/'model_output.rs').open('a') as out:
        out.write('''
// Validation and serialization happen after the timed public do_train call and HWM.
// Keep the training merge order as well as all fields of the loaded native config.
pub fn write_full(
    path: &str, vocab: &Vocab, merges: &[(String, String)], trainer: &tk_train::BpeTrainer,
    special: &[tk_encode::vocab::bucket_added_vocabulary::AddedToken],
) -> Result<(), Error> {
    use tk_encode::models::bpe::{BpeConfig, PipelineBPE};
    let model = PipelineBPE::from_config(BpeConfig {
        vocab: vocab.clone(), merges: merges.to_vec(),
        continuing_subword_prefix: trainer.continuing_subword_prefix.clone(),
        end_of_word_suffix: trainer.end_of_word_suffix.clone(),
        ..Default::default()
    })?;
    let config = model.to_config()?;
    let value = serde_json::json!({
        "training_parts": {"vocab":vocab,"merges":merges},
        "model_config": {"vocab":config.vocab,"merges":config.merges,
            "byte_level":config.byte_level,"unk_token":config.unk_token,
            "continuing_subword_prefix":config.continuing_subword_prefix,
            "end_of_word_suffix":config.end_of_word_suffix,"fuse_unk":config.fuse_unk,
            "byte_fallback":config.byte_fallback,"ignore_merges":config.ignore_merges,
            "dropout":config.dropout,"cache_capacity":config.cache_capacity},
        "special_tokens":special.iter().map(|t| serde_json::json!({
            "content":t.content,"single_word":t.single_word,"lstrip":t.lstrip,
            "rstrip":t.rstrip,"normalized":t.normalized,"special":t.special})).collect::<Vec<_>>()
    });
    // Value canonicalizes object keys only; arrays, IDs and ordered merges are exact.
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer(&mut writer, &value)?;
    writer.flush()?;
    Ok(())
}
''')


if __name__ == '__main__': apply(Path(sys.argv[1]).resolve())
