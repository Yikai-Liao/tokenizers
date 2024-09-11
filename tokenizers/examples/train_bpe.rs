// use the BPE trainer
use tokenizers::models::bpe::{BPE, BpeTrainer, BpeTrainerBuilder};
use tokenizers::{normalizers, pre_tokenizers::split::{Split, SplitPattern}, AddedToken, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer, TokenizerBuilder};
use tokenizers::normalizers::{Sequence, Strip, NFC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <train_file_path>", args[0]);
        std::process::exit(1);
    }
    let train_file_path = &args[1];
    println!("train_file_path: {}", train_file_path);

    let mut trainer = BpeTrainerBuilder::new()
        .max_token_length(Some(32))
        .min_frequency(10)
        .vocab_size(5000)
        .continuing_subword_prefix("##".to_string())
        .end_of_word_suffix("</w>".to_string())
        .show_progress(false)
        .build();

    // Build a New Tokenizer with Pretokenizer and Normalizer
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Split(
            Split::new(
                SplitPattern::Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into()),
                SplitDelimiterBehavior::Isolated,
                false,
            ).unwrap(),
        )))
        .with_post_processor(Some(
            PostProcessorWrapper::Bert(
                tokenizers::processors::bert::BertProcessing::default()
            )
        ))
        .with_decoder(Some(crate::ByteLevel::default()))
        .build()?;

    let start = std::time::Instant::now();
    let train_num = 20;
    for _ in 0..train_num {
        // timestamp
        tokenizer.train_from_files(
            &mut trainer,
            vec![train_file_path.to_string()],
        )?;
    }
    let dur = start.elapsed() / train_num;
    println!("Training took {:?}", dur);
    Ok(())
}