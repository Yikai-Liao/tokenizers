// use the BPE trainer
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence as NSequence, Strip, NFC};
use tokenizers::{PostProcessorWrapper, TokenizerBuilder};

use tokenizers::pre_tokenizers::punctuation::Punctuation;
use tokenizers::pre_tokenizers::sequence::Sequence as PSequence;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

use tokenizers::decoders::bpe::BPEDecoder;
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
        .vocab_size(20000)
        .continuing_subword_prefix("##".to_string())
        .end_of_word_suffix("</w>".to_string())
        .show_progress(false)
        .build();

    // Build a New Tokenizer with Pretokenizer and Normalizer
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(NSequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(PSequence::new(vec![
            Whitespace::default().into(),
            Punctuation::default().into(),
        ])))
        .with_post_processor(Some(PostProcessorWrapper::Bert(
            tokenizers::processors::bert::BertProcessing::default(),
        )))
        .with_decoder(Some(BPEDecoder::default()))
        .build()?;

    // train file path is a dir with multiple text files
    // list all files in the dir
    // let files = std::fs::read_dir(train_file_path)?;
    // let files: Vec<String> = files
    //     .map(|f| f.unwrap().path().to_str().unwrap().to_string())
    //     .collect();
    let files = vec![train_file_path.to_string()];
    // show num of threads rayon will use
    // manually set num of threads
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();
    println!("num of threads: {:?}", rayon::current_num_threads());
    println!("files: {:?}", files);

    let start = std::time::Instant::now();
    let train_num = 1;

    for _ in 0..train_num {
        // timestamp
        tokenizer.train_from_files(&mut trainer, files.clone())?;
    }
    let dur = start.elapsed() / train_num;
    println!("Training took {:?}", dur);
    Ok(())
}
