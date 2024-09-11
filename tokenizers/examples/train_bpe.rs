use rayon::prelude::*;
use std::collections::HashMap;
// use the BPE trainer
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
// use tokenizers::normalizers::{Sequence as NSequence, Strip, NFC};
// use tokenizers::{PostProcessorWrapper, TokenizerBuilder};
//
// use tokenizers::pre_tokenizers::punctuation::Punctuation;
// use tokenizers::pre_tokenizers::sequence::Sequence as PSequence;
// use tokenizers::pre_tokenizers::whitespace::Whitespace;

use rustc_hash::FxHashMap;
use tokenizers::parallelism::MaybeParallelIterator;
// use tokenizers::decoders::bpe::BPEDecoder;

fn build_words(files: Vec<String>) -> HashMap<String, u64> {
    let start = std::time::Instant::now();
    let seps = [
        ' ', 'n', '\t', '\r', '\x0c', '\x0b', ';', ':', ',', '.', '!', '?', '(', ')', '[', ']',
        '{', '}', '<', '>', '\'', '"', '`', '~', '@', '#', '$', '%', '^', '&', '*', '-', '_', '+',
        '=', '\\', '|', '/', ' ', '\n', '\t', '，', '。', '！', '？', '、', '；', '：', '（', '）',
        '【', '】', '《', '》', '‘', '’', '“', '”', '…', '—', '～', '·', '「', '」', '『', '』',
        '〈', '〉',
    ];
    // build an array of bool that returns true if the char is a separator
    let is_sep = seps.iter().map(|&c| (c, true)).collect::<FxHashMap<_, _>>();

    // init a HashMap for each thread
    let words = files
        .into_maybe_par_iter()
        .fold(
            || FxHashMap::default(),
            |mut words, file| {
                let text = match std::fs::read_to_string(&file) {
                    Ok(text) => text,
                    Err(_) => {
                        print!("Error reading file: {}", file);
                        return words;
                    }
                };

                let mut pivot: usize = 0;
                text.char_indices()
                    .filter(|(_, c)| is_sep.contains_key(c))
                    .for_each(|(i, c)| {
                        if i > pivot {
                            let word = &text[pivot..i];
                            words.get_mut(word).map(|v| *v += 1).unwrap_or_else(|| {
                                words.insert(word.to_string(), 1);
                            });
                        }
                        pivot = i + c.len_utf8();
                    });
                words
            },
        )
        .reduce(
            || FxHashMap::default(),
            |words1, words2| {
                let (mut words1, words2) = if words1.len() >= words2.len() {
                    (words1, words2)
                } else {
                    (words2, words1)
                };
                for (word, count) in words2 {
                    words1
                        .get_mut(&word)
                        .map(|v| *v += count)
                        .unwrap_or_else(|| {
                            words1.insert(word, count);
                        });
                }
                words1
            },
        );

    let dur = start.elapsed();
    println!("build_words(hash) took {:?}", dur);
    words.into_iter().collect()
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <train_file_dir>", args[0]);
        std::process::exit(1);
    }
    let train_file_path = &args[1];
    println!("train_file_path: {}", train_file_path);

    let trainer = BpeTrainerBuilder::new()
        .max_token_length(Some(32))
        .min_frequency(10)
        .vocab_size(20000)
        .continuing_subword_prefix("##".to_string())
        .end_of_word_suffix("</w>".to_string())
        .show_progress(false)
        .build();

    // Build a New Tokenizer with Pretokenizer and Normalizer
    // let mut tokenizer = TokenizerBuilder::new()
    //     .with_model(BPE::default())
    //     .with_normalizer(Some(NSequence::new(vec![
    //         Strip::new(true, true).into(),
    //         NFC.into(),
    //     ])))
    //     .with_pre_tokenizer(Some(PSequence::new(vec![
    //         Whitespace::default().into(),
    //         Punctuation::default().into(),
    //     ])))
    //     .with_post_processor(Some(PostProcessorWrapper::Bert(
    //         tokenizers::processors::bert::BertProcessing::default(),
    //     )))
    //     .with_decoder(Some(BPEDecoder::default()))
    //     .build()?;

    // train file path is a dir with multiple text files
    // list all files in the dir
    let files = std::fs::read_dir(train_file_path)?;
    let files: Vec<String> = files
        .map(|f| f.unwrap().path().to_str().unwrap().to_string())
        .collect();
    // let files = vec![train_file_path.to_string()];
    // show num of threads rayon will use
    // manually set num of threads
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();
    println!("num of threads: {:?}", rayon::current_num_threads());
    println!("files: {:?}", files);

    let words = build_words(files.clone());

    let train_num = 10;
    let start = std::time::Instant::now();
    for _ in 0..train_num {
        // timestamp
        let mut model = BPE::default();
        // tokenizer.train_from_files(&mut trainer, files.clone())?;
        let _ = trainer.do_train(&words, &mut model)?;
    }
    let dur = start.elapsed() / train_num;
    println!("Training took {:?}", dur);
    Ok(())
}
