use crate::protocol::{Error, Job};
use tk_encode::{
    pipeline::{Normalize, PreTokenize, PreTokenizerScratch, Span},
    pre_tokenizers::split::{Split, SplitPattern},
    tokenizer::SplitDelimiterBehavior,
    utils::byte_level::GPT2_REGEX_STR,
};
use tk_train::BpeTrainer;

fn byte_alphabet() -> std::collections::HashSet<char> {
    let mut out = std::collections::HashSet::with_capacity(256);
    let mut counter = 0u32;
    for byte in 0u32..=255 {
        let printable = (0x21..=0x7e).contains(&byte) || (0xa1..=0xac).contains(&byte) || (0xae..=0xff).contains(&byte);
        let codepoint = if printable { byte } else { let value = 0x100 + counter; counter += 1; value };
        out.insert(char::from_u32(codepoint).expect("byte alphabet"));
    }
    out
}

fn normalize_bytes(piece: &str) -> Result<String, Error> {
    tk_encode::normalizers::byte_level::ByteLevel::default()
        .normalize(piece, 0)
        .map(|text| text.into_owned())
}

pub fn trainer(job: &Job) -> BpeTrainer {
    let cfg = &job.trainer;
    let mut builder = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(cfg.vocab_size)
        .min_frequency(cfg.min_frequency)
        .max_token_length(cfg.max_token_length);
    if let Some(value) = &cfg.prefix { builder = builder.continuing_subword_prefix(value.clone()); }
    if let Some(value) = &cfg.suffix { builder = builder.end_of_word_suffix(value.clone()); }
    if job.pretokenizer == "bytelevel_regex" {
        builder = builder.initial_alphabet(
            byte_alphabet(),
        );
    }
    builder.build()
}

pub struct Processor { regex: Split }
impl Processor {
    pub fn new(kind: &str) -> Result<Self, Error> {
        if kind != "bytelevel_regex" { return Err("unsupported pretokenizer".into()); }
        Ok(Self { regex: Split::native(
            SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )? })
    }

    pub fn process(&self, line: &str) -> Result<Vec<String>, Error> {
        thread_local! {
            static BUFFERS: std::cell::RefCell<(PreTokenizerScratch, Vec<Span>)> =
                std::cell::RefCell::new((PreTokenizerScratch::default(), Vec::new()));
        }
        BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            let (scratch, spans) = &mut *buffers;
            spans.clear();
            self.regex.pre_tokenize(line, scratch, spans)?;
            spans.iter().map(|span| normalize_bytes(&line[span.range()])).collect()
        })
    }
}
