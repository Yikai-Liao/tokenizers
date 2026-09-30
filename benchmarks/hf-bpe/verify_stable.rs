//! Compile with verify_stable.py against unmodified HF 0.23.2 and this prototype.
use ahash::AHashMap;
use compact_str::CompactString;
use tk_train::BpeTrainer;
use tokenizers::models::bpe::{BPE, BpeTrainer as HfTrainer};
use tokenizers::{AddedToken as HfAdded, Model};

fn rng(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state >> 32
}

fn main() {
    let mut state = 2348;
    let mut reused = 0;
    for case in 0..500 {
        let mut words = AHashMap::<CompactString, u64>::new();
        for _ in 0..1 + rng(&mut state) % 12 {
            let word: String = (0..rng(&mut state) % 20)
                .map(|_| ['a', 'b', 'c', '测'][rng(&mut state) as usize % 4])
                .collect();
            *words.entry(word.into()).or_default() += 1 + rng(&mut state) % 8;
        }
        let mut prototype = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size((4 + rng(&mut state) % 60) as usize)
            .min_frequency(rng(&mut state) % 5)
            .build();
        if case % 3 == 0 {
            prototype.special_tokens = ["ab", "aa", "aaa", "abc", "测测"]
                .map(|s| tk_encode::vocab::bucket_added_vocabulary::AddedToken::from(s, true))
                .to_vec();
        }
        if case % 4 == 0 {
            prototype.continuing_subword_prefix = Some(["##", "a", "测"][(case / 4) % 3].into());
        }
        if case % 5 == 0 {
            prototype.end_of_word_suffix = Some(["</w>", "a"][(case / 5) % 2].into());
        }
        if case % 2 == 0 {
            prototype.max_token_length = Some((rng(&mut state) % 12) as usize);
        }
        if case % 7 == 0 {
            prototype.limit_alphabet = Some(3);
        }
        prototype.initial_alphabet.extend(['a', 'b', 'c']);
        let reference = HfTrainer::builder()
            .show_progress(false)
            .vocab_size(prototype.vocab_size)
            .min_frequency(prototype.min_frequency)
            .build();
        // Both implementations iterate the SAME HashMap, so affix ID initialization
        // uses the same source order. Alphabet pruning uses three forced characters.
        let mut reference = reference;
        reference.special_tokens = prototype
            .special_tokens
            .iter()
            .map(|s| HfAdded::from(s.content.clone(), true))
            .collect();
        reference.continuing_subword_prefix = prototype.continuing_subword_prefix.clone();
        reference.end_of_word_suffix = prototype.end_of_word_suffix.clone();
        reference.max_token_length = prototype.max_token_length;
        reference.limit_alphabet = prototype.limit_alphabet;
        reference.initial_alphabet = prototype.initial_alphabet.clone();
        let mut model = BPE::default();
        reference.do_train(&words, &mut model).unwrap();
        let expected = serde_json::to_value(&model).unwrap();
        let got = prototype.do_train_indexed(&words).unwrap();
        reused += got.stats.reused_ids;
        // HF's model builder retains the LAST rank of a repeated pair.
        // Normalize raw trainer events in rank order before comparing model JSON.
        let last: AHashMap<_, _> = got
            .merges
            .iter()
            .enumerate()
            .map(|(rank, pair)| (pair.clone(), rank))
            .collect();
        let normalized_merges: Vec<_> = got
            .merges
            .iter()
            .enumerate()
            .filter(|(rank, pair)| last[*pair] == *rank)
            .map(|(_, pair)| pair.clone())
            .collect();
        assert_eq!(
            serde_json::to_value(&got.vocab).unwrap(),
            expected["vocab"],
            "vocab case {case}"
        );
        assert_eq!(
            serde_json::to_value(&normalized_merges).unwrap(),
            expected["merges"],
            "merges case {case}"
        );
        // Save a complete HF BPE model JSON and load it through the actual stable
        // model reader; compare encodings before and after this disk round trip.
        let mut document = expected;
        document["vocab"] = serde_json::to_value(&got.vocab).unwrap();
        document["merges"] = serde_json::to_value(&normalized_merges).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        serde_json::to_writer(tmp.as_file(), &document).unwrap();
        let saved = std::fs::read_to_string(tmp.path()).unwrap();
        let reloaded: BPE = serde_json::from_str(&saved).unwrap();
        for word in words.keys() {
            let expected_ids = model
                .tokenize(word)
                .map(|tokens| tokens.iter().map(|t| t.id).collect::<Vec<_>>())
                .map_err(|e| e.to_string());
            let reload_ids = reloaded
                .tokenize(word)
                .map(|tokens| tokens.iter().map(|t| t.id).collect::<Vec<_>>())
                .map_err(|e| e.to_string());
            assert_eq!(expected_ids, reload_ids, "reload case {case}, {word:?}");
        }
    }
    println!("500 stable/prototype model comparisons and disk reloads passed; reused_ids={reused}");
}
