//! Unlimited alphabets need presence, rather than frequencies used for pruning.
use super::*;

const CODEPOINTS: usize = 0x110000;
const BIT_WORDS: usize = CODEPOINTS / 64;

pub(super) fn initialize(
    trainer: &BpeTrainer,
    wc: &AHashMap<CompactString, u64>,
    ids: &mut AHashMap<CompactString, u32>,
    strings: &mut Vec<CompactString>,
    workers: usize,
) -> usize {
    if trainer.limit_alphabet.is_some() {
        // Frequency ties at the pruning boundary inherit HF's hash traversal.
        // Retain that selection unchanged, including forced-alphabet behavior.
        trainer.compute_alphabet(wc, ids, strings);
        return 0;
    }
    let words: Vec<_> = wc.keys().collect();
    let chunk = words.len().div_ceil(workers).max(1);
    let bitmaps: Vec<Vec<u64>> = words
        .par_chunks(chunk)
        .map(|chunk| {
            let mut present = vec![0_u64; BIT_WORDS];
            for word in chunk {
                for c in word.chars() {
                    let c = c as usize;
                    present[c / 64] |= 1_u64 << (c % 64);
                }
            }
            present
        })
        .collect();
    let scratch_bytes = bitmaps.iter().map(|b| b.capacity() * 8).sum::<usize>()
        + BIT_WORDS * 8
        + words.capacity() * std::mem::size_of::<&CompactString>()
        + bitmaps.capacity() * std::mem::size_of::<Vec<u64>>();
    let mut present = vec![0_u64; BIT_WORDS];
    for bitmap in bitmaps {
        for (merged, bits) in present.iter_mut().zip(bitmap) {
            *merged |= bits;
        }
    }
    for c in &trainer.initial_alphabet {
        let c = *c as usize;
        present[c / 64] |= 1_u64 << (c % 64);
    }
    // Iterating set bits in numeric order preserves canonical character IDs.
    // Zero-weight words still contribute characters, exactly as in HF.
    for (word, mut bits) in present.into_iter().enumerate() {
        while bits != 0 {
            let codepoint = word * 64 + bits.trailing_zeros() as usize;
            let character =
                char::from_u32(codepoint as u32).expect("only valid chars set alphabet bits");
            let mut utf8 = [0; 4];
            let text = character.encode_utf8(&mut utf8);
            if !ids.contains_key(text as &str) {
                let id = strings.len() as u32;
                let token = CompactString::from(text as &str);
                strings.push(token.clone());
                ids.insert(token, id);
            }
            bits &= bits - 1;
        }
    }
    scratch_bytes
}

pub(super) fn character_ids(ids: &AHashMap<CompactString, u32>) -> Vec<u32> {
    let mut table = vec![NONE; CODEPOINTS];
    for (text, &id) in ids {
        let mut chars = text.chars();
        if let Some(c) = chars.next()
            && chars.next().is_none()
        {
            table[c as usize] = id;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_presence_preserves_forced_alphabet_special_ids_and_zero_weight_words() {
        let words: AHashMap<CompactString, u64> = [("ba中🙂", 5), ("😀", 0), ("abc", 9)]
            .into_iter()
            .map(|(s, w)| (s.into(), w))
            .collect();
        for limit in [None, Some(2), Some(999)] {
            let mut builder = BpeTrainer::builder()
                .show_progress(false)
                .initial_alphabet(['z', '\u{10ffff}'].into_iter().collect())
                .special_tokens(vec![
                    AddedToken::from("中", true),
                    AddedToken::from("ab", true),
                ]);
            if let Some(limit) = limit {
                builder = builder.limit_alphabet(limit);
            }
            let trainer = builder.build();
            for workers in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let mut original_ids = AHashMap::new();
                let mut original_strings = Vec::new();
                trainer.add_special_tokens(&mut original_ids, &mut original_strings);
                let mut ids = original_ids.clone();
                let mut strings = original_strings.clone();
                // The limited path deliberately calls the original selector;
                // do not compare independently seeded hash maps at tied limits.
                if limit.is_none() {
                    trainer.compute_alphabet(&words, &mut original_ids, &mut original_strings);
                }
                pool.install(|| initialize(&trainer, &words, &mut ids, &mut strings, workers));
                if limit.is_none() {
                    assert_eq!(ids, original_ids);
                    assert_eq!(strings, original_strings);
                    assert!(ids.contains_key("😀"));
                    assert!(ids.contains_key("\u{10ffff}"));
                }
                assert_eq!(ids["中"], 0);
                assert_eq!(ids["ab"], 1);
                let table = character_ids(&ids);
                assert_eq!(table['中' as usize], 0);
                for (token, &id) in &ids {
                    if token.chars().count() == 1 {
                        assert_eq!(table[token.chars().next().unwrap() as usize], id);
                    }
                }
            }
        }
    }
}
