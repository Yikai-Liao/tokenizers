use super::*;
use crate::{BpeTrainer, Trainer};

fn sequence_map_reduce(input: &[String], process: impl Fn(&str) -> Vec<String>) -> WordCounts {
    input
        .iter()
        .map(|sequence| {
            let mut counts = WordCounts::new();
            for word in process(sequence) {
                *counts.entry(CompactString::from(word)).or_default() += 1;
            }
            counts
        })
        .fold(WordCounts::new(), |mut counts, sequence| {
            for (word, count) in sequence {
                *counts.entry(word).or_default() += count;
            }
            counts
        })
}

#[test]
fn feed_matches_sequence_map_reduce() {
    let input: Vec<_> = (0..4097)
        .map(|index| match index % 5 {
            0 => format!("词{index}🙂 shared shared"),
            1 => "shared abc ab abc".to_owned(),
            2 => "very_long_word_that_needs_heap_storage shared".to_owned(),
            3 => String::new(),
            _ => "abc abc ab".to_owned(),
        })
        .collect();
    let split = |sequence: &str| sequence.split_whitespace().map(str::to_owned).collect();
    let expected = sequence_map_reduce(&input, split);
    let mut trainer = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(100)
        .min_frequency(2)
        .build();
    trainer.feed(input.iter(), |text| Ok(split(text))).unwrap();
    assert_eq!(trainer.words, expected);
    assert_eq!(
        trainer.train_vocab().unwrap(),
        trainer.do_train(&expected).unwrap()
    );
}

#[test]
fn counts_survive_flush_boundaries_within_one_sequence() {
    for distinct in [
        LOCAL_KEY_LIMIT - 1,
        LOCAL_KEY_LIMIT,
        LOCAL_KEY_LIMIT + 1,
        2 * LOCAL_KEY_LIMIT + 1,
    ] {
        let words: Vec<_> = std::iter::once(String::new())
            .chain((1..distinct).map(|i| format!("词{i}🙂")))
            .collect();
        let shared = ShardedCounts::default();
        let mut local = LocalCounts::default();
        let mut input = words.clone();
        input.push(words[0].clone());
        local.add(input, &shared);
        let count_before_tail = if distinct < LOCAL_KEY_LIMIT {
            distinct
        } else {
            distinct % LOCAL_KEY_LIMIT + 1
        };
        assert_eq!(local.words.len(), count_before_tail);
        shared.flush(&mut local);
        let counts = shared.into_counts();
        assert_eq!(counts.len(), distinct);
        assert_eq!(counts[words[0].as_str()], 2);
        assert!(words[1..].iter().all(|word| counts[word.as_str()] == 1));
    }
}

#[test]
fn local_counts_flush_and_merge_across_workers() {
    let shared = ShardedCounts::default();
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let mut local = LocalCounts::default();
                local.add(vec!["shared".into(), "shared".into()], &shared);
                shared.flush(&mut local);
            });
        }
    });
    assert_eq!(shared.into_counts(), [("shared".into(), 4)].into());
}

#[test]
fn discard_releases_shards_and_rejects_later_flushes() {
    let shared = ShardedCounts::default();
    let mut local = LocalCounts::default();
    local.add(
        (0..LOCAL_KEY_LIMIT)
            .map(|index| index.to_string())
            .collect(),
        &shared,
    );
    assert_eq!(
        shared
            .shards
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .iter()
            .map(|shard| shard.lock().unwrap().len())
            .sum::<usize>(),
        LOCAL_KEY_LIMIT,
    );
    shared.discard();
    assert!(shared.shards.read().unwrap().is_none());

    local.add(vec!["after".into()], &shared);
    shared.flush(&mut local);
    assert!(shared.shards.read().unwrap().is_none());
    assert!(local.words.is_empty());
}

#[test]
fn feed_error_preserves_counts_and_visits_every_sequence() {
    let mut trainer = BpeTrainer::default();
    trainer
        .feed(["saved"].into_iter(), |text| Ok(vec![text.into()]))
        .unwrap();
    let previous = trainer.clone();
    let called = Mutex::new(Vec::new());
    let error = trainer
        .feed(["flush", "fail", "tail"].into_iter(), |text| {
            called.lock().unwrap().push(text.to_owned());
            match text {
                "flush" => Ok((0..LOCAL_KEY_LIMIT)
                    .map(|index| index.to_string())
                    .collect()),
                "fail" => Err(std::io::Error::other("callback failed").into()),
                _ => Ok(vec![text.into()]),
            }
        })
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().to_string(),
        "callback failed"
    );
    let mut called = called.into_inner().unwrap();
    called.sort();
    assert_eq!(called, ["fail", "flush", "tail"]);
    assert_eq!(trainer, previous);

    trainer
        .feed(["replacement"].into_iter(), |text| Ok(vec![text.into()]))
        .unwrap();
    assert_eq!(trainer.words, [("replacement".into(), 1)].into());
    trainer
        .feed(std::iter::empty::<&str>(), |_| unreachable!())
        .unwrap();
    assert!(trainer.words.is_empty());
}
