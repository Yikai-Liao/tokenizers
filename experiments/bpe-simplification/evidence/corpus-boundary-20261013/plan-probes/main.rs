//! Standard-library contract probe; this is not the BPE training implementation.
use std::collections::HashMap;
use std::mem::{align_of, size_of};
use std::ops::Range;

#[derive(Clone, Copy)]
struct OldMatch {
    start: usize,
    after: usize,
}

type CountMap = HashMap<String, u64>;
struct WordCounts(Vec<(String, u64)>);

#[derive(Clone, Copy)]
struct WordCountsView<'a> {
    map: Option<&'a CountMap>,
    entries: &'a [(String, u64)],
}

impl From<CountMap> for WordCounts {
    fn from(map: CountMap) -> Self {
        Self(map.into_iter().collect())
    }
}

impl<'a> From<&'a CountMap> for WordCountsView<'a> {
    fn from(map: &'a CountMap) -> Self {
        Self { map: Some(map), entries: &[] }
    }
}

impl<'a> From<&'a WordCounts> for WordCountsView<'a> {
    fn from(counts: &'a WordCounts) -> Self {
        Self { map: None, entries: &counts.0 }
    }
}

impl<'a> WordCountsView<'a> {
    fn iter(self) -> impl Iterator<Item = (&'a String, &'a u64)> {
        self.map.into_iter().flat_map(|map| map.iter())
            .chain(self.entries.iter().map(|(word, count)| (word, count)))
    }
}

fn accepted_loop(coordinates: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut calls = Vec::new();
    let mut end = 0;
    let mut accepted = Vec::new();
    for &coordinate in coordinates {
        if coordinate >= end {
            calls.push(coordinate);
            if [1, 4, 6].contains(&coordinate) {
                accepted.push(coordinate);
                end = coordinate + 2;
            }
        }
    }
    (accepted, calls)
}

fn accepted_filter_map(coordinates: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut calls = Vec::new();
    let mut end = 0;
    let accepted = coordinates.iter().copied().filter_map(|coordinate| {
        if coordinate < end {
            return None;
        }
        calls.push(coordinate);
        if ![1, 4, 6].contains(&coordinate) {
            return None;
        }
        end = coordinate + 2;
        Some(coordinate)
    }).collect();
    (accepted, calls)
}

fn main() {
    println!("OldMatch size={} align={}", size_of::<OldMatch>(), align_of::<OldMatch>());
    println!("Range<usize> size={} align={}", size_of::<Range<usize>>(), align_of::<Range<usize>>());
    println!("Option<OldMatch> size={}", size_of::<Option<OldMatch>>());
    println!("Option<Range<usize>> size={}", size_of::<Option<Range<usize>>>());

    // Borrow the non-Copy range through neighbor recording, then read its start.
    let interval = 7..11;
    let mut observed = Vec::new();
    let mut record = |matched: &Range<usize>| observed.push((matched.start, matched.end, matched.len()));
    record(&interval);
    let mut starts = Vec::new();
    starts.push(interval.start);
    assert_eq!(observed, [(7, 11, 4)]);
    assert_eq!(starts, [7]);
    let old = OldMatch { start: 7, after: 11 };
    assert_eq!(old.after - old.start, interval.len());

    let map: CountMap = [("a long owned word".to_owned(), u64::MAX), ("zero".to_owned(), 0)].into();
    let pointers: HashMap<_, _> = map.iter().map(|(word, _)| (word.clone(), word.as_ptr())).collect();
    let borrowed = WordCountsView::from(&map);
    assert_eq!(borrowed.iter().count(), 2);
    let owned = WordCounts::from(map);
    assert!(owned.0.iter().all(|(word, _)| pointers[word] == word.as_ptr()));
    let frozen_view = WordCountsView::from(&owned);
    assert_eq!(frozen_view.iter().map(|(_, count)| *count).max(), Some(u64::MAX));

    let coordinates = [1, 1, 2, 3, 4, 5, 6];
    let old_acceptance = accepted_loop(&coordinates);
    let iterator_acceptance = accepted_filter_map(&coordinates);
    assert_eq!(old_acceptance, iterator_acceptance);
    assert_eq!(old_acceptance, (vec![1, 4, 6], vec![1, 3, 4, 6]));
    println!("borrowed ranges, From signatures/owned moves, filter_map skip and matcher order: passed");
}
