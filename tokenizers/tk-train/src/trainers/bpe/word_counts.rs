//! Frozen owned counts, with an unchanged flat-map public representation.
use ahash::AHashMap;
use compact_str::CompactString;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

type CountMap = AHashMap<CompactString, u64>;
type Entry = (CompactString, u64);

/// Frozen trainer-owned word counts, serialized as the same public flat map.
/// Feed moves its unique entries here, preserving their iteration order.
#[derive(Clone, Default)]
pub(super) struct WordCounts(Vec<Entry>);

impl WordCounts {
    // Requires globally unique keys. Entry order is unspecified.
    pub(super) fn from_unique(entries: Vec<Entry>) -> Self {
        Self(entries)
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}

impl From<CountMap> for WordCounts {
    fn from(map: CountMap) -> Self {
        Self(map.into_iter().collect())
    }
}

impl fmt::Debug for WordCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(WordCountsView::from(self).iter())
            .finish()
    }
}

impl PartialEq for WordCounts {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if self.0 == other.0 {
            return true;
        }

        // Public content equality remains linear, with a temporary borrowed index.
        let lookup: AHashMap<&CompactString, u64> = WordCountsView::from(other)
            .iter()
            .map(|(word, count)| (word, *count))
            .collect();
        WordCountsView::from(self)
            .iter()
            .all(|(word, count)| lookup.get(word) == Some(count))
    }
}

impl Eq for WordCounts {}

impl Serialize for WordCounts {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(word, count)| (word, count)))
    }
}

impl<'de> Deserialize<'de> for WordCounts {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CountMap::deserialize(deserializer).map(Self::from)
    }
}

/// Borrows either count representation, including do_train's caller-owned map.
/// Repeated train(&self) calls leave owned counts intact.
#[derive(Clone, Copy)]
pub(super) struct WordCountsView<'counts> {
    map: Option<&'counts CountMap>,
    entries: &'counts [Entry],
}

impl<'counts> From<&'counts CountMap> for WordCountsView<'counts> {
    fn from(map: &'counts CountMap) -> Self {
        Self {
            map: Some(map),
            entries: &[],
        }
    }
}

impl<'counts> From<&'counts WordCounts> for WordCountsView<'counts> {
    fn from(counts: &'counts WordCounts) -> Self {
        Self {
            map: None,
            entries: &counts.0,
        }
    }
}

impl<'counts> WordCountsView<'counts> {
    pub(super) fn len(self) -> usize {
        // Exactly one representation is active; the absent map or empty entry
        // slice contributes zero to the borrowed view's length.
        self.map.map_or(0, |map| map.len()) + self.entries.len()
    }

    pub(super) fn iter(self) -> impl Iterator<Item = (&'counts CompactString, &'counts u64)> {
        self.map
            .into_iter()
            .flat_map(|map| map.iter())
            .chain(self.entries.iter().map(|(word, count)| (word, count)))
    }
}

impl<'counts> WordCountsView<'counts> {
    pub(super) fn keys(self) -> impl Iterator<Item = &'counts CompactString> {
        self.iter().map(|(word, _)| word)
    }
}
