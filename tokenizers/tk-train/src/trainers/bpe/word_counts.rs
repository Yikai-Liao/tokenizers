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
/// Repeated train(&self) calls leave owned counts intact; only thin references
/// are collected and sorted by CorpusPlan.
#[derive(Clone, Copy)]
pub(super) struct WordCountsView<'a> {
    map: Option<&'a CountMap>,
    entries: &'a [Entry],
}

impl<'a> From<&'a CountMap> for WordCountsView<'a> {
    fn from(map: &'a CountMap) -> Self {
        Self {
            map: Some(map),
            entries: &[],
        }
    }
}

impl<'a> From<&'a WordCounts> for WordCountsView<'a> {
    fn from(counts: &'a WordCounts) -> Self {
        Self {
            map: None,
            entries: &counts.0,
        }
    }
}

impl<'a> WordCountsView<'a> {
    pub(super) fn len(self) -> usize {
        // Exactly one representation is active; the absent map or empty entry
        // slice contributes zero to the borrowed view's length.
        self.map.map_or(0, |map| map.len()) + self.entries.len()
    }

    pub(super) fn iter(self) -> impl Iterator<Item = (&'a CompactString, &'a u64)> {
        self.map
            .into_iter()
            .flat_map(|map| map.iter())
            .chain(self.entries.iter().map(|(word, count)| (word, count)))
    }

    pub(super) fn keys(self) -> impl Iterator<Item = &'a CompactString> {
        self.iter().map(|(word, _)| word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_and_serde() {
        let original = WordCounts::from_unique(vec![("zero".into(), 0), ("wide".into(), u64::MAX)]);
        let counts: CountMap = [("wide".into(), u64::MAX), ("zero".into(), 0)].into();
        let map = WordCounts::from(counts);
        assert_eq!(original, map);
        for json in [
            r#"{"zero":1,"wide":18446744073709551615}"#,
            r#"{"other":0,"wide":18446744073709551615}"#,
            r#"{"zero":0}"#,
        ] {
            assert_ne!(original, serde_json::from_str::<WordCounts>(json).unwrap());
        }
        let json = serde_json::to_value(&original).unwrap();
        assert_eq!(json, serde_json::json!({"zero": 0, "wide": u64::MAX}));
        assert_eq!(
            serde_json::from_value::<WordCounts>(json).unwrap(),
            original
        );
        let duplicate: WordCounts = serde_json::from_str(r#"{"word":1,"word":9}"#).unwrap();
        assert_eq!(duplicate, WordCounts::from_unique(vec![("word".into(), 9)]));
        assert!(serde_json::from_str::<WordCounts>(r#"{"word":-1}"#).is_err());
    }
}
