//! Frozen owned counts, with an unchanged flat-map public representation.
use ahash::AHashMap;
use compact_str::CompactString;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use std::fmt;
type CountMap = AHashMap<CompactString, u64>;
type Entry = (CompactString, u64);
#[derive(Clone)]
// Sequential feed retains its map; parallel feed owns unique entries. Keeping
// both avoids a conversion solely to make their storage types identical.
pub(super) enum WordCounts {
    Map(CountMap),
    Entries(Vec<Entry>),
}
impl Default for WordCounts {
    fn default() -> Self {
        Self::Entries(Vec::new())
    }
}
impl WordCounts {
    // Each entry is a globally unique key; order has no contract.
    pub(super) fn from_entries(entries: Vec<Entry>) -> Self {
        Self::Entries(entries)
    }
    pub(super) fn from_map(map: CountMap) -> Self {
        Self::Map(map)
    }
    pub(super) fn view(&self) -> Words<'_> {
        match self {
            Self::Map(map) => Words::from_map(map),
            Self::Entries(entries) => Words { map: None, entries },
        }
    }
    pub(super) fn len(&self) -> usize {
        self.view().len()
    }
}
impl fmt::Debug for WordCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.view().iter()).finish()
    }
}
impl PartialEq for WordCounts {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        match (self, other) {
            (Self::Map(left), Self::Map(right)) => return left == right,
            (Self::Entries(left), Self::Entries(right)) if left == right => return true,
            (Self::Map(map), Self::Entries(entries)) | (Self::Entries(entries), Self::Map(map)) => {
                return entries
                    .iter()
                    .all(|(word, count)| map.get(word) == Some(count));
            }
            _ => {}
        }
        // Public content equality remains linear, with a temporary borrowed index.
        let lookup: AHashMap<&CompactString, u64> = other
            .view()
            .iter()
            .map(|(word, count)| (word, *count))
            .collect();
        self.view()
            .iter()
            .all(|(word, count)| lookup.get(word) == Some(count))
    }
}
impl Eq for WordCounts {}
impl Serialize for WordCounts {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (word, count) in self.view().iter() {
            map.serialize_entry(word, count)?;
        }
        map.end()
    }
}
impl<'de> Deserialize<'de> for WordCounts {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CountMap::deserialize(deserializer).map(Self::from_map)
    }
}
#[derive(Clone, Copy)]
// Borrow either representation (including public do_train's caller-owned map).
// Repeated train(&self) calls leave owned counts intact; only thin references
// are collected and sorted by CorpusPlan.
pub(super) struct Words<'a> {
    map: Option<&'a CountMap>,
    entries: &'a [Entry],
}
impl<'a> Words<'a> {
    pub(super) fn from_map(map: &'a CountMap) -> Self {
        Self {
            map: Some(map),
            entries: &[],
        }
    }
    pub(super) fn len(self) -> usize {
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
