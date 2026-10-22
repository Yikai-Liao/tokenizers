//! Immutable nondecreasing full-u64 occurrence cohorts; duplicates are retained.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Positions(Vec<u64>);

impl Positions {
    pub(super) fn from_unsorted(mut values: Vec<u64>) -> Self {
        values.sort_unstable();
        Self(values)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.0.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_width_coordinates_and_duplicates_roundtrip() {
        let values = Positions::from_unsorted(vec![u64::MAX, 128, 0, 127, u64::MAX, 128]);
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            [0, 127, 128, 128, u64::MAX, u64::MAX]
        );
    }
}
