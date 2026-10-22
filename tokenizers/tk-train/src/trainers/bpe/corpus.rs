//! Stable original-symbol coordinates for word-local training edits.
use tk_encode::models::bpe::Pair;

const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    span: usize,
}

#[derive(Clone, Default)]
pub(super) struct Word {
    symbols: Vec<Symbol>,
}

impl Word {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn add(&mut self, id: u32) {
        self.symbols.push(Symbol { id, span: 1 });
    }

    pub(super) fn get_chars(&self) -> Vec<u32> {
        let mut position = 0;
        std::iter::from_fn(|| {
            let symbol = *self.symbols.get(position)?;
            position += symbol.span;
            Some(symbol.id)
        })
        .collect()
    }

    pub(super) fn merge(&mut self, c1: u32, c2: u32, id: u32, limit: usize) -> Vec<(Pair, i32)> {
        let mut changes = Vec::new();
        let mut previous: Option<usize> = None;
        let mut position = 0;
        while position < self.symbols.len() {
            let left = self.symbols[position];
            let right = position + left.span;
            if left.id != c1 || right >= self.symbols.len() || self.symbols[right].id != c2 {
                previous = Some(position);
                position = right;
                continue;
            }
            let span = left.span + self.symbols[right].span;
            let after = position + span;
            if let Some(before) = previous {
                let neighbor = self.symbols[before];
                changes.push(((neighbor.id, c1), -1));
                if neighbor.span + span < limit {
                    changes.push(((neighbor.id, id), 1));
                }
            }
            // Heads and tails identify each logical token; spans skip interior holes.
            self.symbols[position] = Symbol { id, span };
            if right + 1 != after {
                self.symbols[right] = Symbol { id: EMPTY, span: 0 };
            }
            self.symbols[after - 1] = Symbol { id, span };
            if after < self.symbols.len() {
                let neighbor = self.symbols[after];
                changes.push(((c2, neighbor.id), -1));
                if span + neighbor.span < limit {
                    changes.push(((id, neighbor.id), 1));
                }
            }
            previous = Some(position);
            position = after;
        }
        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0); // 'h'
        word.add(1); // 'e'
        word.add(2); // 'l'
        word.add(2); // 'l'
        word.add(3); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, usize::MAX);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        // training. This merge affects the counts for the pairs
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), and
        // ('ll', 'o') ~= (4, 3).
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                ((1u32, 4u32), 1i32),  // count for ('e', 'll') should be increased by 1.
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                ((4u32, 3u32), 1i32),  // count for ('ll', 'o') should be increased by 1.
            ]
        );
    }

    #[test]
    fn test_merge_max_length() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0); // 'h'
        word.add(1); // 'e'
        word.add(2); // 'l'
        word.add(2); // 'l'
        word.add(3); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, 2);
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                // ((1u32, 4u32), 1i32),  Missing since this would be larger than 2
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                                       // ((4u32, 3u32), 1i32), Missing since this would be larger than 2
            ]
        );
    }

    #[test]
    fn reused_id_retains_per_occurrence_geometry() {
        let mut word = Word::new();
        for id in [0, 0, 0, 0, 0] {
            word.add(id);
        }
        word.merge(0, 0, 0, usize::MAX);
        assert_eq!(word.get_chars(), [0, 0, 0]);
        word.merge(0, 0, 1, usize::MAX);
        assert_eq!(word.get_chars(), [1, 0]);
        assert_eq!((word.symbols[0].span, word.symbols[4].span), (4, 1));
    }

    #[test]
    fn adjacent_matches_preserve_intermediate_neighbor_events() {
        let mut word = Word::new();
        for id in [0, 1, 0, 1] {
            word.add(id);
        }
        assert_eq!(
            word.merge(0, 1, 2, usize::MAX),
            [((1, 0), -1), ((2, 0), 1), ((2, 0), -1), ((2, 2), 1)]
        );
        assert_eq!(word.get_chars(), [2, 2]);
    }
}
