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
