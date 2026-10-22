//! One insertion-ordered owner for token strings and their IDs.
use ahash::RandomState;
use compact_str::CompactString;
use indexmap::IndexSet;
use tk_encode::models::bpe::{Merges, Pair, Vocab};

pub(super) struct Vocabulary {
    tokens: IndexSet<CompactString, RandomState>,
    prefix: Option<String>,
    merges: Vec<Pair>,
}

impl Vocabulary {
    pub(super) fn new(capacity: usize, prefix: Option<String>) -> Self {
        Self {
            tokens: IndexSet::with_capacity_and_hasher(capacity, RandomState::default()),
            prefix,
            merges: Vec::new(),
        }
    }

    pub(super) fn intern(&mut self, token: CompactString) -> u32 {
        self.tokens.insert_full(token).0 as u32
    }

    pub(super) fn contains(&self, token: &str) -> bool {
        self.tokens.contains(token)
    }

    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn resolve(&mut self, pair: Pair) -> u32 {
        let left = self.tokens[pair.0 as usize].as_str();
        let right = self.tokens[pair.1 as usize].as_str();
        let right = self
            .prefix
            .as_deref()
            .and_then(|p| right.strip_prefix(p))
            .unwrap_or(right);
        let mut text = CompactString::with_capacity(left.len() + right.len());
        text.push_str(left);
        text.push_str(right);
        self.intern(text)
    }

    pub(super) fn commit(&mut self, pair: Pair) {
        self.merges.push(pair);
    }

    pub(super) fn into_parts(self) -> (Vocab, Merges) {
        let Self { tokens, merges, .. } = self;
        let merges = merges
            .into_iter()
            .map(|(left, right)| {
                (
                    tokens[left as usize].to_string(),
                    tokens[right as usize].to_string(),
                )
            })
            .collect();
        (
            tokens
                .into_iter()
                .enumerate()
                .map(|(id, token)| (token.to_string(), id as u32))
                .collect(),
            merges,
        )
    }
}
