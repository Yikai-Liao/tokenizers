//! Immutable AVL epochs with exact argmax augmentation. Bulk replacements split
//! by touched keys, share untouched subtrees and compressed position handles,
//! and switch the root only after all recursive tasks have joined.
use super::super::storage::SortedPositions;
use super::{PairPriority, PairState};
use std::sync::Arc;

type Link<'arena> = Option<Arc<Node<'arena>>>;
#[derive(Clone)]
pub(super) struct Value<'arena> {
    pub(super) count: u64,
    pub(super) positions: Arc<SortedPositions<'arena>>,
}
struct Node<'arena> {
    key: u64,
    value: Value<'arena>,
    left: Link<'arena>,
    right: Link<'arena>,
    height: u8,
    best: PairPriority,
}
pub(super) struct Update<'arena> {
    pub(super) key: u64,
    pub(super) value: Option<Value<'arena>>,
}
pub(super) struct EpochIndex<'arena> {
    root: Link<'arena>,
    floor: u64,
}
fn height(node: &Link<'_>) -> u8 {
    node.as_ref().map_or(0, |node| node.height)
}
fn node<'arena>(
    key: u64,
    value: Value<'arena>,
    left: Link<'arena>,
    right: Link<'arena>,
    floor: u64,
) -> Link<'arena> {
    let mut best = if value.count >= floor {
        PairPriority {
            key,
            priority_count: value.count,
        }
    } else {
        PairPriority {
            key: u64::MAX,
            priority_count: 0,
        }
    };
    for child in [&left, &right].into_iter().flatten() {
        best = best.max(child.best);
    }
    let height = 1 + height(&left).max(height(&right));
    Some(Arc::new(Node {
        key,
        value,
        left,
        right,
        height,
        best,
    }))
}
fn balance<'arena>(
    key: u64,
    value: Value<'arena>,
    left: Link<'arena>,
    right: Link<'arena>,
    floor: u64,
) -> Link<'arena> {
    let left_height = height(&left);
    let right_height = height(&right);
    if left_height > right_height + 1 {
        let root = left.as_ref().expect("heavy left subtree");
        if height(&root.left) >= height(&root.right) {
            let tail = node(key, value, root.right.clone(), right, floor);
            node(root.key, root.value.clone(), root.left.clone(), tail, floor)
        } else {
            let middle = root.right.as_ref().expect("double rotation middle");
            let head = node(
                root.key,
                root.value.clone(),
                root.left.clone(),
                middle.left.clone(),
                floor,
            );
            let tail = node(key, value, middle.right.clone(), right, floor);
            node(middle.key, middle.value.clone(), head, tail, floor)
        }
    } else if right_height > left_height + 1 {
        let root = right.as_ref().expect("heavy right subtree");
        if height(&root.right) >= height(&root.left) {
            let head = node(key, value, left, root.left.clone(), floor);
            node(
                root.key,
                root.value.clone(),
                head,
                root.right.clone(),
                floor,
            )
        } else {
            let middle = root.left.as_ref().expect("double rotation middle");
            let head = node(key, value, left, middle.left.clone(), floor);
            let tail = node(
                root.key,
                root.value.clone(),
                middle.right.clone(),
                root.right.clone(),
                floor,
            );
            node(middle.key, middle.value.clone(), head, tail, floor)
        }
    } else {
        node(key, value, left, right, floor)
    }
}
/// Join two ordered AVL subtrees around one key, copying only affected paths.
fn join<'arena>(
    left: Link<'arena>,
    key: u64,
    value: Value<'arena>,
    right: Link<'arena>,
    floor: u64,
) -> Link<'arena> {
    if height(&left) > height(&right) + 1 {
        let root = left.as_ref().expect("heavy left subtree");
        let tail = join(root.right.clone(), key, value, right, floor);
        balance(root.key, root.value.clone(), root.left.clone(), tail, floor)
    } else if height(&right) > height(&left) + 1 {
        let root = right.as_ref().expect("heavy right subtree");
        let head = join(left, key, value, root.left.clone(), floor);
        balance(
            root.key,
            root.value.clone(),
            head,
            root.right.clone(),
            floor,
        )
    } else {
        node(key, value, left, right, floor)
    }
}
fn split<'arena>(
    root: &Link<'arena>,
    key: u64,
    floor: u64,
) -> (Link<'arena>, Option<Value<'arena>>, Link<'arena>) {
    let Some(root) = root else {
        return (None, None, None);
    };
    match key.cmp(&root.key) {
        std::cmp::Ordering::Equal => (
            root.left.clone(),
            Some(root.value.clone()),
            root.right.clone(),
        ),
        std::cmp::Ordering::Less => {
            let (left, value, middle) = split(&root.left, key, floor);
            (
                left,
                value,
                join(
                    middle,
                    root.key,
                    root.value.clone(),
                    root.right.clone(),
                    floor,
                ),
            )
        }
        std::cmp::Ordering::Greater => {
            let (middle, value, right) = split(&root.right, key, floor);
            (
                join(
                    root.left.clone(),
                    root.key,
                    root.value.clone(),
                    middle,
                    floor,
                ),
                value,
                right,
            )
        }
    }
}
fn pop_min<'arena>(root: &Arc<Node<'arena>>, floor: u64) -> (u64, Value<'arena>, Link<'arena>) {
    if let Some(left) = &root.left {
        let (key, value, tail) = pop_min(left, floor);
        (
            key,
            value,
            balance(
                root.key,
                root.value.clone(),
                tail,
                root.right.clone(),
                floor,
            ),
        )
    } else {
        (root.key, root.value.clone(), root.right.clone())
    }
}
fn concatenate<'arena>(left: Link<'arena>, right: Link<'arena>, floor: u64) -> Link<'arena> {
    match (left, right) {
        (None, right) => right,
        (left, None) => left,
        (left, Some(right)) => {
            let (key, value, tail) = pop_min(&right, floor);
            join(left, key, value, tail, floor)
        }
    }
}
fn bulk<'arena>(root: &Link<'arena>, updates: &[Update<'arena>], floor: u64) -> Link<'arena> {
    if updates.is_empty() {
        return root.clone();
    }
    let middle = updates.len() / 2;
    let update = &updates[middle];
    let (left, _, right) = split(root, update.key, floor);
    let (left, right) = if updates.len() >= 128 {
        rayon::join(
            || bulk(&left, &updates[..middle], floor),
            || bulk(&right, &updates[middle + 1..], floor),
        )
    } else {
        (
            bulk(&left, &updates[..middle], floor),
            bulk(&right, &updates[middle + 1..], floor),
        )
    };
    if let Some(value) = &update.value {
        join(left, update.key, value.clone(), right, floor)
    } else {
        concatenate(left, right, floor)
    }
}
fn build<'arena>(items: &mut [(u64, Option<Value<'arena>>)], floor: u64) -> Link<'arena> {
    if items.is_empty() {
        return None;
    }
    let middle = items.len() / 2;
    let (head, rest) = items.split_at_mut(middle);
    let (center, tail) = rest.split_first_mut().expect("nonempty middle");
    let (left, right) = if head.len() + tail.len() >= 8192 {
        rayon::join(|| build(head, floor), || build(tail, floor))
    } else {
        (build(head, floor), build(tail, floor))
    };
    node(
        center.0,
        center.1.take().expect("one owner per initial state"),
        left,
        right,
        floor,
    )
}
impl<'arena> EpochIndex<'arena> {
    pub(super) fn from_states(
        states: impl Iterator<Item = (u64, PairState<'arena>)>,
        floor: u64,
    ) -> Self {
        let mut items: Vec<_> = states
            .map(|(key, state)| {
                (
                    key,
                    Some(Value {
                        count: state.ledger_count_bits,
                        positions: Arc::new(state.positions),
                    }),
                )
            })
            .collect();
        items.sort_unstable_by_key(|item| item.0);
        Self {
            root: build(&mut items, floor),
            floor,
        }
    }
    pub(super) fn get(&self, key: u64) -> Option<&Value<'arena>> {
        let mut root = self.root.as_ref();
        while let Some(node) = root {
            match key.cmp(&node.key) {
                std::cmp::Ordering::Equal => return Some(&node.value),
                std::cmp::Ordering::Less => root = node.left.as_ref(),
                std::cmp::Ordering::Greater => root = node.right.as_ref(),
            }
        }
        None
    }
    pub(super) fn best(&self) -> Option<PairPriority> {
        self.root
            .as_ref()
            .map(|node| node.best)
            .filter(|priority| priority.priority_count != 0)
    }
    pub(super) fn take_best(&mut self) -> (PairPriority, SortedPositions<'arena>) {
        let best = self.best().expect("selection certified the epoch winner");
        let (left, value, right) = split(&self.root, best.key, self.floor);
        self.root = concatenate(left, right, self.floor);
        let value = value.expect("certified state exists");
        let positions = Arc::try_unwrap(value.positions)
            .unwrap_or_else(|_| panic!("previous epoch readers joined before selection"));
        (best, positions)
    }
    pub(super) fn publish(&mut self, updates: &mut [Update<'arena>]) {
        updates.sort_unstable_by_key(|update| update.key);
        assert!(
            updates.windows(2).all(|pair| pair[0].key < pair[1].key),
            "one complete replacement per key"
        );
        let root = bulk(&self.root, updates, self.floor);
        self.root = root;
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        execution::Execution,
        storage::{AllocationArena, PositionEncodingScratch},
    };
    use super::*;
    use std::collections::BTreeMap;
    fn audit(root: &Link<'_>, floor: u64, output: &mut Vec<(u64, u64)>) -> (u8, PairPriority) {
        let empty = PairPriority {
            key: u64::MAX,
            priority_count: 0,
        };
        let Some(root) = root else {
            return (0, empty);
        };
        let (left_height, left_best) = audit(&root.left, floor, output);
        if let Some(&(key, _)) = output.last() {
            assert!(key < root.key);
        }
        output.push((root.key, root.value.count));
        let (right_height, right_best) = audit(&root.right, floor, output);
        assert!(
            left_height.abs_diff(right_height) <= 1,
            "AVL balance at {}",
            root.key
        );
        assert_eq!(root.height, 1 + left_height.max(right_height));
        let own = if root.value.count >= floor {
            PairPriority {
                key: root.key,
                priority_count: root.value.count,
            }
        } else {
            empty
        };
        let best = own.max(left_best).max(right_best);
        assert_eq!(root.best, best);
        (root.height, best)
    }
    #[test]
    fn immutable_bulk_epochs_match_ordered_map_and_exact_argmax() {
        for workers in [1, 4] {
            let execution = Execution::new(workers).unwrap();
            let arena = AllocationArena::new(workers, 1_000_000);
            execution.pool.install(|| {
                let floor = 2;
                let coordinate = |key: u64| u64::MAX - key;
                let key_of = |index: u64| (index << 32) | (4095 - index);
                let make = |key, count| {
                    let lease = arena.lease(execution.current_worker());
                    let mut scratch = PositionEncodingScratch::default();
                    Value {
                        count,
                        positions: Arc::new(
                            SortedPositions::from_sorted(&[coordinate(key)], &mut scratch, &lease)
                                .unwrap(),
                        ),
                    }
                };
                let mut expected = BTreeMap::new();
                let states: Vec<_> = (0..1024)
                    .map(|index| {
                        let key = key_of(index);
                        let count = index * 37 % 101;
                        expected.insert(key, count);
                        let value = make(key, count);
                        (
                            key,
                            PairState {
                                ledger_count_bits: count,
                                positions: Arc::try_unwrap(value.positions).ok().unwrap(),
                            },
                        )
                    })
                    .collect();
                let mut epoch = EpochIndex::from_states(states.into_iter(), floor);
                let mut random = 19_u64;
                for round in 0..40 {
                    let snapshot = epoch.root.clone();
                    let mut old_items = Vec::new();
                    audit(&snapshot, floor, &mut old_items);
                    let mut replacements = BTreeMap::new();
                    for _ in 0..(129 + round * 7) {
                        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let key = key_of(random >> 32 & 2047);
                        let count = (random >> 13) % 97;
                        replacements.insert(key, if random & 7 == 0 { None } else { Some(count) });
                    }
                    let mut updates: Vec<_> = replacements
                        .into_iter()
                        .map(|(key, count)| {
                            if let Some(count) = count {
                                expected.insert(key, count);
                            } else {
                                expected.remove(&key);
                            }
                            Update {
                                key,
                                value: count.map(|count| make(key, count)),
                            }
                        })
                        .collect();
                    epoch.publish(&mut updates);
                    drop(updates);
                    let mut items = Vec::new();
                    audit(&epoch.root, floor, &mut items);
                    assert_eq!(
                        items,
                        expected
                            .iter()
                            .map(|(&key, &count)| (key, count))
                            .collect::<Vec<_>>()
                    );
                    let mut old_again = Vec::new();
                    audit(&snapshot, floor, &mut old_again);
                    assert_eq!(
                        old_again, old_items,
                        "publication preserves the prior epoch"
                    );
                    let best = expected
                        .iter()
                        .filter(|(_, count)| **count >= floor)
                        .map(|(&key, &count)| PairPriority {
                            key,
                            priority_count: count,
                        })
                        .max();
                    assert_eq!(epoch.best(), best);
                    drop(snapshot);
                }
                while let Some(best) = epoch.best() {
                    let independent = expected
                        .iter()
                        .filter(|(_, count)| **count >= floor)
                        .map(|(&key, &count)| PairPriority {
                            key,
                            priority_count: count,
                        })
                        .max()
                        .unwrap();
                    assert_eq!(best, independent);
                    let (priority, positions) = epoch.take_best();
                    assert_eq!(priority, independent);
                    assert_eq!(
                        positions.iter().collect::<Vec<_>>(),
                        [coordinate(priority.key)]
                    );
                    expected.remove(&priority.key);
                    let mut items = Vec::new();
                    audit(&epoch.root, floor, &mut items);
                }
                assert!(expected.values().all(|&count| count < floor));
            });
        }
    }
}
