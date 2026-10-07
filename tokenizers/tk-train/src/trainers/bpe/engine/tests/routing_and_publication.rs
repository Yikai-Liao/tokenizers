//! Original event order, owner routing, and parallel rule/position order.
use super::*;

#[test]
fn routing_keeps_stable_births_and_original_count_actions() {
    use super::storage::{PositionChain, PositionChains};
    use merge::{ChangeAction, EventChunk, MergeEvents, PairChanges};
    use pair_index::{pair_key, shard_for};

    let chunks = (0..7_u32)
        .map(|producer| {
            let mut chains = PositionChains::new();
            let changes = (0..97_u32)
                .map(|index| {
                    let removed_key = pair_key((u32::MAX - producer, index));
                    let born_key = if index % 5 == 0 {
                        removed_key
                    } else {
                        pair_key((index, u32::MAX - producer))
                    };
                    let mut positions = PositionChain::default();
                    if index % 3 != 0 {
                        chains
                            .push(
                                &mut positions,
                                (u64::from(producer) << 32) + u64::from(index),
                            )
                            .unwrap();
                    }
                    PairChanges {
                        removed_key,
                        born_key,
                        removed_weight: u64::from(index % 4 != 0),
                        born_weight: 0,
                        positions,
                        bucket: (producer * 41 + index * 79) % 512,
                    }
                })
                .collect();
            EventChunk { chains, changes }
        })
        .collect();
    let events = MergeEvents {
        buckets: 512,
        chunks,
    };
    for workers in [1, 4, 16, 64] {
        let routes = events.route(workers);
        for (owner, route) in routes.iter().enumerate() {
            let mut expected_actions = Vec::new();
            for (producer, chunk) in events.chunks.iter().enumerate() {
                for (index, change) in chunk.changes.iter().enumerate() {
                    let removal = change.removed_weight != 0
                        && shard_for(change.removed_key, workers) == owner;
                    let birth = !change.positions.is_empty()
                        && shard_for(change.born_key, workers) == owner;
                    if removal || birth {
                        expected_actions.push((producer, index, removal, birth));
                    }
                }
            }
            let actual_actions: Vec<_> = route
                .changes
                .iter()
                .map(|reference| {
                    (
                        reference.chunk,
                        reference.index(),
                        matches!(
                            reference.action(),
                            ChangeAction::Remove | ChangeAction::Both
                        ),
                        matches!(reference.action(), ChangeAction::Birth | ChangeAction::Both),
                    )
                })
                .collect();
            assert_eq!(
                actual_actions, expected_actions,
                "workers={workers}, owner={owner}"
            );

            // Enumerate buckets explicitly as the oracle. Inside each bucket,
            // the original producer/record order is the required stable order.
            let mut expected_births = Vec::new();
            for bucket in 0..512 {
                for (producer, chunk) in events.chunks.iter().enumerate() {
                    for (index, change) in chunk.changes.iter().enumerate() {
                        if change.bucket == bucket
                            && !change.positions.is_empty()
                            && shard_for(change.born_key, workers) == owner
                        {
                            expected_births.push((producer, index));
                        }
                    }
                }
            }
            let actual_births: Vec<_> = route
                .births
                .iter()
                .map(|&index| {
                    let reference = &route.changes[index];
                    (reference.chunk, reference.index())
                })
                .collect();
            assert_eq!(
                actual_births, expected_births,
                "workers={workers}, owner={owner}"
            );
        }
    }
    assert!(
        MergeEvents {
            buckets: 0,
            chunks: Vec::new()
        }
        .route(4)
        .iter()
        .all(|route| route.changes.is_empty() && route.births.is_empty())
    );
}

#[test]
fn reusable_routes_clear_old_actions_and_regroup_for_new_bucket_domain() {
    use super::storage::{PositionChain, PositionChains};
    use merge::{ChangeAction, EventChunk, MergeEvents, OwnerRoute, PairChanges};
    use pair_index::{ShardRouter, pair_key};

    let mut chains = PositionChains::new();
    let mut changes = Vec::new();
    for i in 0..24_u32 {
        let mut positions = PositionChain::default();
        chains.push(&mut positions, u64::from(i)).unwrap();
        changes.push(PairChanges {
            removed_key: pair_key((i, i + 1)),
            born_key: pair_key((i + 1, i)),
            removed_weight: u64::from(i % 2 == 0),
            born_weight: 0,
            positions,
            bucket: i % 8,
        });
    }
    let first = MergeEvents {
        buckets: 8,
        chunks: vec![EventChunk { chains, changes }],
    };
    let mut routes = (0..4).map(|_| OwnerRoute::default()).collect::<Vec<_>>();
    first.dispatch_into(&mut routes, ShardRouter::new(4));
    for route in &mut routes {
        route.group_births(&first);
    }
    let capacities: Vec<_> = routes
        .iter()
        .map(|route| (route.changes.capacity(), route.births.capacity()))
        .collect();

    let mut chains = PositionChains::new();
    let mut zero_weight_birth = PositionChain::default();
    chains.push(&mut zero_weight_birth, 99).unwrap();
    let mut reused_id_birth = PositionChain::default();
    chains.push(&mut reused_id_birth, 100).unwrap();
    let second = MergeEvents {
        buckets: 1,
        chunks: vec![EventChunk {
            chains,
            changes: vec![
                PairChanges {
                    removed_key: pair_key((1, 2)),
                    born_key: pair_key((2, 1)),
                    removed_weight: 0,
                    born_weight: 0,
                    positions: zero_weight_birth,
                    bucket: 0,
                },
                PairChanges {
                    removed_key: pair_key((3, 4)),
                    born_key: pair_key((3, 4)),
                    removed_weight: 1,
                    born_weight: 1,
                    positions: reused_id_birth,
                    bucket: 0,
                },
            ],
        }],
    };
    second.dispatch_into(&mut routes, ShardRouter::new(4));
    for route in &mut routes {
        route.group_births(&second);
    }
    let active: Vec<_> = routes
        .iter()
        .flat_map(|route| route.changes.iter().map(|reference| reference.action()))
        .collect();
    assert_eq!(active.len(), 2);
    assert!(
        active
            .iter()
            .any(|action| matches!(action, ChangeAction::Birth))
    );
    assert!(
        active
            .iter()
            .any(|action| matches!(action, ChangeAction::Both))
    );
    assert_eq!(
        routes.iter().map(|route| route.births.len()).sum::<usize>(),
        2
    );
    assert!(routes.iter().enumerate().all(|(index, route)| {
        route.changes.capacity() >= capacities[index].0
            && route.births.capacity() >= capacities[index].1
    }));
}

#[test]
fn routed_batches_preserve_rule_and_position_order_with_many_workers() {
    let mut words = counts(&[("aaaaaaa", 5), ("abcabc", 3), ("baab", 0), ("", 1)]);
    for index in 0..128_u32 {
        let word: String = (0..3)
            .map(|offset| char::from_u32(0x4000 + index * 3 + offset).unwrap())
            .collect();
        words.insert(word.into(), 2 + u64::from(index % 3));
    }
    let trainer = BpeTrainer::builder()
        .vocab_size(700)
        .min_frequency(2)
        .show_progress(false)
        .build();
    check_with_workers(&trainer, &words, &[1, 4, 16, 64]);

    let mut aliases = trainer;
    aliases.continuing_subword_prefix = Some("a".into());
    aliases.end_of_word_suffix = Some("a".into());
    aliases.max_token_length = Some(5);
    check_with_workers(
        &aliases,
        &counts(&[("aaaaaaa", 5), ("abcabc", 3), ("baab", 0)]),
        &[1, 16, 64],
    );
}
