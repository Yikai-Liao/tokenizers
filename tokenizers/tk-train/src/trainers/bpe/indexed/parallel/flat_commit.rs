//! Reduce births by their selected rule and neighbor, without a temporary key table.
use super::*;

struct Fragment {
    output: usize,
    head: u32,
    next: u32,
}

pub(super) fn dense<O: Offset, const INLINE: usize>(
    outputs: &[Output<O, INLINE>],
    owner_index: usize,
    ledger: &mut Owner,
    rules: &[Rule],
    identities: usize,
    floor: u64,
    bits: u8,
) -> Result<(Vec<(u64, SmallPosting)>, AHashSet<u64>, usize)> {
    let mut retired = Vec::new();
    for output in outputs {
        for (&k, group) in &output.flat_routes[owner_index].delta {
            debug_assert_eq!(group.occurrences, 0);
            if let Some(entry) = ledger.entries.get_mut(&k) {
                entry.frequency = entry
                    .frequency
                    .checked_sub(group.weight)
                    .ok_or("old pair frequency underflow")?;
                if entry.frequency < floor {
                    ledger.entries.remove(&k);
                    retired.push((k, SmallPosting::default()));
                }
            }
        }
    }
    // Reuse a single vocabulary-sized directory for all rule/direction buckets.
    // Only touched neighbors are reset between buckets; no rules × vocabulary table.
    let mut indices = vec![NONE; identities];
    let mut totals = Vec::<(u32, Group)>::new();
    let mut fragments = Vec::<Fragment>::new();
    let mut dropped = 0;
    for (rank, rule) in rules.iter().enumerate() {
        for direction in 0..2 {
            let bucket = rank * 2 + direction;
            for (job, output) in outputs.iter().enumerate() {
                for (neighbor, group) in &output.flat_births[owner_index][bucket] {
                    let index = &mut indices[*neighbor as usize];
                    if *index == NONE {
                        *index =
                            u32::try_from(totals.len()).map_err(|_| "birth groups exceed u32")?;
                        totals.push((*neighbor, Group::default()));
                    }
                    let total = &mut totals[*index as usize].1;
                    let fragment =
                        u32::try_from(fragments.len()).map_err(|_| "birth fragments exceed u32")?;
                    if fragment == NONE {
                        return Err("birth fragment sentinel collision".into());
                    }
                    fragments.push(Fragment {
                        output: job,
                        head: group.head,
                        next: total.head,
                    });
                    total.head = fragment;
                    total.weight += group.weight;
                    total.occurrences = total
                        .occurrences
                        .checked_add(group.occurrences)
                        .ok_or("birth posting count exceeds u32")?;
                }
            }
            for (neighbor, group) in totals.drain(..) {
                indices[neighbor as usize] = NONE;
                if group.weight < floor {
                    dropped += 1;
                    continue;
                }
                let k = if direction == 0 {
                    key(neighbor, rule.replacement)
                } else {
                    key(rule.replacement, neighbor)
                };
                // A left birth's neighbor is unselected. A boundary joining two
                // selected spans is produced by the left span's right bucket.
                // Thus no final key occurs in two different buckets.
                debug_assert!(!ledger.entries.contains_key(&k));
                let mut fragment = group.head;
                let mut head = fragments[fragment as usize].head;
                let fragments = &fragments;
                let next = move || {
                    while head == NONE {
                        fragment = fragments[fragment as usize].next;
                        head = fragments[fragment as usize].head;
                    }
                    let route =
                        &outputs[fragments[fragment as usize].output].flat_routes[owner_index];
                    let index = head as usize;
                    let node = &route.nodes[index];
                    head = node.next;
                    route.high.address(index, node.position)
                };
                let positions = BlockPosting::from_reversed(group.occurrences, bits, next)?;
                debug_assert!(
                    positions
                        .iter(bits)
                        .collect::<Vec<_>>()
                        .windows(2)
                        .all(|w| w[0] < w[1])
                );
                ledger.entries.insert(
                    k,
                    Entry {
                        frequency: group.weight,
                        blocks: positions,
                    },
                );
                ledger.heap.push(Candidate {
                    key: k,
                    frequency: group.weight,
                });
            }
            fragments.clear();
        }
    }
    Ok((retired, AHashSet::new(), dropped))
}
