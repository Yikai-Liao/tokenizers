//! Task-local neighbor directories aggregate hash work once per touched key.
use super::*;

struct LocalGroup {
    neighbor: u32,
    removed_weight: u64,
    born: Group,
    tail: u32,
}

pub(super) struct Scratch {
    indices: Vec<u32>,
    groups: Vec<LocalGroup>,
}
impl Scratch {
    pub(super) fn new(identities: usize) -> Self {
        Self {
            indices: vec![NONE; identities],
            groups: Vec::new(),
        }
    }
    fn group(&mut self, neighbor: u32) -> &mut LocalGroup {
        let index = &mut self.indices[neighbor as usize];
        if *index == NONE {
            // The directory's capped ID domain also bounds group count.
            *index = self.groups.len() as u32;
            self.groups.push(LocalGroup {
                neighbor,
                removed_weight: 0,
                born: Group::default(),
                tail: NONE,
            });
        }
        &mut self.groups[*index as usize]
    }
    pub(super) fn remove(&mut self, neighbor: u32, weight: u64) {
        self.group(neighbor).removed_weight += weight;
    }
    pub(super) fn birth<O: Offset, const INLINE: usize>(
        &mut self,
        output: &mut Output<O, INLINE>,
        neighbor: u32,
        k: u64,
        position: usize,
        weight: u64,
    ) -> Result<()> {
        let record = self.group(neighbor);
        let o = owner(k, output.flat_routes.len());
        let route = &mut output.flat_routes[o];
        let head = u32::try_from(route.nodes.len()).map_err(|_| "birth chain exceeds u32")?;
        if head == NONE {
            return Err("birth chain sentinel collision".into());
        }
        let position = u32::try_from(position).map_err(|_| "flat birth position exceeds u32")?;
        let count = record
            .born
            .occurrences
            .checked_add(1)
            .ok_or("birth posting count exceeds u32")?;
        route.nodes.push(Node {
            position,
            next: record.born.head,
        });
        if record.born.occurrences == 0 {
            record.tail = head;
        }
        record.born.head = head;
        record.born.occurrences = count;
        record.born.weight += weight;
        Ok(())
    }
    pub(super) fn flush<O: Offset, const INLINE: usize>(
        &mut self,
        output: &mut Output<O, INLINE>,
        rule: &Rule,
        left: bool,
    ) -> Result<()> {
        for record in self.groups.drain(..) {
            self.indices[record.neighbor as usize] = NONE;
            if record.removed_weight != 0 {
                output.remove(
                    if left {
                        key(record.neighbor, rule.edge.0)
                    } else {
                        key(rule.edge.1, record.neighbor)
                    },
                    record.removed_weight,
                );
            }
            if record.born.occurrences == 0 {
                continue;
            }
            let k = if left {
                key(record.neighbor, rule.replacement)
            } else {
                key(rule.replacement, record.neighbor)
            };
            let o = owner(k, output.flat_routes.len());
            let route = &mut output.flat_routes[o];
            let group = route.delta.entry(k).or_default();
            let count = group
                .occurrences
                .checked_add(record.born.occurrences)
                .ok_or("birth posting count exceeds u32")?;
            // This key contains the newly activated identity, so an existing
            // delta can only be an earlier birth, never an old-edge removal.
            debug_assert!(group.occurrences != 0 || group.weight == 0);
            debug_assert_eq!(route.nodes[record.tail as usize].next, NONE);
            route.nodes[record.tail as usize].next = group.head;
            group.head = record.born.head;
            group.occurrences = count;
            group.weight += record.born.weight;
        }
        Ok(())
    }
    pub(super) fn bytes(&self) -> usize {
        self.indices.capacity() * 4 + self.groups.capacity() * std::mem::size_of::<LocalGroup>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(route: &Route, k: u64) -> Vec<u32> {
        let mut head = route.delta[&k].head;
        let mut positions = Vec::new();
        while head != NONE {
            let node = &route.nodes[head as usize];
            positions.push(node.position);
            head = node.next;
        }
        positions
    }
    #[test]
    fn flush_links_multiple_tasks_and_preserves_zero_weight_births() {
        let mut output = Output::<u32, 2>::new(3, true);
        let mut scratch = Scratch::new(16);
        let rule = Rule {
            edge: (1, 2),
            replacement: 9,
            left_len: 1,
            right_len: 1,
        };
        let born = key(3, 9);
        for positions in [&[2, 4][..], &[], &[8, 12]] {
            for &p in positions {
                scratch.remove(3, 1);
                scratch.birth(&mut output, 3, born, p, 0).unwrap();
            }
            scratch.flush(&mut output, &rule, true).unwrap();
        }
        let route = &output.flat_routes[owner(born, 3)];
        assert_eq!(route.delta[&born].weight, 0);
        assert_eq!(route.delta[&born].occurrences, 4);
        assert_eq!(chain(route, born), vec![12, 8, 4, 2]);
        assert_eq!(
            output.flat_routes[owner(key(3, 1), 3)].delta[&key(3, 1)].weight,
            4
        );
        // A selected right neighbor can have different old and final IDs.
        scratch.remove(2, 5);
        scratch.birth(&mut output, 7, key(9, 7), 14, 5).unwrap();
        scratch.flush(&mut output, &rule, false).unwrap();
        assert_eq!(
            output.flat_routes[owner(key(2, 2), 3)].delta[&key(2, 2)].weight,
            5
        );
        assert_eq!(
            chain(&output.flat_routes[owner(key(9, 7), 3)], key(9, 7)),
            vec![14]
        );
        assert!(scratch.indices.iter().all(|&i| i == NONE));
    }
}
