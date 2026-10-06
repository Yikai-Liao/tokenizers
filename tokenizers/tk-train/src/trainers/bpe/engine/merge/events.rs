//! Compact change references and stable routing to pair-count owners.
use super::super::{
    pair_index::ShardRouter,
    storage::{PositionChain, PositionChains},
};

/// One neighbor's removal and birth are committed together. With reusable IDs,
/// the two keys can coincide.
pub(in super::super) struct PairChanges {
    pub(in super::super) removed_key: u64,
    pub(in super::super) born_key: u64,
    pub(in super::super) removed_weight: u64,
    pub(in super::super) born_weight: u64,
    pub(in super::super) positions: PositionChain,

    pub(in super::super) bucket: u32,
}
#[derive(Clone, Copy)]
pub(in super::super) enum ChangeAction {
    Remove,
    Birth,
    Both,
}
/// Two low tag bits share one word with a record index. A resident Vec of
/// 48-byte records bounds its indices well below the available upper bits.
pub(in super::super) struct RoutedChangeRef {
    pub(in super::super) chunk: usize,
    index_and_action: usize,
}
impl RoutedChangeRef {
    fn new(chunk: usize, index: usize, action: ChangeAction) -> Self {
        Self {
            chunk,
            index_and_action: (index << 2) | action as usize,
        }
    }
    pub(in super::super) fn index(&self) -> usize {
        self.index_and_action >> 2
    }
    pub(in super::super) fn action(&self) -> ChangeAction {
        match self.index_and_action & 3 {
            0 => ChangeAction::Remove,
            1 => ChangeAction::Birth,
            2 => ChangeAction::Both,
            _ => unreachable!("record tags are constructed from ChangeAction"),
        }
    }
}
/// One owner's original-order count actions and separately grouped birth indices.
/// Birth grouping may reorder only `births`; `changes` preserves checked update
/// order, including removal-before-birth when both keys route to the same owner.
pub(in super::super) struct OwnerRoute {
    pub(in super::super) changes: Vec<RoutedChangeRef>,
    pub(in super::super) births: Vec<usize>,
}
pub(in super::super) struct EventChunk {
    pub(in super::super) chains: PositionChains,
    pub(in super::super) changes: Vec<PairChanges>,
}
/// Owns buffered position chains until the joined commit finishes borrowing them.
/// Already encoded complete births retain only removal events here.
pub(in super::super) struct MergeEvents {
    pub(in super::super) buckets: usize,
    pub(in super::super) chunks: Vec<EventChunk>,
}
impl MergeEvents {
    /// One directory per owner for the whole batch. Only actual actions and
    /// births occupy entries; producers do not allocate an owner/bucket matrix.
    #[cfg(test)]
    pub(in super::super) fn route(&self, workers: usize) -> Vec<OwnerRoute> {
        {
            let mut routes = self.dispatch_with_router(ShardRouter::new(workers));
            for route in &mut routes {
                route.group_births(self);
            }
            routes
        }
    }

    /// Route only metadata. Birth grouping can run inside the owner task.
    pub(in super::super) fn dispatch_with_router(&self, router: ShardRouter) -> Vec<OwnerRoute> {
        let mut routes: Vec<_> = (0..router.shards())
            .map(|_| OwnerRoute {
                changes: Vec::new(),
                births: Vec::new(),
            })
            .collect();
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            for (index, change) in chunk.changes.iter().enumerate() {
                debug_assert!((change.bucket as usize) < self.buckets);
                let removed =
                    (change.removed_weight != 0).then(|| router.owner(change.removed_key));
                // Zero-weight identity-reuse births still own positions. Only an empty
                // chain has no birth action; weight alone cannot decide this.
                let born = (!change.positions.is_empty()).then(|| router.owner(change.born_key));
                match (removed, born) {
                    (Some(removed), Some(born)) if removed == born => {
                        let route = &mut routes[removed];
                        route.births.push(route.changes.len());
                        route.changes.push(RoutedChangeRef::new(
                            chunk_index,
                            index,
                            ChangeAction::Both,
                        ));
                    }
                    (removed, born) => {
                        if let Some(owner) = removed {
                            routes[owner].changes.push(RoutedChangeRef::new(
                                chunk_index,
                                index,
                                ChangeAction::Remove,
                            ));
                        }
                        if let Some(owner) = born {
                            let route = &mut routes[owner];
                            route.births.push(route.changes.len());
                            route.changes.push(RoutedChangeRef::new(
                                chunk_index,
                                index,
                                ChangeAction::Birth,
                            ));
                        }
                    }
                }
            }
        }
        routes
    }
}
impl OwnerRoute {
    /// Stably group birth references by rule/direction without changing actions.
    /// Equal-bucket fragments keep producer order for the fresh spatial encoder;
    /// reuse uses an encoder that also handles genuinely interleaved chains.
    pub(in super::super) fn group_births(&mut self, events: &MergeEvents) {
        if self.births.len() < 2 {
            return;
        }
        let bucket_of = |index: usize| {
            let reference = &self.changes[index];
            events.chunks[reference.chunk].changes[reference.index()].bucket as usize
        };
        let mut offsets = vec![0_usize; events.buckets];
        for &index in &self.births {
            offsets[bucket_of(index)] += 1;
        }
        let mut total = 0;
        for offset in &mut offsets {
            let count = *offset;
            *offset = total;
            total += count;
        }
        let mut births = vec![0; self.births.len()];
        for &index in &self.births {
            let offset = &mut offsets[bucket_of(index)];
            births[*offset] = index;
            *offset += 1;
        }
        self.births = births;
    }
}
