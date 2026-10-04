//! Storage primitives shared by tokenizer algorithms.
mod arena;
mod id_accumulator;
mod interval_index;
mod position_buffer;
mod position_chains;
mod position_storage;
pub mod radix;
mod sorted_positions;

pub use arena::{AllocationArena, AllocationLease};
pub use id_accumulator::{IdAccumulator, IdDirectory};
pub use interval_index::{IntervalCursor, IntervalIndex};
pub use position_buffer::PositionBuffer;
pub use position_chains::{PositionChain, PositionChains};
pub use sorted_positions::{PositionCursor, PositionEncodingScratch, SortedPositions};

/// Invalid sorted input or a position allocation that cannot be represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageError(pub &'static str);
impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for StorageError {}
type Result<T> = std::result::Result<T, StorageError>;
