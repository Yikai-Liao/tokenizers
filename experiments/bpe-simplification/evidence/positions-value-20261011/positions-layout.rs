#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path = "source/direct/positions.rs"]
mod positions;
struct Candidate { count: u64, pair: (u32, u32), positions: positions::Positions }
fn main() {
    println!("Positions={} Buffer={} Candidate={} usize={}",
        std::mem::size_of::<positions::Positions>(),
        std::mem::size_of::<positions::Buffer>(),
        std::mem::size_of::<Candidate>(), std::mem::size_of::<usize>());
    println!("SmallVec_u64_2={} SmallVec_u64_4={}",
        std::mem::size_of::<smallvec::SmallVec<[u64; 2]>>(),
        std::mem::size_of::<smallvec::SmallVec<[u64; 4]>>());
}
