#![allow(dead_code)]
// Storage declarations copied from enum baseline and box experiment.
enum EnumPositions { Empty, One(u64), Two(u64, u64), Compressed(Box<[u8]>) }
struct BoxPositions(Box<[u8]>);
struct EnumCandidate { count: u64, pair: (u32,u32), positions: EnumPositions }
struct BoxCandidate { count: u64, pair: (u32,u32), positions: BoxPositions }
fn main() {
    println!("enum={} box={} usize={}", std::mem::size_of::<EnumPositions>(),
        std::mem::size_of::<BoxPositions>(), std::mem::size_of::<usize>());
    println!("candidate_enum={} candidate_box={}", std::mem::size_of::<EnumCandidate>(), std::mem::size_of::<BoxCandidate>());
}
