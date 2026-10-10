#![allow(dead_code)]
enum Builder {
    Narrow(smallvec::SmallVec<[u32; 2]>),
    Wide(smallvec::SmallVec<[u64; 2]>),
}
struct BaselineGroup {
    count: u64,
    positions: Builder,
    unordered: Vec<u64>,
}
struct Group {
    count: u64,
    positions: GroupPositions,
}

enum GroupPositions {
    Ordered(Builder),
    Unordered(Vec<u64>),
}
fn main(){println!("baseline={} candidate={} builder={}",std::mem::size_of::<BaselineGroup>(),std::mem::size_of::<Group>(),std::mem::size_of::<Builder>());}
