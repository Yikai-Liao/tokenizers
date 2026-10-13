#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path = "/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012/tokenizers/tk-train/src/trainers/bpe/positions.rs"]
mod positions;
use positions::Positions;
use std::{mem::size_of, ops::Range};
type Pair = (u32, u32);

struct InitialEdge {
    pair: Pair,
    coordinate: u64,
    weight: u64,
}

struct BirthFragment {
    birth_group: usize,
    pair: Pair,
    weight: u64,
    positions: Positions,
}

#[derive(Clone, Copy)]
struct WeightRun {
    start: usize,
    weight: u64,
}

#[derive(Clone, Copy)]
struct WeightLookup {
    weight: u64,
    end: usize,
}

#[derive(Clone, Copy)]
struct AlphabetCharacter {
    character: char,
    observed_in_input: bool,
}

struct State<P> {
    count: u64,
    positions: P,
}

struct PairAggregate<P> {
    count: u64,
    positions: P,
}

fn main() {
    println!("initial_edge tuple={} named={}", size_of::<(Pair, u64, u64)>(), size_of::<InitialEdge>());
    println!("birth_fragment tuple={} named={}", size_of::<((usize, Pair), u64, Positions)>(), size_of::<BirthFragment>());
    println!("weight_run tuple={} named={}", size_of::<(usize, u64)>(), size_of::<WeightRun>());
    println!("weight_lookup tuple={} named={} option_tuple={} option_named={}", size_of::<(u64, usize)>(), size_of::<WeightLookup>(), size_of::<Option<(u64, usize)>>(), size_of::<Option<WeightLookup>>());
    println!("unit_range option_tuple={} option_range={}", size_of::<Option<(usize, usize)>>(), size_of::<Option<Range<usize>>>());
    println!("alphabet_character tuple={} named={}", size_of::<(char, bool)>(), size_of::<AlphabetCharacter>());
    println!("pair_aggregate state={} named={}", size_of::<State<Positions>>(), size_of::<PairAggregate<Positions>>());
}
