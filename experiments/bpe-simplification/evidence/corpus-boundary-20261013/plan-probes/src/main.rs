#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path="/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012/tokenizers/tk-train/src/trainers/bpe/positions.rs"] mod positions;
use positions::Positions;
use std::{cmp::Reverse, mem::{size_of, size_of_val}};
type Pair=(u32,u32);
#[derive(Clone,Copy)] struct Priority {count:u64,pair:Reverse<Pair>}
struct Candidate {priority:Priority,positions:Positions}
struct OldRule {replacement:u32,candidate:Candidate}
struct SelectedMerge {pair:Pair,merged_id:u32,positions:Positions}
struct PendingAction {event_index:usize,remove_count:bool,has_birth:bool}
struct ResolvedMerge {merged_id:u32,already_in_vocabulary:bool}
#[derive(Clone,Copy)] enum TrainingMode {Fresh,Reuse}
#[derive(Clone,Copy)] enum Side {Left,Right}
fn main() {
    println!("rust_type_layout positions={} priority={} candidate={} old_rule={} selected_merge={} tuple_action={} named_action={} tuple_resolution={} named_resolution={} bool={} mode={} side={}", size_of::<Positions>(),size_of::<Priority>(),size_of::<Candidate>(),size_of::<OldRule>(),size_of::<SelectedMerge>(),size_of::<(usize,bool,bool)>(),size_of::<PendingAction>(),size_of::<(u32,bool)>(),size_of::<ResolvedMerge>(),size_of::<bool>(),size_of::<TrainingMode>(),size_of::<Side>());
    let mut positions=Positions::default();
    for value in 0..8194 {positions.push(value).unwrap();}
    type AaPositions=std::iter::Copied<std::slice::Iter<'static,u64>>;
    let old:Vec<_>=positions.chunks(4096).map(|source| (0usize,itertools::Either::<AaPositions,_>::Right(source))).collect();
    let ab:Vec<_>=positions.chunks(4096).map(|source| ((0u32,1u32),2u32,source)).collect();
    let rank:Vec<_>=positions.chunks(4096).map(|source| (0usize,source)).collect();
    let starts:Vec<u64>=(0..8194).step_by(2).collect();
    let aa:Vec<_>=starts.chunks(4096).map(|part| ((0u32,0u32),2u32,part.iter().copied())).collect();
    println!("actual_positions_task_layout old_rank_either={} ab_pair_id={} ab_rank={} aa_pair_id={} old_count={} ab_count={} aa_count={}",size_of_val(&old[0]),size_of_val(&ab[0]),size_of_val(&rank[0]),size_of_val(&aa[0]),old.len(),ab.len(),aa.len());
    let old_values:Vec<_>=old.into_iter().flat_map(|(_,source)|source).collect();
    let ab_values:Vec<_>=ab.into_iter().flat_map(|(_,_,source)|source).collect();
    assert_eq!(old_values,ab_values);
}
