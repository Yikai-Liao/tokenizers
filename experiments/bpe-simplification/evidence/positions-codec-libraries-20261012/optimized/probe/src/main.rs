#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path="hand/positions.rs"] mod hand;
#[path="fmt_push/positions.rs"] mod fmt_push;
#[path="fmt_extend/positions.rs"] mod fmt_extend;
#[path="fmt_zero_push/positions.rs"] mod fmt_zero_push;
#[path="fmt_zero_extend/positions.rs"] mod fmt_zero_extend;
#[path="gimli_inline/positions.rs"] mod gimli_inline;
#[path="fmt_tls_push/positions.rs"] mod fmt_tls_push;
use std::{hint::black_box, time::Instant};
trait List: Default {
    fn push(&mut self,value:u64);
    fn sum(&self)->u64;
    fn seek_sum(&self,value:u64)->u64;
    fn storage(&self)->(usize,usize,bool);
}
macro_rules! list {
    ($module:ident) => {
        impl List for $module::Positions {
            #[inline(always)] fn push(&mut self,value:u64) { self.push(value).unwrap(); }
            #[inline(always)] fn sum(&self)->u64 { self.iter().fold(0u64,u64::wrapping_add) }
            #[inline(always)] fn seek_sum(&self,value:u64)->u64 { self.iter_from_value(value).fold(0u64,u64::wrapping_add) }
            fn storage(&self)->(usize,usize,bool) { self.storage() }
        }
    }
}
list!(hand);list!(fmt_push);list!(fmt_extend);list!(fmt_zero_push);list!(fmt_zero_extend);list!(gimli_inline);list!(fmt_tls_push);
fn run<P: List>(arm:&str,values:&[u64],repeat:u32) {
    let repetitions=(2_000_000/(values.len()+1)).clamp(4096,200_000);
    let now=Instant::now();
    for _ in 0..repetitions {
        let mut positions=P::default();
        for &value in black_box(values) { positions.push(value); }
        black_box(positions);
    }
    let encode=now.elapsed().as_nanos() as f64/repetitions as f64;
    let mut positions=P::default();let mut allocations=0;
    for &value in values {
        let before=positions.storage().1;positions.push(value);
        if positions.storage().1!=before { allocations+=1; }
    }
    let now=Instant::now();for _ in 0..repetitions { black_box(black_box(&positions).sum()); }
    let scan=now.elapsed().as_nanos() as f64/repetitions as f64;
    let target=values.last().copied().unwrap_or(0);
    let now=Instant::now();for _ in 0..repetitions { black_box(black_box(&positions).seek_sum(black_box(target))); }
    let seek=now.elapsed().as_nanos() as f64/repetitions as f64;
    let (bytes,capacity,spilled)=positions.storage();
    println!("{{\"arm\":\"{arm}\",\"len\":{},\"repeat\":{repeat},\"encode_drop_ns\":{encode},\"scan_ns\":{scan},\"seek_ns\":{seek},\"bytes\":{bytes},\"capacity\":{capacity},\"spilled\":{spilled},\"capacity_changes\":{allocations}}}",values.len());
}

fn main() {
    {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = hand::Positions::default();
        let mut new = fmt_push::Positions::default();
        assert_eq!(old.layout(), new.layout());
        for n in 0..8193u64 {
            let value = match kind {0 => 0, 1 => n/3, 2 => n*(u64::MAX/8193), _ => u64::MAX};
            old.push(value).unwrap();new.push(value).unwrap();
            assert_eq!(old.layout(), new.layout(), "kind={kind}, len={}", n+1);
            comparisons += 1;
        }
        assert_eq!(old.iter().collect::<Vec<_>>(), new.iter().collect::<Vec<_>>());
        for target in [0,1,127,128,u64::MAX] {
            assert_eq!(old.iter_from_value(target).collect::<Vec<_>>(),new.iter_from_value(target).collect::<Vec<_>>());
        }
    }
    let mut partitions = 0;
    for len in [0,1,2,3,127,128,129,255,256,257,2049,8193] {
        let values:Vec<u64>=(0..len).map(|n|n as u64/3).collect();
        for width in [1,2,3,127,128,129,255,256,257,511,2049,8193] {
            let old = hand::Positions::concat(values.chunks(width).map(|part| {
                let mut list=hand::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = fmt_push::Positions::concat(values.chunks(width).map(|part| {
                let mut list=fmt_push::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.chunks(target).map(|chunk|chunk.collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
println!("fmt_push layout verified");
    }
    {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = hand::Positions::default();
        let mut new = fmt_extend::Positions::default();
        assert_eq!(old.layout(), new.layout());
        for n in 0..8193u64 {
            let value = match kind {0 => 0, 1 => n/3, 2 => n*(u64::MAX/8193), _ => u64::MAX};
            old.push(value).unwrap();new.push(value).unwrap();
            assert_eq!(old.layout(), new.layout(), "kind={kind}, len={}", n+1);
            comparisons += 1;
        }
        assert_eq!(old.iter().collect::<Vec<_>>(), new.iter().collect::<Vec<_>>());
        for target in [0,1,127,128,u64::MAX] {
            assert_eq!(old.iter_from_value(target).collect::<Vec<_>>(),new.iter_from_value(target).collect::<Vec<_>>());
        }
    }
    let mut partitions = 0;
    for len in [0,1,2,3,127,128,129,255,256,257,2049,8193] {
        let values:Vec<u64>=(0..len).map(|n|n as u64/3).collect();
        for width in [1,2,3,127,128,129,255,256,257,511,2049,8193] {
            let old = hand::Positions::concat(values.chunks(width).map(|part| {
                let mut list=hand::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = fmt_extend::Positions::concat(values.chunks(width).map(|part| {
                let mut list=fmt_extend::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.chunks(target).map(|chunk|chunk.collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
println!("fmt_extend layout verified");
    }
    {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = hand::Positions::default();
        let mut new = fmt_zero_push::Positions::default();
        assert_eq!(old.layout(), new.layout());
        for n in 0..8193u64 {
            let value = match kind {0 => 0, 1 => n/3, 2 => n*(u64::MAX/8193), _ => u64::MAX};
            old.push(value).unwrap();new.push(value).unwrap();
            assert_eq!(old.layout(), new.layout(), "kind={kind}, len={}", n+1);
            comparisons += 1;
        }
        assert_eq!(old.iter().collect::<Vec<_>>(), new.iter().collect::<Vec<_>>());
        for target in [0,1,127,128,u64::MAX] {
            assert_eq!(old.iter_from_value(target).collect::<Vec<_>>(),new.iter_from_value(target).collect::<Vec<_>>());
        }
    }
    let mut partitions = 0;
    for len in [0,1,2,3,127,128,129,255,256,257,2049,8193] {
        let values:Vec<u64>=(0..len).map(|n|n as u64/3).collect();
        for width in [1,2,3,127,128,129,255,256,257,511,2049,8193] {
            let old = hand::Positions::concat(values.chunks(width).map(|part| {
                let mut list=hand::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = fmt_zero_push::Positions::concat(values.chunks(width).map(|part| {
                let mut list=fmt_zero_push::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.chunks(target).map(|chunk|chunk.collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
println!("fmt_zero_push layout verified");
    }
    {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = hand::Positions::default();
        let mut new = fmt_zero_extend::Positions::default();
        assert_eq!(old.layout(), new.layout());
        for n in 0..8193u64 {
            let value = match kind {0 => 0, 1 => n/3, 2 => n*(u64::MAX/8193), _ => u64::MAX};
            old.push(value).unwrap();new.push(value).unwrap();
            assert_eq!(old.layout(), new.layout(), "kind={kind}, len={}", n+1);
            comparisons += 1;
        }
        assert_eq!(old.iter().collect::<Vec<_>>(), new.iter().collect::<Vec<_>>());
        for target in [0,1,127,128,u64::MAX] {
            assert_eq!(old.iter_from_value(target).collect::<Vec<_>>(),new.iter_from_value(target).collect::<Vec<_>>());
        }
    }
    let mut partitions = 0;
    for len in [0,1,2,3,127,128,129,255,256,257,2049,8193] {
        let values:Vec<u64>=(0..len).map(|n|n as u64/3).collect();
        for width in [1,2,3,127,128,129,255,256,257,511,2049,8193] {
            let old = hand::Positions::concat(values.chunks(width).map(|part| {
                let mut list=hand::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = fmt_zero_extend::Positions::concat(values.chunks(width).map(|part| {
                let mut list=fmt_zero_extend::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.chunks(target).map(|chunk|chunk.collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
println!("fmt_zero_extend layout verified");
    }
    {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = hand::Positions::default();
        let mut new = gimli_inline::Positions::default();
        assert_eq!(old.layout(), new.layout());
        for n in 0..8193u64 {
            let value = match kind {0 => 0, 1 => n/3, 2 => n*(u64::MAX/8193), _ => u64::MAX};
            old.push(value).unwrap();new.push(value).unwrap();
            assert_eq!(old.layout(), new.layout(), "kind={kind}, len={}", n+1);
            comparisons += 1;
        }
        assert_eq!(old.iter().collect::<Vec<_>>(), new.iter().collect::<Vec<_>>());
        for target in [0,1,127,128,u64::MAX] {
            assert_eq!(old.iter_from_value(target).collect::<Vec<_>>(),new.iter_from_value(target).collect::<Vec<_>>());
        }
    }
    let mut partitions = 0;
    for len in [0,1,2,3,127,128,129,255,256,257,2049,8193] {
        let values:Vec<u64>=(0..len).map(|n|n as u64/3).collect();
        for width in [1,2,3,127,128,129,255,256,257,511,2049,8193] {
            let old = hand::Positions::concat(values.chunks(width).map(|part| {
                let mut list=hand::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = gimli_inline::Positions::concat(values.chunks(width).map(|part| {
                let mut list=gimli_inline::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.chunks(target).map(|chunk|chunk.collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
println!("gimli_inline layout verified");
    }
    {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = hand::Positions::default();
        let mut new = fmt_tls_push::Positions::default();
        assert_eq!(old.layout(), new.layout());
        for n in 0..8193u64 {
            let value = match kind {0 => 0, 1 => n/3, 2 => n*(u64::MAX/8193), _ => u64::MAX};
            old.push(value).unwrap();new.push(value).unwrap();
            assert_eq!(old.layout(), new.layout(), "kind={kind}, len={}", n+1);
            comparisons += 1;
        }
        assert_eq!(old.iter().collect::<Vec<_>>(), new.iter().collect::<Vec<_>>());
        for target in [0,1,127,128,u64::MAX] {
            assert_eq!(old.iter_from_value(target).collect::<Vec<_>>(),new.iter_from_value(target).collect::<Vec<_>>());
        }
    }
    let mut partitions = 0;
    for len in [0,1,2,3,127,128,129,255,256,257,2049,8193] {
        let values:Vec<u64>=(0..len).map(|n|n as u64/3).collect();
        for width in [1,2,3,127,128,129,255,256,257,511,2049,8193] {
            let old = hand::Positions::concat(values.chunks(width).map(|part| {
                let mut list=hand::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = fmt_tls_push::Positions::concat(values.chunks(width).map(|part| {
                let mut list=fmt_tls_push::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.chunks(target).map(|chunk|chunk.collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
println!("fmt_tls_push layout verified");
    }
    for repeat in 0..3 {
        for len in [0,1,2,3,8,128,129,512] {
            let values:Vec<_>=(0..len).map(|n|n as u64/3).collect();
            if repeat%2==0 {
run::<hand::Positions>("hand",&values,repeat);run::<fmt_push::Positions>("fmt_push",&values,repeat);run::<fmt_extend::Positions>("fmt_extend",&values,repeat);run::<fmt_zero_push::Positions>("fmt_zero_push",&values,repeat);run::<fmt_zero_extend::Positions>("fmt_zero_extend",&values,repeat);run::<gimli_inline::Positions>("gimli_inline",&values,repeat);run::<fmt_tls_push::Positions>("fmt_tls_push",&values,repeat);
            } else {
run::<fmt_tls_push::Positions>("fmt_tls_push",&values,repeat);run::<gimli_inline::Positions>("gimli_inline",&values,repeat);run::<fmt_zero_extend::Positions>("fmt_zero_extend",&values,repeat);run::<fmt_zero_push::Positions>("fmt_zero_push",&values,repeat);run::<fmt_extend::Positions>("fmt_extend",&values,repeat);run::<fmt_push::Positions>("fmt_push",&values,repeat);run::<hand::Positions>("hand",&values,repeat);
            }
        }
    }
}
