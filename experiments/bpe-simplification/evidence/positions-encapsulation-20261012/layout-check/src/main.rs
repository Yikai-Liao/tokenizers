#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path="base/positions.rs"] mod base;
#[path="a/positions.rs"] mod a;
fn main() {
    let mut comparisons = 0;
    for kind in 0..4 {
        let mut old = base::Positions::default();
        let mut new = a::Positions::default();
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
            let old = base::Positions::concat(values.chunks(width).map(|part| {
                let mut list=base::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            let new = a::Positions::concat(values.chunks(width).map(|part| {
                let mut list=a::Positions::default();for &v in part {list.push(v).unwrap();} list
            })).unwrap();
            for target in [0,1,127,128,129,511,usize::MAX] {
                let expected:Vec<Vec<u64>>=old.tasks(target).map(|task|task.iter().collect()).collect();
                let actual:Vec<Vec<u64>>=new.chunks(target).map(|chunk|chunk.collect()).collect();
                assert_eq!(actual,expected,"len={len}, width={width}, target={target}");
                partitions += 1;
            }
        }
    }
    println!("{comparisons} append states: identical encoded bytes, capacity and spill status");
    println!("{partitions} leaf/root partitions: identical chunk boundaries and contents");
}
