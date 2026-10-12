from pathlib import Path
import subprocess,shutil,os
r=Path(__file__).resolve().parent;w=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe/positions.rs')
c=r/'layout-check';(c/'src').mkdir(parents=True,exist_ok=True)
(c/'Cargo.toml').write_text('[package]\nname="positions-layout-check"\nversion="0.0.0"\nedition="2024"\n[dependencies]\nsmallvec={version="1.16",features=["union"]}\n')
helper='''
impl Positions {
    pub(super) fn layout(&self) -> (&[u8], usize, bool) {
        let Repr::Leaf(leaf) = &self.0 else { unreachable!() };
        (&leaf.bytes, leaf.bytes.capacity(), leaf.bytes.spilled())
    }
}
'''
for label in ['base','a']:
 p=c/'src'/label/'positions.rs';p.parent.mkdir(exist_ok=True)
 s=subprocess.check_output(['git','show','3436bba7:'+str(rel)],cwd=w,text=True) if label=='base' else (w/rel).read_text()
 p.write_text(s+helper)
 if label=='a':shutil.copytree(w/rel.parent/'positions',p.parent/'positions',dirs_exist_ok=True)
(c/'src/main.rs').write_text('''#![allow(dead_code)]
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
''')
env=os.environ|{'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0','PATH':'/root/.cargo/bin:'+os.environ['PATH']}
subprocess.run(['cargo','run','--release','--offline','--manifest-path',str(c/'Cargo.toml'),'--target-dir','/tmp/bpe-positions-encapsulation-layout-target-20261012'],env=env,check=True)
