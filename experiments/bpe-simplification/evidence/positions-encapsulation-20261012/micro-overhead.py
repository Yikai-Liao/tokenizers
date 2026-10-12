from pathlib import Path
import subprocess,shutil,os
r=Path(__file__).resolve().parent;w=Path('/tmp/bpe-positions-overhead-20261012');b=Path('/tmp/bpe-positions-two-inline-overhead-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe/positions.rs');c=r/'micro-overhead';(c/'src').mkdir(parents=True,exist_ok=True)
(c/'Cargo.toml').write_text('[package]\nname="positions-encapsulation-micro"\nversion="0.0.0"\nedition="2024"\n[dependencies]\nsmallvec={version="=1.16.2",features=["union"]}\nitertools="0.14"\n[profile.release]\nlto="fat"\ncodegen-units=1\n')
helper='''
impl Positions {
    pub(super) fn storage(&self) -> (usize, usize, bool) {
        let Repr::Leaf(leaf) = &self.0 else { unreachable!() };
        (leaf.bytes.len(), leaf.bytes.capacity(), leaf.bytes.spilled())
    }
}
'''
for label,source in [('base',None),('a',w),('b',b)]:
 p=c/'src'/label/'positions.rs';p.parent.mkdir(exist_ok=True)
 s=subprocess.check_output(['git','show','3436bba7:'+str(rel)],cwd=w,text=True) if source is None else (source/rel).read_text()
 if label=='base':s=s.replace('fn tasks(&self, target_items:', 'fn segments(&self, target_items:')
 p.write_text(s+helper)
 if source:shutil.copytree(source/rel.parent/'positions',p.parent/'positions',dirs_exist_ok=True)
(c/'src/main.rs').write_text('''#![allow(dead_code)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path="base/positions.rs"] mod base;
#[path="a/positions.rs"] mod a;
#[path="b/positions.rs"] mod b;
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
list!(base);list!(a);list!(b);
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
    println!("{{\\"arm\\":\\"{arm}\\",\\"len\\":{},\\"repeat\\":{repeat},\\"encode_drop_ns\\":{encode},\\"scan_ns\\":{scan},\\"seek_ns\\":{seek},\\"bytes\\":{bytes},\\"capacity\\":{capacity},\\"spilled\\":{spilled},\\"capacity_changes\\":{allocations}}}",values.len());
}
fn main() {
    for repeat in 0..3 {
        for len in [0,1,2,3,8,128,129,512] {
            let values:Vec<_>=(0..len).map(|n|n as u64/3).collect();
            if repeat%2==0 { run::<base::Positions>("base",&values,repeat);run::<a::Positions>("a",&values,repeat);run::<b::Positions>("b",&values,repeat); }
            else { run::<b::Positions>("b",&values,repeat);run::<a::Positions>("a",&values,repeat);run::<base::Positions>("base",&values,repeat); }
        }
    }
}
''')
# No collecting intermediary Vecs in the segment scan: this micro measures only
# the storage readers. The normal training runs cover segment consumers.
p=c/'src/main.rs';s=p.read_text().replace('self.segments(128).flat_map(|s|s.iter().collect::<Vec<_>>()).fold(0u64,u64::wrapping_add)', 'self.segments(128).map(|s|s.iter().fold(0u64,u64::wrapping_add)).fold(0u64,u64::wrapping_add)');p.write_text(s)
p=c/'src/main.rs'
s=p.read_text()
layout=r'''
enum LegacySource<'a> { Slice(&'a [u64]), Task(base::ReadTask<'a>) }
fn descriptor_sizes() {
    eprintln!("legacy descriptor bytes {}",std::mem::size_of::<(usize,LegacySource<'_>)>());
    let values=[0u64];
    let mut a=a::Positions::default();a.push(0).unwrap();
    let mut tasks=Vec::new();tasks.push((0usize,itertools::Either::Left(values.iter().copied())));
    tasks.push((0,itertools::Either::Right(a.chunks(128).next().unwrap())));
    eprintln!("a descriptor bytes {}",std::mem::size_of_val(&tasks[0]));
    let mut b=b::Positions::default();b.push(0).unwrap();
    let mut tasks=Vec::new();tasks.push((0usize,itertools::Either::Left(values.iter().copied())));
    tasks.push((0,itertools::Either::Right(b.chunks(128).next().unwrap())));
    eprintln!("b descriptor bytes {}",std::mem::size_of_val(&tasks[0]));
}
'''
s=s.replace('fn main() {',layout+'\nfn main() {\n    descriptor_sizes();')
p.write_text(s)
env=os.environ|{'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0','PATH':'/root/.cargo/bin:'+os.environ['PATH']}
subprocess.run(['cargo','build','--release','--offline','--manifest-path',str(c/'Cargo.toml'),'--target-dir','/tmp/bpe-positions-encapsulation-layout-target-20261012'],env=env,check=True)
with (r/'micro-overhead.jsonl').open('w') as output:subprocess.run(['taskset','-c','0','/tmp/bpe-positions-encapsulation-layout-target-20261012/release/positions-encapsulation-micro'],stdout=output,check=True)
print('micro completed')
