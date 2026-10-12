from pathlib import Path
import subprocess,shutil,os,json
r=Path(__file__).resolve().parent; old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012'); variants=json.loads((r/'variants.json').read_text()); print(variants)
w=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe/positions.rs');c=r/'probe';(c/'src').mkdir(parents=True,exist_ok=True);
(c/'Cargo.toml').write_text('''[package]
name="positions-codec-optimized-probe"
version="0.0.0"
edition="2024"
[dependencies]
smallvec={version="=1.16.2",features=["union","write"]}
leb128="=0.2.7"
leb128fmt={version="=0.1.0",default-features=false}
[profile.release]
lto="fat"
codegen-units=1
''')
helper='''
impl Positions {
    pub(super) fn layout(&self) -> (&[u8], usize, bool) {
        let Repr::Leaf(leaf) = &self.0 else { unreachable!() };
        (&leaf.bytes, leaf.bytes.capacity(), leaf.bytes.spilled())
    }
    pub(super) fn storage(&self) -> (usize, usize, bool) {
        let (b,c,s)=self.layout(); (b.len(),c,s)
    }
}
'''
arms={'hand': '/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012', 'fmt_zero_push': '/tmp/bpe-positions-codec-fmt-zero-push-20261012', 'fmt_small': '/tmp/bpe-positions-codec-fmt-small-20261012'}
arms={k:Path(v) for k,v in arms.items()}
for label,source in arms.items():
 p=c/'src'/label/'positions.rs';p.parent.mkdir(exist_ok=True);p.write_text((source/rel).read_text()+helper);shutil.copytree(source/rel.parent/'positions',p.parent/'positions',dirs_exist_ok=True)
source=(old/'micro/src/main.rs').read_text(); source=source[:source.index('enum LegacySource')]+source[source.index('fn main() {'):];source=source.replace('    descriptor_sizes();','')
source=source.replace('#[path="base/positions.rs"] mod base;\n#[path="a/positions.rs"] mod a;\n#[path="b/positions.rs"] mod b;','\n'.join(f'#[path="{a}/positions.rs"] mod {a};' for a in arms))
source=source.replace('list!(base);list!(a);list!(b);',''.join(f'list!({a});' for a in arms))
source=source[:source.index('fn main() {')]+'''fn main() {
'''
checks=(old/'layout-check/src/main.rs').read_text();checks=checks[checks.index('    let mut comparisons'):checks.rindex('}')]
checks=checks.replace('old.tasks(target).map(|task|task.iter().collect())','old.chunks(target).map(|chunk|chunk.collect())').replace('base::','hand::')
for a in list(arms)[1:]:
 source+='    {\n'+checks.replace('a::',a+'::')+f'println!("{a} layout verified");\n'+'    }\n'
source+='''    for repeat in 0..3 {
        for len in [0,1,2,3,8,128,129,512] {
            let values:Vec<_>=(0..len).map(|n|n as u64/3).collect();
            if repeat%2==0 {
'''+''.join(f'run::<{a}::Positions>("{a}",&values,repeat);' for a in arms)+'''
            } else {
'''+''.join(f'run::<{a}::Positions>("{a}",&values,repeat);' for a in reversed(arms))+'''
            }
        }
    }
}
'''
(c/'src/main.rs').write_text(source)
env=os.environ|{'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0','PATH':'/root/.cargo/bin:'+os.environ['PATH']}
cmd=['cargo','test','--release','--offline','--manifest-path',str(c/'Cargo.toml'),'--target-dir',str(r/'probe-target')];subprocess.run(cmd,env=env,check=True)
subprocess.run(['cargo','build','--release','--offline','--manifest-path',str(c/'Cargo.toml'),'--target-dir',str(r/'probe-target')],env=env,check=True)
with (r/'micro.jsonl').open('w') as f:subprocess.run(['taskset','-c','0',str(r/'probe-target/release/positions-codec-optimized-probe')],stdout=f,check=True)
