from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parent;repo=Path('/root/code/tokenizers');rel=Path('tokenizers/tk-train/src/trainers/bpe/positions/codec.rs');meta=[]
formatter=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012/experiments/bpe-simplification/format-changed.py')
variants=[('fmt-push','6f83f217'),('fmt-extend','6f83f217'),('fmt-zero-push','55a44e7d'),('fmt-zero-extend','55a44e7d'),('gimli-inline','248acd22')]
for name,base in variants:
 w=Path('/tmp/bpe-positions-codec-'+name+'-20261012');assert not w.exists()
 subprocess.run(['git','worktree','add','--no-checkout','-b','experiment/bpe-positions-codec-'+name+'-20261012',str(w),base],cwd=repo,check=True)
 subprocess.run(['git','sparse-checkout','init','--cone'],cwd=w,check=True);subprocess.run(['git','sparse-checkout','set','tokenizers'],cwd=w,check=True);subprocess.run(['git','read-tree','-mu','HEAD'],cwd=w,check=True)
 p=w/rel;s=p.read_text();s=s.replace('#[inline]\npub(super) fn append_gap','#[inline(always)]\npub(super) fn append_gap');s=s.replace('    fn next(&mut self) -> Option<u64> {','    #[inline]\n    fn next(&mut self) -> Option<u64> {')
 if name=='fmt-push':s=s.replace('    bytes.extend_from_slice(&encoded[..width]);','    for &byte in &encoded[..width] {\n        bytes.push(byte);\n    }')
 elif name=='fmt-extend':s=s.replace('    bytes.extend_from_slice(&encoded[..width]);','    bytes.extend(encoded[..width].iter().copied());')
 elif name=='fmt-zero-push':s=s.replace('    bytes.resize(start + width, 0);','    for _ in 0..width {\n        bytes.push(0);\n    }')
 elif name=='fmt-zero-extend':s=s.replace('    bytes.resize(start + width, 0);','    bytes.extend(std::iter::repeat_n(0, width));')
 elif name=='gimli-inline':s=s.replace('    #[inline]\n    fn next','    #[inline(always)]\n    fn next')
 p.write_text(s)
 with (r/(name+'-format.log')).open('w') as f:subprocess.run(['uv','run','--offline',str(formatter),'--base',base],cwd=w,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['git','add',str(rel)],cwd=w,check=True);subprocess.run(['git','commit','-m','perf(bpe): test '+name+' codec integration'],cwd=w,check=True)
 meta.append(dict(name=name,worktree=str(w),base=base,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),upstream_dependencies_unmodified=True))
(r/'variants.json').write_text(json.dumps(meta,indent=2)+'\n')
