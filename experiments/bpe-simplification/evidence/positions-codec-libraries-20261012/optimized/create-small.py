from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parent;repo=Path('/root/code/tokenizers');name='fmt-small';base='20596e98';w=Path('/tmp/bpe-positions-codec-'+name+'-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe/positions/codec.rs');assert not w.exists()
subprocess.run(['git','worktree','add','--no-checkout','-b','experiment/bpe-positions-codec-'+name+'-20261012',str(w),base],cwd=repo,check=True)
subprocess.run(['git','sparse-checkout','init','--cone'],cwd=w,check=True);subprocess.run(['git','sparse-checkout','set','tokenizers'],cwd=w,check=True);subprocess.run(['git','read-tree','-mu','HEAD'],cwd=w,check=True)
p=w/rel;s=p.read_text();old='    let width = gap_width(gap);\n    debug_assert!(bytes.capacity() - bytes.len() >= width);\n    let start = bytes.len();';new='''    let width = gap_width(gap);
    debug_assert!(bytes.capacity() - bytes.len() >= width);
    if gap < 128 {
        bytes.push(leb128fmt::encode_u64(gap).expect("internal u64 gap").0[0]);
        return;
    }
    let start = bytes.len();''';assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
formatter=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012/experiments/bpe-simplification/format-changed.py')
with (r/(name+'-format.log')).open('w') as f:subprocess.run(['uv','run','--offline',str(formatter),'--base',base],cwd=w,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['git','add',str(rel)],cwd=w,check=True);subprocess.run(['git','commit','-m','perf(bpe): specialize library encoding for single-byte gaps'],cwd=w,check=True)
meta=json.loads((r/'variants.json').read_text());meta.append(dict(name=name,worktree=str(w),base=base,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),upstream_dependencies_unmodified=True,encoding_note='Both branches call published leb128fmt encoders; no handwritten LEB128 bit operations.'));(r/'variants.json').write_text(json.dumps(meta,indent=2)+'\n')
