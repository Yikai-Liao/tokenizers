from pathlib import Path
import json,shutil
r=Path(__file__).resolve().parent;old=Path('/tmp/bpe-positions-codec-libraries-results-20261012');variants=json.loads((r/'variants.json').read_text());arms={'hand':Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012')}|{a['name'].replace('-','_'):Path(a['worktree']) for a in variants}
s=(old/'check-and-micro.py').read_text();start=s.index("arms={'hand'");end=s.index('\nfor label,source in arms.items()',start);s=s[:start]+'arms='+repr({k:str(v) for k,v in arms.items()})+'\narms={k:Path(v) for k,v in arms.items()}'+s[end:]
s=s.replace("old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012')","old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012')")
s=s.replace('positions-codec-libraries-probe','positions-codec-optimized-probe')
s=s.replace("c=r/'probe';","c=r/'probe';(c/'src').mkdir(parents=True,exist_ok=True);")
# Prepare only. Build and timed micro run after the original perf runs finish.
end=s.index("cmd=['cargo','test'");setup=s[:end];(r/'prepare-only.py').write_text(setup)
exec(compile(setup,str(r/'prepare-only.py'),'exec'),{'__file__':str(r/'prepare-only.py')})
(r/'check-and-micro.py').write_text(s)
allocation=(old/'allocation-proof.py').read_text().replace("['hand'", "['hand'")
# Generate an allocation-only binary with a counter around each candidate push/drop.
code=(old/'probe/src/bin/allocation-proof.rs').read_text();start=code.index('#[path=');end=code.index('use std::{hint',start)
code=code[:start]+'\n'.join(f'#[path="../{a}/positions.rs"] mod {a};' for a in arms)+'\n'+code[end:]
start=code.index('list!(hand);');end=code.index('use std::alloc',start);code=code[:start]+''.join(f'list!({a});' for a in arms)+'\n'+code[end:]
start=code.index('   let gimli=count');end=code.index('\n  }',start);code=code[:start]+''.join(f'   assert_eq!(hand,count::<{a}::Positions>(&values),"{a} len={{len}} kind={{kind}}");\n' for a in list(arms)[1:])+'   println!("kind={kind} len={len} all candidates alloc/realloc/free={hand:?}");'+code[end:]
p=r/'probe/src/bin/allocation-proof.rs';p.parent.mkdir(exist_ok=True);p.write_text(code)
