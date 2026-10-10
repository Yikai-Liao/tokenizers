from pathlib import Path
import subprocess,shutil,os,json
work=Path('/root/code/tokenizers-workspaces/bpe-heap-experiment');r=Path('/tmp/bpe-sparse-variants')
env=os.environ.copy();env['PATH']='/root/.cargo/bin:'+env['PATH']
sets={}
for arm in ['baseline','sparsley','xsparseset','bevy','cranelift','indexmap']:
    for p in (r/arm).rglob('*'):
        if p.is_file():shutil.copyfile(p,work/p.relative_to(r/arm))
    result=subprocess.check_output(['cargo','tree','--locked','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml','--edges','normal,build','--prefix','none','--format','{p}'],cwd=work,env=env,text=True)
    Path(f'/tmp/bpe-sparse-{arm}-tree.txt').write_text(result)
    sets[arm]=set(line.removesuffix(' (*)').strip() for line in result.splitlines())
for p in (r/'baseline').rglob('*'):
    if p.is_file():shutil.copyfile(p,work/p.relative_to(r/'baseline'))
rows=[{'arm':arm,'additional_normal_and_build_packages':sorted(values-sets['baseline']),'removed_packages':sorted(sets['baseline']-values)} for arm,values in sets.items()]
Path('/tmp/bpe-sparse-dependencies.json').write_text(json.dumps(rows,indent=2))
print([(r['arm'],len(r['additional_normal_and_build_packages'])) for r in rows])
