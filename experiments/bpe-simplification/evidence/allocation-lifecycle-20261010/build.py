from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
work=Path('/root/code/tokenizers-workspaces/bpe-allocation-diagnostic')
variants=Path('/tmp/bpe-allocation-variants')
for arm in sys.argv[1:]:
    source=variants/arm
    for p in source.rglob('*'):
        if p.is_file(): shutil.copyfile(p,work/p.relative_to(source))
    env=os.environ.copy();env['PATH']='/root/.cargo/bin:'+env['PATH'];env['CARGO_INCREMENTAL']='0'
    subprocess.run(['cargo','fmt','--manifest-path','tokenizers/tk-train/Cargo.toml'],cwd=work,env=env,check=True)
    for label,command,target in [
        ('tests',['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--lib'],'/tmp/bpe-global-tests'),
        ('build',['cargo','build','--release','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml'],'/tmp/bpe-global-release-target')]:
        print(f'{arm}: {label}',flush=True);env['CARGO_TARGET_DIR']=target;start=time.monotonic()
        with open(f'/tmp/bpe-allocation-{arm}-{label}.log','w') as log:
            result=subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: print(f'{arm}: FAILED {label}',flush=True);sys.exit(result.returncode)
        print(f'{arm}: {label} passed in {time.monotonic()-start:.1f}s',flush=True)
    shutil.copy2('/tmp/bpe-global-release-target/release/bpe-bench-runner',f'/tmp/bpe-allocation-{arm}-bin')
    patch=subprocess.check_output(['git','diff'],cwd=work)
    (variants/f'{arm}.patch').write_bytes(patch)
    for p in source.rglob('*'):
        if p.is_file(): shutil.copyfile(work/p.relative_to(source),p)
    (variants/f'{arm}-lock.json').write_text(json.dumps({'patch_sha256':hashlib.sha256(patch).hexdigest(),'runner_lock':(work/'experiments/bpe-simplification/runner/Cargo.lock').read_text(),'train_lock':(work/'tokenizers/tk-train/Cargo.lock').read_text()},indent=2))
    print(f'{arm}: immutable binary ready',flush=True)
