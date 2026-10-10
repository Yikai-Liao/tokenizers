from pathlib import Path
import hashlib, json, os, shutil, subprocess

ROOT = Path('/root/code/tokenizers-simplification-results')
OUT = ROOT/'online-initial'
WORK = Path('/root/code/tokenizers-workspaces/bpe-online-experiment-20261010')
BASE = Path('/root/code/tokenizers-workspaces/bpe-online-baseline-probe-20261010')
REL = Path('tokenizers/tk-train/src/trainers/bpe/engine')
shutil.copy2(WORK/REL/'initial_probe.rs', BASE/REL/'initial_probe.rs')
p = BASE/REL/'mod.rs'; p.write_text(p.read_text().replace('mod index;', 'mod index;\nmod initial_probe;'))
p = BASE/REL/'index.rs'; p.write_text(p.read_text().replace('        let work = progress.stage("Count initial pairs", corpus.word_count());',
    '        let _probe = super::initial_probe::start();\n        let work = progress.stage("Count initial pairs", corpus.word_count());', 1))
env = {k:v for k,v in os.environ.items() if not k.startswith(('BPE_','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS'))}
env['PATH'] = '/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin'
env['CARGO_TARGET_DIR'] = str(ROOT/'target')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def patch(work):
    result = subprocess.check_output(['git','diff','--binary'], cwd=work)
    files = subprocess.check_output(['git','ls-files','--others','--exclude-standard'], cwd=work).decode().splitlines()
    for file in files:
        r = subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',file], cwd=work, capture_output=True)
        assert r.returncode == 1
        result += r.stdout
    return result
manifest = dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK).decode().strip(),
    rustc=subprocess.check_output(['/root/.cargo/bin/rustc','--version']).decode().strip(),
    profile=dict(opt_level=3,lto='fat',codegen_units=1),
    features='runner defaults=[]; tk-train default-features=false',
    target_cpu='compiler default; no RUSTFLAGS',
    baseline_plain=dict(path=str(ROOT/'bin/review-final'),sha256=sha(ROOT/'bin/review-final')),
    block_slot_cap=1 << 24, wave_block_count=4, builds=[])
for arm, work in (('experimental',WORK),('baseline-probe',BASE)):
    subprocess.run(['/root/.cargo/bin/cargo','fmt','--manifest-path',str(work/'tokenizers/tk-train/Cargo.toml')],check=True)
    command=['cargo','build','--release','--locked','--offline','--manifest-path',
             str(work/'experiments/bpe-simplification/runner/Cargo.toml'),'-j','2']
    log=OUT/'validation'/f'build-{arm}.log'
    with log.open('w') as f: subprocess.run(command, env=env, stdout=f,stderr=subprocess.STDOUT,check=True)
    binary=OUT/'bin'/f'bpe-bench-{arm}'
    shutil.copy2(ROOT/'target/release/bpe-bench-runner',binary)
    data=patch(work); patch_path=OUT/f'{arm}.patch'; patch_path.write_bytes(data)
    sources={str(p.relative_to(work)):sha(p) for p in sorted((work/REL).rglob('*.rs'))}
    build=dict(arm=arm,worktree=str(work),command=command,binary=str(binary),binary_sha256=sha(binary),
        patch=str(patch_path),patch_sha256=sha(patch_path), sources_sha256=sources,
        runner_lock_sha256=sha(work/'experiments/bpe-simplification/runner/Cargo.lock'),
        runner_sources_sha256={str(p.relative_to(work)):sha(p) for p in sorted((work/'experiments/bpe-simplification/runner/src').rglob('*.rs'))})
    manifest['builds'].append(build)
    print('frozen',arm,build['binary_sha256'],flush=True)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
