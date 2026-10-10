"""Record hashes for the two benchmark arms after compiling them sequentially."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[3]
base='0416ae7a97bbc55f47f7969f4bdb04e8cfed1eff'
sha=lambda data:hashlib.sha256(data).hexdigest()
scope='tokenizers/tk-train/src/trainers/bpe'
paths=[p for p in subprocess.check_output(['git','ls-tree','-r','--name-only',base,'--',scope],cwd=repo,text=True).splitlines() if p.endswith('.rs') and not p.endswith('parity_trainer.rs')]
record=dict(baseline_commit=base,rustc=subprocess.check_output(['/root/.cargo/bin/rustc','-vV'],text=True),
    command=['cargo','build','--locked','--release','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml'],
    environment=dict(CARGO_INCREMENTAL='0',CARGO_TARGET_DIR='/tmp/bpe-box-release-target'),
    profile=dict(opt_level=3,lto='fat',codegen_units=1),
    runner_lock_sha256=sha((repo/'experiments/bpe-simplification/runner/Cargo.lock').read_bytes()),
    builds={})
for arm in ['enum','box']:
    binary=Path(f'/tmp/bpe-box-{arm}-bin')
    source={p:sha(subprocess.check_output(['git','show',f'{base}:{p}'],cwd=repo) if arm=='enum' else (repo/p).read_bytes()) for p in paths}
    record['builds'][arm]=dict(binary=str(binary),binary_sha256=sha(binary.read_bytes()),source_hashes=source)
(root/'builds.json').write_text(json.dumps(record,indent=2)+'\n')
Path('/tmp/bpe-box-builds.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({a:record['builds'][a]['binary_sha256'] for a in record['builds']}))
