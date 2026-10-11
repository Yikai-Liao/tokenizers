"""Build diagnostic copies; never modify the production worktree.

Checkpoints inspect RSS/HWM and allocated layout-vector capacities immediately
before and after the real serial prefix loop. Do not use these binaries for
production timing comparisons.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

here=Path(__file__).resolve().parent
checkout=Path('/tmp/bpe-corpus-b1-baseline-20261011')
target=checkout/'tokenizers/tk-train/src/trainers/bpe/corpus.rs'
helper='''
// Diagnostic only: this helper is absent from production binaries.
fn b1_layout_probe(stage: &str, words: usize, directory_bytes: usize) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let value = |key: &str| -> usize {
        status.lines().find(|line| line.starts_with(key)).unwrap()
            .split_whitespace().nth(1).unwrap().parse().unwrap()
    };
    eprintln!("B1_LAYOUT {} {} {} {} {}", stage, words, directory_bytes,
        value("VmRSS:"), value("VmHWM:"));
}
'''
manifest={}
for arm in ['baseline','candidate']:
    source=(here/f'source/{arm}-corpus.rs').read_text()
    if arm=='baseline':
        source=source.replace('        let mut starts = Vec::with_capacity(words.len());',
            '        b1_layout_probe("measured", words.len(), measured.capacity() * std::mem::size_of::<usize>());\n        let mut starts = Vec::with_capacity(words.len());',1)
        directory='(measured.capacity() + starts.capacity()) * std::mem::size_of::<usize>()'
    else:
        source=source.replace('        let mut weight_regions = Vec::new();',
            '        b1_layout_probe("measured", words.len(), starts.capacity() * std::mem::size_of::<usize>());\n        let mut weight_regions = Vec::new();',1)
        directory='starts.capacity() * std::mem::size_of::<usize>()'
    source=source.replace('        if length > isize::MAX',
        f'        b1_layout_probe("prefixed", words.len(), {directory});\n        if length > isize::MAX',1)+helper
    target.write_text(source)
    (here/f'source/probe-{arm}-corpus.rs').write_text(source)
    cmd=['cargo','build','--offline','--locked','--release','--manifest-path',str(checkout/'experiments/bpe-simplification/runner/Cargo.toml')]
    env={**os.environ,'CARGO_TARGET_DIR':'/tmp/bpe-index-build-20261011'}
    with (here/f'probe-{arm}-build.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=Path(f'/tmp/bpe-corpus-b1-probe-{arm}-bin-20261011')
    shutil.copy('/tmp/bpe-index-build-20261011/release/bpe-bench-runner',binary)
    manifest[arm]={'command':cmd,'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    print(arm,flush=True)
subprocess.run(['git','restore','--',str(target.relative_to(checkout))],cwd=checkout,check=True)
(here/'probe-builds.json').write_text(json.dumps(manifest,indent=2)+'\n')
