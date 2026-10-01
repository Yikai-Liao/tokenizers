#!/usr/bin/env python3
"""One bounded, sequential prototype pass; never run alongside Trainer timings."""
import hashlib, json, pathlib, platform, subprocess, time, resource
ROOT = pathlib.Path(__file__).resolve().parent
PROTO = ROOT / 'prototype'
BIN = PROTO / 'target/release/initialization-algorithm-prototype'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'binary': str(BIN), 'binary_sha256': digest(BIN),
            'rustc': subprocess.check_output(['/root/.cargo/bin/rustc', '--version'], text=True).strip(),
            'cc': subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
            'host': platform.platform(), 'workers': 1, 'floor': 2,
            'ahash_seeds': [1, 2, 3, 4], 'word_order': 'first appearance',
            'scope': 'flat corpus initial index only; contiguous postings, no Trainer or merge',
            'sources': {str(p.relative_to(PROTO)): digest(p) for p in PROTO.rglob('*')
                        if p.is_file() and 'target' not in p.parts and '.git' not in p.parts}, 'runs': []}
for size in [1, 4, 16]:
    input_path = pathlib.Path(f'/tmp/tokenizers-bpe-bench/data/text/zh-{size}m.txt')
    checksums = []
    for mode in ['radix', 'radsort', 'hash2', 'rank3']:
        stem = f'zh-{size}m-{mode}'
        command = [str(BIN), mode, str(input_path), '2', 'verify']
        start = time.time()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with (ROOT / f'{stem}.stdout').open('w') as out, (ROOT / f'{stem}.stderr').open('w') as err:
            result = subprocess.run(command, stdout=out, stderr=err)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        (ROOT / f'{stem}.time').write_text(json.dumps({'process_user_seconds': after.ru_utime-before.ru_utime,
            'process_system_seconds': after.ru_stime-before.ru_stime,
            'process_minor_faults': after.ru_minflt-before.ru_minflt,
            'scope': 'entire process including corpus preprocessing and oracle; maxrss excluded'}, indent=2)+'\n')
        if result.returncode: raise RuntimeError(f'{command}: {result.returncode}')
        row = json.loads((ROOT / f'{stem}.stdout').read_text())
        checksums.append(row['checksum'])
        manifest['runs'].append({'input': str(input_path), 'input_sha256': digest(input_path),
                                 'input_bytes': input_path.stat().st_size, 'command': command,
                                 'process_seconds_with_preprocessing_and_oracle': time.time() - start, **row})
        (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(stem, row, flush=True)
    assert len(set(checksums)) == 1, checksums
