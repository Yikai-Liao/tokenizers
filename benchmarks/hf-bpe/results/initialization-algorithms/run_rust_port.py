#!/usr/bin/env python3
"""Bounded verification of the Rust port, separate from the original matrix."""
import hashlib, json, pathlib, subprocess, resource, time
ROOT = pathlib.Path(__file__).resolve().parent
BIN = ROOT / 'prototype/target/release/initialization-algorithm-prototype'
manifest = {'binary_sha256': hashlib.sha256(BIN.read_bytes()).hexdigest(),
            'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT/'prototype/src').glob('*.rs')}, 'runs': []}
for size in [1, 4, 16]:
    corpus = pathlib.Path(f'/tmp/tokenizers-bpe-bench/data/text/zh-{size}m.txt')
    stem = f'zh-{size}m-radsort-rust'
    command = [str(BIN), 'radsort-rust', str(corpus), '2', 'verify']
    before = resource.getrusage(resource.RUSAGE_CHILDREN); start = time.time()
    with (ROOT/f'{stem}.stdout').open('w') as out, (ROOT/f'{stem}.stderr').open('w') as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert result.returncode == 0
    row = json.loads((ROOT/f'{stem}.stdout').read_text())
    expected = json.loads((ROOT/f'zh-{size}m-radix.stdout').read_text())['checksum']
    assert row['checksum'] == expected
    row.update(command=command, input_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
               process_user_seconds=after.ru_utime-before.ru_utime,
               process_system_seconds=after.ru_stime-before.ru_stime,
               process_seconds_with_preprocessing_and_oracle=time.time()-start)
    manifest['runs'].append(row)
    (ROOT/'rust-port-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(row, flush=True)
