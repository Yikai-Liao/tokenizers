#!/usr/bin/env python3
"""Prepare a line-weighted efficient_bpe fixture for a rough native comparison."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=16 << 20)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    with args.source.open('rb') as source:
        for line in source:
            if len(raw) + len(line) > args.max_bytes:
                break
            raw.extend(line)
    text_path = args.directory / 'zh-prefix.txt'
    fixture_path = args.directory / 'zh-prefix.prepared.json'
    if text_path.exists() or fixture_path.exists():
        raise SystemExit('comparison inputs already exist')
    text_path.write_bytes(raw)
    # Keep line endings, like the HF benchmark's BufRead::read_line iterator.
    words = Counter(io.StringIO(raw.decode('utf-8'), newline='\n'))
    alphabet = sorted(set(''.join(words)))
    ids = {char: index for index, char in enumerate(alphabet, 1)}
    pivots, weights = [], []
    positions = 1
    with fixture_path.open('w') as stream:
        stream.write('{"corpus":[0')
        for word, weight in words.items():
            pivots.append(positions)
            weights.append(weight)
            values = [ids[char] for char in word]
            stream.write(',' + json.dumps(values, separators=(',', ':'))[1:-1] + ',0')
            positions += len(values) + 1
        stream.write('],"initial_lengths":')
        json.dump([1] * (len(alphabet) + 1), stream)
        stream.write(',"pivots":')
        json.dump(pivots, stream)
        stream.write(',"weights":')
        json.dump(weights, stream)
        stream.write('}\n')
    manifest = dict(source=str(args.source.resolve()), input_bytes=len(raw),
                    input_sha256=hashlib.sha256(raw).hexdigest(),
                    selection='complete-line prefix; no replication',
                    preparation='unique lines including line endings, occurrence weights',
                    unique_words=len(words), alphabet_size=len(alphabet),
                    symbols=sum(map(len, words)), corpus_positions=positions,
                    fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                    hf_vocab=50000, efficient_max_rules=50000-len(alphabet))
    (args.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
