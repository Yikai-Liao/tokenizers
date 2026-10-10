"""Generate fixed core-runner inputs for AA and active-ID reuse cost checks."""
import json
from pathlib import Path
import hashlib

root = Path('/tmp/bpe-readability-directed-inputs')
root.mkdir(exist_ok=True)
cases = []
for name, words, suffix in [
    ('aa', [(('a' * 4097) + f'{i:05}', 1 + i % 17) for i in range(8192)], None),
    ('reuse', [(f'{i:05}' + 'aaaaaaaaabcd' * 32 + 'baaba', 1 + i % 17)
               for i in range(32768)], 'a'),
]:
    path = root / f'{name}.json'
    path.write_text(json.dumps(dict(schema_version=1, pretokenizer='Whitespace',
                                   hash_seeds=[11, 13, 17, 19],
                                   ordered_words=sorted(words))))
    cases.append(dict(language='zh', case=name, prepared_path=str(path),
                      prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      pretokenizer='Whitespace',
                      trainer=dict(vocab_size=4096, min_frequency=2, prefix=None,
                                   suffix=suffix, max_token_length=None)))
(root / 'inputs.json').write_text(json.dumps(cases, indent=2) + '\n')
print(root / 'inputs.json')
