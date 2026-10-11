"""Recreate frozen directed words; no tokenizer preprocessing is involved."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('/tmp/bpe-std-logic-results'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
weights=[0,1,7,13]
fixtures={
 'aa-neighbors':[(chr(0x4e00+i)+'a'*2048+chr(0x4e00+i),1+i%7) for i in range(2048)],
 'reuse-affix':[(str(i)+'aaaaaaaaabcd'*1024+'baaba',weights[i%4]) for i in range(512)],
 'reuse-find':[(str(i)+'xy'*8192+'z',weights[i%4]) for i in range(256)]+[('baaba',200)],
}
inputs=json.loads((Path(__file__).resolve().parent/'directed-inputs.json').read_text())
for item in inputs:
    kind=item['case']
    data={'schema_version':1,'pretokenizer':'none','ordered_words':sorted(fixtures[kind]),'hash_seeds':[11,13,17,19]}
    content=json.dumps(data,ensure_ascii=False).encode()
    assert hashlib.sha256(content).hexdigest()==item['prepared_sha256'],kind
    target=args.output/f'{kind}-words.json'
    target.write_bytes(content)
    item['path']=item['prepared_path']=str(target.resolve())
(args.output/'directed-inputs.json').write_text(json.dumps(inputs,indent=2))
