from pathlib import Path
import hashlib, json
root=Path('/tmp/bpe-write-pressure-inputs')
root.mkdir(exist_ok=True)
cases=[]
for n in [1<<20,8<<20,32<<20]:
    for kind in ['ab','aa']:
        name=f'writes-{kind}-{n}'
        path=root/f'{name}.json'
        word='ab'*n if kind=='ab' else 'a'*(2*n)
        path.write_text(json.dumps(dict(schema_version=1,pretokenizer='whitespace',ordered_words=[(word,1)],hash_seeds=[11,13,17,19])))
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        cases.append(dict(case=name,prepared_path=str(path),prepared_sha256=h,pretokenizer='whitespace',vocab_size=3 if kind=='ab' else 2,min_frequency=1,n=n,kind=kind))
        print(name,path.stat().st_size,h,flush=True)
(root/'manifest.json').write_text(json.dumps(cases,indent=2))
