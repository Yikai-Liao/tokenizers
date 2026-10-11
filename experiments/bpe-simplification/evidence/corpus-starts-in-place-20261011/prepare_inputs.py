"""Freeze four prepared inputs from the same pinned 256 MiB source corpora."""
import hashlib,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
root=Path('/tmp/bpe-corpus-b1-inputs-20261011')
records=[];cases={}
for lang in ['zh','en']:
    for pretokenizer in ['whitespace','bytelevel_regex']:
        folder=root/f'{lang}-{pretokenizer}'
        job=json.loads((folder/'job.json').read_text())
        if pretokenizer=='whitespace':
            prepared=Path(f'/root/code/tokenizers-bpe-benchmarks/.bench/owner/{lang}256-words/words.json')
        else:
            prepared=folder/'words.json'
            with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
                subprocess.run(['taskset','-c','0-3','/tmp/bpe-corpus-b1-baseline-bin-20261011',str(folder/'job.json')],stdout=out,stderr=err,check=True)
        source=Path(job['input'])
        with prepared.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        with source.open('rb') as f: source_sha=hashlib.file_digest(f,'sha256').hexdigest()
        records.append(dict(language=lang,pretokenizer=pretokenizer,source=str(source),source_sha256=source_sha,source_bytes=source.stat().st_size,prepared=str(prepared),prepared_sha256=sha,prepared_bytes=prepared.stat().st_size,preparation='existing fixed-seed lexical cache' if pretokenizer=='whitespace' else job))
        cases[f'{lang}-{pretokenizer}-256MiB']=dict(path=str(prepared),pretokenizer=pretokenizer,vocab_size=50000,min_frequency=2,suffix=None)
        print(records[-1],flush=True)
(here/'inputs.json').write_text(json.dumps(records,indent=2)+'\n')
(here/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
