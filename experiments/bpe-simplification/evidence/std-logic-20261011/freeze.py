from pathlib import Path
import hashlib,json,shutil,subprocess,sys
root=Path(__file__).resolve().parent
arm=sys.argv[1]
shutil.copy2('/tmp/bpe-interface-release/release/bpe-bench-runner',root/arm)
files=[*Path('tokenizers/tk-train/src/trainers/bpe').glob('*.rs'),Path('tokenizers/tk-train/src/trainers/bpe/AGENTS.md'),Path('tokenizers/tk-train/src/trainers/bpe/README.md')]
for p in files:
    target=root/'source'/arm/p.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
manifest=json.loads((root/'builds.json').read_text())
manifest['builds'][arm]={'binary':str(root/arm),'binary_sha256':hashlib.file_digest((root/arm).open('rb'),'sha256').hexdigest(),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(root/'builds.json').write_text(json.dumps(manifest,indent=2))
(root/f'{arm}.patch').write_bytes(subprocess.check_output(['git','diff','--','tokenizers/tk-train/src/trainers/bpe']))
print(arm,manifest['builds'][arm]['binary_sha256'])
