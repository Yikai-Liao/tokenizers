from pathlib import Path
import json,shutil
r=Path(__file__).resolve().parent;old=Path('/tmp/bpe-positions-codec-libraries-results-20261012');variants={x['name']:x for x in json.loads((r/'variants.json').read_text())};selected=['fmt-zero-push','fmt-tls-push','gimli-inline'];(r/'selected.json').write_text(json.dumps([variants[x] for x in selected],indent=2)+'\n')
s=(old/'build.py').read_text();start=s.index("arms={'hand'");end=s.index('\nfor name,source in arms.items():');s=s[:start]+'arms='+repr({a:variants[a]['worktree'] for a in selected})+'\narms={k:Path(v) for k,v in arms.items()}'+s[end:];(r/'build.py').write_text(s)
for n in ['hand-bin','hand-build.json']:shutil.copyfile(old/n,r/n)
(r/'hand-bin').chmod(0o755);shutil.copytree(old/'source/hand',r/'source/hand',dirs_exist_ok=True)
s=(old/'native-checks.py').read_text().replace('bpe-positions-codec-libraries-native-20261012','bpe-positions-codec-optimized-native-20261012');s=s.replace("json.loads((r/'variants.json').read_text())","[a for a in json.loads((r/'selected.json').read_text()) if a['name'].startswith('fmt')]")
(r/'native-checks.py').write_text(s)
