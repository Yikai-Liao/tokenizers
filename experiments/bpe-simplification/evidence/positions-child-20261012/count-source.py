# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.2"]
# ///
from pathlib import Path
import subprocess, json, difflib
prefix=Path(__file__).with_name('count-and-audit.py').read_text().split('args = argparse.ArgumentParser()')[0]
prefix=prefix.replace('    return count, syntax(prod_root)','    production_cleaned = bytes(ch if not test_mask[i] or ch == 10 else 32 for i, ch in enumerate(cleaned))\n    return count, syntax(prod_root), [line.strip().decode() for line in production_cleaned.splitlines() if line.strip()]')
exec(prefix)
result={}
for name in ['positions','index','merge','mod']:
 path=Path('tokenizers/tk-train/src/trainers/bpe')/f'{name}.rs'
 before=subprocess.check_output(['git','show',f'02620e0a:{path}']);after=path.read_bytes()
 a,_,lines_a=inspect(before);b,_,lines_b=inspect(after)
 additions=removals=0
 for tag,i,j,k,l in difflib.SequenceMatcher(None,lines_a,lines_b,autojunk=False).get_opcodes():
  if tag in ('replace','delete'):removals+=j-i
  if tag in ('replace','insert'):additions+=l-k
 result[name]=dict(production_before=len(lines_a),production_after=len(lines_b),production_added=additions,production_removed=removals,test_before=a,test_after=b)
result['production_delta']=sum(item['production_after']-item['production_before'] for item in result.values())
Path(__file__).with_name('source-counts.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
