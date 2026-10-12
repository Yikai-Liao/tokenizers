# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.2"]
# ///
from pathlib import Path
import subprocess,json
from tree_sitter import Language,Parser
import tree_sitter_rust
r=Path(__file__).resolve().parent;w=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012');b=Path('/tmp/bpe-positions-two-inline-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe');parser=Parser(Language(tree_sitter_rust.language()))
def cost(data):
 tree=parser.parse(data);assert not tree.root_node.has_error
 lines=data.splitlines();tests=set();comment_bytes=[];pending=[tree.root_node]
 while pending:
  node=pending.pop();children=node.named_children;pending.extend(children)
  if node.type in {'line_comment','block_comment'}:comment_bytes.append((node.start_byte,node.end_byte))
  for i,node in enumerate(children):
   if node.type=='attribute_item' and data[node.start_byte:node.end_byte].replace(b' ',b'')==b'#[cfg(test)]':
    tests.update(range(node.start_point.row,node.end_point.row+1))
    rest=next((n for n in children[i+1:] if n.type not in {'attribute_item','line_comment','block_comment'}),None)
    assert rest is not None
    tests.update(range(rest.start_point.row,rest.end_point.row+1))
 masked=bytearray(data)
 for start,end in comment_bytes:
  for i in range(start,end):
   if masked[i] not in (10,13):masked[i]=32
 no_comments=masked.splitlines();out={}
 for label,selected in [('production',set(range(len(lines)))-tests),('tests',tests)]:
  out[label]={'physical_lines':len(selected),'code_lines':sum(bool(no_comments[i].strip()) for i in selected)}
 return out
out={'method':'AST excludes cfg(test) items/statements/fields; code lines exclude blanks and Rust comments','variants':{}}
for label,path in [('baseline',None),('a',w),('ab',b)]:
 files={};total={k:{'physical_lines':0,'code_lines':0} for k in ['production','tests']}
 for f in ['positions.rs','positions/codec.rs','merge.rs']:
  if path is None:
   p=subprocess.run(['git','show','3436bba7:'+str(rel/f)],cwd=w,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
   if p.returncode:continue
   data=p.stdout
  else:data=(path/rel/f).read_bytes()
  counts=cost(data);files[f]=counts
  for kind in total:
   for metric in total[kind]:total[kind][metric]+=counts[kind][metric]
 out['variants'][label]={'files':files,'total':total}
(r/'production-costs.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
