"""Count production Rust lines and diff after removing comments and test items.
Requires tree-sitter==0.25.2 and tree-sitter-rust==0.24.2.
"""
import json,re,subprocess,tempfile
from pathlib import Path
from tree_sitter import Language,Parser
import tree_sitter_rust
parser=Parser(Language(tree_sitter_rust.language()))
def effective(source):
    tree=parser.parse(source)
    if tree.root_node.has_error:
        raise RuntimeError('Rust parse failed')
    ranges=[]
    observations=[]
    def walk(node):
        if node.type in ('line_comment','block_comment'):
            ranges.append((node.start_byte,node.end_byte))
            return
        if node.type=='attribute_item':
            attr=re.sub(rb'\s+',b'',source[node.start_byte:node.end_byte])
            if attr in (b'#[cfg(test)]',b'#[test]') or attr.startswith(b'#[doc=') or attr.startswith(b'#[doc('):
                target=node.next_named_sibling
                if target is None: raise RuntimeError('No cfg target')
                while target.type=='attribute_item':
                    target=target.next_named_sibling
                    if target is None: raise RuntimeError('No attr target')
                end=target.end_byte
                # Parameters and arguments leave a punctuation sibling outside their node.
                sibling=target.next_sibling
                if sibling is not None and sibling.type==',': end=sibling.end_byte
                start=node.start_byte
                previous=node.prev_named_sibling
                while previous is not None and previous.type=='attribute_item':
                    start=previous.start_byte
                    previous=previous.prev_named_sibling
                ranges.append((start,end))
                observations.append({'line':node.start_point.row+1,'kind':target.type,'attr':attr.decode()})
        for child in node.children: walk(child)
    walk(tree.root_node)
    filtered=bytearray(source)
    for start,end in ranges:
        for i in range(start,end):
            if filtered[i] not in (10,13): filtered[i]=32
    # Preserve source indentation; remove blank/comment-only lines and trailing space.
    lines=[line.rstrip() for line in filtered.decode().splitlines() if line.strip()]
    return '\n'.join(lines)+'\n',observations

root=Path(__file__).resolve().parent
before=effective((root/'source/positions-enum.rs').read_bytes())[0]
after=effective((root/'source/positions-box.rs').read_bytes())[0]
with tempfile.TemporaryDirectory(prefix='bpe-box-loc-') as tmp:
    a=Path(tmp)/'enum.rs';b=Path(tmp)/'box.rs'
    a.write_text(before);b.write_text(after)
    result=subprocess.run(['git','diff','--no-index','--numstat',str(a),str(b)],capture_output=True,text=True)
    assert result.returncode in (0,1),result.stderr
    added,deleted=map(int,result.stdout.split('\t')[:2])
report=dict(before=len(before.splitlines()),after=len(after.splitlines()),added=added,deleted=deleted,net_removed=deleted-added,
    scope='positions.rs only; exclude test items, comments, doc comments and blank lines; preserve formatting')
(root/'code-count.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
