# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.2"]
# ///
"""Count nonblank test lines after removing Rust comments; compare non-test syntax."""
import argparse, json, subprocess
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_rust

parser = Parser(Language(tree_sitter_rust.language()))

def inspect(source, whole_test=False):
    root = parser.parse(source).root_node
    assert not root.has_error
    test_ranges, comments = [], []
    def visit(node):
        if node.type in ('line_comment', 'block_comment'):
            comments.append((node.start_byte, node.end_byte))
            return
        children = node.named_children
        for i, child in enumerate(children):
            if child.type == 'attribute_item' and child.text.replace(b' ', b'') == b'#[cfg(test)]':
                if node.type in ('field_initializer', 'shorthand_field_initializer'):
                    end = node.end_byte
                    following = node.next_sibling
                    if following and following.type == ',':
                        end = following.end_byte
                    test_ranges.append((node.start_byte, end))
                    continue
                j = i + 1
                while j < len(children) and children[j].type in ('attribute_item', 'line_comment', 'block_comment'):
                    j += 1
                assert j < len(children), child.text
                end = children[j].end_byte
                following = children[j].next_sibling
                if following and following.type in (',', ';'):
                    end = following.end_byte
                test_ranges.append((child.start_byte, end))
        for child in children:
            visit(child)
    visit(root)
    test_mask = bytearray(len(source))
    if whole_test:
        test_mask[:] = b'\1' * len(source)
    else:
        for a, b in test_ranges:
            test_mask[a:b] = b'\1' * (b-a)
    cleaned = bytearray(source)
    for a, b in comments:
        for i in range(a,b):
            if cleaned[i] != 10:
                cleaned[i] = 32
    selected = bytes(ch if test_mask[i] or ch == 10 else 32 for i, ch in enumerate(cleaned))
    count = sum(bool(line.strip()) for line in selected.splitlines())
    production = bytes(ch if not test_mask[i] or ch == 10 else 32 for i, ch in enumerate(source))
    prod_root = parser.parse(production).root_node
    if prod_root.has_error:
        pending = [prod_root]
        while pending:
            node = pending.pop()
            if node.type == 'ERROR' or node.is_missing:
                print('Parse error:', node.type, node.start_point, node.text[:100])
            pending.extend(node.children)
        raise ValueError('Production masking needs to retain valid Rust syntax')
    def syntax(node):
        if node.type in ('line_comment', 'block_comment'):
            return []
        if not node.children:
            return [(node.type, node.text.decode())]
        return [item for child in node.children for item in syntax(child)]
    production_cleaned = bytes(ch if not test_mask[i] or ch == 10 else 32 for i, ch in enumerate(cleaned))
    return count, syntax(prod_root), [line.strip().decode() for line in production_cleaned.splitlines() if line.strip()]

import subprocess,json,difflib,argparse
cli=argparse.ArgumentParser();cli.add_argument('--base',default='890ea1f58e65f8d8503f7cd3fc97ad1651eaf284');args=cli.parse_args()
result={}
for name in subprocess.check_output(['git','diff','--name-only','HEAD'],text=True).splitlines():
 path=Path(name)
 if path.suffix!='.rs':continue
 before=subprocess.check_output(['git','show',f'{args.base}:{path}']);after=path.read_bytes()
 a,_,la=inspect(before,path.name=='tests.rs');b,_,lb=inspect(after,path.name=='tests.rs')
 added=removed=0
 for tag,i,j,k,l in difflib.SequenceMatcher(None,la,lb,autojunk=False).get_opcodes():
  if tag in ('replace','delete'):removed+=j-i
  if tag in ('replace','insert'):added+=l-k
 result[name]=dict(production_before=len(la),production_after=len(lb),production_added=added,production_removed=removed,production_delta=len(lb)-len(la),test_before=a,test_after=b)
print(json.dumps(result,indent=2))
