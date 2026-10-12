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
    return count, syntax(prod_root)

args = argparse.ArgumentParser()
args.add_argument('--base', default='b106736b')
args.add_argument('--output', required=True)
options = args.parse_args()
folder = Path('tokenizers/tk-train/src/trainers/bpe')
result = {'base': options.base, 'count_definition': 'tests.rs entirely plus all #[cfg(test)] items, fields and statements in ordinary BPE modules (excluding parity_trainer.rs and its word.rs); Rust AST comments removed, then nonblank lines counted', 'files': {}}
for path in sorted(folder.glob('*.rs')):
    before = subprocess.check_output(['git', 'show', f'{options.base}:{path}'])
    after = path.read_bytes()
    if path.name in ('parity_trainer.rs', 'word.rs'):
        assert before == after, 'Parity code changed'
        continue
    a, prod_a = inspect(before, path.name == 'tests.rs')
    b, prod_b = inspect(after, path.name == 'tests.rs')
    result['files'][path.name] = {'before': a, 'after': b, 'delta': b-a, 'production_syntax_equal': prod_a == prod_b}
miri_path = Path('experiments/bpe-simplification/miri-codec/src/lib.rs')
miri_before = subprocess.check_output(['git', 'show', f'{options.base}:{miri_path}'])
miri_after = miri_path.read_bytes() if miri_path.exists() else subprocess.check_output(['git', 'show', f'HEAD:{miri_path}'])
a, _ = inspect(miri_before, True)
b, _ = inspect(miri_after, True)
result['files']['miri-codec/src/lib.rs'] = {'before': a, 'after': b, 'delta': b-a, 'production_syntax_equal': miri_before == miri_after}
result['count_definition'] += '; all four Miri entry lines included'
result['before'] = sum(item['before'] for item in result['files'].values())
result['after'] = sum(item['after'] for item in result['files'].values())
result['delta'] = result['after'] - result['before']
result['production_syntax_equal'] = all(item['production_syntax_equal'] for item in result['files'].values())
Path(options.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
