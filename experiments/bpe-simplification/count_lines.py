"""Report BPE source size and Git differences without line-count limits."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def strip_comments(source):
    # Preserve strings while removing Rust line/block comments, including nesting.
    pattern = re.compile(r'(?P<string>r(?P<hash>\#*)".*?"(?P=hash)|"(?:\\.|[^"\\])*")|(?P<line>//[^\n]*)|(?P<block>/\*)', re.S)
    result = []
    cursor = 0
    while match := pattern.search(source, cursor):
        result.append(source[cursor:match.start()])
        if match.group('string'):
            result.append(match.group())
            cursor = match.end()
        elif match.group('line'):
            cursor = match.end()
        else:
            end = match.end()
            depth = 1
            while depth:
                token = re.search(r'/\*|\*/', source[end:])
                if token is None:
                    raise ValueError('Unclosed block comment')
                result.append('\n' * source[end:end + token.start()].count('\n'))
                end += token.end()
                depth += 1 if token.group() == '/*' else -1
            cursor = end
    result.append(source[cursor:])
    return ''.join(result)


def separate_test_items(source):
    # BPE cfg(test) items are complete attributes + declarations/blocks.
    # Braces inside strings are hidden while locating item boundaries.
    text = strip_comments(source)
    masked = re.sub(r'r(\#*)".*?"\1|"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m.group()), text, flags=re.S)
    test_ranges = []
    cursor = 0
    # The optional parity feature is off in the default BPE build; its
    # any(test, feature=...) imports therefore belong to the test count.
    marker = re.compile(r'#\[cfg\((?:test|any\(test,\s*feature\s*=\s+\))\)\]')
    while match := marker.search(masked, cursor):
        start, begin = match.span()
        following_line = masked[begin:].lstrip().splitlines()[0]
        if following_line.startswith('mut observe:') or following_line == 'None,':
            # This test-only parameter is one rustfmt line, not the train body.
            declaration = masked.index(following_line, begin)
            end = declaration + len(following_line)
            test_ranges.append((start, end))
            cursor = end
            continue
        brace = masked.find('{', begin)
        semi = masked.find(';', begin)
        if semi >= 0 and (brace < 0 or semi < brace):
            end = semi + 1
        else:
            depth = 1
            end = brace + 1
            while depth:
                depth += (masked[end] == '{') - (masked[end] == '}')
                end += 1
        test_ranges.append((start, end))
        cursor = end
    production = list(text)
    tests = []
    for start, end in test_ranges:
        tests.append(text[start:end])
        for index in range(start, end):
            if production[index] != '\n':
                production[index] = ' '
    count = lambda s: sum(bool(line.strip()) for line in s.splitlines())
    return count(''.join(production)), count('\n'.join(tests))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base', default='origin/main', help='Upstream Git revision for comparison')
    parser.add_argument('--change-base', help='Optional starting revision for this change')
    args = parser.parse_args()
    root = args.root / 'tokenizers/tk-train'
    bpe = root / 'src/trainers/bpe'
    # Check ordinary BPE and its tests; upstream sibling trainers and legacy
    # parity files retain their original formatting.
    format_paths = [p for p in bpe.rglob('*.rs') if p.name not in ('parity_trainer.rs', 'word.rs')]
    subprocess.run(['/root/.cargo/bin/rustfmt', '--edition', '2024', '--check',
                    '--config', 'skip_children=true',
                    *map(str, sorted(format_paths))], check=True)
    production = {}
    tests = {}
    # Include the public entry and feed/count representation alongside algorithm
    # components, so moving implementation between files cannot change its scope.
    paths = [p for p in bpe.rglob('*.rs') if p.name not in ('reference.rs', 'parity_trainer.rs', 'word.rs')]
    for path in sorted(paths):
        name = str(path.relative_to(bpe))
        if 'tests' in path.parts or path.name == 'tests.rs':
            tests[name] = sum(bool(line.strip()) for line in strip_comments(path.read_text()).splitlines())
        else:
            prod, test = separate_test_items(path.read_text())
            production[name] = prod
            if test:
                tests[name + ' [cfg(test)]'] = test
    for filename in ['reference.rs']:
        path = bpe / filename
        tests[filename + ' [oracle and shared helpers]'] = sum(bool(line.strip()) for line in strip_comments(path.read_text()).splitlines())
    # The shared trainer wrapper's sole default test exercises BpeTrainer.
    _, wrapper_tests = separate_test_items((bpe.parent / 'mod.rs').read_text())
    tests['trainers/mod.rs [cfg(test) BPE wrapper]'] = wrapper_tests
    for path in sorted((args.root / 'experiments/bpe-simplification/miri-codec/src').glob('*.rs')):
        tests[str(path.relative_to(args.root)) + ' [Miri harness]'] = sum(bool(line.strip()) for line in strip_comments(path.read_text()).splitlines())
    scopes = ['tokenizers/tk-train/src/trainers/bpe',
              'tokenizers/tk-train/src/trainers/wordpiece.rs',
              'tokenizers/tk-train/src/trainers/mod.rs',
              'experiments/bpe-simplification/miri-codec/src']
    diffs = {}
    for label, revision in [('upstream', args.base), ('change', args.change_base)]:
        if revision is None:
            continue
        commit = subprocess.check_output(['git', 'rev-parse', revision], cwd=args.root, text=True).strip()
        stat = subprocess.check_output(['git', 'diff', '--find-renames', '--stat', commit, '--', *scopes], cwd=args.root, text=True)
        numstat = subprocess.check_output(['git', 'diff', '--find-renames', '--numstat', commit, '--', *scopes], cwd=args.root, text=True)
        diffs[label] = dict(revision=revision, commit=commit, scopes=scopes, stat=stat, numstat=numstat)
    report = dict(production=production, tests=tests, production_total=sum(production.values()), test_total=sum(tests.values()),
                  counting_rule='Descriptive nonblank noncomment lines after rustfmt; ordinary BPE components, public API, feed and word counts. Optional parity implementation excluded. Test count includes embedded cfg(test), reference, shared trainer wrapper and Miri harness, each once. No hard limits.',
                  diffs=diffs,
                  diff_rule='Git text additions/deletions include comments, formatting and documentation; rename detection is enabled. Counts and diffs describe different scopes and are not complexity or correctness gates.')
    content = json.dumps(report, indent=2)
    print(content)
    if args.output:
        args.output.write_text(content)


if __name__ == '__main__':
    main()
