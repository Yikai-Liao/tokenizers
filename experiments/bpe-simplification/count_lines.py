"""Count rustfmt source lines, including test oracle/helpers in the test budget."""
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
    # Engine cfg(test) items are complete attributes + declarations/blocks.
    # Braces inside strings are hidden while locating item boundaries.
    text = strip_comments(source)
    masked = re.sub(r'r(\#*)".*?"\1|"(?:\\.|[^"\\])*"', lambda m: ' ' * len(m.group()), text, flags=re.S)
    test_ranges = []
    cursor = 0
    marker = '#[cfg(test)]'
    while (start := masked.find(marker, cursor)) >= 0:
        begin = start + len(marker)
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
    parser.add_argument('--max-production', type=int, default=2100)
    parser.add_argument('--max-tests', type=int, default=800)
    args = parser.parse_args()
    root = args.root / 'tokenizers/tk-train'
    subprocess.run(['/root/.cargo/bin/cargo', 'fmt', '--manifest-path', str(root / 'Cargo.toml'), '--check'], check=True)
    bpe = root / 'src/trainers/bpe'
    engine = bpe / 'engine'
    production = {}
    tests = {}
    for path in sorted(engine.rglob('*.rs')):
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
    for filename in ['mod.rs', 'feed.rs', 'word_counts.rs']:
        _, public_tests = separate_test_items((bpe / filename).read_text())
        tests[filename + ' [cfg(test) public API/helpers]'] = public_tests
    for path in sorted((args.root / 'experiments/bpe-simplification/miri-codec/src').glob('*.rs')):
        tests[str(path.relative_to(args.root)) + ' [Miri harness]'] = sum(bool(line.strip()) for line in strip_comments(path.read_text()).splitlines())
    report = dict(production=production, tests=tests, production_total=sum(production.values()), test_total=sum(tests.values()),
                  counting_rule='Nonblank noncomment lines after cargo fmt --check; all engine implementation; all default BPE cfg(test) including feed/word_counts, independent reference and Miri harness. The oracle no longer uses the optional parity trainer Word. No implementation relocated outside engine.')
    report['production_limit'] = args.max_production
    report['test_limit'] = args.max_tests
    report['within_budget'] = report['production_total'] <= args.max_production and report['test_total'] <= args.max_tests
    content = json.dumps(report, indent=2)
    print(content)
    if args.output:
        args.output.write_text(content)
    if not report['within_budget']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
