"""Recheck the flattening against its exact starting revision, without a build."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASE = '245ac0ffaff8927c36b9d97ee0326f5e96edec41'
BPE = 'tokenizers/tk-train/src/trainers/bpe/'


def before(path):
    return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT, text=True)


def after(path):
    return (ROOT / path).read_text()


old = before(BPE + 'engine/mod.rs')
new = after(BPE + 'mod.rs')
expected = old[old.index('pub(super) fn train('):old.index('\n#[cfg(test)]\nmod tests;')].replace('pub(super) fn train(', 'fn train(', 1).strip()
actual = new[new.index('\nfn train(') + 1:new.index('\nimpl Trainer for BpeTrainer {')].strip()
assert expected == actual
assert before(BPE + 'engine/positions.rs') == after(BPE + 'positions.rs')
for name, marker in [('corpus.rs', 'pub(super) struct Corpus {'), ('index.rs', '// Highest count first'), ('merge.rs', "struct Rule<'arena>")]:
    old, new = before(BPE + 'engine/' + name), after(BPE + name)
    assert old[old.index(marker):] == new[new.index(marker):], name
old = before(BPE + 'mod.rs')
selector = old[old.index('    fn select_alphabet('):old.index('    /// Train the collected weighted words')]
old_body = selector[selector.index('{') + 1:selector.rindex('}')].replace('self.', 'trainer.').replace('= self\n', '= trainer\n')
new = after(BPE + 'vocabulary.rs')
selector = new[new.index('pub(super) fn select_alphabet('):]
new_body = selector[selector.index('{') + 1:selector.rindex('}')]
assert re.sub(r'\s+', '', old_body) == re.sub(r'\s+', '', new_body)
public = old[old.index('struct Config {'):old.index('\nimpl Trainer for BpeTrainer {')]
a = public.index('    /// Select the alphabet with the existing frequency-tie and codepoint order.')
b = public.index('    /// Train the collected weighted words', a)
public = (public[:a] + public[b:]).replace('engine::train(', 'train(').strip()
new = after(BPE + 'mod.rs')
assert public == new[new.index('struct Config {'):new.index('\nfn add(')].strip()
assert old[old.index('\nimpl Trainer for BpeTrainer {'):] == new[new.index('\nimpl Trainer for BpeTrainer {'):]
checks = dict(coordinator_body_byte_equal=True, positions_byte_equal=True,
              corpus_index_merge_bodies_byte_equal=True,
              alphabet_body_equal_after_receiver_rename_and_whitespace=True,
              rejected_writes_and_birth_preserved=True,
              public_config_builder_trainer_and_entry_equal_after_selector_move=True,
              public_trainer_impl_byte_equal=True)
Path(__file__).with_name('source-checks.json').write_text(json.dumps(checks, indent=2) + '\n')

spec = importlib.util.spec_from_file_location('counts', ROOT / 'experiments/bpe-simplification/count_lines.py')
counts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counts)
production, tests = {}, {}
paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--', BPE], cwd=ROOT, text=True).splitlines()
for path in paths:
    p = Path(path)
    if p.suffix != '.rs' or p.name in ('reference.rs', 'parity_trainer.rs', 'word.rs'):
        continue
    name = path.removeprefix(BPE)
    source = before(path)
    if 'tests' in p.parts:
        tests[name] = sum(bool(line.strip()) for line in counts.strip_comments(source).splitlines())
    else:
        prod, test = counts.separate_test_items(source)
        production[name] = prod
        if test:
            tests[name + ' [cfg(test)]'] = test
tests['reference.rs [oracle and shared helpers]'] = sum(bool(line.strip()) for line in counts.strip_comments(before(BPE + 'reference.rs')).splitlines())
_, tests['trainers/mod.rs [cfg(test) BPE wrapper]'] = counts.separate_test_items(before('tokenizers/tk-train/src/trainers/mod.rs'))
path = 'experiments/bpe-simplification/miri-codec/src/lib.rs'
tests[path + ' [Miri harness]'] = sum(bool(line.strip()) for line in counts.strip_comments(before(path)).splitlines())
report = dict(revision=BASE, production=production, tests=tests,
              production_total=sum(production.values()), test_total=sum(tests.values()),
              method='Exact git show inputs to the same count_lines.py functions; no thresholds or build.')
Path(__file__).with_name('counts-before.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(checks=checks, before_production=report['production_total'], before_tests=report['test_total'])))
