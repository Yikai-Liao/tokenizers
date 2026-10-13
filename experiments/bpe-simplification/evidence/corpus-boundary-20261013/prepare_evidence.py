"""Freeze inputs/resources and make separate production-grain diagnostic copies.

Run once before implementation; use --diagnostic ROOT DEST trace|stage to
instrument a copy. Formal timing binaries are built from the original ROOT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BPE = Path('tokenizers/tk-train/src/trainers/bpe')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(path, old, new):
    text = path.read_text()
    assert old in text, (path, old)
    path.write_text(text.replace(old, new, 1))


def diagnostic(root, dest, mode):
    dest.mkdir(parents=True, exist_ok=True)
    for directory in ['tokenizers/tk-train', 'tokenizers/tk-encode',
                      'tokenizers/tk-convert', 'tokenizers/tk-serialize', 'tokenizers/bitcannon', 'experiments/bpe-simplification/runner']:
        shutil.copytree(root / directory, dest / directory,
                        ignore=shutil.ignore_patterns('target', '.git'), dirs_exist_ok=True)
    source = dest / BPE
    merge = source / 'merge.rs'
    modern = (source / 'corpus/merging.rs').exists()
    if mode == 'trace':
        if modern:
            anchor = 'let candidate = index.take(priority);'
            event = '''eprintln!("BPE_DIAG {}", serde_json::json!({"kind":"select", "pair":pair,
                "count":format!("{:?}", priority), "merged_id":resolved.merged_id,
                "existing":resolved.already_in_vocabulary, "reuse":reuse}));
            '''
            restart = 'return Ok(MergeOutcome::RestartForReuse);'
            batch_anchor = 'let changes = corpus.merge(selected)?;'
            batch = 'selected.len()'
        else:
            anchor = 'let candidate = index.take(priority);'
            event = '''eprintln!("BPE_DIAG {}", serde_json::json!({"kind":"select", "pair":pair,
                "count":format!("{:?}", priority), "merged_id":replacement,
                "existing":reserved, "reuse":reuse}));
            '''
            restart = 'return Ok(MergeOutcome::RestartForReuse);'
            batch_anchor = 'let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;'
            batch = 'batch.rules.len()'
        replace(merge, anchor, event + anchor)
        replace(merge, restart, 'eprintln!("BPE_DIAG {}", serde_json::json!({"kind":"restart"}));\n                ' + restart)
        replace(merge, batch_anchor, f'''eprintln!("BPE_DIAG {{}}", serde_json::json!({{"kind":"batch", "rules":{batch}, "reuse":reuse}}));
        ''' + batch_anchor)
        commit = 'vocabulary.commit(merges);' if modern else 'vocabulary.commit_merges(records);'
        replace(merge, commit, commit + '\n        eprintln!("BPE_DIAG {}", serde_json::json!({"kind":"commit"}));')
        preparation = source / 'corpus/merging.rs' if modern else merge
        anchor = 'if !corpus.whole_words() {'
        replace(preparation, anchor, 'eprintln!("BPE_DIAG {}", serde_json::json!({"kind":"reuse_scan", "whole_words":corpus.whole_words()}));\n            ' + anchor)
    else:
        # Stage telemetry has no full trace buffer or per-coordinate counters.
        mod = source / 'mod.rs'
        with mod.open('a') as output:
            output.write('''
fn diagnostic_stage(stage: &str, begin: std::time::Instant, tasks: usize, starts: usize, events: usize) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let rss = status.lines().find(|s| s.starts_with("VmRSS:")).unwrap().to_owned();
    eprintln!("BPE_STAGE {}", serde_json::json!({"stage":stage, "seconds":begin.elapsed().as_secs_f64(),
        "tasks":tasks, "starts":starts, "events":events, "rss":rss}));
}
''')
        if modern:
            prep = source / 'corpus/merging.rs'
            replace(prep, 'let chunks = self.prepare(selected)?;', '''let begin = std::time::Instant::now();
        super::super::diagnostic_stage("before_prepare", begin, selected.len(), 0, 0);
        let chunks = self.prepare(selected)?;
        super::super::diagnostic_stage("prepare", begin, chunks.len(), chunks.iter().map(|c| c.starts.len()).sum(), chunks.iter().map(|c| c.changes.len()).sum());
        let begin = std::time::Instant::now();''')
            replace(prep, 'Ok(chunks', 'let changes = chunks')
            replace(prep, '.collect())', '''.collect();
        super::super::diagnostic_stage("apply", begin, 0, 0, 0);
        Ok(changes)''')
        else:
            replace(merge, 'let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;', '''let begin = std::time::Instant::now();
        super::diagnostic_stage("before_prepare", begin, batch.rules.len(), 0, 0);
        let jobs = batch.prepare(corpus, self.max_token_length.unwrap_or(usize::MAX))?;
        super::diagnostic_stage("prepare", begin, jobs.len(), jobs.iter().map(|c| c.starts.len()).sum(), jobs.iter().map(|c| c.changes.len()).sum());
        let begin = std::time::Instant::now();''')
            replace(merge, 'index.commit(changes)?;', 'super::diagnostic_stage("apply", begin, 0, 0, 0);\n        index.commit(changes)?;')
        replace(merge, 'index.commit(changes)?;', '''let begin = std::time::Instant::now();
        index.commit(changes)?;
        super::diagnostic_stage("commit", begin, 0, 0, 0);''')
        index = source / 'index.rs'
        replace(index, 'drop(shards);', '''super::diagnostic_stage("before_retire", std::time::Instant::now(), queue.len(), 0, 0);
        drop(shards);''')
    (dest / 'instrumented-source-manifest.json').write_text(json.dumps(
        {str(p.relative_to(dest)): sha(p) for p in source.rglob('*.rs')}, indent=2))


def freeze():
    fixtures = HERE / 'inputs'
    fixtures.mkdir(exist_ok=True)
    cases = {
        'aa': ([('a' * 32771, 3), ('ab' * 12000, 1), ('xy', 12000)], {}),
        'ab': ([('abacdbdc', 2), ('ab', 10), ('ac', 9), ('db', 8), ('dc', 7)], {}),
        'reserved': ([('abab', 10), ('ab', 3), ('cdcd', 7)], {'special_tokens':['ab']}),
        'alias': ([('baaba', 1)], {'suffix':'a'}),
        'length-reuse': ([('baaba', 1)], {'suffix':'a', 'max_token_length':9}),
    }
    manifest = {}
    for name, (words, config) in cases.items():
        path = fixtures / (name + '.json')
        path.write_text(json.dumps({'schema_version':1, 'pretokenizer':'none',
                                  'ordered_words':sorted(words), 'hash_seeds':[11,13,17,19]}))
        manifest[name] = {'input':str(path), 'sha256':sha(path), 'pretokenizer':'none',
                          'trainer':dict(vocab_size=64, min_frequency=1, prefix=None,
                                         suffix=None, max_token_length=None, **{}) | config}
    prepared = Path('/root/code/tokenizers-bpe-benchmarks/.bench/main-yttm-inspired/inputs')
    for name in ['en256', 'zh256']:
        path = prepared / (name + '-words/words.json')
        data = json.loads(path.read_text())
        # Share frozen prepared words; no preprocessing is charged to training.
        manifest[name] = {'input':str(path), 'sha256':sha(path), 'pretokenizer':data['pretokenizer'],
                          'trainer':{'vocab_size':50000, 'min_frequency':2, 'prefix':None,
                                     'suffix':None, 'max_token_length':None}}
    (HERE / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    environment = {'rustc':subprocess.check_output(['/root/.cargo/bin/rustc','-Vv'], text=True),
        'affinity':sorted(os.sched_getaffinity(0)), 'RUSTFLAGS':os.getenv('RUSTFLAGS',''),
        'baseline':'826479034460c18530ebe2522aaa1585133f97fe', 'profile':{'opt-level':3,'lto':'fat','codegen-units':1}}
    cgroup = Path('/sys/fs/cgroup/system.slice/paseo.service')
    environment['cgroup'] = {name:(cgroup / name).read_text().strip() for name in
                              ['cpu.max','cpuset.cpus.effective','memory.max','memory.swap.max']}
    environment['source_hashes'] = {str(p.relative_to(ROOT)):sha(p) for p in (ROOT / BPE).rglob('*') if p.is_file()}
    environment['locks'] = {str(p.relative_to(ROOT)):sha(p) for p in
                           [ROOT/'tokenizers/tk-train/Cargo.lock', ROOT/'experiments/bpe-simplification/runner/Cargo.lock']}
    (HERE / 'baseline-manifest.json').write_text(json.dumps(environment, indent=2))


def prepare_bytelevel(binary):
    manifest = json.loads((HERE / 'inputs.json').read_text())
    for language in ['en', 'zh']:
        original = Path('/root/code/tokenizers-bpe-benchmarks/.bench/main-yttm-inspired/inputs') / (language + '256/text.txt')
        text = HERE / 'inputs' / (language + '-bytelevel.txt')
        # Preserve complete UTF-8 lines from the same fixed corpus, capped at 8 MiB.
        with original.open('rb') as source, text.open('wb') as dest:
            size = 0
            for line in source:
                if size + len(line) > 8 * 1024 * 1024:
                    break
                dest.write(line)
                size += len(line)
        words = HERE / 'inputs' / (language + '-bytelevel.json')
        config = {'vocab_size':30000,'min_frequency':2,'prefix':None,'suffix':None,'max_token_length':None}
        job = dict(protocol_version=1, attempt_id=language+'-prepare', build_id='baseline-production',
                   input_id=language+'-bytelevel', mode='prepare', input=str(text), output=str(words),
                   workers=1, pretokenizer='bytelevel_regex', trainer=config)
        path = HERE / 'inputs' / (language + '-prepare-job.json')
        path.write_text(json.dumps(job, indent=2))
        result = subprocess.check_output([str(binary), str(path)], text=True)
        (HERE / 'inputs' / (language + '-preprocessing.json')).write_text(result)
        manifest[language+'-bytelevel'] = {'input':str(words), 'sha256':sha(words),
            'pretokenizer':'bytelevel_regex','trainer':config, 'source_text_sha256':sha(text)}
        # Timing runs need only the frozen words; source hash/length identify this prefix.
        text.unlink()
    (HERE / 'inputs.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic', nargs=3, metavar=('ROOT','DEST','MODE'))
    parser.add_argument('--prepare-bytelevel', type=Path)
    args = parser.parse_args()
    if args.diagnostic:
        root, dest, mode = args.diagnostic
        assert mode in ['trace','stage']
        diagnostic(Path(root), Path(dest), mode)
    elif args.prepare_bytelevel:
        prepare_bytelevel(args.prepare_bytelevel)
    else:
        freeze()
