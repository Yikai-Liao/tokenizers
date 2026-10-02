#!/usr/bin/env python3
"""Build an instrumented, isolated Tokenizers runner with immutable provenance."""
import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def source_hashes(tree):
    paths = subprocess.check_output(['git', '-C', str(tree), 'ls-files', '-z']).decode().split('\0')
    result = {}
    for rel in paths:
        if not rel:
            continue
        path = tree / rel
        if path.is_file():
            result[rel] = sha(path)
    return result

def install_cohort_switches(source):
    """Add benchmark-only switches from the frozen source's complete Options schema."""
    indexed = source/'tk-train/src/trainers/bpe/indexed.rs'
    module = source/'tk-train/src/trainers/bpe/indexed/cohort_parallel.rs'
    if not module.is_file():
        return [], {}
    t = module.read_text()
    match = re.search(r'pub(?:\(super\))? struct Options\s*\{(.*?)\n\}', t, re.S)
    if not match:
        return [], {}
    fields = re.findall(r'pub(?:\(super\))?\s+(\w+)\s*:\s*bool\s*,', match.group(1))
    if not fields:
        raise SystemExit('cohort Options struct found, but it has no public boolean fields')
    indexed_text = indexed.read_text()
    module_text = module.read_text()
    if 'fn from_bench_env()' in module_text:
        raise SystemExit('frozen source already contains from_bench_env; refuse ambiguous instrumentation')
    dispatch = '''        if supports_compact(self) {
            parallel::train(self, word_counts, config)
        } else {
            let options = cohort_parallel::Options::default();'''
    if indexed_text.count(dispatch) == 1:
        indexed_text = indexed_text.replace(dispatch, dispatch.replace('Options::default()', 'Options::from_bench_env()'))
    else:
        # Earlier cohort implementations constructed Options inside this helper
        # instead of in do_train_indexed_parallel's dispatch arm.
        start_anchor = '    fn train_cohorts_parallel('
        end_anchor = '    fn train_cohorts_parallel_options('
        if indexed_text.count(start_anchor) != 1 or indexed_text.count(end_anchor) != 1:
            raise SystemExit('could not find unique general-cohort dispatch for benchmark options')
        start = indexed_text.index(start_anchor)
        end = indexed_text.index(end_anchor, start)
        body = indexed_text[start:end]
        old = 'cohort_parallel::Options::default()'
        if body.count(old) != 1:
            raise SystemExit('legacy cohort helper has no unique Options default construction')
        indexed_text = indexed_text[:start] + body.replace(old, 'cohort_parallel::Options::from_bench_env()') + indexed_text[end:]
    indexed.write_text(indexed_text)
    defaults_match = re.search(r'impl\s+Default\s+for\s+Options\s*\{.*?Self\s*\{(.*?)\n\s*\}', module_text, re.S)
    defaults = {}
    if defaults_match:
        defaults = {name: value == 'true' for name, value in re.findall(
            r'(\w+)\s*:\s*(true|false)\s*,', defaults_match.group(1)) if name in fields}
    fields_disabled = ',\n            '.join(f'{name}: false' for name in fields)
    impl = '''

impl Options {
    pub(super) fn from_bench_env() -> Self {
        let mut options = Self::default();
        let disabled = std::env::var("HF_BPE_COHORT_DISABLE").unwrap_or_default();
        for name in disabled.split(',').map(str::trim).filter(|name| !name.is_empty()) {
            match name {
                "all" => return Self { FIELDS },
                __MATCHES__
                other => panic!("unknown HF_BPE_COHORT_DISABLE option: {other}"),
            }
        }
        options
    }
}
'''.replace('FIELDS', fields_disabled)
    arms = '\n                '.join(f'"{name}" => options.{name} = false,' for name in fields)
    impl = impl.replace('__MATCHES__', arms)
    module.write_text(module_text.rstrip()+impl+'\n')
    return fields, defaults

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('worktree', type=Path)
    p.add_argument('--label', required=True)
    p.add_argument('--posting-block-bits', type=int, choices=(16, 32), default=32,
                   help='benchmark-only fixed posting layout; production source stays unchanged')
    p.add_argument('--shared-target', type=Path,
                   help='reuse Cargo dependencies through a target symlink; runner labels stay unique')
    a = p.parse_args()
    wt = a.worktree.resolve()
    dirty = subprocess.check_output(['git', '-C', str(wt), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise SystemExit(f'worktree must be clean: {wt}')
    commit = subprocess.check_output(['git', '-C', str(wt), 'rev-parse', 'HEAD'], text=True).strip()
    build = ROOT / '.build' / f'native-{a.label}'
    if build.exists():
        raise SystemExit(f'refusing to overwrite existing build: {build}')
    build.mkdir(parents=True)
    if a.shared_target is not None:
        shared_target = a.shared_target.resolve(strict=True)
        (build / 'target').symlink_to(shared_target, target_is_directory=True)
    source = build / 'source/tokenizers'
    shutil.copytree(wt / 'tokenizers', source, ignore=shutil.ignore_patterns('target', '.git', 'data'))

    trainer = source / 'tk-train/src/trainers/bpe/mod.rs'
    t = trainer.read_text()
    if a.posting_block_bits != 32:
        anchor = '                posting_block_bits: 32,'
        if t.count(anchor) != 1:
            raise SystemExit('could not find unique public posting layout for benchmark control')
        t = t.replace(anchor, f'                posting_block_bits: {a.posting_block_bits},')
    anchor = '        let trained = self.do_train_indexed_parallel('
    assert t.count(anchor) == 1
    t = t.replace(anchor, '''        if std::env::var("HF_BPE_ORACLE").as_deref() == Ok("1") {
            return self.do_train_observed(word_counts, |_, _, _| {});
        }
        let trained = self.do_train_indexed_parallel(''')
    anchor = '        Ok((trained.vocab, trained.merges, trained.special_tokens))'
    assert t.count(anchor) == 1
    t = t.replace(anchor, '        eprintln!("{}", serde_json::json!({"bench_indexed_stats": trained.stats}));\n' + anchor)
    start = t.index('    fn feed<I, S, F>')
    end = t.index('\n#[cfg(test)]', start)
    feed = t[start:end]
    feed = feed.replace('AHashMap::new()', 'AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19))')
    t = t[:start] + feed + t[end:]
    trainer.write_text(t)

    indexed = source / 'tk-train/src/trainers/bpe/indexed.rs'
    t = indexed.read_text()
    # Newer candidate sources expose the cohort scan counters in production.
    # Preserve the legacy instrumentation path for older frozen baselines.
    if 'pub cohort_scan_activations: usize,' not in t:
        anchor = '    pub word_scan_steps: usize,'
        assert t.count(anchor) == 1
        t = t.replace(anchor, anchor + '\n    pub cohort_scan_activations: usize,\n    pub cohort_words_scanned: usize,\n    pub cohort_scan_max_words: usize,')
        anchor = '            let mut words: Vec<u32> = top'
        assert t.count(anchor) == 1
        t = t.replace(anchor, '            stats.cohort_scan_activations += 1;\n' + anchor)
        anchor = '            words.sort_unstable();'
        assert t.count(anchor) == 1
        t = t.replace(anchor, '            stats.cohort_words_scanned += words.len();\n            stats.cohort_scan_max_words = stats.cohort_scan_max_words.max(words.len());\n' + anchor)
        indexed.write_text(t)

    cohort_switch_names, cohort_option_defaults = install_cohort_switches(source)

    stats_file = source/'tk-train/src/trainers/bpe/indexed.rs'
    stats_text = stats_file.read_text()
    stats_match = re.search(r'pub\(super\) struct IndexedTrainingStats\s*\{(.*?)\n\}', stats_text, re.S)
    stats_fields = re.findall(r'pub\s+(\w+)\s*:', stats_match.group(1)) if stats_match else []
    extended_diagnostics = {
        'alias_guarded', 'alias_fallback', 'corpus_slot_bytes',
        'speculative_selected_merges', 'speculative_applied_merges',
        'speculative_initialize_ms', 'speculative_merge_ms', 'speculative_total_ms',
        'speculative_posting_allocations',
    }.issubset(stats_fields)

    runner = build / 'runner'
    (runner / 'src').mkdir(parents=True)
    (runner / 'src/main.rs').write_text(r'''use serde_json::json;
use sha2::{Digest, Sha256};
use std::{env, fs::File, io::{BufRead, BufReader}, time::Instant};
use tk_train::{BpeTrainer, Trainer};
struct Lines<R>(R);
impl<R: BufRead> Iterator for Lines<R> { type Item=String; fn next(&mut self)->Option<String>{let mut s=String::new(); match self.0.read_line(&mut s){Ok(0)=>None,Ok(_)=>Some(s),Err(e)=>panic!("read: {e}")}} }
fn main()->Result<(),Box<dyn std::error::Error+Send+Sync>>{
 let args:Vec<String>=env::args().collect(); assert!(args.len()==6);
 let (input,split,backend)=(&args[1],&args[2],&args[3]); assert_eq!(split,"none");
 let mut b=BpeTrainer::builder().show_progress(false).vocab_size(args[4].parse()?).min_frequency(args[5].parse()?);
 if let Ok(x)=env::var("HF_BPE_MAX_TOKEN_LENGTH"){b=b.max_token_length(Some(x.parse()?))}
 if let Ok(x)=env::var("HF_BPE_PREFIX"){b=b.continuing_subword_prefix(x)}
 if let Ok(x)=env::var("HF_BPE_SUFFIX"){b=b.end_of_word_suffix(x)}
 let mut trainer=b.build(); tk_encode::parallelism::set_num_threads(env::var("HF_BPE_BENCH_WORKERS").unwrap_or_else(|_|"1".into()).parse()?); tk_encode::parallelism::set_parallelism(false); let begin=Instant::now(); trainer.feed(Lines(BufReader::new(File::open(input)?)),|s|Ok(vec![s.to_owned()]))?;
 let feed_ms=begin.elapsed().as_secs_f64()*1000.0; tk_encode::parallelism::set_parallelism(true);
 let (vocab,merges,stats)=match backend.as_str(){"reference"=>{let(v,m,_)=trainer.train_vocab()?;(v,m,None::<serde_json::Value>)},_=>panic!("bad backend")};
 let elapsed_ms=begin.elapsed().as_secs_f64()*1000.0; let mut digest=Sha256::new(); let mut entries:Vec<_>=vocab.iter().collect(); entries.sort_by_key(|(_,id)|**id);
 for(s,id)in entries{digest.update(id.to_le_bytes());digest.update((s.len()as u32).to_le_bytes());digest.update(s.as_bytes())}
 for(a,b)in &merges{digest.update(2_u32.to_le_bytes());for s in[a,b]{digest.update((s.len()as u32).to_le_bytes());digest.update(s.as_bytes())}}
 let hwm=std::fs::read_to_string("/proc/self/status")?.lines().find(|s|s.starts_with("VmHWM:")).unwrap().split_whitespace().nth(1).unwrap().parse::<u64>()?;
 println!("{}",json!({"backend":backend,"input":input,"input_bytes":std::fs::metadata(input)?.len(),"split":split,"vocab_size":trainer.vocab_size,"min_frequency":trainer.min_frequency,"elapsed_ms":elapsed_ms,"feed_ms":feed_ms,"train_ms":elapsed_ms-feed_ms,"unique_words":trainer.get_word_count(),"actual_vocab":vocab.len(),"actual_merges":merges.len(),"maxrss_kib":hwm,"model_sha256":format!("{:x}",digest.finalize()),"indexed_stats":stats})); Ok(())
}
''')
    shutil.copyfile(ROOT / 'Cargo.lock', runner / 'Cargo.lock')
    (runner / 'Cargo.toml').write_text(f'''[package]\nname = "hf-bpe-native-{a.label}"\nversion = "0.1.0"\nedition = "2024"\n[features]\nindexed = []\n[dependencies]\ntk-train = {{ path = {json.dumps(str(source/'tk-train'))}, default-features = false }}\ntk-encode = {{ path = {json.dumps(str(source/'tk-encode'))}, default-features = false }}\nserde_json = "1"\nsha2 = "0.10"\nhf-front-end = {{ package = "tokenizers", version = "=0.23.2", default-features = false, features = ["fancy-regex"] }}\n''')

    diffs = []
    for f in source.rglob('*.rs'):
        original = wt / 'tokenizers' / f.relative_to(source)
        if original.exists():
            diffs.extend(difflib.unified_diff(original.read_text().splitlines(True), f.read_text().splitlines(True), fromfile=str(original), tofile=str(f)))
    patch_path = build / 'instrumentation.patch'
    patch_path.write_text(''.join(diffs))
    command = [str(Path.home()/'.cargo/bin/cargo'), 'build', '--offline', '--release', '--manifest-path', str(runner/'Cargo.toml')]
    env = dict(os.environ, CARGO_TARGET_DIR=str(build/'target'))
    for key in ('HF_BPE_PREFIX', 'HF_BPE_SUFFIX', 'HF_BPE_ORACLE'):
        env.pop(key, None)
    manifest = {
        'label': a.label, 'worktree': str(wt), 'worktree_commit': commit, 'worktree_clean': True,
        'source_tree_sha256': source_hashes(wt), 'build_script_sha256': sha(Path(__file__)),
        'runner_source_sha256': sha(runner/'src/main.rs'), 'runner_cargo_sha256': sha(runner/'Cargo.toml'),
        'runner_lock_sha256': sha(runner/'Cargo.lock'), 'instrumentation_patch_sha256': sha(patch_path),
        'benchmark_cohort_switches_enabled': bool(cohort_switch_names),
        'benchmark_cohort_switch_names': cohort_switch_names,
        'benchmark_cohort_option_defaults': cohort_option_defaults,
        'benchmark_extended_diagnostics': extended_diagnostics,
        'benchmark_stats_fields': stats_fields,
        'benchmark_posting_block_bits': a.posting_block_bits,
        'shared_target': str(a.shared_target.resolve()) if a.shared_target else None,
        'cargo_command': command, 'build_environment_overrides': {'CARGO_TARGET_DIR': str(build/'target')},
        'build_log_stdout': 'build.stdout', 'build_log_stderr': 'build.stderr'
    }
    manifest_path = build / 'build_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with (build/'build.stdout').open('w') as out, (build/'build.stderr').open('w') as err:
        result = subprocess.run(command, env=env, stdout=out, stderr=err, text=True)
    manifest['build_returncode'] = result.returncode
    binary = build/'target/release'/f'hf-bpe-native-{a.label}'
    if result.returncode == 0:
        manifest['binary_sha256'] = sha(binary)
        manifest['binary_path'] = str(binary)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(f'build failed; see {build}/build.stdout and build.stderr')
    print(binary)

if __name__ == '__main__':
    main()
