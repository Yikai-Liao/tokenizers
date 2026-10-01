#!/usr/bin/env python3
"""Build matched legacy/fused batch trainers, seeded feed, and optional diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import os

ROOT = Path(__file__).resolve().parent

def install_diagnostics(source, phase_controls):
    from fused_rewrite_diagnostics import install
    install(source, phase_controls)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('worktree',type=Path)
    parser.add_argument('--label', default='fused-rewrite-v1')
    parser.add_argument('--phase-controls', action='store_true')
    args = parser.parse_args()
    worktree=args.worktree.resolve()
    if subprocess.check_output(['git','-C',str(worktree),'status','--porcelain'],text=True).strip():
        raise SystemExit('worktree must be committed and clean')
    build=ROOT/'.build'/f'native-{args.label}'
    source=build/'source/tokenizers'
    shutil.copytree(worktree/'tokenizers',source,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('target','.git','data'))
    trainer=source/'tk-train/src/trainers/bpe/mod.rs'
    text=trainer.read_text()
    anchor='        Ok((trained.vocab, trained.merges, trained.special_tokens))'
    assert text.count(anchor)==1
    trainer.write_text(text.replace(anchor,
        '        eprintln!("{}", serde_json::json!({"bench_indexed_stats": trained.stats}));\n'+anchor))
    parallel = source/'tk-train/src/trainers/bpe/indexed/parallel.rs'
    text = parallel.read_text()
    anchor = '    train_with_policy(trainer, wc, config, super::posting_arena::Policy::Auto)'
    assert text.count(anchor) == 1
    replacement = """    let policy = match std::env::var("HF_BPE_ARENA_MODE").as_deref() {
        Ok("0") => super::posting_arena::Policy::System,
        Ok("256") => super::posting_arena::Policy::Fixed(256),
        Ok("auto") | Err(_) => super::posting_arena::Policy::Auto,
        _ => panic!("invalid diagnostic arena policy"),
    };
    train_with_policy(trainer, wc, config, policy)"""
    parallel.write_text(text.replace(anchor, replacement))
    # Seed feed maps before corpus construction. Feed is serial in the fair
    # runner, so iteration is reproducible without sorting inside train timing.
    text = trainer.read_text()
    start = text.index('    fn feed<I, S, F>')
    end = text.index('\n#[cfg(test)]', start)
    feed = text[start:end]
    assert feed.count('AHashMap::new()') == 2
    feed = feed.replace('AHashMap::new()',
        'AHashMap::with_hasher(ahash::RandomState::with_seeds(11, 13, 17, 19))')
    trainer.write_text(text[:start] + feed + text[end:])
    # Embed the exact current baseline in the same diagnostic binary.
    baseline_sha = 'adb219cda4600e52ea2809d2537ac551c168dd27'
    legacy = subprocess.check_output(['git', '-C', str(worktree), 'show',
        baseline_sha + ':tokenizers/tk-train/src/trainers/bpe/indexed/parallel.rs'], text=True)
    current = parallel.read_text()
    prefix = legacy[:legacy.index('pub(super) fn train(')].replace('mod fused_batch;', 'mod fused_batch;\nmod fused_rewrite;\nmod flat_commit;')
    # Candidate births use additional empty-by-default output buckets. Keep
    # this common output type while restoring legacy shared-slot access and Plan.
    output_start = prefix.index('struct Output<')
    output_end = prefix.index('// Each split is', output_start)
    modern_start = current.index('struct Output<')
    modern_end = current.index('// Each split is', modern_start)
    prefix = prefix[:output_start] + current[modern_start:modern_end] + prefix[output_end:]
    current = prefix + current[current.index('pub(super) fn train('):]
    start = current.index('        let selected_postings: Vec<_>')
    end = current.index('        let stage = Instant::now();\n        let commits:', start)
    fused_body = current[start:end]
    group_counter = '        stats.flat_route_group_visits += outputs.iter().map(Output::flat_group_visits).sum::<usize>();\n'
    assert group_counter in fused_body
    fused_body = fused_body.replace(group_counter, '')
    legacy_start = legacy.index('        let outputs: Vec<Output<O, INLINE>> = if flat')
    legacy_end = legacy.index('        let stage = Instant::now();\n        let commits:', legacy_start)
    legacy_body = legacy[legacy_start:legacy_end]
    current = current[:start] + '        let outputs = if bench_fused {\n' + fused_body + '            outputs\n        } else {\n' + legacy_body + '            outputs\n        };\n' + group_counter + current[end:]
    current = current.replace('    let max_length = trainer.max_token_length.unwrap_or(usize::MAX);',
        '    let bench_fused = std::env::var("HF_BPE_FUSED_MODE").as_deref() != Ok("baseline");\n'
        '    stats.bench_fused_mode = if bench_fused { "fused" } else { "baseline" };\n'
        '    let max_length = trainer.max_token_length.unwrap_or(usize::MAX);')
    # Keep the baseline's original commit too; fusion's spatial fragmentation
    # requires a different birth assembly, which is part of the candidate.
    commit = current.index('        let commits: Vec<_>')
    a = current.index('                    if flat {', commit)
    z = current.index('                    let mut sums', a)
    old_commit = legacy.index('        let commits: Vec<_>')
    old_a = legacy.index('                    if flat {', old_commit)
    old_z = legacy.index('                    let mut sums', old_a)
    old_flat = legacy[old_a:old_z].replace('if flat {', 'if flat && !bench_fused {', 1)
    current = current[:a] + old_flat + current[a:]
    parallel.write_text(current)
    neighbors = source/'tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs'
    legacy_neighbors = subprocess.check_output(['git', '-C', str(worktree), 'show',
        baseline_sha + ':tokenizers/tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs'], text=True)
    # Retain the common Selected implementation and append the legacy stages.
    neighbors.write_text(neighbors.read_text() + '\nuse super::weight_lookup::WeightLookup;\n' + legacy_neighbors[legacy_neighbors.index('struct Task'):])
    legacy_current = neighbors.read_text()
    legacy_current = legacy_current.replace('    let selected = Selected::new(rules, lengths.len());',
        '    let bench_dense = GROUPED && std::env::var("HF_BPE_DENSE_COMMIT").as_deref() == Ok("1");\n'
        '    let selected = Selected::new(rules, lengths.len());')
    legacy_current = legacy_current.replace('            let mut output = Output::new(workers, true);',
        '            let mut output = Output::new(workers, true);\n'
        '            if bench_dense { output.enable_dense_births(rules.len()); }')
    flush = '''                    left_cache.flush(&mut output, rule, true)?;
                    right_cache.flush(&mut output, rule, false)?;'''
    assert legacy_current.count(flush) == 1
    legacy_current = legacy_current.replace(flush, '''                    if bench_dense {
                        left_cache.flush_dense(&mut output, rule, true, task.rank)?;
                        right_cache.flush_dense(&mut output, rule, false, task.rank)?;
                    } else {
                        left_cache.flush(&mut output, rule, true)?;
                        right_cache.flush(&mut output, rule, false)?;
                    }''')
    neighbors.write_text(legacy_current)
    stats_path = source/'tk-train/src/trainers/bpe/indexed.rs'
    current = stats_path.read_text().replace('pub(super) struct IndexedTrainingStats {',
        'pub(super) struct IndexedTrainingStats {\n    pub bench_fused_mode: &\'static str,\n    pub bench_merge_worker_cpu_ms: Vec<f64>,')
    stats_path.write_text(current)
    current = trainer.read_text()
    config_start = current.index('    pub fn do_train(')
    config_end = current.index('    fn do_train_observed', config_start)
    config = current[config_start:config_end]
    assert config.count('atomic_corpus: true,') == 1
    config = config.replace('atomic_corpus: true,', 'atomic_corpus: std::env::var("HF_BPE_FUSED_ATOMIC").as_deref() != Ok("0"),')
    config = config.replace('posting_block_bits: 32,', 'posting_block_bits: std::env::var("HF_BPE_FUSED_BLOCK_BITS").map_or(32, |v| v.parse().unwrap()),')
    trainer.write_text(current[:config_start] + config + current[config_end:])
    # Detailed diagnostics are installed only in the benchmark overlay.
    install_diagnostics(source, args.phase_controls)

    # Preserve every overlay change, including the existing scalar stats probe.
    import difflib
    patch = []
    for candidate in source.rglob('*.rs'):
        original = worktree/'tokenizers'/candidate.relative_to(source)
        if original.exists():
            patch.extend(difflib.unified_diff(original.read_text().splitlines(True),
                candidate.read_text().splitlines(True), fromfile=str(original), tofile=str(candidate)))
    (build/'instrumentation.patch').write_text(''.join(patch))
    runner=build/'runner'
    (runner/'src').mkdir(parents=True,exist_ok=True)
    text=(ROOT/'src/main.rs').read_text()
    anchor='    let reader = BufReader::new(File::open(input)?);'
    assert text.count(anchor)==1
    text=text.replace(anchor,'    let bench_workers = env::var("HF_BPE_BENCH_WORKERS")\n'
                            '        .map(|value| value.parse::<usize>().expect("invalid benchmark worker count"))\n'
                            '        .unwrap_or(4);\n'
                            '    assert!(matches!(bench_workers, 1 | 4), "benchmark workers must be 1 or 4");\n'
                            '    tk_encode::parallelism::set_num_threads(bench_workers);\n'
                            '    tk_encode::parallelism::set_parallelism(false);\n'+anchor)
    anchor='    let feed_ms = begin.elapsed().as_secs_f64() * 1000.0;'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n    tk_encode::parallelism::set_parallelism(true);')
    (runner/'src/main.rs').write_text(text)
    shutil.copyfile(ROOT/'Cargo.lock',runner/'Cargo.lock')
    (runner/'Cargo.toml').write_text(f'''[package]
name = "hf-bpe-native-{args.label}"
version = "0.1.0"
edition = "2024"
[features]
indexed = []
[dependencies]
tk-train = {{ path = {json.dumps(str(source/'tk-train'))}, default-features = false }}
tk-encode = {{ path = {json.dumps(str(source/'tk-encode'))}, default-features = false }}
serde_json = "1"
sha2 = "0.10"
hf-front-end = {{ package = "tokenizers", version = "=0.23.2", default-features = false, features = ["fancy-regex"] }}
''')
    subprocess.run([str(Path.home()/'.cargo/bin/cargo'),'build','--offline','--release','--manifest-path',str(runner/'Cargo.toml')],
                   check=True,env=dict(os.environ,CARGO_TARGET_DIR=str(build/'target')))
    (build/'baseline.json').write_text(json.dumps(dict(baseline_sha=baseline_sha, source_sha=subprocess.check_output(['git','-C',str(worktree),'rev-parse','HEAD'],text=True).strip()), indent=2)+'\n')
    print(build/f'target/release/hf-bpe-native-{args.label}')

if __name__=='__main__':
    main()
