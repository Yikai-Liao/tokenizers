"""Prepare inputs, build a source ref, or measure all refs on one machine."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys
import measure

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())


def command(argv, *, cwd=None, env=None, log=None):
    if log:
        with Path(log).open('w') as stream:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {argv}\n{result.stderr if not log else log}')
    return '' if log else result.stdout


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_inputs(args):
    args.shared.mkdir(parents=True, exist_ok=True)
    for language, expected in PLAN['inputs'].items():
        output = args.shared / 'inputs' / f'{language}{PLAN["size_mib"]}'
        command([sys.executable, '-m', 'bench', 'corpus', '--dataset', str(HERE / f'wikipedia-{language}.json'), '--size-mib', str(PLAN['size_mib']), '--out', str(output), '--cache', str(args.shared.parent / 'shards')], cwd=args.protocol)
        path = output / 'text.txt'
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise RuntimeError(f'input verification failed for {language}')
    (args.shared / 'verified-inputs.json').write_text(json.dumps(PLAN['inputs'], indent=2))


def resolve_ref(repository, ref):
    output = command(['git', 'ls-remote', f'https://github.com/{repository}.git', f'refs/heads/{ref}'])
    lines = output.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f'ref not found or ambiguous: {repository}:{ref}')
    return lines[0].split()[0]


def resolve(args):
    baseline = {'version': 'baseline', 'repository': 'huggingface/tokenizers', 'ref': args.baseline_ref, 'sha': resolve_ref('huggingface/tokenizers', args.baseline_ref)}
    refs = [ref.strip() for ref in args.candidate_refs.replace(',', '\n').splitlines() if ref.strip()]
    if not refs:
        raise RuntimeError('candidate_refs must contain at least one branch')
    candidates = []
    for index, ref in enumerate(refs, 1):
        candidates.append({'version': f'candidate-{index}', 'repository': args.repository, 'ref': ref, 'sha': resolve_ref(args.repository, ref)})
    print(json.dumps({'baseline': baseline, 'candidates': candidates}, separators=(',', ':')))


def build(args):
    source = args.source.resolve()
    if command(['git', 'rev-parse', 'HEAD'], cwd=source).strip() != args.head:
        raise RuntimeError('source checkout does not match resolved SHA')
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CARGO_INCREMENTAL='0', CARGO_BUILD_JOBS='2', CARGO_TARGET_DIR=str(args.work / 'target'), RUSTUP_TOOLCHAIN=PLAN['rust_toolchain'], CARGO_PROFILE_RELEASE_OPT_LEVEL='3', CARGO_PROFILE_RELEASE_LTO='fat', CARGO_PROFILE_RELEASE_CODEGEN_UNITS='1')
    runner = args.work / 'runner'
    shutil.copytree(HERE / 'runner', runner)
    (runner / 'Cargo.toml').write_text((HERE / 'runner' / 'Cargo.toml.in').read_text().replace('@SOURCE@', str(source)))
    manifest_dir = args.out / 'build'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    command(['cargo', 'build', '--release', '--manifest-path', str(runner / 'Cargo.toml')], env=env, log=manifest_dir / 'build.log')
    binary = Path(env['CARGO_TARGET_DIR']) / 'release' / 'bpe-bench-runner'
    copied = manifest_dir / 'bpe-bench-runner'
    shutil.copy2(binary, copied)
    manifest = {'version': args.version, 'ref': args.ref, 'head': args.head, 'binary_sha256': sha(copied), 'rustc': command(['rustc', '-Vv'], env=env)}
    (manifest_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def find_builds(directory, versions):
    found = {}
    for manifest_path in Path(directory).glob('bpe-build-*/build/manifest.json'):
        manifest = json.loads(manifest_path.read_text())
        binary = manifest_path.parent / 'bpe-bench-runner'
        if manifest['version'] in versions:
            if measure.sha(binary) != manifest['binary_sha256']:
                raise RuntimeError(f'binary checksum mismatch for {manifest["version"]}')
            binary.chmod(0o755)  # artifact downloads drop the executable bit
            found[manifest['version']] = (manifest, binary)
    missing = [version for version in versions if version not in found]
    if missing:
        raise RuntimeError(f'missing build artifacts: {missing}')
    return found


def measure_job(args):
    versions = json.loads(args.versions_json)
    order = [versions['baseline'], *versions['candidates']]
    version_names = [item['version'] for item in order]
    builds = find_builds(args.binaries, version_names)
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='true')
    corpora = {}
    for language, info in PLAN['inputs'].items():
        path = args.shared / 'inputs' / info['path']
        if path.stat().st_size != info['bytes'] or measure.sha(path) != info['sha256']:
            raise RuntimeError(f'downloaded input verification failed for {language}')
        if args.language == language:
            corpora[language] = path
    reference = args.out / 'reference.json.gz'

    def run(version, round_name, warmup):
        manifest, binary = builds[version]
        print(f'{version} {round_name} ...', flush=True)
        record, model = measure.run_one(version=version, ref=manifest['ref'], head=manifest['head'], binary=binary, corpus=corpora[args.language], expected_sha=PLAN['inputs'][args.language]['sha256'], language=args.language, workers=args.workers, round_name=round_name, warmup=warmup, output_root=args.out, environment=environment)
        metrics = record['metrics']
        print(f"  feed {metrics['feed_seconds']:.2f}s train {metrics['train_seconds']:.2f}s rss {record['process']['maxrss_kib'] / 1024:.0f} MiB", flush=True)
        return record, model

    # Warm up every version; the baseline warmup model is the reference for all timed runs.
    for version in version_names:
        _, model = run(version, 'warmup', True)
        if version == 'baseline':
            measure.save_reference(model, reference)
    records = []
    for round_number in range(1, args.rounds + 1):
        start = round_number % len(version_names)
        for version in version_names[start:] + version_names[:start]:
            record, model = run(version, f'round-{round_number}', False)
            record['model_equal_baseline'] = measure.compare_model(model, reference)
            records.append(record)
            if not record['model_equal_baseline']:
                raise RuntimeError(f'model mismatch for {version}/{args.language}/workers={args.workers}')
    (args.out / 'results.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    (args.out / 'completion.json').write_text(json.dumps({'language': args.language, 'workers': args.workers, 'rounds': args.rounds, 'versions': version_names}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('--protocol', type=Path, required=True); prepare.add_argument('--shared', type=Path, required=True)
    resolve_parser = sub.add_parser('resolve'); resolve_parser.add_argument('--repository', required=True); resolve_parser.add_argument('--baseline-ref', default='main'); resolve_parser.add_argument('--candidate-refs', required=True)
    build_parser = sub.add_parser('build'); build_parser.add_argument('--version', required=True); build_parser.add_argument('--ref', required=True); build_parser.add_argument('--head', required=True); build_parser.add_argument('--source', type=Path, required=True); build_parser.add_argument('--out', type=Path, required=True); build_parser.add_argument('--work', type=Path, required=True)
    measure_parser = sub.add_parser('measure'); measure_parser.add_argument('--versions-json', required=True); measure_parser.add_argument('--binaries', type=Path, required=True); measure_parser.add_argument('--shared', type=Path, required=True); measure_parser.add_argument('--out', type=Path, required=True); measure_parser.add_argument('--language', choices=('en', 'zh'), required=True); measure_parser.add_argument('--workers', type=int, choices=(1, 4), required=True); measure_parser.add_argument('--rounds', type=int, default=PLAN['rounds'])
    args = parser.parse_args()
    if args.command == 'prepare': prepare_inputs(args)
    elif args.command == 'resolve': resolve(args)
    elif args.command == 'build': build(args)
    else: measure_job(args)


if __name__ == '__main__':
    main()
