from pathlib import Path
import json, os, subprocess, time

e = Path(__file__).resolve().parent
root = e.parents[3]
env = dict(os.environ, PATH='/root/.cargo/bin:' + os.environ['PATH'], CARGO_TARGET_DIR='/tmp/bpe-feed-uniform-integration-tests-20261011', CARGO_INCREMENTAL='0', CARGO_PROFILE_TEST_DEBUG='0', CARGO_PROFILE_DEV_DEBUG='0', CARGO_BUILD_JOBS='2')
base = ['cargo', 'test', '--offline', '--locked', '--manifest-path', 'tokenizers/tk-train/Cargo.toml', '--lib']
skip = ['--', '--skip', 'feed_training_tokenizer_json_and_encoding_roundtrip']
checks = [
    ('default', base + skip),
    ('no-default', base + ['--no-default-features'] + skip),
    ('clippy', ['cargo', 'clippy', '--offline', '--locked', '--manifest-path', 'tokenizers/tk-train/Cargo.toml', '--all-targets', '--', '-D', 'warnings']),
    ('format', ['uv', 'run', 'experiments/bpe-simplification/format-changed.py', '--check', 'tokenizers/tk-train/src/trainers/bpe/word_counts.rs', 'tokenizers/tk-train/src/trainers/bpe/tests.rs']),
]
results = []
for name, command in checks:
    log = e / ('integration-' + name + '.log')
    start = time.monotonic()
    with log.open('w') as output:
        run = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
    results.append(dict(name=name, command=command, exit_code=run.returncode, seconds=time.monotonic()-start, log=log.name))
    (e / 'integration-validation.json').write_text(json.dumps(dict(parent=json.loads((e/'integration.json').read_text())['integration_parent'], environment={key: env[key] for key in ['CARGO_TARGET_DIR', 'CARGO_INCREMENTAL', 'CARGO_PROFILE_TEST_DEBUG', 'CARGO_PROFILE_DEV_DEBUG', 'CARGO_BUILD_JOBS']}, checks=results), indent=2) + '\n')
    print(json.dumps(results[-1]), flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)
