"""Freeze the actual clean baseline; isolate only fixed-length storage changes."""
from pathlib import Path
import hashlib, json, shutil, subprocess

r = Path(__file__).resolve().parent
repo = Path('/home/lyk/code/tokenizers')
base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=repo, text=True)
assert not status, status
shutil.copy2('/home/lyk/.paseo/uploads/upload_3208077e-c944-476d-8d39-65145c2b0485/bpe-vec-to-box-experiment-plan.md.txt', r/'plan.md')
manifest = {'baseline_commit': base, 'initial_status': status, 'builds': {}}
source = repo/'experiments/bpe-simplification/evidence/feed-cost-20261011/production/source/drain'
for arm in ('baseline', 'boxed'):
    w = r/('worktree-'+arm)
    subprocess.run(['git', 'worktree', 'add', '--detach', str(w), base], cwd=repo, check=True)
    for f in ('tokenizers/tk-train/Cargo.lock', 'experiments/bpe-simplification/runner/Cargo.lock'):
        shutil.copy2(repo/f, w/f)
    for name in ('pipeline.rs', 'feed_stats.rs', 'protocol.rs'):
        f = Path('experiments/bpe-simplification/runner/src')/name
        shutil.copy2(source/f, w/f)
    f = w/'experiments/bpe-simplification/runner/src/pipeline.rs'
    s = f.read_text()
    marker = '    let pipeline_seconds = begin.elapsed().as_secs_f64();'
    s = s.replace(marker, '''    // Complete model output belongs to the full-pipeline measurement.
    model_output::write(&job.output, &vocab, &merges)?;
    let special_path = std::path::Path::new(&job.output).with_extension("special.json");
    let special_json: Vec<_> = special.iter().map(|token| json!({
        "content":token.content,"single_word":token.single_word,"lstrip":token.lstrip,
        "rstrip":token.rstrip,"normalized":token.normalized,"special":token.special
    })).collect();
    serde_json::to_writer(File::create(&special_path)?, &special_json)?;
''' + marker)
    s = s.replace('    model_output::write(&job.output, &vocab, &merges)?;\n    Ok(', '    Ok(')
    s = s.replace('"counts_path":counts_path,', '"special_tokens_path":special_path,"counts_path":counts_path,')
    s = s.replace('public-feed-and-train-before-serialization', 'public-feed-train-and-complete-model-output-before-validation')
    f.write_text(s)
    manifest['builds'][arm] = {'worktree': str(w), 'binary': str(r/arm), 'source_commit': base, 'binary_sha256': None}
(r/'builds.json').write_text(json.dumps(manifest, indent=2))
shutil.copy2(repo/'experiments/bpe-simplification/evidence/initialization-rounds-20261011/inputs.json', r/'inputs.json')
(r/'decision-rule.json').write_text(json.dumps({
    'prior_evidence': 'Prior reports treat around one percent on this shared laptop as noise, do not claim stable gains, and permit small increases for simpler ownership.',
    'screen_before_results': 'Retain only if correctness is unchanged and no repeated material regression. Material screen: pipeline wall or CPU at least +3% in all paired runs, or HWM at least +3% and +32 MiB in all pairs; investigate lifecycle groups if repeated.',
    'third_pair': 'After two pairs add one reversed-order alternating pair if time directions conflict with a magnitude above 1%, absolute median wall/CPU exceeds 1%, or HWM exceeds 3% and 32 MiB. Differences entirely below 1% cannot support resource benefits and need no extra pair to choose semantic clarity.',
    'claims': 'A third pair is descriptive, not statistical significance. Never claim speed or RSS benefits from descriptor sizes or requested payload alone.'
}, indent=2))
print(base, flush=True)
