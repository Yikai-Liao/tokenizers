from pathlib import Path
import subprocess, json, os, time, re, hashlib, csv, resource, argparse

root = Path(__file__).resolve().parent
inputs = json.loads((root / 'inputs.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--suite', default='screen')
parser.add_argument('--arms', nargs='+', default=['baseline', 'cache', 'split'])
args = parser.parse_args()
rows = []
expected = {}

def vm():
    return {k: int(v) for k, v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin', 'pswpout')}

def sample(case, arm, job, repeat):
    run = root / args.suite / f'{case}-r{repeat}-{arm}'
    run.mkdir(parents=True, exist_ok=False)
    job |= dict(attempt_id=run.name, build_id=arm, output=str(run / 'model.json'))
    (run / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    before = vm()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    swap = rss = faults = 0
    with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(['taskset', '-c', '0-3', str(root / (arm + '-bin')), str(run / 'job.json')], stdout=stdout, stderr=stderr)
        while process.poll() is None:
            try:
                status = (Path('/proc') / str(process.pid) / 'status').read_text()
                for field in ['VmSwap', 'VmRSS']:
                    match = re.search(r'^' + field + r':\s+(\d+)', status, re.M)
                    if match:
                        if field == 'VmSwap':
                            swap = max(swap, int(match[1]))
                        else:
                            rss = max(rss, int(match[1]))
                stat = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
                faults = max(faults, int(stat[9]))
            except FileNotFoundError:
                pass
            time.sleep(.05)
    if process.returncode:
        raise RuntimeError((run, process.returncode, (run / 'stderr.log').read_text()))
    result = json.loads((run / 'stdout.log').read_text())
    stderr = (run / 'stderr.log').read_text()
    initialization = [float(x) for x in re.findall(r'BPE_VOCABULARY_INITIALIZATION_RAW_SECONDS ([\d.]+)', stderr)]
    selection = [float(x) for x in re.findall(r'BPE_ALPHABET_SELECTION_SECONDS ([\d.]+)', stderr)]
    if not selection:
        combined = [float(x) for x in re.findall(r'BPE_INITIALIZATION_SECONDS ([\d.]+)', stderr)]
        # mod.rs times select_alphabet once and adds that duration to first
        # attempt initialization. The target initialization uses its raw timer.
        selection = [sum(combined) - sum(initialization)]
    assert initialization and selection
    model = (run / 'model.json').read_bytes()
    expected.setdefault(case, model)
    assert model == expected[case], (case, arm, 'vocabulary or ordered merges differ')
    after = vm()
    final_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # The historical architecture selects inside Vocabulary::initialize. Its
    # exclusive initialization subtracts the nested selection timer; current
    # baseline/cache/split raw initialization already excludes selection.
    initialization_seconds = sum(initialization) - (sum(selection) if arm == 'oldflag' else 0)
    assert initialization_seconds >= 0
    row = dict(case=case, repeat=repeat, arm=arm, vocabulary_initialization_seconds=initialization_seconds, raw_initialization_timer_seconds=sum(initialization), initialization_includes_selection=arm=='oldflag', alphabet_selection_seconds=sum(selection), attempts=len(initialization), **result['metrics'], **result['output'], child_max_swap_kib=swap, child_major_faults=final_usage.ru_majflt-usage.ru_majflt, child_major_faults_sampled=faults, child_max_rss_kib=rss, system_swapin_pages=after['pswpin']-before['pswpin'], system_swapout_pages=after['pswpout']-before['pswpout'], external_wall_seconds=time.monotonic()-start, model_sha256=hashlib.sha256(model).hexdigest())
    rows.append(row)
    (root / (args.suite + '.json')).write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)

if args.suite == 'controls':
    previous = Path('/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011')
    for case in ['aa', 'reuse']:
        job = json.loads((previous / 'controls' / (case + '-baseline') / 'job.json').read_text())
        for arm in args.arms:
            sample(case, arm, job.copy(), 0)
            assert rows[-1]['attempts'] == (2 if case == 'reuse' else 1)
else:
    for decorated in ([True] if args.suite == 'flags' else [False, True]):
        for item in inputs:
            case = item['language'] + ('-affix' if decorated else '-plain')
            for repeat in range(2 if decorated else 1):
                arms = args.arms if (repeat + (item['language'] == 'en')) % 2 == 0 else list(reversed(args.arms))
                for arm in arms:
                    job = dict(protocol_version=1, input_id=item['language']+'256', mode='core', input=item['prepared_path'], workers=4, pretokenizer='whitespace', trainer=dict(vocab_size=50000, min_frequency=2, prefix='##' if decorated else None, suffix='</w>' if decorated else None, max_token_length=None, limit_alphabet=item['limit_alphabet']))
                    sample(case, arm, job, repeat)
with (root / (args.suite + '.csv')).open('w') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
