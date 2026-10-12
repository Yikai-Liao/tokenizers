from pathlib import Path
import subprocess, json, gzip, hashlib, time, os, re

r = Path(__file__).resolve().parent
assert len(json.loads((r / 'normal.json').read_text())) == 18
assert 'Library comparisons complete' in (r / 'normal-controller.log').read_text()
subprocess.run(['python3', str(r / 'summarize.py')], check=True)
rows = []
for case in ['en-whitespace-w6', 'en-bytelevel_regex-w6', 'zh-bytelevel_regex-w1']:
    repeats = 2 if case.startswith('en') else 1
    for repeat in range(repeats):
        for arm in ['fmt-small']:
            run = r / 'perf-final' / f'{case}-r{repeat}-{arm}'
            run.mkdir(parents=True, exist_ok=False)
            ref = r / 'normal' / f'{case}-r0-{arm}'
            job = json.loads((ref / 'job.json').read_text()) | dict(attempt_id=run.name, output=str(run / 'model.json'))
            (run / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
            inner = [str(r / (arm + '-bin')), str(run / 'job.json')]
            if case.startswith('en'):
                inner = ['perf', 'record', '--no-buildid-cache', '-q', '-o', str(run / 'perf.data'), '-e', 'cpu-clock', '-F', '999', '--call-graph', 'dwarf,4096', '--', *inner]
            cmd = ['taskset', '-c', '0-1', 'perf', 'stat', '-x', ';', '-o', str(run / 'stat.csv'), '-e', 'instructions:u,branches:u,branch-misses:u,cache-misses:u,task-clock', '--', *inner]
            (run / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
            print('start diagnostic', run.name, flush=True)
            with (run / 'stdout.log').open('w') as out, (run / 'stderr.log').open('w') as err:
                p = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
                while p.poll() is None:
                    available = int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)[1])
                    if available < 2_000_000:
                        os.killpg(p.pid, 15)
                        p.wait()
                        raise RuntimeError('memory guard interrupted own diagnostic')
                    time.sleep(.1)
            assert p.returncode == 0, (run.name, p.returncode)
            result = json.loads((run / 'stdout.log').read_text())
            model = (run / 'model.json').read_bytes()
            assert model == gzip.decompress((ref / 'model.json.gz').read_bytes())
            (run / 'model.json.gz').write_bytes(gzip.compress(model, mtime=0))
            (run / 'model.json').unlink()
            if case.startswith('en'):
                with (run / 'self-report.txt').open('w') as out:
                    subprocess.run(['perf', 'report', '-i', str(run / 'perf.data'), '--stdio', '--no-children', '--sort', 'symbol', '--percent-limit', '.3'], stdout=out, stderr=subprocess.STDOUT, check=True)
            row = dict(case=case, arm=arm, repeat=repeat, diagnostic_only=True, metrics=result['metrics'], affinity=result['effective_affinity'], model_sha256=hashlib.sha256(model).hexdigest())
            rows.append(row)
            (r / 'perf-final.json').write_text(json.dumps(rows, indent=2) + '\n')
            print('done diagnostic', run.name, flush=True)

