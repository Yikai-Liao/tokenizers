"""Sequential perf diagnostics, separate from uninstrumented rankings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def children(pid):
    try:
        found = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:
        return []
    return [int(child) for child in found] + [nested for child in found for nested in children(child)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/root/code/tokenizers-simplification-results'))
    parser.add_argument('--labels', nargs='+', default=['baseline', 'head-ring'])
    parser.add_argument('--tag', default='zh-core-4')
    args = parser.parse_args()
    template = json.loads((args.root / 'runs/geometry/zh-core-w4-v50000/b1-B/job.json').read_text())
    expected = json.loads((args.root / 'reference/zh-core-w4-v50000.json').read_text())
    for label in args.labels:
        out = args.root / 'profiles' / (args.tag + '-' + label)
        out.mkdir(parents=True, exist_ok=False)
        model = out / 'model.json'
        job = dict(template, attempt_id=out.name, build_id=label, output=str(model))
        (out / 'job.json').write_text(json.dumps(job, indent=2))
        binary = args.root / 'bin' / label
        command = ['taskset', '-c', '0-3', 'perf', 'stat', '--json-output', '-o', str(out / 'stat.json'),
                   '-e', 'task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses',
                   '--', 'perf', 'record', '-e', 'cycles', '-F', '99', '--call-graph', 'dwarf,8192',
                   '-o', str(out / 'perf.data'), '--', str(binary), str(out / 'job.json')]
        swap = 0
        before = time.monotonic()
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                for pid in children(process.pid):
                    try:
                        status = Path(f'/proc/{pid}/status').read_text()
                        swap = max(swap, next((int(s.split()[1]) for s in status.splitlines() if s.startswith('VmSwap:')), 0))
                    except FileNotFoundError:
                        pass
                time.sleep(0.25)
        if process.returncode:
            raise RuntimeError(f'perf failed: {out}')
        metrics = json.loads((out / 'stdout.log').read_text())
        equal = json.loads(model.read_text()) == expected
        record = dict(label=label, command=command, diagnostic=True, model_equal=equal,
                      max_swap_kib=swap, subprocess_wall=time.monotonic()-before,
                      binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), runner=metrics)
        (out / 'result.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        if not equal or swap:
            raise RuntimeError(f'Invalid diagnostic: {out}')
        model.unlink()
        for suffix, options in [('flat', ['--no-children', '--call-graph', 'none']),
                                ('callers', ['--children', '--call-graph', 'graph,0.5,caller'])]:
            with (out / (suffix + '.txt')).open('w') as output:
                subprocess.run(['perf', 'report', '-i', str(out / 'perf.data'), '--stdio',
                                '--percent-limit', '0.5', '--sort', 'dso,symbol', *options],
                               stdout=output, stderr=subprocess.DEVNULL, check=True)


if __name__ == '__main__':
    main()
