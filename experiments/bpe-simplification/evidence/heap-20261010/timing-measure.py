import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/tmp/bpe-heap-timing-measurements')
root.mkdir(exist_ok=False)
binary=Path('/tmp/bpe-heap-timed')
inputs=[r for r in json.loads(Path('/root/code/tokenizers-simplification-results/online-initial/input-pretokenizers.json').read_text()) if r['language']=='zh']
records=[]
digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=dict(binary=str(binary),binary_sha256=digest(binary),inputs=inputs,workers=[4,6],scope='Initial candidate collect+heapify; best including count certification/lazy repair/stale drops; take including count removal; joined serial birth push; final index cleanup including queue/list/count-map/routes destruction. Nonoverlapping coordinator wall intervals. Instant/atomic instrumentation overhead included; no per-occurrence instrumentation.',limitations='One phase profile per case/worker count. Share within instrumented do_train, not baseline vs candidate performance evidence. best/take and index cleanup are queue-path upper bounds including hash lookup/free/count maps, not pure heap CPU. Parallel owner Candidate creation/counting excluded. Six usable CPUs, no extrapolation beyond six.')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
for workers in [4,6]:
    for item in inputs:
        out=root/f'{item["case"]}-w{workers}'
        out.mkdir()
        job=dict(protocol_version=1,attempt_id=out.name,build_id='timing',input_id=item['case'],mode='core',input=item['prepared_path'],output=str(out/'model.json'),workers=workers,pretokenizer=item['pretokenizer'],trainer=dict(vocab_size=50000,min_frequency=2,prefix=None,suffix=None,max_token_length=None))
        (out/'job.json').write_text(json.dumps(job,indent=2))
        competitors=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try:
                comm=(entry/'comm').read_text().strip()
                exe=str((entry/'exe').resolve())
                if comm in ('cargo','rustc') or comm.startswith('bpe-bench') or exe in [str(binary),'/tmp/bpe-global-measurements/candidate','/tmp/bpe-heap-clean','/tmp/bpe-heap-diagnostic-baseline','/tmp/bpe-heap-diagnostic-candidate']:
                    competitors.append((int(entry.name),comm))
            except (FileNotFoundError,ProcessLookupError): pass
        if competitors: raise RuntimeError(f'Concurrent job: {competitors}')
        env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','BPE_HEAP_DIAGNOSTIC','BPE_HEAP_TIMING','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE'))}
        env['BPE_HEAP_TIMING']='1'
        command=['taskset','-c',f'0-{workers-1}',str(binary),str(out/'job.json')]
        swap=0
        with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
            process=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
            while process.poll() is None:
                try:
                    status=Path(f'/proc/{process.pid}/status').read_text().splitlines()
                    swap=max(swap,next((int(s.split()[1]) for s in status if s.startswith('VmSwap:')),0))
                except (FileNotFoundError,ProcessLookupError): pass
                time.sleep(.05)
        record=dict(case=item['case'],workers=workers,command=command,returncode=process.returncode,max_swap_kib=swap,concurrent_builds_or_benchmarks=competitors)
        if process.returncode==0:
            record.update(json.loads((out/'stdout.log').read_text()))
            phase=[json.loads(s.removeprefix('BPE_HEAP_TIMING ')) for s in (out/'stderr.log').read_text().splitlines() if s.startswith('BPE_HEAP_TIMING ')]
            assert len(phase)==1
            record['phase']=phase[0]
            reference=Path('/tmp/bpe-heap-measurements')/f'clean-{item["case"]}-b1-baseline/model.json'
            record['model_equal']=json.loads((out/'model.json').read_text())==json.loads(reference.read_text())
            record['model_sha256']=digest(out/'model.json')
            record['valid']=record['model_equal'] and swap==0
            total=record['metrics']['train_seconds']
            ns_names=['initial_ns','best_ns','take_ns','serial_birth_push_ns','index_cleanup_ns']
            record['phase_seconds']={k:record['phase'][k]/1e9 for k in ns_names}
            record['phase_percent']={k:record['phase'][k]/1e9/total*100 for k in ns_names}
            record['queue_path_total_seconds']=sum(record['phase_seconds'].values())
            record['queue_path_percent']=record['queue_path_total_seconds']/total*100
        else: record['valid']=False
        records.append(record)
        (root/'runs.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
        if not record['valid']: raise RuntimeError(f'Invalid timing profile: {out}')
(root/'summary.json').write_text(json.dumps(records,indent=2))
