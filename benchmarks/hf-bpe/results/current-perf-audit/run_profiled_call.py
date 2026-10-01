#!/usr/bin/env python3
"""One original-API candidate training call under bounded perf sampling and proc monitoring."""
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
BENCH=HERE.parents[1]
PERF=Path('/root/code/tokenizers/benchmarks/hf-bpe/.build/native-j-current-perf.perf')
BIN=Path('/root/code/tokenizers/benchmarks/hf-bpe/.build/native-j-current-perf-target/release/hf-bpe-native-j-current-perf')
CORPUS=Path('/root/code/tokenizers/benchmarks/hf-bpe/.build/gb-corpus/zh-512m.txt')
OUT=HERE/'current.jsonl'
if OUT.exists() or PERF.exists(): raise SystemExit('refusing to overwrite audit outputs')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def meminfo():
    d={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)}
    return {'mem_available_bytes':d['MemAvailable'],'swap_free_bytes':d['SwapFree'],'pswpin':d.get('pswpin',0),'pswpout':d.get('pswpout',0)}
def descendants(root):
    todo=[root]; seen=set()
    while todo:
        pid=todo.pop()
        if pid in seen: continue
        seen.add(pid)
        try: kids=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except OSError: kids=[]
        todo.extend(map(int,kids))
    return seen
def proc_status(pid):
    try:
        lines=Path(f'/proc/{pid}/status').read_text().splitlines()
        d={s.split(':',1)[0]:int(s.split(':',1)[1].split()[0])*1024 for s in lines if s.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        return d,cmd
    except OSError:return {},''
def cpustat():
    vals=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    return vals[:8]
if not BIN.is_file() or not CORPUS.is_file(): raise SystemExit('binary/input missing')
before=meminfo()
if before['mem_available_bytes'] <= 1<<30: raise SystemExit('MemAvailable <= 1 GiB before start')
metadata={'source_commit':'029ab45bd0b0f446035f71acbd247c804bea60b8','binary':str(BIN),'binary_sha256':sha(BIN),'build_id':'7faa1d5a8f37cd203ba61fb94200d5ba80ebfa42','input':str(CORPUS),'input_sha256':sha(CORPUS),'input_bytes':CORPUS.stat().st_size,'parameters':{'split':'none','backend':'reference','vocab_size':50000,'min_frequency':2,'initialization_workers':4,'merge_workers':4,'atomic_corpus':True,'expected_layout':'parallel_u32_flat32','allocator':'system default Rust allocator'},'environment':{'TOKENIZERS_PARALLELISM':'false','RAYON_NUM_THREADS':'4','HF_BPE_BENCH_WORKERS':'4'},'perf_command':['perf','record','-F','99','-e','cycles:u','-e','cache-misses:u','--call-graph','dwarf,16384','-o',str(PERF),'--',str(BIN),str(CORPUS),'none','reference','50000','2'],'perf_event_paranoid':Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),'system_before':before}
(HERE/'current.provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
env=dict(os.environ,TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='4',HF_BPE_BENCH_WORKERS='4')
cmd=metadata['perf_command']
log=(HERE/'perf-record.stderr').open('w')
start=time.monotonic(); startcpu=cpustat()
p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=True)
peak_rss=peak_hwm=peak_swap=0; min_avail=before['mem_available_bytes']; sampled=0; target_pids=set(); stop_reason=None
while p.poll() is None:
    now=meminfo(); min_avail=min(min_avail,now['mem_available_bytes']); sampled+=1
    for pid in descendants(p.pid):
        d,c=proc_status(pid)
        if BIN.name in c:
            target_pids.add(pid); peak_rss=max(peak_rss,d.get('VmRSS',0)); peak_hwm=max(peak_hwm,d.get('VmHWM',0)); peak_swap=max(peak_swap,d.get('VmSwap',0))
    if now['mem_available_bytes'] <= 1<<30:
        stop_reason='MemAvailable <= 1 GiB'; os.killpg(p.pid,signal.SIGTERM); break
    time.sleep(.5)
returncode=p.wait(); wall=time.monotonic()-start; log.close(); end=meminfo(); endcpu=cpustat()
stdout=p.stdout.read()
(HERE/'profiled.stdout').write_text(stdout)
(HERE/'current.provenance.json').write_text(json.dumps({**metadata,'perf_returncode':returncode,'profiling_wall_seconds':wall,'monitor_samples':sampled,'target_pids_seen':sorted(target_pids),'min_mem_available_bytes':min_avail,'target_sampled_peak_rss_bytes':peak_rss,'target_observed_vmhwm_bytes':peak_hwm,'target_sampled_peak_vmswap_bytes':peak_swap,'system_after':end,'cpu_ticks_delta':[b-a for a,b in zip(startcpu,endcpu)],'stop_reason':stop_reason,'perf_data_sha256':sha(PERF) if PERF.exists() else None},indent=2)+'\n')
if returncode or stop_reason: raise SystemExit(f'profile call failed: returncode={returncode}, stop={stop_reason}; outputs retained')
rows=[json.loads(x) for x in stdout.splitlines() if x.strip()]
if len(rows)!=1: raise SystemExit(f'expected one runner output row, got {len(rows)}')
row=rows[0]
err=(HERE/'perf-record.stderr').read_text()
stats=[json.loads(line)['bench_indexed_stats'] for line in err.splitlines() if line.startswith('{') and 'bench_indexed_stats' in line]
if len(stats)!=1: raise SystemExit(f'expected one stats row, got {len(stats)}')
row['indexed_stats']=stats[0]; row.update(engine='j-current-perf',commit=metadata['source_commit'],input_sha256=metadata['input_sha256'],provenance='current.provenance.json',profiling_wall_seconds=wall)
OUT.write_text(json.dumps(row)+'\n')
print(json.dumps({'train_ms':row.get('train_ms'),'initialize_ms':stats[0].get('initialize_ms'),'merge_ms':stats[0].get('merge_ms'),'input_sha256':row.get('input_sha256'),'model_sha256':row.get('model_sha256'),'min_mem_available_bytes':min_avail,'peak_rss_bytes':peak_hwm,'peak_vmswap_bytes':peak_swap,'perf_sha256':sha(PERF)}))
