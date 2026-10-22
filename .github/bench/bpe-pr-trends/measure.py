"""Public pipeline timing and full-process phase observation, shared with local evidence."""
from pathlib import Path
import gzip,hashlib,json,os,subprocess,time,shutil
OUT=None
CORPORA={}
EXPECTED={}
ENV={}
CPUS=[0,1,2,3]
SIZE_MIB=128

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()

def phase_summary(events, samples, peak):
    stack=[];intervals=[];updates=[];hi=0;previous=None
    for e in events:
        at=e['unix_ns'];value=e['ru_maxrss_kib']
        phase=stack[-1] if stack else 'startup_or_teardown'
        if previous is not None:intervals.append((previous,at,phase))
        if value>hi:updates.append({'at_ns':at,'phase':phase,'peak_kib':value,'boundary':e['action']});hi=value
        if e['action']=='begin':stack.append(e['phase'])
        else:
            if not stack or stack[-1]!=e['phase']:raise ValueError(f'unbalanced phases {stack} {e}')
            stack.pop()
        previous=at
    phase=next((u['phase'] for u in updates if u['peak_kib']>=peak),'startup_or_teardown')
    # Samples are instantaneous RSS envelopes, not guaranteed per-phase maxima.
    envelopes={};j=0
    for sample in samples:
        at=sample['unix_ns']
        while j<len(intervals) and at>intervals[j][1]:j+=1
        name=intervals[j][2] if j<len(intervals) and at>=intervals[j][0] else 'startup_or_teardown'
        envelopes[name]=max(envelopes.get(name,0),sample['rss_kib'])
    return {'global_peak_phase':phase,'highwater_updates':updates,'sampled_phase_rss_kib':envelopes,'peak_boundary_hwm_kib':hi,'near_peak_phases':list(dict.fromkeys(u['phase'] for u in updates if u['peak_kib']>=peak-1024))}

def run(number,head,binary,language,attempt='main'):
    dest=OUT/'runs'/f'{number:02d}-{language}-{attempt}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    output=dest/'model.json'
    job={'protocol_version':1,'attempt_id':f'{number:02d}-{language}-{attempt}','build_id':head,'input_id':language+str(SIZE_MIB)+'-sha256:'+EXPECTED[language],'mode':'pipeline','input':str(CORPORA[language]),'output':str(output),'workers':4,'pretokenizer':'bytelevel_regex','trainer':{'vocab_size':100000,'min_frequency':2,'prefix':None,'suffix':None,'max_token_length':None}}
    (dest/'job.json').write_text(json.dumps(job,indent=2))
    args=['/usr/bin/time','-o',str(dest/'time.json'),'-f','{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"maxrss_kib":%M,"major_faults":%F,"swaps":%W,"exit_code":%x}','taskset','-c',','.join(map(str,CPUS)),str(binary),str(dest/'job.json')]
    def host_snapshot():
        return {'unix_ns':time.time_ns(),'loadavg':Path('/proc/loadavg').read_text().strip(),'memory':Path('/proc/meminfo').read_text(),'cpu_stat':Path('/proc/stat').read_text().splitlines()[:7]}
    (dest/'host-before.json').write_text(json.dumps(host_snapshot(),indent=2))
    samples=[]
    with (dest/'result.json').open('w') as stdout,(dest/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(args,env=ENV,stdout=stdout,stderr=stderr)
        child=None
        while proc.poll() is None:
            if child is None:
                try:
                    children=Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split();child=int(children[0]) if children else None
                except FileNotFoundError:pass
            if child is not None:
                try:
                    fields={}
                    for line in Path(f'/proc/{child}/status').read_text().splitlines():
                        if line.startswith(('VmRSS:','VmHWM:','VmSwap:','Threads:')):
                            key,value,*_=line.split();fields[key[:-1]]=int(value)
                    if 'VmRSS' in fields:samples.append({'unix_ns':time.time_ns(),'rss_kib':fields['VmRSS'],'hwm_kib':fields['VmHWM'],'swap_kib':fields.get('VmSwap',0),'threads':fields.get('Threads',0)})
                except (FileNotFoundError,ProcessLookupError):pass
            time.sleep(.01)
    (dest/'host-after.json').write_text(json.dumps(host_snapshot(),indent=2))
    # Save the sampling evidence even when a process is interrupted or fails.
    with gzip.open(dest/'samples.jsonl.gz','wt',compresslevel=1) as f:
        for item in samples:f.write(json.dumps(item)+'\n')
    if proc.returncode:
        (dest/'failure.json').write_text(json.dumps({'returncode':proc.returncode,'sample_count':len(samples),'host_after':str(dest/'host-after.json')},indent=2))
        raise RuntimeError(f'benchmark failed {dest}; return {proc.returncode}')
    events=[json.loads(line.split('BPE_REVIEW_PHASE ',1)[1]) for line in (dest/'stderr.log').read_text().splitlines() if 'BPE_REVIEW_PHASE ' in line]
    external=json.loads((dest/'time.json').read_text());result=json.loads((dest/'result.json').read_text())
    if result['effective_affinity']!=CPUS:raise ValueError(result['effective_affinity'])
    reference=OUT/'reference'/f'{language}-model.json.gz';reference.parent.mkdir(exist_ok=True)
    modelsha=sha(output)
    if number==0 and attempt=='main':
        with open(output,'rb') as inp,gzip.open(reference,'wb',compresslevel=6) as out:shutil.copyfileobj(inp,out)
        equal=True
    else:
        # The protocol writes vocab in ID order and merges in commit order.
        with gzip.open(reference,'rt') as f:expected=json.load(f)
        with open(output) as f:equal=json.load(f)==expected
    with open(output,'rb') as inp,gzip.open(str(output)+'.gz','wb',compresslevel=6) as out:shutil.copyfileobj(inp,out)
    output.unlink()
    summary={'number':number,'head':head,'language':language,'attempt':attempt,'metrics':result['metrics'],'process':external,'model_sha256':modelsha,'model_equal_baseline':equal,'effective_affinity':result['effective_affinity'],'max_process_swap_kib':max((s['swap_kib'] for s in samples),default=0),'feed_train_hwm_kib':next((e['ru_maxrss_kib'] for e in reversed(events) if e['phase']=='train_other' and e['action']=='end'),None),'sample_interval_seconds':.01,'sample_count':len(samples),'phase':phase_summary(events,samples,external['maxrss_kib']),'path':str(dest)}
    (dest/'phases.json').write_text(json.dumps(events,indent=2));(dest/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'results.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    print(json.dumps({'number':number,'language':language,'attempt':attempt,'seconds':result['metrics']['pipeline_seconds'],'rss_mib':external['maxrss_kib']/1024,'peak_phase':summary['phase']['global_peak_phase'],'model_equal':equal,'swap_kib':summary['max_process_swap_kib']}),flush=True)
    if not equal:raise RuntimeError(f'large model differs at {dest}')
    return summary
