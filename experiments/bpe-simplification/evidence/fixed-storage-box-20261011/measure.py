"""Serial warmups, forward/reverse formal pairs, and rule-selected third pairs."""
import codecs,hashlib,json,os,statistics,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'measurements';out.mkdir(exist_ok=True)
build=json.loads((r/'builds.json').read_text());inputs=json.loads((r/'inputs.json').read_text())
cases=[dict(x,workers=w,scenario=x['case']+'-w'+str(w)) for x in inputs for w in (1,4)]
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
for b in build['builds'].values():assert digest(b['binary'])==b['binary_sha256']
for x in inputs:
    assert digest(x['source'])==x['source_sha256'] and Path(x['source']).stat().st_size==x['source_bytes']
    decoder=codecs.getincrementaldecoder('utf-8')('strict')
    with Path(x['source']).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):decoder.decode(chunk)
    decoder.decode(b'',final=True)
manifest={'builds':build,'inputs':inputs,'workers':[1,4],'warmups_per_arm_scenario':1,'formal_pairs':2,'decision_rule':json.loads((r/'decision-rule.json').read_text()),
 'cpu':subprocess.check_output(['lscpu'],text=True),'rustc':subprocess.check_output(['rustc','-Vv'],text=True),
 'allocator':'Rust System, glibc default; no allocator replacement',
 'timing':'streaming feed + train_vocab + complete vocabulary/ordered merges and special-token JSON output; HWM/RSS and faults sampled before count export, canonical sort or validation; train excludes output; no encode',
 'cache':'source hash/UTF8 verification before matrix; one uncounted fresh-process warmup per arm per scenario; ordinary OS cache, no eviction; serial processes and no concurrent cargo/rustc/benchmarks',
 'limitations':'normal RandomState::default; no trainer phase instrumentation; child VmSwap/RSS polled every 25 ms; HWM cumulative'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
rp=out/'runs.json';rows=json.loads(rp.read_text()) if rp.exists() else [];refs={}
for row in rows:
    if row['valid'] and row['case'] not in refs:
        d=out/row['run'];refs[row['case']]=(row['counts_sha256'],json.loads((d/'model.json').read_text()),json.loads((d/'model.special.json').read_text()))
def sample(x,phase,block,arm):
    if any(a['scenario']==x['scenario'] and a['phase']==phase and a['block']==block and a['arm']==arm for a in rows):return
    competing=[]
    for e in Path('/proc').iterdir():
        if not e.name.isdigit():continue
        try:
            comm=(e/'comm').read_text().strip();exe=str((e/'exe').resolve())
            if comm in ('cargo','rustc') or comm.startswith('bpe-bench') or exe in [a['binary'] for a in build['builds'].values()]:competing.append([e.name,comm])
        except (FileNotFoundError,ProcessLookupError):pass
    assert not competing,competing
    name=f"{x['scenario']}-{phase}{block}-{arm}";d=out/name;d.mkdir(exist_ok=False)
    job={'protocol_version':1,'attempt_id':name,'build_id':arm,'input_id':x['case'],'mode':'pipeline','input':x['source'],'output':str(d/'model.json'),'workers':x['workers'],'pretokenizer':x['pretokenizer'],'trainer':{'vocab_size':50000,'min_frequency':2,'prefix':None,'suffix':None,'max_token_length':None}}
    (d/'job.json').write_text(json.dumps(job,indent=2))
    env={k:v for k,v in os.environ.items() if not k.startswith(('BPE_TRACE_','TK_WORD_COUNTS_CACHE','TK_WRITE_WORD_COUNTS_CACHE','BPE_HEAP_DIAGNOSTIC'))}
    env.update(RAYON_NUM_THREADS=str(x['workers']),LC_ALL='C',RUST_BACKTRACE='full')
    command=['taskset','-c','0' if x['workers']==1 else '0-3',build['builds'][arm]['binary'],str(d/'job.json')]
    swap=rss=0;start=time.monotonic()
    print('START',name,flush=True)
    with (d/'stdout.log').open('w') as stdout,(d/'stderr.log').open('w') as stderr:
        proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
        while proc.poll() is None:
            try:
                status=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                field=lambda n:next((int(s.split()[1]) for s in status if s.startswith(n)),0)
                swap=max(swap,field('VmSwap:'));rss=max(rss,field('VmRSS:'))
            except (FileNotFoundError,ProcessLookupError):pass
            time.sleep(.025)
    row={'run':name,'case':x['case'],'scenario':x['scenario'],'workers':x['workers'],'phase':phase,'block':block,'arm':arm,'command':command,'environment':{k:env[k] for k in ('RAYON_NUM_THREADS','LC_ALL','RUST_BACKTRACE')},'returncode':proc.returncode,'max_swap_kib':swap,'sampled_max_rss_kib':rss,'subprocess_wall':time.monotonic()-start,'valid':False,'concurrent_builds_or_benchmarks':competing}
    if proc.returncode==0:
        row.update(json.loads((d/'stdout.log').read_text()));assert row['metrics']['feed_workers']==x['workers']
        counts=Path(row['output']['counts_path']);ordered=d/'counts.sorted.tsv'
        subprocess.run(['sort','-S','128M','-T',str(d),str(counts),'-o',str(ordered)],env=env,check=True)
        row.update(counts_sha256=digest(ordered),count_records=sum(1 for _ in ordered.open('rb')),model_sha256=digest(d/'model.json'),special_sha256=digest(d/'model.special.json'))
        model=json.loads((d/'model.json').read_text());special=json.loads((d/'model.special.json').read_text())
        if x['case'] not in refs:refs[x['case']]=(row['counts_sha256'],model,special)
        count_ref,model_ref,special_ref=refs[x['case']]
        row['counts_equal']=row['counts_sha256']==count_ref;row['model_equal']=model==model_ref;row['special_equal']=special==special_ref
        row['valid']=row['counts_equal'] and row['model_equal'] and row['special_equal'] and row['count_records']==row['metrics']['feed_unique_words'] and swap==0 and row['metrics']['pipeline_major_faults']==0
        if row['valid']:counts.unlink();ordered.unlink()
    rows.append(row);rp.write_text(json.dumps(rows,indent=2))
    print('DONE',name,'valid',row['valid'],'metrics',json.dumps(row.get('metrics')),flush=True)
    assert row['valid'],f'Blocked by invalid run; all input/log/output retained: {d}'
def summarize():
    summaries=[]
    for x in cases:
        cr=[a for a in rows if a['scenario']==x['scenario'] and a['phase']=='formal' and a['valid']]
        blocks=sorted({a['block'] for a in cr})
        if not blocks:continue
        result={'scenario':x['scenario'],'pairs':len(blocks),'arms':{},'paired_delta_percent':{},'paired_values':{}}
        keys=('pipeline_seconds','pipeline_cpu_seconds','process_hwm_kib_before_validation','train_seconds','train_cpu_seconds','feed_seconds','train_rss_kib')
        for arm in ('baseline','boxed'):
            ar=[a for a in cr if a['arm']==arm]
            result['arms'][arm]={k:{'values':[a['metrics'][k] for a in ar],'median':statistics.median(a['metrics'][k] for a in ar),'min':min(a['metrics'][k] for a in ar),'max':max(a['metrics'][k] for a in ar)} for k in keys}
        for k in keys:
            vals=[(next(a for a in cr if a['arm']=='boxed' and a['block']==b)['metrics'][k]/next(a for a in cr if a['arm']=='baseline' and a['block']==b)['metrics'][k]-1)*100 for b in blocks]
            result['paired_values'][k]=vals;result['paired_delta_percent'][k]=statistics.median(vals)
        summaries.append(result)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2));return summaries
for x in cases:
    for arm in ('baseline','boxed'):sample(x,'warmup',0,arm)
    for block,order in enumerate((('baseline','boxed'),('boxed','baseline'))):
        for arm in order:sample(x,'formal',block,arm)
    result=next(a for a in summarize() if a['scenario']==x['scenario']);reasons=[]
    for k in ('pipeline_seconds','pipeline_cpu_seconds'):
        v=result['paired_values'][k]
        if min(v)<0<max(v) and max(abs(n) for n in v)>1:reasons.append(k+' conflicting directions above 1%')
        if abs(statistics.median(v))>1:reasons.append(k+' absolute median above 1%')
    k='process_hwm_kib_before_validation';v=result['paired_values'][k]
    a=result['arms']['baseline'][k]['median'];b=result['arms']['boxed'][k]['median']
    if abs(statistics.median(v))>3 and abs(b-a)>32*1024:reasons.append('HWM change above 3% and 32 MiB')
    decision={'scenario':x['scenario'],'third_pair':bool(reasons),'reasons':reasons,'initial_two_pairs':result}
    (out/(x['scenario']+'-additional-decision.json')).write_text(json.dumps(decision,indent=2))
    if reasons:
        for arm in ('baseline','boxed'):sample(x,'formal',2,arm)
    summarize()
print('COMPLETE',len(rows),'runs',sum(a['phase']=='formal' for a in rows),'formal',flush=True)
