from pathlib import Path
import shutil,subprocess,os,json,gzip,hashlib,re
r=Path(__file__).resolve().parent;w=Path('/tmp/bpe-positions-encapsulation-measurement-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe');runner=w/'experiments/bpe-simplification/runner';env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0'}
assert len(json.loads((r/'overhead.json').read_text()))==38
for arm in ['a','b']:
 shutil.copytree(r/'source'/arm/rel,w/rel,dirs_exist_ok=True)
 shutil.copytree(r/'source'/arm/'experiments/bpe-simplification/runner',runner,dirs_exist_ok=True)
 p=runner/'src/main.rs';p.write_text('mod allocation_phases;\n'+p.read_text())
 shutil.copyfile(r/'allocation-phases.rs',runner/'src/allocation_phases.rs')
 p=runner/'src/core.rs';s=p.read_text();s=s.replace('    let (vocab, merges, special) = trainer.do_train(&words)?;','    crate::allocation_phases::ACTIVE.store(true,std::sync::atomic::Ordering::Relaxed);\n    let (vocab, merges, special) = trainer.do_train(&words)?;',1);s=s.replace('    let train_seconds = begin.elapsed().as_secs_f64();','    crate::allocation_phases::report();\n    let train_seconds = begin.elapsed().as_secs_f64();',1);p.write_text(s)
 for name in ['index.rs','merge.rs','mod.rs']:
  p=w/rel/name;s=p.read_text()
  if name=='index.rs':
   s=s.replace('        let count_started = std::time::Instant::now();','        allocation_phase(0,true);\n        let count_started = std::time::Instant::now();',1)
   s=s.replace('        eprintln!("BPE_INITIAL_COUNT_SECONDS','        allocation_phase(0,false);\n        allocation_phase(1,true);\n        eprintln!("BPE_INITIAL_COUNT_SECONDS',1)
   s=s.replace('        eprintln!("BPE_INITIAL_OWNER_SECONDS','        allocation_phase(1,false);\n        eprintln!("BPE_INITIAL_OWNER_SECONDS',1)
  elif name=='merge.rs':
   for phase,key in [(2,'PREPARE'),(3,'APPLY'),(4,'COMMIT')]:
    s=s.replace('        let '+key.lower()+'_started = std::time::Instant::now();','        allocation_phase('+str(phase)+',true);\n        let '+key.lower()+'_started = std::time::Instant::now();',1)
    s=s.replace('        BENCH_'+key+'_NS.fetch_add','        allocation_phase('+str(phase)+',false);\n        BENCH_'+key+'_NS.fetch_add',1)
  else:
   s=s.replace('            index.retire();','            allocation_phase(5,true);\n            index.retire();')
   s=s.replace('            eprintln!("BPE_DROP_SECONDS','            allocation_phase(5,false);\n            eprintln!("BPE_DROP_SECONDS')
  s+='\nfn allocation_phase(phase:u8,start:bool){\n unsafe extern "C" {fn bpe_allocation_phase(phase:u8,start:bool);}\n // SAFETY: diagnostic runner defines this function with the exact ABI.\n unsafe {bpe_allocation_phase(phase,start)};\n}\n';p.write_text(s)
 snapshot=r/'source'/('allocation-'+arm);shutil.copytree(w/rel,snapshot/rel,dirs_exist_ok=True);shutil.copytree(runner,snapshot/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 command=['cargo','build','--release','--locked','--offline','--target-dir','/tmp/bpe-positions-encapsulation-release-20261012']
 with (r/('allocation-'+arm+'-build.log')).open('w') as log:subprocess.run(command,cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=r/('allocation-'+arm+'-bin');shutil.copyfile('/tmp/bpe-positions-encapsulation-release-20261012/release/bpe-bench-runner',binary);binary.chmod(0o755)
 print('built',arm,flush=True)
rows=[]
for arm in ['a','b']:
 run=r/'allocation-counts'/('zh-bytelevel_regex-w1-'+arm);run.mkdir(parents=True,exist_ok=False);reference=r/'normal/zh-bytelevel_regex-w1-r0-a';job=json.loads((reference/'job.json').read_text())|dict(attempt_id=run.name,build_id='allocation-'+arm,output=str(run/'model.json'));(run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
 print('start',run.name,flush=True)
 with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:subprocess.run(['taskset','-c','0-1',str(r/('allocation-'+arm+'-bin')),str(run/'job.json')],stdout=out,stderr=err,check=True)
 model=(run/'model.json').read_bytes();assert model==gzip.decompress((reference/'model.json.gz').read_bytes());(run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
 text=(run/'stderr.log').read_text();row=dict(arm=arm,case='zh-bytelevel_regex-w1',diagnostic_only=True,phases=[json.loads(x) for x in re.findall(r'BPE_ALLOCATION_PHASE (.+)',text)],totals=json.loads(re.search(r'BPE_ALLOCATIONS (.+)',text)[1]),binary_sha256=hashlib.sha256((r/('allocation-'+arm+'-bin')).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(model).hexdigest());rows.append(row);(r/'allocation-counts-summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(row['totals'],flush=True)
