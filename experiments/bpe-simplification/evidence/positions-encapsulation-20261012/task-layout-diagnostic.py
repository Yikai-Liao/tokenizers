from pathlib import Path
import shutil,subprocess,os,json,gzip,hashlib,re
r=Path(__file__).resolve().parent;w=Path('/tmp/bpe-positions-encapsulation-measurement-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe');runner=w/'experiments/bpe-simplification/runner'
# Run only after all normal timings finish. This reuses the final A snapshot,
# adding counters solely to an independent diagnostic binary.
assert len(json.loads((r/'normal.json').read_text()))==48
shutil.copytree(r/'source/a'/rel,w/rel,dirs_exist_ok=True)
p=w/rel/'merge.rs';s=p.read_text().replace('let tasks = snapshot.tasks();','let tasks = snapshot.tasks();\n        observe_task_layout(&tasks);',1)
s=s.replace('pub(super) fn bench_report_prepare() {','pub(super) fn bench_report_prepare() {\n    report_task_layout();',1)
s+='''
static TASK_ITEM_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TASK_MAX_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TASK_MAX_CAPACITY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TASK_MAX_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TASK_TOTAL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn observe_task_layout<T>(tasks: &Vec<T>) {
    use std::sync::atomic::Ordering::Relaxed;
    let item_bytes = std::mem::size_of::<T>();
    TASK_ITEM_BYTES.store(item_bytes, Relaxed);
    TASK_MAX_COUNT.fetch_max(tasks.len(), Relaxed);
    TASK_MAX_CAPACITY.fetch_max(tasks.capacity(), Relaxed);
    TASK_MAX_BYTES.fetch_max(tasks.capacity() * item_bytes, Relaxed);
    TASK_TOTAL_COUNT.fetch_add(tasks.len(), Relaxed);
}
fn report_task_layout() {
    use std::sync::atomic::Ordering::Relaxed;
    eprintln!("BPE_TASK_LAYOUT {{\\"item_bytes\\":{},\\"max_tasks\\":{},\\"max_capacity\\":{},\\"max_vector_bytes\\":{},\\"total_tasks\\":{}}}", TASK_ITEM_BYTES.load(Relaxed), TASK_MAX_COUNT.load(Relaxed), TASK_MAX_CAPACITY.load(Relaxed), TASK_MAX_BYTES.load(Relaxed), TASK_TOTAL_COUNT.load(Relaxed));
}
''';p.write_text(s)
shutil.copytree(w/rel,r/'source/a-task-layout'/rel,dirs_exist_ok=True)
env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0'}
command=['cargo','build','--release','--locked','--offline','--target-dir','/tmp/bpe-positions-encapsulation-release-20261012']
with (r/'task-layout-build.log').open('w') as log:subprocess.run(command,cwd=runner,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
binary=r/'a-task-layout-bin';shutil.copyfile('/tmp/bpe-positions-encapsulation-release-20261012/release/bpe-bench-runner',binary);binary.chmod(0o755)
run=r/'task-layout';run.mkdir(exist_ok=False);reference=r/'normal/en-whitespace-w6-r0-a';job=json.loads((reference/'job.json').read_text())|dict(attempt_id='a-task-layout',build_id='a-task-layout',output=str(run/'model.json'))
(run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:subprocess.run(['taskset','-c','0-1',str(binary),str(run/'job.json')],stdout=out,stderr=err,check=True)
model=(run/'model.json').read_bytes();assert model==gzip.decompress((reference/'model.json.gz').read_bytes());(run/'model.json.gz').write_bytes(gzip.compress(model,mtime=0));(run/'model.json').unlink()
reports=[json.loads(x) for x in re.findall(r'BPE_TASK_LAYOUT (.+)',(run/'stderr.log').read_text())];assert reports and all(x['item_bytes']==112 for x in reports)
(r/'task-layout-summary.json').write_text(json.dumps({'diagnostic_only':True,'input':'en-whitespace-w6','affinity':[0,1],'reports':reports,'legacy_item_bytes':40,'b_item_bytes':128,'command':command,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(model).hexdigest()},indent=2)+'\n')
print('Task layout:',reports)
