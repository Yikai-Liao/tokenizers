"""Prepare pinned inputs, build one source version, and measure three full pipelines."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,platform,shutil,subprocess,sys,tomllib,time
import instrument
import measure

HERE=Path(__file__).resolve().parent
BPE='tokenizers/tk-train/src/trainers/bpe'


def command(args,*,cwd=None,env=None,log=None):
    if log:
        with Path(log).open('w') as stream:
            p=subprocess.run(args,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,text=True)
    else:
        p=subprocess.run(args,cwd=cwd,env=env,capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}): {args}; {log or p.stderr}')
    return p.stdout if not log else ''


def git(source,*args):
    return command(['git',*args],cwd=source)


def load(path):
    return json.loads(Path(path).read_text())


def point(plan,number):
    if number==0:return plan['base']
    if number==-1:return plan['original']
    return next(row for row in plan['points'] if row['number']==number)


def prepare_inputs(args,plan):
    for language,expected in plan['inputs'].items():
        out=args.shared/'inputs'/(language+str(plan['size_mib']))
        command([sys.executable,'-m','bench','corpus','--dataset',
                 str(HERE/f'wikipedia-{language}.json'),'--size-mib',str(plan['size_mib']),
                 '--out',str(out),'--cache',str(args.shared.parent/'shards')],cwd=args.protocol)
        path=out/'text.txt'
        assert path.stat().st_size==expected['bytes']
        assert measure.sha(path)==expected['sha256'],language
    (args.shared/'verified-inputs.json').write_text(json.dumps(plan['inputs'],indent=2))


def adapt_original(source,original):
    """Place frozen ordinary sources in the common current upstream crate shell."""
    git(source,'fetch','--depth','1','origin',original)
    folder=source/BPE
    (folder/'merge_changes.rs').unlink(missing_ok=True)
    for path in git(source,'ls-tree','--name-only',original,BPE+'/').splitlines():
        if path.endswith('.rs'):
            (folder/Path(path).name).write_text(git(source,'show',original+':'+path))
    mod=folder/'mod.rs'
    old='use tk_encode::utils::progress::{ProgressBar, ProgressFormat, ProgressStyle};'
    text=mod.read_text();assert text.count(old)==1
    mod.write_text(text.replace(old,'use crate::progress::{ProgressBar, ProgressFormat, ProgressStyle};'))
    vocab=folder/'vocabulary.rs';text=vocab.read_text()
    old='    utils::progress::ProgressBar,\n';assert text.count(old)==1
    text=text.replace(old,'');end=text.index('\n')+1
    vocab.write_text(text[:end]+'use crate::progress::ProgressBar;\n'+text[end:])
    index=folder/'index.rs';text=index.read_text()
    old='use tk_encode::{Result, models::bpe::Pair, parallelism::*, utils::progress::ProgressBar};'
    assert text.count(old)==1
    index.write_text(text.replace(old,'use crate::progress::ProgressBar;\nuse tk_encode::{Result, models::bpe::Pair, parallelism::*};'))


def package_versions(path):
    return {(p['name'],p['version'],p.get('source','local'))
            for p in tomllib.loads(path.read_text())['package']}


def build(args,plan,row,env):
    source=args.source.resolve()
    checkout=row.get('checkout_head',row['head'])
    assert git(source,'rev-parse','HEAD').strip()==checkout
    expected_tree=row.get('tree')
    if expected_tree:assert git(source,'rev-parse','HEAD^{tree}').strip()==expected_tree
    dest=args.out/'builds'/f"{row['number']:02d}"
    dest.mkdir(parents=True)
    if row['number']==-1:adapt_original(source,row['head'])
    git(source,'add','-A',BPE)
    source_tree=git(source,'write-tree').strip()
    (dest/'source.patch').write_text(git(source,'diff','HEAD','--binary'))
    source_hashes={p.name:measure.sha(p) for p in sorted((source/BPE).glob('*.rs'))}
    instrument.BENCH=source
    instrument.instrument()
    runner=args.work/'runner'
    shutil.copytree(HERE/'runner/src',runner/'src')
    (runner/'Cargo.toml').write_text((HERE/'runner/Cargo.toml.in').read_text().replace('@SOURCE@',str(source)))
    shutil.copy2(HERE/'runner/Cargo.lock',runner/'Cargo.lock')
    metadata=command(['cargo','metadata','--format-version','1','--manifest-path',str(runner/'Cargo.toml')],env=env)
    (dest/'dependency-metadata.json').write_text(metadata)
    actual=package_versions(runner/'Cargo.lock');canonical=package_versions(HERE/'runner/Cargo.lock')
    assert actual<=canonical,sorted(actual-canonical)
    shutil.copy2(runner/'Cargo.lock',dest/'Cargo.lock')
    command(['cargo','build','--locked','--release','--manifest-path',str(runner/'Cargo.toml')],env=env,log=dest/'build.log')
    binary=dest/'bpe-bench-runner'
    shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/bpe-bench-runner',binary)
    git(source,'add','-A',BPE,'tokenizers/tk-train/Cargo.toml')
    build_tree=git(source,'write-tree').strip()
    (dest/'instrumentation.patch').write_text(git(source,'diff','HEAD','--binary'))
    shutil.copy2(source/BPE/'review_phase.rs',dest/'review_phase.rs')
    manifest={'number':row['number'],'head':row['head'],'checkout_head':checkout,
              'source_tree':source_tree,'build_tree':build_tree,'source_files_before_instrumentation':source_hashes,
              'binary_source_head':row['head'],'binary_sha256':measure.sha(binary),
              'source_patch_sha256':measure.sha(dest/'source.patch'),
              'patch_sha256':measure.sha(dest/'instrumentation.patch'),
              'observer_sha256':measure.sha(dest/'review_phase.rs'),
              'canonical_dependency_lock_sha256':measure.sha(HERE/'runner/Cargo.lock'),
              'actual_dependency_lock_sha256':measure.sha(dest/'Cargo.lock'),
              'actual_package_versions':sorted(actual),'canonical_packages_not_used':sorted(canonical-actual),
              'rustc':command(['rustc','-Vv'],env=env),'profile':{k:v for k,v in env.items() if k.startswith('CARGO_PROFILE_RELEASE_')},
              'runner_sources':{str(p.relative_to(runner)):measure.sha(p) for p in [*sorted((runner/'src').rglob('*.rs')),runner/'Cargo.toml']},
              'workers':4,'control_sha':os.environ.get('GITHUB_SHA'),
              'adaptations':['Progress namespace imports only; frozen ordinary source in current common dependency shell.'] if row['number']==-1 else []}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return binary,manifest


def run_point(args,plan):
    assert 1<=args.rounds<=9
    row=point(plan,args.number)
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'results.jsonl').exists(),'Use a fresh output directory for a new measurement job.'
    args.work.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CARGO_INCREMENTAL='0',CARGO_BUILD_JOBS='2',CARGO_TARGET_DIR=str(args.work/'target'),
             RAYON_NUM_THREADS='4',TOKENIZERS_PARALLELISM='true',RUSTUP_TOOLCHAIN=plan['rust_toolchain'],
             CARGO_PROFILE_RELEASE_OPT_LEVEL='3',CARGO_PROFILE_RELEASE_LTO='fat',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='1')
    env.pop('RUSTFLAGS',None);env.pop('CARGO_ENCODED_RUSTFLAGS',None)
    inputs={l:(args.shared/'inputs'/v['path']).resolve() for l,v in plan['inputs'].items()}
    for l,p in inputs.items():
        assert p.stat().st_size==plan['inputs'][l]['bytes'] and measure.sha(p)==plan['inputs'][l]['sha256']
    binary,manifest=build(args,plan,row,env)
    # Finish compilation before warming and verifying the timed input files.
    for language,path in inputs.items():
        assert measure.sha(path)==plan['inputs'][language]['sha256']
    allowed=sorted(os.sched_getaffinity(0));assert len(allowed)>=4,allowed
    cpus=allowed[:4];observer=allowed[4:5] or allowed
    os.sched_setaffinity(0,observer)
    environment={'unix_ns':time.time_ns(),'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text(),
                 'meminfo':Path('/proc/meminfo').read_text(),'swaps':Path('/proc/swaps').read_text(),
                 'runner_name':os.environ.get('RUNNER_NAME'),'runner_os':os.environ.get('RUNNER_OS'),
                 'runner_arch':os.environ.get('RUNNER_ARCH'),'image_version':os.environ.get('ImageVersion'),
                 'allowed_cpus':allowed,'benchmark_cpus':cpus,'observer_cpus':observer,
                 'run_id':os.environ.get('GITHUB_RUN_ID'),'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT')}
    (args.out/'environment.json').write_text(json.dumps(environment,indent=2))
    measure.OUT=args.out;measure.CORPORA=inputs;measure.EXPECTED={l:v['sha256'] for l,v in plan['inputs'].items()}
    measure.ENV=env;measure.CPUS=cpus;measure.SIZE_MIB=plan['size_mib']
    if args.number!=0:
        shutil.copytree(args.shared/'reference',args.out/'reference')
    records=[]
    attempts=['main','repeat']+[f'round-{i}' for i in range(3,args.rounds+1)]
    for i,attempt in enumerate(attempts[:args.rounds]):
        languages=('en','zh') if (i+args.number)%2==0 else ('zh','en')
        for language in languages:
            raw=measure.run(args.number,row['head'],binary,language,attempt)
            dest=Path(raw['path']);(dest/'measured-summary.json').write_text(json.dumps(raw,indent=2))
            result={**raw,'source_tree':manifest['source_tree'],'measured_source_tree':manifest['source_tree'],
                    'measured_head':row['head'],'binary_source_head':row['head'],'job_head':row['head'],
                    'binary_sha256':manifest['binary_sha256'],'size_mib':plan['size_mib'],
                    'artifact_relative_run_path':str(dest.relative_to(args.out)),
                    'measurement_start_unix_ns':load(dest/'host-before.json')['unix_ns'],
                    'ci_run_id':environment['run_id'],'ci_run_attempt':environment['run_attempt'],
                    'canonical_dependency_lock_sha256':manifest['canonical_dependency_lock_sha256']}
            assert result['model_equal_baseline'] and result['process']['exit_code']==0
            records.append(result)
            (dest/'summary.json').write_text(json.dumps(result,indent=2))
            (args.out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    if args.number==0:shutil.copytree(args.out/'reference',args.shared/'reference',dirs_exist_ok=True)
    (args.out/'completion.json').write_text(json.dumps({'number':args.number,'head':row['head'],'rounds':args.rounds,
        'languages':['en','zh'],'runs':len(records),'size_mib':plan['size_mib'],'workers':4,
        'phase_observers':True,'run_id':environment['run_id'],'run_attempt':environment['run_attempt']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=HERE/'plan.json')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--protocol',type=Path,required=True);q.add_argument('--shared',type=Path,required=True)
    sub.add_parser('matrix')
    q=sub.add_parser('run');q.add_argument('--number',type=int,required=True);q.add_argument('--source',type=Path,required=True)
    q.add_argument('--shared',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--work',type=Path,required=True);q.add_argument('--rounds',type=int,default=3)
    args=p.parse_args();plan=load(args.plan)
    if args.command=='prepare':prepare_inputs(args,plan)
    elif args.command=='matrix':
        rows=[*plan['points'],plan['original']]
        matrix={'include':[{'number':r['number'],'head':r.get('checkout_head',r['head']),'label':r['label']} for r in rows]}
        print(json.dumps(matrix,separators=(',',':')))
    else:run_point(args,plan)


if __name__=='__main__':main()
