from pathlib import Path
import hashlib,json,os,shutil,subprocess
r=Path(__file__).resolve().parent;spec=json.loads((r/'diagnostics.json').read_text());records=[]
fixture={"schema_version":1,"pretokenizer":"none","ordered_words":[["baaba",1],["cbaaba",7],["dbaaba",0]],"hash_seeds":[11,13,17,19]}
(r/'alias-words.json').write_text(json.dumps(fixture,indent=2))
test='''
#[test]
fn storage_experiment_canonical_outputs() {
    for (case, items, target, prefix, suffix) in [
        ("empty",vec![],64,None,None),
        ("zero",vec![("ab",1)],0,None,None),
        ("no-merge",vec![("猫",0),("🙂",7)],64,None,None),
        ("alias",vec![("baaba",1),("cbaaba",7),("dbaaba",0)],64,None,Some("a")),
        ("unicode-affix",vec![("猫🙂猫",3),("🙂é",7),("猫",0)],64,Some("前"),Some("尾")),
    ] {
        let mut words=AHashMap::with_hasher(ahash::RandomState::with_seeds(11,13,17,19));
        for (word,weight) in items {words.insert(CompactString::from(word),weight);}
        let mut t=trainer();t.vocab_size=target;t.continuing_subword_prefix=prefix.map(str::to_owned);t.end_of_word_suffix=suffix.map(str::to_owned);
        t.special_tokens=vec![AddedToken::from("<s>",true).lstrip(true),AddedToken::from("aa",true)];
        for workers in [1,4,8] {
            let result=t.do_train_impl(WordCountsView::from_map(&words),Some(workers)).unwrap();
            let (vocab,merges,special)=result.parts;
            let mut vocab: Vec<_>=vocab.into_iter().collect();vocab.sort_by_key(|(_,id)|*id);
            let special: Vec<_>=special.iter().map(|s| serde_json::json!({"content":s.content,"single_word":s.single_word,"lstrip":s.lstrip,"rstrip":s.rstrip,"normalized":s.normalized,"special":s.special})).collect();
            eprintln!("BPE_MODEL {}",serde_json::json!({"case":case,"workers":workers,"vocab":vocab,"merges":merges,"special":special,"trace":result.trace}));
        }
    }
}
'''
for arm,b in spec.items():
    w=Path(b['worktree']);p=w/'tokenizers/tk-train/src/trainers/bpe/tests.rs'
    if 'fn storage_experiment_canonical_outputs()' not in p.read_text():p.write_text(p.read_text()+test)
    hashes={}
    for p in (w/'tokenizers/tk-train/src/trainers/bpe').glob('*.rs'):hashes[str(p.relative_to(w))]=hashlib.sha256(p.read_bytes()).hexdigest()
    b['source_hashes']=hashes
    cmd=['cargo','test','--manifest-path','tokenizers/tk-train/Cargo.toml','--offline','--locked','--lib','storage_experiment_canonical_outputs','--','--nocapture']
    env=dict(os.environ,CARGO_TARGET_DIR='/tmp/bpe-feed-native')
    with (r/(arm+'-diagnostic-fixtures.log')).open('w') as f:subprocess.run(cmd,cwd=w,env=env,stdout=f,stderr=f,check=True)
    for name in ('overlap_floor_alias_and_alphabet_boundaries','real_wide_ids_preserve_pair_order_and_separator_distinction'):
        directed=cmd[:];directed[directed.index('storage_experiment_canonical_outputs')]=name
        with (r/(arm+'-diagnostic-'+name+'.log')).open('w') as f:subprocess.run(directed,cwd=w,env=env,stdout=f,stderr=f,check=True)
    cmd=['cargo','build','--release','--offline','--locked','--manifest-path','experiments/bpe-simplification/runner/Cargo.toml']
    env=dict(os.environ,CARGO_TARGET_DIR='/tmp/bpe-feed-release')
    with (r/(arm+'-diagnostic-build.log')).open('w') as f:subprocess.run(cmd,cwd=w,env=env,stdout=f,stderr=f,check=True)
    shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/bpe-bench-runner',b['binary']);b['binary_sha256']=hashlib.sha256(Path(b['binary']).read_bytes()).hexdigest()
    b['build_command']=cmd
    (r/'diagnostics.json').write_text(json.dumps(spec,indent=2));print('diagnostic build',arm,flush=True)
    job={'protocol_version':1,'attempt_id':'alias-'+arm,'build_id':'diagnostic-'+arm,'input_id':'alias','mode':'core','input':str(r/'alias-words.json'),'output':str(r/('alias-'+arm+'-model.json')),'workers':1,'pretokenizer':'none','trainer':{'vocab_size':64,'min_frequency':1,'prefix':None,'suffix':'a','max_token_length':None}}
    jp=r/('alias-'+arm+'-job.json');jp.write_text(json.dumps(job,indent=2))
    cmd=['valgrind','--tool=memcheck','--trace-malloc=yes','--leak-check=no','--error-exitcode=99','--log-file='+str(r/(arm+'-allocator.log')),b['binary'],str(jp)]
    with (r/(arm+'-alias.stdout.log')).open('w') as stdout,(r/(arm+'-alias.stderr.log')).open('w') as stderr:
        p=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=dict(os.environ,RAYON_NUM_THREADS='1'))
    records.append({'arm':arm,'fixture':'alias-allocator','command':cmd,'exit_code':p.returncode});(r/'diagnostic-runs.json').write_text(json.dumps(records,indent=2))
    assert p.returncode==0
    # Two representative real corpora for capacity and phase lifetimes, not timings.
    inputs=json.loads((r/'inputs.json').read_text())
    for x,workers in ((inputs[0],4),(inputs[3],1)):
        name=x['case']+'-diag-'+arm;d=r/'diagnostic-runs'/name;d.mkdir(parents=True)
        job.update(attempt_id=name,input_id=x['case'],mode='pipeline',input=x['source'],output=str(d/'model.json'),workers=workers,pretokenizer=x['pretokenizer'],trainer={'vocab_size':50000,'min_frequency':2,'prefix':None,'suffix':None,'max_token_length':None})
        jp=d/'job.json';jp.write_text(json.dumps(job,indent=2))
        cmd=['taskset','-c','0-3' if workers==4 else '0',b['binary'],str(jp)]
        with (d/'stdout.log').open('w') as stdout,(d/'stderr.log').open('w') as stderr:
            p=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=dict(os.environ,RAYON_NUM_THREADS=str(workers)))
        records.append({'arm':arm,'fixture':x['case'],'command':cmd,'exit_code':p.returncode});(r/'diagnostic-runs.json').write_text(json.dumps(records,indent=2));assert p.returncode==0
        # Diagnostics need no duplicate large TSV retained; model remains available.
        Path(json.loads((d/'stdout.log').read_text())['output']['counts_path']).unlink()
        print('diagnostic real input',name,flush=True)
models={}
for arm in spec:
    lines=(r/(arm+'-diagnostic-fixtures.log')).read_text().splitlines()
    models[arm]=[json.loads(s.removeprefix('BPE_MODEL ')) for s in lines if s.startswith('BPE_MODEL ')]
assert len(models['baseline'])==15 and models['baseline']==models['boxed']
assert json.loads((r/'alias-baseline-model.json').read_text())==json.loads((r/'alias-boxed-model.json').read_text())
(r/'canonical-fixtures.json').write_text(json.dumps({'equal':True,'models':models['baseline']},indent=2))
print('15 full canonical fixture/trace/special results match across A/B',flush=True)
