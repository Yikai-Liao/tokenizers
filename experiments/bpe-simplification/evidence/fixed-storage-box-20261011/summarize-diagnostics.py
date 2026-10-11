from pathlib import Path
import collections,hashlib,json,re
r=Path(__file__).resolve().parent
def events(path):
    return [json.loads(s[12:]) for s in path.read_text().splitlines() if s.startswith('BPE_STORAGE ')]
summary={'note':'Diagnostic times/RSS/HWM are excluded from formal comparisons. Spare bytes are requested payload, not measured RSS savings; pointer stability does not exclude in-place realloc. HWM is cumulative.', 'real':{},'allocator_fixture':{},'sizes':{}}
for arm in ('baseline','boxed'):
    for case in ('bytelevel_regex','en-whitespace'):
        es=events(r/'diagnostic-runs'/(case+'-diag-'+arm)/'stderr.log');by={}
        for name in sorted({x['name'] for x in es if x['event']=='freeze'}):
            xs=[x for x in es if x['event']=='freeze' and x['name']==name]
            by[name]={'conversions':len(xs),'len_range':[min(x['len'] for x in xs),max(x['len'] for x in xs)],'capacities':sorted({x['capacity'] for x in xs}) if len(xs)<10 else 'see raw log', 'element_bytes':xs[0]['element_bytes'],'potential_spare_bytes_sum':sum(x['potential_spare_bytes'] for x in xs),'conversion_pointer_moves':sum(x['pointer_changed'] for x in xs),'conversion_seconds_sum':sum(x['seconds'] for x in xs)}
        summary['real'][case+'-'+arm]={'candidates':by,'phases':[x for x in es if x['event']=='phase'],'weight_region_growths':[x for x in es if x['event']=='capacity_growth']}
        for x in es:
            if x['event']=='sizes':
                for name,n in x['values']:summary['sizes'].setdefault(name,{})[arm]=n
    es=events(r/(arm+'-alias.stderr.log'));log=(r/(arm+'-allocator.log')).read_text()
    realloc=[{'old_pointer':int(a,16),'new_bytes':int(n),'new_pointer':int(b,16)} for a,n,b in re.findall(r'realloc\((0x[0-9A-Fa-f]+),(\d+)\) = (0x[0-9A-Fa-f]+)',log)]
    candidate=[]
    for x in es:
        if x['event']!='freeze':continue
        matches=[a for a in realloc if a['old_pointer']==x['old_pointer'] and a['new_bytes']==x['len']*x['element_bytes']]
        candidate.append(dict(x,matching_allocator_reallocs=matches))
    summary['allocator_fixture'][arm]={'candidate_events':candidate,'candidate_shrink_reallocs':sum(bool(x['matching_allocator_reallocs']) for x in candidate),'all_process_reallocs':len(realloc),'heap_summary':re.search(r'total heap usage:.*',log).group(),'error_summary':re.search(r'ERROR SUMMARY:.*',log).group(), 'limitation':'Valgrind replaces malloc; its address moves and process allocation totals include diagnostic JSON/logging and cannot predict production allocator movement or performance.'}
(r/'diagnostic-summary.json').write_text(json.dumps(summary,indent=2))
for arm,x in summary['allocator_fixture'].items():print(arm,'candidate shrink calls',x['candidate_shrink_reallocs'])
print('descriptor sizes',summary['sizes'])
build=json.loads((r/'builds.json').read_text());bpe='tokenizers/tk-train/src/trainers/bpe/'
audits={}
for arm,b in build['builds'].items():
    w=Path(b['worktree'])
    assert all(hashlib.sha256((w/f).read_bytes()).hexdigest()==h for f,h in b['source_hashes'].items())
    audits[arm]={'source_matches_frozen_manifest':True,'atomic_ordering_counts':{n:(w/bpe/(n+'.rs')).read_text().count('Ordering::Relaxed') for n in ('corpus','index','merge')},'new_unsafe':False}
baseline=build['builds']['baseline']['source_hashes'];boxed=build['builds']['boxed']['source_hashes']
for name in ('feed','positions','word_counts','mod','tests','parity_trainer','word'):
    assert baseline[bpe+name+'.rs']==boxed[bpe+name+'.rs'],name
assert audits['baseline']['atomic_ordering_counts']==audits['boxed']['atomic_ordering_counts']
audits['unchanged_contracts']=['B1 prefix computation in place','WordCounts representation/construction/sort/release','Feed and codec','owner routing and shard inner buffers','Positions format','activation spans Vec transfer and later push','directory scratch reuse','atomic load/store orderings','joined preparation/application/publication and error/unwind protocol']
(r/'audit.json').write_text(json.dumps(audits,indent=2))
