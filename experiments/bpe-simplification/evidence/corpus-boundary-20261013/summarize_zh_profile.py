import subprocess,json,collections,gzip,shutil,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEMANGLES={}
def role(s):
    if 'bridge_producer_consumer' in s:
        if 'Batch>::prepare' in s or 'Corpus>::prepare_fresh' in s:return 'fresh_prepare_helper'
        if 'PairIndex>::build' in s:return 'initial_count_helper'
        if 'PairIndex>::commit' in s:return 'index_commit_helper'
        if 'BpeTrainer>::merge_next' in s or 'Corpus>::merge' in s:return 'endpoint_write_helper'
    if 'Neighbors>::record' in s or 'NeighborChanges>::record' in s:return 'neighbor_record'
    if 'Neighbors>::group' in s or 'NeighborChanges>::group' in s:return 'neighbor_group'
    if 'Neighbors>::finish' in s or 'NeighborChanges>::finish' in s:return 'neighbor_finish'
    if 'sort' in s and 'Shard>::commit' in s:return 'birth_fragment_sort'
    if 'BpeTrainer>::merge_next' in s:return 'merge_next_remaining'
    if 'max_sibling' in s:return 'queue_max_sibling'
    if 'InitialTokenIds>::symbols' in s:return 'initial_symbol_iterator'
    if 'serde_json' in s:return 'json_load_export'
    if 'CorpusPlan>::materialize' in s:return 'corpus_materialize'
    if 'Cursor as' in s and 'Iterator>::next' in s:return 'position_decoder'
    return 'other'
def summarize():
    rows=[]
    for d in sorted((HERE/'zh-aligned-profile').glob('record-*')):
        periods=collections.Counter()
        with tempfile.TemporaryDirectory(prefix='bpe-perf-replay-') as scratch:
            data = d/'perf.data'
            if not data.exists():
                data = Path(scratch)/'perf.data'
                with gzip.open(d/'perf.data.gz', 'rb') as source, data.open('wb') as target:
                    shutil.copyfileobj(source, target)
            output=subprocess.check_output(['perf','script','-i',str(data),'-F','period,ip,sym','-G'],text=True)
        for line in output.splitlines():
            fields=line.split(maxsplit=2)
            if len(fields)!=3:continue
            try:period=int(fields[0]);int(fields[1],16)
            except ValueError:continue
            raw=fields[2].split()[0].split('+')[0]
            if raw not in DEMANGLES:DEMANGLES[raw]=subprocess.check_output(['c++filt','-s','rust',raw],text=True).strip()
            periods[role(DEMANGLES[raw])]+=period
        result=json.loads((d/'result.json').read_text())
        rows.append({'arm':result['arm'],'repeat':result['repeat'],'sampled_cycle_periods':dict(periods),'train_cpu_seconds':result['output']['metrics']['train_cpu_seconds'],'train_seconds':result['output']['metrics']['train_seconds'],'total_sampled_cycle_period':sum(periods.values()),'source':str(d)})
    totals={arm:collections.Counter() for arm in ['baseline','candidate']}
    for row in rows:totals[row['arm']].update(row['sampled_cycle_periods'])
    compare=[]
    for name in sorted(totals['baseline']):
        old=totals['baseline'][name];new=totals['candidate'][name]
        compare.append({'role':name,'baseline_sum_sampled_cycles':old,'candidate_sum_sampled_cycles':new,'change_cycles':new-old,'change_percent':100*(new/old-1) if old else None})
    return {'boundary':'whole-process cycles:P sampling; input load/export included; two alternating pairs; sampling cycles are diagnostic, not core timer','runs':rows,'aggregate_difference':compare}
if __name__=='__main__':
    result=summarize();(HERE/'zh-profile-difference.json').write_text(json.dumps(result,indent=2))
    for row in result['aggregate_difference']:
        print(row['role'],round(row['change_cycles']/1e6,3),round(row['change_percent'],3) if row['change_percent'] is not None else None)
