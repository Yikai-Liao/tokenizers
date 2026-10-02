#!/usr/bin/env python3
"""Summarize frozen baseline timings and fallback counters."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/affix-analysis'
V1=ROOT/'results/affix-general-v1'
def load(name): return json.loads((DATA/(name+'.jsonl')).read_text())
def load_from(directory,name): return json.loads((directory/(name+'.jsonl')).read_text())
def row(name):
 d=load(name); s=d.get('indexed_stats') or {}; env=json.loads((DATA/(name+'.environment.json')).read_text()); return {'case':name,'input_bytes':d['input_bytes'],'train_ms':d['train_ms'],'wall_seconds':d['wall_seconds'],'requested_threads':env.get('threads_requested',env.get('requested_threads')),'reported_workers':s.get('workers'),'reported_initialization_workers':s.get('initialization_workers'),'layout':s.get('layout'),'atomic_corpus':s.get('atomic_corpus'),'initialize_ms':s.get('initialize_ms'),'merge_ms':s.get('merge_ms'),'tokenize_ms':s.get('tokenize_ms'),'posting_visits':s.get('posting_visits'),'stale_posting_visits':s.get('stale_posting_visits'),'word_scan_steps':s.get('word_scan_steps'),'cohort_scan_activations':s.get('cohort_scan_activations'),'cohort_words_scanned':s.get('cohort_words_scanned'),'monotone_pairs':s.get('monotone_pairs'),'reused_ids':s.get('reused_ids'),'actual_os_threads':d.get('max_process_threads'),'peak_rss_bytes':d.get('peak_rss_bytes'),'peak_vmswap_bytes':d.get('peak_vmswap_bytes'),'min_memavailable_bytes':d.get('min_memavailable_bytes'),'model_sha256':d['model_sha256']}

def candidate_v1_summary():
 affix_cases=['en16m-prefix','en16m-suffix','en16m-both','zh16m-prefix','zh16m-suffix','zh16m-both']
 comparisons=[]
 for name in affix_cases:
  old=load(name); new=load_from(V1,name)
  lang=name[:2]
  none1=load(f'{lang}16m-none')
  none4=load(f'{lang}16m-none-workers4')
  env_old=json.loads((DATA/(name+'.environment.json')).read_text())
  env_new=json.loads((V1/(name+'.environment.json')).read_text())
  if env_old['corpus_sha256'] != env_new['corpus_sha256']:
   raise SystemExit(f'input SHA differs for paired case {name}')
  if new['indexed_stats']['workers'] != 1 or new['indexed_stats']['initialization_workers'] != 1:
   raise SystemExit(f'candidate affix case is not the measured one-worker path: {name}')
  if none1['indexed_stats']['workers'] != 1 or none4['indexed_stats']['workers'] != 4:
   raise SystemExit(f'no-affix controls have mislabeled worker counts: {name}')
  comparisons.append({'case':name,'affix_train_ms':new['train_ms'],'affix_requested_threads':env_new['threads_requested'],'affix_workers':new['indexed_stats']['workers'],'none1_train_ms':none1['train_ms'],'none4_train_ms':none4['train_ms'],'ratio_to_none1':round(new['train_ms']/none1['train_ms'],3),'ratio_to_none4':round(new['train_ms']/none4['train_ms'],3),'input_sha256':env_new['corpus_sha256'],'model_sha256_matches_baseline':new['model_sha256']==old['model_sha256'],'baseline_model_sha256':old['model_sha256'],'candidate_model_sha256':new['model_sha256'],'candidate_peak_rss_bytes':new['peak_rss_bytes'],'arena_backing_bytes':new['indexed_stats']['posting_allocations'].get('backing_bytes')})
 aliases={}
 for name in ['alias-suffix-a-exact','alias-suffix-a-baseline4','alias-suffix-a-oracle']:
  if (V1/(name+'.jsonl')).exists(): aliases[name]=load_from(V1,name)
 timeout=json.loads((V1/'zh512m-suffix-w1-vocab50k.environment.json').read_text())
 return {'candidate_build_manifest':str(ROOT/'.build/native-affix-general-v1/build_manifest.json'),'affix_vs_noaffix':comparisons,'en_none4_pair':{name:{k:load_from(V1,name).get(k) for k in ['train_ms','wall_seconds','peak_rss_bytes','model_sha256','indexed_stats']} for name in ['en16m-none-t4-check-baseline','en16m-none-t4-check-candidate']},'alias_fixture':{name:{'model_sha256':d.get('model_sha256'),'reused_ids':(d.get('indexed_stats') or {}).get('reused_ids'),'cohort_scan_activations':(d.get('indexed_stats') or {}).get('cohort_scan_activations'),'cohort_words_scanned':(d.get('indexed_stats') or {}).get('cohort_words_scanned')} for name,d in aliases.items()},'zh512m_suffix':{'failure':timeout.get('failure'),'wall_seconds':timeout.get('wall_seconds'),'peak_rss_bytes':timeout.get('peak_rss_bytes'),'peak_vmswap_bytes':timeout.get('peak_vmswap_bytes'),'min_memavailable_bytes':timeout.get('min_memavailable_bytes'),'max_process_threads':timeout.get('max_process_threads'),'corpus_sha256':timeout.get('corpus_sha256')},'failed_wrong_input':{'case':'zh16m-prefix','moved_to':'failed-wrong-input/','reason':'wrong source corpus; model SHA mismatch was caught after the run and is excluded from all comparisons'}}
def main():
 names=[p.stem for p in DATA.glob('*.jsonl') if p.stem not in ('en16m-pua-prefix',)]
 rows=[row(n) for n in sorted(names) if n != 'en16m-none-t4' and 'model_sha256' in load(n)]
 # Same-runner one-thread no-affix comparisons are paired with each affix mode.
 groups={
 'en16m-serial': ['en16m-none','en16m-prefix','en16m-suffix','en16m-both','en16m-pua-prefix-verified','en16m-pua-both-verified'],
 'zh16m-serial': ['zh16m-none','zh16m-prefix','zh16m-suffix','zh16m-both'],
 'zh512m': ['zh512m-none-workers4-vocab50k','zh512m-suffix-w1-vocab50k'],
 }
 for k,ns in groups.items():
  present=[n for n in ns if (DATA/(n+'.jsonl')).exists() and 'model_sha256' in load(n)]
  if len(present)>=2:
   base=load(present[0])['train_ms']; groups[k]={'cases':present,'relative_to_first':{n:round(load(n)['train_ms']/base,3) for n in present}}
 out={'rows':rows,'comparison_groups':groups,'excluded_artifacts':{'en16m-none-t4':'Old artifact name implied 4 workers, but stats show workers=1; excluded from four-thread comparisons.'},'interpretation':{'workers_zero':'In the HF affix fallback, zero-valued worker fields in older baseline binaries are unpopulated IndexedTrainingStats defaults; the source directly calls the serial cohort engine. Candidate v1 records workers=1 and initialization_workers=1.','layout':'hf_cohorts means HF word-cohort semantics were retained; the ordinary compact parallel path was bypassed.','scope':'Public BpeTrainer feed + train_vocab; timed train_ms excludes feed time, wall_seconds includes harness and input feed.'}}
 (DATA/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 if V1.exists(): (V1/'analysis.json').write_text(json.dumps(candidate_v1_summary(),ensure_ascii=False,indent=2)+'\n')
 for r in rows: print(r['case'],f"train={r['train_ms']/1000:.3f}s",f"layout={r['layout']}",f"workers={r['reported_workers']}",f"OSthreads={r['actual_os_threads']}",f"RSS={r['peak_rss_bytes']/2**30:.2f}GiB",f"swap={r['peak_vmswap_bytes']}")
 print('wrote',DATA/'summary.json')
if __name__=='__main__': main()
