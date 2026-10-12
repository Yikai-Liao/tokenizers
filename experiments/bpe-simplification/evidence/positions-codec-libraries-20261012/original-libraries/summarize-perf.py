from pathlib import Path
import subprocess,json,re
r=Path(__file__).resolve().parent;rows=[]
for arm in ['hand','gimli','fmt-slice','fmt-array']:
 run=r/'perf'/('zh-bytelevel_regex-w1-r0-'+arm)
 if not (run/'self-report.txt').exists():continue
 raw=subprocess.check_output(['nm','-n',str(r/(arm+'-bin'))],text=True).splitlines();dem=subprocess.check_output(['nm','-n','-C',str(r/(arm+'-bin'))],text=True).splitlines();names={}
 for a,b in zip(raw,dem):
  pa=a.split(maxsplit=2);pb=b.split(maxsplit=2)
  assert pa[:2]==pb[:2],(pa,pb)
  if len(pa)==3:names[pa[2]]=pb[2]
 metrics={}
 for line in (run/'stat.csv').read_text().splitlines():
  p=line.split(';')
  if len(p)>4:
   try:metrics[p[2]]=dict(count=float(p[0]),running_percent=float(p[4]))
   except ValueError:pass
 report=(run/'self-report.txt').read_text();tot=re.search(r'Event count \(approx\.\): (\d+)',report);cpu_ns=int(tot[1]);hot=[]
 for percent,name in re.findall(r'^\s*([0-9.]+)%\s+\[.\]\s+(\S+)',report,re.M):
  pct=float(percent);human=names.get(name,name);hot.append(dict(percent=pct,approx_cpu_seconds=cpu_ns/1e9*pct/100,symbol=human,raw_symbol=name))
 demangled=re.sub(r'_R[\w.$]+',lambda m:names.get(m[0],m[0]),report);(run/'self-report-demangled.txt').write_text(demangled)
 for filename in ['callgraph-report.txt']:
  txt=(run/filename).read_text();(run/filename.replace('.txt','-demangled.txt')).write_text(re.sub(r'_R[\w.$]+',lambda m:names.get(m[0],m[0]),txt))
 buckets={}
 for name,needle in [('writer','leb128::write::unsigned'),('append','positions::codec::append_gap'),('resize','SmallVec<[u8; 16]>>::resize'),('decoder','positions::codec::Cursor as'),('neighbors','merge::Neighbors>::record'),('memmove','memmove'),('memcpy','memcpy')]:
  matches=[x for x in hot if needle in x['symbol']];buckets[name]=dict(percent=sum(x['percent'] for x in matches),approx_cpu_seconds=sum(x['approx_cpu_seconds'] for x in matches),symbols=[x['symbol'] for x in matches])
 row=dict(arm=arm,event='cpu-clock @ 199Hz dwarf4096',self_percentages_not_source_line_attribution=True,approx_sample_cpu_seconds=cpu_ns/1e9,stat=metrics,buckets=buckets,hot=hot);rows.append(row)
 print(arm,'instructions',metrics.get('instructions:u',{}).get('count'),{k:round(v['approx_cpu_seconds'],3) for k,v in buckets.items()})
(r/'perf-findings.json').write_text(json.dumps(rows,indent=2)+'\n')
