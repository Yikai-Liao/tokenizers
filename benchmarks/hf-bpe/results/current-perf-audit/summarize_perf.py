#!/usr/bin/env python3
"""Make weighted top-IP summaries without full callchain expansion."""
import json, re, subprocess
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
PERF=Path('/root/code/tokenizers/benchmarks/hf-bpe/.build/native-j-current-perf.perf')
RAW=HERE/'.samples-with-offset.txt'
with RAW.open('w') as out:
    subprocess.run(['perf','script','-i',str(PERF),'-G','--no-inline','--no-demangle','-F','hw:event,period,ip,dso,dsoff,sym,symoff','--max-stack','0'],check=True,stdout=out)
rows=[]; totals=defaultdict(lambda:[0,0]); symbols=defaultdict(lambda:[0,0]); unresolved=defaultdict(lambda:[0,0])
for line in RAW.read_text().splitlines():
    m=re.match(r'\s*(\d+)\s+(cycles:u|cache-misses:u):\s*(.*)',line)
    if not m: continue
    period,event,body=int(m[1]),m[2],m[3].strip()
    totals[event][0]+=1; totals[event][1]+=period
    tok=body.split(); sym=tok[1] if len(tok)>1 else '[unknown]'; sym=sym.split('+0x')[0]
    symbols[(event,sym)][0]+=1; symbols[(event,sym)][1]+=period
    if sym=='[unknown]': unresolved[event][0]+=1; unresolved[event][1]+=period
    rows.append((period,event,sym))
names=sorted({sym for (_,sym) in symbols if sym!='[unknown]'})
demangled=subprocess.run(['c++filt','-s','rust'],input='\n'.join(names)+'\n',text=True,capture_output=True,check=True).stdout.splitlines()
dem=dict(zip(names,demangled)); out={'input_perf':str(PERF),'sample_rows':len(rows),'events':{}}
for event in ('cycles:u','cache-misses:u'):
    count,period=totals[event]
    ordered=sorted(((v[1],v[0],sym) for (ev,sym),v in symbols.items() if ev==event),reverse=True)
    reserve=sorted(((v[1],v[0],sym) for (ev,sym),v in symbols.items() if ev==event and 'reserve_rehash' in sym),reverse=True)
    out['events'][event]={'samples':count,'period_total':period,'top_ip_unresolved_samples':unresolved[event][0],'top_ip_unresolved_period':unresolved[event][1],'top_ip_unresolved_pct':100*unresolved[event][1]/period if period else None,'top_symbols':[{'symbol':dem.get(sym,sym),'mangled':sym,'samples':n,'period':p,'pct':100*p/period} for p,n,sym in ordered[:25]],'reserve_rehash_total_pct':100*sum(p for p,_,_ in reserve)/period if period else None,'reserve_rehash_symbols':[{'symbol':dem.get(sym,sym),'mangled':sym,'samples':n,'period':p,'pct':100*p/period} for p,n,sym in reserve]}
(HERE/'event-summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
RAW.unlink()
print(HERE/'event-summary.json')
