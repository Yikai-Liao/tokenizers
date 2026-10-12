from pathlib import Path
from collections import defaultdict
import re,bisect,json
r=Path(__file__).resolve().parent
for p in (r/'perf').glob('zh-bytelevel_regex-w1-r0-*'):
 text=(p/'sample-locations.txt').read_text();rows=re.findall(r'^\s*(\d+)\s+[0-9a-f]+\s+(\S+)',text,re.M);by_symbol=defaultdict(lambda:defaultdict(int));total=sum(int(n) for n,_ in rows)
 for period,location in rows:
  if '+0x' in location:
   sym,offset=location.rsplit('+0x',1);by_symbol[sym][int(offset,16)]+=int(period)
 for name in ['prepare','neighbors']:
  sym=(p/(name+'-symbol.txt')).read_text().strip();asm=(p/(name+'-disassembly.txt')).read_text();instructions=[(int(a,16),s) for a,s in re.findall(r'^\s*([0-9a-f]+):\s+(.*)',asm,re.M)];base=instructions[0][0];count=by_symbol[sym];addresses=[a for a,_ in instructions];joined=defaultdict(int)
  for offset,period in count.items():
   index=bisect.bisect_right(addresses,base+offset)-1
   if index>=0:joined[addresses[index]]+=period
  fn_total=sum(joined.values());out=[]
  for address,instruction in instructions:
   period=joined[address];out.append(f'{period/total*100:7.3f}% global {period/fn_total*100:7.3f}% local {address:x}: {instruction}')
  (p/(name+'-weighted-assembly.txt')).write_text('\n'.join(out)+'\n')
  top=sorted(zip(instructions,[joined[a] for a,_ in instructions]),key=lambda x:x[1],reverse=True)[:10]
  print(p.name,name,'top',[(round(n/fn_total*100,2),hex(a),s[:100]) for (a,s),n in top])
