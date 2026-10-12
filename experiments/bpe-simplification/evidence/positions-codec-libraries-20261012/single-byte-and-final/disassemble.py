from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parent;summary={}
for arm in ['hand','fmt-zero-push','fmt-small']:
 binary=r/(arm+'-bin');lines=subprocess.check_output(['nm','-n','-C',str(binary)],text=True).splitlines();summary[arm]={}
 for label,needle in [('neighbors','<tk_train::trainers::bpe::merge::Neighbors>::record'),('cursor','<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next'),('prepare','tk_train::trainers::bpe::merge::prepare_job'),('writer','leb128::write::unsigned::'),('append','tk_train::trainers::bpe::positions::codec::append_gap'),('resize','<smallvec::SmallVec<[u8; 16]>>::resize')]:
  indices=[i for i,l in enumerate(lines) if needle in l]
  if not indices:continue
  idx=indices[0];start=int(lines[idx].split()[0],16);end=next(int(l.split()[0],16) for l in lines[idx+1:] if int(l.split()[0],16)>start)
  output=subprocess.check_output(['objdump','-d','--demangle',f'--start-address={start}',f'--stop-address={end}',str(binary)],text=True);(r/f'{arm}-{label}-disassembly.txt').write_text(output)
  summary[arm][label]=dict(start=hex(start),end=hex(end),bytes=end-start,calls=[l.strip() for l in output.splitlines() if '\tcall' in l])
(r/'disassembly-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for a,g in summary.items():
 print(a,{k:v['bytes'] for k,v in g.items()})
 for k in ['neighbors','writer','cursor']:
  for call in g.get(k,{}).get('calls',[]):
   if any(n in call for n in ['leb128','smallvec','memset','memcpy']):print(k,call)
