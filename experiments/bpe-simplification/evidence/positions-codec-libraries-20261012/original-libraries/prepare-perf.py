from pathlib import Path
r=Path(__file__).resolve().parent;old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012')
s=(old/'perf-compare.py').read_text();s=s.replace('==48','==32').replace("[['a','b'],['b','a']]","[['hand','gimli','fmt-slice','fmt-array']]")
s=s.replace("'cycles:u,instructions:u,branches:u,branch-misses:u,cache-misses:u,task-clock'","'instructions:u,branches:u,branch-misses:u,cache-misses:u,task-clock'").replace("'-e','cycles:u','-F','199'","'-e','cpu-clock','-F','199'")
s=s.replace("p=subprocess.Popen(command,stdout=out,stderr=err)","p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)").replace('p.terminate();p.wait();raise',"os.killpg(p.pid,15);p.wait();raise")
s=s.replace("for name,extra in", "for name,extra in")
(r/'perf-compare.py').write_text(s)
