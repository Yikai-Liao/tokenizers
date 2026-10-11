from pathlib import Path
import json,os,select,subprocess,time
r=Path('/root/code/tokenizers-workspaces/bpe-affix-initial-final-results-20261012');rows=[]
for arm in ['baseline','candidate','flags']:
 run=r/'profile-init'/arm;run.mkdir(parents=True,exist_ok=False)
 job=json.loads(Path('/root/code/tokenizers-workspaces/bpe-affix-initial-results-20261012/primary/zh-affix-w4-r0-baseline/job.json').read_text());job=job|{'attempt_id':run.name,'build_id':arm,'output':str(run/'model.json')};(run/'job.json').write_text(json.dumps(job,indent=2)+'\n')
 control_path=run/'control.fifo';ack_path=run/'ack.fifo';os.mkfifo(control_path);os.mkfifo(ack_path);control=os.open(control_path,os.O_RDWR);ack=os.open(ack_path,os.O_RDWR);commands=[]
 command=['perf','stat','--all-user','-e','cycles,instructions,branches,branch-misses','-D','-1','--control',f'fifo:{control_path},{ack_path}','-x','\t','-o',str(run/'counters.tsv'),'--','taskset','-c','0-3',str(r/f'{arm}-bin'),str(run/'job.json')]
 (run/'command.json').write_text(json.dumps(command,indent=2)+'\n')
 with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
  proc=subprocess.Popen(command,stdout=out,stderr=subprocess.PIPE,text=True,bufsize=1)
  for line in proc.stderr:
   err.write(line);action=None
   if line.startswith('BPE_SELECTION_SECONDS'):action='enable'
   elif line.startswith('BPE_VOCABULARY_CPU_SECONDS'):action='disable'
   if action:
    sent=time.monotonic();os.write(control,(action+'\n').encode());assert select.select([ack],[],[],3)[0];response=os.read(ack,4096).decode();commands.append({'action':action,'sent':sent,'acknowledged':time.monotonic(),'response':response})
  proc.wait()
 os.close(control);os.close(ack);control_path.unlink();ack_path.unlink();assert proc.returncode==0;assert [x['action'] for x in commands]==['enable','disable']
 row={'arm':arm,'command':command,'control_times':commands,'raw_counters':(run/'counters.tsv').read_text()};rows.append(row);(r/'profile-init.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
