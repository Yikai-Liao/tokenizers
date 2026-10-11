from pathlib import Path
import subprocess,re,json
r=Path(__file__).resolve().parent;b=Path('tokenizers/tk-train/src/trainers/bpe');repo=Path('/home/lyk/code/tokenizers')
up=subprocess.check_output(['git','rev-parse','upstream/main'],cwd=repo,text=True).strip();before='2f4803a936619aa575ba4f6cb49fc660057ef5e4';checks={};raw={}
def show(rev,path):return subprocess.check_output(['git','show',rev+':'+str(path)],cwd=repo,text=True)
current=(repo/b/'mod.rs').read_text();upsource=show(up,b/'mod.rs');(r/'upstream/mod.rs').write_text(upsource)
def normalized(t):return re.sub(r'\s+','',re.sub(r'//[^\n]*','',t))
def rust_body(s,a,z):return normalized(s[s.index(a):s.index(z,s.index(a))])
checks['config_matches_upstream']=rust_body(current,'struct Config {','/// A `BpeTrainerBuilder`')==rust_body(upsource,'struct Config {','/// A `BpeTrainerBuilder`')
for a,z in [('impl Default for BpeTrainerBuilder {','impl BpeTrainerBuilder {'),('impl Default for BpeTrainer {','impl BpeTrainer {')]:checks[a]=rust_body(current,a,z)==rust_body(upsource,a,z)
a='impl BpeTrainerBuilder {';z='/// In charge of training';normal=rust_body(current,a,z).replace('words:WordCounts::default(),','words:AHashMap::new(),');checks['builder_matches_upstream_except_count_storage']=normal==rust_body(upsource,a,z)
for name in ('setup_progress','emit_json_progress','finalize_progress','update_progress'):
 def fn(s):
  start=s.index('    fn '+name+'(');depth=0;beg=s.index('{',start)
  for k in range(beg,len(s)):
   depth += (s[k]=='{')-(s[k]=='}')
   if depth==0:return s[start:k+1]
 raw['upstream_progress_'+name]=fn(current)==fn(upsource)
 checks['upstream_progress_'+name]=normalized(fn(current))==normalized(fn(upsource))
for name in ('feed.rs','positions.rs','index.rs'):
 c=(repo/b/name).read_text();old=show(before,b/name)
 if name=='index.rs':c=c.split('#[cfg(test)]')[0];old=old.split('#[cfg(test)]')[0]
 checks['production_unchanged_'+name]=c==old
for name in ('parity_trainer.rs','word.rs'):checks['upstream_unchanged_'+name]=(repo/b/name).read_text()==show(up,b/name)
final=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
(r/'upstream/review.json').write_text(json.dumps({'upstream_commit':up,'before_commit':before,'final_commit':final,'checks':checks,'progress_byte_equality':raw,'progress_note':'update_progress has one preexisting blank line; Rust operations match upstream'},indent=2));assert all(checks.values())
(r/'upstream/full.diff').write_bytes(subprocess.check_output(['git','diff',up,final,'--',str(b)],cwd=repo))
(r/'upstream/a1-a3.diff').write_bytes(subprocess.check_output(['git','diff',before,final,'--',str(b)],cwd=repo))
(r/'upstream/full.stat').write_bytes(subprocess.check_output(['git','diff','--stat',up,final,'--',str(b)],cwd=repo))
print('Upstream review: all',len(checks),'checks pass',up)
