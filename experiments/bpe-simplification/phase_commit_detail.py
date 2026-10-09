"""Add coarse commit substage diagnostics after phase_timing.py.

Only isolated diagnostic worktrees are modified. Worker CPU uses thread clocks;
parallel substage wall times overlap and must not be summed as coordinator wall.
"""
from pathlib import Path
from phase_timing import TIMER,wrap
names='"commit_route", "commit_counts", "commit_completed", "commit_partial_group", "commit_partial_encode_publish", "commit_loop_mixed", "commit_partial_finalize", "commit_cleanup"'
timer=TIMER.replace('[&str; 10]', '[&str; 18]').replace('"model_output"]','"model_output", '+names+']').replace('[AtomicU64; 10]', '[AtomicU64; 18]').replace('}; 10]', '}; 18]')
timer=timer.replace('fn cpu_ns() -> u64','fn cpu_ns(thread: bool) -> u64').replace('clock_gettime(2, &mut time)','clock_gettime(if thread { 3 } else { 2 }, &mut time)').replace('cpu: cpu_ns()','cpu: cpu_ns(index >= 10)').replace('cpu_ns() - self.cpu','cpu_ns(self.index >= 10) - self.cpu')
for kind in ['main','lean']:
 e=Path('/root/code/tokenizers-workspaces/bpe-phase-'+kind+'-20261009/tokenizers/tk-train/src/trainers/bpe/engine')
 (e/'phase_timing.rs').write_text(timer)
 if kind=='main':
  p=e/'pair_index/commit.rs';s=p.read_text();s='use super::super::phase_timing;\n'+s if not s.startswith('//!') else s.replace('use super::*;','use super::*;\nuse super::super::phase_timing;')
  s=s.replace('        let policy = self.policy;','        let _route_phase = phase_timing::stage(10);\n        let policy = self.policy;',1).replace('        let result = self','        drop(_route_phase);\n        let result = self',1)
  for call,stage in [('route.group_births(',13),('shard.apply_ordered_counts(',11),('shard.publish_completed_births(',12),('shard.reduce_encode_and_publish_births(',14)]:s=wrap(s,call,stage)
  p.write_text(s)
 else:
  p=e/'index.rs';s=p.read_text();s=s.replace('use super::merge::Change;','use super::merge::Change;\nuse super::phase_timing;');s=s.replace('        let workers = self.shards.len();\n        let mut routes:', '        let _route_phase = phase_timing::stage(10);\n        let workers = self.shards.len();\n        let mut routes:',1).replace('        let reuse = self.reuse;\n        let floor = self.floor;','        drop(_route_phase);\n        let reuse = self.reuse;\n        let floor = self.floor;',1)
  s=s.replace('                for (change, remove, birth) in route {','                let _loop_phase = phase_timing::stage(15);\n                for (change, remove, birth) in route {',1).replace('                let mut candidates = Vec::new();','                drop(_loop_phase);\n                let _final_phase = phase_timing::stage(16);\n                let mut candidates = Vec::new();',1)
  p.write_text(s)
