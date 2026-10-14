from pathlib import Path
import shutil,ast
r=Path(__file__).resolve().parent;name='history-buffer-alloc';src=r/f'source-{name}';shutil.copytree(r/'source-history-buffer-detail',src,symlinks=True);runner=r/f'runner-{name}';shutil.copytree(r/'runner-history-buffer-detail',runner);p=runner/'Cargo.toml';p.write_text(p.read_text().replace('source-history-buffer-detail','source-'+name))
script=ast.parse((r/'prepare_allocation_diagnostics.py').read_text());allocator=next(ast.literal_eval(x.value) for x in script.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='allocator' for t in x.targets))
p=src/'tokenizers/tk-train/src/trainers/bpe/mod.rs';s=p.read_text();a='''    pub(super) fn timer(tag: usize) -> Timer { Timer { tag, wall: Instant::now(), cpu: cpu() } }''';b='''    fn allocation_phase(tag: usize) -> Option<usize> { match tag { 13 => Some(1), 14 => Some(2), 15 => Some(3), 16 => Some(5), 17 => Some(11), 18 => Some(9), 19 => Some(6), 9 => Some(7), 0|1|2|10 => Some(8), _ => None } }
    pub(super) fn timer(tag: usize) -> Timer { if let Some(phase) = allocation_phase(tag) { super::DIAG_ALLOC_PHASE.store(phase, Relaxed); } Timer { tag, wall: Instant::now(), cpu: cpu() } }''';assert a in s;s=s.replace(a,b,1)
a='''CALLS[self.tag].fetch_add(1, Relaxed);''';b=a+''' if allocation_phase(self.tag).is_some() { super::DIAG_ALLOC_PHASE.store(12, Relaxed); }''';assert a in s;s=s.replace(a,b,1)
a='''pub(super) fn report() { eprintln!''';b='''pub(super) fn report() { super::diagnostic_allocation_report(); eprintln!''';assert a in s;s=s.replace(a,b,1);p.write_text(s+allocator)
print(name+' ready')
