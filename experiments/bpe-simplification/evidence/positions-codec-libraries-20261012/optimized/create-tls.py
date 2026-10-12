from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parent;repo=Path('/root/code/tokenizers');name='fmt-tls-push';base='55a44e7d';w=Path('/tmp/bpe-positions-codec-'+name+'-20261012');rel=Path('tokenizers/tk-train/src/trainers/bpe/positions/codec.rs');assert not w.exists()
subprocess.run(['git','worktree','add','--no-checkout','-b','experiment/bpe-positions-codec-'+name+'-20261012',str(w),base],cwd=repo,check=True)
subprocess.run(['git','sparse-checkout','init','--cone'],cwd=w,check=True);subprocess.run(['git','sparse-checkout','set','tokenizers'],cwd=w,check=True);subprocess.run(['git','read-tree','-mu','HEAD'],cwd=w,check=True)
p=w/rel;s=p.read_text();start=s.index('/// Leaf reserves');end=s.index('\npub(super) fn read_absolute',start)
s=s[:start]+'''thread_local! {
    static GAP_SCRATCH: std::cell::RefCell<[u8; 10]> = const { std::cell::RefCell::new([0; 10]) };
}

/// Leaf reserves the complete append before changing its logical contents.
#[inline(always)]
pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, gap: u64) {
    debug_assert!(bytes.capacity() - bytes.len() >= gap_width(gap));
    GAP_SCRATCH.with_borrow_mut(|scratch| {
        let mut end = 0;
        leb128fmt::encode_uint_slice::<u64, 64>(gap, scratch, &mut end)
            .expect("internal u64 gap buffer");
        for &byte in &scratch[..end] {
            bytes.push(byte);
        }
    });
}
''' + s[end:]
s=s.replace('    fn next(&mut self) -> Option<u64> {','    #[inline]\n    fn next(&mut self) -> Option<u64> {');p.write_text(s)
formatter=Path('/root/code/tokenizers-workspaces/bpe-positions-encapsulation-20261012/experiments/bpe-simplification/format-changed.py')
with (r/(name+'-format.log')).open('w') as f:subprocess.run(['uv','run','--offline',str(formatter),'--base',base],cwd=w,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['git','add',str(rel)],cwd=w,check=True);subprocess.run(['git','commit','-m','perf(bpe): test thread-local FMT scratch buffer'],cwd=w,check=True)
meta=json.loads((r/'variants.json').read_text());meta.append(dict(name=name,worktree=str(w),base=base,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),upstream_dependencies_unmodified=True));(r/'variants.json').write_text(json.dumps(meta,indent=2)+'\n')
