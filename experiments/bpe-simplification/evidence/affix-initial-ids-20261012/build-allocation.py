from pathlib import Path
import shutil,subprocess,os,hashlib,json,sys
r=Path('/root/code/tokenizers-workspaces/bpe-affix-initial-final-results-20261012');w=Path('/tmp/bpe-selected-rules-measurement-20261011');rel=Path('tokenizers/tk-train/src/trainers/bpe')
for arm in (sys.argv[1:] or ['baseline','candidate']):
 source=r/'source'/arm
 for f in (source/rel).glob('*.rs'):shutil.copyfile(f,w/rel/f.name)
 shutil.copytree(source/'experiments/bpe-simplification/runner',w/'experiments/bpe-simplification/runner',dirs_exist_ok=True)
 shutil.copyfile(source/'tokenizers/tk-train/Cargo.toml',w/'tokenizers/tk-train/Cargo.toml')
 runner=w/'experiments/bpe-simplification/runner/src'
 allocator=Path('/tmp/bpe-child-allocator.rs').read_text()+'''
#[unsafe(no_mangle)]
pub extern "C" fn bpe_init_allocation(kind: u8) -> u64 {
    match kind { 0 => ALLOC.load(Relaxed), 1 => REALLOC.load(Relaxed), 2 => ALLOC_BYTES.load(Relaxed), _ => FREE.load(Relaxed) }
}
'''
 (runner/'allocation_diagnostic.rs').write_text(allocator)
 main=runner/'main.rs';main.write_text('mod allocation_diagnostic;\n'+main.read_text())
 core=runner/'core.rs';s=core.read_text().replace('    let (vocab, merges, special) = trainer.do_train(&words)?;','    crate::allocation_diagnostic::ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);\n    let (vocab, merges, special) = trainer.do_train(&words)?;',1);core.write_text(s)
 mod=w/rel/'mod.rs';s=mod.read_text()
 needle='        let init_cpu = unsafe { bpe_init_cpu() };'
 s=s.replace(needle,'        let allocations_before = std::array::from_fn::<_, 4, _>(|kind| unsafe { bpe_init_allocation(kind as u8) });\n'+needle,1)
 needle='        eprintln!("BPE_VOCABULARY_CPU_SECONDS'
 at=s.index('\n',s.index(needle));s=s[:at]+'''\n        let allocations_after = std::array::from_fn::<_, 4, _>(|kind| unsafe { bpe_init_allocation(kind as u8) });
        eprintln!("BPE_INITIAL_ALLOCATIONS {:?}", std::array::from_fn::<_, 4, _>(|kind| allocations_after[kind] - allocations_before[kind]));
        ids.report_initial(&vocabulary);
        std::process::exit(0);'''+s[at:]
 s+='\nunsafe extern "C" { fn bpe_init_allocation(kind: u8) -> u64; }\n';mod.write_text(s)
 vocab=w/rel/'vocabulary.rs';s=vocab.read_text();row='row.0' if arm.startswith('candidate') else '*row';rowtype='DecoratedIds' if arm.startswith('candidate') else '[u32; 3]'
 s+='''
impl InitialTokenIds { pub(super) fn report_initial(&self, vocabulary: &Vocabulary) {
    eprintln!("BPE_INITIAL_LAYOUT rows={} row_bytes={} ids_bytes={} table_bytes={}", self.decorated.len(), std::mem::size_of::<'''+rowtype+'''>(), std::mem::size_of::<Self>(), std::mem::size_of_val(self.decorated.as_ref()));
    println!("{}", serde_json::json!({
        "tokens_by_id": vocabulary.tokens.iter().map(|token| token.as_str()).collect::<Vec<_>>(),
        "spans": self.spans,
        "characters": self.characters.iter().enumerate().filter(|(_, id)| **id != WORD_SEPARATOR_ID).map(|(ch,id)| (ch,*id)).collect::<Vec<_>>(),
        "decorated": self.decorated.iter().map(|row| '''+row+''').collect::<Vec<_>>(),
        "prefix": self.prefix, "suffix": self.suffix, "complete_alphabet": self.complete_alphabet
    }));
} }
''';vocab.write_text(s)
 snapshot=r/'diagnostic-source'/arm;shutil.copytree(w/rel,snapshot/rel,dirs_exist_ok=True);shutil.copytree(w/'experiments/bpe-simplification/runner',snapshot/'experiments/bpe-simplification/runner',dirs_exist_ok=True);shutil.copyfile(w/'tokenizers/tk-train/Cargo.toml',snapshot/'tokenizers/tk-train/Cargo.toml')
 cmd=['cargo','build','--release','--locked','--offline','--target-dir','/root/code/tokenizers-build/bpe-affix-core-build-20261011']
 with (r/f'{arm}-diagnostic-build.log').open('w') as log:subprocess.run(cmd,cwd=w/'experiments/bpe-simplification/runner',env=os.environ|{'PATH':'/root/.cargo/bin:'+os.environ['PATH'],'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'2'},stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=r/f'{arm}-diagnostic-bin';shutil.copyfile('/root/code/tokenizers-build/bpe-affix-core-build-20261011/release/bpe-bench-runner',binary);binary.chmod(0o755);print(arm,hashlib.sha256(binary.read_bytes()).hexdigest(),flush=True)
