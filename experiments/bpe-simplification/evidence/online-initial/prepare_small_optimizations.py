from pathlib import Path
import subprocess,json,shutil
OUT=Path('/root/code/tokenizers-simplification-results/online-initial')
REPO=Path('/root/code/tokenizers-workspaces/bpe-simplification')
REL=Path('tokenizers/tk-train/src/trainers/bpe/engine')

def ordered(w):
 p=w/REL/'positions.rs';s=p.read_text();old='''        if let Self::Narrow(values) = self {
            if let Ok(position) = u32::try_from(position) {''';assert s.count(old)==1
 s=s.replace(old,'''        self.push_ordered(position)
    }
    // Snapshot scans visit disjoint ordered matches; callers retain that order.
    // Generic append/input paths use `push` and keep its validation.
    pub(super) fn push_ordered(&mut self, position: u64) -> Result<()> {
        if let Self::Narrow(values) = self {
            if let Ok(position) = u32::try_from(position) {''');p.write_text(s)
 p=w/REL/'merge.rs';s=p.read_text();assert s.count('group.positions.push(position as u64)?;')==1;s=s.replace('group.positions.push(position as u64)?;','group.positions.push_ordered(position as u64)?;')
 assert s.count('Self::Compact { positions, .. } => positions.push(matched.start as u64),')==1;s=s.replace('Self::Compact { positions, .. } => positions.push(matched.start as u64),','Self::Compact { positions, .. } => positions.push_ordered(matched.start as u64),');p.write_text(s)

def lookup(w):
 p=w/REL/'merge.rs';s=p.read_text();old='''        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;''';assert s.count(old)==1
 s=s.replace(old,'''        let mut heads = vec![None; corpus.id_count()];
        let mut tails = vec![false; corpus.id_count()];
        for rule in &self.rules {
            let slot = &mut heads[rule.pair.0 as usize];
            // The separator cannot be a real rule endpoint; it marks a shared head.
            *slot = Some(if slot.is_some() {
                (WORD_SEPARATOR_ID, 0)
            } else {
                (rule.pair.1, rule.replacement)
            });
            tails[rule.pair.1 as usize] = true;
        }
        let selected_id = |pair: Pair| match heads.get(pair.0 as usize).copied().flatten() {
            Some((right, id)) if right != WORD_SEPARATOR_ID => (right == pair.1).then_some(id),
            None => None,
            _ => selected.get(&pair).copied(),
        };
        let aa = self.rules[0].pair.0 == self.rules[0].pair.1;''')
 old='''                                before != 0
                                    && selected.contains_key(&(corpus.token(before - 1), prior))''';assert old in s
 s=s.replace(old,'''                                before != 0
                                    && tails[prior as usize]
                                    && selected_id((corpus.token(before - 1), prior)).is_some()''')
 old='''                            } else {
                                selected
                                    .get(&(
                                        next,
                                        corpus.token(matched.after + corpus.id_span(next)),
                                    ))
                                    .copied()
                            };''';assert old in s
 s=s.replace(old,'''                            } else if heads[next as usize].is_some() {
                                selected_id((
                                    next,
                                    corpus.token(matched.after + corpus.id_span(next)),
                                ))
                            } else {
                                None
                            };''');p.write_text(s)
 # A compatible batch with shared heads and tails exercises the exact-key fallback.
 p=w/REL/'tests/mod.rs';s=p.read_text();old='''fn overlap_floor_alias_and_alphabet_boundaries() {''';assert s.count(old)==1
 s=s.replace(old,old+'''\n    check(&trainer(), &counts(&[("abac", 4), ("dbdc", 4), ("abdb", 3), ("acdc", 3)]));''');p.write_text(s)

if __name__=='__main__':
 for arm in ('ordered','lookup','combined'):
  w=Path('/root/code/tokenizers-workspaces')/('bpe-small-'+arm+'-20261010')
  if not w.exists():
   subprocess.run(['git','worktree','add','--detach',str(w),'7e77262b'],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
  origin=Path('/root/code/tokenizers-workspaces/bpe-online-candidate-20261010')
  for p in (origin/REL).rglob('*.rs'):shutil.copy2(p,w/REL/p.relative_to(origin/REL))
  if arm in ('ordered','combined'):ordered(w)
  if arm in ('lookup','combined'):lookup(w)
  subprocess.run(['/root/.cargo/bin/cargo','fmt','--manifest-path',str(w/'tokenizers/tk-train/Cargo.toml')],check=True)
  r=subprocess.run(['python3',str(REPO/'experiments/bpe-simplification/count_lines.py'),'--root',str(w),'--max-production','2200','--max-tests','800','--output',str(OUT/'validation'/('small-'+arm+'-lines.json'))],check=True,capture_output=True,text=True)
  print(arm,r.stdout,flush=True)
  (OUT/('small-'+arm+'.patch')).write_bytes(subprocess.check_output(['git','diff','--binary','--',str(REL)],cwd=w))
