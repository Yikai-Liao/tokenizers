from pathlib import Path
import shutil, subprocess

ROOT=Path('/root/code/tokenizers-simplification-results/online-initial')
WORK=Path('/root/code/tokenizers-workspaces/bpe-online-candidate-20261010')
EXP=Path('/root/code/tokenizers-workspaces/bpe-online-experiment-20261010')
BASE=Path('/root/code/tokenizers-workspaces/bpe-simplification')
REL=Path('tokenizers/tk-train/src/trainers/bpe/engine')
for file in ('positions.rs','corpus.rs'):
    shutil.copy2(EXP/REL/file,WORK/REL/file)
p=WORK/REL/'corpus.rs';s=p.read_text();a=s.index('    pub(super) fn range_slots(');b=s.index('    pub(super) fn initial_edges(',a);p.write_text(s[:a]+s[b:])
p=WORK/REL/'index.rs';s=(BASE/REL/'index.rs').read_text()
a=s.index('        let chunk = corpus');b=s.index('                let domain = corpus.small_pair_domain();',a)
s=s[:a]+'''        let pieces = corpus
            .initial_ranges(if cfg!(test) { 16 } else { 1 << 24 })
            .into_par_iter()
            .map(|range| -> Result<_> {
'''+s[b:]
s=s.replace('begin..(begin + chunk).min(corpus.word_count()),','range.clone(),',1)
s=s.replace('work.complete(chunk.min(corpus.word_count() - begin));','work.complete(range.len());',1)
s=s.replace('                Ok(match domain {','                let states = match domain {',1)
needle='''                })
            })
            .collect::<Result<Vec<_>>>()?;'''
replacement='''                };
                let mut lease = arena.lease();
                states.into_iter().map(|(pair, state)| {
                    let positions = Positions::from_sorted_owned(
                        Input::Builder(&state.positions), &mut lease)?;
                    Ok((pair, State { count: state.count, positions }))
                }).collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;'''
assert needle in s
s=s.replace(needle,replacement,1)
s=s.replace('let mut states = AHashMap::<Pair, State<Builder>>::new();','let mut states = AHashMap::<Pair, State<Vec<Positions>>>::new();',1)
s=s.replace('total.positions.append(state.positions)?;','total.positions.push(state.positions);',1)
s=s.replace('                for (pair, state) in states {','                for (pair, mut state) in states {',1)
s=s.replace('''                        let positions =
                            Positions::from_sorted(Input::Builder(&state.positions), &mut lease)?;''','''                        let positions = if state.positions.len() == 1 {
                            state.positions.pop().unwrap()
                        } else {
                            Positions::from_sorted(Input::Fragments(&state.positions), &mut lease)?
                        };''',1)
p.write_text(s)
# Reuse each existing Builder fixture to also freeze temporary fragments. This
# retains the original boundary/seek assertions and covers all concrete inputs.
p=WORK/REL/'positions.rs';s=p.read_text()
s=s.replace('''                let mut builder = Builder::default();
                for chunk in values.chunks(17) {''','''                let mut builder = Builder::default();
                let mut fragments = vec![Positions::default()];
                for chunk in values.chunks(17) {''',1)
s=s.replace('''                    builder.append(piece).unwrap();''','''                    fragments.push(Positions::from_sorted_owned(Input::Builder(&piece), &mut arena.lease()).unwrap());
                    builder.append(piece).unwrap();''',1)
a=s.index('                let fragments: Vec<_> = std::iter::once(');b=s.index('                let positions =',a)
s=s[:a]+'''                fragments.push(Positions::default());
'''+s[b:]
a=s.index('        let fragments = [\n',s.index('fn unsafe_storage_rejects'));b=s.index('        let mut builder = Builder::default();',a)
s=s[:a]+'''        for (a, b) in [(&[u64::MAX][..], &[0][..]), (&[0, u64::MAX][..], &[1][..])] {
            let fragments = [a, b].map(|v| Positions::from_sorted_owned(Input::Slice(v), &mut lease).unwrap());
            assert!(Positions::from_sorted(Input::Fragments(&fragments), &mut lease).is_err());
        }
'''+s[b:]
p.write_text(s)
shutil.copy2(BASE/'tokenizers/tk-train/Cargo.lock',WORK/'tokenizers/tk-train/Cargo.lock')
subprocess.run(['/root/.cargo/bin/cargo','fmt','--manifest-path',str(WORK/'tokenizers/tk-train/Cargo.toml')],check=True)
print('Prepared candidate without experiment flags')
