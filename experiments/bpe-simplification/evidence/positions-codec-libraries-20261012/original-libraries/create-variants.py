from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parent;repo=Path('/root/code/tokenizers');base='5d617efe';rel=Path('tokenizers/tk-train/src/trainers/bpe/positions/codec.rs')
old='pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, mut gap: u64) {\n    debug_assert!(bytes.capacity() - bytes.len() >= gap_width(gap));\n    while gap >= 128 {\n        bytes.push(gap as u8 | 128);\n        gap >>= 7;\n    }\n    bytes.push(gap as u8);\n}'
old_read='            let mut delta = 0u64;\n            let mut shift = 0;\n            loop {\n                let byte = self.bytes[0];\n                self.bytes = &self.bytes[1..];\n                delta |= u64::from(byte & 127) << shift;\n                if byte < 128 {\n                    break;\n                }\n                shift += 7;\n            }\n            self.position += delta;'
variants={
'gimli':"""pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, gap: u64) {
    debug_assert!(bytes.capacity() - bytes.len() >= gap_width(gap));
    leb128::write::unsigned(bytes, gap).expect("internal reserved gap buffer");
}""",
'fmt-slice':"""pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, gap: u64) {
    let width = gap_width(gap);
    debug_assert!(bytes.capacity() - bytes.len() >= width);
    let start = bytes.len();
    bytes.resize(start + width, 0);
    let mut end = start;
    leb128fmt::encode_uint_slice::<u64, 64>(gap, bytes.as_mut_slice(), &mut end)
        .expect("internal reserved gap buffer");
    debug_assert_eq!(end, bytes.len());
}""",
'fmt-array':"""pub(super) fn append_gap(bytes: &mut SmallVec<[u8; 16]>, gap: u64) {
    debug_assert!(bytes.capacity() - bytes.len() >= gap_width(gap));
    let (encoded, width) = leb128fmt::encode_u64(gap).expect("internal u64 gap");
    bytes.extend_from_slice(&encoded[..width]);
}"""
}
metadata=[]
for name,append in variants.items():
 w=Path('/tmp/bpe-positions-codec-'+name+'-20261012')
 if not w.exists():
  subprocess.run(['git','worktree','add','--no-checkout','-b','experiment/bpe-positions-codec-'+name+'-20261012',str(w),base],cwd=repo,check=True)
  subprocess.run(['git','sparse-checkout','init','--cone'],cwd=w,check=True)
  subprocess.run(['git','sparse-checkout','set','tokenizers','experiments/bpe-simplification/runner'],cwd=w,check=True)
 if not (w/rel).exists():subprocess.run(['git','read-tree','-mu','HEAD'],cwd=w,check=True)
 s=(w/rel).read_text();assert old in s and old_read in s;s=s.replace(old,append,1)
 read='            self.position += leb128::read::unsigned(&mut self.bytes)\n                .expect("internal LEB128 gap");' if name=='gimli' else '            let mut consumed = 0;\n            self.position += leb128fmt::decode_uint_slice::<u64, 64>(self.bytes, &mut consumed)\n                .expect("internal LEB128 gap");\n            self.bytes = &self.bytes[consumed..];'
 s=s.replace(old_read,read,1);(w/rel).write_text(s)
 p=w/'tokenizers/tk-train/Cargo.toml';s=p.read_text()
 if name=='gimli':s=s.replace('features = ["union"]','features = ["union", "write"]',1)
 dep='leb128 = "=0.2.7"\n' if name=='gimli' else 'leb128fmt = { version = "=0.1.0", default-features = false }\n'
 s=s.replace('slice-group-by =',dep+'slice-group-by =',1);p.write_text(s)
 metadata.append({'name':name,'worktree':str(w),'base':base,'dependency':dep.strip()})
(r/'variants.json').write_text(json.dumps(metadata,indent=2)+'\n');print('Created isolated sparse candidates')
