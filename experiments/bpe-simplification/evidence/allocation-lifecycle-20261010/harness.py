from pathlib import Path
import shutil
root=Path('/tmp/bpe-allocation-harness');(root/'src').mkdir(parents=True,exist_ok=True)
(root/'Cargo.toml').write_text('''[package]
name="bpe-allocation-diagnostic-storage-check"
version="0.0.0"
edition="2024"
[dependencies]
thread_local="1.1.10"
bumpalo="3.19"
smallvec={version="1.16",features=["union"]}
itertools="0.14"
rayon="1.10"
serde={version="1",features=["derive"]}
serde_json="1"
''')
lib='''#![allow(dead_code, unused_imports)]
extern crate self as tk_encode;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
mod diagnostic;
'''
for arm in ['arena','arena-off','box-mutex','box-tls']:
    name=arm.replace('-','_')
    shutil.copyfile(Path('/tmp/bpe-allocation-variants')/arm/'tokenizers/tk-train/src/trainers/bpe/positions.rs',root/'src'/f'{name}.rs')
    lib+=f'mod {name};\n'
shutil.copyfile('/tmp/bpe-allocation-variants/box-tls/tokenizers/tk-train/src/trainers/bpe/diagnostic.rs',root/'src/diagnostic.rs')
(root/'src/lib.rs').write_text(lib)
print(root)
