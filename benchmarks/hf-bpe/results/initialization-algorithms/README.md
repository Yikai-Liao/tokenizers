# Independent initialization algorithm prototypes

See [INITIALIZATION_ALGORITHM_RESEARCH.md](../../INITIALIZATION_ALGORITHM_RESEARCH.md) for the mathematical setup, candidate selection, measurements, and evidence limits.

`prototype` is a standalone Rust crate. All modes share line-preserving deduplication, a flat scalar-token corpus with NONE boundaries, mixed word weights, and the same contiguous output representation. It intentionally does not invoke a Trainer or its merge implementation.

Modes:

- `radix`: conventional stable 4-pass 32-bit key radix, 8-byte key/position records and equal-sized scratch.
- `radsort`: authors' BSD-2-Clause C block-reuse stable radix reference.
- `radsort-rust`: Rust translation, key in high32 and position in low32, with contiguous finalize.
- `hash2`: two exact key queries per physical edge, count then fill.
- `rank3`: membership bitmap, dense bitmap rank, exact count then fill.
- `selftest`: boundary/floor/weight/AA differential cases.

Compile and validate:

```sh
cd prototype
/root/.cargo/bin/cargo test --release --offline
/root/.cargo/bin/cargo build --release --offline
target/release/initialization-algorithm-prototype selftest -
cd ..
python3 run.py
python3 run_rust_port.py
```

Do not run these scripts alongside formal Trainer benchmark timings. Each script executes its configurations sequentially. `index_ms` excludes corpus preprocessing, checksums, and the oracle. `hwm_bytes` is read before the checksum and oracle, but includes preprocessing and allocator retained pages. `.time` CPU values cover the whole child process including its oracle. The capacity model excludes corpus, feed data, and allocator overhead.

`preprocessing-v1` preserves the first completed runs and source. Its preprocessing unnecessarily collected every character into one Vec before deduplication. The second version uses a fixed Unicode membership table. Source/binary hashes for each measurement are in the respective manifests; do not assume the latest main binary is the binary from the earlier matrix.

Reference C source: https://github.com/clausecker/radsort at `f69e816c3cd79d312cd67aea5b9cf1c338c1b371`. Files `radixsort_permuted.c` and `radixsort.h` are preserved in `prototype/vendor`. The full BSD-2-Clause license is in `prototype/vendor/COPYING`. The Rust port derives from this source; preserve its notice and license when integrating it.

Paper: Robert Clausecker and Florian Schintke, *Parallel O(√n) Overhead LSD Radix Sort*, arXiv:2607.05302v1, 2026-07-06. The fixed 512-element implementation uses O(n/512) block metadata and 2 MiB scratch, not literal O(√n) storage for all input sizes. The generalized variable-block analysis in the paper obtains the O(√n) bound.
