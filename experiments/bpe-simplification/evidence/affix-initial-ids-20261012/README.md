# InitialTokenIds owns construction and decoration

Candidate `62263dbd` is based on `a2b20f32` (the completed Positions read refinement). It is pushed to `experiment/bpe-affix-initial-20261012` for review. The normal training branch remains at the accepted Positions implementation. This Affix candidate has not met the requested performance target; adopting its initialization cost requires a decision. The separate old-flags diagnostic is `83630bd0`, on `experiment/bpe-affix-old-flags-20261012`.

Vocabulary::initialize reserves special and alphabet identities, then calls InitialTokenIds::build. InitialTokenIds now owns construction of ordinary and decorated lookup tables, original-order activation, Affix interpretation and cache misses. Vocabulary::intern receives only token text and the existing spans buffer. Queries after construction remain immutable and do not intern.

DecoratedIds contains exactly [u32; 3]. The private Affix values Prefix=0, Suffix=1, Both=2 map directly to the three columns; Plain=3 has no column. Integer casts occur only in the row implementation. Row get handles Plain by returning the ordinary ID, and remember ignores Plain. Both activation and lookup branch on Plain before accessing the table. Empty affixes keep the empty table; zero remains a valid token ID and WORD_SEPARATOR_ID remains the unresolved sentinel. The original UTF-8 first/last coordinates, prefix-character-suffix order, cache-hit behavior and identity discovery order are retained. Construction reuses one local String, which is dropped after construction.

Callers use the existing interface without the cache protocol:

```rust
let (vocabulary, ids) = Vocabulary::initialize(trainer, words, alphabet, progress)?;
let count = ids.symbol_count(word);
for token_id in ids.symbols(word) {
    // Count or materialize the same initial token sequence.
}
let token_id = ids.id(character, first_in_original_word, last_in_original_word);
let spans = ids.into_spans();
```

Default and no-default runs each execute 30 passing BPE tests, excluding only the previously established JSON roundtrip failure. Existing cases cover four affix configurations, single-character words, UTF-8/cutoff, specials, reuse and model/thread parity. The additions check 12-byte rows, all three columns, Plain no-op, cached token zero, the empty-table Plain query and repeated lookup after activation. The full AST-based test budget is 1406→1428 (+22), including the four Miri entry lines in the count; this task did not run Miri. Production nonblank/noncomment/non-test lines in vocabulary.rs are 366→392 (+26, +142/−116 in the relocation-sensitive diff). This is a responsibility change with an explicit source-size increase. The changed file passes rustfmt; whole-crate fmt still reports unrelated pre-existing import ordering in other trainers. Clippy passes with only the pre-existing redundant_field_names lint allowed.

Normal measurements use serial fresh processes, four workers pinned to CPUs 0–3, identical 256 MiB-derived prepared Chinese/English Whitespace inputs, vocab 50000, min frequency 2 and the existing alphabet cutoffs (2047/128). Affix runs enable prefix ## and suffix </w>; controls disable both. A primary comparison was followed by reverse order because timings were noisy. Every full exported model, including token IDs and ordered merges, matches byte-for-byte within each case. No sampled child process used swap. Whole-training wall excludes loading/export; whole-training CPU is process CPU. Initialization CPU is measured across Vocabulary::initialize with getrusage, identically in all arms. Frozen diagnostic hooks are excluded from production.

| Suite / case | Base init s | Candidate init s | Flags init s | Candidate vs flags | Base train s | Candidate train s | Flags train s |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary / zh-affix-w4 | 1.734683 | 1.953356 | 1.703269 | +14.68% | 25.399 | 26.390 | 25.697 |
| primary / en-affix-w4 | 0.043144 | 0.056639 | 0.050472 | +12.22% | 1.611 | 1.589 | 1.496 |
| reverse / zh-affix-w4 | 1.768599 | 1.846832 | 1.731880 | +6.64% | 26.255 | 27.008 | 27.167 |
| reverse / en-affix-w4 | 0.043556 | 0.053588 | 0.052188 | +2.68% | 1.640 | 1.632 | 1.711 |

The candidate initialization is consistently slower than flags in the paired normal runs: Chinese +6.64% to +14.68%, English +2.68% to +12.22%. It is also slower than the current base. The Chinese difference is roughly 115–250 ms; the English difference is 1.4–6.2 ms. Whole-training directions vary with surrounding phases, so these measurements do not establish a repeatable whole-training slowdown or speedup. No-affix controls follow the unchanged plain path and also show whole-training noise. Full CPU, phase, HWM and swap data are in primary.json/reverse.json.

| Suite / case | Base CPU s | Candidate CPU s | Base HWM MiB | Candidate HWM MiB | Base train s | Candidate train s |
|---|---:|---:|---:|---:|---:|---:|
| primary / zh-affix-w4 | 67.920 | 68.915 | 1700.4 | 1712.3 | 25.399 | 26.390 |
| primary / en-affix-w4 | 3.477 | 3.372 | 130.0 | 130.7 | 1.611 | 1.589 |
| primary / zh-plain-w4 | 61.161 | 59.162 | 1689.7 | 1684.1 | 21.713 | 21.397 |
| primary / en-plain-w4 | 3.643 | 3.661 | 128.4 | 127.9 | 1.510 | 1.506 |
| reverse / zh-affix-w4 | 70.017 | 73.341 | 1745.3 | 1711.7 | 26.255 | 27.008 |
| reverse / en-affix-w4 | 3.517 | 3.389 | 130.7 | 130.5 | 1.640 | 1.632 |
| reverse / zh-plain-w4 | 63.706 | 66.287 | 1675.4 | 1672.5 | 22.670 | 23.778 |
| reverse / en-plain-w4 | 3.668 | 3.626 | 130.7 | 132.6 | 1.486 | 1.502 |

Allocation diagnostics isolate initialization. Counters are read immediately before/after Vocabulary::initialize; the snapshot serialization happens afterwards, and the process exits before corpus planning. The snapshot reads every token by ID, spans, all resolved characters, every decorated row and the three configuration bits, preserving activation writes against compiler elimination. Baseline and candidate snapshots match exactly. Diagnostic times are not used as normal performance gates.

| Case | Base / candidate allocs | Reallocs | Requested bytes | Frees | Rows | Row bytes | IDs bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| en-affix-w4 | 7 / 7 | 8 | 6656010 | 1 | 128 | 12 | 64 |
| en-plain-w4 | 4 / 4 | 5 | 6648304 | 0 | 0 | 12 | 64 |
| zh-affix-w4 | 7 / 7 | 12 | 6801918 | 1 | 2047 | 12 | 64 |
| zh-plain-w4 | 4 / 4 | 9 | 6679024 | 0 | 0 | 12 | 64 |

Initialization allocation counts, reallocations, requested bytes, frees and table layout are identical in every diagnostic pair. Rows are 12 bytes and InitialTokenIds is 64 bytes in both representations; no-affix rows/table bytes are zero. This checks initialization allocation cost, not all-training allocation totals. Normal HWM fluctuates with parallel scheduling/allocation and cannot prove a memory reduction.

An initialization perf stat diagnostic measured baseline / candidate / flags: cycles 4,948,347,701 / 5,124,962,041 / 5,388,145,960; instructions 5,663,605,531 / 5,467,468,259 / 5,554,271,239; branches 1,059,419,437 / 1,187,279,427 / 1,052,989,480; branch misses 14,659,512 / 14,609,572 / 14,804,261. FIFO enable/disable acknowledgment took about 26–34 ms, so the window is approximate. In this diagnostic the candidate initialized faster than flags, contrary to normal paired runs. More branches and fewer instructions are observations, not proof that Enum itself is slow or a complete explanation of the timing difference.

Generated-code inspection finds the original candidate helpers fully inlined and the three-column array bounds checks removed. Two bounded diagnostic trials were discarded: outlining the cache-miss insertion with cold/inline(never) did not recover flags performance; swapping the order of the two boolean conditions still regressed initialization by +3.31% Chinese / +13.76% English against flags. Frozen sources, binaries’ hashes, normal jobs and results of both trials are retained; production retains the original simple constructor and cache code. No unsafe, fourth cache column, extra persistent String or bare-flags fallback was introduced.

The preliminary pre-read comparison is retained under pre-read/ and uses the earlier Positions source. It is historical evidence, excluded from the final paired gate. A diagnostic launch initially failed protocol parsing because attempt_id was missing; no training ran on that launch. Its job/error are retained separately.

The unresolved issue is recovering the original flags initialization performance while retaining the requested responsibility boundary and representation. This report provides the completed candidate for a decision rather than silently adopting the measured cost. Prefetch implementation is queued after that decision.

All ordinary run jobs/stdout/stderr, exact full compressed models, allocation snapshots, profile counters, source overlays, lockfiles, compiler/binary/input hashes and AST count scripts are retained. Per-run reference JSON points to content-addressed gzip blobs; decompression reconstructs exact bytes. Use replay.py with a fresh detached checkout at a2b20f32 to build a frozen arm; replace the original machine input/output paths in an archived job with matching prepared data before running the binary. The build and measurement scripts retain original paths/scheduling as evidence. `uv run count-source.py` from the candidate checkout reproduces production counts; `uv run count-tests.py --base a2b20f32` reproduces the complete test budget.
