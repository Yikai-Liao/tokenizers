# Affix performance restored with original-character activation

Production commit `a583a0b4` completes the InitialTokenIds responsibility change on top of `a2b20f32`, including the earlier boundary/row commit `62263dbd`. It is pushed to simplify/bpe-flat-20261010. The original candidate report remains under ../affix-initial-ids-20261012/. Its unresolved performance gate is addressed here without restoring bare flags or changing the requested three-column representation.

Vocabulary::initialize reserves special and alphabet identities, then delegates construction to InitialTokenIds::build. InitialTokenIds owns the ordinary/decorated tables, original-order activation and cache misses. Vocabulary::intern receives text and spans, without an Affix/column protocol. The private Affix remains Prefix=0, Suffix=1, Both=2, Plain=3; DecoratedIds stores exactly three inline u32 values per contiguous heap row. Column casts remain inside its get/remember methods. Plain is handled before any table row access, its row get returns the ordinary ID, and its row remember is a no-op. Token zero stays valid and WORD_SEPARATOR_ID stays unresolved. Cache hits do not construct text; construction reuses one local String.

The final activation loop uses word.chars(), remembers whether the current character was the original first character, and checks the remaining original text for the last character. The first flag advances before the alphabet filter, so discarded leading characters do not rebase the prefix boundary. Remaining text includes discarded trailing characters, so suffixes remain tied to the original word end. This removes a numeric UTF-8 offset accumulation and a second character.len_utf8 calculation. The simple original four-state match is retained. Immutable id/symbols queries and their caller-facing signatures are unchanged.

```rust
let (vocabulary, ids) = Vocabulary::initialize(trainer, words, alphabet, progress)?;
let count = ids.symbol_count(word);
let symbols = ids.symbols(word);
let id = ids.id(character, first_in_original_word, last_in_original_word);
let spans = ids.into_spans();
```

Default and no-default runs each execute 30 passing BPE tests, excluding the same pre-existing JSON roundtrip failure. Existing cases cover four affix configurations, token-zero cache hits, single-character words, original UTF-8 first/last boundaries with cutoff, special reuse, model parity and thread parity. No extra fixture was added for the final loop. Clippy passes with only the pre-existing redundant_field_names lint allowed, and the changed file passes rustfmt. Whole-crate fmt has the previously recorded unrelated import-order issues. AST production count for vocabulary.rs is 366→396 (+30, relocation-sensitive +146/−116); the full test budget is 1406→1428 (+22), including four Miri entry lines in the count. This task did not run Miri.

Final ordinary runs are serial fresh processes using four workers pinned to CPUs 0–3, identical 256 MiB-derived prepared Chinese/English Whitespace inputs, vocab 50000, min frequency 2, alphabet cutoffs 2047/128, prefix ## and suffix </w>. Initial comparison uses current base / candidate / flags; the reverse comparison uses flags / candidate and includes no-affix controls. English changed direction, so two extra English pairs (opposite orders) were run. These 18 completed final runs were free of simultaneous compilation, tests and profiling. Every full exported model is byte-identical within each case, including IDs and ordered merges. No child process swapped.

| Normal pair | Flags init wall s | Candidate init wall s | Wall change | Flags init CPU s | Candidate init CPU s |
|---|---:|---:|---:|---:|---:|
| chars / zh-affix-w4 / 0 | 1.648693 | 1.633611 | -0.91% | 1.647246 | 1.628446 |
| chars / en-affix-w4 / 0 | 0.049548 | 0.055141 | +11.29% | 0.049472 | 0.055097 |
| chars-confirm / zh-affix-w4 / 0 | 1.699718 | 1.645431 | -3.19% | 1.696987 | 1.642640 |
| chars-confirm / en-affix-w4 / 0 | 0.051919 | 0.050298 | -3.12% | 0.051761 | 0.050250 |
| chars-en-check / en-affix-w4 / 0 | 0.050377 | 0.049583 | -1.58% | 0.050224 | 0.049517 |
| chars-en-check / en-affix-w4 / 1 | 0.051514 | 0.048796 | -5.28% | 0.051462 | 0.048752 |

| Language / aggregate of all normal pairs | Flags mean init wall s | Candidate mean init wall s | Wall change | CPU change |
|---|---:|---:|---:|---:|
| zh | 1.674206 | 1.639521 | -2.07% | -2.19% |
| en | 0.050840 | 0.050955 | +0.23% | +0.34% |

The high English primary sample (+11.29%) is retained. The other three English normal pairs are faster; all-pair mean wall/CPU differences are +0.23%/+0.34%, about 0.12/0.17 ms. Chinese mean initialization is about 2% faster. This restores the measured flags performance level without a persistent significant initialization regression; it does not establish that Enum has zero intrinsic cost. No Positions 5% allowance is used.

| Normal pair | Flags train wall s | Candidate train wall s | Flags train CPU s | Candidate train CPU s | Flags HWM MiB | Candidate HWM MiB |
|---|---:|---:|---:|---:|---:|---:|
| chars / zh-affix-w4 / 0 | 24.084 | 23.828 | 64.408 | 64.503 | 1715.5 | 1700.4 |
| chars / en-affix-w4 / 0 | 1.654 | 1.590 | 3.539 | 3.436 | 130.3 | 130.4 |
| chars-confirm / zh-affix-w4 / 0 | 24.453 | 24.116 | 65.511 | 64.723 | 1731.4 | 1717.8 |
| chars-confirm / en-affix-w4 / 0 | 1.661 | 1.595 | 3.521 | 3.423 | 130.1 | 131.0 |
| chars-en-check / en-affix-w4 / 0 | 1.633 | 1.626 | 3.425 | 3.437 | 129.5 | 129.1 |
| chars-en-check / en-affix-w4 / 1 | 1.632 | 1.643 | 3.453 | 3.459 | 130.4 | 131.3 |

Whole-train values include initialization, planning, indexing, materialization, merging and cleanup, excluding loading/export. Directions and surrounding phases vary; these small comparisons do not establish a general whole-training speedup. The current-base samples and complete phase/CPU/swap values remain in the raw JSON.

| No-affix control | Base train wall s | Candidate train wall s | Base train CPU s | Candidate train CPU s | Base HWM MiB | Candidate HWM MiB |
|---|---:|---:|---:|---:|---:|---:|
| en-plain-w4 | 1.564 | 1.541 | 3.826 | 3.760 | 129.1 | 129.2 |
| zh-plain-w4 | 20.436 | 20.174 | 57.187 | 56.752 | 1674.6 | 1678.9 |

Initialization allocation diagnostics were rebuilt for the final candidate. Counter deltas bracket Vocabulary::initialize, before snapshot serialization; processes then exit before corpus planning. Full snapshots read tokens by ID, spans, every resolved character/decorated row and configuration bits, preventing elimination of activation writes. They match the base exactly in all four affix/plain cases.

| Case | Base / candidate allocs | Reallocs | Requested bytes | Frees | Rows | Row bytes | IDs bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| en-affix-w4 | 7 / 7 | 8 | 6656010 | 1 | 128 | 12 | 64 |
| en-plain-w4 | 4 / 4 | 5 | 6648304 | 0 | 0 | 12 | 64 |
| zh-affix-w4 | 7 / 7 | 12 | 6801918 | 1 | 2047 | 12 | 64 |
| zh-plain-w4 | 4 / 4 | 9 | 6679024 | 0 | 0 | 12 | 64 |

Allocation/reallocation counts, requested bytes and frees are identical. Rows remain 12 bytes and InitialTokenIds remains 64 bytes; no-affix table bytes are zero. These are initialization totals, not complete-training allocation totals. Sampled whole-process HWM fluctuates and does not prove a memory reduction.

The discarded condition-selection attempts are retained separately: named Enum choices reduced branches but did not consistently recover initialization; grouping those choices reduced generated operations but still regressed English; checking remaining text with CharIndices recovered Chinese only. The final version additionally drops unnecessary numeric offsets and retains the simple original Enum constructor. No failed variant or switch remains in production. The standalone code-generation probe is diagnostic only; its exported numeric returns are not production API. One approximately 0.2-second probe overlapped a discarded choices-confirm screening run; those timings are not part of the final gate.

The screen-chars experiment uses the exact final normal binary but terminates each child after its initialization timer. Its favorable CPU result selected the candidate for full validation; it is not counted as a completed model/training run. All subsequent final normal data, including the higher English primary sample, are retained.

Frozen source overlays, runner locks, exact binary/input/compiler hashes, raw jobs/logs, all completed models and allocation snapshots are included. There are 44 completed normal runs here (18 final and 26 discarded-candidate samples), plus eight allocation snapshots and four explicitly terminated initialization screens. Model bytes are stored once by SHA-256; reference JSON identifies each content-addressed gzip blob. replay.py builds a frozen arm in a fresh detached a2b20f32 checkout. Replace original machine paths in archived jobs with matching prepared inputs. From the repository root, run `uv run experiments/bpe-simplification/evidence/affix-activation-20261012/count-source.py` and `uv run experiments/bpe-simplification/evidence/affix-activation-20261012/count-tests.py --base a2b20f32 --output /tmp/bpe-affix-test-counts.json` to reproduce the counts. The original old-flags diagnostic remains commit 83630bd0 on its separate experiment branch.
