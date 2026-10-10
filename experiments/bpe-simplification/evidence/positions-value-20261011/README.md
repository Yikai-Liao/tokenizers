# Owned BPE positions and buffer ablations

`Positions` owns an immutable nondecreasing sequence. Construction no longer
requires an encoding service, scratch lease, or input wrapper. Empty, one-value,
and two-value lists use the original Enum representations; longer lists retain
the existing length/restart-directory/unsigned-LEB128-gap format in an owned Box.

The retained interface exposes `Positions::from_sorted(&values)`,
`Positions::concat(fragments)`, and iteration. Producers accumulate coordinates
in `SmallVec<[u64; 2]>` and pass a slice to construction.
`concat` consumes its fragments, discards empty fragments, returns a sole
nonempty fragment without encoding, and otherwise concatenates in order. It
preserves duplicates and rejects descending coordinates; it does not sort or
merge fragments. Existing block reads and seeking remain available.

## Construction and ownership

The constructors obtain cardinality from a buffer, slice, or fragment lengths.
They reserve the length and directory in the output Vec and write seeds/gaps
directly after that prefix, filling each directory entry when its restart is
reached. The initial capacity includes the minimum stream width; larger gaps
can grow Vec. Conversion to Box can also reallocate or copy. A single encoding
pass therefore does not imply a single allocation.

This removes `Codec`, `CodecScratch`, `Lease`, and `Input`, both temporary
encoding buffers, and the training crate's `thread_local` dependency. PairIndex,
Batch preparation, FreshSnapshot, CohortPreparation, and OwnerCommit no longer
carry an encoder or its lifetime. Cursor remains private reader state. There
are no new raw pointers, custom allocation tags, or unsafe implementations.

## Controlled variants

All variants use four workers, the same runner/profile and frozen input maps.
Source snapshots, hashes, exact commands, and build logs identify each binary.
The cached original Enum binary differs from the pre-change branch only in
`positions.rs`; the other BPE source hashes were checked against that branch.

| Arm | Mutable positions | Frozen positions | Construction |
| --- | --- | --- | --- |
| enum | Original narrow builder | Original Enum | Reusable encoder scratch, then copy |
| direct | Narrow u32/2 or wide u64/2 Buffer | Original Enum | Direct Vec, permits growth |
| u642 | SmallVec u64/2, no custom Buffer | Original Enum | Same direct encoding |
| u644 | SmallVec u64/4, no custom Buffer | Original Enum | Same direct encoding |

The SmallVec arms remove the custom container rather than retaining an alias or
wrapper. Producers supply ordered coordinates and compression validates order.
Thus this ablation measures the proposed container and API together, including
promotion/validation overhead, rather than isolating only integer width.

On the measured x86-64 build, Positions is 24 bytes and a queued Candidate is
40 bytes. The original narrow Buffer is 32 bytes; SmallVec u64/2 is 24 bytes and
SmallVec u64/4 is 40 bytes. The four-item mutable SmallVec does not change the
two-item inline representation of immutable Positions.

## Measurement protocol

The initial screen covers English and Chinese, ByteLevel-regex and Whitespace,
at 16 and 256 MiB of original text. Each case has one excluded warmup block and
two formal blocks, with reversed then forward arm order. Training uses a 50,000
vocabulary target, minimum frequency 2, no affixes, and no maximum token length.
All workers are pinned to CPUs 0–3. Builds and other benchmarks do not run during
measurement. Each completed run checks its complete model against the case
reference and records swap, source/input/binary hashes, and individual metrics.

CPU and elapsed time cover public `do_train`, excluding input loading and model
serialization. Peak memory is process high-water RSS before validation; it
includes the caller's loaded word map and earlier allocations. Original text
size is not the size of the deduplicated map. Percentage deltas are medians of
within-block ratios, not ratios of absolute medians. Two formal blocks on a
shared VM provide descriptive observations, not a general speedup guarantee.

## Narrowing at larger corpus sizes

The narrowing threshold is a coordinate value, not input bytes. CorpusPlan
counts initial symbol slots plus word separators after preprocessing and word
deduplication. Each Buffer starts narrow and promotes its complete contents once
a coordinate exceeds u32::MAX (about 4.29 billion slots). Merges retain their
original coordinates. GB-scale input does not by itself cause promotion; above
that slot boundary, buffers spanning it lose narrowing, while buffers confined
to lower coordinates can remain narrow. Final compressed storage always supports
full-u64 coordinates. The measured corpus sizes do not establish the benefit
on GB-scale, nearly nonrepeating input.

## Validation

The first direct-encoding implementation passed 14 scoped BPE tests, including
the sequential model/rule oracle,
full-u64 coordinates and seeking, parallel immutable readers, concat ownership and
duplicates, and exact restart/gap bytes when Vec grows. One existing tokenizer
encoding assertion fails identically on the untouched baseline; the user
explicitly excluded it from this work. All-target Clippy with warnings denied
passes for the direct narrow and u64/4 implementations. Capacity-only variants
are checked through release builds and complete model comparisons rather than
repeating the unit-test suite.

The user chose uniform u64 storage for the large, low-repetition target corpus,
rather than relying on narrowing below a coordinate threshold. The custom Buffer
and its promotion machinery are therefore removed. A proposed u32/4 follow-up
was cancelled before building or measuring it. The current choice is u64/2;
u64/4 remains a measured control. No further comparison was run after the user
requested stopping measurement and pushing the change.

## Completed results

The 256 MiB cases are the main comparison. The retained u64/2 arm compared
with the original Enum and encoder scratch has these paired median deltas.
Negative time deltas mean less time; positive memory deltas mean more memory.
Absolute RSS columns are formal-run medians in MiB.

| 256 MiB case | CPU | Elapsed | Peak RSS | Enum MiB | u64/2 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| English ByteLevel | -3.12% | -6.49% | +8.30% | 168.2 | 182.2 |
| English Whitespace | +6.09% | +5.10% | +17.84% | 136.2 | 160.5 |
| Chinese ByteLevel | +1.01% | +2.83% | +5.28% | 2285.1 | 2405.8 |
| Chinese Whitespace | +2.79% | +1.49% | +0.37% | 1790.4 | 1797.0 |

The controls below use the same Enum reference. Direct encoding with the narrow
Buffer isolates the encoder-service removal more closely. The four-item u64
SmallVec has no consistent timing benefit and a larger mutable object.

| 256 MiB case | Narrow direct CPU / elapsed / RSS | u64/4 CPU / elapsed / RSS |
| --- | ---: | ---: |
| English ByteLevel | -0.49% / -0.51% / +0.88% | -2.78% / -3.25% / +10.79% |
| English Whitespace | +0.12% / -1.98% / +0.84% | -0.40% / -0.43% / +15.00% |
| Chinese ByteLevel | +5.85% / +6.29% / +0.52% | +1.92% / +5.50% / +5.91% |
| Chinese Whitespace | +11.10% / +10.81% / +0.34% | +10.00% / +10.18% / +0.53% |

All 96 screen runs (32 excluded warmups and 64 formal runs) produce the same
complete model as their case reference. Four additional AA-overlap/reuse runs
compare the Enum and retained u64/2 models once per case/arm; all four match.
Those directed runs are model validation rather than repeated timing evidence.
No run records swap. The retained source hashes match the measured u64/2
snapshot exactly. No GB-scale measurement was performed.

The uniform u64 choice removes width promotion and the custom builder at the
cost of more mutable heap storage in these inputs. The decision prioritizes
the requested large, low-repetition Chinese corpus and simpler ownership; the
measurements do not demonstrate a universal speed or memory improvement.
The 16 MiB results and individual runs remain in the full JSON archive.

## Artifacts

- [Build commands, binary/source hashes and retained arm](builds.json),
  [source snapshots](source), and compiler/Cargo logs in this directory.
- [Screen protocol](screen/core-manifest.json),
  [all screen runs](screen/core-runs.json),
  [complete screen summary](screen/core-summary.json), and `screen/jobs/` logs.
- [Directed validation protocol](directed/core-manifest.json),
  [directed runs](directed/core-runs.json), and `directed/jobs/` logs.
- Per-case compressed reference models and hashes in each `models/` directory
  and `models.json`; model equality was checked as parsed complete JSON.
- [Layout output](positions-layout.txt), [probe](positions-layout.rs), and
  [layout provenance](layout-provenance.json). The probe imports the frozen
  direct/narrow control to report its Buffer alongside both library SmallVec
  layouts.

Archived logs have trailing blank lines removed for repository whitespace checks;
metric records and source snapshots are preserved.
