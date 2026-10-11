# Positions: appendable leaves and fixed roots

Kept the 32-byte child implementation in production commit `ea1d4d56`, based on `02620e0a`. On 2026-10-11 the user explicitly accepted the English single-worker cost and requested this implementation be retained and finished. Thus this is an accepted exception to the original per-scenario 5% gate, not a claim that every scenario passed that gate. Chinese ByteLevel single-worker round 0 was +5.03%; its reversed pair was +3.04%.

## Representation and deleted protocol

Callers append directly to `Positions` and publish immutable occurrence lists. A private leaf uses SmallVec<[u8;16]> plus NonZeroU64 len+1; roots own Box<[Leaf]> plus cached length. The actual x86-64 default and no-default tests assert a 32-byte Positions. Empty leaves, repeated coordinates, and the full u64 coordinate domain remain supported. Root inputs to concat are rejected before empty filtering, even a sole root. Heap compressed payloads move without re-encoding; inline payloads and root descriptors are moved.

Leaves keep an eight-byte last-value cache in their byte storage, followed by the geometric restart directory and absolute seeds / LEB128 gaps. Append validates order and arithmetic before mutation, reserves once, then writes directly. Restart handling remains a separate cold function; the hot push calls inline. The 1.5x growth factor remains unchanged at user request. Restart directory growth moves only the data area when directory capacity crosses a geometric boundary, giving amortized linear copy volume.

Deleted Birth Complete/Partial, Source::complete, Neighbors.complete/rank, producer floor plumbing, initial raw-coordinate conversion, owner raw-u64 accumulation, Complete-first publication and final bulk compression. PairIndex.floor, checked weights, removal-before-birth, reuse signed cohorts, rule task rank, AA start matching and prepare/write/owner joins remain. Fresh fragments concatenate in proven spatial order; reuse merges ordered borrowed streams directly into a leaf. Read tasks start at a leaf/restart boundary and split a large leaf; no task scans the root prefix from the beginning. Index retirement explicitly distributes owning entries through the existing Rayon pool without a new per-leaf Drop protocol.

## Final ordinary release comparison

The input text sources are 256 MiB each. Core mode trains from identical prepared word-count files; train_seconds includes selection, vocabulary, plan, index, materialization, merge and retirement, excluding load/validation. vocab_size=50000, min_frequency=2, no affix; workers are 1 or 4, affinity CPUs 0–3. Each pair runs serially. Opposite orders were measured where near the threshold or noisy. Whole serialized model bytes (vocabulary by ID and ordered merges) match in all paired runs. No measured child had swap; global swap counters can change because of unrelated processes. English runs are short and direction varies with four workers; no fixed speedup is claimed.

| Case | Round | Base wall s | Child wall s | Change | Base CPU s | Child CPU s | Base HWM MiB | Child HWM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en-bytelevel_regex-w1 | 0 | 2.437 | 2.655 | +8.92% | 2.431 | 2.631 | 170.7 | 139.6 |
| en-bytelevel_regex-w1 | 1 | 2.227 | 2.485 | +11.62% | 2.213 | 2.483 | 169.4 | 137.8 |
| en-bytelevel_regex-w4 | 0 | 1.731 | 1.686 | -2.60% | 4.145 | 4.064 | 164.1 | 159.2 |
| en-bytelevel_regex-w4 | 1 | 1.635 | 1.696 | +3.69% | 3.847 | 4.088 | 163.9 | 158.9 |
| en-whitespace-w1 | 0 | 1.899 | 2.168 | +14.17% | 1.889 | 2.167 | 144.4 | 121.8 |
| en-whitespace-w1 | 1 | 1.836 | 2.138 | +16.44% | 1.818 | 2.135 | 142.2 | 123.4 |
| en-whitespace-w4 | 0 | 1.583 | 1.490 | -5.89% | 3.689 | 3.531 | 154.0 | 132.4 |
| en-whitespace-w4 | 1 | 1.439 | 1.507 | +4.72% | 3.505 | 3.600 | 165.2 | 134.5 |
| zh-bytelevel_regex-w1 | 0 | 69.633 | 73.134 | +5.03% | 69.526 | 72.983 | 2271.6 | 2508.1 |
| zh-bytelevel_regex-w1 | 1 | 70.941 | 73.098 | +3.04% | 70.786 | 72.948 | 2279.9 | 2509.2 |
| zh-bytelevel_regex-w4 | 0 | 22.695 | 22.540 | -0.68% | 70.781 | 72.555 | 2391.6 | 2521.2 |
| zh-bytelevel_regex-w4 | 1 | 22.550 | 23.545 | +4.41% | 70.752 | 75.299 | 2436.2 | 2535.9 |
| zh-whitespace-w1 | 0 | 61.064 | 59.925 | -1.87% | 60.886 | 59.798 | 1700.5 | 1745.8 |
| zh-whitespace-w4 | 0 | 23.904 | 20.685 | -13.47% | 64.752 | 59.457 | 1792.3 | 1749.9 |
| zh-whitespace-w4 | 1 | 22.102 | 21.844 | -1.17% | 60.414 | 62.727 | 1837.9 | 1752.7 |

English one-worker Whitespace costs about 0.26–0.30 s (+14.17% / +16.44%); ByteLevel costs about 0.22–0.27 s (+8.93% / +11.62%). The stable increase is primarily merge work, not input IO. Chinese four-worker Whitespace is −13.47% / −1.17%, ByteLevel −0.68% / +4.41%; these ranges show machine variability. Chinese one-worker index improves, but ByteLevel merge and Whitespace cleanup grow. Memory is not unchanged: Chinese four-worker ByteLevel HWM grows about 4–5%; the structural retained-memory increase is detailed below.

### Phase evidence

| Case / round | Arm | Selection+vocab+plan s | Index s | Materialize s | Merge s | Drop s |
|---|---|---:|---:|---:|---:|---:|
| en-bytelevel_regex-w1 / 0 | baseline | 0.179 | 0.415 | 0.150 | 1.616 | 0.051 |
| en-bytelevel_regex-w1 / 0 | child | 0.172 | 0.367 | 0.149 | 1.865 | 0.081 |
| en-bytelevel_regex-w1 / 1 | baseline | 0.148 | 0.396 | 0.142 | 1.467 | 0.051 |
| en-bytelevel_regex-w1 / 1 | child | 0.152 | 0.373 | 0.141 | 1.724 | 0.073 |
| en-bytelevel_regex-w4 / 0 | baseline | 0.083 | 0.400 | 0.044 | 1.148 | 0.031 |
| en-bytelevel_regex-w4 / 0 | child | 0.095 | 0.377 | 0.053 | 1.102 | 0.036 |
| en-bytelevel_regex-w4 / 1 | baseline | 0.086 | 0.398 | 0.056 | 1.040 | 0.033 |
| en-bytelevel_regex-w4 / 1 | child | 0.094 | 0.373 | 0.056 | 1.113 | 0.034 |
| en-whitespace-w1 / 0 | baseline | 0.142 | 0.356 | 0.112 | 1.228 | 0.037 |
| en-whitespace-w1 / 0 | child | 0.147 | 0.385 | 0.114 | 1.435 | 0.061 |
| en-whitespace-w1 / 1 | baseline | 0.134 | 0.377 | 0.107 | 1.162 | 0.033 |
| en-whitespace-w1 / 1 | child | 0.125 | 0.369 | 0.111 | 1.445 | 0.064 |
| en-whitespace-w4 / 0 | baseline | 0.086 | 0.367 | 0.035 | 1.047 | 0.024 |
| en-whitespace-w4 / 0 | child | 0.079 | 0.401 | 0.038 | 0.927 | 0.025 |
| en-whitespace-w4 / 1 | baseline | 0.074 | 0.338 | 0.034 | 0.946 | 0.026 |
| en-whitespace-w4 / 1 | child | 0.078 | 0.373 | 0.035 | 0.975 | 0.025 |
| zh-bytelevel_regex-w1 / 0 | baseline | 4.242 | 15.359 | 4.976 | 44.123 | 0.909 |
| zh-bytelevel_regex-w1 / 0 | child | 4.667 | 12.007 | 4.866 | 50.171 | 1.398 |
| zh-bytelevel_regex-w1 / 1 | baseline | 4.529 | 15.716 | 4.951 | 44.768 | 0.952 |
| zh-bytelevel_regex-w1 / 1 | child | 4.673 | 11.735 | 5.116 | 50.215 | 1.335 |
| zh-bytelevel_regex-w4 / 0 | baseline | 1.203 | 4.261 | 1.388 | 14.561 | 1.258 |
| zh-bytelevel_regex-w4 / 0 | child | 1.262 | 3.157 | 1.450 | 15.277 | 1.370 |
| zh-bytelevel_regex-w4 / 1 | baseline | 1.217 | 4.229 | 1.443 | 14.341 | 1.298 |
| zh-bytelevel_regex-w4 / 1 | child | 1.314 | 3.133 | 1.683 | 15.834 | 1.557 |
| zh-whitespace-w1 / 0 | baseline | 2.769 | 37.243 | 2.614 | 16.702 | 1.718 |
| zh-whitespace-w1 / 0 | child | 2.856 | 32.505 | 2.344 | 17.522 | 4.680 |
| zh-whitespace-w4 / 0 | baseline | 0.908 | 12.568 | 0.712 | 7.629 | 2.067 |
| zh-whitespace-w4 / 0 | child | 0.969 | 10.987 | 0.712 | 6.469 | 1.529 |
| zh-whitespace-w4 / 1 | baseline | 0.839 | 12.138 | 0.616 | 6.748 | 1.741 |
| zh-whitespace-w4 / 1 | child | 1.072 | 11.297 | 0.667 | 6.714 | 2.073 |

## Separate allocation and layout diagnostics

Instrumentation binaries are separate from ordinary timed binaries. TLS timers sum worker spans and are not a normal training gate. Global allocator totals cover the entire train region, not just Positions. Retention snapshots count currently owning position containers; payload means used bytes, retained heap means capacity, and root child arrays are separately counted. Baseline singleton/pair inline payload is normalized to its actual logical 8/16 bytes. Initial build State<Positions> is 40 bytes versus 32, and an initial aggregation State<SmallVec<[Positions;1]>> is 48 versus 32; this is a temporary map entry, not the persistent Box root layout. Candidate is 48 versus 40. Fresh owner commit uses borrowed linear metadata, avoiding that temporary fragment map.

| Scene / arm | Initial roots | Initial leaves | Initial heap MB | Final roots | Final leaves | Final payload MB | Final retained heap MB | Final root child-array MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zh-bytelevel_regex-w4-r0-baseline-final-diagnostic | 0 | 15049 | 373.8 | 0 | 6958865 | 613.4 | 572.1 | 0.0 |
| zh-bytelevel_regex-w4-r0-child-inline-fast-final-diagnostic | 14695 | 149214 | 466.9 | 31459 | 7066103 | 630.0 | 758.8 | 4.4 |
| zh-whitespace-w4-r0-baseline-final-diagnostic | 0 | 1721017 | 194.6 | 0 | 6453766 | 292.1 | 253.1 | 0.0 |
| zh-whitespace-w4-r0-child-inline-fast-final-diagnostic | 1574108 | 5642885 | 261.0 | 1608159 | 10348204 | 349.7 | 371.2 | 176.1 |

For the child, singleton leaves fit inline, two/three-item leaves spill. Whitespace finishes with 2,543,328 inline singletons among 10,348,204 leaves; ByteLevel has 316,387 among 7,066,103. Child retained heap is larger despite fewer allocation calls. ByteLevel includes about 134 MB of unused heap capacity; growth-factor tuning was stopped by the user, and no shrink policy was added. Full histograms, per-bin payload/capacity and actual worker task loads are in final-storage-summary.json.

| Scene / arm | Allocations | Frees | Reallocations | Requested GB | Directory growths | Moved MB | Copy ms (worker sum) | Concat calls | Concat s (worker sum) | Tasks | Position items |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zh-bytelevel_regex-w4-r0-baseline-final-diagnostic | 9544773 | 9395236 | 18325444 | 36.37 | 0 | 0.0 | 0.00 | 15049 | 4.253 | n/a | n/a |
| zh-bytelevel_regex-w4-r0-child-inline-fast-final-diagnostic | 7567291 | 7417754 | 11106252 | 27.86 | 540573 | 1218.1 | 214.56 | 7066363 | 1.170 | 50019 | 162681608 |
| zh-whitespace-w4-r0-baseline-final-diagnostic | 14576314 | 14461783 | 21408189 | 21.95 | 0 | 0.0 | 0.00 | 1721017 | 5.903 | n/a | n/a |
| zh-whitespace-w4-r0-child-inline-fast-final-diagnostic | 11572204 | 11457673 | 13266033 | 22.08 | 165866 | 124.9 | 41.04 | 6490086 | 3.452 | 32607 | 36130172 |

Baseline task counts were not instrumented (n/a). Baseline concat counters cover initial construction only; child counters cover initial and subsequent publication, so call totals are not an identical workload boundary. These real inputs did not activate identity reuse. AA and alias/reuse controls have identical full models in final-controls and final-diagnostic-controls; the reuse diagnostic records three ordered stream merges, about 22 µs total. Functional coverage does not establish a realistic reuse performance result. Task item counts show distribution, not equal per-item processing cost.

## Bounded ablations and hot-path evidence

At user request SmallVec<[u8;24]> was tried as an isolated 40-byte Positions. It cut whole-train allocations (Whitespace 11.57M to 7.31M; ByteLevel 7.57M to 4.02M), but two four-worker Chinese pairs did not show a stable net gain versus 32 bytes (Whitespace +4.32% / +1.15%, ByteLevel −4.02% / +0.60%), and HWM grew about 2–4%. That trial was stopped. A separately requested thin-vec 0.2.21 trial reduced Positions to 24 bytes but slowed Whitespace by 26.8% and ByteLevel by 17.3%. Both trials preserve full model equality; neither dependency nor size variant is in production.

Profile summaries are retained for the final implementation. The Chinese ByteLevel merge profile attributes about 32.82% self samples to fresh prepare, 10.92% to owner commit, 9.28% to append and 7.82% to write. English merge push_restart self samples are 1.73%; this does not justify another encoding experiment. Historical cleanup comparisons are archived separately; final cleanup timing varies, and the isolated early parallel cleanup improvement is not presented as a stable speedup of this final implementation.

## Validation and code size

Default targeted BPE run: 29 passed, 0 failed, eight filtered including the previously confirmed JSON roundtrip “测” failure. No-default full BPE run: 29 passed, the same one known failure, seven unrelated tests filtered. Baseline logs reproduce that failure (32 pass before three obsolete protocol tests were removed). Clippy --lib -D warnings with only the existing redundant_field_names lint allowed passes; baseline Clippy logs show the two same old literals. Changed files pass rustfmt. Trial correctness logs are included, but production validation is tied to the 32-byte implementation.

Rust AST removes cfg(test), comments and blank lines for production counts. Positions 227→333, index 356→360, merge 560→506, mod 348→352: +345 / −285, net +60 production lines. The old lifecycle and conversion protocol is deleted; this is not a total-LOC reduction claim. Ordinary BPE tests including four Miri entry lines are 1500→1407 under the existing count definition. This stage did not run Miri.

## Reproduction and artifacts

environment.json records base and production SHAs, binary hashes, compiler, affinity, input hashes and build command. source/ contains frozen timed source snapshots and runner locks; final-diagnostic-source/ contains instrumented snapshots. Baseline and child-inline-fast are the decision pair. Other source arms and measurements are historical tuning evidence. The final formatted production differs from its frozen timed source only by formatting and removed timing probes, not behavior.

Create a disposable checkout of environment.json base_sha, provide the exact prepared inputs (hashes in that file), then run `python replay.py CHECKOUT multi-cn/zh-whitespace-w4-r0-child-inline-fast --input ORIGINAL_PATH=LOCAL_PATH`. replay.py overlays the frozen source, builds locked release, and executes the archived job. Original builders and measure.py also preserve machine-specific paths and command details. `uv run count-source.py` from the production repository root reproduces production counts. count-and-audit.py reproduces the existing full test budget.

final-pairs.json links ordinary paired metrics, phases and complete model hashes. Each run has its raw job/stdout/stderr. model-reference.json points to a losslessly compressed, content-addressed full model in models/; model-manifest.json records every archived run and checksum. Identical model bytes are stored once. Diagnostic models use the same equality check. Normal and instrumented timing results remain distinguishable; perf trace binaries and input corpora are not embedded.
