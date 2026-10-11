# Feed uniform owned entries — experiment on 8f078a8c

This experiment implements the supplied [plan](experiment-plan.md) in a detached candidate worktree. The frozen baseline is `8f078a8ca11a62bb6f81fe99a73f028f9cc4a5d2`, the latest fork commit at experiment start, including B1. The original worktree's production source remained unchanged throughout measurement. After reviewing the result, the user requested adoption and push; integration is recorded below.

During execution the user clarified that the decision should prioritize maintenance cost and whole-pipeline peak memory. Feed peak is retained as allocation diagnostics; when it remains below the later pipeline peak it is not treated as a decisive memory result. The user accepts VPS noise. Observed fluctuations are still reported without automatically attributing every difference to VPS scheduling.

## Production change and compatibility

[production.patch](production.patch) changes only `word_counts.rs` and its existing Feed tests. Trainer-owned `WordCounts::Map/Entries` becomes `WordCounts(Vec<(CompactString, u64)>)`. Serial Feed's existing `from_map` completion now moves the map's iterator into a Vec; parallel Feed passes its existing Vec directly. `feed.rs`, hash initialization, local-key limit 2048, batches of 32, startup, callbacks, vocabulary initialization, Corpus/B1, index and position decoding are unchanged.

The owned representation dispatch disappears from `view`, equality and serialization, together with the enum and manual Default implementation. The borrowed `WordCountsView` still accepts caller-owned maps for public `do_train`. Content equality, map-style Debug and flat-map serde remain explicit compatibility code. Deserialization still uses CountMap first, retaining last-wins duplicate keys, then moves entries into the owned Vec. The existing borrowed sorting Vec remains; repeated training continues to borrow the counts.

Same-order equality avoids a lookup allocation. Unordered equality uses a temporary index of borrowed keys, without copying owned strings. This existing entries/entries fallback now also applies to independently fed serial trainers; baseline map/map equality did not allocate that temporary index. The performance matrix measures Feed/Train, not the cost of large trainer equality comparisons.

The non-test portion of `word_counts.rs` has 22 fewer lines. Test changes add 16 lines, so the complete production/test patch has six fewer lines overall. This is a reduction in owned states and dispatch, with an added serial O(unique words) move/allocation; the borrowed compatibility representation remains.

## Correctness

Commands, exit codes and logs are in [validation.json](validation.json), [all-lib-validation.json](all-lib-validation.json) and [harness-validation.json](harness-validation.json). Default and no-default full-library checks pass 42 baseline tests and 43 candidate tests, each excluding one independently reproduced existing encoder failure. Candidate Clippy (`--all-targets -- -D warnings`) and changed-file formatting pass.

The unfiltered BPE checks reproduce `feed_training_tokenizer_json_and_encoding_roundtrip` failing its encoding assertion for `测`: actual `[PipelineToken(3)]`, expected `[6]`, in both baseline and candidate. The failure is retained in both default logs; no encoder source was changed. Other BPE tests pass. The full-library checks add the seven tests outside that BPE name filter. No counts or model-equivalence result is inferred from the excluded encoder test.

Existing content equality and flat serde coverage is reused. The new duplicate-key case checks last-wins, flat output, Debug shape and negative-count rejection. Existing Feed tests retain empty input, serial/parallel, local flush and batch boundaries, startup and ordinary callback error coverage. The successful Feed scenario trains twice and verifies its stored counts remain intact; public map `do_train` is still compared with trainer-owned training.

The identical measurement harness's existing full-count exporter test passes on both sides. An initial serializer harness build failed because its enabled model/normalizer features did not cover the features already active in tk-train. Only the harness configuration was corrected; [the initial log](baseline-harness-tests-serializer-feature-failure.log) is retained. Harness source, lockfiles and patches are byte-identical between arms; production source differences are separately verified in [source-hashes.json](source-hashes.json).

## Measurement method

Both binaries use release opt-level 3, fat LTO and one codegen unit. Runner default features are empty; tk-train uses no default features. [builds.json](builds.json) records binary hashes and commands; [snapshot.json](snapshot.json), [environment.txt](environment.txt), the source snapshots and both lockfiles capture the version and build inputs. The source tree snapshot excludes unrelated untracked experiment directories. Sparse worktrees preserve the same full Git tree while checking out only required build sources.

Run `PATH=/root/.cargo/bin:$PATH python3 measure_pipeline.py` from this evidence directory, using the frozen binaries and corpus paths in the manifest. All builds and checks complete before sequential benchmark execution. No microbenchmark or alternative production variant is used. The script first performs 16 separate 1 MiB complete-line-prefix warmups, one per side and scenario, then 32 full processes: Chinese/English × Whitespace/ByteLevel-regex × 1/4 workers, two opposite-order A/B pairs per scenario. Any justified follow-up pair is recorded separately.

Full corpora are the existing pinned approximately 256 MiB UTF-8 text files, streamed by BufReader lines without preloading. Both input size and SHA256 are verified against their manifests. Parameters are vocabulary 50,000, minimum frequency 2, no prefix/suffix/max-token-length limit; ByteLevel uses the existing GPT-2 regex/normalization path and initial byte alphabet. Production RandomState initialization and iteration-order semantics are untouched; no fixed Feed seed or pre-ID sorting is introduced. CPU affinity is CPU 0 for one worker and CPUs 0–3 for four workers. Input hashes warm the ordinary OS page cache; no candidate-specific cache eviction or preload is used.

Feed wall uses the same process-internal Instant interval across lazy reading, preprocessing, aggregation and candidate map→Vec conversion. CPU is getrusage user plus system CPU. Pipeline wall/CPU includes Feed, intervening identical checkpoints and `train_vocab`; both arms exclude Processor initialization and input open. Feed RSS is current VmRSS; Feed HWM is cumulative process VmHWM through Feed return. Pipeline HWM is captured before count export and model construction/serialization. Parent-polled whole-process peaks including validation are separate fields and are not attributed to training.

Every successful sample exports complete JSON-key/count TSV after these timing and memory boundaries. LC_ALL=C external sorting and SHA256 cover the entire count set; record count is checked against unique words. Complete original vocabulary-to-ID entries and ordered merges are compared without changing IDs or sorting merges. Complete native model JSON, including all model configuration fields, is independently compared. Large TSV exports are deleted only after equality succeeds; original and native model files, jobs and stdout/stderr remain. Native export constructs a model after the measured interval and is excluded from pipeline HWM.

Machine load, memory, vmstat, pressure, CPU MHz and available thermal readings are recorded before and after each process. The machine had pre-existing system swap usage; each child is independently polled every 25 ms for VmSwap. Major/minor faults use process getrusage. The actual layout is 32 bytes per owned entry and 16 bytes per borrowed reference pair. Uniform ownership does not remove the borrowed sorting array. During serial conversion, map buckets and the new Vec coexist, so peak and steady-state memory can move in opposite directions.

## Results and decision

**Recommendation: retain the candidate for its modest, concrete maintenance reduction.** Owned counts have one representation; equality no longer maintains a cross-representation matrix, and view/serialization no longer dispatch on Feed history. This does not simplify Feed aggregation or the BPE algorithm itself. Public-map borrowing and the flat-map/content traits remain explicit, with no additional caller protocol, dependency or production measurement code.

The user prioritizes whole-pipeline HWM. The serial Chinese cases reduce it in both original pairs; four-worker Chinese Whitespace remains effectively equal. Other whole-pipeline peaks vary, and English ByteLevel serial does not show a uniform reduction. The smaller held Vec is useful context, but neither lower Feed RSS nor changes in Feed peak are presented as a universal reduction in final peak memory.

The table reports the **median of paired candidate/baseline percentage differences**, not the ratio of independent arm medians. A negative time/HWM percentage is lower. Only English ByteLevel serial and Chinese ByteLevel four-worker have a third pair. Every original run remains included; the [original two-pair summary](original-two-pairs/pipeline-summary.json) and [follow-up justification](extra-block-2.json) preserve the initial matrix separately.

| Scenario | Pairs | Feed wall | Feed CPU | Pipeline wall | Pipeline CPU | Pipeline HWM | Paired HWM difference, MiB (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| zh-whitespace-w1 | 2 | -7.21% | -7.04% | -1.70% | -1.52% | -3.20% | -54.56 (-55.31 to -53.80) |
| zh-bytelevel_regex-w1 | 2 | -3.08% | -3.08% | -1.01% | -0.94% | -2.47% | -56.45 (-73.15 to -39.74) |
| en-whitespace-w1 | 2 | -7.70% | -7.78% | -6.40% | -6.53% | -10.13% | -14.35 (-15.02 to -13.68) |
| en-bytelevel_regex-w1 | 3 | -7.22% | -7.10% | -5.99% | -5.88% | +3.36% | +5.71 (-24.31 to +15.47) |
| zh-whitespace-w4 | 2 | +2.56% | +4.73% | -0.88% | -0.32% | +0.03% | +0.46 (+0.23 to +0.68) |
| zh-bytelevel_regex-w4 | 3 | -4.71% | -2.32% | -1.12% | -0.63% | +0.78% | +18.63 (-5.95 to +53.41) |
| en-whitespace-w4 | 2 | -11.89% | -9.91% | -8.58% | -8.08% | -0.11% | -0.22 (-3.60 to +3.17) |
| en-bytelevel_regex-w4 | 2 | -5.58% | -5.83% | -5.11% | -5.84% | -4.92% | -8.74 (-16.73 to -0.75) |

These are observations from two/three pairs on this VPS with the production random hash policy. They establish correctness and expose material tradeoffs; they do not establish a lasting speedup. In particular, parallel Feed already owned entries in the baseline, so its apparent faster/slower samples are not attributed to eliminating a conversion.

### The initially reported 6.5% slowdown

Chinese ByteLevel, four workers, pipeline wall pairs were **+6.55%, -1.12%, -2.69%**. The first pair was 28.19585s baseline versus 30.04192s candidate: Feed contributed +0.77079s and training +1.07528s. The original positive sample is retained; the later pairs refute describing it as a sustained 6.5% regression.

The first pair switches from a lower-allocation Feed sample in baseline to a higher-allocation sample in candidate (minor faults 289,735 → 434,013). The reverse-order pair switches the other way (candidate 282,074; baseline 436,949). CPU MHz reported by the VPS is unchanged, child swap and major faults are zero, and recorded CPU pressure is small. Thus a pure VPS-only explanation is not established.

[The exact dependency source and analysis](allocation-explanation.json) show scc 3.8.8 uses estimated occupancy from bucket samples, including a full sample of up to 128 buckets. Its enlargement threshold is 13/16: for 2^23 slots, 6,815,744 entries. The two Chinese word sets, 6,827,958 and 6,859,987, lie close to that threshold. Old bucket arrays are linked during incremental rehash and reclaimed later. Existing sampled expansion/reclamation and random layout are plausible contributors to the allocation regimes observed on both arms. Actual capacities and resize events were not traced; this is source-backed inference, not proof of the cause of every time difference, including the training portion. Hash/table policy remains unchanged.

### Equivalence, failures and memory scope

All **36 formal processes plus 16 warmups** succeed, process the complete expected input, and match complete canonical counts, original vocabulary/ordered merges and complete native model JSON across both arms/workers. All 52 children have zero observed swap and zero measured pipeline major faults. The corpus-coordinate panic mentioned in the plan reproduces in neither arm for these scenarios. This is distinct from the existing encoder assertion reproduced by unfiltered tests on both arms; no independent issue was fixed.

All timing boundaries, absolute paired differences and ratios, per-arm ranges, Feed/current RSS, cumulative Feed HWM, pre-validation pipeline HWM, separate process-wide validation peaks, commands and machine snapshots are retained in [pipeline-runs.json](pipeline-runs.json), [pipeline-summary.json](pipeline-summary.json) and per-run artifacts. Feed conversion adds a full O(unique) move/allocation and keeps the borrowed sorting array; large unordered trainer Eq additionally needs a temporary borrowed-key index. Those costs remain part of the design tradeoff.

No instrumented map→Vec timer or microbenchmark was added. Exact observed Feed deltas include the entire conversion and other process variation; the experiment does not isolate conversion CPU cost. At experiment completion, the patch remained isolated without a merge, commit or push. [Final verification](final-verification.json) captures that historical state and confirms frozen source/binary hashes and all 52 equivalence checks.

## Accepted integration

The user subsequently requested push and specified native tests without another benchmark. The branch was fast-forwarded to `8afc869f`, retaining the intervening neighbor-directory, vocabulary-insertion and fixed-length Box storage changes. The two accepted Rust files remain byte-identical to the measured candidate; the measurement harness remains confined to the archived source snapshots. The BPE guide now describes the uniform owned representation.

[Integration metadata](integration.json) and [native checks](integration-validation.json) record the parent, source equivalence, commands and results. Default and no-default full-library suites each pass 44 tests; all-target Clippy with denied warnings and changed-file formatting pass. These integration checks do not extend the performance results to the newer combined source. The independently reproduced encoder assertion is excluded from both full-library test configurations, as in the original experiment. The final verification record above is preserved as evidence of the original isolated measurement.
