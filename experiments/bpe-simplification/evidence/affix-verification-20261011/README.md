# Nonempty-affix verification and separate training stages

This compares `38b283c0` (before the flags/alphabet changes) with published `0dd1ca48` (current). `flags-only` applies only the named-bool/three-slot decoration change to `38b283c0`; it retains the old alphabet selection and activation behavior. This round makes no production Vocabulary changes.

The previous initialization figures described a stage containing alphabet selection, and must not be described as whole-training gains. With fixed input traversal here, the plain combined stage improves while alphabet selection itself becomes slower. The separate table gives the whole-training result.

Both-affix cases show no comparable initialization acceleration. The current combined stage is slower for Chinese and for the English repeat, while whole-train results are close to baseline. The isolated flags initialization is slower in both Chinese samples and both English samples (about 6–13%). Its whole-train deltas change direction in both reverse-order repeats, so this is an observed initialization regression rather than an established whole-train regression. The initial flags-only whole-training slowdowns cannot be dismissed from the plain case: reverse-order repeats are included below. These few samples do not establish a stable affix throughput gain or rule out a small initialization regression. No sample is removed because of its direction or host paging.

Four workers pinned to CPUs 0–3; all samples run serially. The release runner uses opt-level 3, fat LTO, one codegen unit, and no tk-train default features. Exact source snapshots, dependency locks, binaries, models, build logs and raw job/stdout/stderr files are retained in the durable directory named below. Compilation and tests did not overlap timed samples.

The raw Chinese and English corpora are about 256 MiB each. Both use the same public Whitespace preprocessing; input and prepared-file hashes are in `inputs.json`. The fixed prepared words are reconstructed into the same borrowed caller AHashMap with seeds [11,13,17,19]. This fixes word traversal and affix ID assignment. Load, preprocessing, and serialization are outside `train_seconds`; this is public `do_train` time, not Feed-plus-train pipeline time. The Chinese prepared file has 6,827,958 unique words; English has 724,596. Limits 2047/128 have actual frequency cutoffs without a tie at the boundary. Vocabulary size is 50,000 and min frequency is 2.

Plain cases have no affixes. Affix cases set **both nonempty prefix `##` and suffix `</w>`** and exercise prefix, suffix, and combined decoration slots. Every case compares the entire vocabulary in ID order and every ordered merge, using byte-identical exported model files, not only counts. Native tokenizer JSON or encoding parity is not claimed by this performance export.

Alphabet selection is measured inside `select_alphabet`. Vocabulary initialization ends immediately after `Vocabulary::initialize`, before `CorpusPlan::build`; it excludes selection. Combined is selection plus that initialization. Whole train includes both stages and the remaining training work. HWM is the process peak RSS sampled before validation/export; it also includes prepared-input loading and is not a phase-specific peak. CPU seconds are summed process CPU during the train timer. Stage logging is diagnostic-only; production code has no new timers.

For the old architecture, the raw Vocabulary timer includes selection; the reported exclusive initialization subtracts the separately measured selection. The current architecture selects at entry and the raw Vocabulary timer already excludes selection. The same hooks capture the boundary once, and the driver checks stage sums. All performance cases finish in one attempt.

The first ten samples use baseline/current for Chinese, current/baseline for English, then flags-only for each decorated case. The English decorated repeat reverses to flags/current/baseline. The Chinese decorated repeat also uses flags/current/baseline to check the initial flags slowdown.

A separate `perf stat` diagnostic enables user-mode cycles/instructions/branches/branch-misses between the selection log and initialization log, using an external FIFO controller. Compared with the old flags path, the isolated named-bool path executes about 4.4% more instructions and 7.6% more branches for Chinese, and about 5.6% more instructions and 9.7% more branches for English. Branch misses do not increase. This supports extra cached-lookup control work rather than new string allocations or worse branch prediction. These are approximate external-control windows, especially for the short English stage; the enable acknowledgment can take 4–34 ms. They are diagnostics, not the primary timing samples. Raw counters, commands, controller times, and full models are retained under `profile-init/` and summarized in `profile-init.json`.

Durable artifacts: `/root/code/tokenizers-workspaces/bpe-affix-verification-results-20261011`. `runs.json`/CSV retain all timings and memory data; `pairs.json` contains all deltas. `artifact-hashes.json` records frozen files; raw models remain outside Git.

| Case | Arm | Alphabet s | Vocab init s | Combined s | Whole train s | Train CPU s | HWM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| zh-plain | baseline | 1.853162 | 1.690044 | 3.543206 | 24.774031 | 63.079029 | 1730.54 |
| zh-plain | candidate | 2.445022 | 0.003489 | 2.448619 | 23.384253 | 61.336722 | 1731.47 |
| en-plain | candidate | 0.061137 | 0.004491 | 0.065718 | 1.619972 | 3.770552 | 139.44 |
| en-plain | baseline | 0.045770 | 0.045458 | 0.091228 | 1.690758 | 3.751729 | 146.08 |
| zh-affix | baseline | 1.866246 | 1.910369 | 3.776615 | 25.925682 | 65.888833 | 1774.64 |
| zh-affix | candidate | 2.067898 | 2.034303 | 4.102287 | 25.773050 | 65.751797 | 1776.05 |
| zh-affix | flags-only | 1.942008 | 2.024223 | 3.966232 | 28.021348 | 72.109718 | 1773.93 |
| en-affix | candidate | 0.051203 | 0.054242 | 0.105505 | 1.725884 | 3.624918 | 157.94 |
| en-affix | baseline | 0.057578 | 0.054105 | 0.111682 | 1.742945 | 3.654308 | 159.18 |
| en-affix | flags-only | 0.043944 | 0.060927 | 0.104871 | 1.965713 | 4.152058 | 156.50 |
| en-affix-repeat | flags-only | 0.047159 | 0.056754 | 0.103913 | 1.593664 | 3.292109 | 156.36 |
| en-affix-repeat | candidate | 0.053963 | 0.055230 | 0.109252 | 1.651534 | 3.388051 | 160.43 |
| en-affix-repeat | baseline | 0.049276 | 0.050458 | 0.099734 | 1.673967 | 3.404529 | 158.62 |
| zh-affix-repeat | flags-only | 1.791877 | 1.846351 | 3.638228 | 25.986227 | 66.294591 | 1774.75 |
| zh-affix-repeat | candidate | 1.938830 | 1.864488 | 3.803405 | 25.559798 | 65.086577 | 1771.79 |
| zh-affix-repeat | baseline | 1.719203 | 1.638561 | 3.357764 | 26.361705 | 68.106796 | 1770.38 |

Deltas against the baseline in the same case; negative time means faster.

| Case | Arm | Alphabet | Vocab init | Combined | Whole train | HWM |
|---|---|---:|---:|---:|---:|---:|
| zh-plain | candidate | +31.94% | -99.79% | -30.89% | -5.61% | +0.05% |
| en-plain | candidate | +33.57% | -90.12% | -27.96% | -4.19% | -4.55% |
| zh-affix | candidate | +10.81% | +6.49% | +8.62% | -0.59% | +0.08% |
| zh-affix | flags-only | +4.06% | +5.96% | +5.02% | +8.08% | -0.04% |
| en-affix | candidate | -11.07% | +0.25% | -5.53% | -0.98% | -0.78% |
| en-affix | flags-only | -23.68% | +12.61% | -6.10% | +12.78% | -1.68% |
| en-affix-repeat | flags-only | -4.30% | +12.48% | +4.19% | -4.80% | -1.43% |
| en-affix-repeat | candidate | +9.51% | +9.46% | +9.54% | -1.34% | +1.14% |
| zh-affix-repeat | flags-only | +4.23% | +12.68% | +8.35% | -1.42% | +0.25% |
| zh-affix-repeat | candidate | +12.77% | +13.79% | +13.27% | -3.04% | +0.08% |

Process and host paging over the entire child lifetime, including load and model export. These are not restricted to the train timer. Host swap counts include unrelated processes.

| Case | Arm | Child major faults | Max child swap KiB | Host swap-in pages | Host swap-out pages |
|---|---|---:|---:|---:|---:|
| zh-plain | baseline | 7 | 0 | 7 | 102259 |
| zh-plain | candidate | 0 | 0 | 5 | 0 |
| en-plain | candidate | 0 | 0 | 0 | 0 |
| en-plain | baseline | 0 | 0 | 0 | 0 |
| zh-affix | baseline | 0 | 0 | 23 | 1784 |
| zh-affix | candidate | 0 | 0 | 1 | 19646 |
| zh-affix | flags-only | 0 | 0 | 7 | 773 |
| en-affix | candidate | 0 | 0 | 0 | 0 |
| en-affix | baseline | 0 | 0 | 0 | 0 |
| en-affix | flags-only | 0 | 0 | 1 | 0 |
| en-affix-repeat | flags-only | 0 | 0 | 0 | 0 |
| en-affix-repeat | candidate | 0 | 0 | 1 | 0 |
| en-affix-repeat | baseline | 0 | 0 | 0 | 0 |
| zh-affix-repeat | flags-only | 0 | 0 | 4 | 202 |
| zh-affix-repeat | candidate | 0 | 0 | 2 | 118 |
| zh-affix-repeat | baseline | 0 | 0 | 26 | 6 |
