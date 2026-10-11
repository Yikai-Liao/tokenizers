# Selected rules stored as a plain HashMap

Candidate production commit `b4599744` is based on current `0dd1ca48` in the independent worktree `/tmp/bpe-selected-rules-hashmap-20261011`. Only `merge.rs` changes. Ordinary preparation holds the pair-to-replacement AHashMap directly; `Head`, preparation heads/tails arrays, and `SelectedRules` are removed. Production code adds 12 lines and removes 66, net −54. Selection conflict sets remain. AA and reuse handling, Positions storage and task splitting, Source, Birth, and neighbor event order are unchanged. This experiment has not been merged into the published simplification branch and does not incorporate the other computer's Positions changes.

Independent implementation review and validation are in `review.md` and `validation.json`. Both default and no-default BPE configurations pass 43 tests, including the new shared-head/shared-tail exact-neighbor-events regression, and retain one previously reported encoding-roundtrip failure. Complete raw test stdout was not saved by the implementation agent; that limitation is recorded. Clippy with all targets/no default features and `-D warnings`, formatting, and whitespace checks pass.

This comparison is current `0dd1ca48` versus the map candidate with identical diagnostic stage hooks and input protocol. The preparation arrays no longer allocate, while ordinary neighbor resolution performs extra exact-pair hashing and may read an extra corpus endpoint. Four case pairs alone support only measured deltas, not a stable general speedup claim. Changes in alphabet and initialization timers are noise for this experiment: those production stages are byte-identical in both arms.

Four workers pinned to CPUs 0–3; all samples run serially. The release runner uses opt-level 3, fat LTO, one codegen unit, and no tk-train default features. Exact source snapshots, dependency locks, binaries, models, build logs and raw job/stdout/stderr files are retained in the durable directory named below. Compilation and tests did not overlap timed samples.

The raw Chinese and English corpora are about 256 MiB each. Both use the same public Whitespace preprocessing; input and prepared-file hashes are in `inputs.json`. The fixed prepared words are reconstructed into the same borrowed caller AHashMap with seeds [11,13,17,19]. This fixes word traversal and affix ID assignment. Load, preprocessing, and serialization are outside `train_seconds`; this is public `do_train` time, not Feed-plus-train pipeline time. The Chinese prepared file has 6,827,958 unique words; English has 724,596. Limits 2047/128 have actual frequency cutoffs without a tie at the boundary. Vocabulary size is 50,000 and min frequency is 2.

Plain cases have no affixes. Affix cases set **both nonempty prefix `##` and suffix `</w>`** and exercise prefix, suffix, and combined decoration slots. Every case compares the entire vocabulary in ID order and every ordered merge, using byte-identical exported model files, not only counts. Native tokenizer JSON or encoding parity is not claimed by this performance export.

Alphabet selection is measured inside `select_alphabet`. Vocabulary initialization ends immediately after `Vocabulary::initialize`, before `CorpusPlan::build`; it excludes selection. Combined is selection plus that initialization. Whole train includes both stages and the remaining training work. HWM is the process peak RSS sampled before validation/export; it also includes prepared-input loading and is not a phase-specific peak. CPU seconds are summed process CPU during the train timer. Stage logging is diagnostic-only; production code has no new timers.

Order is baseline/candidate for Chinese and candidate/baseline for English. The Chinese affix repeat reverses to candidate/baseline because the first pair was 3.4% slower. The repeat is 0.08% slower, with about 2% more train CPU time; the first wall-time difference is not stable across the two pairs. The baseline binary is the exact current binary used in the affix report. Its source snapshot is retained there; candidate source is frozen here. `production-source-hashes.json` ties the measured merge code to the tested production commit.

Separate AA and reuse controls compare full models. The long-AA control runs one attempt; the forced alias/reuse control uses suffix `a` and must run two attempts in each arm. Their tiny times are not performance results. Raw jobs, models, and logs are retained under `controls/`, with outcomes in `controls.json`.

Durable artifacts: `/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011`. `runs.json`/CSV retain all stages, whole-train CPU/wall time and memory; `pairs.json` retains deltas. `artifact-hashes.json` records frozen sources/binaries/models. The copied baseline initially lacked execute permission, so its first startup returned 126 before training; that failed job is retained under `failed-startup-permission/`. It is excluded from timed samples because it never ran. The execute bit was corrected without changing binary bytes.

| Case | Arm | Alphabet s | Vocab init s | Combined s | Whole train s | Train CPU s | HWM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| zh-plain | baseline | 1.937884 | 0.004135 | 1.942104 | 22.577912 | 60.692738 | 1761.54 |
| zh-plain | candidate | 1.971822 | 0.004401 | 1.976302 | 22.509471 | 60.996542 | 1696.25 |
| en-plain | candidate | 0.051337 | 0.004196 | 0.055607 | 1.518471 | 3.587947 | 138.48 |
| en-plain | baseline | 0.060714 | 0.005033 | 0.065863 | 1.579173 | 3.699161 | 138.65 |
| zh-affix | baseline | 1.930271 | 1.839863 | 3.770196 | 25.143558 | 64.101474 | 1770.93 |
| zh-affix | candidate | 1.935921 | 1.732236 | 3.668234 | 26.007040 | 65.942727 | 1774.28 |
| en-affix | candidate | 0.053794 | 0.048926 | 0.102809 | 1.607023 | 3.382140 | 160.59 |
| en-affix | baseline | 0.071805 | 0.054904 | 0.126767 | 1.615832 | 3.359076 | 157.82 |
| zh-affix-repeat | candidate | 1.998712 | 1.752962 | 3.751776 | 26.478172 | 68.744492 | 1815.60 |
| zh-affix-repeat | baseline | 1.981383 | 1.921853 | 3.903383 | 26.455707 | 67.379032 | 1775.96 |

Deltas against the baseline in the same case; negative time means faster.

| Case | Arm | Alphabet | Vocab init | Combined | Whole train | HWM |
|---|---|---:|---:|---:|---:|---:|
| zh-plain | candidate | +1.75% | +6.44% | +1.76% | -0.30% | -3.71% |
| en-plain | candidate | -15.44% | -16.63% | -15.57% | -3.84% | -0.12% |
| zh-affix | candidate | +0.29% | -5.85% | -2.70% | +3.43% | +0.19% |
| en-affix | candidate | -25.08% | -10.89% | -18.90% | -0.55% | +1.76% |
| zh-affix-repeat | candidate | +0.87% | -8.79% | -3.88% | +0.08% | +2.23% |

Process and host paging over the entire child lifetime, including load and model export. These are not restricted to the train timer. Host swap counts include unrelated processes.

| Case | Arm | Child major faults | Max child swap KiB | Host swap-in pages | Host swap-out pages |
|---|---|---:|---:|---:|---:|
| zh-plain | baseline | 0 | 0 | 2 | 492 |
| zh-plain | candidate | 0 | 0 | 5 | 84 |
| en-plain | candidate | 0 | 0 | 0 | 0 |
| en-plain | baseline | 0 | 0 | 0 | 0 |
| zh-affix | baseline | 0 | 0 | 1 | 188 |
| zh-affix | candidate | 0 | 0 | 2 | 210 |
| en-affix | candidate | 0 | 0 | 1 | 0 |
| en-affix | baseline | 0 | 0 | 0 | 0 |
| zh-affix-repeat | candidate | 0 | 0 | 7 | 0 |
| zh-affix-repeat | baseline | 0 | 0 | 3 | 481 |
