# Affix cache activation and original-character scanning experiments

These two independent production candidates are based on `9063c557`. Cache activation (`7073b6a4`) activates a decorated identity after intern and before filling its cache, so hits return without writing spans. Plain characters still activate on every retained occurrence. The original-character scan (`45b439fa`) partitions original first/middle/last characters with `Chars::next` and `next_back`, filters afterward, and creates the last identity only after the middle. It retains baseline activation writes. Its shared character helper takes a fixed middle slot; no combined candidate was measured.

Neither candidate was integrated. Cache activation has no consistent measured benefit: Chinese initialization is 3.87% faster in the first screen pair and 0.27% faster in the reverse pair; English is 5.01% and 1.71% slower. Original-character scanning is slower in both languages and both orders. Whole-train results are also retained below; initialization deltas do not establish matching whole-train changes. No machine-code cause is claimed.

The user stopped further affix work while a historical Flag comparison was in progress. The complete screen has 18 samples; AA/reuse controls have six samples. The historical comparison has five completed Chinese samples, one cancelled run, and no English samples. The cancelled run is excluded from results, and its partial job/stdout/stderr remain under `flags/zh-affix-r1-oldflag`. Candidate debug fixture validation was not run before cancellation; the production patches remain experiments. The separately completed test compaction is published without these production changes.

Both 256 MiB input corpora and prepared-file hashes are in `inputs.json`. Limits 2047/128 exercise actual alphabet cutoffs without a boundary tie. Vocabulary target is 50,000, min frequency 2. Affix cases set both prefix `##` and suffix `</w>`; plain controls set neither. The runner reconstructs the caller AHashMap with seeds [11,13,17,19], preserving word traversal and decorated ID allocation. Every completed sample compares the complete vocabulary in ID order and every ordered merge by byte-identical model export. AA uses one attempt; the forced alias/reuse control uses two attempts in all three arms.

Release builds use Rust 1.98.1, opt-level 3, fat LTO, one codegen unit and no tk-train default features. Four workers are pinned to CPUs 0–3. Samples run serially; builds and tests do not overlap sampling. Diagnostic timers are confined to measured sources. `train_seconds` is public do_train time and excludes prepared-input load and output export. The raw current Vocabulary timer excludes selection and CorpusPlan construction. Current selection is derived from combined initialization minus the raw timer. Historical Flag `38b283c0` selects inside Vocabulary::initialize, so its exclusive initialization subtracts its nested selection timer. Raw timer values and the boundary flag are retained in `flags.json`; current initialization never subtracts selection.

HWM includes prepared-input loading and is sampled before validation/export; it is not phase-specific memory. Child swap is zero in every completed sample. Child major faults are 0–1; host paging varies and includes unrelated processes. Paging columns are retained rather than used to discard samples. The first control invocation stopped at a parser assertion because the current selection log had moved; it is retained in `failed-control-parser` and was rerun successfully after deriving selection from the mod.rs timers.

Completed screen samples (seconds; HWM in MiB):

| Case | Round | Arm | Vocab init | Whole train | Train CPU | HWM |
|---|---:|---|---:|---:|---:|---:|
| zh-plain | 1 | baseline | 0.004034 | 25.117443 | 65.295108 | 1732.98 |
| zh-plain | 1 | cache | 0.002930 | 23.747168 | 62.922456 | 1696.93 |
| zh-plain | 1 | split | 0.002933 | 24.031397 | 63.950722 | 1752.05 |
| en-plain | 1 | split | 0.002918 | 1.569941 | 3.670731 | 138.69 |
| en-plain | 1 | cache | 0.004157 | 1.765618 | 3.685431 | 137.08 |
| en-plain | 1 | baseline | 0.003390 | 1.505755 | 3.545696 | 138.69 |
| zh-affix | 1 | baseline | 1.813660 | 26.211516 | 66.411543 | 1776.77 |
| zh-affix | 1 | cache | 1.743540 | 26.048844 | 66.585058 | 1775.02 |
| zh-affix | 1 | split | 2.056142 | 26.558093 | 66.136132 | 1809.53 |
| zh-affix | 2 | split | 2.028431 | 26.241437 | 66.246982 | 1773.30 |
| zh-affix | 2 | cache | 1.798464 | 26.915513 | 68.816141 | 1773.23 |
| zh-affix | 2 | baseline | 1.803247 | 25.846292 | 65.816113 | 1771.64 |
| en-affix | 1 | split | 0.067033 | 1.709648 | 3.549842 | 161.75 |
| en-affix | 1 | cache | 0.048821 | 1.827933 | 3.942448 | 161.53 |
| en-affix | 1 | baseline | 0.046493 | 1.773756 | 3.841807 | 157.67 |
| en-affix | 2 | baseline | 0.045365 | 1.654195 | 3.493581 | 158.94 |
| en-affix | 2 | cache | 0.046143 | 1.668254 | 3.463292 | 156.98 |
| en-affix | 2 | split | 0.076897 | 1.689746 | 3.573427 | 157.82 |

Paired initialization and whole-train deltas against the current baseline in each round; negative means faster:

| Case | Round | Arm | Vocab init | Whole train |
|---|---:|---|---:|---:|
| en-affix | 1 | cache | +5.01% | +3.05% |
| en-affix | 1 | split | +44.18% | -3.61% |
| en-affix | 2 | cache | +1.71% | +0.85% |
| en-affix | 2 | split | +69.51% | +2.15% |
| en-plain | 1 | cache | +22.65% | +17.26% |
| en-plain | 1 | split | -13.91% | +4.26% |
| zh-affix | 1 | cache | -3.87% | -0.62% |
| zh-affix | 1 | split | +13.37% | +1.32% |
| zh-affix | 2 | cache | -0.27% | +4.14% |
| zh-affix | 2 | split | +12.49% | +1.53% |
| zh-plain | 1 | cache | -27.38% | -5.46% |
| zh-plain | 1 | split | -27.30% | -4.32% |

Partial historical Flag comparison (completed samples only):

| Case | Round | Arm | Vocab init s | Whole train s |
|---|---:|---|---:|---:|
| zh-affix | 1 | oldflag | 1.718611 | 26.603077 |
| zh-affix | 1 | baseline | 1.855561 | 25.331590 |
| zh-affix | 1 | cache | 1.795211 | 26.482284 |
| zh-affix | 2 | cache | 1.783272 | 27.094428 |
| zh-affix | 2 | baseline | 1.783550 | 27.262914 |

The first complete historical triplet puts old Flag at 1.718611 s, current at 1.855561 s and cache activation at 1.795211 s. Cache activation closes part of that one observed gap but remains slower than old Flag. In the next reverse-order pair, current and cache are effectively equal (1.783550/1.783272 s); the matching old Flag sample was cancelled. These partial results establish neither stable recovery toward the old Flag baseline nor an English comparison.

`cache-production.patch` and `split-production.patch` reconstruct the independent candidates from `9063c557`. Measured source snapshots, dependency locks, build commands/logs, raw jobs/stdout/stderr, complete exported models, pairs, controls and hashes accompany this report. Frozen binaries and the full durable artifact directory remain at `/root/code/tokenizers-workspaces/bpe-affix-core-results-20261011`; published evidence omits the binaries and unused combined-source draft. `oldflag-provenance.json` identifies the reused historical binary and source snapshot.
