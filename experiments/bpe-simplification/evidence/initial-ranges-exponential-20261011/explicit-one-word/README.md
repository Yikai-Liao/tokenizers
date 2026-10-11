This is the current implementation. It retains `slice-group-by` 0.3.1,
with default features disabled, while making mandatory progress explicit:

1. Admit one word with `one_word_end = begin + 1`, even if it exceeds the budget.
2. Finish immediately for the last word or a fitting final tail. `starts` has no
   terminal corpus boundary, so that tail cannot be searched in the slice.
3. Search end boundaries for additional words. `Ok(exact)` includes the exact
   budget end; `Err(insertion)` already counts the fitting additional ends.
4. Append `begin..end` and advance to `end`.

Each entry in `starts[one_word_end + 1..]` closes an additional word. The explicit
first-word admission removes the earlier `fitted.max(1)` fallback and makes the
search result a word count with a single interpretation. The library continues
to own probe growth and bounded binary search. It does not own oversized-word or
final-tail rules; these remain together in the whole-word loop. The interface,
word-start representation, Shard and restored position Range API are preserved.

Validation uses the actual current production function. Default and
no-default-features library tests each pass 38 tests with the previously
reproduced baseline encoder roundtrip assertion excluded. Clippy with warnings
denied, changed-file formatting and whitespace checks pass. The procedural
linear oracle and direct 3/4/2/6 expectations cover exact budgets, oversized
words and final tails. The extracted microbenchmark also checks empty inputs,
zero budgets, power-of-two-adjacent lengths and coordinates near `usize::MAX`.

The microbenchmark retains the pushed binary baseline and paused linear source
as comparisons. All arms share the same inputs; one warmup block is discarded
and twelve formal blocks cover each of six arm orders twice on CPU 0. Each arm
calibrates its own iteration count to approximately 80 ms. Allocation and drop
are included. The following are medians in microseconds:

| Layout | Binary | Linear | Current exponential |
| --- | ---: | ---: | ---: |
| 100k words, all fit | 0.0565 | 83.3679 | 0.0598 |
| 2m words, 3 large segments | 0.1633 | 3182.7953 | 0.2473 |
| 100k words, one per segment | 6435.1530 | 322.7450 | 998.3837 |
| 100k words, about 3 per segment | 2245.7409 | 148.4176 | 331.8672 |
| 100k words, 417 medium segments | 41.6474 | 94.4977 | 21.6892 |

The selected search remains a compromise across segment sizes. The binary
baseline is faster for a few large segments, where the absolute difference is
below 0.1 microsecond in this matrix. Linear is faster for very short segments;
exponential avoids its long scans and avoids repeated full-remainder binary
searches for short segments. The raw measurements include the empty and
single-word cases as well. These observations do not establish a general
training speedup or attribute a speed difference solely to code readability.

Full training compares frozen locked runners against the pushed `ac32723d`
binary baseline, with opt-level 3, fat LTO and one codegen unit. Chinese and
English word counts come from 256 MiB Whitespace corpora, vocabulary target
50000 and minimum frequency 2. Four workers are pinned to CPUs 0–3. Each corpus
excludes one warmup pair and includes four alternating formal pairs; loading and
serialization are outside the timer, while output construction is included.
HWM includes input loading. Builds, tests and the microbenchmark finish before
formal training starts. All twenty parsed-model comparisons pass with zero swap.
Positive paired changes mean an increase:

| Corpus | Wall time | CPU time | HWM |
| --- | ---: | ---: | ---: |
| Chinese | -4.35% | -2.33% | +0.69% |
| English | -1.94% | -2.99% | -0.51% |

The earlier formulation's wall-time results were mixed (+0.17% Chinese, -0.42%
English). The current production budget usually produces a few large ranges;
the microbenchmark difference for that layout is below 0.1 microsecond. These
samples therefore do not attribute the percent-scale training-time change to
this search or establish a stable end-to-end speedup. The dependency is kept for
the chosen search algorithm, with its remaining segmentation rules expressed
locally. Raw manifests, runs and logs are in `training-zh/` and `training-en/`.

`builds.json` records source, runner, dependency-lock and extracted-source hashes.
The initial trial remains in the parent directory with its own frozen sources
and measurements. The source and measurement evidence accompany the training change.
