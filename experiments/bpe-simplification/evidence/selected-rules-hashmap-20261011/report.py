from pathlib import Path
import json

base = Path('/root/code/tokenizers-workspaces')
fields = ['alphabet_selection_seconds', 'vocabulary_initialization_seconds',
          'alphabet_plus_initialization_seconds', 'train_seconds',
          'train_cpu_seconds', 'process_hwm_kib_before_validation']

def tables(root):
    rows = json.loads((root / 'runs.json').read_text())
    pairs = []
    for case in dict.fromkeys(r['case'] for r in rows):
        group = [r for r in rows if r['case'] == case]
        baseline = next(r for r in group if r['arm'] == 'baseline')
        for r in group:
            if r['arm'] == 'baseline':
                continue
            assert r['model_sha256'] == baseline['model_sha256']
            pairs.append(dict(case=case, arm=r['arm'], full_model_equal=True,
                              model_sha256=r['model_sha256'],
                              delta_percent={f: (r[f] / baseline[f] - 1) * 100 for f in fields}))
    (root / 'pairs.json').write_text(json.dumps(pairs, indent=2) + '\n')
    text = '| Case | Arm | Alphabet s | Vocab init s | Combined s | Whole train s | Train CPU s | HWM MiB |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        text += f"| {r['case']} | {r['arm']} | {r[fields[0]]:.6f} | {r[fields[1]]:.6f} | {r[fields[2]]:.6f} | {r[fields[3]]:.6f} | {r[fields[4]]:.6f} | {r[fields[5]] / 1024:.2f} |\n"
    text += '\nDeltas against the baseline in the same case; negative time means faster.\n\n| Case | Arm | Alphabet | Vocab init | Combined | Whole train | HWM |\n|---|---|---:|---:|---:|---:|---:|\n'
    for p in pairs:
        d = p['delta_percent']
        text += f"| {p['case']} | {p['arm']} | {d[fields[0]]:+.2f}% | {d[fields[1]]:+.2f}% | {d[fields[2]]:+.2f}% | {d[fields[3]]:+.2f}% | {d[fields[5]]:+.2f}% |\n"
    text += '\nProcess and host paging over the entire child lifetime, including load and model export. These are not restricted to the train timer. Host swap counts include unrelated processes.\n\n| Case | Arm | Child major faults | Max child swap KiB | Host swap-in pages | Host swap-out pages |\n|---|---|---:|---:|---:|---:|\n'
    for r in rows:
        text += f"| {r['case']} | {r['arm']} | {r['child_major_faults']} | {r['child_max_swap_kib']} | {r['system_swapin_pages']} | {r['system_swapout_pages']} |\n"
    return text, pairs

protocol = '''Four workers pinned to CPUs 0–3; all samples run serially. The release runner uses opt-level 3, fat LTO, one codegen unit, and no tk-train default features. Exact source snapshots, dependency locks, binaries, models, build logs and raw job/stdout/stderr files are retained in the durable directory named below. Compilation and tests did not overlap timed samples.

The raw Chinese and English corpora are about 256 MiB each. Both use the same public Whitespace preprocessing; input and prepared-file hashes are in `inputs.json`. The fixed prepared words are reconstructed into the same borrowed caller AHashMap with seeds [11,13,17,19]. This fixes word traversal and affix ID assignment. Load, preprocessing, and serialization are outside `train_seconds`; this is public `do_train` time, not Feed-plus-train pipeline time. The Chinese prepared file has 6,827,958 unique words; English has 724,596. Limits 2047/128 have actual frequency cutoffs without a tie at the boundary. Vocabulary size is 50,000 and min frequency is 2.

Plain cases have no affixes. Affix cases set **both nonempty prefix `##` and suffix `</w>`** and exercise prefix, suffix, and combined decoration slots. Every case compares the entire vocabulary in ID order and every ordered merge, using byte-identical exported model files, not only counts. Native tokenizer JSON or encoding parity is not claimed by this performance export.

Alphabet selection is measured inside `select_alphabet`. Vocabulary initialization ends immediately after `Vocabulary::initialize`, before `CorpusPlan::build`; it excludes selection. Combined is selection plus that initialization. Whole train includes both stages and the remaining training work. HWM is the process peak RSS sampled before validation/export; it also includes prepared-input loading and is not a phase-specific peak. CPU seconds are summed process CPU during the train timer. Stage logging is diagnostic-only; production code has no new timers.
'''

affix = base / 'bpe-affix-verification-results-20261011'
selected = base / 'bpe-selected-rules-results-20261011'
affix_table, _ = tables(affix)
selected_table, _ = tables(selected)

(affix / 'README.md').write_text('''# Nonempty-affix verification and separate training stages

This compares `38b283c0` (before the flags/alphabet changes) with published `0dd1ca48` (current). `flags-only` applies only the named-bool/three-slot decoration change to `38b283c0`; it retains the old alphabet selection and activation behavior. This round makes no production Vocabulary changes.

The previous initialization figures described a stage containing alphabet selection, and must not be described as whole-training gains. With fixed input traversal here, the plain combined stage improves while alphabet selection itself becomes slower. The separate table gives the whole-training result.

Both-affix cases show no comparable initialization acceleration. The current combined stage is slower for Chinese and for the English repeat, while whole-train results are close to baseline. The isolated flags initialization is slower in both Chinese samples and both English samples (about 6–13%). Its whole-train deltas change direction in both reverse-order repeats, so this is an observed initialization regression rather than an established whole-train regression. The initial flags-only whole-training slowdowns cannot be dismissed from the plain case: reverse-order repeats are included below. These few samples do not establish a stable affix throughput gain or rule out a small initialization regression. No sample is removed because of its direction or host paging.

''' + protocol + '''
For the old architecture, the raw Vocabulary timer includes selection; the reported exclusive initialization subtracts the separately measured selection. The current architecture selects at entry and the raw Vocabulary timer already excludes selection. The same hooks capture the boundary once, and the driver checks stage sums. All performance cases finish in one attempt.

The first ten samples use baseline/current for Chinese, current/baseline for English, then flags-only for each decorated case. The English decorated repeat reverses to flags/current/baseline. The Chinese decorated repeat also uses flags/current/baseline to check the initial flags slowdown.

A separate `perf stat` diagnostic enables user-mode cycles/instructions/branches/branch-misses between the selection log and initialization log, using an external FIFO controller. Compared with the old flags path, the isolated named-bool path executes about 4.4% more instructions and 7.6% more branches for Chinese, and about 5.6% more instructions and 9.7% more branches for English. Branch misses do not increase. This supports extra cached-lookup control work rather than new string allocations or worse branch prediction. These are approximate external-control windows, especially for the short English stage; the enable acknowledgment can take 4–34 ms. They are diagnostics, not the primary timing samples. Raw counters, commands, controller times, and full models are retained under `profile-init/` and summarized in `profile-init.json`.

Durable artifacts: `/root/code/tokenizers-workspaces/bpe-affix-verification-results-20261011`. `runs.json`/CSV retain all timings and memory data; `pairs.json` contains all deltas. `artifact-hashes.json` records frozen files; raw models remain outside Git.

''' + affix_table)

(selected / 'README.md').write_text('''# Selected rules stored as a plain HashMap

Candidate production commit `b4599744` is based on current `0dd1ca48` in the independent worktree `/tmp/bpe-selected-rules-hashmap-20261011`. Only `merge.rs` changes. Ordinary preparation holds the pair-to-replacement AHashMap directly; `Head`, preparation heads/tails arrays, and `SelectedRules` are removed. Production code adds 12 lines and removes 66, net −54. Selection conflict sets remain. AA and reuse handling, Positions storage and task splitting, Source, Birth, and neighbor event order are unchanged. This experiment has not been merged into the published simplification branch and does not incorporate the other computer's Positions changes.

Independent implementation review and validation are in `review.md` and `validation.json`. Both default and no-default BPE configurations pass 43 tests, including the new shared-head/shared-tail exact-neighbor-events regression, and retain one previously reported encoding-roundtrip failure. Complete raw test stdout was not saved by the implementation agent; that limitation is recorded. Clippy with all targets/no default features and `-D warnings`, formatting, and whitespace checks pass.

This comparison is current `0dd1ca48` versus the map candidate with identical diagnostic stage hooks and input protocol. The preparation arrays no longer allocate, while ordinary neighbor resolution performs extra exact-pair hashing and may read an extra corpus endpoint. Four case pairs alone support only measured deltas, not a stable general speedup claim. Changes in alphabet and initialization timers are noise for this experiment: those production stages are byte-identical in both arms.

''' + protocol + '''
Order is baseline/candidate for Chinese and candidate/baseline for English. The Chinese affix repeat reverses to candidate/baseline because the first pair was 3.4% slower. The repeat is 0.08% slower, with about 2% more train CPU time; the first wall-time difference is not stable across the two pairs. The baseline binary is the exact current binary used in the affix report. Its source snapshot is retained there; candidate source is frozen here. `production-source-hashes.json` ties the measured merge code to the tested production commit.

Separate AA and reuse controls compare full models. The long-AA control runs one attempt; the forced alias/reuse control uses suffix `a` and must run two attempts in each arm. Their tiny times are not performance results. Raw jobs, models, and logs are retained under `controls/`, with outcomes in `controls.json`.

Durable artifacts: `/root/code/tokenizers-workspaces/bpe-selected-rules-results-20261011`. `runs.json`/CSV retain all stages, whole-train CPU/wall time and memory; `pairs.json` retains deltas. `artifact-hashes.json` records frozen sources/binaries/models. The copied baseline initially lacked execute permission, so its first startup returned 126 before training; that failed job is retained under `failed-startup-permission/`. It is excluded from timed samples because it never ran. The execute bit was corrected without changing binary bytes.

''' + selected_table)
