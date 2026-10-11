# Alphabet initialization cleanup

This change removes the decoration flag arithmetic, resolves alphabet selection once at the training entry, and uses retained presence to activate plain characters. Fresh/reuse attempts borrow the same immutable `Alphabet`.

## Behavior and state

- Decoration slots stay prefix=0, suffix=1, both=2; no decoration returns the plain ID. The existing three-slot table and reusable string buffer remain. `add_prefix` and `add_suffix` replace numeric flags; no decoration wrapper type is added.
- Limited selection keeps weighted frequencies, initial-alphabet priority, the existing tie selector, and codepoint insertion order. Presence includes zero-weight input words. The unlimited path still uses parallel presence without frequencies.
- Plain activation happens while reserving characters. Limited plain input skips the later word scan, including when configured affixes are empty strings. Nonempty affixes still resolve decorated IDs in original input order and original UTF-8 coordinates.
- Single-character special tokens outside the selected alphabet retain their original activation and symbol behavior. Multi-character special tokens remain inactive as input symbols.
- `complete_alphabet` remains separate from plain activation: unlimited input can count characters directly, limited input checks character IDs even though its activation scan is skipped.
- Moving selection out of `Vocabulary::initialize` removes the mutable `Option` / `get_or_insert_with` protocol. The baseline already selected a limited alphabet once; this ownership change is not presented as reducing repeated limited selection.

## Memory lifetime

The limited selector now uses a 139,264-byte (136 KiB) presence bitmap while accumulating frequencies. It converts presence into selected `(char, bool)` entries and observed single-character special entries before returning. `Alphabet` contains no bitmap; selection scratch is dropped before vocabulary/ID-table/corpus allocation. Allocator retention can still affect RSS.

On this target `(char, bool)` occupies 8 bytes, verified by `layout.rs`. These entries remain until all attempts finish. The old limited cache stored 4-byte characters: selected-entry storage rises from 8,188 to 16,376 bytes for the Chinese cutoff of 2,047, and from 512 to 1,024 bytes for the English cutoff of 128, excluding vector capacity/metadata. The unlimited path now also retains the selected entry vector across attempts; it still releases its presence bitmaps at selection return.

Peak process RSS did not fall in these samples. The English increase was explicitly accepted by the user. The bitmap lifetime and scan reduction are the intended tradeoff; no overall memory reduction is claimed.

## Checks

Both default and no-default BPE test configurations: 42 passed, one known baseline failure (`feed_training_tokenizer_json_and_encoding_roundtrip`), seven unrelated tests filtered out. The failure is the pre-existing tokenizer encoding roundtrip issue, also documented at base commit 38b283c0. Clippy with all targets and `-D warnings`, changed-file format check, and diff whitespace check passed.

Two added regression tests cover all three decoration combinations, no/empty/nonempty affixes, zero-weight presence, forced unused characters, limit zero, and single-character special tokens outside the cutoff. Existing tests cover an actual reuse restart with a frequency tie at the cutoff, UTF-8 filtering boundaries, sequential-model parity, and worker counts 1/4/8.

## Measurement

Base: `38b283c0b82a475c6a888eadd01fd32d2a97dfc5`. Both measurement worktrees use the same runner parameter for `limit_alphabet` and the same timing boundary: alphabet selection plus `Vocabulary::initialize`, before corpus-plan building. In the baseline selection occurs inside initialization; in the candidate its earlier selection duration is added. These lightweight timing hooks are confined to measurement worktrees. Production sources are uninstrumented.

Public Feed + train_vocab, Whitespace, 4 workers pinned to CPUs 0–3, vocabulary size 50,000, minimum frequency 2, no affixes. Release opt-level=3, fat LTO, one codegen unit. Chinese input 268,435,162 bytes; English 268,434,441 bytes. Actual limits 2,047 and 128 filter 17,759 and 5,083 observed nonspace characters respectively. Boundary counts differ on each side of the cutoff to avoid an ambiguous frequency tie; input identities and boundaries are in `inputs.json`. Production hash randomization remains enabled.

First order: Chinese baseline then candidate; English candidate then baseline. The English repeat reverses the order to check the initial time/RSS difference. No other cases or worker matrices were added. Ordinary shared OS cache; corpus reads for frequency-boundary discovery before measurement. No cache drops, swap disabling, global memory tuning, or background workload changes.

| Case | Selection + initialization, baseline → candidate | Change | Train, baseline → candidate | Pipeline, baseline → candidate | Peak RSS MiB, baseline → candidate |
|---|---:|---:|---:|---:|---:|
| zh | 3.152066 → 1.821061 s | -42.2% | 23.606 → 21.703 s | 27.314 → 25.317 s | 1734.15 → 1744.05 |
| en | 0.081633 → 0.055315 s | -32.2% | 1.522 → 1.735 s | 5.188 → 5.371 s | 140.80 → 149.95 |
| en repeat | 0.081367 → 0.058233 s | -28.4% | 1.792 → 1.597 s | 5.642 → 5.302 s | 139.81 → 152.77 |

Initialization improved in every paired sample: Chinese about 42%, English about 28–32%. English total training time changed direction on the repeat; the stage benefit is clearer than the overall timing with this sample count.

Every sample produced byte-identical complete vocabulary-by-ID and ordered merges within its language, including the English repeat. Feed unique-word count and unique UTF-8 byte totals also matched. All benchmark-child VmSwap and major-fault samples were zero. Global process-window swap activity remained: Chinese baseline 12 pages in / 128 out, Chinese candidate 8 in / 0 out; English first pair 0/0, repeat candidate 2 in / 0 out. These global counters include process startup and output, and are retained as background context; no claim of a globally paging-free host is made.

## Artifacts and reproduction

`runs.csv` / `runs.json` retain each baseline and candidate sample; `pairs.csv` / `pairs.json` retain both values and paired deltas. `validation.json`, raw check logs, source hashes, measurement patches and runner scripts accompany this report. The larger archive with frozen measurement sources, exact locks, binaries, raw model outputs and logs is `/root/code/tokenizers-workspaces/bpe-alphabet-results-20261011`.

To reproduce, make separate worktrees at the base; apply the production diff to the candidate, then apply the matching recorded measurement modifications. Copy the recorded lockfiles, run `build.py`, then `measure.py`. `measure-en-repeat.py` performs the additional reversed English pair. Paths in scripts identify the recorded local inputs and worktrees. All source/binary/input hashes are retained.
