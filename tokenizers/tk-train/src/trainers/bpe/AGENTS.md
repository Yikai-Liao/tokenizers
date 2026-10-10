# Temporary formatting rule for this BPE change

- Keep a blank line between adjacent functions, including methods and tests.
  Keep a function's attributes and documentation next to that function.
- Format only Rust files changed by the current task. Do not run workspace-wide
  `cargo fmt` or format unchanged files.
- From the repository root, run
  `uv run experiments/bpe-simplification/format-changed.py <changed-file> ...`.
  The formatter applies rustfmt and then adds missing function separators using
  the Rust syntax tree. It only accepts changed files under this directory and
  disables traversal into child modules.
- Add `--check` for a read-only check. The default base is `HEAD`; use
  `--base <PR-base>` to include committed changes from the PR.
- This file and `experiments/bpe-simplification/format-changed.py` are temporary
  and may be removed together before the upstream PR.
