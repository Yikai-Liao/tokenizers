# Initialization and block-count archive

Main report: [INITIALIZATION_MEMORY_REPORT.md](../../INITIALIZATION_MEMORY_REPORT.md).

`bundle.summary.json` contains 35 completed formal calls, source/binary/data hashes, nine workload signatures and resource checks. Workload includes input, split and requested vocabulary. Smokes and failed attempts stay in their original directories and are excluded from formal counts.

| Directory | Formal calls | Source commit | Scope |
|---|---:|---|---|
| `initial-owner-waves` | 3 | `012b8f62` | classic radix installation widths 4/2/1 |
| `block-radix` | 3 | `1740479c` | classic2, block4, block2 |
| `block-radix-stages` | 4 | `32a78a08` | separate sorting/installation, fixed cutoffs |
| `frontier` | 6 | `56dd3227` | direct scatter, bounded heap; final two configurations n=2 |
| `frontier-sort-cost` | 1 | `8bd048f6` | diagnostic sorting/finalization costs |
| `block-count` | 8 | `add44e14` | 16 MiB, u32 local offsets, 2²⁰-slot address block proxy |
| `block-count-stable` | 8 | `add44e14` | 32 MiB, fixed lexical word order, 2²⁴-slot proxy |
| `block-summary-waves` | 2 | `7e794db1` | same binary all summaries vs four-block waves |

Each directory stores an overlay `instrumentation.patch` against its listed commit and its successful build log. `patch-validation.json` records checks in temporary git indexes. `candidate-source.patch` recreates the final candidate `7e794db1` from J `376363d2`; `f16dee08` was the intervening comment-only correction. The candidate patch includes source copyright/license notices. It does not include experimental arena instrumentation.

Test logs include 57 passing final library tests plus earlier meaningful gates. `candidate-range-fallback.tests.legacy-oracle.log` preserves the first failed large-frequency comparison against the original HF i32-frequency implementation. The corrected wide serial oracle and affix fallback pass in their named logs.

Rebuilds must use the historical source commit plus that directory's overlay, rather than current candidate HEAD alone. Raw metadata records actual paths, compiler runner and dependency locks. Proxy tests verify generic dictionary semantics at small physical spans; no run allocated a corpus above 2³² slots. Fixed lexical ordering is diagnostic only.

Plots are standalone PNG/SVG. Run aggregation from `benchmarks/hf-bpe`:

```bash
python3 analyze_initialization.py
uv run --no-project --with matplotlib python analyze_initialization.py --plot
```

Aggregation checks the retained local source overlays and binaries; it does not launch training. Reproduction timings are not expected to equal saved timings.
