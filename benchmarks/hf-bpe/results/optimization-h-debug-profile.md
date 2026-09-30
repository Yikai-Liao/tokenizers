# H debug profile: bounded source attribution

This is a diagnostic profile of the unchanged H source (`weight-one-buckets`, commit `00216d914186e42458d45e72276b13c700749c6a`) using a separate opt-3 release binary built with debug level 2 and stripping disabled (the compilation units report DWARF version 4). The formal H binary was not replaced or modified: SHA-256 `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd`. Debug binary SHA-256 `b712704589be5228a7ba1295d39f989993ac1d987b8f96197f9504c30b701d34`, Build ID `e80ac816f0da8a7b8019205f97b665ba08d13045`. `readelf` found `.debug_info` and `.debug_line`; representative `addr2line -afiC` checks resolved Rust source and inline frames.

The call used the same 536,870,289-byte corpus (SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`), reference backend, `none` pre-tokenizer, vocab 50,000/min frequency 2, initialization/merge workers 4/4, atomic corpus, and u32 layout. The full model signature matched H: `d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`.

## Profile result

`perf record -F 99 -e cycles:u -e cache-misses:u --call-graph dwarf,16384` recorded 8,084 cycles samples and 7,813 cache-miss samples, with **zero lost samples**. Weighted event periods were 199,974,980,103 cycles and 1,411,570,990 cache misses. Top-IP symbols were unresolved for 9.55% of cycle period and 9.63% of miss period. These are sampled event shares across the full measured process, not wall-clock shares or phase timers.

Top-IP attribution (exclusive sample IP, weighted by event period):

| Top IP function | Cycles | Cache misses |
|---|---:|---:|
| `Output<u32, 2>::birth` | 5.35% | 1.69% |
| `Output<u32, 2>::remove` | 5.46% | 3.25% |
| hashbrown `RawTable` functions | 3.40% | 7.41% |
| `cfree` | 3.56% | 1.76% |

The two `Output` rows total 10.81% of cycle samples, but **that is not all hash cost**: these functions include map entry/probing, group updates, and route-node work. The 3.40% cycle / 7.41% miss RawTable aggregate is almost entirely table growth/rehashing rather than steady-state probes. Direct `reserve_rehash` top IPs account for 3.256% cycles / 7.176% misses: `BirthGroup` growth 2.102% / 4.121%, `(u64,(u64,u32))` growth 0.799% / 2.149%, parallel `Entry` growth 0.187% / 0.844%, and `(CompactString,u64)` growth 0.168% / 0.062%. These are different exclusive IPs from the `Output` function IPs in the table. The selected inline `find_inner`/`match_tag` samples below establish route-map probing, but their total share was not isolated. Top-IP categories in the table are mutually exclusive; inclusive call-chain shares are separate and overlap. Inclusive shares were 50.51% cycles / 48.48% misses in the owner/training worker closure and 34.08% / 24.30% in fused prepare. Do not sum inclusive categories with one another or with top-IP shares.

The largest source-resolved probe samples map into inline hashbrown and SSE2 code:

- Birth offset `Output::birth + 0x745` contributed 1.43% of cycle period. Its inline chain is `RawTableInner::find_inner` (`hashbrown/raw.rs:2034`), `RawTable::find` (`raw.rs:1202`), `HashMap::rustc_entry` (`rustc_entry.rs:37`), `HashMap::entry` (`std::collections/hash/map.rs:1014`), then `Output::birth` (`parallel.rs:308`). Other hot offsets map to SSE2 `Group::match_tag` (`hashbrown/control/group/sse2.rs:84`) within that same probe chain.
- Remove offset `Output::remove + 0x14b` contributed 1.81% of cycle period and resolves through the same probe path to `Output::remove` (`parallel.rs:294`). AHash hash computation also appears inline under remove (`hashbrown/map.rs:241`, AHash `fallback_hash.rs:96`, `Hasher::write_u64`).
- Other direct birth offsets include `+0x7de` at `parallel.rs:310` and `+0x7e3` at `:312`, covering per-group updates / node-position construction. The birth call then appends one physical node and links it to the group's chain (`:309–315`). Remove updates only the group weight after the entry lookup (`:294–299`).

The `cfree` top-IP samples are 3.56% of cycle period. A filtered caller check attributes 2.02% cycles / 1.21% miss period to `Vec<Owner>::drop`, plus 1.02% cycles / 0.39% misses to an owner/training closure. This confirms owner-ledger destruction is sampled work, but whole-process sampling cannot say how much wall time it occupies inside the post-merge interval.

## Timer boundary and limits

The debug run measured train 28.569 s, initialize 6.933 s, merge 17.156 s. Therefore the unaccounted interval after initialization and merge is **4.480 s**. The initialization timer already includes tokenization; it is not subtracted again. In source, `stats.merge_ms` ends at `parallel.rs:1375`, before vocab/merge string construction and function-scope local destruction. This profile suggests owner teardown as one contributor, but it cannot uniquely divide the 4.480 s between string conversion, `Owner`/posting destruction, corpus buffers, and report/return work. A separate phase-boundary/count probe is required for that decision.

The earlier H profile from the formal non-DWARF binary remains function/symbol-offset evidence only. `perf --call-graph dwarf` captures stack data; it does not add debug information to a binary.

## Artifacts

- Run row: `optimization-h-debug-profile.jsonl`
- Runner environment and provenance: `optimization-h-debug-profile.environment.json`
- Standard output/error: `optimization-h-debug-profile.stdout` and `.stderr`
- Raw perf data: `../.build/optimization-h-debug-profile.perf`
- Bounded sample / IP scratch data: `../.build/optimization-h-debug-profile.ip-only.txt` and `.topip.txt`
- Target binary metadata: `optimization-h-debug-profile.target.json`
