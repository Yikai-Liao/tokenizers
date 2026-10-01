#!/usr/bin/env python3
"""Conditional arithmetic, never a throughput or memory measurement."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
reference_path = ROOT / 'results/frontier/zh512m-direct-auto-t256.jsonl'
reference = json.loads(reference_path.read_text().splitlines()[0])
s = reference['indexed_stats']
GIB = 2**30
rows = []
for raw_gib in [32, 100]:
    scale = raw_gib * GIB / reference['input_bytes']
    n, e, u = (s['initial_slots'] * scale,
               s['initial_edges'] * scale, reference['unique_words'] * scale)
    c, p, w = 4*n/GIB, 4*e/GIB, 12*u/GIB
    rows.append(dict(raw_gib=raw_gib, slots=n, initial_edges=e,
                     unique_words=u, corpus_gib=c, initial_postings_gib=p,
                     word_metadata_gib=w,
                     sort16_gib=16*e/GIB, sort24_gib=24*e/GIB,
                     corpus_postings_metadata_gib=c+p+w,
                     init_peak16_including_raw_gib=2*16*e/GIB+c+p+w+raw_gib,
                     init_peak24_including_raw_gib=2*24*e/GIB+c+p+w+raw_gib,
                     historical_postings_3E_gib=12*e/GIB,
                     full_corpus_read_per_30000_rules_tib=c*30000/1024,
                     one_read_one_write_per_rule_30000_tib=2*c*30000/1024))
result = {'kind': 'conditional-arithmetic-not-measurement',
          'reference': str(reference_path.relative_to(ROOT)),
          'reference_input_bytes': reference['input_bytes'],
          'assumptions': ['same unique-word/slot/edge density as reference',
                          'u32 corpus, 4-byte local posting',
                          '12-byte logical mixed-weight word metadata',
                          'all initial postings retained',
                          '2 full sort payloads temporarily coexist',
                          'raw source retained',
                          'pair/PQ/directory/dedup/log overhead excluded'],
          'rows': rows,
          'dense_32byte_ledger_16byte_queue_equal_cardinality_gib': {
              str(k): 48*k/GIB for k in [10_000_000, 100_000_000, 430_853_049]},
          'initial_pair_universe_fixed_V0': s.get('initial_identities', 20757)**2}
print(json.dumps(result, indent=2))
