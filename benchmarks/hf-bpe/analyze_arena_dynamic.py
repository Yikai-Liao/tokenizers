#!/usr/bin/env python3
"""Offline dynamic-cutoff memory scenarios from archived lifecycle traces; no training."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/arena-dynamic'
GIB = 1 << 30
MIB = 1 << 20


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(rows, key):
    xs = [math.log(r['physical_edges']) for r in rows]
    ys = [math.log(r[key]) for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)


libc = ctypes.CDLL(None)
libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]
libc.malloc_usable_size.argtypes = [ctypes.c_void_p]
libc.malloc_usable_size.restype = ctypes.c_size_t
padding_cache = {}


def padding(size):
    if size not in padding_cache:
        pointer = libc.malloc(size)
        if not pointer:
            raise MemoryError(size)
        usable = libc.malloc_usable_size(pointer)
        libc.free(pointer)
        padding_cache[size] = usable - size
    return padding_cache[size]


rows = []
for path in sorted((ROOT / 'results/posting-distribution').glob('*-r*.summary.json')):
    d = json.loads(path.read_text())
    assert d['invariant_checks'] == 'PASS' and d['probe_summary']['grows'] == 0
    edge = d['physical_edges']
    cutoff = max(256, math.isqrt(edge // 256))
    caps = [c*4 for c in d['probe_summary']['capacity_bounds']]
    lo = max(j for j, c in enumerate(caps) if c <= cutoff)
    hi = next(j for j, c in enumerate(caps) if c >= cutoff)
    t = d['terminal']
    alive_count = alive_padding = live_payload = 0
    for length, count in d['exact_live_length_histogram']:
        if length <= 2:
            continue
        size = max(length, 4)*4
        if size <= cutoff:
            alive_count += count
            alive_padding += count*padding(size)
            live_payload += count*size
    rows.append(dict(case=path.name.removesuffix('.summary.json'),
        source_sha256=sha(path), language=d['input']['language'], split=d['input']['split'],
        size_mib=d['input']['size_mib_label'], rules=d['actual_rules'], physical_edges=edge,
        input_bytes=d['input']['bytes'], cutoff_bytes=cutoff,
        capacity_bracket_bytes=[caps[lo], caps[hi]],
        retired_lower_bytes=sum(t['retired_bytes'][:lo+1]),
        retired_upper_bytes=sum(t['retired_bytes'][:hi+1]),
        all_floor_retired_bytes=sum(d['probe_summary']['floor_bytes']),
        alive_small_heap_buffers=alive_count, alive_small_heap_requested_bytes=live_payload,
        measured_malloc_padding_bytes=alive_padding,
        modeled_live_allocator_overhead_bytes=alive_padding+8*alive_count))

fits = []
for language in ['en', 'zh', 'de', 'ja']:
    sample = sorted([r for r in rows if r['language'] == language and r['split'] == 'none'
                     and r['rules'] == 16000 and 4 <= r['size_mib'] <= 32],
                    key=lambda r: r['physical_edges'])
    fits.append(dict(language=language, cases=[r['case'] for r in sample],
        retired_upper_gamma=fit(sample, 'retired_upper_bytes'),
        all_floor_retired_gamma=fit(sample, 'all_floor_retired_bytes'),
        alive_small_gamma=fit(sample, 'alive_small_heap_buffers'),
        allocator_overhead_gamma=fit(sample, 'modeled_live_allocator_overhead_bytes')))
zh32 = next(r for r in rows if r['case'] == 'zh-32m-none-r16000')
zh512 = next(r for r in rows if r['case'] == 'zh-512m-none-r16000')
zh_large_retired_gamma = fit([zh32, zh512], 'retired_upper_bytes')
zh_large_overhead_gamma = fit([zh32, zh512], 'modeled_live_allocator_overhead_bytes')

# Calibrate the current 29,243-rule Chinese run with its exact <=256 B
# retained payload, using the same-input 16k-rule capacity bins only for the
# 256->890 B shape. This is a scenario estimate, not a confidence interval.
full_inventory = json.loads((ROOT / 'results/arena-threshold/zh512m-t256.summary.json').read_text())
full_inventory = full_inventory['inventory']
live256 = sum(full_inventory['heap_bytes_histogram'][:5])
full256 = json.loads((OUT / 'zh512m-seeded-256.jsonl').read_text())
full890 = json.loads((OUT / 'zh512m-seeded-auto.jsonl').read_text())
allocated256 = full256['indexed_stats']['posting_allocations']['arena_requested_bytes']
retired256 = allocated256 - live256
old = json.loads((ROOT / 'results/posting-distribution/zh-512m-none-r16000.summary.json').read_text())
oldr = old['terminal']['retired_bytes']
old256 = sum(oldr[:5])
retired890_bracket_estimate = [retired256*sum(oldr[:6])/old256,
                              retired256*sum(oldr[:7])/old256]
# The independent full-run inventory gives a wide exact bracket for the live
# bytes at 890 B; keep it separate from the estimated same-input bin shape.
allocated890 = full890['indexed_stats']['posting_allocations']['arena_requested_bytes']
retired890_hard_bracket = [max(retired256, allocated890-sum(full_inventory['heap_bytes_histogram'][:6])),
                          allocated890-live256]
alive_full_bracket = [sum(full_inventory['heap_capacity_histogram'][:5]),
                      sum(full_inventory['heap_capacity_histogram'][:6])]
alive_full_mid = statistics.mean(alive_full_bracket)
full_overhead_reference = zh512['modeled_live_allocator_overhead_bytes'] * alive_full_mid / zh512['alive_small_heap_buffers']

forecasts = []
for language in ['en', 'zh', 'de', 'ja']:
    f = next(v for v in fits if v['language'] == language)
    base = max([r for r in rows if r['language'] == language and r['split'] == 'none'
                and r['rules'] == 16000], key=lambda r: r['size_mib'])
    if language == 'zh':
        rlo, rhi = retired890_bracket_estimate
        alo = min(f['allocator_overhead_gamma'], zh_large_overhead_gamma)
        ahi = max(f['allocator_overhead_gamma'], zh_large_overhead_gamma)
        r_gamma_lo = zh_large_retired_gamma
        r_gamma_hi = max(zh_large_retired_gamma, f['all_floor_retired_gamma'])
        overhead_reference = full_overhead_reference
        rules_scope = 'current 29,243 rules, calibrated from same-input 16k bins'
    else:
        rlo, rhi = base['retired_lower_bytes'], base['retired_upper_bytes']
        alo = ahi = f['allocator_overhead_gamma']
        r_gamma_lo = f['retired_upper_gamma']
        r_gamma_hi = max(f['retired_upper_gamma'], zh_large_retired_gamma, f['all_floor_retired_gamma'])
        overhead_reference = base['modeled_live_allocator_overhead_bytes']
        rules_scope = '16k rules; other merge budgets may change retention'
    for raw_gib in [1, 4, 16, 32, 100]:
        scale = raw_gib*GIB/base['input_bytes']
        physical_edges = int(base['physical_edges']*scale)
        cutoff = max(256, math.isqrt(physical_edges//256))
        retained = [rlo*scale**r_gamma_lo, rhi*scale**r_gamma_hi]
        savings = [overhead_reference*scale**alo, overhead_reference*scale**ahi]
        # Arithmetic scenarios for allocator-managed resident payload only.
        # RSS, glibc arenas, untouched bump reservation and phase overlap are not
        # simulated. No probability or guarantee is inferred from these bounds.
        net = [retained[0]-savings[1], retained[1]-savings[0]]
        linear_retired = rhi*scale
        forests_item = dict(language=language, raw_gib=raw_gib, rules_scope=rules_scope,
            physical_edges_scenario=physical_edges, cutoff_bytes=cutoff,
            retained_payload_scenario_bytes=retained,
            modeled_live_allocator_overhead_saved_bytes=savings,
            net_retention_minus_allocator_overhead_scenario_bytes=net,
            linear_retention_stress_bytes=linear_retired,
            linear_stress_net_bytes=linear_retired-savings[0],
            corpus_slot_bytes_reference=4*physical_edges,
            posting_blocks_lower_reference=math.ceil(physical_edges/(1<<32)),
            retired_gamma_scenario=[r_gamma_lo, r_gamma_hi],
            allocator_overhead_gamma_scenario=[alo, ahi])
        forecasts.append(forests_item)

# Preserve all completed calls and distinguish formal calls from diagnostics.
formal = []
work_keys = ['initial_symbols', 'initial_slots', 'initial_edges', 'initial_pairs',
             'posting_visits', 'stale_posting_visits', 'pruned_pairs']
for path in sorted(OUT.glob('*seeded*.jsonl')):
    if path.name.startswith('smoke'):
        continue
    d = json.loads(path.read_text())
    formal.append(dict(case=path.stem, path=str(path.relative_to(ROOT)),
        source_sha256=sha(path), model_sha256=d['model_sha256'],
        train_ms=d['train_ms'], initialize_ms=d['initialize_ms'], merge_ms=d['merge_ms'],
        full_peak_rss_bytes=max(d['maxrss_kib']*1024, d['memory']['sampled_peak_rss_bytes']),
        cutoff_bytes=d['indexed_stats']['posting_arena_cutoff_bytes'],
        allocations=d['indexed_stats']['posting_allocations'],
        work={k:d['indexed_stats'][k] for k in work_keys},
        process_swap_bytes=d['memory']['sampled_peak_process_swap_bytes'],
        extra_environment=json.loads(path.with_suffix('.control.json').read_text())['extra_environment']))
for a in [r for r in formal if r['case'].endswith('-auto')]:
    b = next(r for r in formal if r['case'] == a['case'].removesuffix('-auto')+'-0')
    assert a['work'] == b['work'] and a['model_sha256'] == b['model_sha256']
    a['full_time_change_fraction'] = a['train_ms']/b['train_ms']-1
    a['full_peak_change_bytes'] = a['full_peak_rss_bytes']-b['full_peak_rss_bytes']
    assert a['process_swap_bytes'] == b['process_swap_bytes'] == 0

result = dict(method=dict(
    purpose='offline rough memory scenarios for unchanged max(256,isqrt(E/256)); no runtime fallback',
    lifecycle_scope='43 archived flat32 cases, 4 languages, two merge budgets, sizes 1/4/16/32 MiB and one 512 MiB holdout; requested retirement excludes final model teardown',
    cutoff_brackets='sum exact archived capacity bins below and above cutoff; retired bytes bracket is exact for each archived trace',
    current_29k_calibration='exact full-run R256; estimate R890 shape using same-input 16k 512/1024 B ratios; keep wide full-inventory bounds separately',
    allocator_model='local measured malloc_usable_size padding; add nominal 8 B in-use chunk header per live small allocation as a sensitivity model; ignores heap fragmentation and thread-arena page residency',
    extrapolation='fixed rule budget, same-language new unique material; physical edges scaled by largest observed input ratio; use observed local power slopes as scenarios, not a universal law or confidence interval',
    larger_blocks='predicted >2^32 positions use generic block dictionaries; flat histograms do not validate their actual RSS; directory buffers, growth and local-list repartition are unmodeled',
    risk='no quantitative probability or no-peak guarantee; linear retirement is a stress case; chunk reservation is not resident RSS'),
    completed_formal_calls=len(formal), formal_calls=formal,
    archived_lifecycle_case_count=len(rows), lifecycle_cases=rows, local_slopes=fits,
    large_zh_holdout_slopes=dict(retired_upper_gamma=zh_large_retired_gamma,
                                allocator_overhead_gamma=zh_large_overhead_gamma),
    current_zh_29k=dict(exact_retired_256_bytes=retired256,
        estimated_retired_890_bytes=retired890_bracket_estimate,
        full_inventory_retired_890_wide_bounds_bytes=retired890_hard_bracket,
        modeled_live_allocator_overhead_reference_bytes=full_overhead_reference,
        arena_backing_bytes=full890['indexed_stats']['posting_allocations']['backing_bytes'],
        arena_requested_bytes=allocated890),
    malloc_padding_by_requested_size=padding_cache, forecasts=forecasts)
(OUT/'offline-memory-scenarios.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print('formal calls',len(formal),'archived lifecycle cases',len(rows))
print('current 29k R256 MiB',retired256/MIB,'estimated R890 MiB',[v/MIB for v in retired890_bracket_estimate],
      'modeled live overhead MiB',full_overhead_reference/MIB)
for x in forecasts:
    if x['raw_gib'] in [32,100]:
        print(x['language'],x['raw_gib'],'GiB T',x['cutoff_bytes'],
              'retained GiB',[round(v/GIB,2) for v in x['retained_payload_scenario_bytes']],
              'modeled overhead GiB',[round(v/GIB,2) for v in x['modeled_live_allocator_overhead_saved_bytes']],
              'stress net GiB',round(x['linear_stress_net_bytes']/GIB,2))
