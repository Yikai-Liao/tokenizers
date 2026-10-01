"""Benchmark-only worker timing, task traces, and perf merge-window control."""

def install(source, phase_controls=False):
    parallel = source/'tk-train/src/trainers/bpe/indexed/parallel.rs'
    text = parallel.read_text()
    anchor = '    let begin = Instant::now();\n    let bench_fused ='
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    let bench_diagnostics = std::env::var("HF_BPE_FUSED_DIAGNOSTICS").as_deref() == Ok("1");
    let bench_cpu_begin = if bench_diagnostics { pool.broadcast(|_| bench_thread_cpu_ms()) } else { Vec::new() };
    bench_perf_control("enable");
    let begin = Instant::now();
    let bench_fused =''')
    anchor = '    stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    bench_perf_control("disable");
    stats.merge_ms = begin.elapsed().as_secs_f64() * 1000.0;
    if bench_diagnostics {
        stats.bench_merge_worker_cpu_ms = pool.broadcast(|_| bench_thread_cpu_ms()).into_iter()
            .zip(bench_cpu_begin).map(|(end, begin)| end - begin).collect();
    }''')
    text += '''
fn bench_thread_cpu_ms() -> f64 {
    #[repr(C)]
    struct Timespec { sec: std::ffi::c_long, nsec: std::ffi::c_long }
    unsafe extern "C" { fn clock_gettime(clock: std::ffi::c_int, value: *mut Timespec) -> std::ffi::c_int; }
    let mut value = Timespec { sec: 0, nsec: 0 };
    assert_eq!(unsafe { clock_gettime(3, &mut value) }, 0);
    value.sec as f64 * 1000.0 + value.nsec as f64 / 1_000_000.0
}
fn bench_perf_control(command: &str) {
    use std::io::{Read, Write};
    let Ok(control) = std::env::var("HF_BPE_PERF_CONTROL") else { return; };
    let ack = std::env::var("HF_BPE_PERF_ACK").unwrap();
    let mut writer = std::fs::OpenOptions::new().write(true).open(control).unwrap();
    writeln!(writer, "{command}").unwrap();
    let mut reader = std::fs::File::open(ack).unwrap();
    let mut byte = [0];
    loop { reader.read_exact(&mut byte).unwrap(); if byte[0] == b'\\n' { break; } }
}
'''
    parallel.write_text(text)

    if phase_controls:
        text = parallel.read_text()
        text = text.replace('    bench_perf_control("enable");\n    let begin', '''    let bench_perf_phase = match std::env::var("HF_BPE_PERF_PHASE").as_deref() {
        Ok("kernel") => "kernel", Ok("commit") => "commit", _ => "merge",
    };
    stats.bench_perf_phase = bench_perf_phase;
    if bench_perf_phase == "merge" { bench_perf_control("enable"); }
    let begin''')
        text = text.replace('    bench_perf_control("disable");\n    stats.merge_ms',
            '    if bench_perf_phase == "merge" { bench_perf_control("disable"); }\n    stats.merge_ms')
        text = text.replace('        let outputs = if bench_fused {',
            '        if bench_perf_phase == "kernel" { bench_perf_control("enable"); }\n        let outputs = if bench_fused {')
        text = text.replace('        stats.flat_route_group_visits +=',
            '        if bench_perf_phase == "kernel" { bench_perf_control("disable"); }\n        stats.flat_route_group_visits +=')
        text = text.replace('        let commits: Vec<_> = pool.install(|| {',
            '        stats.bench_dense_commit_batches += usize::from(flat && !outputs[0].flat_births.is_empty());\n'
            '        if bench_perf_phase == "commit" { bench_perf_control("enable"); }\n        let commits: Vec<_> = pool.install(|| {')
        text = text.replace('        stats.commit_ms += stage.elapsed()',
            '        if bench_perf_phase == "commit" { bench_perf_control("disable"); }\n        stats.commit_ms += stage.elapsed()')
        parallel.write_text(text)
        stats = source/'tk-train/src/trainers/bpe/indexed.rs'
        text = stats.read_text().replace('pub(super) struct IndexedTrainingStats {',
            'pub(super) struct IndexedTrainingStats {\n    pub bench_perf_phase: &\'static str,\n    pub bench_dense_commit_batches: usize,')
        stats.write_text(text)

    fused = source/'tk-train/src/trainers/bpe/indexed/parallel/fused_rewrite.rs'
    text = fused.read_text()
    # The clock and flag are outside the workers; optional trace I/O is absent
    # from formal timings. The ordinary scalar worker elapsed metric stays on.
    anchor = '    let parallel = Instant::now();'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    let bench_trace = std::env::var("HF_BPE_FUSED_DIAGNOSTICS").as_deref() == Ok("1");
    static BENCH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let bench_id = if bench_trace { BENCH_ID.fetch_add(1, AtomicOrdering::Relaxed) } else { 0 };
    let parallel = Instant::now();''')
    anchor = '            Ok((\n                output,'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''            if bench_trace {
                eprintln!("{}", serde_json::json!({"bench_task": {"phase": "fused", "batch": bench_id,
                    "worker": rayon::current_thread_index(), "job": job, "visits": partition.visits,
                    "start_ms": begin.duration_since(parallel).as_secs_f64() * 1000.0,
                    "end_ms": parallel.elapsed().as_secs_f64() * 1000.0}}));
            }
            Ok((
                output,''')
    fused.write_text(text)

    if phase_controls:
        text = fused.read_text().replace('HF_BPE_FUSED_DIAGNOSTICS', 'HF_BPE_FUSED_TRACE')
        text = text.replace('    let parallel = Instant::now();',
            '    let bench_dense = GROUPED && std::env::var("HF_BPE_DENSE_COMMIT").as_deref() == Ok("1");\n    let parallel = Instant::now();')
        text = text.replace('if GROUPED { output.enable_dense_births(rules.len()); }',
            'if bench_dense { output.enable_dense_births(rules.len()); }')
        anchor = '''                    left_cache.flush_dense(&mut output, rule, true, list.rank)?;
                    right_cache.flush_dense(&mut output, rule, false, list.rank)?;'''
        assert text.count(anchor) == 1
        text = text.replace(anchor, '''                    if bench_dense {
                        left_cache.flush_dense(&mut output, rule, true, list.rank)?;
                        right_cache.flush_dense(&mut output, rule, false, list.rank)?;
                    } else {
                        left_cache.flush(&mut output, rule, true)?;
                        right_cache.flush(&mut output, rule, false)?;
                    }''')
        fused.write_text(text)

    legacy = source/'tk-train/src/trainers/bpe/indexed/parallel/fused_batch.rs'
    text = legacy.read_text()
    anchor = '    let prepared: Vec<_> = jobs'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    let bench_trace = std::env::var("HF_BPE_FUSED_DIAGNOSTICS").as_deref() == Ok("1");
    static BENCH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let bench_id = if bench_trace { BENCH_ID.fetch_add(1, AtomicOrdering::Relaxed) } else { 0 };
    let bench_parallel = Instant::now();
    let prepared: Vec<_> = jobs''')
    anchor = '        .map(|tasks| -> Result<_> {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n            let bench_begin = Instant::now();')
    anchor = '            Ok((valid, output, left_cache.bytes() + right_cache.bytes()))'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''            if bench_trace {
                eprintln!("{}", serde_json::json!({"bench_task": {"phase": "baseline_prepare", "batch": bench_id,
                    "worker": rayon::current_thread_index(), "visits": tasks.iter().map(|t| t.positions.len()).sum::<usize>(),
                    "start_ms": bench_begin.duration_since(bench_parallel).as_secs_f64() * 1000.0,
                    "end_ms": bench_parallel.elapsed().as_secs_f64() * 1000.0}}));
            }
''' + anchor)
    # Keep the prepare identifier for its separate rewrite phase.
    anchor = 'pub(super) struct Prepared<O: Offset, const INLINE: usize> {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n    bench_id: u64,')
    anchor = '    Ok(Prepared {\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '        bench_id,\n')
    anchor = '        self.valid.par_iter().for_each(|job| {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''        let bench_trace = std::env::var("HF_BPE_FUSED_DIAGNOSTICS").as_deref() == Ok("1");
        let bench_parallel = Instant::now();
        self.valid.par_iter().for_each(|job| {
            let bench_begin = Instant::now();''')
    anchor = '            }\n        });\n    }\n}'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''            }
            if bench_trace {
                eprintln!("{}", serde_json::json!({"bench_task": {"phase": "baseline_rewrite", "batch": self.bench_id,
                    "worker": rayon::current_thread_index(), "visits": job.iter().map(|v| v.positions.len()).sum::<usize>(),
                    "start_ms": bench_begin.duration_since(bench_parallel).as_secs_f64() * 1000.0,
                    "end_ms": bench_parallel.elapsed().as_secs_f64() * 1000.0}}));
            }
        });
    }
}''')
    legacy.write_text(text)
    if phase_controls:
        legacy.write_text(legacy.read_text().replace('HF_BPE_FUSED_DIAGNOSTICS', 'HF_BPE_FUSED_TRACE'))
