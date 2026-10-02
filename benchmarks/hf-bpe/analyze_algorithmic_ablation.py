#!/usr/bin/env python3
"""Validate and summarize the frozen six-candidate ablation artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/algorithmic-ablation"
SOURCE = Path("/root/code/tokenizers-worktrees/permanent-boundaries")
REFERENCE = Path("/root/code/tokenizers-worktrees/weight-explanation")
BUILD = ROOT / ".build/native-algorithmic-ablation"
REFERENCE_BINARY = ROOT / ".build/native-weight-intervals-inline/target/release/hf-bpe-native-weight-intervals-inline"
BASELINE_BINARY_SHA256 = "9cd06c0d9fcacf945316b4d2f1ae75b288b173564601a855d45fad9c71589b4c"
CANDIDATE_BINARY_SHA256 = "04af228849603a495168382eafcc8a2867548d7b4485718493797098dada14af"
EXPECTED_MODELS = {
    "en1m": "ece74188f0b6566ab9eecd14f73b2cce6d1f48cc5b6ae9d4f89078ceb810a67a",
    "en16m": "854e76eae8d379f8f2cb26f22363c55c56b6429c595649b31f05c29a55ed7d9e",
    "zh512m": "d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd",
}
MODES = ["none", "boundaries", "lazy-counts", "lazy-births", "positions", "hot256", "final", "all"]
FORMAL_CORPORA = ["en16m", "zh512m"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_commit(path):
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return not dirty, commit


def read_case(case):
    path = RESULTS / f"{case}.jsonl"
    row = json.loads(path.read_text())
    control = json.loads(path.with_suffix(".control.json").read_text())
    environment_path = path.with_suffix(".environment.json")
    environment = json.loads(environment_path.read_text())
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    return row, control, environment, summary, sha256(environment_path)


def validate_case(case, corpus, mode, workers, classification, reference=False):
    row, control, env, summary, env_digest = read_case(case)
    stats = row["indexed_stats"]
    assert row["model_sha256"] == EXPECTED_MODELS[corpus], (case, row["model_sha256"])
    assert control["workers"] == workers and control["unchanged_reference"] == reference
    assert summary["gates"] == "PASS"
    assert env["worktree_clean"] is True
    assert env["binary_sha256"] == (BASELINE_BINARY_SHA256 if reference else CANDIDATE_BINARY_SHA256)
    assert env["input_sha256"]
    assert row["memory"]["sampled_peak_process_swap_bytes"] == 0
    assert row["memory"]["minimum_available_bytes"] >= 1024**3
    alloc = stats["posting_allocations"]
    assert alloc["arena_requested_bytes"] == alloc["arena_retired_bytes"]
    assert alloc["heap_requested_bytes"] == alloc["heap_freed_bytes"]
    assert alloc["heap_buffers"] == alloc["heap_frees"]
    assert stats["workers"] == stats["initialization_workers"] == workers
    assert stats["queue_selection_mode"] == "bulk4" and stats["corpus_weight_order"] == "weight_sorted"
    assert stats["atomic_corpus"] is True and stats["layout"] == "parallel_u32_flat32"
    assert env["worktree_source_sha256"]
    return {
        "case": case,
        "classification": classification,
        "corpus": corpus,
        "mode": mode,
        "workers": workers,
        "model_sha256": row["model_sha256"],
        "train_ms": row["train_ms"],
        "initialize_ms": stats["initialize_ms"],
        "merge_ms": stats["merge_ms"],
        "peak_rss_bytes": max(row["maxrss_kib"] * 1024, row["memory"]["sampled_peak_rss_bytes"]),
        "minimum_available_bytes": row["memory"]["minimum_available_bytes"],
        "vm_swap_bytes": row["memory"]["sampled_peak_process_swap_bytes"],
        "initial": {k: stats[k] for k in ("initial_edges", "initial_slots", "initial_symbols", "initial_pairs")},
        "posting_allocations": alloc,
        "indexed_stats": stats,
        "worktree": env["worktree"],
        "worktree_commit": env["worktree_commit"],
        "worktree_clean": env["worktree_clean"],
        "binary_path": env["binary_path"],
        "binary_sha256": env["binary_sha256"],
        "input_path": env["input_path"],
        "input_sha256": env["input_sha256"],
        "environment_digest_sha256": env_digest,
        "environment_digest_path": str((RESULTS / f"{case}.environment.json").relative_to(ROOT)),
        "gates": "PASS",
    }


def fmt_ms(value):
    return f"{value / 1000:.3f}"


def pct_delta(x, base):
    return 100 * (x / base - 1)


def main():
    rows = []
    for mode in MODES:
        rows.append(validate_case(f"en1m-{mode}-r16000-smoke", "en1m", mode, 4, "smoke"))
    for corpus in FORMAL_CORPORA:
        for mode in MODES:
            rows.append(validate_case(f"{corpus}-{mode}-r16000-formal", corpus, mode, 4, "formal matrix"))
    for corpus, case in [
        ("en16m", "en16m-weight-intervals-inline-reference"),
        ("zh512m", "zh512m-weight-intervals-inline-reference"),
    ]:
        rows.append(validate_case(case, corpus, "none", 4, "fresh unchanged reference", reference=True))
    for mode in ["none", "final"]:
        rows.append(validate_case(f"zh512m-{mode}-r16000-inverse", "zh512m", mode, 4, "adjacent repeat"))
    rows.append(validate_case("en16m-final-r16000-one-thread", "en16m", "final", 1, "one-thread check"))
    rows.append(validate_case("en16m-weight-intervals-inline-r16000-one-thread", "en16m", "none", 1, "one-thread unchanged reference", reference=True))
    assert len(rows) == 30

    binary_sha = sha256(BUILD / "target/release/hf-bpe-native-algorithmic-ablation")
    reference_sha = sha256(REFERENCE_BINARY)
    assert binary_sha == CANDIDATE_BINARY_SHA256 and reference_sha == BASELINE_BINARY_SHA256
    source_clean, source_commit = clean_commit(SOURCE)
    reference_clean, reference_commit = clean_commit(REFERENCE)
    assert source_clean and source_commit == "9d81ffb1b1a1e08ff064155224701ef124588158"
    assert reference_clean and reference_commit.startswith("c9af0cf2")
    input_paths = {
        "en1m": ROOT / ".build/posting-distribution-inputs/en-1m-none-r16000/input.txt",
        "en16m": ROOT / ".build/posting-distribution-inputs/en-16m-none-r16000/input.txt",
        "zh512m": ROOT / ".build/gb-corpus/zh-512m.txt",
    }
    input_sha = {key: sha256(path) for key, path in input_paths.items()}
    for corpus in input_sha:
        expected_input = next(r["input_sha256"] for r in rows if r["corpus"] == corpus)
        assert input_sha[corpus] == expected_input

    by_case = {r["case"]: r for r in rows}
    comparisons = []
    for corpus in FORMAL_CORPORA:
        base = by_case[f"{corpus}-none-r16000-formal"]
        for mode in MODES[1:]:
            candidate = by_case[f"{corpus}-{mode}-r16000-formal"]
            assert candidate["initial"] == base["initial"]
            comparisons.append({
                "corpus": corpus,
                "mode": mode,
                "baseline_case": base["case"],
                "candidate_case": candidate["case"],
                "baseline_train_ms": base["train_ms"],
                "candidate_train_ms": candidate["train_ms"],
                "delta_ms": candidate["train_ms"] - base["train_ms"],
                "delta_percent": pct_delta(candidate["train_ms"], base["train_ms"]),
                "same_binary": True,
            })
    references = []
    for corpus in FORMAL_CORPORA:
        for mode in ["none", "final"]:
            candidate = by_case[f"{corpus}-{mode}-r16000-formal"]
            baseline = by_case[f"{corpus}-weight-intervals-inline-reference"]
            references.append({
                "corpus": corpus, "mode": mode, "candidate_case": candidate["case"],
                "reference_case": baseline["case"], "candidate_train_ms": candidate["train_ms"],
                "reference_train_ms": baseline["train_ms"], "delta_ms": candidate["train_ms"] - baseline["train_ms"],
                "delta_percent": pct_delta(candidate["train_ms"], baseline["train_ms"]),
                "reference_commit": reference_commit,
            })
    reverse = {
        "actual_repeat_order": ["none", "final"],
        "artifact_name_note": "The original case IDs use inverse, but the adjacent repeat actually ran none before final.",
        "initial_pair": {m: by_case[f"zh512m-{m}-r16000-formal"]["train_ms"] for m in ["none", "final"]},
        "repeat_pair": {m: by_case[f"zh512m-{m}-r16000-inverse"]["train_ms"] for m in ["none", "final"]},
        "initial_final_delta_percent": pct_delta(by_case["zh512m-final-r16000-formal"]["train_ms"], by_case["zh512m-none-r16000-formal"]["train_ms"]),
        "repeat_final_delta_percent": pct_delta(by_case["zh512m-final-r16000-inverse"]["train_ms"], by_case["zh512m-none-r16000-inverse"]["train_ms"]),
        "final_skipped_posting_positions": by_case["zh512m-final-r16000-formal"]["indexed_stats"]["skipped_final_posting_positions"],
        "initial_posting_positions": by_case["zh512m-final-r16000-formal"]["indexed_stats"]["posting_visits"],
    }
    reverse["skipped_fraction_percent"] = 100 * reverse["final_skipped_posting_positions"] / reverse["initial_posting_positions"]
    assert reverse["initial_final_delta_percent"] < 0 and reverse["repeat_final_delta_percent"] > 0

    tests = sorted(p.name for p in (RESULTS / "correctness").glob("*.log"))
    build_meta = json.loads((RESULTS / "build-and-tests.json").read_text())
    analysis = {
        "generated_from": "frozen six-candidate ablation results",
        "source_worktree": str(SOURCE), "source_commit": source_commit, "source_clean": source_clean,
        "candidate_binary_sha256_after_all_runs": binary_sha,
        "reference_worktree": str(REFERENCE), "reference_commit": reference_commit, "reference_clean": reference_clean,
        "reference_binary_sha256_after_all_runs": reference_sha,
        "expected_models": EXPECTED_MODELS,
        "input_sha256_after_all_runs": input_sha,
        "build": build_meta,
        "correctness_logs": tests,
        "call_count": {"smoke": 8, "formal_matrix": 16, "fresh_reference": 2, "adjacent_repeat": 2, "one_thread": 2, "total": 30},
        "cases": rows,
        "same_binary_formal_comparisons": comparisons,
        "fresh_unchanged_reference_comparisons": references,
        "zh_final_repeat": reverse,
        "all_checks": "PASS",
    }
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    lines = [
        "# 六个无损算法候选的统一消融报告",
        "",
        "## 结论",
        "",
        "本轮没有建立可稳定优于已推送基线 `48223a58` 的统一组合。保持生产默认不变；六个候选及其组合继续保留在独立实验提交 `9d81ffb1`。30 次原生调用的模型 SHA、posting arena 与 heap 生命周期、工作线程/布局和资源门禁均通过。该结论只适用于下列确定输入、权重降序区间、Bulk(4)、当前自动 arena 和已测实现。",
        "",
        "## 完整训练耗时",
        "",
        "`train_ms` 覆盖完整训练，包括候选的初始化、额外扫描、证书、witness 展开和重建。下表为 4 线程正式矩阵；括号内是相对同二进制 `none` 的变化。小语料只验证开关和模型。",
        "",
        "| 开关 | EN 16 MiB / 4 线程 | 中文 512 MiB / 4 线程 | EN 1 MiB smoke |",
        "|---|---:|---:|---:|",
    ]
    smoke = {r["mode"]: r for r in rows if r["classification"] == "smoke"}
    for mode in MODES:
        if mode == "none":
            en = by_case[f"en16m-none-r16000-formal"]["train_ms"]
            zh = by_case[f"zh512m-none-r16000-formal"]["train_ms"]
            en_cell, zh_cell = f"{fmt_ms(en)} s", f"{fmt_ms(zh)} s"
        else:
            en_cmp = next(c for c in comparisons if c["corpus"] == "en16m" and c["mode"] == mode)
            zh_cmp = next(c for c in comparisons if c["corpus"] == "zh512m" and c["mode"] == mode)
            en_cell = f"{fmt_ms(en_cmp['candidate_train_ms'])} s ({en_cmp['delta_percent']:+.1f}%)"
            zh_cell = f"{fmt_ms(zh_cmp['candidate_train_ms'])} s ({zh_cmp['delta_percent']:+.1f}%)"
        lines.append(f"| `{mode}` | {en_cell} | {zh_cell} | {fmt_ms(smoke[mode]['train_ms'])} s |")
    lines += [
        "",
        "峰值 RSS 正式矩阵范围：EN 为 0.259–0.305 GiB，中文为 3.305–3.669 GiB。中文最小可用内存为 3.67 GiB，进程 VmSwap 始终为 0。其余每个调用也满足至少 1 GiB 可用内存。",
        "",
        "## 冻结基线与相邻复测",
        "",
        "六项候选实验二进制含候选分支和计数插桩；旧冻结二进制没有这些改动，因此另以未修改的权重排序二进制 fresh reference 对照。进一步对照复用已有的 `final` 单项正式记录。EN 16 MiB / 4 线程：实验 `none` 为 1.967 s、`final` 为 1.902 s，未修改参考为 1.937 s；实验相对参考快 1.8%，幅度很小。中文 512 MiB / 4 线程：实验 `none` 为 19.434 s、`final` 为 17.221 s，fresh reference 为 17.928 s。之后的中文相邻复测按 `none → final` 运行（历史文件名中的 `inverse` 只是用例标识），得到 `none` 17.221 s、`final` 17.416 s，方向翻转，因此先前观察到的 11.4% 优势被复测推翻。参考与实验初始 posting 安装阶段分别有明显波动；单次总耗时不能证明收益来自末批跳过。",
        "",
        "末批候选跳过了 19,264 个 posting 位置，占基线 125,409,599 次选择计划 posting 访问的 0.0154%。它的初始化阶段不应受此开关影响；两次正式运行的初始化却为 5.440 s 和 4.241 s。`delta_ms` 为 7.390 s 对 7.487 s，略慢；`commit_ms` 为 5.023 s 对 4.064 s。只将真实嵌套的计时与父阶段分开解释：例如 `corpus_sort_ms` 包含在初始化内，`lazy_recount_ms` 和 `position_certificate_ms` 包含在选择阶段内，`lazy_birth_expand_ms` 位于合并阶段内但在 prepare/commit 外，worker prefetch 计时也包含在现有阶段中。这些子阶段不能再和父阶段重复相加；`initialize_ms` 与 `merge_ms` 是分开的主阶段，`delta_ms`、`rewrite_ms`、`commit_ms` 则按各自边界比较，不把它们直接拼成总耗时。此操作量和方向不支持稳定净收益归因。",
        "",
        "EN 16 MiB / 1 线程：`final` 为 4.315 s，未修改参考为 4.304 s，差约 +0.3%。没有发现明显单线程收益，也没有硬件专用路径的依据。",
        "",
        "## 各候选实际省下与增加的工作",
        "",
        "以下操作计数来自中文单项正式运行；这些数字解释所测原型的代价，不是时间收益的替代指标。",
        "",
        "| 候选原型 | 减少的工作 | 增加的工作或储存 |",
        "|---|---|---|",
    ]
    zhstats = {m: by_case[f"zh512m-{m}-r16000-formal"]["indexed_stats"] for m in MODES}
    s = zhstats["boundaries"]
    lines.append(f"| 永久边界 | 跳过相邻候选检查 {s['boundary_skipped_neighbors']:,} 次 | 退役 posting 扫描 {s['boundary_retired_posting_visits']:,} 次；位图 {s['boundary_mask_bytes']/1024**2:.1f} MiB；低频出生标记 {s['boundary_low_birth_positions']:,} 次 |")
    s = zhstats["lazy-counts"]
    lines.append(f"| 按需计频（全冷上界原型） | 免去旧 pair 逐批扣减、remove 聚合和 ledger 更新 | 重算访问 {s['lazy_recount_posting_visits']:,} 个 posting，过滤无效位置 {s['lazy_recount_invalid_positions']:,} 个，重算计时 {s['lazy_recount_ms']/1000:.2f} s；目录峰值 {s['peak_lazy_count_directory_bytes']/1024**2:.2f} MiB |")
    s = zhstats["lazy-births"]
    lines.append(f"| 延迟出生（全局上界、批量 witness 展开） | 延后 {s['lazy_birth_deferred_positions']:,} 个 posting 位置的物化 | {s['lazy_birth_expansions']} 次展开、访问 witness {s['lazy_birth_witness_visits']:,} 次、展开计时 {s['lazy_birth_expand_ms']/1000:.2f} s；位图 {s['lazy_birth_mask_bytes']/1024**2:.1f} MiB |")
    s = zhstats["positions"]
    lines.append(f"| 位置批次证书 | 批次从 1,550 降至 {s['batch_rounds']:,} | 认证访问 {s['position_certificate_visits']:,} 个位置、认证计时 {s['position_certificate_ms']/1000:.2f} s；接受 {s['position_certificate_accepts']} 个疑似符号冲突候选，拒绝 {s['position_certificate_rejects']} 个候选 |")
    s = zhstats["hot256"]
    edges = by_case["zh512m-none-r16000-formal"]["initial"]["initial_edges"]
    cov256 = 100 * s["initial_hot_coverage_256"] / edges
    cov512 = 100 * s["initial_hot_coverage_512"] / edges
    cov1024 = 100 * s["initial_hot_coverage_1024"] / edges
    lines.append(f"| 热字符初始化（256） | 热区覆盖 {s['initial_hot_coverage_256']:,} 个边，约 {cov256:.1f}% | 256/512/1024 字符覆盖约 {cov256:.1f}% / {cov512:.1f}% / {cov1024:.1f}%；目录 {s['initial_hot_directory_bytes']/1024**2:.1f} MiB，位置缓冲 {s['initial_hot_position_buffer_bytes']/1024**2:.1f} MiB；分类与安装 {s['initial_hot_classify_ms']/1000+s['initial_hot_install_ms']/1000:.2f} s |")
    s = zhstats["final"]
    lines.append(f"| 末批跳过 | 最终批已认证并写入模型后省略 prepare/rewrite/commit 的后续部分 | 跳过 {s['skipped_final_batches']} 批、{s['skipped_final_posting_positions']:,} 个 posting 位置 |")
    lines += [
        "",
        "按需计频实现是全冷上界加候选扫描，没有按热度混合精确维护和惰性维护。延迟出生实现保存全局频率上界，胜出可能时批量展开全部 witness；它没有按父集合局部延迟同步。完整 `all` 在 EN 16 MiB 为 9.163 s，在中文 512 MiB 为 85.185 s，包含这些负收益路径。单项收益不能相加；本轮没有再挑一个组合宣称优于基线。",
        "",
        "## 可复现资料与门禁",
        "",
        f"正式/参考/单线程/复测/smoke 原生调用共 **{len(rows)} 次**。构建耗时 54.42 s。候选源提交 `{source_commit}`，未修改参考提交 `{reference_commit}`。消融二进制 SHA256 `{binary_sha}`；未修改参考二进制 SHA256 `{reference_sha}`。EN1M、EN16M、ZH512M 输入 SHA256 分别为 `{input_sha['en1m']}`、`{input_sha['en16m']}`、`{input_sha['zh512m']}`。",
        "",
        "逐调用控制、环境、stdout/stderr、摘要及原始 JSONL 位于 `results/algorithmic-ablation/`。每个环境记录有源码文件 SHA 列表；`analysis.json` 记录逐项环境文件摘要、模型、输入、二进制 SHA、生命周期/资源门禁和对照。计划中已有的 correctness 日志保留在 `results/algorithmic-ablation/correctness/`，75 项库测试和 4,096 个组合差分此前均通过。",
        "",
        "数据仅覆盖 EN1M/EN16M 与中文 512 MiB、当前随机 hash 布局和所列参数。它没有证明其它语料、非空 affix、不同权重分布、更多线程或其它机器上的结果。",
        "",
    ]
    (ROOT / "ALGORITHMIC_ABLATION_REPORT.md").write_text("\n".join(lines))
    print(f"validated {len(rows)} calls; wrote results/algorithmic-ablation/analysis.json and ALGORITHMIC_ABLATION_REPORT.md")


if __name__ == "__main__":
    main()
