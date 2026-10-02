#!/usr/bin/env python3
"""Compare completed NONE metadata; optionally census each input once; never launch native runs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_METADATA_BYTES = 1 << 20


def metadata(path):
    path = Path(path).resolve()
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds 1 MiB: {path}")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def case(path):
    path = Path(path).resolve()
    row, digest = metadata(path)
    env_path = Path(row.get("env_metadata", path.with_suffix(".environment.json")))
    env, env_digest = metadata(env_path)
    if row.get("environment_sha256") != env_digest:
        raise ValueError(f"environment digest mismatch: {path}")
    stats = row["indexed_stats"]
    if not isinstance(stats, dict):
        raise ValueError(f"missing indexed stats: {path}")
    if env.get("prefix") is not None or env.get("suffix") is not None:
        raise ValueError("language comparison requires NONE on both sides")
    if env.get("failure") is not None or env.get("returncode") != 0:
        raise ValueError(f"run did not complete successfully: {path}")
    if row["input"] != env["corpus_path"] or row["input_bytes"] != env["corpus_bytes"]:
        raise ValueError(f"input metadata mismatch: {path}")
    for key in ("vocab_size", "min_frequency"):
        if row[key] != env[key]:
            raise ValueError(f"configuration mismatch: {key}: {path}")
    if row["model_sha256"] != env["expected_model_sha256"]:
        raise ValueError(f"model signature mismatch: {path}")
    manifest_path = Path(env["build_manifest_path"])
    manifest, manifest_digest = metadata(manifest_path)
    if manifest_digest != env["build_manifest_sha256"]:
        raise ValueError(f"build manifest digest mismatch: {path}")
    for key in ("binary_path", "binary_sha256", "worktree_commit"):
        if env[key] != manifest[key]:
            raise ValueError(f"manifest provenance mismatch: {key}: {path}")
    values = {
        key: row[key] for key in (
            "input_bytes", "unique_words", "actual_vocab", "actual_merges",
            "feed_ms", "train_ms", "elapsed_ms", "wall_seconds", "peak_rss_bytes",
        )
    }
    for key in (
        "initialize_ms", "merge_ms", "alphabet_ms", "tokenize_ms",
        "initial_route_ms", "initial_count_ms", "initial_radix_sort_ms",
        "initial_group_count_ms", "initial_posting_install_ms", "select_ms",
        "plan_ms", "delta_ms", "fused_prepare_ms", "rewrite_ms", "commit_ms", "route_ms",
        "initial_symbols", "initial_edges", "initial_pairs", "posting_visits",
        "initial_slots", "corpus_bytes", "initial_posting_bytes",
        "initial_pair_table_bytes", "initial_route_buffer_bytes",
        "peak_initial_route_buffer_bytes", "initial_group_buffer_bytes",
        "batch_rounds", "fused_batches", "max_batch_rules", "weight_interval_count",
        "queue_owner_probes", "queue_truth_checks", "queue_stale_corrections",
    ):
        values[key] = stats[key]
    for key, value in stats["posting_allocations"].items():
        values[f"posting_allocations.{key}"] = value
    # Remainder is a boundary difference; it has no attributed operation name.
    values["train_minus_initialize_minus_merge_ms"] = (
        row["train_ms"] - stats["initialize_ms"] - stats["merge_ms"]
    )
    values["vocab_minus_merges"] = row["actual_vocab"] - row["actual_merges"]
    values["symbols_per_unique_line"] = stats["initial_symbols"] / row["unique_words"]
    values["posting_visits_per_merge"] = stats["posting_visits"] / row["actual_merges"]
    values["merges_per_batch"] = row["actual_merges"] / stats["batch_rounds"]
    values["fused_prepare_ns_per_posting_visit"] = stats["fused_prepare_ms"] * 1e6 / stats["posting_visits"]
    values["child_user_seconds"] = env["child_user_seconds"]
    values["child_system_seconds"] = env["child_system_seconds"]
    return {
        "result_path": str(path), "result_sha256": digest,
        "environment_path": str(env_path), "environment_sha256": env_digest,
        "build_manifest_path": str(manifest_path), "build_manifest_sha256": manifest_digest,
        "provenance": {key: env[key] for key in (
            "worktree_commit", "binary_path", "binary_sha256", "corpus_path",
            "corpus_sha256", "env_overrides", "threads_requested", "vocab_size",
            "min_frequency", "max_token_length", "cohort_disable",
        )},
        "model_sha256": row["model_sha256"],
        "runtime": {key: stats[key] for key in (
            "workers", "initialization_workers", "layout", "corpus_slot_bytes",
            "initial_count_backend", "reused_ids", "alias_fallback",
        )},
        "resources": {key: env[key] for key in (
            "peak_vmswap_bytes", "min_memavailable_bytes", "max_process_threads",
            "pswpin_delta", "pswpout_delta",
        )},
        "values": values,
    }


def compare(en, zh):
    for key in ("worktree_commit", "binary_path", "binary_sha256", "env_overrides",
                "threads_requested", "vocab_size", "min_frequency", "max_token_length", "cohort_disable"):
        if en["provenance"][key] != zh["provenance"][key]:
            raise ValueError(f"comparison configuration mismatch: {key}")
    if en["runtime"] != zh["runtime"]:
        raise ValueError("comparison runtime paths differ")
    if (en["runtime"]["workers"], en["runtime"]["initialization_workers"]) != (4, 4):
        raise ValueError("comparison requires actual workers/init = 4/4")
    for entry in (en, zh):
        if entry["resources"]["peak_vmswap_bytes"] != 0:
            raise ValueError("process swap observed")
        if entry["resources"]["min_memavailable_bytes"] <= 1 << 30:
            raise ValueError("MemAvailable gate failed")
    differences = {}
    for key, left in en["values"].items():
        right = zh["values"][key]
        differences[key] = {"en": left, "zh": right, "en_minus_zh": left - right,
                            "en_over_zh": left / right if right else None}
    gap = differences["train_ms"]["en_minus_zh"]
    shares = {key: differences[key]["en_minus_zh"] / gap if gap else None
              for key in ("initialize_ms", "merge_ms", "train_minus_initialize_minus_merge_ms")}
    return {"status": "historical evidence; BLOCK freeze and CPU release pending",
            "en": en, "zh": zh, "differences": differences,
            "train_gap_boundary_shares": shares,
            "limitations": [
                "Input and binary digests are recorded provenance, not rehashed artifacts in this lightweight analysis.",
                "One timing per language is used in the formal comparison; owner hash layout remains randomized.",
                "Nested phase times must not be added; boundary remainder has no operation attribution.",
                "fused_prepare ns/visit is a composite phase average, not an individual posting operation latency.",
                "vocab-minus-merges needs trainer defaults and identity behavior to infer alphabet.",
                "Process user/sys and perf command totals include feed and post-training work.",
                "These fixtures do not isolate language from line structure and corpus distribution.",
            ]}


def corpus_census(path):
    """One physical read, retaining exactly the runner's LF-delimited bytes."""
    lines = Counter()
    digest = hashlib.sha256()
    byte_count = 0
    with Path(path).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            byte_count += len(raw)
            lines[raw.decode("utf-8")] += 1
    alphabet = set()
    weights = Counter()
    lengths = Counter()
    weighted_lengths = Counter()
    symbols = weighted_symbols = edges = weighted_edges = unique_bytes = 0
    crlf = lf = unterminated = 0
    for line, weight in lines.items():
        n = len(line)
        alphabet.update(line)
        weights[weight] += 1
        lengths[n] += 1
        weighted_lengths[n] += weight
        symbols += n
        weighted_symbols += n * weight
        edges += max(0, n - 1)
        weighted_edges += max(0, n - 1) * weight
        unique_bytes += len(line.encode("utf-8"))
        if line.endswith("\r\n"):
            crlf += weight
        elif line.endswith("\n"):
            lf += weight
        else:
            unterminated += weight
    def quantiles(histogram):
        total = sum(histogram.values())
        result = {}
        for fraction in (.5, .9, .99, 1.):
            rank = max(1, int(total * fraction + .999999))
            cumulative = 0
            for length, count in sorted(histogram.items()):
                cumulative += count
                if cumulative >= rank:
                    result[str(fraction)] = length
                    break
        return result
    occurrences = sum(lines.values())
    return {
        "path": str(Path(path).resolve()), "sha256": digest.hexdigest(),
        "input_bytes": byte_count, "physical_read_passes": 1,
        "line_semantics": "UTF-8 strict; split only at LF, preserve CR and LF",
        "lines": occurrences, "unique_lines": len(lines),
        "duplicate_line_occurrences_beyond_first": occurrences - len(lines),
        "duplicate_line_occurrence_fraction": (occurrences - len(lines)) / occurrences,
        "weight_histogram_unique_lines": dict(sorted(weights.items())),
        "unit_weight_unique_lines": weights[1], "max_line_weight": max(weights),
        "alphabet_size": len(alphabet),
        "alphabet_utf8_sha256": hashlib.sha256("".join(sorted(alphabet)).encode()).hexdigest(),
        "unique_line_unicode_symbols": symbols, "all_line_unicode_symbols": weighted_symbols,
        "unique_line_utf8_bytes": unique_bytes,
        "all_line_utf8_bytes_per_unicode_symbol": byte_count / weighted_symbols,
        "unique_line_utf8_bytes_per_unicode_symbol": unique_bytes / symbols,
        "unweighted_edges": edges, "weighted_edges": weighted_edges,
        "line_endings": {"CRLF": crlf, "LF_without_CR": lf, "unterminated": unterminated},
        "unique_line_length_quantiles_unicode": quantiles(lengths),
        "all_line_length_quantiles_unicode": quantiles(weighted_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    historical = ROOT / "results/affix-all-fast-v4"
    parser.add_argument("--en-row", type=Path, default=historical / "en16m-none4.jsonl")
    parser.add_argument("--zh-row", type=Path, default=historical / "zh16m-none4.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/language-training-gap/historical-none-v4.json")
    census_group = parser.add_mutually_exclusive_group()
    census_group.add_argument("--census", action="store_true", help="one full 16MiB pass per row input; only after CPU release")
    census_group.add_argument("--reuse-census", type=Path, help="reuse a checked census JSON without rereading inputs")
    parser.add_argument("--perf-summary", type=Path, help="attach the existing bounded perf analysis without decoding or resolving again")
    parser.add_argument("--final", action="store_true", help="mark comparison as the completed BLOCK freeze")
    args = parser.parse_args()
    analysis = compare(case(args.en_row), case(args.zh_row))
    if args.final:
        analysis["status"] = "final BLOCK freeze; one formal run per language"
    if args.census or args.reuse_census:
        reused = metadata(args.reuse_census)[0]["census"] if args.reuse_census else None
        analysis["census"] = {}
        for language in ("en", "zh"):
            entry = analysis[language]
            census = reused[language] if reused else corpus_census(entry["provenance"]["corpus_path"])
            checks = {
                "sha256": entry["provenance"]["corpus_sha256"],
                "input_bytes": entry["values"]["input_bytes"],
                "unique_lines": entry["values"]["unique_words"],
                "unique_line_unicode_symbols": entry["values"]["initial_symbols"],
                "unweighted_edges": entry["values"]["initial_edges"],
                "alphabet_size": entry["values"]["vocab_minus_merges"],
            }
            for key, expected in checks.items():
                if census[key] != expected:
                    raise ValueError(f"census disagrees with native {language}: {key}")
            analysis["census"][language] = census
        analysis["limitations"][0] = "Corpus SHA and counts checked by one prior census per input; binary/ELF verification is recorded separately in elf-validation.json."
    if args.perf_summary:
        analysis["limitations"].append("The omitted EN unsampled diagnostic control leaves EN sampling overhead unisolated; ZH single-run control differences include measurement noise.")
        perf, digest = metadata(args.perf_summary)
        analysis["perf_summary"] = {"path": str(args.perf_summary.resolve()), "sha256": digest,
                                    "event": perf["event"], "denominator": perf["denominator"],
                                    "binary_sha256": perf["binary_sha256"]}
        analysis["diagnostic"] = {}
        for language in ("en", "zh"):
            base = args.perf_summary.parent
            record, _ = metadata(base / (language + "-final-record.jsonl"))
            env, _ = metadata(base / (language + "-final-record.environment.json"))
            if (env["worktree_commit"], env["corpus_sha256"], env["binary_sha256"], record["model_sha256"]) != (
                    analysis[language]["provenance"]["worktree_commit"], analysis[language]["provenance"]["corpus_sha256"],
                    perf["binary_sha256"], analysis[language]["model_sha256"]):
                raise ValueError(f"perf/source/input/model provenance mismatch: {language}")
            stats = record["indexed_stats"]
            histogram = stats["diagnostic_batch_sizes"]
            if sum(histogram) != stats["batch_rounds"] or sum(i * c for i, c in enumerate(histogram)) != record["actual_merges"]:
                raise ValueError(f"batch histogram totals mismatch: {language}")
            visits, rewrites = stats["posting_visits"], stats["diagnostic_effective_rewrites"]
            aa_visits, aa_rewrites = stats["diagnostic_aa_visits"], stats["diagnostic_aa_rewrites"]
            analysis["diagnostic"][language] = {
                "stat": perf["languages"][language]["stat"],
                "profile_quality": perf["languages"][language]["quality"],
                "profile_source_resolution": perf["languages"][language]["source_resolution"],
                "sample_count": perf["languages"][language]["sample_count"],
                "sample_period_sum": perf["languages"][language]["sample_period_sum"],
                "counters": {key: value for key, value in stats.items() if key.startswith("diagnostic_")},
                "derived_work": {
                    "postings_without_rewrite": visits - rewrites,
                    "postings_without_rewrite_fraction": (visits - rewrites) / visits,
                    "non_aa_postings_without_rewrite": visits - aa_visits - rewrites + aa_rewrites,
                    "aa_rewrite_fraction": aa_rewrites / rewrites,
                    "remaining_active_symbols": stats["initial_symbols"] - rewrites,
                    "symbol_reduction_fraction": rewrites / stats["initial_symbols"],
                },
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "train_ratio": analysis["differences"]["train_ms"]["en_over_zh"],
                      "train_gap_boundary_shares": analysis["train_gap_boundary_shares"]}))


if __name__ == "__main__":
    main()
