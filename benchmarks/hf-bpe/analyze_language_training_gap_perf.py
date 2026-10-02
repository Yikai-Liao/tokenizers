#!/usr/bin/env python3
"""Bounded offline analysis of the two captured profiles; launches no native work."""
import collections
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/language-training-gap"


def samples(path):
    header = re.compile(r"^(.*?)\s+(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+\.\d+):\s+(\d+) cycles:u:\s*$")
    frame = re.compile(r"^\s+([0-9a-f]+) (.+?) \((.+)\)$")
    result = []
    current = None
    for line in path.read_text().splitlines():
        match = header.match(line)
        if match:
            if current:
                result.append(current)
            sec, fraction = match[5].split(".")
            current = {"comm": match[1], "pid": int(match[2]), "tid": int(match[3]), "cpu": int(match[4]),
                       "monotonic_ns": int(sec) * 1_000_000_000 + int(fraction.ljust(9, "0")),
                       "period": int(match[6]), "frames": []}
        elif (match := frame.match(line)) and current is not None:
            symbol = match[2]
            offset = re.search(r"\+0x([0-9a-f]+)$", symbol)
            current["frames"].append({"runtime_ip": int(match[1], 16), "symbol": symbol[:offset.start()] if offset else symbol,
                                      "offset": int(offset[1], 16) if offset else 0, "dso": match[3]})
        elif line.strip():
            raise ValueError(f"unexpected perf script line: {line[:180]}")
    if current:
        result.append(current)
    if len(result) > 40000:
        raise ValueError("sample parse budget exceeded")
    return result


def raw_records(path):
    if path.stat().st_size > 128 << 20:
        raise ValueError("perf.data exceeds 128MiB")
    raw = path.read_bytes()
    header = struct.unpack_from("<8s12Q", raw)
    if header[0] != b"PERFILE2":
        raise ValueError("unexpected perf.data endian/format")
    attrs_offset = header[3]
    sample_type = struct.unpack_from("<Q", raw, attrs_offset + 24)[0]
    read_format = struct.unpack_from("<Q", raw, attrs_offset + 32)[0]
    regs_mask = struct.unpack_from("<Q", raw, attrs_offset + 80)[0]
    offset, size = header[5:7]
    end = offset + size
    types = collections.Counter()
    lost_events = lost_samples = lost_records = 0
    period_sum = 0
    stack_sizes = collections.Counter()
    sample_count = 0
    while offset < end:
        kind, misc, n = struct.unpack_from("<IHH", raw, offset)
        if n < 8 or offset + n > end:
            raise ValueError("invalid perf record boundary")
        types[kind] += 1
        if kind == 2:
            lost_records += 1
            lost_events += struct.unpack_from("<Q", raw, offset + 16)[0]
        if kind == 13:
            lost_records += 1
            lost_samples += struct.unpack_from("<Q", raw, offset + 8)[0]
        if kind == 9:
            sample_count += 1
            # Exact capture schema verified in the perf header, including order.
            expected = 1 | 2 | 4 | 8 | 32 | 128 | 256 | 4096 | 8192 | 32768
            if sample_type != expected:
                raise ValueError(f"unsupported raw sample schema: {sample_type}")
            cursor = offset + 8 + 8 + 8 + 8 + 8
            cursor += 8  # CPU + reserved
            period_sum += struct.unpack_from("<Q", raw, cursor)[0]
            cursor += 8
            chains = struct.unpack_from("<Q", raw, cursor)[0]
            cursor += 8 + chains * 8
            abi = struct.unpack_from("<Q", raw, cursor)[0]
            cursor += 8 + (regs_mask.bit_count() * 8 if abi else 0)
            stack_size = struct.unpack_from("<Q", raw, cursor)[0]
            cursor += 8 + stack_size
            dynamic_size = struct.unpack_from("<Q", raw, cursor)[0] if stack_size else 0
            stack_sizes[(stack_size, dynamic_size)] += 1
        offset += n
    return {"file_bytes": len(raw), "data_bytes": size, "sample_type": sample_type,
            "read_format": read_format, "record_type_counts": dict(types), "sample_count": sample_count,
            "sample_period_sum": period_sum, "recorded_lost_records": lost_records,
            "recorded_lost_events": lost_events, "recorded_lost_samples": lost_samples,
            "stack_capture_size_histogram": [{"requested": k[0], "dynamic": k[1], "samples": v} for k, v in sorted(stack_sizes.items())],
            "lost_scope": "recorded PERF_RECORD_LOST/LOST_SAMPLES, not a proof of zero unreported hardware loss"}


def stat(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")
        if fields[0].startswith("<"):
            result[fields[2]] = {"status": fields[0]}
            continue
        raw = float(fields[0])
        percent = float(fields[4])
        result[fields[2]] = {"raw_count": raw, "unit": fields[1], "time_running_ns": int(fields[3]),
                             "running_percent": percent,
                             "time_enabled_ns_estimate": int(fields[3]) * 100 / percent,
                             "scaled_count_estimate": raw * 100 / percent,
                             "multiplexed": percent != 100}
    result["derived"] = {
        "ipc_whole_command_user": result["instructions:u"]["scaled_count_estimate"] / result["cycles:u"]["scaled_count_estimate"],
        "generic_cache_miss_fraction": result["cache-misses:u"]["scaled_count_estimate"] / result["cache-references:u"]["scaled_count_estimate"],
        "generic_cache_mpki": result["cache-misses:u"]["scaled_count_estimate"] * 1000 / result["instructions:u"]["scaled_count_estimate"],
    }
    return result


def rank(counter, total, count=20):
    return [{"symbol": key, "sample_period_sum": value, "percent_of_all_command_periods": value * 100 / total}
            for key, value in counter.most_common(count)]


def analyze():
    parsed = {language: samples(RESULTS / (language + "-final-record.script.txt")) for language in ("en", "zh")}
    symbols = sorted({f["symbol"] for ss in parsed.values() for s in ss for f in s["frames"]})
    demangled = subprocess.check_output(["c++filt", "-s", "rust", "-n"], input="\n".join(symbols) + "\n", text=True).splitlines()
    if len(demangled) != len(symbols):
        raise ValueError("demangle alignment failed")
    names = dict(zip(symbols, demangled))
    manifest = json.loads((ROOT / ".build/native-language-gap-dwarf32/build_manifest.json").read_text())
    binary = manifest["binary_path"]
    if hashlib.sha256(Path(binary).read_bytes()).hexdigest() != manifest["binary_sha256"]:
        raise ValueError("profile source/binary provenance changed")
    nm = subprocess.check_output(["nm", "-n", "--defined-only", "--format=posix", binary], text=True)
    addresses = {}
    for line in nm.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            addresses[fields[0]] = int(fields[2], 16)
    result = {"event": "cycles:u", "denominator": "whole-command period-weighted user-cycle samples; inclusive paths overlap",
              "clock": "CLOCK_MONOTONIC, perf clockid=1; train window is a sample estimate, not wall-clock attribution",
              "demangler": "GNU c++filt -s rust -n in one batch; perf's --demangle left v0 symbols encoded",
              "binary_sha256": manifest["binary_sha256"], "languages": {}}
    all_addresses = set()
    for language, ss in parsed.items():
        total = sum(s["period"] for s in ss)
        record = json.loads((RESULTS / (language + "-final-record.jsonl")).read_text())
        env = json.loads((RESULTS / (language + "-final-record.environment.json")).read_text())
        markers = {p["diagnostic_phase"]: p["monotonic_ns"] for p in record["diagnostic_phases"]}
        lo, hi = markers["train_start"], markers["train_end"]
        self_weight = collections.Counter()
        inclusive = collections.Counter()
        self_samples = collections.Counter()
        ip_weight = collections.Counter()
        ip_count = collections.Counter()
        threads = collections.Counter()
        window_self = collections.Counter()
        window_inclusive = collections.Counter()
        window_periods = window_samples = 0
        unknown_leaf_periods = unknown_frames = total_frames = no_stack = short_stack = 0
        root_periods = 0
        unresolved_names = set()
        representative = {}
        for s in ss:
            if s["pid"] not in env["target_pids"]:
                raise ValueError("sample belongs to a non-target PID")
            w = s["period"]
            frames = s["frames"]
            threads[s["tid"]] += w
            total_frames += len(frames)
            unknown_frames += sum(f["symbol"] == "[unknown]" for f in frames)
            if not frames:
                no_stack += 1
                continue
            if len(frames) < 3:
                short_stack += 1
            if any(f["symbol"] in ("_start", "__clone3", "clone", "start_thread") or "thread_start" in names[f["symbol"]] for f in frames):
                root_periods += w
            leaf = frames[0]
            symbol = names[leaf["symbol"]]
            if leaf["symbol"] == "[unknown]":
                unknown_leaf_periods += w
            if symbol.startswith("_R"):
                unresolved_names.add(symbol)
            self_weight[symbol] += w
            self_samples[symbol] += 1
            ancestors = {names[f["symbol"]] for f in frames}
            for name in ancestors:
                inclusive[name] += w
            in_window = lo <= s["monotonic_ns"] <= hi
            if in_window:
                window_samples += 1
                window_periods += w
                window_self[symbol] += w
                for name in ancestors:
                    window_inclusive[name] += w
            if leaf["dso"] == binary and leaf["symbol"] in addresses:
                address = addresses[leaf["symbol"]] + leaf["offset"]
                ip_weight[address] += w
                ip_count[address] += 1
                representative[address] = {"runtime_ip": hex(leaf["runtime_ip"]), "elf_address": hex(address),
                                           "raw_symbol": leaf["symbol"], "symbol": symbol, "symbol_offset": hex(leaf["offset"])}
        picked = ip_weight.most_common(200)
        all_addresses.update(address for address, _ in picked)
        raw = raw_records(RESULTS / (language + "-final-record.perf.data"))
        if (raw["sample_count"], raw["sample_period_sum"]) != (len(ss), total):
            raise ValueError("raw sample count/periods and decoded script disagree")
        result["languages"][language] = {
            "raw_records": raw, "sample_count": len(ss), "sample_period_sum": total,
            "train_window": {"start_ns": lo, "end_ns": hi, "duration_ms": (hi - lo) / 1e6,
                             "samples": window_samples, "sample_period_sum": window_periods,
                             "fraction_of_command_sample_periods": window_periods / total,
                             "self_top20": rank(window_self, total), "inclusive_top20": rank(window_inclusive, total)},
            "self_top20": rank(self_weight, total), "inclusive_top20": rank(inclusive, total),
            "self_sample_counts_top20": [{"symbol": key, "samples": value} for key, value in self_samples.most_common(20)],
            "threads": [{"tid": key, "sample_period_sum": value, "percent": value * 100 / total} for key, value in threads.most_common()],
            "quality": {"unknown_leaf_period_fraction": unknown_leaf_periods / total,
                        "unknown_frames": unknown_frames, "all_frames": total_frames,
                        "empty_stacks": no_stack, "stacks_under_three_frames": short_stack,
                        "period_fraction_with_recognized_entry_frame": root_periods / total,
                        "rust_names_still_encoded": sorted(unresolved_names),
                        "truncation_limit": "8KiB capture cannot exclude truncation; recognized root fraction is observable, not a guarantee of all inline frames"},
            "instruction_ips": [{**representative[address], "sample_period_sum": w, "samples": ip_count[address],
                                 "percent_of_all_command_periods": w * 100 / total} for address, w in picked],
            "stat": stat(RESULTS / (language + "-final-stat.perf-stat.csv")),
        }
    # A single batch resolves the union of both per-language 200-IP budgets.
    address_list = sorted(all_addresses)
    command = ["addr2line", "-a", "-C", "-f", "-i", "-e", binary, *[hex(a) for a in address_list]]
    resolved = subprocess.check_output(command, text=True)
    (RESULTS / "sampled-addresses.addr2line.txt").write_text(resolved)
    mapping = {}
    address = None
    lines = []
    for line in resolved.splitlines():
        if re.fullmatch(r"0x[0-9a-f]+", line):
            if address is not None:
                mapping[address] = lines
            address = int(line, 16)
            lines = []
        else:
            lines.append(line)
    if address is not None:
        mapping[address] = lines
    for entry in result["languages"].values():
        resolved_periods = 0
        picked_periods = 0
        for ip in entry["instruction_ips"]:
            frames = mapping[int(ip["elf_address"], 16)]
            ip["dwarf_inline_functions_and_locations"] = frames
            picked_periods += ip["sample_period_sum"]
            if any(re.search(r":\d+(?: \(discriminator \d+\))?$", line) for line in frames):
                resolved_periods += ip["sample_period_sum"]
        entry["source_resolution"] = {"ips_selected": len(entry["instruction_ips"]),
                                       "selected_period_fraction_of_command": picked_periods / entry["sample_period_sum"],
                                       "source_resolved_period_fraction_of_selected": resolved_periods / picked_periods}
    result["batch_address_resolution"] = {"unique_addresses": len(address_list), "command": command}
    (RESULTS / "final-perf-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    analysis = analyze()
    print(json.dumps({lang: {"samples": x["sample_count"], "quality": x["quality"], "source_resolution": x["source_resolution"]}
                      for lang, x in analysis["languages"].items()}))
