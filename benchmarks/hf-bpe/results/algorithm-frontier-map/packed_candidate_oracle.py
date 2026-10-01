"""Bounded arithmetic oracle, independent of the Trainer heap implementation.

Checks the proposed 64-bit order embedding and boundary eligibility rules.
Does not measure performance or exercise Trainer integration.
"""
import itertools
import json
import random
from pathlib import Path

U32 = (1 << 32) - 1


def canonical(a, b):
    return (a << 32) | b


def pack(frequency, a, b):
    assert 0 <= frequency <= U32
    assert 0 <= a <= 65535 and 0 <= b <= 65535
    return (frequency << 32) | (U32 ^ ((a << 16) | b))


def decode(value):
    code = U32 ^ (value & U32)
    return value >> 32, code >> 16, code & 65535


def eligible(initial_id_count, target_size, weighted_edge_mass):
    return max(initial_id_count, target_size) <= 65536 and weighted_edge_mass <= U32


def main():
    boundaries = list(itertools.product([0, 1, U32 - 1, U32], [0, 1, 65534, 65535], [0, 1, 65534, 65535]))
    rng = random.Random(376363)
    values = boundaries + [(rng.randrange(1 << 32), rng.randrange(1 << 16), rng.randrange(1 << 16)) for _ in range(10000)]
    assert all(decode(pack(*value)) == value for value in values)
    expected = sorted(values, key=lambda value: (value[0], -canonical(value[1], value[2])))
    actual = sorted(values, key=lambda value: pack(*value))
    assert actual == expected
    assert eligible(65536, 50000, U32)
    assert eligible(50000, 65536, U32)
    assert not eligible(65537, 50000, U32)
    assert not eligible(50000, 65537, U32)
    assert not eligible(50000, 50000, U32 + 1)
    assert eligible(0, 0, 0)
    result = {"status": "passed", "roundtrip_and_order_records": len(values), "seed": 376363,
              "eligibility_boundaries": 6, "scope": "mathematical encoding only; no Trainer test or benchmark"}
    Path(__file__).with_name("packed_candidate_oracle.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
