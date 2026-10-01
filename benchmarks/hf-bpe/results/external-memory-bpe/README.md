# External-memory BPE preliminary research

Report: [EXTERNAL_MEMORY_BPE.md](../../EXTERNAL_MEMORY_BPE.md).

This directory contains 26 primary research/official sources in `provenance.json`, plus reproducible conditional capacity arithmetic. No external-memory Trainer implementation or measured disk throughput is delivered. HF model output is the acceptance criterion for any future implementation.

Run the arithmetic from any directory:

```bash
python3 capacity_model.py
```

The JSON assumes the archived Chinese 512 MiB input's unique slot/edge/word density. It does not extrapolate speed, K/H/Q cardinality, or actual RSS. Staging calculations include the declared full sorting copies and raw input; additional dedup, directory, queue and log capacity is required.

Latest user scope is research only; implementation remains deferred. A tiny exploratory oracle had run before this scope correction and is excluded from this deliverable. It made no Trainer changes.
