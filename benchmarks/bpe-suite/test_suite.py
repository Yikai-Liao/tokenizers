"""Check failure accounting and exact model comparison, independent of BPE."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import suite


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        (self.out / "binaries").mkdir()
        self.config = dict(workers=1, min_available_gib=0, max_process_rss_gib=1, timeout_seconds=5)
        self.case = dict(name="case", input="unused", split="whitespace", vocab_size=100, min_frequency=2)

    def binary(self, model, delay=0):
        script = self.out / "binaries/fake"
        script.write_text("#!/usr/bin/env python3\nimport json,sys,time\n"
                          f"time.sleep({delay})\n"
                          "job=json.load(open(sys.argv[1]))\n"
                          f"open(job['output'],'w').write({json.dumps(model)!r})\n"
                          "print(json.dumps(dict(train_seconds=1.0,elapsed_seconds=2.0,"
                          "train_cpu_seconds=1.0,maxrss_kib=100,actual_vocab=2,actual_merges=1)))\n")
        script.chmod(0o755)

    def test_changed_token_ids_stop_comparison(self):
        self.binary([[['a', 0], ['b', 1]], [['a', 'b']]])
        first = suite.execute(self.out, self.config, self.case, "fake", self.out / "first")
        self.assertEqual(first["status"], "ok")
        self.assertFalse((self.out / "first/model.json").exists())
        self.binary([[['b', 0], ['a', 1]], [['a', 'b']]])
        with self.assertRaisesRegex(ValueError, "ordered merges differ"):
            suite.execute(self.out, self.config, self.case, "fake", self.out / "second")
        self.assertEqual(suite.read(self.out / "second/result.json")["status"], "model_mismatch")
        self.assertTrue((self.out / "second/model.json").exists())

    def test_changed_merge_order_stops_comparison(self):
        self.binary([[['a', 0], ['b', 1]], [['a', 'b'], ['b', 'a']]])
        suite.execute(self.out, self.config, self.case, "fake", self.out / "first")
        self.binary([[['a', 0], ['b', 1]], [['b', 'a'], ['a', 'b']]])
        with self.assertRaises(ValueError):
            suite.execute(self.out, self.config, self.case, "fake", self.out / "second")

    def test_memory_gate_does_not_launch_process(self):
        self.config["min_available_gib"] = 1
        with patch.object(suite, "available_memory", return_value=0), patch.object(suite.subprocess, "Popen") as launch:
            row = suite.execute(self.out, self.config, self.case, "fake", self.out / "guard")
        self.assertEqual(row["status"], "resource_gate_before_run")
        launch.assert_not_called()

    def test_timeout_is_retained_as_failure(self):
        self.binary([], delay=10)
        self.config["timeout_seconds"] = 0.02
        row = suite.execute(self.out, self.config, self.case, "fake", self.out / "timeout")
        self.assertEqual(row["status"], "timeout")
        self.assertNotIn("metrics", row)

    def test_failed_pair_and_interrupted_block_do_not_enter_statistics(self):
        metrics = dict(train_seconds=1, elapsed_seconds=2, train_cpu_seconds=1, maxrss_kib=100)
        good = dict(arm="full", status="ok", metrics=metrics)
        fail = dict(arm="hf", status="memory_guard")
        suite.write(self.out / "runs/primary/case/block-01/block.json",
                    dict(comparison_valid=False, results=[good, fail]))
        suite.write(self.out / "runs/primary/case/block-02.interrupted-1/block.json",
                    dict(comparison_valid=True, results=[good]))
        suite.report(self.out)
        summary = suite.read(self.out / "summary.json")
        self.assertEqual(summary["valid_paired_samples"], 0)
        self.assertEqual(len(summary["failed_runs"]), 1)
        self.assertEqual(summary["summary"], [])


if __name__ == "__main__": unittest.main()
