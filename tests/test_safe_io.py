"""Tests for the safe-write helpers in functions.io_utils.

Goals
-----
1. Writes to a denied path return False instead of raising.
2. Successful writes leave the file at chmod 0o770.
3. Higher-level wrappers (save_masked_registry, save_baseline_registry,
   write_baseline_summary_csv, safe_savefig) honour both rules.
4. A failed write does NOT prevent subsequent writes from succeeding
   (the property the iDLG_mask.py crash was about).

Run with:
    conda activate stable-ginv
    python tests/test_safe_io.py            # stdlib unittest, no pytest needed
    python -m unittest tests.test_safe_io   # equivalent
"""
import contextlib
import csv
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest

# Make project root importable when run as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functions.io_utils import (
    safe_chmod,
    safe_makedirs,
    safe_savefig,
    safe_write,
    save_baseline_registry,
    save_masked_registry,
    write_baseline_summary_csv,
)


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@contextlib.contextmanager
def _capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def _make_denied_dir():
    """Create a read-only parent so writing inside it fails with EACCES.

    Returns (denied_file_path, cleanup_fn). Skips the test if not root-permitted to chmod.
    """
    parent = tempfile.mkdtemp(prefix="safe_io_denied_")
    inner = os.path.join(parent, "locked")
    os.makedirs(inner)
    os.chmod(inner, 0o500)  # r-x only: cannot create files inside

    def cleanup():
        try:
            os.chmod(inner, 0o700)
        except OSError:
            pass
        shutil.rmtree(parent, ignore_errors=True)

    return os.path.join(inner, "out.json"), cleanup


class SafeMakedirsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="safe_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_nested(self):
        target = os.path.join(self.tmp, "a", "b", "c")
        self.assertTrue(safe_makedirs(target))
        self.assertTrue(os.path.isdir(target))

    def test_idempotent(self):
        target = os.path.join(self.tmp, "existing")
        os.makedirs(target)
        self.assertTrue(safe_makedirs(target))

    def test_empty_path_returns_true(self):
        self.assertTrue(safe_makedirs(""))

    def test_denied_returns_false_and_logs(self):
        denied_file, cleanup = _make_denied_dir()
        denied_dir = os.path.join(os.path.dirname(denied_file), "newsub")
        try:
            with _capture_stdout() as out:
                ok = safe_makedirs(denied_dir)
            self.assertFalse(ok)
            self.assertIn("[WARNING]", out.getvalue())
            self.assertIn("Failed to create directory", out.getvalue())
        finally:
            cleanup()


class SafeChmodTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="safe_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sets_mode(self):
        p = os.path.join(self.tmp, "x.txt")
        with open(p, "w") as f:
            f.write("hi")
        os.chmod(p, 0o600)
        safe_chmod(p, 0o770)
        self.assertEqual(_mode(p), 0o770)

    def test_missing_path_does_not_raise(self):
        with _capture_stdout() as out:
            safe_chmod("/no/such/path/here_xxxxxxxx")
        self.assertIn("[WARNING] Failed to chmod", out.getvalue())


class SafeWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="safe_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_success_chmods_770(self):
        p = os.path.join(self.tmp, "sub", "nested", "out.txt")
        self.assertTrue(safe_write(p, lambda f: f.write("payload")))
        with open(p) as f:
            self.assertEqual(f.read(), "payload")
        self.assertEqual(_mode(p), 0o770)

    def test_append_mode(self):
        p = os.path.join(self.tmp, "log.txt")
        self.assertTrue(safe_write(p, lambda f: f.write("a"), mode="a"))
        self.assertTrue(safe_write(p, lambda f: f.write("b"), mode="a"))
        with open(p) as f:
            self.assertEqual(f.read(), "ab")

    def test_newline_kwarg_for_csv(self):
        p = os.path.join(self.tmp, "out.csv")

        def writer(f):
            w = csv.DictWriter(f, fieldnames=["a", "b"])
            w.writeheader()
            w.writerow({"a": 1, "b": 2})

        self.assertTrue(safe_write(p, writer, newline=""))
        with open(p) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows, [{"a": "1", "b": "2"}])

    def test_denied_returns_false_no_raise(self):
        denied_file, cleanup = _make_denied_dir()
        try:
            with _capture_stdout() as out:
                ok = safe_write(denied_file, lambda f: f.write("x"))
            self.assertFalse(ok)
            self.assertIn("[WARNING]", out.getvalue())
        finally:
            cleanup()

    def test_failure_does_not_block_later_success(self):
        """The property whose absence caused the iDLG_mask.py crash."""
        denied_file, cleanup = _make_denied_dir()
        good = os.path.join(self.tmp, "good.txt")
        try:
            with _capture_stdout():
                self.assertFalse(safe_write(denied_file, lambda f: f.write("x")))
            self.assertTrue(safe_write(good, lambda f: f.write("y")))
            with open(good) as f:
                self.assertEqual(f.read(), "y")
        finally:
            cleanup()


class SafeSavefigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="safe_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_success(self):
        p = os.path.join(self.tmp, "plots", "fig.png")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ok = safe_savefig(fig, p, dpi=50)
        plt.close(fig)
        self.assertTrue(ok)
        self.assertGreater(os.path.getsize(p), 0)
        self.assertEqual(_mode(p), 0o770)

    def test_denied(self):
        denied_file, cleanup = _make_denied_dir()
        denied_png = denied_file.replace(".json", ".png")
        fig, _ = plt.subplots()
        try:
            with _capture_stdout() as out:
                ok = safe_savefig(fig, denied_png)
            self.assertFalse(ok)
            self.assertIn("[WARNING]", out.getvalue())
        finally:
            plt.close(fig)
            cleanup()


class RegistryWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="safe_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_masked_registry_success(self):
        p = os.path.join(self.tmp, "masked_registry.json")
        payload = {"k1": {"a": 1}, "k2": {"b": [1, 2, 3]}}
        self.assertTrue(save_masked_registry(p, payload))
        with open(p) as f:
            self.assertEqual(json.load(f), payload)
        self.assertEqual(_mode(p), 0o770)

    def test_save_masked_registry_denied_reproduces_original_crash_path(self):
        denied_file, cleanup = _make_denied_dir()
        try:
            with _capture_stdout() as out:
                ok = save_masked_registry(denied_file, {"k": "v"})
            self.assertFalse(ok)
            self.assertIn("[WARNING]", out.getvalue())
        finally:
            cleanup()

    def test_save_baseline_registry_success(self):
        p = os.path.join(self.tmp, "baselines", "idlg_baselines_registry.json")
        payload = {"hashkey": {"args": {"lr": 0.1}, "best_psnr_list": [1.0, 2.0]}}
        self.assertTrue(save_baseline_registry(p, payload))
        with open(p) as f:
            self.assertEqual(json.load(f), payload)
        self.assertEqual(_mode(p), 0o770)

    def test_save_baseline_registry_denied(self):
        denied_file, cleanup = _make_denied_dir()
        try:
            with _capture_stdout() as out:
                self.assertFalse(save_baseline_registry(denied_file, {}))
            self.assertIn("[WARNING]", out.getvalue())
        finally:
            cleanup()

    def test_write_baseline_summary_csv_success(self):
        p = os.path.join(self.tmp, "summary.csv")
        registry = {
            "abc123": {
                "args": {
                    "dataset": "CIFAR10", "network": "LeNet", "pretrained": False,
                    "lr": 0.1, "gamma": 0.9, "grad_loss": "l2", "num_dummy": 1,
                    "iteration": 300, "num_exp": 5, "run_id": 0, "tv_weight": 0.0,
                    "optimizer": "lbfgs", "num_restarts": 1, "max_iteration": 20,
                    "history_size": 100,
                },
                "best_psnr_list": [10.0, 12.0, 14.0],
                "best_mse_list": [0.01, 0.02, 0.03],
            }
        }
        self.assertTrue(write_baseline_summary_csv(p, registry))
        self.assertEqual(_mode(p), 0o770)
        with open(p) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["baseline_key"], "abc123")
        self.assertEqual(rows[0]["network"], "LeNet")

    def test_write_baseline_summary_csv_denied(self):
        denied_file, cleanup = _make_denied_dir()
        try:
            with _capture_stdout() as out:
                self.assertFalse(write_baseline_summary_csv(denied_file, {}))
            self.assertIn("[WARNING]", out.getvalue())
        finally:
            cleanup()


class EndToEndResilienceTests(unittest.TestCase):
    """Simulate the iDLG_mask.py tail: three registry/summary writes fail,
    but a subsequent per-experiment CSV append still succeeds — no exception
    bubbles up to crash main()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="safe_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failures_dont_block_later_writes(self):
        denied_file, cleanup = _make_denied_dir()
        good_csv = os.path.join(self.tmp, "exp_results.csv")
        try:
            with _capture_stdout() as out:
                self.assertFalse(save_masked_registry(denied_file, {"a": 1}))
                self.assertFalse(save_baseline_registry(denied_file, {"a": 1}))
                self.assertFalse(write_baseline_summary_csv(denied_file, {}))

                def append_csv(f):
                    w = csv.DictWriter(f, fieldnames=["method", "psnr"])
                    w.writeheader()
                    w.writerow({"method": "iDLG", "psnr": 12.34})

                self.assertTrue(safe_write(good_csv, append_csv, mode="a", newline=""))

            self.assertTrue(os.path.exists(good_csv))
            self.assertEqual(_mode(good_csv), 0o770)
            self.assertGreaterEqual(out.getvalue().count("[WARNING]"), 3)
        finally:
            cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
