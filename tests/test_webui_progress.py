"""The job page follows the phases and bars T1Prep prints.

Up to 0.8.2 the page only read bar lines that start with "[", the Python
renderer's; the shell renderer T1Prep uses starts them with "|", so a
single-subject job sat at 0 % until it ended, and its surface panel never
started because only several subjects got the phase headers.  After the job
ended the page did not fetch the final state either.
"""

import shutil
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from t1prep.webui import app as webui
except ImportError:  # pragma: no cover - flask is a dependency
    webui = None


def bar(percent, label="", style="|"):
    """A bar line as progress_bar_multi.sh (|) or the Python renderer ([) prints it."""
    filled = 40 * percent // 100
    left, right = ("[", "]") if style == "[" else ("|", "|")
    return f"{left}{'█' * filled}{' ' * (40 - filled)}{right} {percent}% {label}"


# One subject in the web UI: a header per phase, and each phase's bar
# counts from zero (scripts/T1Prep with T1PREP_WEBUI)
VOLUME = ["Volume Segmentation"] + [
    bar(round(100 * step / 7), "step") for step in range(1, 8)
]
SURFACE = ["Surface Estimation"] + [
    bar(round(100 * step / 7), "step") for step in range(1, 5)
]


@unittest.skipIf(webui is None, "web UI not importable")
class TestJobProgress(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.object(webui, "find_parallelize_log", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = webui.app.test_client()

    def start(self, lines, exit_code=0):
        """Run a stand-in T1Prep that prints lines, then waits for release()."""
        self.go = self.tmp / "go"
        script = self.tmp / "t1prep.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import os, sys, time
                for line in {lines!r}:
                    print(line, flush=True)
                print("printed", flush=True)
                while not os.path.exists({str(self.go)!r}):
                    time.sleep(0.02)
                sys.exit({exit_code})
                """
            )
        )
        self.job_id = f"test{id(self)}"
        webui.JOBS[self.job_id] = {
            "id": self.job_id,
            "status": "queued",
            "command": [sys.executable, str(script)],
            "log_path": str(self.tmp / "t1prep.log"),
            "progress_dir": str(self.tmp / "progress"),
            "upload_dir": None,
        }
        self.addCleanup(webui.JOBS.pop, self.job_id, None)
        self.worker = threading.Thread(target=webui.run_job, args=(self.job_id,))
        self.worker.start()
        self.addCleanup(self.release)
        deadline = time.time() + 30
        while "printed" not in self.log():
            self.assertLess(time.time(), deadline, "stand-in T1Prep printed nothing")
            time.sleep(0.02)

    def release(self):
        self.go.touch()
        self.worker.join(30)

    def log(self):
        path = self.tmp / "t1prep.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def progress(self):
        return self.client.get(f"/jobs/{self.job_id}/progress").get_json()

    def status(self):
        return self.client.get(f"/jobs/{self.job_id}/status").get_json()

    def test_the_volume_panel_follows_its_bar(self):
        self.start(VOLUME[:4])
        volume = self.progress()["volume"]
        self.assertEqual((volume["percent"], volume["started"]), (43, True))
        self.assertFalse(self.progress()["surface"]["started"])
        self.assertEqual(self.status()["phase"], "Volume Segmentation")

    def test_a_single_subject_fills_both_panels(self):
        self.start(VOLUME + SURFACE)
        progress = self.progress()
        self.assertEqual(progress["volume"]["percent"], 100)
        self.assertTrue(progress["volume"]["complete"])
        self.assertRegex(progress["volume"]["time"], r"^\d+s$")
        self.assertEqual(progress["surface"]["percent"], 57)
        self.assertFalse(progress["surface"]["complete"])
        self.assertEqual(self.status()["phase"], "Surface Estimation")

        self.release()
        progress = self.progress()
        for phase in ("volume", "surface"):
            self.assertEqual(progress[phase]["percent"], 100)
            self.assertTrue(progress[phase]["complete"])
            self.assertFalse(progress[phase]["failed"])

    def test_a_phase_that_did_not_run_is_skipped(self):
        self.start(VOLUME)
        self.release()
        surface = self.progress()["surface"]
        self.assertTrue(surface["skipped"])
        self.assertFalse(surface["complete"])

    def test_a_failure_marks_the_phase_it_happened_in(self):
        self.start(VOLUME + SURFACE[:3], exit_code=1)
        self.release()
        progress = self.progress()
        self.assertTrue(progress["volume"]["complete"])
        self.assertFalse(progress["volume"]["failed"])
        self.assertTrue(progress["surface"]["failed"])
        self.assertEqual(progress["surface"]["percent"], 29)

    def test_a_failure_before_any_phase_is_the_volume_s(self):
        self.start(["ERROR: x.nii could not be found."], exit_code=1)
        self.release()
        progress = self.progress()
        self.assertTrue(progress["volume"]["failed"])
        self.assertFalse(progress["surface"]["failed"])

    def test_the_python_renderer_s_bars_count_too(self):
        self.start(["Volume Segmentation", bar(50, "step", style="[")])
        self.assertEqual(self.progress()["volume"]["percent"], 50)

    def test_the_log_keeps_each_step_once(self):
        # parallelize's bar over several subjects repeats every second
        repeated = [bar(0, "T1Prep")] * 5 + [bar(50, "T1Prep ETA 00:01:00")] * 5
        self.start(["Volume Segmentation", *repeated])
        log = self.log()
        self.assertEqual(log.count("% T1Prep"), 2)
        self.assertEqual(self.progress()["volume"]["percent"], 50)


@unittest.skipIf(webui is None, "web UI not importable")
class TestParallelizeLog(unittest.TestCase):
    def test_a_log_from_before_the_job_is_not_appended(self):
        logs = {"/tmp/parallelize_h_1.log": 100.0, "/tmp/parallelize_h_2.log": 200.0}
        with mock.patch.object(webui.glob, "glob", return_value=list(logs)), \
                mock.patch.object(webui.os.path, "getmtime", logs.get):
            self.assertEqual(
                webui.find_parallelize_log(), Path("/tmp/parallelize_h_2.log")
            )
            self.assertEqual(
                webui.find_parallelize_log(since=150.0),
                Path("/tmp/parallelize_h_2.log"),
            )
            self.assertIsNone(webui.find_parallelize_log(since=300.0))


if __name__ == "__main__":
    unittest.main()
