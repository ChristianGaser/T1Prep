"""The state of each image while T1Prep runs.

T1Prep reports differently depending on how it runs: one image in its own
loop, several images in one job of ``parallelize``, several images in several
jobs.  ``RunTracker`` has to give a table of images the same answer in all
three — which image is queued, running, done or failed in which phase — and
the lines and files it is fed here are the ones ``scripts/T1Prep`` and
``scripts/parallelize`` write.
"""

import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from t1prep.launch.progress import (  # noqa: E402
    DONE, FAILED, QUEUED, RUNNING, SKIPPED, RunTracker, find_parallelize_logs,
    follow_line,
    new_progress, read_progress_dir, step_text,
)

BOTH = ("volume", "surface")


def bar(percent, label="step"):
    """A bar line as progress_bar_multi.sh prints it for a front end."""
    filled = 40 * percent // 100
    return f"|{'█' * filled}{' ' * (40 - filled)}| {percent}% {label}"


class Disk:
    """Stands in for the results on disk: what ``finished`` is asked."""

    def __init__(self):
        self.there = set()

    def write(self, path, stage):
        self.there.add((path, stage))

    def __call__(self, path, stage):
        return (path, stage) in self.there


class TestTheLinesT1PrepPrints(unittest.TestCase):
    def test_headers_bars_and_failures(self):
        live = new_progress()
        self.assertEqual(follow_line(live, "Volume Segmentation\n", 1.0), ("phase", "volume"))
        self.assertEqual(follow_line(live, bar(42), 2.0), ("bar", 42))
        self.assertEqual(live["volume"]["percent"], 42)
        self.assertEqual(follow_line(live, "Segmentation failed", 3.0), ("failed", "volume"))
        self.assertTrue(live["volume"]["failed"])
        self.assertIsNone(follow_line(live, "Use MPS in Python", 4.0))

    def test_the_closing_bar_of_a_phase_is_not_a_header(self):
        live = new_progress()
        follow_line(live, "Volume Segmentation", 1.0)
        self.assertEqual(
            follow_line(live, bar(100, "Volume segmentation finished"), 2.0),
            ("bar", 100))
        self.assertEqual(live["volume"]["entries"], 1)

    def test_a_bar_reads_as_a_step_in_a_log(self):
        self.assertEqual(step_text(bar(25, "Affine registration         ETA 00:00:36")),
                         "  25%  Affine registration")
        self.assertEqual(step_text("[████    ]  50% Atlas creation"), "  50%  Atlas creation")
        self.assertEqual(step_text(bar(100, "")), " 100%")
        self.assertIsNone(step_text("Volume Segmentation"))

    def test_a_bar_before_any_header_belongs_to_the_first_phase(self):
        live = new_progress()
        follow_line(live, bar(10), 1.0)
        self.assertEqual(live["phase"], "volume")


class TestOneImage(unittest.TestCase):
    """T1Prep's own loop: a header and bars per phase."""

    def setUp(self):
        self.disk = Disk()
        self.run = RunTracker(["/d/a.nii"], BOTH, self.disk)
        self.state = self.run.states["/d/a.nii"]

    def feed(self, *lines):
        for line in lines:
            self.run.feed(line, 1.0)

    def test_it_goes_through_both_phases(self):
        self.assertEqual(self.state, {"volume": QUEUED, "surface": QUEUED})
        self.feed("Processing /d/a.nii", "Volume Segmentation", bar(50))
        self.assertEqual(self.state, {"volume": RUNNING, "surface": QUEUED})
        self.assertAlmostEqual(self.run.fraction(), 0.25)
        self.disk.write("/d/a.nii", "volume")
        self.feed(bar(100, "Volume segmentation finished"), "Surface Estimation")
        self.assertEqual(self.state, {"volume": DONE, "surface": RUNNING})
        self.disk.write("/d/a.nii", "surface")
        self.feed("Finished after 4min 52s")
        self.assertEqual(self.state, {"volume": DONE, "surface": DONE})
        self.run.finish()
        self.assertEqual(self.run.failures(), 0)
        self.assertEqual(self.run.fraction(), 1.0)

    def test_a_failed_segmentation_leaves_the_surfaces_out(self):
        self.feed("Processing /d/a.nii", "Volume Segmentation", bar(30),
                  "Segmentation failed", "Error in /d/a.nii")
        self.assertEqual(self.state, {"volume": FAILED, "surface": SKIPPED})
        self.run.finish()
        self.assertEqual(self.state, {"volume": FAILED, "surface": SKIPPED})
        self.assertEqual(self.run.failures(), 1)

    def test_a_failed_surface_keeps_the_volume(self):
        self.disk.write("/d/a.nii", "volume")
        self.feed("Processing /d/a.nii", "Volume Segmentation", "Surface Estimation",
                  "Surface estimation failed", "Error in a")
        self.assertEqual(self.state, {"volume": DONE, "surface": FAILED})

    def test_a_step_that_ends_without_results_has_failed(self):
        # No failure notice, the process just died: the files decide
        self.feed("Processing /d/a.nii", "Volume Segmentation", bar(30))
        self.run.finish()
        self.assertEqual(self.state, {"volume": FAILED, "surface": SKIPPED})

    def test_stopping_is_not_failing(self):
        self.feed("Processing /d/a.nii", "Volume Segmentation", bar(30))
        self.run.finish(cancelled=True)
        self.assertEqual(self.state, {"volume": SKIPPED, "surface": SKIPPED})
        self.assertEqual(self.run.failures(), 0)

    def test_only_the_phases_of_the_run_are_tracked(self):
        run = RunTracker(["/d/a.nii"], ["surface"], self.disk)
        run.feed("Processing /d/a.nii", 1.0)
        run.feed("Surface Estimation", 1.0)
        self.assertEqual(run.states["/d/a.nii"], {"surface": RUNNING})


class TestSeveralImagesOneJob(unittest.TestCase):
    """``parallelize`` with one process: the bar counts the finished images."""

    def setUp(self):
        self.files = ["/d/a.nii", "/d/b.nii", "/d/c.nii"]
        self.disk = Disk()
        self.run = RunTracker(self.files, BOTH, self.disk)

    def states(self, stage):
        return [self.run.states[path][stage] for path in self.files]

    def test_the_images_are_taken_in_the_order_given(self):
        self.run.feed("Volume Segmentation", 1.0)
        self.run.feed(bar(0, "T1Prep"), 1.0)
        self.assertEqual(self.states("volume"), [RUNNING, QUEUED, QUEUED])
        self.disk.write("/d/a.nii", "volume")
        self.run.feed(bar(33, "T1Prep"), 2.0)
        self.assertEqual(self.states("volume"), [DONE, RUNNING, QUEUED])
        # b ends without results: nothing says so, the files do
        self.run.feed(bar(66, "T1Prep"), 3.0)
        self.assertEqual(self.states("volume"), [DONE, FAILED, RUNNING])
        self.assertEqual(self.run.counts("volume"), (2, 1, 3))

    def test_the_next_header_closes_the_phase(self):
        self.run.feed("Volume Segmentation", 1.0)
        for path in self.files:
            self.disk.write(path, "volume")
        self.run.feed("Surface Estimation", 2.0)
        self.assertEqual(self.states("volume"), [DONE, DONE, DONE])
        self.assertEqual(self.states("surface"), [QUEUED, QUEUED, QUEUED])

    def test_an_error_line_names_the_image(self):
        self.run.feed("Volume Segmentation", 1.0)
        self.run.feed("Error in /d/b.nii", 1.0)
        self.assertEqual(self.states("volume"), [QUEUED, FAILED, QUEUED])
        self.run.feed("Surface Estimation", 2.0)
        self.run.feed("Error in c", 2.0)          # the surfaces name the stem
        self.assertEqual(self.run.states["/d/c.nii"]["surface"], FAILED)


class TestSeveralImagesSeveralJobs(unittest.TestCase):
    """``parallelize`` with several processes: the lists in PROGRESS_DIR."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.jobs = self.tmp / "progress"
        self.files = ["/d/a.nii", "/d/b.nii", "/d/c.nii", "/d/e f.nii"]
        self.disk = Disk()
        self.run = RunTracker(self.files, BOTH, self.disk, self.jobs)

    def job(self, number, items, done=None, failed=()):
        self.jobs.mkdir(exist_ok=True)
        (self.jobs / f"job{number}.list").write_text("".join(f"{i}\n" for i in items))
        if done is not None:
            (self.jobs / f"job{number}.progress").write_text(f"{done}/{len(items)}\n")
        (self.jobs / f"job{number}.failed").write_text("".join(f"{i}\n" for i in failed))

    def states(self, stage="volume"):
        return [self.run.states[path][stage] for path in self.files]

    def test_reading_the_lists(self):
        self.job(0, self.files[:2], done=1)
        self.job(1, self.files[2:])                      # not started yet
        self.assertEqual(read_progress_dir(self.jobs), {
            "/d/a.nii": DONE, "/d/b.nii": RUNNING,
            "/d/c.nii": QUEUED, "/d/e f.nii": QUEUED})
        self.assertEqual(read_progress_dir(self.tmp / "nowhere"), {})

    def test_each_job_works_through_its_list(self):
        self.run.feed("Volume Segmentation", 1.0)
        self.job(0, self.files[:2], done=0)
        self.job(1, self.files[2:], done=0)
        self.run.poll()
        self.assertEqual(self.states(), [RUNNING, QUEUED, RUNNING, QUEUED])
        self.job(0, self.files[:2], done=1)
        self.job(1, self.files[2:], done=1, failed=[self.files[2]])
        self.run.poll()
        self.assertEqual(self.states(), [DONE, RUNNING, FAILED, RUNNING])

    def test_the_overall_bar_does_not_guess_an_order(self):
        self.run.feed("Volume Segmentation", 1.0)
        self.job(0, self.files[:2], done=0)
        self.job(1, self.files[2:], done=0)
        self.run.feed(bar(50, "T1Prep"), 2.0)
        self.assertEqual(self.states(), [QUEUED] * 4)

    def test_what_is_settled_survives_the_lists(self):
        self.run.feed("Volume Segmentation", 1.0)
        self.job(0, self.files, done=2, failed=[self.files[1]])
        self.run.poll()
        shutil.rmtree(self.jobs)                 # parallelize removes them
        self.run.poll()
        self.assertEqual(self.states()[:2], [DONE, FAILED])
        # The two that were open are asked of the disk at the next header
        self.disk.write("/d/c.nii", "volume")
        self.run.feed("Surface Estimation", 3.0)
        self.assertEqual(self.states(), [DONE, FAILED, DONE, FAILED])

    def test_the_failed_items_parallelize_lists_at_the_end(self):
        self.run.feed("Volume Segmentation", 1.0)
        for line in ("Finished after 3min 2s with 1 error(s)", "Failed items:",
                     "  /d/e f.nii", "Surface Estimation"):
            self.run.feed(line, 2.0)
        self.assertEqual(self.run.states["/d/e f.nii"]["volume"], FAILED)

    def test_the_lists_of_the_surfaces_do_not_touch_the_volumes(self):
        self.run.feed("Volume Segmentation", 1.0)
        for path in self.files:
            self.disk.write(path, "volume")
        self.run.feed("Surface Estimation", 2.0)
        self.job(0, self.files, done=1)
        self.run.poll()
        self.assertEqual(self.states("volume"), [DONE] * 4)
        self.assertEqual(self.states("surface"), [DONE, RUNNING, QUEUED, QUEUED])


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class TestAgainstParallelize(unittest.TestCase):
    """The real ``scripts/parallelize``, printing and writing what it does."""

    def follow(self, processes):
        """Run five items, one of which fails; return the tracker and the exit status."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        files = []
        for name in ("a", "b", "bad", "d", "e"):
            (tmp / f"{name}.nii").write_text("")
            files.append(str(tmp / f"{name}.nii"))
        step = tmp / "step.sh"
        step.write_text('#!/bin/bash\nsleep 0.3\n'
                        'case "$1" in *bad.nii) exit 3;; esac\ntouch "$1.done"\n')
        step.chmod(0o755)
        run = RunTracker(files, ["volume"],
                         lambda path, stage: os.path.exists(path + ".done"),
                         tmp / "progress")
        run.feed("Volume Segmentation", time.time())      # T1Prep's header
        process = subprocess.Popen(
            [str(_ROOT / "scripts" / "parallelize"), "--logdir", str(tmp),
             "--processes", str(processes), "--command", str(step), *files],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=dict(os.environ, T1PREP_WEBUI="1", PROGRESS_DIR=str(tmp / "progress")))
        lines = queue.Queue()
        reader = threading.Thread(
            target=lambda: [lines.put(line) for line in process.stdout], daemon=True)
        reader.start()
        was_running = set()
        deadline = time.time() + 120
        while (process.poll() is None or reader.is_alive() or not lines.empty()):
            self.assertLess(time.time(), deadline, "parallelize did not end")
            try:
                run.feed(lines.get(timeout=0.05), time.time())
            except queue.Empty:
                pass
            run.poll()
            was_running |= {path for path, per in run.states.items()
                            if per["volume"] == RUNNING}
        before_the_end = {Path(path).stem: per["volume"] for path, per in run.states.items()}
        run.finish()
        states = {Path(path).stem: per["volume"] for path, per in run.states.items()}
        return states, before_the_end, was_running

    EXPECTED = {"a": DONE, "b": DONE, "bad": FAILED, "d": DONE, "e": DONE}

    def test_one_job(self):
        states, before_the_end, was_running = self.follow(1)
        self.assertEqual(states, self.EXPECTED)
        # Known while it ran, not only once the process had ended
        self.assertEqual(before_the_end, self.EXPECTED)
        self.assertEqual(len(was_running), 5)

    def test_two_jobs(self):
        states, before_the_end, was_running = self.follow(2)
        self.assertEqual(states, self.EXPECTED)
        self.assertEqual(before_the_end, self.EXPECTED)
        self.assertTrue(was_running)


class TestParallelizeLog(unittest.TestCase):
    def test_only_the_run_s_own_log_directory_is_read(self):
        with tempfile.TemporaryDirectory() as mine, tempfile.TemporaryDirectory() as other:
            host = socket.gethostname()
            first = Path(mine) / f"parallelize_{host}_1.log"
            second = Path(mine) / f"parallelize_{host}_2.log"
            (Path(other) / f"parallelize_{host}_3.log").write_text("x")
            first.write_text("a")
            second.write_text("b")
            os.utime(first, (100, 100))
            os.utime(second, (200, 200))
            # Oldest first: the order the phases ran
            self.assertEqual(find_parallelize_logs(mine), [first, second])
            self.assertEqual(find_parallelize_logs(None), [])


class TestImagesOfTheSameName(unittest.TestCase):
    def test_a_name_that_two_images_share_names_neither(self):
        run = RunTracker(["/x/t1.nii", "/y/t1.nii"], ["surface"], Disk())
        run.feed("Surface Estimation", 1.0)
        run.feed("Error in t1", 1.0)
        self.assertEqual([per["surface"] for per in run.states.values()],
                         [QUEUED, QUEUED])
        run.feed("Error in /y/t1.nii", 1.0)
        self.assertEqual(run.states["/y/t1.nii"]["surface"], FAILED)


if __name__ == "__main__":
    unittest.main()
