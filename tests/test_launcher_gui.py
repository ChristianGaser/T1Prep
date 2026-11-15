"""The launcher window, driven end to end with a stand-in T1Prep.

The window is built on Qt's "offscreen" platform in a process of its own.
That it can be is part of what is tested: the launcher imports no VTK and
opens no OpenGL context, which is what lets it start on a machine where the
viewers cannot.  The separate process also keeps the offscreen platform away
from the viewer tests of the same session, which need a real one.

The stand-in prints what ``scripts/T1Prep`` prints for a front end and writes
the files the phases end with, so a run goes through the same code as a real
one: the process group, the reading of the output, the states of the table,
the look at the disk afterwards.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

try:
    import PySide6  # noqa: F401
except Exception as exc:  # pragma: no cover - PySide6 is a dependency
    raise unittest.SkipTest(f"PySide6 unavailable: {exc}")
if not hasattr(os, "setsid"):  # pragma: no cover - the launcher is POSIX only
    raise unittest.SkipTest("no process sessions on this platform")

# What stands in for scripts/T1Prep: T1Prep's own loop over the images, with
# a header and bars per phase.  An image called bad* fails its segmentation;
# with STANDIN=hang it waits after the volumes, to be stopped.
_STANDIN = '''\
#!{python}
import json, os, sys, time
from pathlib import Path

report = {{"qualitymeasures": {{"IQR": {{"value": 2.08, "grade": "A-"}}}},
          "subjectmeasures": {{"vol_TIV": {{"value": 1433.0}}}}}}
files = [a for a in sys.argv[1:] if a.endswith((".nii", ".nii.gz")) and os.path.isfile(a)]
failed = 0
for image in files:
    base, name = Path(image).parent, Path(image).name.split(".nii")[0]
    print("Processing " + image, flush=True)
    print("Volume Segmentation", flush=True)
    for percent in (25, 25, 50, 100):
        print("|%-40s| %d%% step" % ("#" * (percent * 40 // 100), percent), flush=True)
    if name.startswith("bad"):
        print("Segmentation failed", flush=True)
        print("Error in " + image, flush=True)
        failed += 1
        continue
    for folder in ("report", "mri", "surf"):
        (base / folder).mkdir(exist_ok=True)
    (base / "report" / ("log_%s.json" % name)).write_text(json.dumps(report))
    (base / "mri" / ("p0%s.nii" % name)).write_text("")
    if os.environ.get("STANDIN") == "hang":
        print("waiting", flush=True)
        time.sleep(120)
    print("Surface Estimation", flush=True)
    for percent in (50, 100):
        print("|%-40s| %d%% step" % ("#" * (percent * 40 // 100), percent), flush=True)
    for hemi in ("lh", "rh"):
        (base / "surf" / ("%s.thickness.%s" % (hemi, name))).write_text("")
    print("Finished after 1s", flush=True)
sys.exit(failed)
'''

# Runs in the child: drives the window and prints what it saw as JSON
_DRIVER = '''\
import json, os, sys, time
from pathlib import Path

tmp, standin = Path(sys.argv[1]), sys.argv[2]
from PySide6 import QtCore
from t1prep.gui import theme
from t1prep.gui.launcher import Launcher
from t1prep.gui.viewer_common import qt_application

seen = {"vtk": sorted(name for name in sys.modules if name.startswith("vtk"))}
app = qt_application()
theme.apply(app)


def settings():
    return QtCore.QSettings(str(tmp / "settings.ini"), QtCore.QSettings.Format.IniFormat)


def wait(done, seconds=60):
    deadline = time.time() + seconds
    while not done() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    return done()


def rows(win):
    return {row.subject.name: {"volume": row.shown("volume"), "surface": row.shown("surface"),
                               "grade": row.grade, "fixed": row.fixed}
            for row in win.model.rows}


win = Launcher(command=standin, settings=settings())
win.show()
box = win.options._controls
seen["run_enabled_when_empty"] = win.run_button.isEnabled()

# -- the form ----------------------------------------------------------
box["fmriprep"].setChecked(True)
seen["fmriprep"] = {"bids_shown": box["bids"].isChecked(), "bids_enabled": box["bids"].isEnabled(),
                    "gz_shown": box["gz"].isChecked(), "mwp_shown": box["mwp"].isChecked(),
                    "bids_chosen": win.options.values()["bids"]}
box["fmriprep"].setChecked(False)
seen["after_fmriprep"] = {"bids_shown": box["bids"].isChecked(), "bids_enabled": box["bids"].isEnabled(),
                          "mwp_shown": box["mwp"].isChecked()}
box["surf"].setChecked(False)
seen["without_surfaces"] = {"pial_white_enabled": box["pial_white"].isEnabled(),
                            "mwp_enabled": box["mwp"].isEnabled()}
box["seg"].setChecked(False)
seen["both_off"] = {"seg": win.options.values()["seg"], "surf": win.options.values()["surf"]}
win.options.reset()

# -- images ------------------------------------------------------------
data = tmp / "data"
data.mkdir()
images = [data / "good.nii", data / "bad.nii"]
for image in images:
    image.write_text("")
(data / "notes.txt").write_text("")
seen["added"] = [win.add_images(images + [data / "notes.txt"]), win.add_images(images)]
seen["run_enabled_with_images"] = win.run_button.isEnabled()

# -- a run: one image passes, one fails ----------------------------------
seen["started"] = win.start()
seen["locked_while_running"] = not box["seg"].isEnabled()
seen["finished"] = wait(lambda: not win.is_running())
seen["steps_logged"] = win.log.toPlainText().count("  25%  step")
seen["first_run"] = {"rows": rows(win), "status": win.status_label.text(),
                     "log_shown": win.log.isVisible(), "count": win.count_label.text(),
                     "unlocked": box["seg"].isEnabled()}

# -- again, leaving out what is finished ---------------------------------
box["skip_existing"].setChecked(True)
win.start()
wait(lambda: not win.is_running())
seen["second_command"] = win.log.toPlainText().splitlines()[0]
box["skip_existing"].setChecked(False)

# -- stopping ------------------------------------------------------------
other = tmp / "other"
other.mkdir()
(other / "slow.nii").write_text("")
win.add_images([other / "slow.nii"])
slow = win.model.rows[-1]
os.environ["STANDIN"] = "hang"
win.start([slow])
# The table follows the output a moment later (it is redrawn on a timer)
wait(lambda: "waiting" in win.log.toPlainText() and slow.states.get("volume") == "running")
pid = int(win.process.processId())
seen["running_states"] = dict(slow.states)
win._terminate()
seen["stopped"] = wait(lambda: not win.is_running(), 20)
time.sleep(0.2)
try:
    os.kill(pid, 0)
    seen["process_left"] = True
except OSError:
    seen["process_left"] = False
seen["after_stop"] = {"rows": rows(win)["slow"], "status": win.status_label.text()}
del os.environ["STANDIN"]

# -- results found on disk -----------------------------------------------
win.remove_all()
seen["found"] = win.add_results(tmp)
seen["results"] = rows(win)
win.table.selectRow(0)
app.processEvents()
seen["details"] = {"name": win.details.name.text(),
                   "side": win.side_switch.current(),
                   "segmentation": win.details._views["segmentation"].isEnabled(),
                   "thickness": win.details._views["thickness"].isEnabled()}

# -- the options are kept ------------------------------------------------
box["p"].setChecked(True)
win.close()
again = Launcher(command=standin, settings=settings())
seen["kept"] = again.options.values()["p"]
again.close()
print("RESULT " + json.dumps(seen))
'''


class TestTheLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp()).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        standin = cls.tmp / "T1Prep"
        standin.write_text(_STANDIN.format(python=sys.executable))
        standin.chmod(0o755)
        driver = cls.tmp / "driver.py"
        driver.write_text(textwrap.dedent(_DRIVER))
        environment = dict(
            os.environ, QT_QPA_PLATFORM="offscreen", T1PREP_NO_APPS="1",
            PYTHONPATH=os.pathsep.join(
                [str(_SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        environment.pop("STANDIN", None)
        done = subprocess.run(
            [sys.executable, str(driver), str(cls.tmp), str(standin)],
            capture_output=True, text=True, timeout=300, env=environment)
        cls.output = done.stdout + done.stderr
        line = next((text for text in done.stdout.splitlines()
                     if text.startswith("RESULT ")), None)
        cls.seen = json.loads(line[7:]) if line else None

    def setUp(self):
        self.assertIsNotNone(self.seen, f"the window did not get through:\n{self.output}")

    def test_no_vtk_is_imported(self):
        self.assertEqual(self.seen["vtk"], [])

    def test_nothing_to_run_without_images(self):
        self.assertFalse(self.seen["run_enabled_when_empty"])
        self.assertTrue(self.seen["run_enabled_with_images"])

    def test_only_images_are_added_and_only_once(self):
        self.assertEqual(self.seen["added"], [2, 0])

    def test_what_an_option_forces_is_shown_and_given_back(self):
        forced = self.seen["fmriprep"]
        self.assertTrue(forced["bids_shown"] and forced["gz_shown"])
        self.assertFalse(forced["bids_enabled"])
        self.assertFalse(forced["mwp_shown"])
        self.assertFalse(forced["bids_chosen"])          # the user's own choice
        after = self.seen["after_fmriprep"]
        self.assertFalse(after["bids_shown"])
        self.assertTrue(after["bids_enabled"] and after["mwp_shown"])

    def test_what_belongs_to_a_step_is_disabled_without_it(self):
        self.assertFalse(self.seen["without_surfaces"]["pial_white_enabled"])
        self.assertTrue(self.seen["without_surfaces"]["mwp_enabled"])

    def test_one_step_always_runs(self):
        self.assertEqual(self.seen["both_off"], {"seg": False, "surf": True})

    def test_a_run_marks_what_passed_and_what_failed(self):
        self.assertTrue(self.seen["started"] and self.seen["finished"])
        self.assertTrue(self.seen["locked_while_running"])
        run = self.seen["first_run"]
        self.assertEqual(run["rows"]["good"],
                         {"volume": "done", "surface": "done", "grade": "A-", "fixed": False})
        self.assertEqual(run["rows"]["bad"]["volume"], "failed")
        self.assertEqual(run["rows"]["bad"]["surface"], "")
        self.assertIn("1 of 2 images failed", run["status"])
        self.assertTrue(run["log_shown"])
        self.assertIn("1 with results", run["count"])
        self.assertTrue(run["unlocked"])

    def test_the_log_keeps_each_step_once(self):
        # The stand-in prints the 25 % bar twice per image, as a terminal
        # bar is redrawn; the log names the step once for each of the two
        self.assertEqual(self.seen["steps_logged"], 2)

    def test_finished_images_are_left_out_on_request(self):
        self.assertIn("bad.nii", self.seen["second_command"])
        self.assertNotIn("good.nii", self.seen["second_command"])

    def test_stopping_ends_the_process_and_is_not_a_failure(self):
        self.assertEqual(self.seen["running_states"],
                         {"volume": "running", "surface": "queued"})
        self.assertTrue(self.seen["stopped"])
        self.assertFalse(self.seen["process_left"])
        self.assertEqual(self.seen["after_stop"]["status"], "Stopped.")
        # The report was written before the stop: that phase is there
        self.assertEqual(self.seen["after_stop"]["rows"]["volume"], "done")
        self.assertEqual(self.seen["after_stop"]["rows"]["surface"], "")

    def test_results_are_found_on_disk(self):
        self.assertEqual(self.seen["found"], 2)
        self.assertEqual(sorted(self.seen["results"]), ["good", "slow"])
        self.assertTrue(all(row["fixed"] for row in self.seen["results"].values()))
        details = self.seen["details"]
        self.assertEqual(details["name"], "good")
        self.assertEqual(details["side"], "results")
        self.assertTrue(details["thickness"])
        # The stand-in keeps no corrected image, so there is nothing to
        # draw the label on once the input is unknown
        self.assertFalse(details["segmentation"])

    def test_the_options_are_kept_between_sessions(self):
        self.assertTrue(self.seen["kept"])


class TestTheCommandLine(unittest.TestCase):
    """``t1prep-gui`` answers like every tool; only a bare call opens the window."""

    def call(self, *arguments):
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=os.pathsep.join(
            [str(_SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
        return subprocess.run(
            [sys.executable, "-m", "t1prep.gui.launcher", *arguments],
            capture_output=True, text=True, timeout=120, env=environment)

    def test_help_and_version(self):
        done = self.call("--help")
        self.assertEqual(done.returncode, 0)
        self.assertIn("usage: t1prep-gui", done.stdout)
        self.assertIn("--out-dir", done.stdout)
        done = self.call("--version")
        self.assertEqual(done.returncode, 0)
        self.assertTrue(done.stdout.startswith("t1prep-gui "))

    def test_an_unknown_option_points_at_help(self):
        done = self.call("--no-such-option")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("t1prep-gui --help", done.stderr)


if __name__ == "__main__":
    unittest.main()
