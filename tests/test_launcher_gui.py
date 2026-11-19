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

# What stands in for the two viewers when a picture is rendered: it notes how
# it was called and writes a picture -- wider than high for the thickness,
# higher than wide otherwise, as the real ones are.  A subject called broken*
# gets none.
_VIEWER = '''\
import json, os, struct, sys, zlib

arguments = sys.argv[1:]
with open(os.environ["VIEWER_LOG"], "a") as log:
    log.write(json.dumps(arguments) + "\\n")
if any("broken" in os.path.basename(argument) for argument in arguments[:1]):
    sys.exit(1)
target = arguments[arguments.index(
    "--output" if "--output" in arguments else "--screenshot") + 1]
width, height = (60, 30) if "--views" in arguments else (30, 40)


def chunk(kind, body):
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff))


rows = (b"\\x00" + bytes((200, 60, 60)) * width) * height
with open(target, "wb") as picture:
    picture.write(b"\\x89PNG\\r\\n\\x1a\\n"
                  + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
'''

# Runs in the child: drives the window and prints what it saw as JSON
_DRIVER = '''\
import json, os, sys, time
from pathlib import Path

tmp, standin = Path(sys.argv[1]), sys.argv[2]
from PySide6 import QtCore, QtGui, QtWidgets
from t1prep.gui import icons, theme
from t1prep.gui.launcher import Launcher
from t1prep.gui.viewer_common import qt_application
from t1prep.launch import results as res

# The pictures are drawn by the stand-in next to this file, not by the viewers
res.VOLUME_VIEWER = res.SURFACE_VIEWER = "standin_viewer"

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


def viewer_calls():
    log = tmp / "viewer.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


win = Launcher(command=standin, settings=settings(), cache=tmp / "cache")
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
seen["renders"] = {"before": win.previews.workers()}
seen["started"] = win.start()
seen["renders"]["in_a_run"] = win.previews.workers()
seen["locked_while_running"] = not box["seg"].isEnabled()
seen["finished"] = wait(lambda: not win.is_running())
seen["renders"]["after"] = win.previews.workers()
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
details = win.details
seen["details"] = {"name": details.name.full_text(),
                   "side": win.side_switch.current(),
                   "segmentation": details.pictures._buttons["volume"].isEnabled(),
                   "thickness": details.pictures._buttons["thickness"].isEnabled(),
                   "shown": details.picture(),
                   "open": details.open_button.isEnabled()}

# -- the pictures: there before they are asked for, in a box of one size ---
# (as T1Prep names the results of sub-..._T1w.nii.gz: without the _T1w)
LONG = "sub-0123456789_ses-baseline_acq-MPRAGE_run-01_desc-aVeryLongDescriptionOfTheScanAndItsSite"
report = {"qualitymeasures": {"IQR": {"value": 2.08, "grade": "A-"}}}


def results_of(folder, name):
    for path in ("report/log_%s.json", "mri/p0%s.nii", "mri/m%s.nii",
                 "surf/lh.central.%s.gii", "surf/rh.central.%s.gii",
                 "surf/lh.thickness.%s", "surf/rh.thickness.%s"):
        target = folder / (path % name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report) if path.endswith(".json") else "")


for subject in ("a", "b", LONG):
    results_of(tmp / "study", subject)
win.remove_all()
before = len(viewer_calls())
seen["study_found"] = win.add_results(tmp / "study")
seen["waiting_at_once"] = win.previews.waiting()
seen["count_while_preparing"] = win.count_label.text()
seen["prepared"] = wait(lambda: win.previews.waiting() == 0)
cache = tmp / "cache"
seen["unfinished_left"] = len(list(cache.glob("*.part.png")))
seen["pictures_of_study"] = len(viewer_calls()) - before
seen["viewer_calls"] = viewer_calls()[before:]
names = [row.subject.name for row in win.model.rows]
boxes, drawn = [], []
for index in range(len(names)):
    win.table.selectRow(index)
    for picture in ("volume", "surface", "thickness"):
        details.pictures._buttons[picture].click()
        app.processEvents()
        boxes.append([details.preview.width(), details.preview.height()])
        drawn.append(details.preview.has_picture())
seen["boxes"], seen["drawn_at_once"] = boxes, drawn
seen["drawn_without_another_render"] = len(viewer_calls()) - before
win.table.selectRow(names.index(LONG))
app.processEvents()
seen["long"] = {"full": details.name.full_text() == LONG, "shown": details.name.text(),
                "tooltip": details.name.toolTip() == LONG,
                "body": details.widget().width(), "viewport": details.viewport().width()}

# -- again on request ----------------------------------------------------------
before = len(viewer_calls())
win._render_again()
wait(lambda: win.previews.waiting() == 0)
seen["rendered_again"] = len(viewer_calls()) - before
seen["drawn_after_again"] = details.preview.has_picture()

# -- several are drawn at once, and the one that is looked at does not wait ----------
keys = [win.previews.key(row.found(), picture) for row in win.model.rows
        for picture in ("volume", "surface", "thickness")]


def draw_all_again(workers):
    win.table.clearSelection()
    wait(lambda: win.previews.waiting() == 0)
    win.previews.set_workers(workers)
    win.previews.forget(keys)
    win._prepare_pictures()
    return win.previews.rendering()


at_once = {"one": draw_all_again(1)}
win.table.selectRow(2)          # a picture other than the one being drawn
at_once["and_the_one_asked_for"] = win.previews.rendering()
at_once["three"] = draw_all_again(3)
wait(lambda: win.previews.waiting() == 0)
at_once["all_there"] = all(win.previews.path(key).is_file() for key in keys)
at_once["none_left"] = win.previews.rendering()
seen["at_once"] = at_once

# -- a picture that cannot be drawn is not tried over and over -------------------
results_of(tmp / "bad", "broken")
before = len(viewer_calls())
win.add_results(tmp / "bad")
wait(lambda: win.previews.waiting() == 0)
for _ in range(3):
    win._prepare_pictures()
    wait(lambda: win.previews.waiting() == 0)
seen["tries_of_a_broken_one"] = len(viewer_calls()) - before
win.table.selectRow(len(win.model.rows) - 1)
wait(lambda: win.previews.waiting() == 0)
app.processEvents()
seen["broken_says"] = details.preview.text()

# -- an image of a BIDS dataset whose results an earlier T1Prep wrote ---------------
dataset = tmp / "dataset"
image = dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
image.parent.mkdir(parents=True)
image.write_text("")
earlier = dataset / "derivatives" / "T1Prep-v0.1.0" / "sub-01" / "anat"
earlier.mkdir(parents=True)
(earlier / "log_sub-01.json").write_text(json.dumps(report))
(earlier / "p0sub-01.nii").write_text("")
win.remove_all()
win.add_images([image])
win.table.selectRow(0)
app.processEvents()
row = win.model.rows[0]
seen["earlier_version"] = {
    "volume": row.shown("volume"), "grade": row.grade, "fixed": row.fixed,
    "folder_shown": details.path.full_text() == str(earlier),
    "run_writes_elsewhere": str(row.subject.folder) != str(earlier),
    "side": win.side_switch.current(),
    "picture": wait(lambda: details.preview.has_picture(), 30)}
wait(lambda: win.previews.waiting() == 0)

# -- the window is dark, the rows are numbered, the form is short ---------------------
def lightness(role):
    return app.palette().color(role).lightness()


seen["dark"] = {"window": lightness(QtGui.QPalette.ColorRole.Window),
                "text": lightness(QtGui.QPalette.ColorRole.WindowText)}
seen["menus"] = {menu.title(): [item.text() for item in menu.actions() if item.text()]
                 for menu in win.menuBar().findChildren(QtWidgets.QMenu)}
win.remove_all()
win.add_results(tmp / "study")
vertical = QtCore.Qt.Orientation.Vertical
seen["numbers"] = {
    "shown": not win.table.verticalHeader().isHidden(),
    "texts": [win.model.headerData(row, vertical) for row in range(len(win.model.rows))],
    "first_name": win.model.index(0, 0).data()}
seen["form"] = sorted(box)

# -- icons ---------------------------------------------------------------------------
def painted(name):
    image = icons.pixmap(name, "#000000", 16, 2.0).toImage()
    return sum(image.pixelColor(x, y).alpha() > 0
               for x in range(image.width()) for y in range(image.height()))


seen["icons"] = {name: painted(name) for name in icons.NAMES}
seen["buttons_without_icon"] = [
    button.text() for button in (
        win.add_images_button, win.add_folder_button, win.open_results_button,
        win.run_button, win.stop_button, details.open_button, details.log_link,
        details.folder_link, *details.pictures._buttons.values())
    if button.icon().isNull()]

# -- the options and the look are kept ---------------------------------------------
box["p"].setChecked(True)
win.close()
again = Launcher(command=standin, settings=settings(), cache=tmp / "cache")
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
        (cls.tmp / "standin_viewer.py").write_text(textwrap.dedent(_VIEWER))
        environment = dict(
            os.environ, QT_QPA_PLATFORM="offscreen", T1PREP_NO_APPS="1",
            VIEWER_LOG=str(cls.tmp / "viewer.log"),
            PYTHONPATH=os.pathsep.join(
                [str(_SRC), str(cls.tmp)]
                + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
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
        # draw the label on once the input is unknown ...
        self.assertFalse(details["segmentation"])
        # ... and the panel turns to a picture there is something to show of
        self.assertEqual(details["shown"], "thickness")
        self.assertTrue(details["open"])

    def test_the_options_are_kept_between_sessions(self):
        self.assertTrue(self.seen["kept"])

    def test_the_pictures_are_there_before_they_are_asked_for(self):
        # Three images with three pictures each, queued when the results are
        # opened and not when a row is selected
        self.assertEqual(self.seen["study_found"], 3)
        self.assertGreater(self.seen["waiting_at_once"], 0)
        self.assertIn("preparing", self.seen["count_while_preparing"])
        self.assertTrue(self.seen["prepared"])
        self.assertEqual(self.seen["pictures_of_study"], 9)
        self.assertEqual(self.seen["unfinished_left"], 0)
        # Going through the rows and the pictures then draws nothing anew
        self.assertEqual(self.seen["drawn_at_once"], [True] * 9)
        self.assertEqual(self.seen["drawn_without_another_render"], 9)

    def test_every_picture_gets_the_same_box(self):
        # The stand-in's pictures are 30 x 40 and 60 x 30; the names are one
        # letter and ninety
        boxes = self.seen["boxes"]
        self.assertEqual(len(boxes), 9)
        self.assertEqual({tuple(box) for box in boxes}, {tuple(boxes[0])})
        width, height = boxes[0]
        self.assertEqual(width, height)
        self.assertGreater(width, 200)

    def test_a_long_name_is_shortened_not_given_the_room(self):
        long = self.seen["long"]
        self.assertTrue(long["full"] and long["tooltip"])
        self.assertIn("…", long["shown"])
        self.assertTrue(long["shown"].startswith("sub-0123") and long["shown"].endswith("Site"))
        # The panel's content is as wide as the panel, not as the name
        self.assertEqual(long["body"], long["viewport"])

    def test_the_viewers_draw_the_pictures_without_a_window(self):
        calls = self.seen["viewer_calls"]
        volumes = [call for call in calls if "--contour" in call]
        surfaces = [call for call in calls if "--contour" not in call and "--headless" in call]
        thickness = [call for call in calls if "--output" in call]
        self.assertEqual((len(volumes), len(surfaces), len(thickness)), (3, 3, 3))
        for call in volumes:
            at = call.index("--contour-level")
            self.assertEqual(call[at + 1:at + 3], ["1.5", "2.5"])    # both borders
            self.assertIn("--headless", call)
        for call in thickness:
            self.assertEqual(call[call.index("--views") + 1], "4")
        # A picture is written under another name until it is complete
        for call in calls:
            target = call[call.index("--output" if "--output" in call else "--screenshot") + 1]
            self.assertTrue(target.endswith(".part.png"), target)

    def test_the_pictures_of_an_image_are_drawn_again_on_request(self):
        self.assertEqual(self.seen["rendered_again"], 3)
        self.assertTrue(self.seen["drawn_after_again"])

    def test_several_pictures_are_drawn_at_once(self):
        at_once = self.seen["at_once"]
        self.assertEqual(at_once["one"], 1)
        self.assertEqual(at_once["three"], 3)
        self.assertTrue(at_once["all_there"])
        self.assertEqual(at_once["none_left"], 0)

    def test_the_picture_that_is_looked_at_does_not_wait_for_room(self):
        # One render at a time, and it is busy with a picture drawn ahead:
        # the one of the selected image starts beside it
        self.assertEqual(self.seen["at_once"]["and_the_one_asked_for"], 2)

    def test_a_run_leaves_one_render_at_a_time(self):
        from t1prep.gui.launcher import render_processes

        renders = self.seen["renders"]
        self.assertEqual(renders["before"], render_processes())
        self.assertTrue(1 <= renders["before"] <= 8)
        self.assertEqual(renders["in_a_run"], 1)
        self.assertEqual(renders["after"], renders["before"])

    def test_a_picture_that_fails_is_not_tried_over_and_over(self):
        # Once per picture when it is prepared; asking for it by selecting
        # the image tries the one that is shown once more
        self.assertEqual(self.seen["tries_of_a_broken_one"], 3)
        self.assertEqual(self.seen["broken_says"], "No picture available")

    def test_an_image_shows_the_results_an_earlier_version_left(self):
        # derivatives/T1Prep-v0.1.0 of a BIDS dataset, while this T1Prep
        # writes to the folder of its own version
        earlier = self.seen["earlier_version"]
        self.assertEqual(earlier["volume"], "done")
        self.assertEqual(earlier["grade"], "A-")
        self.assertTrue(earlier["folder_shown"] and earlier["run_writes_elsewhere"])
        self.assertEqual(earlier["side"], "results")
        self.assertTrue(earlier["picture"])
        self.assertFalse(earlier["fixed"])          # it can still be run

    def test_the_window_is_dark_and_only_dark(self):
        dark = self.seen["dark"]
        self.assertLess(dark["window"], 60)
        self.assertGreater(dark["text"], 190)
        # ... with nothing to switch it by
        self.assertNotIn("Appearance", self.seen["menus"].get("View", []))
        self.assertIn("Prepare All Pictures", self.seen["menus"]["View"])

    def test_the_images_are_numbered(self):
        numbers = self.seen["numbers"]
        self.assertTrue(numbers["shown"])
        self.assertEqual(numbers["texts"], ["1", "2", "3"])
        self.assertEqual(numbers["first_name"], "a")       # the name stays the name

    def test_the_form_leaves_the_settings_of_the_method_out(self):
        form = self.seen["form"]
        for option in ("thickness_method", "correct_folding", "median_filter", "pre_fwhm"):
            self.assertNotIn(option, form)
        for option in ("amap", "vessel", "nn_backend", "skullstrip", "hemi", "debug"):
            self.assertIn(option, form)                    # the other expert options stay

    def test_every_button_of_a_file_a_folder_a_run_or_a_view_has_its_icon(self):
        self.assertEqual(self.seen["buttons_without_icon"], [])
        for name, pixels in self.seen["icons"].items():
            with self.subTest(icon=name):
                self.assertGreater(pixels, 40)


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
