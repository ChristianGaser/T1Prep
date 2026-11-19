"""The T1Prep launcher: choose images, run the pipeline, look at the results.

One window with the images in a table, the options beside it and a button
that starts the bash orchestrator ``T1Prep`` on them.  While it runs, the
table shows where each image is; once a phase has written its results they
can be looked at — as a picture in the window and in the viewers.

How the pieces divide:

* ``t1prep.launch`` holds everything that is not Qt: the option table and the
  arguments it becomes, the states of a run, the files of a subject;
* ``launcher_widgets`` holds the table, the form and the results panel;
* this module holds the window, the T1Prep process and the two kinds of
  viewer processes.

The launcher never imports VTK and opens no OpenGL context.  ``CAT_VolView``
and ``CAT_SurfView`` do the drawing, each time in a process of its own:
without a window for the pictures in the results panel, with one when a view
is opened.  A graphics driver that fails therefore costs the pictures, not
the launcher or the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..cli_help import ArgumentParser, version
from ..launch import options as opt
from ..launch import results as res
from ..launch.inputs import find_images, is_image
from ..launch.progress import (
    DONE, PHASE_TITLES, RUNNING, RunTracker, find_parallelize_logs,
    format_duration, step_text,
)
from ..launch.results import PICTURES, SURFACE, VOLUME, Subject
from . import icons, theme
from .launcher_widgets import (
    STAGES, DetailsPanel, FolderDialog, OptionsPanel, Row, Segments,
    SubjectModel, SubjectTable,
)
from .viewer_common import (
    VOLUME_SUFFIXES, finder_open_files, qt_application, shorten_path,
)

DOCUMENTATION = "https://github.com/ChristianGaser/T1Prep#readme"

# Runs T1Prep as the leader of a session of its own, so that stopping reaches
# everything it started (parallelize, the Python and the surface processes)
# and nothing else.  exec keeps the process id, which is then the group's.
_OWN_SESSION = ("import os, sys; os.setsid(); "
                "os.execvp(sys.argv[1], sys.argv[1:])")

#: Approximate half-width in pixels of the pictures the viewer renders
PREVIEW_SIZE = 340


def _black_surround(image: QtGui.QImage) -> QtGui.QImage:
    """Turn the white around the panes of a viewer screenshot black.

    ``CAT_VolView`` draws its three panes on white; in the results panel the
    picture sits on black.  Only the white connected to the border is
    changed, so bright tissue inside a pane stays as it is.
    """
    try:
        import numpy as np
        from scipy import ndimage

        image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.constBits(), np.uint8).reshape(
            image.height(), image.bytesPerLine())[:, : image.width() * 4]
        pixels = pixels.reshape(image.height(), image.width(), 4).copy()
        labels, _ = ndimage.label((pixels[..., :3] > 250).all(axis=2))
        border = np.unique(np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        pixels[np.isin(labels, border[border > 0])] = (0, 0, 0, 255)
        result = QtGui.QImage(pixels.data, image.width(), image.height(),
                              image.width() * 4, QtGui.QImage.Format.Format_RGBA8888)
        return result.copy()           # owns its pixels
    except Exception:
        return image


#: Part of the name of every cached picture; raised when the pictures change
#: what they show, so those of an earlier version are not taken for current
PICTURE_VERSION = 2

#: One picture to draw: whose, which of PICTURES, and the name it is kept under
Job = Tuple[Subject, str, str]


#: The command that starts a process at low priority, where there is one
_NICE = shutil.which("nice")


def render_processes() -> int:
    """How many pictures are rendered at once on this machine.

    A render is one process on one core, with about 0.6 GB at its peak, and
    they do not get in each other's way: eight took 6.7 times the pictures
    of one on a 24-core machine.  Half the cores, eight at most, leave the
    desktop and a run its share.
    """
    return max(1, min(8, (os.cpu_count() or 2) // 2))


class Previews(QtCore.QObject):
    """The pictures of the results panel, rendered by the viewers without a window.

    A picture is kept in the cache folder under a name made of what it shows —
    the files it is drawn from, with their sizes and dates — so it is rendered
    once and again only after a run has replaced those files.

    Each picture is rendered in a process of its own, several of them at a
    time (:meth:`set_workers`).  The one the user is looking at goes first
    (:meth:`request`), at once and whatever else is being rendered.  After it
    come those :meth:`prepare` was last given, in its order: the pictures of
    the images around the selected one, so they are there when the selection
    moves on.

    Args:
        folder: Where the pictures are kept; the cache folder of the
            application by default.
        workers: How many are rendered at once; :func:`render_processes` by
            default.
    """

    ready = Signal(str, str)        # key, path of the PNG ("" when it failed)
    changed = Signal()              # how many pictures are waiting has changed

    #: A render that takes longer than this is ended, in milliseconds
    TIMEOUT = 120_000

    def __init__(self, parent=None, folder: Optional[os.PathLike] = None,
                 workers: Optional[int] = None):
        super().__init__(parent)
        if folder is None:
            base = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.StandardLocation.CacheLocation)
            folder = Path(base or tempfile.gettempdir()) / "previews"
        self.folder = Path(folder)
        self._workers = max(1, int(workers or render_processes()))
        #: The renders that are running, each with the picture it draws
        self._running: Dict[QtCore.QProcess, Job] = {}
        self._wanted: Optional[Job] = None
        self._queue: List[Job] = []
        self._failed: Set[str] = set()
        self._discard_old()

    def _discard_old(self, days: int = 30) -> None:
        limit = time.time() - days * 86400
        try:
            for path in self.folder.glob("*.png"):
                if path.stat().st_mtime < limit:
                    path.unlink()
        except OSError:
            pass

    def key(self, subject: Subject, picture: str) -> Optional[str]:
        """Name of the picture *picture*; None while there is nothing to draw."""
        sources = res.preview_sources(subject, picture)
        if len(sources) < 2:
            return None
        digest = hashlib.sha1(f"{PICTURE_VERSION}:{picture}:{PREVIEW_SIZE}".encode())
        for path in sources:
            try:
                info = path.stat()
            except OSError:
                return None
            digest.update(f"|{path}:{info.st_size}:{info.st_mtime_ns}".encode())
        return digest.hexdigest()[:20]

    def path(self, key: str) -> Path:
        return self.folder / f"{key}.png"

    def _unfinished(self, key: str) -> Path:
        # A picture appears under its name only when it is complete
        return self.folder / f"{key}.part.png"

    def workers(self) -> int:
        """How many pictures are rendered at once."""
        return self._workers

    def set_workers(self, number: int) -> None:
        """Render *number* pictures at once from now on.

        Renders that are running are left to finish; with a smaller number
        no new one starts until enough of them have.
        """
        self._workers = max(1, int(number))
        self._next()

    def rendering(self) -> int:
        """How many renders are running."""
        return len(self._running)

    def waiting(self) -> int:
        """How many pictures are being rendered or still to render."""
        return len(self._queue) + len(self._running) + (self._wanted is not None)

    def _is_rendering(self, key: str) -> bool:
        return any(job[2] == key for job in self._running.values())

    def request(self, subject: Subject, picture: str, key: str) -> None:
        """Render the picture *key* now; :attr:`ready` says when it is there."""
        self._failed.discard(key)
        if self._is_rendering(key):
            return
        self._wanted = (subject, picture, key)
        self._next()
        self.changed.emit()

    def prepare(self, jobs: Sequence[Job]) -> None:
        """Render *jobs* when nothing is asked for, in their order.

        They replace what was given before: the pictures worth having ahead
        change with the selection.  A picture that did not render is not
        tried again until it is asked for by :meth:`request`.
        """
        self._queue = [job for job in jobs if job[2] not in self._failed]
        self._next()
        self.changed.emit()

    def forget(self, keys: Sequence[str]) -> None:
        """Drop the pictures *keys*, so that they are rendered anew."""
        for key in keys:
            self._failed.discard(key)
            try:
                self.path(key).unlink()
            except OSError:
                pass

    def _next(self) -> None:
        """Start as many renders as there is room for.

        The picture that is asked for does not wait for room: it is what the
        user is looking at an empty box for, so it gets a process beyond the
        number that are rendered ahead.
        """
        while True:
            if self._wanted is not None:
                job, self._wanted = self._wanted, None
                ahead = False
            elif self._queue and len(self._running) < self._workers:
                job = self._queue.pop(0)
                # The first of the queue is the picture nearest to what is
                # shown; the others are drawn ahead and yield to it
                ahead = bool(self._running)
            else:
                return
            key = job[2]
            if self._is_rendering(key):
                continue                # its "ready" is on the way
            if self.path(key).is_file():
                self.ready.emit(key, str(self.path(key)))
            elif not self._start(job, ahead):
                self._failed.add(key)
                self.ready.emit(key, "")

    def _start(self, job: Job, ahead: bool = False) -> bool:
        """Start the render of *job*; *ahead* of need it runs at low priority.

        Several renders starting together slow each other down on a machine
        that is busy, and the picture the user is waiting for is one of
        them.  The others are therefore run "nice": they take the time that
        is free and give way to it.
        """
        subject, picture, key = job
        command = res.preview(subject, picture, self._unfinished(key), PREVIEW_SIZE)
        if command is None:
            return False
        if ahead and _NICE:
            command = [_NICE, "-n", "10"] + command
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        process = QtCore.QProcess(self)
        environment = QtCore.QProcessEnvironment()
        for variable, value in res.viewer_environment().items():
            environment.insert(variable, value)
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.finished.connect(lambda *_args, p=process: self._finished(p))
        process.errorOccurred.connect(lambda _error, p=process: self._error(p))
        # One that hangs must not hold its place for good
        watchdog = QtCore.QTimer(process)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(process.kill)      # "finished" follows
        watchdog.start(self.TIMEOUT)
        self._running[process] = job
        process.start(command[0], command[1:])
        return True

    def _error(self, process: QtCore.QProcess) -> None:
        # A process that crashed still reports "finished"; only one that did
        # not start gets no further signal
        if process.state() == QtCore.QProcess.ProcessState.NotRunning:
            self._finished(process)

    def _finished(self, process: QtCore.QProcess) -> None:
        job = self._running.pop(process, None)
        if job is None:
            return                      # reported twice: as an error and as finished
        process.deleteLater()           # and its watchdog with it
        key = job[2]
        part, path = self._unfinished(key), self.path(key)
        done = False
        if (process.exitStatus() == QtCore.QProcess.ExitStatus.NormalExit
                and process.exitCode() == 0 and part.is_file()):
            try:
                os.replace(part, path)
                done = True
            except OSError:
                pass
        if not done:
            self._failed.add(key)
            try:
                part.unlink()
            except OSError:
                pass
        self.ready.emit(key, str(path) if done else "")
        self._next()
        self.changed.emit()

    def shutdown(self) -> None:
        self._wanted = None
        self._queue = []
        for process in list(self._running):
            process.kill()
        for process in list(self._running):
            process.waitForFinished(1000)


class Viewers(QtCore.QObject):
    """The viewer windows the launcher has opened, each a process of its own.

    They are started detached and outlive the launcher.  A viewer that ends
    with an error shortly after the start — no display, a graphics driver
    that fails — is reported, with what it wrote to standard error.
    """

    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._started: List[tuple] = []
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._check)

    def open(self, command: Sequence[str]) -> None:
        errors = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(
                list(command), env=res.viewer_environment(),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=errors, start_new_session=True)
        except OSError as exc:
            errors.close()
            self.failed.emit(f"The viewer could not be started: {exc}")
            return
        self._started.append((process, errors, Path(command[2]).suffix or command[2]))
        self._timer.start()

    def _check(self) -> None:
        for entry in list(self._started):
            process, errors, name = entry
            status = process.poll()
            if status is None:
                continue
            self._started.remove(entry)
            if status != 0:
                errors.seek(0)
                lines = errors.read().decode("utf-8", "replace").strip().splitlines()
                reason = lines[-1] if lines else f"exit status {status}"
                self.failed.emit(f"The viewer ended with an error: {reason}")
            errors.close()
        if not self._started:
            self._timer.stop()


class Launcher(QtWidgets.QMainWindow):
    """The window: images, options, run and results.

    Args:
        command: The ``T1Prep`` to run; the one on ``PATH`` by default.
        settings: Where the window keeps its state between sessions.
        cache: Where the pictures of the results are kept; the cache folder
            of the application by default.
    """

    #: Pictures are rendered ahead for this many images around the selected
    #: one; the whole table on request ("Prepare All Pictures")
    PREPARE_AROUND = 100

    def __init__(self, command: Optional[str] = None,
                 settings: Optional[QtCore.QSettings] = None,
                 cache: Optional[os.PathLike] = None):
        super().__init__()
        self.command = command or opt.t1prep_command()
        self.defaults = opt.load_defaults(opt.defaults_file(self.command))
        self.settings = settings if settings is not None else QtCore.QSettings()
        self.colors = theme.apply(QtWidgets.QApplication.instance())
        self.out_dir: Optional[Path] = None
        self.process: Optional[QtCore.QProcess] = None
        self.tracker: Optional[RunTracker] = None
        self._run_rows: Dict[str, Row] = {}
        self._run_dir: Optional[Path] = None
        self._started = 0.0
        self._cancelled = False
        self._buffer = b""
        self._logged_percent: Optional[int] = None
        self._naming = (False, False)          # (BIDS names, AMAP) of the subjects
        self._wanted_preview: Optional[str] = None
        self._shown_row: Optional[Row] = None
        self._prepare_everything = False

        self.model = SubjectModel(self)
        self.previews = Previews(self, cache)
        self._render_processes = self.previews.workers()
        self.previews.ready.connect(self._preview_ready)
        self.previews.changed.connect(self._update_count)
        self.viewers = Viewers(self)
        self.viewers.failed.connect(self._viewer_failed)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)

        self.setWindowTitle("T1Prep")
        self.setAcceptDrops(True)
        self._build()
        self._build_menu()
        self._set_icons()
        self._restore()
        self._options_changed()
        self._selection_changed()

    # ------------------------------------------------------------------
    # Building the window
    # ------------------------------------------------------------------

    def _build(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        # -- what to process -------------------------------------------
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.add_images_button = QtWidgets.QPushButton("Add images…")
        self.add_images_button.clicked.connect(self.choose_images)
        self.add_folder_button = QtWidgets.QPushButton("Add folder…")
        self.add_folder_button.clicked.connect(self.choose_folder)
        self.open_results_button = QtWidgets.QPushButton("Open results…")
        self.open_results_button.setToolTip(
            "Show the results of an earlier run without processing anything.")
        self.open_results_button.clicked.connect(self.choose_results)
        self.count_label = QtWidgets.QLabel()
        self.count_label.setObjectName("muted")
        for button in (self.add_images_button, self.add_folder_button,
                       self.open_results_button):
            top.addWidget(button)
        top.addStretch(1)
        top.addWidget(self.count_label)
        outer.addLayout(top)

        # -- images and side panel --------------------------------------
        self.table = SubjectTable(self.model, self.colors)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(self._open_best_view)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.remove_requested.connect(self.remove_selected)

        self.side_switch = Segments((("options", "Options"), ("results", "Results")))
        self.options = OptionsPanel(self.defaults)
        self.options.changed.connect(self._options_changed)
        self.details = DetailsPanel(self.colors)
        self.details.open_requested.connect(lambda: self._open_view())
        self.details.picture_changed.connect(self._picture_changed)
        self.details.render_requested.connect(lambda: self._render_again())
        self.details.log_requested.connect(self._show_subject_log)
        self.details.folder_requested.connect(self._open_folder)
        self.side = QtWidgets.QStackedWidget()
        self.side.addWidget(self.options)
        self.side.addWidget(self.details)
        self.side_switch.changed.connect(self._show_side)
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(1, 10, 1, 1)
        panel_layout.setSpacing(6)
        panel_layout.addWidget(self.side_switch, 0, Qt.AlignmentFlag.AlignHCenter)
        panel_layout.addWidget(self.side, 1)
        # Wide enough for the switch between the three pictures
        panel.setMinimumWidth(388)

        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(10)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([640, 410])
        outer.addWidget(self.splitter, 1)

        # -- where the results go, and the run --------------------------
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(8)
        label = QtWidgets.QLabel("Output")
        label.setObjectName("muted")
        self.output = QtWidgets.QComboBox()
        self.output.setMinimumWidth(240)
        self.output.setToolTip(
            "Next to each image: in mri/, surf/, report/ and label/ of its "
            "folder, or in derivatives/ of a BIDS dataset.")
        self.output.activated.connect(self._output_chosen)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.hide()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(lambda: self.start())
        bottom.addWidget(label)
        bottom.addWidget(self.output)
        bottom.addStretch(1)
        bottom.addWidget(self.stop_button)
        bottom.addWidget(self.run_button)
        outer.addLayout(bottom)

        status = QtWidgets.QHBoxLayout()
        status.setSpacing(10)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(200)
        self.progress.hide()
        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("muted")
        self.log_link = QtWidgets.QPushButton("Show log")
        self.log_link.setObjectName("link")
        self.log_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.log_link.clicked.connect(lambda: self.show_log(not self.log.isVisible()))
        status.addWidget(self.progress)
        status.addWidget(self.status_label, 1)
        status.addWidget(self.log_link)
        outer.addLayout(status)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSizeF(max(9.0, font.pointSizeF() - 1))
        self.log.setFont(font)
        self.log.setFixedHeight(170)
        self.log.hide()
        outer.addWidget(self.log)

        self.setCentralWidget(central)
        self._fill_output()
        self.resize(1100, 790)
        self.setMinimumSize(900, 540)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        def action(menu, text, slot, shortcut=None):
            item = QtGui.QAction(text, self)
            if shortcut:
                item.setShortcut(QtGui.QKeySequence(shortcut))
            item.triggered.connect(lambda _checked=False: slot())
            menu.addAction(item)
            return item

        file_menu = bar.addMenu("File")
        self._add_actions = [
            action(file_menu, "Add Images…", self.choose_images, "Ctrl+O"),
            action(file_menu, "Add Folder…", self.choose_folder, "Ctrl+Shift+O"),
        ]
        action(file_menu, "Open Results…", self.choose_results, "Ctrl+E")
        file_menu.addSeparator()
        self._remove_action = action(file_menu, "Remove Selected", self.remove_selected)
        self._remove_all_action = action(file_menu, "Remove All", self.remove_all)

        run_menu = bar.addMenu("Run")
        self._run_action = action(run_menu, "Run", self.start, "Ctrl+R")
        self._run_selected_action = action(
            run_menu, "Run Selected", lambda: self.start(self._selected_rows()))
        self._stop_action = action(run_menu, "Stop", self.stop, "Ctrl+.")

        view_menu = bar.addMenu("View")
        action(view_menu, "Options", lambda: self._show_side("options"), "Ctrl+1")
        action(view_menu, "Results", lambda: self._show_side("results"), "Ctrl+2")
        view_menu.addSeparator()
        self._prepare_action = action(
            view_menu, "Prepare All Pictures", self.prepare_all_pictures)
        self._prepare_action.setToolTip(
            "Render the pictures of every image now, not only of those "
            "around the selected one.")
        view_menu.addSeparator()
        action(view_menu, "Log", lambda: self.show_log(not self.log.isVisible()), "Ctrl+L")

        help_menu = bar.addMenu("Help")
        action(help_menu, "T1Prep Documentation",
               lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(DOCUMENTATION)))

    def _set_icons(self) -> None:
        """Give the buttons their icons, in the colours of the window."""
        c = self.colors
        for button, name in ((self.add_images_button, "add_file"),
                             (self.add_folder_button, "add_folder"),
                             (self.open_results_button, "results")):
            button.setIcon(icons.icon(name, c.text, c.faint))
        self.run_button.setIcon(icons.icon("play", c.on_accent, c.faint))
        self.stop_button.setIcon(icons.icon("stop", c.danger, c.faint))
        self.log_link.setIcon(icons.icon("log", c.accent, c.faint))

    # ------------------------------------------------------------------
    # State between sessions
    # ------------------------------------------------------------------

    def _restore(self) -> None:
        s = self.settings
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        split = s.value("window/splitter")
        if split is not None:
            self.splitter.restoreState(split)
        try:
            saved = json.loads(str(s.value("options/values", "{}")))
        except ValueError:
            saved = {}
        if isinstance(saved, dict):
            for key, value in list(saved.items()):
                if isinstance(value, list):
                    saved[key] = tuple(value)
            self.options.set_values(saved)
        try:
            sections = json.loads(str(s.value("options/sections", "{}")))
        except ValueError:
            sections = {}
        if isinstance(sections, dict):
            self.options.set_sections_open(sections)
        folder = str(s.value("output/folder", "") or "")
        if folder and Path(folder).is_dir():
            self.out_dir = Path(folder)
        self._fill_output()

    def _save(self) -> None:
        s = self.settings
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/splitter", self.splitter.saveState())
        s.setValue("options/values", json.dumps(self.options.values()))
        s.setValue("options/sections", json.dumps(self.options.sections_open()))
        s.setValue("output/folder", str(self.out_dir) if self.out_dir else "")
        s.sync()

    def _last_folder(self) -> str:
        folder = str(self.settings.value("paths/last", "") or "")
        return folder if folder and Path(folder).is_dir() else str(Path.home())

    def _remember_folder(self, path: os.PathLike) -> None:
        self.settings.setValue("paths/last", str(path))

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self.process is not None

    def _subject_for(self, image: Path) -> Subject:
        bids, amap = self._naming
        return res.subject_for_input(image, self.out_dir, bids=bids, amap=amap)

    def add_images(self, paths: Sequence[os.PathLike]) -> int:
        """Add *paths* to the table; images already in it are left as they are.

        Returns:
            The number of images added.
        """
        known = {str(row.subject.input) for row in self.model.rows
                 if row.subject.input is not None}
        rows = []
        for path in paths:
            image = Path(os.path.abspath(str(path)))
            if str(image) in known or not image.is_file() or not is_image(image):
                continue
            known.add(str(image))
            row = Row(self._subject_for(image))
            self._look_at_disk(row)
            rows.append(row)
        self.model.append(rows)
        self._rows_changed()
        return len(rows)

    def add_results(self, folder: os.PathLike) -> int:
        """Add the subjects whose results lie in *folder*; return how many."""
        known = {(str(row.subject.folder), row.subject.name, row.subject.bids)
                 for row in self.model.rows}
        rows = []
        for subject in res.find_results(folder):
            key = (str(subject.folder), subject.name, subject.bids)
            if key in known:
                continue
            known.add(key)
            row = Row(subject, fixed=True)
            self._look_at_disk(row)
            rows.append(row)
        self.model.append(rows)
        self._rows_changed()
        return len(rows)

    def add_paths(self, paths: Sequence[os.PathLike], ask: bool = False) -> None:
        """Add images and folders, as dropped on the window or given at start.

        A folder that holds T1Prep reports is opened as results; any other
        folder contributes its images — after asking which ones when *ask* is
        set, otherwise those directly in it.
        """
        images: List[Path] = []
        for path in map(Path, paths):
            if path.is_dir():
                if self.add_results(path):
                    self._show_side("results")
                elif ask:
                    dialog = FolderDialog(path, self)
                    if dialog.exec():
                        images.extend(dialog.images)
                else:
                    images.extend(find_images(path))
            elif is_image(path):
                images.append(path)
        if images:
            self.add_images(images)

    def choose_images(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add images", self._last_folder(),
            "NIfTI images (*.nii *.nii.gz);;All files (*)")
        if files:
            self._remember_folder(Path(files[0]).parent)
            self.add_images(files)

    def choose_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add the images of a folder", self._last_folder())
        if not folder:
            return
        self._remember_folder(folder)
        dialog = FolderDialog(Path(folder), self)
        if dialog.exec():
            self.add_images(dialog.images)

    def choose_results(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open the results in a folder", self._last_folder())
        if not folder:
            return
        self._remember_folder(folder)
        if self.add_results(folder):
            self._show_side("results")
        else:
            self._say(f"No T1Prep results in {shorten_path(folder)}.")

    def _selected_rows(self) -> List[Row]:
        return [self.model.rows[index] for index in self.table.selected_rows()]

    def remove_selected(self) -> None:
        if self.is_running():
            return
        self.model.remove(self.table.selected_rows())
        self._rows_changed()

    def remove_all(self) -> None:
        if self.is_running():
            return
        self.model.remove(range(len(self.model.rows)))
        self._prepare_everything = False
        self._rows_changed()

    def _look_at_disk(self, row: Row) -> None:
        """Take a row's finished phases and its quality from its files.

        From wherever they are: an image keeps the results of an earlier run
        that used the other names or, in a BIDS dataset, another T1Prep
        version, whose folder is not the one this run writes to.
        """
        found = row.subject if row.fixed else res.results_on_disk(row.subject)
        row.results = None if found == row.subject else found
        row.on_disk = {stage: found.stage_done(stage) for stage in STAGES}
        grade = res.quality(found.report()) if row.on_disk[VOLUME] else None
        row.grade, row.iqr = grade if grade else ("", None)

    def _rows_changed(self) -> None:
        self._update_count()
        self._update_actions()
        self._selection_changed()

    def _update_count(self) -> None:
        """What the table holds, and how many pictures are still being drawn."""
        rows = self.model.rows
        total = len(rows)
        finished = sum(row.has_results() for row in rows)
        text = f"{total} image{'s' if total != 1 else ''}" if total else ""
        if finished:
            text += f"  ·  {finished} with results"
        waiting = self.previews.waiting()
        if waiting:
            text += f"  ·  preparing {waiting} picture{'s' if waiting != 1 else ''}"
        self.count_label.setText(text)

    def _update_actions(self) -> None:
        running = self.is_running()
        runnable = any(not row.fixed for row in self.model.rows)
        selected = self._selected_rows()
        self.run_button.setEnabled(not running and runnable)
        self.run_button.setVisible(not running)
        self.stop_button.setVisible(running)
        self.output.setEnabled(not running)
        for button in (self.add_images_button, self.add_folder_button):
            button.setEnabled(not running)
        for item in self._add_actions:
            item.setEnabled(not running)
        self._run_action.setEnabled(not running and runnable)
        self._run_selected_action.setEnabled(
            not running and any(not row.fixed for row in selected))
        self._stop_action.setEnabled(running)
        self._remove_action.setEnabled(not running and bool(selected))
        self._remove_all_action.setEnabled(not running and bool(self.model.rows))
        self._prepare_action.setEnabled(
            any(row.has_results() for row in self.model.rows))

    # -- drag and drop ---------------------------------------------------

    def _droppable(self, event) -> List[str]:
        paths = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.isdir(path) or path.lower().endswith(VOLUME_SUFFIXES[:2]):
                paths.append(path)
        return paths

    def dragEnterEvent(self, event):       # noqa: N802
        if not self.is_running() and self._droppable(event):
            event.acceptProposedAction()

    def dropEvent(self, event):            # noqa: N802
        paths = self._droppable(event)
        if paths and not self.is_running():
            event.acceptProposedAction()
            # After the drop has returned: a dialog inside it blocks the
            # application the files were dragged from
            QtCore.QTimer.singleShot(0, lambda: self.add_paths(paths, ask=True))

    # ------------------------------------------------------------------
    # Options and output folder
    # ------------------------------------------------------------------

    def _options_changed(self) -> None:
        current = opt.effective(self.options.values())
        naming = (bool(current.get("bids")), bool(current.get("amap")))
        if naming != self._naming:
            self._naming = naming
            self._move_subjects()
        self._update_actions()

    def _move_subjects(self) -> None:
        """Options or output folder changed where the results go: look again."""
        for row in self.model.rows:
            if row.fixed or row.subject.input is None:
                continue
            row.subject = self._subject_for(row.subject.input)
            row.states = {}
            self._look_at_disk(row)
        self.model.refresh()
        self._rows_changed()

    def _fill_output(self) -> None:
        self.output.blockSignals(True)
        self.output.clear()
        self.output.addItem("Next to each image", "")
        if self.out_dir is not None:
            self.output.addItem(shorten_path(str(self.out_dir), max_chars=46),
                                str(self.out_dir))
        self.output.addItem("Choose folder…", "?")
        self.output.setCurrentIndex(1 if self.out_dir is not None else 0)
        self.output.blockSignals(False)

    def _output_chosen(self, index: int) -> None:
        chosen = self.output.itemData(index)
        before = self.out_dir
        if chosen == "?":
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Folder for the results",
                str(self.out_dir or self._last_folder()))
            if folder:
                self.out_dir = Path(folder)
        elif chosen == "":
            self.out_dir = None
        self._fill_output()
        if self.out_dir != before:
            self._move_subjects()

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------

    def start(self, only: Optional[Sequence[Row]] = None) -> bool:
        """Run T1Prep on the images of the table, or on *only* those given.

        Returns:
            Whether a run was started.
        """
        if self.is_running():
            return False
        values = self.options.values()
        current = opt.effective(values)
        issues = opt.problems(values)
        if issues:
            self._say(issues[0])
            return False
        if shutil.which(self.command) is None and not os.path.isfile(self.command):
            self._say("T1Prep was not found. Is it installed in this environment?")
            return False

        stages = [stage for stage, key in ((VOLUME, "seg"), (SURFACE, "surf"))
                  if current[key]]
        skull_only = bool(current["seg"]) and current.get("skullstrip") == "only"
        rows = [row for row in (self.model.rows if only is None else only)
                if not row.fixed and row.subject.input is not None
                and row.subject.input.is_file()]
        if current.get("skip_existing") and only is None:
            rows = [row for row in rows if not all(
                row.subject.stage_done(stage, skullstrip_only=skull_only)
                for stage in stages)]
            if not rows:
                self._say("Nothing to do: every image is finished.")
                return False
        if not rows:
            self._say("There is no image to process.")
            return False
        places = [(str(row.subject.folder), row.subject.name) for row in rows]
        if len(set(places)) < len(places):
            self._say("Some images share a name and would overwrite each "
                      "other's results. Process them into separate folders.")
            return False

        self._run_dir = Path(tempfile.mkdtemp(prefix="t1prep-gui-"))
        logs = self._run_dir / "logs"
        logs.mkdir()
        # Not created here: parallelize makes it when it distributes jobs,
        # and that is how the tracker knows it did
        progress_dir = self._run_dir / "progress"

        command = [self.command] + opt.build_arguments(values, self.defaults)
        if self.out_dir is not None:
            command += ["--out-dir", str(self.out_dir)]
        inputs = [str(row.subject.input) for row in rows]
        command += inputs

        self._started = time.time()
        self._cancelled = False
        self._buffer = b""
        self._logged_percent = None
        self._run_rows = dict(zip(inputs, rows))
        since = self._started - 2      # a clock on a file server may lag
        self.tracker = RunTracker(
            inputs, stages,
            lambda path, stage: self._run_rows[path].subject.stage_done(
                stage, since=since, skullstrip_only=skull_only),
            progress_dir)
        for row in self.model.rows:
            row.states = {}
        self._copy_states()

        process = QtCore.QProcess(self)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("T1PREP_WEBUI", "1")    # phase headers, bars as lines
        environment.insert("PROGRESS_DIR", str(progress_dir))
        environment.insert("T1PREP_LOGDIR", str(logs))
        if not environment.contains("PYTHONUNBUFFERED"):
            environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process

        # The run comes first: a segmentation needs 10-16 GB and the surfaces
        # every core, so the pictures are rendered one at a time while it lasts
        self.previews.set_workers(1)
        self.log.clear()
        self.log.appendPlainText(opt.format_command(command) + "\n")
        self.options.set_locked(True)
        self.progress.setValue(0)
        self.progress.show()
        self._update_actions()
        self.table.set_spinning(True)
        self._timer.start()
        self._update_status()
        process.start(sys.executable, ["-c", _OWN_SESSION] + command)
        return True

    def stop(self) -> None:
        """Ask, then end the run and everything it started."""
        if not self.is_running() or self._cancelled:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Stop the run",
            "Stop processing?\n\nImages that are finished keep their results; "
            "the others can be continued later with “Skip finished images”.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._terminate()

    def _terminate(self) -> None:
        if self.process is None:
            return
        self._cancelled = True
        self.status_label.setText("Stopping…")
        self._signal_run(signal.SIGTERM)
        QtCore.QTimer.singleShot(4000, lambda: self._signal_run(signal.SIGKILL))

    def _signal_run(self, number: int) -> None:
        """Send a signal to T1Prep's process group."""
        if self.process is None:
            return
        pid = int(self.process.processId())
        if pid <= 0:
            return
        try:
            os.killpg(pid, number)
        except (ProcessLookupError, PermissionError):
            try:       # the session was never set up: the process alone
                os.kill(pid, number)
            except OSError:
                pass

    def _read_output(self) -> None:
        if self.process is None or self.tracker is None:
            return
        data = self._buffer + bytes(self.process.readAllStandardOutput())
        lines = data.replace(b"\r", b"\n").split(b"\n")
        self._buffer = lines.pop()             # the unfinished last line
        now = time.time()
        keep = []
        for raw in lines:
            line = raw.decode("utf-8", "replace")
            event = self.tracker.feed(line, now)
            if event is not None and event[0] == "phase":
                self._logged_percent = None
            elif event is not None and event[0] == "bar":
                # A bar is redrawn many times: one line per step is enough,
                # and that line says the step, it does not draw the bar
                if event[1] == self._logged_percent:
                    continue
                self._logged_percent = int(event[1])
                line = step_text(line) or line
            keep.append(line.rstrip())
        if keep:
            self.log.appendPlainText("\n".join(keep))

    def _tick(self) -> None:
        """While a run is on: read what it printed and show where it is."""
        if self.tracker is None:
            return
        self._read_output()
        self.tracker.poll()
        self._copy_states()
        self._update_status()

    def _copy_states(self) -> None:
        """Put the tracker's states into the rows; look up what just finished."""
        if self.tracker is None:
            return
        fresh = False
        for path, row in self._run_rows.items():
            states = dict(self.tracker.states[path])
            for stage, state in states.items():
                if state == DONE and row.states.get(stage) != DONE:
                    fresh = True
                    self._look_at_disk(row)
            row.states = states
        self.model.refresh()
        if fresh:
            self._rows_changed()

    def _update_status(self) -> None:
        tracker = self.tracker
        if tracker is None:
            return
        self.progress.setValue(int(round(1000 * tracker.fraction())))
        if self._cancelled:
            return
        phase = tracker.live["phase"]
        elapsed = format_duration(time.time() - self._started)
        if phase not in tracker.stages:
            self.status_label.setText(f"Starting…  ·  {elapsed}")
            return
        through, _failed, total = tracker.counts(phase)
        if total > 1:
            where = f"{min(through + 1, total)} of {total}"
        else:
            where = f"{tracker.live[phase]['percent']} %"
        self.status_label.setText(f"{PHASE_TITLES[phase]}  ·  {where}  ·  {elapsed}")

    def _process_error(self, error) -> None:
        if (error == QtCore.QProcess.ProcessError.FailedToStart
                and self.process is not None):
            self.log.appendPlainText("T1Prep could not be started.")
            self._process_finished(-1, QtCore.QProcess.ExitStatus.CrashExit)

    def _process_finished(self, code: int, _status=None) -> None:
        process, tracker = self.process, self.tracker
        if process is None or tracker is None:
            return
        self._read_output()
        if self._buffer:
            self.log.appendPlainText(self._buffer.decode("utf-8", "replace"))
            self._buffer = b""
        self._timer.stop()
        tracker.finish(cancelled=self._cancelled)
        self.process = None
        process.deleteLater()
        self._copy_states()
        for row in self._run_rows.values():
            self._look_at_disk(row)
        self.model.refresh()

        total = len(self._run_rows)
        failures = tracker.failures()
        took = format_duration(time.time() - self._started)
        if self._cancelled:
            message = "Stopped."
        elif failures:
            message = (f"{failures} of {total} image{'s' if total != 1 else ''} "
                       f"failed after {took}.")
        elif code != 0:
            message = f"T1Prep ended with an error (exit status {code})."
        else:
            message = (f"Finished {total} image{'s' if total != 1 else ''} "
                       f"in {took}.")
        if failures or (code != 0 and not self._cancelled):
            # What went wrong in a job of parallelize is only in its log
            for path in find_parallelize_logs(str(self._run_dir / "logs")):
                try:
                    self.log.appendPlainText(
                        f"\n--- {path.name} ---\n"
                        + path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
            self.show_log(True)
        shutil.rmtree(self._run_dir, ignore_errors=True)
        self._run_dir = None
        self.tracker = None
        self._run_rows = {}

        self.progress.hide()
        self.options.set_locked(False)
        self.table.set_spinning(False)
        self._say(message)
        self.previews.set_workers(self._render_processes)
        self._rows_changed()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _show_side(self, key: str) -> None:
        self.side_switch.set_current(key)
        self.side.setCurrentIndex(0 if key == "options" else 1)

    def _current_row(self) -> Optional[Row]:
        rows = self._selected_rows()
        return rows[0] if rows else None

    @staticmethod
    def _views(subject: Subject) -> Dict[str, Optional[List[str]]]:
        """The viewer call per picture; None where its files are missing."""
        return {picture: res.view(subject, picture) for picture in PICTURES}

    def _selection_changed(self, *_args) -> None:
        row = self._current_row()
        if row is None:
            self._shown_row = None
            self.details.show_row(None)
        else:
            views = self._views(row.found())
            self.details.show_row(
                row, row.found().report() if row.on_disk.get(VOLUME) else None,
                {picture: command is not None for picture, command in views.items()})
            if row is not self._shown_row and row.has_results():
                self._show_side("results")
            self._shown_row = row
        self._update_preview()
        self._prepare_pictures()
        self._update_actions()

    def _picture_changed(self, _picture: str) -> None:
        row = self._current_row()
        if row is not None:
            self.details.open_button.setEnabled(
                self._views(row.found()).get(self.details.picture()) is not None)
        self._update_preview()
        self._prepare_pictures()

    def _update_preview(self) -> None:
        row = self._current_row()
        self._wanted_preview = None
        if row is None:
            return
        picture = self.details.picture()
        key = self.previews.key(row.found(), picture)
        if key is None:
            running = RUNNING in row.states.values()
            self.details.preview.show_picture(
                None, "Running…" if running else "No results yet")
            return
        self._wanted_preview = key
        path = self.previews.path(key)
        if path.is_file():
            self._preview_ready(key, str(path))
        else:
            self.details.preview.show_picture(None, "Rendering…")
            self.previews.request(row.found(), picture, key)

    def _preview_ready(self, key: str, path: str) -> None:
        if key != self._wanted_preview:
            return
        image = QtGui.QImage(path) if path else QtGui.QImage()
        if image.isNull():
            self.details.preview.show_picture(None, "No picture available")
            return
        self.details.preview.show_picture(
            QtGui.QPixmap.fromImage(_black_surround(image)))

    def _prepare_pictures(self) -> None:
        """Have the pictures rendered that are likely to be looked at next.

        Those of the images around the selected one, nearest first and the
        kind of picture that is shown before the other two: stepping through
        the table then finds them ready.  The whole table only on request --
        a picture takes a second or more to render and a megabyte to keep,
        which a study of a thousand images should not cost unasked.
        """
        rows = self.model.rows
        current = self._current_row()
        centre = next((i for i, row in enumerate(rows) if row is current), 0)
        order = sorted(range(len(rows)), key=lambda i: (abs(i - centre), i < centre))
        if not self._prepare_everything:
            order = order[:self.PREPARE_AROUND]
        shown = self.details.picture()
        jobs = []
        for picture in (shown,) + tuple(p for p in PICTURES if p != shown):
            for index in order:
                subject = rows[index].found()
                if not rows[index].has_results():
                    continue
                key = self.previews.key(subject, picture)
                if key is not None and not self.previews.path(key).is_file():
                    jobs.append((subject, picture, key))
        self.previews.prepare(jobs)

    def prepare_all_pictures(self) -> None:
        """Render the pictures of every image of the table, not only of those
        around the selected one."""
        self._prepare_everything = True
        self._prepare_pictures()
        waiting = self.previews.waiting()
        self._say(f"Preparing {waiting} picture{'s' if waiting != 1 else ''}…"
                  if waiting else "The pictures of all images are there.")

    def _render_again(self, rows: Optional[Sequence[Row]] = None) -> None:
        """Draw the pictures of *rows* (the selected image) anew."""
        if rows is None:
            current = self._current_row()
            rows = [current] if current is not None else []
        keys = [self.previews.key(row.found(), picture)
                for row in rows for picture in PICTURES]
        self.previews.forget([key for key in keys if key is not None])
        self._update_preview()
        self._prepare_pictures()

    def _open_view(self, picture: Optional[str] = None,
                   row: Optional[Row] = None) -> None:
        """Open the viewer on *picture* (the one shown) of *row* (the selected)."""
        row = row or self._current_row()
        if row is None:
            return
        command = self._views(row.found()).get(picture or self.details.picture())
        if command is not None:
            self.viewers.open(command)

    def _open_best_view(self, index=None) -> None:
        """A double click: the surfaces if there are any, else the label."""
        row = self._current_row()
        if row is None:
            return
        views = self._views(row.found())
        for picture in (SURFACE, VOLUME):
            if views[picture] is not None:
                self.viewers.open(views[picture])
                return

    def _viewer_failed(self, message: str) -> None:
        self.log.appendPlainText(message)
        self._say(message)

    def _show_subject_log(self) -> None:
        row = self._current_row()
        path = row.found().log_file() if row is not None else None
        if path is None:
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self._say(f"The log could not be read: {exc}")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(path.name)
        layout = QtWidgets.QVBoxLayout(dialog)
        view = QtWidgets.QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setFont(self.log.font())
        view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(view)
        dialog.resize(760, 520)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    def _open_folder(self) -> None:
        row = self._current_row()
        if row is not None and row.found().folder.is_dir():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(row.found().folder)))

    def _context_menu(self, position) -> None:
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        row = self.model.rows[index.row()]
        views = self._views(row.found())
        menu = QtWidgets.QMenu(self)
        for picture, text, icon, _tip in DetailsPanel.PICTURES:
            item = menu.addAction(icons.icon(icon, self.colors.text, self.colors.faint),
                                  f"Show {text}")
            item.setEnabled(views[picture] is not None)
            item.triggered.connect(
                lambda _c=False, p=picture, r=row: self._open_view(p, r))
        item = menu.addAction("Render Pictures Again")
        item.setEnabled(row.has_results())
        item.triggered.connect(
            lambda _c=False, r=row: self._render_again(self._selected_rows() or [r]))
        menu.addSeparator()
        menu.addAction(self._run_selected_action)
        item = menu.addAction("Open Folder")
        item.setEnabled(row.found().folder.is_dir())
        item.triggered.connect(self._open_folder)
        menu.addSeparator()
        menu.addAction(self._remove_action)
        menu.exec(self.table.viewport().mapToGlobal(position))

    # ------------------------------------------------------------------
    # Messages, log, closing
    # ------------------------------------------------------------------

    def _say(self, message: str) -> None:
        self.status_label.setText(message)

    def show_log(self, visible: bool) -> None:
        self.log.setVisible(visible)
        self.log_link.setText("Hide log" if visible else "Show log")
        if visible:
            bar = self.log.verticalScrollBar()
            bar.setValue(bar.maximum())

    def closeEvent(self, event):           # noqa: N802
        if self.is_running():
            answer = QtWidgets.QMessageBox.question(
                self, "Quit T1Prep",
                "A run is in progress. Stop it and quit?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._terminate()
            if self.process is not None:
                self.process.waitForFinished(5000)
        self.previews.shutdown()
        self._save()
        event.accept()


def _build_parser() -> ArgumentParser:
    """Return the command-line parser for ``t1prep-gui``."""
    parser = ArgumentParser(
        prog="t1prep-gui",
        description=(
            "Open the T1Prep launcher: choose images and options, run the "
            "pipeline and look at the results.\n\n"
            "Images given on the command line are added to the table; a "
            "folder that holds T1Prep results is opened to browse them."),
        epilog=(
            "examples:\n"
            "  t1prep-gui\n"
            "  t1prep-gui sub-01_T1w.nii.gz sub-02_T1w.nii.gz\n"
            "  t1prep-gui --out-dir /data/results /data/study/*.nii\n"
            "  t1prep-gui /data/results"),
    )
    parser.add_argument(
        "inputs", nargs="*", metavar="FILE|FOLDER",
        help="images to add, or a folder with T1Prep results to browse")
    parser.add_argument(
        "--out-dir", metavar="DIR",
        help="write the results to DIR instead of next to each image")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Console-script entry point for ``t1prep-gui``.

    Called without an argument it opens the window — that is what the tool is
    for — rather than printing the synopsis the file-consuming tools show.
    """
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    QtCore.QCoreApplication.setOrganizationName("T1Prep")
    QtCore.QCoreApplication.setApplicationName("T1Prep")
    app = qt_application()
    app.setApplicationDisplayName("T1Prep")
    app.setApplicationVersion(version())
    window = Launcher()         # gives the application its look
    if args.out_dir:
        folder = Path(os.path.abspath(args.out_dir))
        folder.mkdir(parents=True, exist_ok=True)
        window.out_dir = folder
        window._fill_output()
        window._move_subjects()
    if args.inputs:
        window.add_paths(args.inputs)
    window.show()
    # Images dropped on the application icon or opened from Finder
    finder_open_files(app).set_handler(window.add_paths)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
