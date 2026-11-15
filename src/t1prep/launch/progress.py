"""Follow a running T1Prep: its phases, its bars and the state of each image.

T1Prep reports in two ways.  On standard output it prints a header before
each phase (``Volume Segmentation``, ``Surface Estimation``), a bar line per
step and a notice when a step failed.  For several images it hands the work to
``parallelize``, which prints one bar for all of them and keeps the list of
each job, what it has finished and what failed in ``PROGRESS_DIR``.

:func:`follow_line` turns the printed lines into the phase and percentage a
front end shows.
:class:`RunTracker` adds what a table of images needs: which image is queued,
running, done or failed in which phase.

Standard output and ``PROGRESS_DIR`` say what is *running*.  Whether a step
that ended left its results behind is asked of the files (the ``finished``
callable of :class:`RunTracker`): T1Prep prints no failure notice in every
mode, and a result that is not there has failed whatever was printed.
"""

from __future__ import annotations

import glob
import math
import os
import re
import socket
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# A bar line of progress_bar_multi.sh ("|███     |  42% label ETA ...") or of
# the Python renderer ("[███     ]  42% label")
BAR_LINE = re.compile(r"^[|\[][^|\]]*[|\]]\s*(\d{1,3})%")

# The same line taken apart: percentage and label, without the bar and the ETA
_BAR_TEXT = re.compile(
    r"^[|\[][^|\]]*[|\]]\s*(\d{1,3})%\s*(.*?)\s*(?:ETA\s+[\d:]+)?\s*$")

# The headers T1Prep prints before each phase, in the order the phases run
PHASES = (("Volume Segmentation", "volume"), ("Surface Estimation", "surface"))
PHASE_ORDER = tuple(name for _, name in PHASES)

#: What a phase is called in a sentence
PHASE_TITLES = {"volume": "Volume segmentation", "surface": "Surface estimation"}

# States of one image in one phase
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
#: Not attempted: the run was stopped, or the phase before it failed
SKIPPED = "skipped"

# "Processing /data/sub-01.nii" or "2/5 Processing /data/sub-02.nii": the
# image T1Prep's own loop has turned to
_PROCESSING = re.compile(r"^(?:\d+/\d+\s+)?Processing\s+(.+?)\s*$")
# "Error in /data/sub-01.nii" (segmentation) or "Error in sub-01" (surfaces)
_ERROR_IN = re.compile(r"^Error in\s+(.+?)\s*$")


# ---------------------------------------------------------------------------
# Phases and bars
# ---------------------------------------------------------------------------

def new_progress() -> dict:
    """Return the live progress of a job that has printed nothing yet."""
    return {
        "phase": None,
        "since": None,
        **{
            name: {
                "percent": 0,
                "started": False,
                "seconds": 0.0,
                "entries": 0,
                "failed": False,
            }
            for name in PHASE_ORDER
        },
    }


def stop_clock(live: dict, now: float) -> None:
    """Add the time since the running phase started to that phase."""
    if live["phase"] is not None and live["since"] is not None:
        live[live["phase"]]["seconds"] += now - live["since"]
    live["since"] = None


def enter_phase(live: dict, phase: str, now: float) -> None:
    """Start ``phase`` for one more subject; its bar counts from zero."""
    stop_clock(live, now)
    live["phase"], live["since"] = phase, now
    live[phase]["started"] = True
    live[phase]["entries"] += 1
    live[phase]["percent"] = 0


def failed_phase(stripped: str) -> Optional[str]:
    """Name the phase whose failure notice ``stripped`` is, if it is one."""
    if "Segmentation failed" in stripped:
        return "volume"
    if "Surface estimation failed" in stripped:
        return "surface"
    return None


def follow_line(live: dict, line: str, now: float) -> Optional[Tuple[str, object]]:
    """Update *live* from one line T1Prep printed.

    Args:
        live: The progress :func:`new_progress` started.
        line: The line, with or without its line end.
        now: The time the line was read, in seconds.

    Returns:
        What the line was: ``("phase", name)`` for a phase header,
        ``("failed", name)`` for a failure notice — kept in the phase where it
        happened, not read as progress —, ``("bar", percent)`` for a bar line,
        and None for anything else.
    """
    stripped = line.strip()
    failure = failed_phase(stripped)
    if failure:
        live[failure]["failed"] = True
        return ("failed", failure)
    phase = next((name for header, name in PHASES if header in stripped), None)
    if phase:
        enter_phase(live, phase, now)
        return ("phase", phase)
    match = BAR_LINE.match(stripped)
    if match:
        percent = min(100, int(match.group(1)))
        if live["phase"] is None:
            enter_phase(live, PHASE_ORDER[0], now)
        live[live["phase"]]["percent"] = percent
        return ("bar", percent)
    return None


def step_text(line: str) -> Optional[str]:
    """A bar line as a log would say it: ``" 25%  Affine registration"``.

    The bar itself is drawn for a terminal that redraws it in place; in a
    log that keeps every line it is noise.  None for a line that is no bar.
    """
    match = _BAR_TEXT.match(line.strip())
    if match is None:
        return None
    return f"{int(match.group(1)):>4}%  {match.group(2)}".rstrip()


def format_duration(seconds: float) -> str:
    """Spell a duration as T1Prep's "Finished after" does: ``1hrs 2min 3s``."""
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours}hrs " if hours else ""
    if minutes:
        text += f"{minutes}min "
    return f"{text}{secs}s"


def find_parallelize_logs(log_dir: Optional[str]) -> List[Path]:
    """Return the parallelize logs written into a job's own log directory.

    The job hands ``log_dir`` to T1Prep (``T1PREP_LOGDIR``), so no other
    job's log can be found here.  Oldest first: the order the phases ran.
    """
    if not log_dir:
        return []
    hostname = socket.gethostname()
    pattern = os.path.join(glob.escape(str(log_dir)), f"parallelize_{hostname}_*.log")
    return [Path(path) for path in sorted(glob.glob(pattern), key=os.path.getmtime)]


# ---------------------------------------------------------------------------
# The jobs of parallelize
# ---------------------------------------------------------------------------

def read_progress_dir(directory: Path) -> Dict[str, str]:
    """The state of every image ``parallelize`` has distributed.

    Each of its jobs works through ``job<N>.list`` from the top, writes the
    number it has finished to ``job<N>.progress`` (``3/7``) and names what
    failed in ``job<N>.failed``.

    Returns:
        A state per listed image: :data:`DONE` or :data:`FAILED` for what the
        job is through, :data:`RUNNING` for the one it is at, :data:`QUEUED`
        for the rest.  Empty when there is nothing to read (yet).
    """
    states: Dict[str, str] = {}
    try:
        lists = sorted(Path(directory).glob("job*.list"))
    except OSError:
        return states
    for list_file in lists:
        try:
            items = [item for item in
                     list_file.read_text(encoding="utf-8").splitlines() if item]
        except OSError:
            continue
        stem = list_file.with_suffix("")
        done: Optional[int] = None
        try:
            done = int(stem.with_suffix(".progress")
                       .read_text(encoding="utf-8").split("/", 1)[0])
        except (OSError, ValueError):
            pass                       # the job has not started
        try:
            failed = set(stem.with_suffix(".failed")
                         .read_text(encoding="utf-8").splitlines())
        except OSError:
            failed = set()
        for index, item in enumerate(items):
            if item in failed:
                states[item] = FAILED
            elif done is None or index > done:
                states[item] = QUEUED
            elif index == done:
                states[item] = RUNNING
            else:
                states[item] = DONE
    return states


# ---------------------------------------------------------------------------
# One run, image by image
# ---------------------------------------------------------------------------

def _stem(path: str) -> str:
    """The name T1Prep derives from an image: no folder, no ``.nii(.gz)``."""
    name = os.path.basename(path)
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class RunTracker:
    """The state of every image of one T1Prep run, phase by phase.

    Feed it what T1Prep prints (:meth:`feed`), let it look into
    ``PROGRESS_DIR`` now and then (:meth:`poll`) and tell it when the process
    has ended (:meth:`finish`); :attr:`states` then holds, per image and
    phase, one of :data:`QUEUED`, :data:`RUNNING`, :data:`DONE`,
    :data:`FAILED` and :data:`SKIPPED`.

    T1Prep runs in one of three ways, and each says something else:

    * one image, in its own loop: a header and bars per phase;
    * several images, one job: ``parallelize`` runs them in the order given
      and prints one bar that counts the finished ones;
    * several images, several jobs: the lists in ``PROGRESS_DIR``.

    Args:
        inputs: The images as they are given to T1Prep, in that order.
        stages: The phases the run goes through, of :data:`PHASE_ORDER`.
        finished: Called with an image and a phase; True when the results of
            that phase are on disk.  It settles every step that has ended
            without an explicit failure notice.
        progress_dir: The ``PROGRESS_DIR`` the job runs with.  It must not
            exist beforehand: its presence is how the third way is told from
            the second.
    """

    def __init__(self, inputs: Sequence[str], stages: Iterable[str],
                 finished: Callable[[str, str], bool],
                 progress_dir: Optional[os.PathLike] = None):
        wanted = set(stages)
        self.inputs: List[str] = [str(path) for path in inputs]
        self.stages: Tuple[str, ...] = tuple(
            stage for stage in PHASE_ORDER if stage in wanted)
        self.finished = finished
        self.progress_dir = Path(progress_dir) if progress_dir else None
        self.live = new_progress()
        self.states: Dict[str, Dict[str, str]] = {
            path: {stage: QUEUED for stage in self.stages} for path in self.inputs
        }
        self._current: Optional[str] = None     # the image of T1Prep's own loop
        self._listing_failures = False
        # T1Prep names an image by its path, its file name or its stem
        self._names: Dict[str, Optional[str]] = {}
        for path in self.inputs:
            for name in {os.path.basename(path), _stem(path)}:
                # Two images of the same name in different folders: the name
                # alone says nothing
                self._names[name] = None if name in self._names else path
        for path in self.inputs:
            self._names[path] = path

    # -- what T1Prep prints ------------------------------------------------

    def feed(self, line: str, now: Optional[float] = None) -> Optional[Tuple[str, object]]:
        """Take one printed line; return what :func:`follow_line` made of it."""
        now = time.time() if now is None else now
        stripped = line.strip()
        event = follow_line(self.live, stripped, now)
        if event is not None:
            self._listing_failures = False
            kind, value = event
            if kind == "phase":
                self._enter(str(value))
            elif kind == "failed":
                self._fail(self._current, str(value))
            elif kind == "bar":
                self._count(int(value))
            return event

        match = _PROCESSING.match(stripped)
        if match:
            self._settle_current()
            self._current = self._names.get(match.group(1))
            return None
        match = _ERROR_IN.match(stripped)
        if match:
            self._fail(self._names.get(match.group(1)), self.live["phase"])
            return None
        if stripped.startswith("Failed items:"):
            self._listing_failures = True      # parallelize names them below
            return None
        if self._listing_failures:
            path = self._names.get(stripped)
            if path is None:
                self._listing_failures = False
            else:
                self._fail(path, self.live["phase"])
            return None
        if stripped.startswith("Finished after"):
            self._settle_current()
        return None

    def _enter(self, phase: str) -> None:
        """A phase header: the phases before it have ended."""
        if phase not in self.stages:
            return
        before = self.stages[: self.stages.index(phase)]
        if self._current is not None:
            for stage in before:
                self._settle(self._current, stage)
            if self.states[self._current][phase] == QUEUED:
                self.states[self._current][phase] = RUNNING
            return
        for path in self.inputs:           # parallelize: the header comes once
            for stage in before:
                self._settle(path, stage)

    def _count(self, percent: int) -> None:
        """One job for several images: the bar counts the finished ones."""
        phase = self.live["phase"]
        if (self._current is not None or len(self.inputs) < 2
                or phase not in self.stages or self._has_job_lists()):
            return
        total = len(self.inputs)
        through = min(total, max(0, math.ceil(percent * total / 100 - 1e-9)))
        for index, path in enumerate(self.inputs):
            if index < through:
                self._settle(path, phase)
            elif index == through and self.states[path][phase] == QUEUED:
                self.states[path][phase] = RUNNING

    def _has_job_lists(self) -> bool:
        return self.progress_dir is not None and self.progress_dir.is_dir()

    # -- what parallelize keeps --------------------------------------------

    def poll(self) -> None:
        """Read ``PROGRESS_DIR``; call this while the job runs."""
        phase = self.live["phase"]
        if phase not in self.stages or not self._has_job_lists():
            return
        for item, state in read_progress_dir(self.progress_dir).items():
            path = self._names.get(item)
            if path is None:
                continue
            if self.states[path][phase] in (DONE, FAILED):
                continue                   # settled; the lists are gone later
            self.states[path][phase] = state

    # -- settling ----------------------------------------------------------

    def _settle(self, path: str, stage: str) -> None:
        """A step that has ended: done if its results are there."""
        if self.states[path].get(stage) in (QUEUED, RUNNING):
            self.states[path][stage] = (
                DONE if self.finished(path, stage) else FAILED)
            self._skip_after_failure(path)

    def _settle_current(self) -> None:
        if self._current is None:
            return
        for stage in self.stages:
            if self.states[self._current][stage] == RUNNING:
                self._settle(self._current, stage)

    def _fail(self, path: Optional[str], stage: Optional[str]) -> None:
        if path is None or stage not in self.stages:
            return
        self.states[path][stage] = FAILED
        self._skip_after_failure(path)

    def _skip_after_failure(self, path: str) -> None:
        """T1Prep's own loop gives up on an image whose segmentation failed."""
        if path != self._current:
            return
        failed = False
        for stage in self.stages:
            if failed and self.states[path][stage] == QUEUED:
                self.states[path][stage] = SKIPPED
            failed = failed or self.states[path][stage] == FAILED

    def finish(self, cancelled: bool = False, now: Optional[float] = None) -> None:
        """The process has ended: settle what is still open.

        Args:
            cancelled: The run was stopped.  What it had not finished is then
                :data:`SKIPPED`, not failed.
        """
        stop_clock(self.live, time.time() if now is None else now)
        for path in self.inputs:
            failed = False
            for stage in self.stages:
                state = self.states[path][stage]
                if state in (QUEUED, RUNNING):
                    if cancelled or failed:
                        self.states[path][stage] = SKIPPED
                    else:
                        self.states[path][stage] = (
                            DONE if self.finished(path, stage) else FAILED)
                failed = failed or self.states[path][stage] == FAILED

    # -- summaries ---------------------------------------------------------

    def counts(self, stage: str) -> Tuple[int, int, int]:
        """``(finished, failed, total)`` images of *stage*."""
        states = [per.get(stage) for per in self.states.values()]
        return (sum(state in (DONE, FAILED, SKIPPED) for state in states),
                sum(state == FAILED for state in states), len(states))

    def failures(self) -> int:
        """Images with a failed phase."""
        return sum(FAILED in per.values() for per in self.states.values())

    def fraction(self) -> float:
        """How far the whole run is, from 0 to 1."""
        total = len(self.inputs) * len(self.stages)
        if not total:
            return 0.0
        through = sum(state in (DONE, FAILED, SKIPPED)
                      for per in self.states.values() for state in per.values())
        phase = self.live["phase"]
        if (self._current is not None and phase in self.stages
                and self.states[self._current][phase] == RUNNING):
            through += self.live[phase]["percent"] / 100
        return min(1.0, through / total)
