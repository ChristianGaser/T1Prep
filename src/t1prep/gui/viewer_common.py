"""What CAT_SurfView and CAT_VolView both need.

The two viewers grew the same handful of behaviours independently — taking
events away from the interactor, deciding which dropped files they can open,
listing their keys, saving a PNG — and the same bug then had to be fixed twice.
Those pieces live here instead.

The module deliberately imports neither viewer (cat_surf_view imports
cat_vol_view, so anything shared has to sit below both) and knows nothing about
their data: it is given the interactor, the file name or the callable to use.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Iterable, List, Optional, Sequence

from PySide6 import QtWidgets

#: Whether the viewers report what they fall back on.  Off by default: a
#: reader that fails and is replaced by the next one is normal, not an error.
#: The volume viewer sets it from --verbose, and T1PREP_VERBOSE=1 sets it for
#: both.
_verbose = bool(os.environ.get("T1PREP_VERBOSE"))


def set_verbose(verbose: bool) -> None:
    """Report fallbacks and the reasons they were taken."""
    global _verbose
    _verbose = bool(verbose)


def note(message: str) -> None:
    """Say what happened when the viewer worked around something.

    Reading a surface or an overlay means trying one reader after another, and
    a silent ``except`` there is what hides a genuine failure among the
    expected ones.
    """
    if _verbose:
        print(f"[t1prep] {message}", file=sys.stderr)


#: Files holding a surface mesh or per-vertex values
SURFACE_SUFFIXES = ('.gii', '.annot', '.vtk', '.vtp', '.obj', '.stl', '.ply', '.txt')

#: Files holding a volume
VOLUME_SUFFIXES = ('.nii', '.nii.gz', '.mnc', '.mha', '.mhd', '.nrrd', '.nhdr',
                   '.img', '.hdr')


def claim_event(interactor, event: str, keep_alive: List,
                should_abort: Callable[[], bool],
                handler: Optional[Callable] = None) -> None:
    """Handle *event* before the interactor style, and optionally hide it from it.

    Overriding the style's ``On...`` methods in Python has no effect: the
    interactor dispatches to the C++ implementation, which knows nothing about
    a Python subclass.  An observer with a higher priority does get called, and
    aborting there stops the event before the style sees it — that is how the
    viewers keep VTK from zooming with the mouse, or from acting on a key they
    handle themselves.

    Args:
        interactor: The render window interactor to observe.
        event: VTK event name, e.g. ``"MouseWheelForwardEvent"``.
        keep_alive: List the callback is appended to; without a reference the
            observer dies with the local scope.
        should_abort: Called per event; True takes the event away from the
            style.
        handler: Called with the interactor before the abort decision.
    """
    tag = [None]

    def callback(obj, _event):
        if handler is not None:
            handler(obj)
        if should_abort() and tag[0] is not None:
            command = obj.GetCommand(tag[0])
            if command is not None:
                command.AbortFlagOn()

    keep_alive.append(callback)
    tag[0] = interactor.AddObserver(event, callback, 1.0)


#: The events through which VTK's interactor styles change the zoom
ZOOM_EVENTS = ("RightButtonPressEvent", "RightButtonReleaseEvent",
               "MouseWheelForwardEvent", "MouseWheelBackwardEvent",
               "StartPinchEvent", "PinchEvent")


def droppable_url(url, suffixes: Sequence[str]) -> bool:
    """True when a dragged URL is a local file the viewer can open."""
    return bool(url.isLocalFile()
                and str(url.toLocalFile()).lower().endswith(tuple(suffixes)))


def dropped_files(event, suffixes: Sequence[str]) -> List[str]:
    """The local files of a drop event that the viewer can open."""
    return [url.toLocalFile() for url in event.mimeData().urls()
            if droppable_url(url, suffixes)]


def show_shortcuts(parent, shortcuts: Iterable[Sequence[str]], footer: str = "") -> None:
    """List keys and what they do, as a dialog.

    Printing them to the terminal helps nobody once the viewer is started from
    a Finder double-click or an app bundle.

    Args:
        shortcuts: Rows of (keys, description); further columns are ignored, so
            a viewer can keep the slot it uses to name the handler.
        footer: HTML appended below the table.
    """
    rows = "".join(f"<tr><td><b>{row[0]}</b>&nbsp;&nbsp;</td><td>{row[1]}</td></tr>"
                   for row in shortcuts)
    QtWidgets.QMessageBox.information(parent, "Keyboard shortcuts",
                                      f"<table>{rows}</table>{footer}")


def ask_and_save_png(parent, default_path: str,
                     write: Callable[[str], object]) -> Optional[str]:
    """Ask where to put a PNG, write it there and return the file name.

    Args:
        default_path: What the dialog starts with.
        write: Called with the chosen path; anything it raises is reported to
            the user rather than to the terminal.

    Returns:
        The path written, or None when the dialog was cancelled or the write
        failed.
    """
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, "Save image", default_path, "PNG image (*.png)")
    if not path:
        return None
    try:
        write(path)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(parent, "Screenshot",
                                      f"Could not save the image:\n{exc}")
        return None
    return path


def qt_application():
    """The QApplication, created once.

    A second instance raises, and both viewers may need one early — to ask for
    a file when started from the macOS app bundle — and again for the window.
    """
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])


def shorten_path(path: str, max_parts: int = 3, max_chars: int = 40) -> str:
    """A directory for the title bar: its last components, "…" for the rest.

    The end is what identifies a volume (``…/sub-01/mri``), so that is what is
    kept.  It has to stay short as well: a title that does not fit the window
    is shortened by macOS itself, and that drops the end.
    """
    text = str(path).rstrip(os.sep)
    parts = [part for part in text.split(os.sep) if part]
    if not parts:
        return text
    if len(parts) <= max_parts and len(text) <= max_chars:
        return text
    for count in range(min(max_parts, len(parts)), 1, -1):
        candidate = f"…{os.sep}{os.sep.join(parts[-count:])}"
        if len(candidate) <= max_chars:
            return candidate
    tail = parts[-1]
    if len(tail) + 2 <= max_chars:
        return f"…{os.sep}{tail}"
    return "…" + tail[-(max_chars - 1):]   # one very long name: cut it as well
