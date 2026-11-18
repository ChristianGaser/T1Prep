"""The launcher's icons, drawn in code.

Small line drawings on a 16-unit grid, painted with ``QPainter`` in the colour
they are asked for.  That keeps them out of the package data (no image files,
no SVG module to import), sharp at every pixel ratio, and in the colours of
the window: each takes the colour of the text it stands next to.

Qt Gui only; nothing here needs a display beyond what ``QPixmap`` needs.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt

#: Edge of the grid the drawings are made on
GRID = 16.0
#: Width of their lines, in grid units
STROKE = 1.35


def _path(*commands) -> QtGui.QPainterPath:
    """A path from ``("M", x, y)``, ``("L", x, y)``, ``("C", six numbers)``,
    ``("Q", four numbers)`` and ``("Z",)`` commands, as in an SVG path."""
    path = QtGui.QPainterPath()
    for command in commands:
        kind, numbers = command[0], command[1:]
        if kind == "M":
            path.moveTo(*numbers)
        elif kind == "L":
            path.lineTo(*numbers)
        elif kind == "C":
            path.cubicTo(*numbers)
        elif kind == "Q":
            path.quadTo(*numbers)
        elif kind == "Z":
            path.closeSubpath()
    return path


def _lines(painter: QtGui.QPainter, *segments) -> None:
    for x1, y1, x2, y2 in segments:
        painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))


def _folder_outline() -> QtGui.QPainterPath:
    return _path(("M", 1.9, 4.6), ("Q", 1.9, 3.4, 3.1, 3.4), ("L", 5.9, 3.4),
                 ("L", 7.5, 5.0), ("L", 12.9, 5.0), ("Q", 14.1, 5.0, 14.1, 6.2),
                 ("L", 14.1, 11.6), ("Q", 14.1, 12.8, 12.9, 12.8), ("L", 3.1, 12.8),
                 ("Q", 1.9, 12.8, 1.9, 11.6), ("Z",))


def _page_outline() -> QtGui.QPainterPath:
    return _path(("M", 4.6, 1.9), ("L", 9.3, 1.9), ("L", 12.6, 5.2), ("L", 12.6, 13.0),
                 ("Q", 12.6, 14.1, 11.5, 14.1), ("L", 4.6, 14.1),
                 ("Q", 3.5, 14.1, 3.5, 13.0), ("L", 3.5, 3.0), ("Q", 3.5, 1.9, 4.6, 1.9),
                 ("Z",))


# ---- the drawings: each gets a painter with pen and grid set up --------------

def _add_file(p: QtGui.QPainter) -> None:
    p.drawPath(_page_outline())
    p.drawPath(_path(("M", 9.3, 1.9), ("L", 9.3, 5.2), ("L", 12.6, 5.2)))
    _lines(p, (8.0, 8.0, 8.0, 11.6), (6.2, 9.8, 9.8, 9.8))


def _add_folder(p: QtGui.QPainter) -> None:
    p.drawPath(_folder_outline())
    _lines(p, (8.0, 7.2, 8.0, 10.8), (6.2, 9.0, 9.8, 9.0))


def _folder(p: QtGui.QPainter) -> None:
    p.drawPath(_folder_outline())


def _results(p: QtGui.QPainter) -> None:
    """Bars over an axis: what a run measured."""
    p.drawPath(_path(("M", 2.4, 2.6), ("L", 2.4, 13.4), ("L", 13.8, 13.4)))
    _lines(p, (5.5, 10.9, 5.5, 8.2), (8.5, 10.9, 8.5, 4.6), (11.5, 10.9, 11.5, 6.6))


def _play(p: QtGui.QPainter) -> None:
    p.setBrush(p.pen().color())
    p.drawPath(_path(("M", 5.2, 3.4), ("L", 12.4, 8.0), ("L", 5.2, 12.6), ("Z",)))


def _stop(p: QtGui.QPainter) -> None:
    p.setBrush(p.pen().color())
    p.drawRoundedRect(QtCore.QRectF(4.4, 4.4, 7.2, 7.2), 1.2, 1.2)


def _segmentation(p: QtGui.QPainter) -> None:
    """A slice of the image with a region outlined in it."""
    p.drawRoundedRect(QtCore.QRectF(2.0, 2.0, 12.0, 12.0), 2.2, 2.2)
    p.drawPath(_path(
        ("M", 5.0, 8.4), ("C", 4.8, 6.0, 6.5, 4.7, 8.4, 4.9),
        ("C", 10.5, 5.1, 11.5, 6.6, 11.1, 8.2), ("C", 10.8, 9.4, 9.7, 9.5, 9.5, 10.5),
        ("C", 9.3, 11.5, 7.8, 11.7, 6.7, 11.1), ("C", 5.7, 10.5, 5.1, 9.6, 5.0, 8.4),
        ("Z",)))


def _surface(p: QtGui.QPainter) -> None:
    """A hemisphere from the side, with two sulci."""
    p.drawPath(_path(
        ("M", 2.2, 8.6), ("C", 2.2, 5.1, 4.9, 2.9, 8.2, 2.9),
        ("C", 11.7, 2.9, 13.9, 5.0, 13.9, 7.7), ("C", 13.9, 9.7, 12.6, 10.7, 11.0, 10.8),
        ("C", 10.6, 12.1, 9.4, 12.8, 8.1, 12.5), ("C", 7.5, 11.5, 6.6, 11.3, 5.6, 11.5),
        ("C", 3.6, 11.8, 2.2, 10.5, 2.2, 8.6), ("Z",)))
    p.drawPath(_path(("M", 5.2, 6.1), ("C", 6.5, 6.5, 6.9, 7.8, 6.3, 9.0)))
    p.drawPath(_path(("M", 8.8, 4.9), ("C", 10.0, 5.6, 10.4, 6.9, 9.8, 8.1)))


def _thickness(p: QtGui.QPainter) -> None:
    """Two surfaces and the distance between them."""
    p.drawPath(_path(("M", 2.3, 12.6), ("C", 2.3, 6.4, 6.2, 2.6, 13.7, 2.6)))
    p.drawPath(_path(("M", 7.4, 13.6), ("C", 7.4, 10.4, 9.4, 8.2, 13.7, 8.2)))
    # the measure, with a stop at each surface
    _lines(p, (5.3, 5.9, 8.6, 9.4), (4.4, 6.8, 6.2, 5.0), (7.7, 10.3, 9.5, 8.5))


def _open(p: QtGui.QPainter) -> None:
    """Out of the box: into a window of its own."""
    p.drawPath(_path(("M", 6.6, 3.6), ("L", 4.0, 3.6), ("Q", 2.8, 3.6, 2.8, 4.8),
                     ("L", 2.8, 12.0), ("Q", 2.8, 13.2, 4.0, 13.2), ("L", 11.2, 13.2),
                     ("Q", 12.4, 13.2, 12.4, 12.0), ("L", 12.4, 9.4)))
    p.drawPath(_path(("M", 9.4, 2.8), ("L", 13.2, 2.8), ("L", 13.2, 6.6)))
    _lines(p, (13.2, 2.8, 7.6, 8.4))


def _log(p: QtGui.QPainter) -> None:
    _lines(p, (3.0, 4.2, 13.0, 4.2), (3.0, 8.0, 13.0, 8.0), (3.0, 11.8, 9.2, 11.8))


def _refresh(p: QtGui.QPainter) -> None:
    """Most of a circle, clockwise, into an arrow head."""
    box = QtCore.QRectF(2.6, 2.6, 10.8, 10.8)
    path = QtGui.QPainterPath()
    path.arcMoveTo(box, -19.5)
    path.arcTo(box, -19.5, -295.5)
    path.lineTo(14.6, 6.8)
    p.drawPath(path)
    p.drawPath(_path(("M", 14.6, 3.2), ("L", 14.6, 6.8), ("L", 11.0, 6.8)))


_DRAW: Dict[str, Callable[[QtGui.QPainter], None]] = {
    "add_file": _add_file,
    "add_folder": _add_folder,
    "folder": _folder,
    "results": _results,
    "play": _play,
    "stop": _stop,
    "segmentation": _segmentation,
    "surface": _surface,
    "thickness": _thickness,
    "open": _open,
    "log": _log,
    "refresh": _refresh,
}

#: The icons there are
NAMES: Tuple[str, ...] = tuple(sorted(_DRAW))


def pixmap(name: str, color: str, size: int = 16, ratio: float = 1.0) -> QtGui.QPixmap:
    """The icon *name* as a pixmap of *size* logical pixels.

    Args:
        name: One of :data:`NAMES`.
        color: ``#rrggbb`` of its lines.
        size: Edge length in logical pixels.
        ratio: Device pixels per logical pixel.
    """
    draw = _DRAW[name]
    edge = max(1, int(round(size * ratio)))
    result = QtGui.QPixmap(edge, edge)
    result.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(result)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.scale(edge / GRID, edge / GRID)
    pen = QtGui.QPen(QtGui.QColor(color), STROKE)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    draw(painter)
    painter.end()
    result.setDevicePixelRatio(ratio)
    return result


def icon(name: str, color: str, disabled: Optional[str] = None,
         checked: Optional[str] = None, size: int = 16) -> QtGui.QIcon:
    """The icon *name* for a button.

    Args:
        name: One of :data:`NAMES`.
        color: Colour of its lines.
        disabled: Colour while the button is disabled; *color* otherwise.
        checked: Colour while a checkable button is on; *color* otherwise.
        size: Edge length in logical pixels.
    """
    result = QtGui.QIcon()
    mode, state = QtGui.QIcon.Mode, QtGui.QIcon.State
    for ratio in (1.0, 2.0, 3.0):
        for which, off, on in ((mode.Normal, color, checked or color),
                               (mode.Active, color, checked or color),
                               (mode.Selected, color, checked or color),
                               (mode.Disabled, disabled or color, disabled or color)):
            result.addPixmap(pixmap(name, off, size, ratio), which, state.Off)
            result.addPixmap(pixmap(name, on, size, ratio), which, state.On)
    return result
