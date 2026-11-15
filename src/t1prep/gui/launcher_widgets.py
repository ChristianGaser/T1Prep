"""The parts of the T1Prep launcher window.

The table of images, the options form and the results panel.  Each is built
from what ``t1prep.launch`` knows — the option table, the states of a run,
the report of a subject — and holds no pipeline knowledge of its own.

Qt Widgets only: no VTK, no OpenGL.  The pictures in the results panel are
PNG files the volume viewer renders in a process of its own.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..launch import options as opt
from ..launch.inputs import DEFAULT_PATTERN, find_images
from ..launch.progress import DONE, FAILED, QUEUED, RUNNING
from ..launch.results import SURFACE, VOLUME, Subject, describe
from .theme import Colors, blend, grade_color
from .viewer_common import shorten_path

STAGES = (VOLUME, SURFACE)
STATE_ROLE = Qt.ItemDataRole.UserRole + 1


# ---------------------------------------------------------------------------
# The table of images
# ---------------------------------------------------------------------------

@dataclass
class Row:
    """One image of the table.

    Attributes:
        subject: Where its results are.
        fixed: The results were opened from disk; the options of a run do
            not move them.
        states: Its state per phase in the current or the last run.
        on_disk: Per phase, whether the results exist.
        grade, iqr: The image quality of its report.
    """

    subject: Subject
    fixed: bool = False
    states: Dict[str, str] = field(default_factory=dict)
    on_disk: Dict[str, bool] = field(default_factory=dict)
    grade: str = ""
    iqr: Optional[float] = None

    def shown(self, stage: str) -> str:
        """The state the table shows: the run's, else what is on disk."""
        state = self.states.get(stage, "")
        if state in (QUEUED, RUNNING, DONE, FAILED):
            return state
        return DONE if self.on_disk.get(stage) else ""

    def has_results(self) -> bool:
        return any(self.on_disk.get(stage) for stage in STAGES)


class SubjectModel(QtCore.QAbstractTableModel):
    """The images, one per row: name, state of each phase, quality."""

    HEADERS = ("Image", "Volume", "Surface", "Quality")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: List[Row] = []

    # -- Qt's interface ----------------------------------------------------

    def rowCount(self, parent=QtCore.QModelIndex()):      # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):   # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return self.HEADERS[section]
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignVCenter | (
                    Qt.AlignmentFlag.AlignLeft if section == 0
                    else Qt.AlignmentFlag.AlignHCenter))
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = self.rows[index.row()], index.column()
        if column == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                return row.subject.name
            if role == Qt.ItemDataRole.ToolTipRole:
                return str(row.subject.input or row.subject.folder)
        elif column in (1, 2):
            state = row.shown(STAGES[column - 1])
            if role == STATE_ROLE:
                return state
            if role == Qt.ItemDataRole.ToolTipRole and state:
                return {QUEUED: "Waiting", RUNNING: "Running",
                        DONE: "Finished", FAILED: "Failed"}[state]
        elif column == 3:
            if role == STATE_ROLE:
                return row.grade
            if role == Qt.ItemDataRole.ToolTipRole and row.iqr is not None:
                return f"Image quality rating {row.iqr:.2f} (1 excellent, 6 poor)"
        return None

    # -- changes -----------------------------------------------------------

    def append(self, rows: Sequence[Row]) -> None:
        if not rows:
            return
        first = len(self.rows)
        self.beginInsertRows(QtCore.QModelIndex(), first, first + len(rows) - 1)
        self.rows.extend(rows)
        self.endInsertRows()

    def remove(self, indices: Sequence[int]) -> None:
        for index in sorted(set(indices), reverse=True):
            self.beginRemoveRows(QtCore.QModelIndex(), index, index)
            del self.rows[index]
            self.endRemoveRows()

    def refresh(self) -> None:
        """Redraw every row; the states and measures were changed in place."""
        if self.rows:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self.rows) - 1, len(self.HEADERS) - 1))


class StateDelegate(QtWidgets.QStyledItemDelegate):
    """Draws the state of a phase as a mark and the quality as a grade."""

    def __init__(self, colors: Colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.angle = 0          # of the running mark, advanced by the table

    def paint(self, painter, option, index):
        item = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item, index)
        item.text = ""
        style = item.widget.style() if item.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
                          item, painter, item.widget)
        value = index.data(STATE_ROLE)
        if not value:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if index.column() == 3:
            self._grade(painter, option.rect, str(value))
        else:
            self._mark(painter, option.rect, str(value))
        painter.restore()

    def _mark(self, painter, cell, state: str) -> None:
        c = self.colors
        size = 16.0
        box = QtCore.QRectF(cell.center().x() - size / 2 + 0.5,
                            cell.center().y() - size / 2 + 0.5, size, size)
        if state in (DONE, FAILED):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(c.success if state == DONE else c.danger))
            painter.drawEllipse(box)
            pen = QtGui.QPen(QtGui.QColor("#ffffff"), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            x, y, s = box.left(), box.top(), size
            if state == DONE:
                painter.drawPolyline(QtGui.QPolygonF([
                    QtCore.QPointF(x + 0.28 * s, y + 0.52 * s),
                    QtCore.QPointF(x + 0.44 * s, y + 0.68 * s),
                    QtCore.QPointF(x + 0.73 * s, y + 0.35 * s)]))
            else:
                painter.drawLine(QtCore.QPointF(x + 0.34 * s, y + 0.34 * s),
                                 QtCore.QPointF(x + 0.66 * s, y + 0.66 * s))
                painter.drawLine(QtCore.QPointF(x + 0.66 * s, y + 0.34 * s),
                                 QtCore.QPointF(x + 0.34 * s, y + 0.66 * s))
            return
        ring = box.adjusted(1.5, 1.5, -1.5, -1.5)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if state == QUEUED:
            painter.setPen(QtGui.QPen(QtGui.QColor(c.faint), 1.5))
            painter.drawEllipse(ring)
        elif state == RUNNING:
            painter.setPen(QtGui.QPen(QtGui.QColor(c.border), 2.0))
            painter.drawEllipse(ring)
            pen = QtGui.QPen(QtGui.QColor(c.accent), 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(ring, -self.angle * 16, -100 * 16)

    def _grade(self, painter, cell, grade: str) -> None:
        colour = grade_color(grade, self.colors)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        width = max(30, painter.fontMetrics().horizontalAdvance(grade) + 16)
        pill = QtCore.QRectF(cell.center().x() - width / 2, cell.center().y() - 9.5,
                             width, 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(blend(self.colors.surface, colour, 0.18)))
        painter.drawRoundedRect(pill, 10, 10)
        painter.setPen(QtGui.QColor(colour))
        painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), grade)


class SubjectTable(QtWidgets.QTableView):
    """The table of images, with a hint while it is empty."""

    remove_requested = Signal()

    def __init__(self, model: SubjectModel, colors: Colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.setModel(model)
        self.delegate = StateDelegate(colors, self)
        for column in (1, 2, 3):
            self.setItemDelegateForColumn(column, self.delegate)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(32)
        header = self.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionsClickable(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 76), (2, 76), (3, 76)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            header.resizeSection(column, width)
        # Turns the mark of a running phase; only ticks while one is shown
        self._spin = QtCore.QTimer(self)
        self._spin.setInterval(60)
        self._spin.timeout.connect(self._advance)

    def set_spinning(self, spinning: bool) -> None:
        if spinning and not self._spin.isActive():
            self._spin.start()
        elif not spinning and self._spin.isActive():
            self._spin.stop()

    def _advance(self) -> None:
        self.delegate.angle = (self.delegate.angle + 24) % 360
        self.viewport().update()

    def selected_rows(self) -> List[int]:
        return sorted(index.row() for index in self.selectionModel().selectedRows())

    def keyPressEvent(self, event):    # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_requested.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):       # noqa: N802
        super().paintEvent(event)
        if self.model().rowCount():
            return
        painter = QtGui.QPainter(self.viewport())
        area = self.viewport().rect()
        font = painter.font()
        font.setPointSizeF(font.pointSizeF() + 2)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(self.colors.muted))
        painter.drawText(area.adjusted(0, -14, 0, -14),
                         int(Qt.AlignmentFlag.AlignCenter),
                         "Drop T1 images or folders here")
        font.setPointSizeF(font.pointSizeF() - 2)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(self.colors.faint))
        painter.drawText(area.adjusted(0, 16, 0, 16),
                         int(Qt.AlignmentFlag.AlignCenter),
                         "NIfTI files, .nii or .nii.gz")
        painter.end()


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

def rule() -> QtWidgets.QFrame:
    """A hairline between two parts of a panel."""
    line = QtWidgets.QFrame()
    line.setObjectName("rule")
    line.setFixedHeight(1)
    return line


class Segments(QtWidgets.QFrame):
    """A switch between a few named views, one of which is on."""

    changed = Signal(str)

    def __init__(self, entries: Sequence[Sequence[str]], parent=None):
        super().__init__(parent)
        self.setObjectName("segments")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self._group = QtWidgets.QButtonGroup(self)
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        for key, text in entries:
            button = QtWidgets.QPushButton(text)
            button.setObjectName("segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            bold = QtGui.QFont(button.font())
            bold.setBold(True)     # the one that is on; reserve its width
            button.setMinimumWidth(
                QtGui.QFontMetrics(bold).horizontalAdvance(text) + 34)
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button)
            button.clicked.connect(lambda _checked=False, k=key: self.changed.emit(k))
        next(iter(self._buttons.values())).setChecked(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def current(self) -> str:
        return next(key for key, button in self._buttons.items() if button.isChecked())

    def set_current(self, key: str) -> None:
        self._buttons[key].setChecked(True)

    def set_available(self, key: str, available: bool) -> None:
        self._buttons[key].setEnabled(available)


class Section(QtWidgets.QWidget):
    """A titled part of the form that folds away."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.header = QtWidgets.QToolButton()
        self.header.setObjectName("section")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.toggled.connect(self._fold)
        self.body = QtWidgets.QWidget()
        self.content = QtWidgets.QVBoxLayout(self.body)
        self.content.setContentsMargins(2, 0, 0, 6)
        self.content.setSpacing(4)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self._fold(True)

    def _fold(self, open_: bool) -> None:
        self.header.setArrowType(Qt.ArrowType.DownArrow if open_
                                 else Qt.ArrowType.RightArrow)
        self.body.setVisible(open_)

    def is_open(self) -> bool:
        return self.header.isChecked()

    def set_open(self, open_: bool) -> None:
        self.header.setChecked(bool(open_))


class AtlasPicker(QtWidgets.QToolButton):
    """Chooses any number of atlases from a list that drops down.

    Args:
        names: The atlases to choose from.
        describe: Called with a name; the text it returns is the tool tip of
            that atlas in the list.
    """

    changed = Signal()

    def __init__(self, names: Sequence[str],
                 describe: Optional[Callable[[str], str]] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("picker")
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setFixedWidth(132)
        self._list = QtWidgets.QListWidget()
        self._list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            text = describe(name) if describe is not None else ""
            if text:
                # As rich text, so that Qt wraps it and takes nothing in it
                # for markup of its own
                item.setToolTip("<p style='white-space: pre-wrap'>"
                                f"{html.escape(text)}</p>")
            self._list.addItem(item)
        rows = max(1, min(len(names), 10))
        self._list.setFixedSize(
            max(220, self._list.sizeHintForColumn(0) + 36),
            rows * max(22, self._list.sizeHintForRow(0)) + 8)
        self._list.itemChanged.connect(self._item_changed)
        # A click on the name ticks the box as well
        self._list.itemClicked.connect(self._toggle)
        self._ignore = False
        menu = QtWidgets.QMenu(self)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(self._list)
        menu.addAction(action)
        self.setMenu(menu)
        self._caption()

    def _toggle(self, item) -> None:
        if self._ignore:           # the click was on the box itself
            self._ignore = False
            return
        item.setCheckState(Qt.CheckState.Unchecked
                           if item.checkState() == Qt.CheckState.Checked
                           else Qt.CheckState.Checked)
        self._ignore = False

    def _item_changed(self, _item) -> None:
        self._ignore = True
        self._caption()
        if not self.signalsBlocked():
            self.changed.emit()

    def _caption(self) -> None:
        chosen = self.value()
        if not chosen:
            text = "None"
        elif len(chosen) == 1:
            text = chosen[0]
        else:
            text = f"{len(chosen)} selected"
        self.setText(self.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideRight, 98))
        self.setToolTip("\n".join(chosen))

    def value(self) -> tuple:
        return tuple(self._list.item(i).text() for i in range(self._list.count())
                     if self._list.item(i).checkState() == Qt.CheckState.Checked)

    def set_value(self, names: Sequence[str]) -> None:
        wanted = set(names)
        blocked = self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() in wanted
                               else Qt.CheckState.Unchecked)
        self._list.blockSignals(blocked)
        self._caption()


# ---------------------------------------------------------------------------
# The options form
# ---------------------------------------------------------------------------

class OptionsPanel(QtWidgets.QScrollArea):
    """The form of T1Prep's options, built from ``launch.options.OPTIONS``.

    It keeps what the user chose apart from what is shown: an option another
    choice forces (``--bids`` compresses the volumes) shows the forced value
    and is disabled, and returns to the user's own choice when the force goes.
    """

    changed = Signal()

    def __init__(self, defaults: Mapping[str, str], parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._defaults = opt.default_values(defaults)
        self._values: Dict[str, object] = dict(self._defaults)
        self._controls: Dict[str, QtWidgets.QWidget] = {}
        self._labels: Dict[str, QtWidgets.QLabel] = {}
        self._sections: Dict[str, Section] = {}
        self._locked = False

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(2)
        for index, (key, title, open_) in enumerate(opt.GROUPS):
            items = [item for item in opt.OPTIONS if item.group == key]
            if not items:
                continue
            if index:
                layout.addWidget(rule())
            if key == "steps":         # what the run does: always in view
                target = QtWidgets.QVBoxLayout()
                target.setContentsMargins(2, 4, 0, 8)
                target.setSpacing(4)
                layout.addLayout(target)
            else:
                section = Section(title)
                section.set_open(open_)
                self._sections[key] = section
                layout.addWidget(section)
                target = section.content
            for item in items:
                self._add(item, target)
        layout.addWidget(rule())
        self.reset_button = QtWidgets.QPushButton("Restore defaults")
        self.reset_button.setObjectName("link")
        self.reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_button.clicked.connect(self.reset)
        layout.addSpacing(6)
        layout.addWidget(self.reset_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        self.setWidget(body)
        self._sync()

    # -- building ----------------------------------------------------------

    def _add(self, item: opt.Option, target: QtWidgets.QVBoxLayout) -> None:
        if item.kind == "bool":
            control = QtWidgets.QCheckBox(item.label)
            control.toggled.connect(lambda value, k=item.key: self._edited(k, bool(value)))
            control.setToolTip(item.help)
            self._controls[item.key] = control
            target.addWidget(control)
            return
        if item.kind == "int":
            control = QtWidgets.QSpinBox()
            control.setRange(item.minimum, item.maximum)
            control.setSuffix(item.suffix)
            if item.auto is not None:
                control.setSpecialValueText("Auto")
            control.setFixedWidth(132)
            control.valueChanged.connect(lambda value, k=item.key: self._edited(k, int(value)))
        elif item.kind == "choice":
            control = QtWidgets.QComboBox()
            for value, text in opt.choices(item):
                control.addItem(text, value)
            control.setFixedWidth(132)
            control.currentIndexChanged.connect(
                lambda _i, k=item.key, c=control: self._edited(k, str(c.currentData())))
        else:
            surface = item.key == "atlas_surf"
            names = (opt.list_surface_atlas_names() if surface
                     else opt.list_atlas_names())
            control = AtlasPicker(
                names, lambda name, s=surface: opt.atlas_description(name, s))
            control.changed.connect(lambda k=item.key, c=control: self._edited(k, c.value()))
        label = QtWidgets.QLabel(item.label)
        label.setToolTip(item.help)
        if item.kind != "atlas":
            control.setToolTip(item.help)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 1, 0, 1)
        row.addWidget(label, 1)
        row.addWidget(control, 0)
        target.addLayout(row)
        self._controls[item.key] = control
        self._labels[item.key] = label

    # -- values ------------------------------------------------------------

    def values(self) -> Dict[str, object]:
        """What the user chose; ``launch.options.effective`` adds the forced."""
        return dict(self._values)

    def set_values(self, values: Mapping[str, object]) -> None:
        """Take over *values* where they fit the option they are for."""
        for key, value in values.items():
            if key not in self._defaults:
                continue
            item, default = opt.option(key), self._defaults[key]
            if item.kind == "bool" and isinstance(value, bool):
                if value != default and not opt.can_change(item, default):
                    continue
                self._values[key] = value
            elif item.kind == "int" and isinstance(value, int) and not isinstance(value, bool):
                self._values[key] = min(max(value, item.minimum), item.maximum)
            elif item.kind == "choice" and value in [v for v, _ in opt.choices(item)]:
                self._values[key] = value
            elif item.kind == "atlas" and isinstance(value, (list, tuple)):
                known = (opt.list_atlas_names() if key == "atlas"
                         else opt.list_surface_atlas_names())
                self._values[key] = tuple(name for name in known if name in value)
        self._sync()
        self.changed.emit()

    def reset(self) -> None:
        """Back to what T1Prep does by itself."""
        self._values = dict(self._defaults)
        self._sync()
        self.changed.emit()

    def set_locked(self, locked: bool) -> None:
        """Keep the form from being changed while a run uses it."""
        self._locked = bool(locked)
        self._sync()

    def sections_open(self) -> Dict[str, bool]:
        return {key: section.is_open() for key, section in self._sections.items()}

    def set_sections_open(self, state: Mapping[str, object]) -> None:
        for key, section in self._sections.items():
            if key in state:
                section.set_open(str(state[key]).lower() in ("1", "true"))

    def _edited(self, key: str, value: object) -> None:
        self._values[key] = value
        # One of the two steps has to run: switching the last one off
        # switches the other one on
        current = opt.effective(self._values)
        if key in ("seg", "surf") and not current["seg"] and not current["surf"]:
            other = "surf" if key == "seg" else "seg"
            self._values[other] = True
            if opt.problems(self._values):     # the other one is forced off
                self._values[key] = True
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        """Show the values as they will be used and disable what cannot change."""
        forced = opt.implied(self._values)
        unused = opt.irrelevant(self._values)
        for key, control in self._controls.items():
            item = opt.option(key)
            value = forced.get(key, self._values[key])
            enabled = (not self._locked and key not in forced and key not in unused
                       and opt.can_change(item, self._defaults[key]))
            blocked = control.blockSignals(True)
            if item.kind == "bool":
                control.setChecked(bool(value))
            elif item.kind == "int":
                control.setValue(int(value))
            elif item.kind == "choice":
                control.setCurrentIndex(max(0, control.findData(str(value))))
            else:
                control.set_value(tuple(value or ()))
            control.blockSignals(blocked)
            control.setEnabled(enabled)
            if key in self._labels:
                self._labels[key].setEnabled(enabled)
        self.reset_button.setEnabled(
            not self._locked and self._values != self._defaults)


# ---------------------------------------------------------------------------
# The results of one image
# ---------------------------------------------------------------------------

class PreviewLabel(QtWidgets.QLabel):
    """Shows a picture scaled to its width, or a line of text in its place."""

    #: Height while there is no picture
    EMPTY_HEIGHT = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(self.EMPTY_HEIGHT)
        self._picture: Optional[QtGui.QPixmap] = None

    def show_picture(self, picture: Optional[QtGui.QPixmap], text: str = "") -> None:
        self._picture = picture if picture is not None and not picture.isNull() else None
        if self._picture is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText(text)
            self.setFixedHeight(self.EMPTY_HEIGHT)
        else:
            self._scale()

    def _scale(self) -> None:
        """Fit the picture to the width; the height follows its shape."""
        if self._picture is None or self.width() < 20:
            return
        ratio = self.devicePixelRatioF()
        width = self.width() - 8
        height = round(width * self._picture.height() / max(1, self._picture.width()))
        scaled = self._picture.scaled(
            QtCore.QSize(int(width * ratio), int(height * ratio)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(ratio)
        self.setFixedHeight(height + 8)
        self.setPixmap(scaled)

    def resizeEvent(self, event):      # noqa: N802
        super().resizeEvent(event)
        self._scale()


class DetailsPanel(QtWidgets.QScrollArea):
    """What a run left for the selected image: a picture, measures, viewers."""

    view_requested = Signal(str)        # "segmentation", "surfaces", "thickness"
    stage_changed = Signal(str)         # the picture wanted: "volume" or "surface"
    log_requested = Signal()
    folder_requested = Signal()

    VIEWS = (("segmentation", "Segmentation", "Image and tissue label in CAT_VolView"),
             ("surfaces", "Surfaces", "Central surfaces outlined on the image in CAT_VolView"),
             ("thickness", "Thickness", "Cortical thickness on the surface in CAT_SurfView"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        self.name = QtWidgets.QLabel()
        self.name.setObjectName("title")
        self.name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path = QtWidgets.QLabel()
        self.path.setObjectName("path")
        layout.addWidget(self.name)
        layout.addWidget(self.path)
        layout.addSpacing(4)

        self.stages = Segments(((VOLUME, "Segmentation"), (SURFACE, "Surfaces")))
        self.stages.changed.connect(self.stage_changed)
        layout.addWidget(self.stages, 0, Qt.AlignmentFlag.AlignHCenter)
        self.preview = PreviewLabel()
        layout.addWidget(self.preview)

        heading = QtWidgets.QLabel("Open in viewer")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        self._views: Dict[str, QtWidgets.QPushButton] = {}
        for key, text, tip in self.VIEWS:
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, k=key: self.view_requested.emit(k))
            buttons.addWidget(button)
            self._views[key] = button
        layout.addLayout(buttons)

        self.measures = QtWidgets.QGridLayout()
        self.measures.setContentsMargins(0, 0, 0, 0)
        self.measures.setHorizontalSpacing(10)
        self.measures.setVerticalSpacing(3)
        self.measures.setColumnStretch(0, 1)
        layout.addLayout(self.measures)

        layout.addSpacing(6)
        links = QtWidgets.QHBoxLayout()
        self.log_link = QtWidgets.QPushButton("Show log")
        self.folder_link = QtWidgets.QPushButton("Open folder")
        for link, signal in ((self.log_link, self.log_requested),
                             (self.folder_link, self.folder_requested)):
            link.setObjectName("link")
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.clicked.connect(signal)
            links.addWidget(link)
            links.addSpacing(14)
        links.addStretch(1)
        layout.addLayout(links)
        layout.addStretch(1)
        self.setWidget(body)
        self.show_row(None)

    def stage(self) -> str:
        """The phase whose picture is asked for."""
        return self.stages.current()

    def show_row(self, row: Optional[Row], report: Optional[dict] = None,
                 available: Optional[Mapping[str, bool]] = None) -> None:
        """Fill the panel for *row*; None empties it.

        Args:
            report: The parsed report of the row's subject.
            available: Per view of :attr:`VIEWS`, whether its files exist.
        """
        available = available or {}
        for key, button in self._views.items():
            button.setEnabled(bool(available.get(key)))
        self._clear_measures()
        if row is None:
            self.name.setText("No image selected")
            self.path.setText("Select an image to see its results.")
            self.stages.setVisible(False)
            self.preview.show_picture(None, "")
            self.log_link.setEnabled(False)
            self.folder_link.setEnabled(False)
            return
        subject = row.subject
        self.name.setText(subject.name)
        self.path.setText(shorten_path(str(subject.folder), max_chars=44))
        self.path.setToolTip(str(subject.folder))
        self.stages.setVisible(True)
        self.stages.set_available(VOLUME, bool(available.get("segmentation")))
        self.stages.set_available(SURFACE, bool(available.get("surfaces")))
        view_of = {VOLUME: "segmentation", SURFACE: "surfaces"}
        if not available.get(view_of[self.stages.current()]):
            # Show the phase that has something to show
            for stage in (VOLUME, SURFACE):
                if available.get(view_of[stage]):
                    self.stages.set_current(stage)
                    break
        self.log_link.setEnabled(subject.log_file() is not None)
        self.folder_link.setEnabled(subject.folder.is_dir())
        self._fill_measures(report)

    def _clear_measures(self) -> None:
        while self.measures.count():
            item = self.measures.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _fill_measures(self, report: Optional[dict]) -> None:
        line = 0
        for title, rows in describe(report):
            heading = QtWidgets.QLabel(title)
            heading.setObjectName("heading")
            self.measures.addWidget(heading, line, 0, 1, 2)
            line += 1
            for label, text in rows:
                name = QtWidgets.QLabel(label)
                name.setObjectName("muted")
                value = QtWidgets.QLabel(text)
                value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                self.measures.addWidget(name, line, 0)
                self.measures.addWidget(value, line, 1)
                line += 1


# ---------------------------------------------------------------------------
# Adding a folder
# ---------------------------------------------------------------------------

class FolderDialog(QtWidgets.QDialog):
    """Which images of a folder to add: a name pattern, with or without subfolders."""

    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add folder")
        self.folder = Path(folder)
        self.images: List[Path] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        where = QtWidgets.QLabel(shorten_path(str(self.folder), max_chars=60))
        where.setObjectName("title")
        where.setToolTip(str(self.folder))
        layout.addWidget(where)
        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("File names"))
        self.pattern = QtWidgets.QLineEdit(DEFAULT_PATTERN)
        self.pattern.setToolTip("Shell pattern, e.g. *_T1w.nii.gz")
        self.pattern.setMinimumWidth(220)
        form.addWidget(self.pattern, 1)
        layout.addLayout(form)
        self.recursive = QtWidgets.QCheckBox("Include subfolders")
        self.recursive.setToolTip(
            "Folders with T1Prep results (mri, surf, report, label, "
            "derivatives) are left out.")
        layout.addWidget(self.recursive)
        self.count = QtWidgets.QLabel()
        self.count.setObjectName("muted")
        layout.addWidget(self.count)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.accept_button = QtWidgets.QPushButton("Add")
        self.accept_button.setObjectName("primary")
        self.accept_button.setDefault(True)
        self.accept_button.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self.accept_button)
        layout.addLayout(buttons)
        self.pattern.textChanged.connect(self._search)
        self.recursive.toggled.connect(self._search)
        self._search()

    def _search(self) -> None:
        self.images = find_images(self.folder, self.pattern.text(),
                                  self.recursive.isChecked())
        number = len(self.images)
        self.count.setText("No image matches." if not number else
                           f"{number} image{'s' if number != 1 else ''} found.")
        self.accept_button.setEnabled(bool(number))
        self.accept_button.setText(f"Add {number}" if number else "Add")
