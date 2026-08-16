"""The control panel shared by the surface and the volume viewer.

Both offer the same handles on an overlay — value range, clip window,
colormap, opacity, inversion, discrete levels and the p-value thresholds — so
the widget lives here and each viewer wires it to its own data.
"""

from __future__ import annotations

import math
from typing import Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

ORIENT_H = Qt.Orientation.Horizontal

try:
    from .colormaps import COLORMAP_NAMES
except ImportError:  # direct invocation as a script
    from colormaps import COLORMAP_NAMES


#: Thresholds offered for -log10(p) overlays, as in cat_surf_results: the label
#: shown in the control panel and the value in -log10(p) units.  Selecting one
#: hides everything between -value and +value.
LOGP_THRESHOLDS = (
    ('none', 0.0),
    ('p<0.05', -math.log10(0.05)),
    ('p<0.01', -math.log10(0.01)),
    ('p<0.001', -math.log10(0.001)),
)


class HistogramWidget(QtWidgets.QWidget):
    """Intensity histogram with the displayed window drawn over it.

    Setting a display range by typing numbers means guessing where the tissue
    classes are; over a histogram the same range is one drag, and the two
    handles are the ends of the window.  Emits ``windowChanged(low, high)``
    while a handle is dragged.
    """

    windowChanged = Signal(float, float)

    #: How close to a handle a click has to be, in pixels
    GRAB = 6

    def __init__(self, parent=None, bins: int = 128):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMinimumWidth(160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Drag the two handles to set the displayed intensity "
                        "range; double-click for the full range")
        self._bins = int(bins)
        self._counts: list = []
        self._low = 0.0
        self._high = 1.0
        self._window = (0.0, 1.0)
        self._dragging = None       # 'low', 'high' or None

    def sizeHint(self):
        """A form layout gives a field its size hint, and without one — which
        is what a plain QWidget reports — the histogram ends up 0 px wide."""
        return QtCore.QSize(320, 90)

    def minimumSizeHint(self):
        return QtCore.QSize(160, 70)

    # ---- data ----
    def set_values(self, values, low: float = None, high: float = None):
        """Take the intensities to show (any sequence of numbers)."""
        try:
            import numpy as np
        except ImportError:       # pragma: no cover - numpy is a dependency
            return
        data = np.asarray(values, dtype=float).ravel()
        data = data[np.isfinite(data)]
        if data.size == 0:
            self._counts = []
            self.update()
            return
        if data.size > 200000:    # a histogram does not need every voxel
            data = data[:: max(1, data.size // 200000)]
        self._low = float(low if low is not None else data.min())
        self._high = float(high if high is not None else data.max())
        if self._high <= self._low:
            self._high = self._low + 1.0
        counts, _ = np.histogram(data, bins=self._bins,
                                 range=(self._low, self._high))
        # The background peak of an MRI dwarfs everything else, so the bars are
        # drawn on a log scale — otherwise only the air is visible
        self._counts = np.log1p(counts).tolist()
        self.update()

    def set_window(self, low: float, high: float):
        """Show *low*..*high* as the selected window."""
        self._window = (float(low), float(high))
        self.update()

    def window(self) -> Tuple[float, float]:
        return self._window

    # ---- geometry ----
    def _to_x(self, value: float) -> float:
        span = self._high - self._low or 1.0
        return (value - self._low) / span * max(1, self.width() - 1)

    def _to_value(self, x: float) -> float:
        span = self._high - self._low or 1.0
        return self._low + x / max(1, self.width() - 1) * span

    # ---- painting ----
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        width, height = self.width(), self.height()
        if self._counts:
            peak = max(self._counts) or 1.0
            step = width / len(self._counts)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(150, 150, 150))
            for i, count in enumerate(self._counts):
                bar = (count / peak) * (height - 4)
                painter.drawRect(QtCore.QRectF(i * step, height - bar, step, bar))
        # the window as a bright band, its ends as handles
        low_x, high_x = self._to_x(self._window[0]), self._to_x(self._window[1])
        painter.setBrush(QtGui.QColor(80, 140, 220, 60))
        painter.drawRect(QtCore.QRectF(low_x, 0, max(1.0, high_x - low_x), height))
        pen = QtGui.QPen(QtGui.QColor(120, 190, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        for x in (low_x, high_x):
            painter.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, height))

    # ---- dragging ----
    def mousePressEvent(self, event):
        x = event.position().x()
        distances = {'low': abs(x - self._to_x(self._window[0])),
                     'high': abs(x - self._to_x(self._window[1]))}
        nearest = min(distances, key=distances.get)
        # Clicking well away from both handles moves the nearer one there, so
        # the window can be set without hitting a two-pixel line
        self._dragging = nearest
        self._drag_to(x)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_to(event.position().x())

    def mouseReleaseEvent(self, event):
        self._dragging = None

    def mouseDoubleClickEvent(self, event):
        """Back to the full range."""
        self._window = (self._low, self._high)
        self.update()
        self.windowChanged.emit(*self._window)

    def _drag_to(self, x: float):
        value = self._to_value(max(0.0, min(float(self.width()), x)))
        low, high = self._window
        if self._dragging == 'low':
            low = min(value, high - 1e-9)
        else:
            high = max(value, low + 1e-9)
        self._window = (low, high)
        self.update()
        self.windowChanged.emit(low, high)


class ControlPanel(QtWidgets.QWidget):
    """The docked panel both viewers use to control an overlay.

    Overlay picker, value range, clip window, p-value thresholds, background
    range with its histogram, opacity, colormap, discrete levels and
    inversion.  The rows that only make sense for a surface are hidden by
    :meth:`configure_for_volume`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10,10,10,10)
        form = QtWidgets.QFormLayout()
        self.form = form
        # Internal bounds for slider mapping (min,max)
        self._overlay_bounds = (-1.0, 1.0)
        self._clip_bounds = (-1.0, 1.0)
        self._bkg_bounds = (-1.0, 1.0)

        # Overlay selector (editable combo for long names + direct selection) — FIRST ROW
        self.overlay_combo = QtWidgets.QComboBox()
        self.overlay_combo.setEditable(True)
        self.overlay_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.overlay_combo.setMinimumContentsLength(50)
        try:
            # Widen dropdown popup for long paths
            self.overlay_combo.view().setMinimumWidth(600)
        except Exception:
            pass
        self.overlay_btn = QtWidgets.QPushButton("…")
        ov_box = QtWidgets.QHBoxLayout(); ov_box.addWidget(self.overlay_combo, 1); ov_box.addWidget(self.overlay_btn)
        self._overlay_row = self._wrap(ov_box)
        form.addRow("Overlay", self._overlay_row)

        # Volume (orthogonal view) — simple button
        self.volume_btn = QtWidgets.QPushButton("Open NIfTI…")
        form.addRow("Volume", self.volume_btn)
        self.volume_row = self.volume_btn

        # Range (overlay)
        self.range_min = QtWidgets.QDoubleSpinBox(); self.range_min.setDecimals(6); self.range_min.setRange(-1e9, 1e9)
        self.range_max = QtWidgets.QDoubleSpinBox(); self.range_max.setDecimals(6); self.range_max.setRange(-1e9, 1e9)
        self.range_slider_min = QtWidgets.QSlider(ORIENT_H); self.range_slider_min.setRange(0, 1000)
        self.range_slider_max = QtWidgets.QSlider(ORIENT_H); self.range_slider_max.setRange(0, 1000)
        range_box = QtWidgets.QHBoxLayout(); range_box.addWidget(self.range_min); range_box.addWidget(self.range_slider_min); range_box.addWidget(self.range_slider_max); range_box.addWidget(self.range_max)
        form.addRow("Range (overlay)", self._wrap(range_box))

        # Clip
        self.clip_min = QtWidgets.QDoubleSpinBox(); self.clip_min.setDecimals(6); self.clip_min.setRange(-1e9, 1e9)
        self.clip_max = QtWidgets.QDoubleSpinBox(); self.clip_max.setDecimals(6); self.clip_max.setRange(-1e9, 1e9)
        self.clip_slider_min = QtWidgets.QSlider(ORIENT_H); self.clip_slider_min.setRange(0, 1000)
        self.clip_slider_max = QtWidgets.QSlider(ORIENT_H); self.clip_slider_max.setRange(0, 1000)
        clip_box = QtWidgets.QHBoxLayout(); clip_box.addWidget(self.clip_min); clip_box.addWidget(self.clip_slider_min); clip_box.addWidget(self.clip_slider_max); clip_box.addWidget(self.clip_max)
        form.addRow("Clip window", self._wrap(clip_box))

        # Threshold for -log10(p) overlays, as in cat_surf_results.  The row is
        # only shown for such overlays, where these are the values that matter.
        self.threshold = QtWidgets.QComboBox()
        self.threshold.addItems([label for label, _ in LOGP_THRESHOLDS])
        thr_box = QtWidgets.QHBoxLayout()
        thr_box.addWidget(self.threshold)
        thr_box.addStretch(1)
        self.threshold_row = self._wrap(thr_box)
        self.threshold_label = QtWidgets.QLabel("Threshold")
        form.addRow(self.threshold_label, self.threshold_row)
        self.set_threshold_visible(False)

        # Range bkg
        self.bkg_min = QtWidgets.QDoubleSpinBox(); self.bkg_min.setDecimals(6); self.bkg_min.setRange(-1e9, 1e9)
        self.bkg_max = QtWidgets.QDoubleSpinBox(); self.bkg_max.setDecimals(6); self.bkg_max.setRange(-1e9, 1e9)
        self.bkg_slider_min = QtWidgets.QSlider(ORIENT_H); self.bkg_slider_min.setRange(0, 1000)
        self.bkg_slider_max = QtWidgets.QSlider(ORIENT_H); self.bkg_slider_max.setRange(0, 1000)
        bkg_box = QtWidgets.QHBoxLayout(); bkg_box.addWidget(self.bkg_min); bkg_box.addWidget(self.bkg_slider_min); bkg_box.addWidget(self.bkg_slider_max); bkg_box.addWidget(self.bkg_max)
        self._bkg_row = self._wrap(bkg_box)
        form.addRow("Range (bkg)", self._bkg_row)
        # Histogram of the image, with the same range as draggable handles
        self.histogram = HistogramWidget()
        self.histogram_label = QtWidgets.QLabel("Histogram")
        form.addRow(self.histogram_label, self.histogram)
        self.set_histogram_visible(False)
        # Opacity
        self.opacity = QtWidgets.QSlider(ORIENT_H); self.opacity.setRange(0,100); self.opacity.setValue(80)
        form.addRow("Opacity", self.opacity)
        # Toggles
        self.cb_colorbar = QtWidgets.QCheckBox("Show colorbar")
        self.cb_discrete = QtWidgets.QCheckBox("Discrete")
        # Colormap selector
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(list(COLORMAP_NAMES))  # order visible to user
        # Title mode selector (shape | stats | none)
        self.title_mode = QtWidgets.QComboBox(); self.title_mode.addItems(["shape","stats","none"])
        self.cb_inverse = QtWidgets.QCheckBox("Inverse (flip sign)")
        self.cb_fix_scaling = QtWidgets.QCheckBox("Fix scaling")
        self.cb_histogram = QtWidgets.QCheckBox("Show histogram")
        # Put Show colorbar and Colorbar title on one row (aligned with other checkboxes)
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(8)
        row.addWidget(self.cb_colorbar)
        row.addWidget(self.cb_discrete)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Colormap"))
        row.addWidget(self.colormap)
        self.title_mode_label = QtWidgets.QLabel("Colorbar title")
        row.addWidget(self.title_mode_label)
        row.addWidget(self.title_mode)
        form.addRow(self._wrap(row))
        form.addRow(self.cb_inverse)
        form.addRow(self.cb_fix_scaling)
        form.addRow(self.cb_histogram)
        self.layout.addLayout(form)
        # Action buttons (none for now)
        self.layout.addStretch(1)

        # --- Wiring: bidirectional sync between sliders and spin boxes ---
        # Overlay range
        self.range_slider_min.valueChanged.connect(lambda v: self._slider_to_spin('overlay', 'min', v))
        self.range_slider_max.valueChanged.connect(lambda v: self._slider_to_spin('overlay', 'max', v))
        self.range_min.valueChanged.connect(lambda v: self._spin_to_slider('overlay', 'min', float(v)))
        self.range_max.valueChanged.connect(lambda v: self._spin_to_slider('overlay', 'max', float(v)))
        # Clip window
        self.clip_slider_min.valueChanged.connect(lambda v: self._slider_to_spin('clip', 'min', v))
        self.clip_slider_max.valueChanged.connect(lambda v: self._slider_to_spin('clip', 'max', v))
        self.clip_min.valueChanged.connect(lambda v: self._spin_to_slider('clip', 'min', float(v)))
        self.clip_max.valueChanged.connect(lambda v: self._spin_to_slider('clip', 'max', float(v)))
        # Background
        self.bkg_slider_min.valueChanged.connect(lambda v: self._slider_to_spin('bkg', 'min', v))
        self.bkg_slider_max.valueChanged.connect(lambda v: self._slider_to_spin('bkg', 'max', v))
        self.bkg_min.valueChanged.connect(lambda v: self._spin_to_slider('bkg', 'min', float(v)))
        self.bkg_max.valueChanged.connect(lambda v: self._spin_to_slider('bkg', 'max', float(v)))

    def _wrap(self, hbox: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); w.setLayout(hbox); return w

    def configure_for_volume(self):
        """Drop the rows that only make sense for a surface.

        What is left — overlay picker, range, clip, threshold, background
        range, opacity, colormap, discrete and inverse — is exactly what the
        volume viewer needs for its overlay.
        """
        try:
            self.form.setRowVisible(self.volume_row, False)
        except Exception:
            self.volume_btn.setVisible(False)
        for widget in (self.cb_colorbar, self.title_mode, self.title_mode_label,
                       self.cb_fix_scaling, self.cb_histogram):
            widget.setVisible(False)
        self.form.labelForField(self._bkg_row).setText("Range (image)")
        # macOS leaves form fields at their size hint, which would keep the
        # histogram narrower than the row it belongs to
        try:
            self.form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        except Exception:
            pass
        self.set_histogram_visible(True)

    def set_labels_for_volume(self):
        """Wording that fits volumes rather than surfaces."""
        self.form.labelForField(self._overlay_row).setText("Overlay volume")

    def set_histogram_visible(self, visible: bool):
        """Show the intensity histogram (the volume viewer fills it)."""
        for widget in (self.histogram_label, self.histogram):
            widget.setVisible(bool(visible))

    def set_threshold_visible(self, visible: bool):
        """Show the p-value threshold row (only useful for -log10(p) overlays)."""
        for widget in (self.threshold_label, self.threshold_row):
            widget.setVisible(bool(visible))

    def set_threshold_from_clip(self, clip: Tuple[float, float]):
        """Select the entry matching *clip*, or 'none' for anything else."""
        index = 0
        if clip[1] > clip[0]:
            for i, (_, value) in enumerate(LOGP_THRESHOLDS):
                if value and abs(clip[1] - value) < 0.05 and abs(clip[0] + value) < 0.05:
                    index = i
                    break
        self.threshold.blockSignals(True)
        self.threshold.setCurrentIndex(index)
        self.threshold.blockSignals(False)


    def set_overlay_controls_enabled(self, enabled: bool):
        """Enable or disable overlay-related controls based on whether an overlay is loaded."""
        # Ensure a strict boolean is used (avoid None/[] leaking from callers)
        enabled = bool(enabled)
        # Range controls
        self.range_min.setEnabled(enabled)
        self.range_max.setEnabled(enabled)
        self.range_slider_min.setEnabled(enabled)
        self.range_slider_max.setEnabled(enabled)
        # Clip controls
        self.clip_min.setEnabled(enabled)
        self.clip_max.setEnabled(enabled)
        self.clip_slider_min.setEnabled(enabled)
        self.clip_slider_max.setEnabled(enabled)
        # Colorbar and title controls
        self.cb_colorbar.setEnabled(enabled)
        # Discrete applies to the overlay LUT regardless of colorbar visibility
        self.cb_discrete.setEnabled(enabled)
        # Title is relevant only when the colorbar is shown
        self.title_mode.setEnabled(enabled and self.cb_colorbar.isChecked())
        # Colormap selector
        self.colormap.setEnabled(enabled)
        # Inverse control
        self.cb_inverse.setEnabled(enabled)
        # Background and opacity are also not meaningful until data is loaded
        self.bkg_min.setEnabled(enabled)
        self.bkg_max.setEnabled(enabled)
        self.bkg_slider_min.setEnabled(enabled)
        self.bkg_slider_max.setEnabled(enabled)
        self.opacity.setEnabled(enabled)
        # Fix scaling only makes sense with at least one overlay
        self.cb_fix_scaling.setEnabled(enabled)
        # Histogram available only when overlay is loaded
        self.cb_histogram.setEnabled(enabled)
        if not enabled:
            try:
                self.cb_histogram.blockSignals(True)
                self.cb_histogram.setChecked(False)
            finally:
                self.cb_histogram.blockSignals(False)

    # ---- Slider helpers ----
    def _bounds(self, which: str):
        return {
            'overlay': self._overlay_bounds,
            'clip': self._clip_bounds,
            'bkg': self._bkg_bounds,
        }[which]

    @staticmethod
    def _to_slider(value: float, bounds: tuple) -> int:
        a, b = bounds
        if b <= a:
            return 0
        t = (float(value) - float(a)) / (float(b) - float(a))
        t = max(0.0, min(1.0, t))
        return int(round(t * 1000.0))

    @staticmethod
    def _from_slider(ticks: int, bounds: tuple) -> float:
        a, b = bounds
        if b <= a:
            return float(a)
        t = max(0, min(1000, int(ticks))) / 1000.0
        return float(a) + t * (float(b) - float(a))

    def _slider_to_spin(self, which: str, part: str, ticks: int):
        bounds = self._bounds(which)
        val = self._from_slider(ticks, bounds)
        if which == 'overlay':
            if part == 'min':
                # Enforce min <= max
                if self.range_slider_min.value() > self.range_slider_max.value():
                    self.range_slider_max.blockSignals(True)
                    self.range_slider_max.setValue(self.range_slider_min.value())
                    self.range_slider_max.blockSignals(False)
                self.range_min.blockSignals(True); self.range_min.setValue(val); self.range_min.blockSignals(False)
            else:
                if self.range_slider_max.value() < self.range_slider_min.value():
                    self.range_slider_min.blockSignals(True)
                    self.range_slider_min.setValue(self.range_slider_max.value())
                    self.range_slider_min.blockSignals(False)
                self.range_max.blockSignals(True); self.range_max.setValue(val); self.range_max.blockSignals(False)
        elif which == 'clip':
            if part == 'min':
                if self.clip_slider_min.value() > self.clip_slider_max.value():
                    self.clip_slider_max.blockSignals(True)
                    self.clip_slider_max.setValue(self.clip_slider_min.value())
                    self.clip_slider_max.blockSignals(False)
                self.clip_min.blockSignals(True); self.clip_min.setValue(val); self.clip_min.blockSignals(False)
            else:
                if self.clip_slider_max.value() < self.clip_slider_min.value():
                    self.clip_slider_min.blockSignals(True)
                    self.clip_slider_min.setValue(self.clip_slider_max.value())
                    self.clip_slider_min.blockSignals(False)
                self.clip_max.blockSignals(True); self.clip_max.setValue(val); self.clip_max.blockSignals(False)
        elif which == 'bkg':
            if part == 'min':
                if self.bkg_slider_min.value() > self.bkg_slider_max.value():
                    self.bkg_slider_max.blockSignals(True)
                    self.bkg_slider_max.setValue(self.bkg_slider_min.value())
                    self.bkg_slider_max.blockSignals(False)
                self.bkg_min.blockSignals(True); self.bkg_min.setValue(val); self.bkg_min.blockSignals(False)
            else:
                if self.bkg_slider_max.value() < self.bkg_slider_min.value():
                    self.bkg_slider_min.blockSignals(True)
                    self.bkg_slider_min.setValue(self.bkg_slider_max.value())
                    self.bkg_slider_min.blockSignals(False)
                self.bkg_max.blockSignals(True); self.bkg_max.setValue(val); self.bkg_max.blockSignals(False)

    def _spin_to_slider(self, which: str, part: str, value: float):
        bounds = self._bounds(which)
        ticks = self._to_slider(value, bounds)
        if which == 'overlay':
            if part == 'min':
                if ticks > self.range_slider_max.value():
                    self.range_slider_max.blockSignals(True)
                    self.range_slider_max.setValue(ticks)
                    self.range_slider_max.blockSignals(False)
                self.range_slider_min.blockSignals(True); self.range_slider_min.setValue(ticks); self.range_slider_min.blockSignals(False)
            else:
                if ticks < self.range_slider_min.value():
                    self.range_slider_min.blockSignals(True)
                    self.range_slider_min.setValue(ticks)
                    self.range_slider_min.blockSignals(False)
                self.range_slider_max.blockSignals(True); self.range_slider_max.setValue(ticks); self.range_slider_max.blockSignals(False)
        elif which == 'clip':
            if part == 'min':
                if ticks > self.clip_slider_max.value():
                    self.clip_slider_max.blockSignals(True)
                    self.clip_slider_max.setValue(ticks)
                    self.clip_slider_max.blockSignals(False)
                self.clip_slider_min.blockSignals(True); self.clip_slider_min.setValue(ticks); self.clip_slider_min.blockSignals(False)
            else:
                if ticks < self.clip_slider_min.value():
                    self.clip_slider_min.blockSignals(True)
                    self.clip_slider_min.setValue(ticks)
                    self.clip_slider_min.blockSignals(False)
                self.clip_slider_max.blockSignals(True); self.clip_slider_max.setValue(ticks); self.clip_slider_max.blockSignals(False)
        elif which == 'bkg':
            if part == 'min':
                if ticks > self.bkg_slider_max.value():
                    self.bkg_slider_max.blockSignals(True)
                    self.bkg_slider_max.setValue(ticks)
                    self.bkg_slider_max.blockSignals(False)
                self.bkg_slider_min.blockSignals(True); self.bkg_slider_min.setValue(ticks); self.bkg_slider_min.blockSignals(False)
            else:
                if ticks < self.bkg_slider_min.value():
                    self.bkg_slider_min.blockSignals(True)
                    self.bkg_slider_min.setValue(ticks)
                    self.bkg_slider_min.blockSignals(False)
                self.bkg_slider_max.blockSignals(True); self.bkg_slider_max.setValue(ticks); self.bkg_slider_max.blockSignals(False)

    # Public: set slider bounds (min,max) and align slider positions to current spin values
    def set_overlay_bounds(self, vmin: float, vmax: float):
        """Set what the overlay range sliders span (the spin boxes keep their value)."""
        self._overlay_bounds = (float(vmin), float(vmax))
        self._spin_to_slider('overlay', 'min', float(self.range_min.value()))
        self._spin_to_slider('overlay', 'max', float(self.range_max.value()))

    def set_clip_bounds(self, vmin: float, vmax: float):
        self._clip_bounds = (float(vmin), float(vmax))
        self._spin_to_slider('clip', 'min', float(self.clip_min.value()))
        self._spin_to_slider('clip', 'max', float(self.clip_max.value()))

    def set_bkg_bounds(self, vmin: float, vmax: float):
        self._bkg_bounds = (float(vmin), float(vmax))
        self._spin_to_slider('bkg', 'min', float(self.bkg_min.value()))
        self._spin_to_slider('bkg', 'max', float(self.bkg_max.value()))
