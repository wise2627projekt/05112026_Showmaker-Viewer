"""Image Trace panel (Illustrator-style controls) as a reusable Qt widget."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .palette_editor import PaletteEditor
from .trace import PRESETS, TraceParams, params_from_preset, used_palette_indices

MODES = [("Black and White", "bw"), ("Grayscale", "gray"), ("Color", "color")]
VIEWS = [("Tracing Result", "result"), ("Outlines", "outline"),
         ("Outlines with Tracing", "result_outline"),
         ("Outlines with Source Image", "outline_source"), ("Source Image", "source")]
RES = [("Fast (1600 px)", 1600), ("Standard (2400 px)", 2400), ("Full resolution", 0)]


class SliderRow(QWidget):
    valueChanged = Signal(int)

    def __init__(self, lo: int, hi: int, val: int, suffix: str = "", less: str = "", more: str = ""):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        h = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(val)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(val)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(70)
        h.addWidget(self.slider, 1)
        h.addWidget(self.spin)
        v.addLayout(h)
        if less or more:
            c = QHBoxLayout()
            c.setContentsMargins(0, 0, 0, 0)
            a, b = QLabel(less), QLabel(more)
            for x in (a, b):
                x.setStyleSheet("color:#888;font-size:10px")
            c.addWidget(a)
            c.addStretch(1)
            c.addWidget(b)
            c.addSpacing(74)
            v.addLayout(c)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        with QSignalBlocker(self.spin):
            self.spin.setValue(v)
        self.valueChanged.emit(v)

    def _from_spin(self, v):
        with QSignalBlocker(self.slider):
            self.slider.setValue(v)
        self.valueChanged.emit(v)

    def value(self) -> int:
        return self.slider.value()

    def set(self, v: int) -> None:
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin):
            self.slider.setValue(v)
            self.spin.setValue(v)


def _row(label: str, w: QWidget) -> QWidget:
    r = QWidget()
    h = QHBoxLayout(r)
    h.setContentsMargins(0, 0, 0, 0)
    lab = QLabel(label)
    lab.setFixedWidth(70)
    h.addWidget(lab)
    h.addWidget(w, 1)
    return r


def _title(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet("font-weight:600;margin-top:6px")
    return l


class TracePanel(QWidget):
    changed = Signal()            # any parameter edit (live Preview hook)
    view_changed = Signal(str)    # view mode only
    trace_clicked = Signal()
    pick_clicked = Signal()
    palette_changed = Signal()

    def __init__(self):
        super().__init__()
        self._busy = False
        self._custom_added = False
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.view = QComboBox()
        for n, _ in VIEWS:
            self.view.addItem(n)
        self.mode = QComboBox()
        for n, _ in MODES:
            self.mode.addItem(n)
        self.palette = QComboBox()
        self.palette.addItems(["Limited", "Automatic"])
        v.addWidget(_row("Preset", self.preset))
        v.addWidget(_row("View", self.view))
        v.addWidget(_row("Mode", self.mode))
        self.palette_row = _row("Palette", self.palette)
        v.addWidget(self.palette_row)
        self.colors = SliderRow(2, 64, 6, "", "2", "64")
        self.colors_row = _row("Colors", self.colors)
        v.addWidget(self.colors_row)
        self.threshold = SliderRow(0, 255, 128, "", "Less", "More")
        self.threshold_row = _row("Threshold", self.threshold)
        v.addWidget(self.threshold_row)
        self.palette_editor = PaletteEditor()
        self.palette_editor.changed.connect(self.palette_changed)
        v.addWidget(self.palette_editor)

        self.adv_btn = QToolButton()
        self.adv_btn.setText("Advanced")
        self.adv_btn.setCheckable(True)
        self.adv_btn.setChecked(True)
        self.adv_btn.setArrowType(Qt.DownArrow)
        self.adv_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.adv_btn.setStyleSheet("border:none;font-weight:600")
        self.adv_btn.toggled.connect(self._toggle_adv)
        v.addWidget(self.adv_btn)

        self.adv = QWidget()
        a = QVBoxLayout(self.adv)
        a.setContentsMargins(0, 0, 0, 0)
        self.paths = SliderRow(0, 100, 50, " %", "Low", "High")
        self.corners = SliderRow(0, 100, 75, " %", "Less", "More")
        self.noise = SliderRow(1, 100, 25, " px", "1", "100")
        a.addWidget(_row("Paths", self.paths))
        a.addWidget(_row("Corners", self.corners))
        a.addWidget(_row("Noise", self.noise))
        mrow = QWidget()
        mh = QHBoxLayout(mrow)
        mh.setContentsMargins(0, 0, 0, 0)
        self.abut = QToolButton()
        self.abut.setText("Abutting")
        self.abut.setCheckable(True)
        self.abut.setChecked(True)
        self.abut.setToolTip("Cut-out shapes that touch; no overlaps (adds a base colour layer).")
        self.over = QToolButton()
        self.over.setText("Overlapping")
        self.over.setCheckable(True)
        self.over.setToolTip("Stacked shapes that overlap; no gaps possible.")
        mh.addWidget(self.abut)
        mh.addWidget(self.over)
        mh.addStretch(1)
        a.addWidget(_row("Method", mrow))
        a.addWidget(_title("Create"))
        fr = QWidget()
        fh = QHBoxLayout(fr)
        fh.setContentsMargins(0, 0, 0, 0)
        self.fills = QCheckBox("Fills")
        self.fills.setChecked(True)
        self.strokes = QCheckBox("Strokes")
        self.stroke_w = QDoubleSpinBox()
        self.stroke_w.setRange(1, 100)
        self.stroke_w.setValue(10)
        self.stroke_w.setSuffix(" px")
        self.stroke_w.setToolTip("Features narrower than this become centre-line strokes.")
        fh.addWidget(self.fills)
        fh.addWidget(self.strokes)
        fh.addWidget(self.stroke_w)
        a.addWidget(fr)
        orow = QWidget()
        oh = QHBoxLayout(orow)
        oh.setContentsMargins(0, 0, 0, 0)
        self.outlines = QCheckBox("Outlines")
        self.outlines.setToolTip("Draw the outline of every fill as a stroke (works with or without Fills).")
        self.outline_w = QDoubleSpinBox()
        self.outline_w.setRange(0.2, 50)
        self.outline_w.setSingleStep(0.5)
        self.outline_w.setValue(1.5)
        self.outline_w.setSuffix(" px")
        self.outline_fixed = QCheckBox("Fixed")
        self.outline_fixed.setToolTip("Use one fixed outline colour instead of each fill's own colour.")
        self.outline_color = QColor(0, 0, 0)
        self.outline_btn = QPushButton()
        self.outline_btn.setFixedSize(34, 22)
        self.outline_btn.clicked.connect(self._choose_outline_color)
        self._paint_outline_swatch()
        for w in (self.outlines, self.outline_w, self.outline_fixed, self.outline_btn):
            oh.addWidget(w)
        oh.addStretch(1)
        a.addWidget(orow)
        self.gradients = QCheckBox("Gradients")
        self.gradients.setEnabled(False)
        self.gradients.setToolTip("Not supported yet.")
        a.addWidget(self.gradients)
        self.smooth = SliderRow(0, 100, 1, " %", "Min", "Max")
        a.addWidget(_row("Smooth", self.smooth))
        self.shapes = QCheckBox("Shapes")
        self.shapes.setChecked(True)
        self.shapes.setToolTip("Recognise ellipses/circles and draw them as exact curves.")
        a.addWidget(self.shapes)
        a.addWidget(_title("Options"))
        self.snap = QCheckBox("Snap Curves To Lines")
        self.snap.setChecked(True)
        self.transp = QCheckBox("Transparency")
        self.transp.setEnabled(False)
        self.transp.setToolTip("Not supported yet.")
        a.addWidget(self.snap)
        a.addWidget(self.transp)
        ir = QWidget()
        ih = QHBoxLayout(ir)
        ih.setContentsMargins(0, 0, 0, 0)
        self.ign = QCheckBox("Ignore Color")
        self.ign_color = QColor(255, 255, 255)
        self.ign_btn = QPushButton()
        self.ign_btn.setFixedSize(54, 22)
        self._paint_swatch()
        self.ign_btn.clicked.connect(self._choose_color)
        self.eye = QPushButton("Pick")
        self.eye.setToolTip("Click, then click a colour in the image.")
        self.eye.clicked.connect(self.pick_clicked)
        ih.addWidget(self.ign)
        ih.addWidget(self.ign_btn)
        ih.addWidget(self.eye)
        ih.addStretch(1)
        a.addWidget(ir)
        self.res = QComboBox()
        for n, _ in RES:
            self.res.addItem(n)
        self.res.setCurrentIndex(1)
        a.addWidget(_row("Resolution", self.res))
        v.addWidget(self.adv)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        v.addWidget(sep)
        self.info = QLabel("Paths 0    Anchors 0    Colors 0")
        v.addWidget(self.info)
        self.group = QCheckBox("Auto Grouping")
        v.addWidget(self.group)
        pr = QWidget()
        ph = QHBoxLayout(pr)
        ph.setContentsMargins(0, 0, 0, 0)
        self.preview = QCheckBox("Preview")
        self.preview.setChecked(True)
        self.trace_btn = QPushButton("Trace")
        self.trace_btn.clicked.connect(self.trace_clicked)
        ph.addWidget(self.preview)
        ph.addWidget(self.trace_btn, 1)
        v.addWidget(pr)
        v.addStretch(1)

        self.preset.currentTextChanged.connect(self._preset_chosen)
        self.view.currentIndexChanged.connect(lambda i: self.view_changed.emit(VIEWS[i][1]))
        self.mode.currentIndexChanged.connect(self._edited)
        for w in (self.palette, self.res):
            w.currentIndexChanged.connect(self._edited)
        for w in (self.colors, self.threshold, self.paths, self.corners, self.noise, self.smooth):
            w.valueChanged.connect(self._edited)
        for w in (self.fills, self.strokes, self.shapes, self.snap, self.ign, self.group,
                  self.outlines, self.outline_fixed):
            w.toggled.connect(self._edited)
        self.outline_w.valueChanged.connect(self._edited)
        self.stroke_w.valueChanged.connect(self._edited)
        self.abut.clicked.connect(lambda: self._method("abutting"))
        self.over.clicked.connect(lambda: self._method("overlapping"))
        self._sync_visibility()

    # --- helpers --------------------------------------------------------------------
    def _toggle_adv(self, on: bool) -> None:
        self.adv.setVisible(on)
        self.adv_btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def _paint_swatch(self) -> None:
        self.ign_btn.setStyleSheet(f"background:{self.ign_color.name()};border:1px solid #666")

    def _paint_outline_swatch(self) -> None:
        self.outline_btn.setStyleSheet(f"background:{self.outline_color.name()};border:1px solid #666")

    def _choose_outline_color(self) -> None:
        dlg = QColorDialog(self.outline_color, self)
        dlg.setOption(QColorDialog.DontUseNativeDialog, True)
        if dlg.exec():
            self.outline_color = dlg.currentColor()
            self._paint_outline_swatch()
            self.outline_fixed.setChecked(True)
            self._edited()

    def set_palette(self, res) -> None:
        """Feed the palette of a fresh trace result into the replace-colours editor."""
        p = res.params
        sig = (p.mode, p.palette, p.colors, len(res.palette))
        self.palette_editor.set_palette(res.palette, used_palette_indices(res), sig)

    def palette_overrides(self):
        return self.palette_editor.overrides()

    def _choose_color(self) -> None:
        c = QColorDialog.getColor(self.ign_color, self)
        if c.isValid():
            self.set_ignore_color((c.red(), c.green(), c.blue()), enable=True)

    def set_ignore_color(self, rgb, enable: bool = True) -> None:
        self.ign_color = QColor(*rgb)
        self._paint_swatch()
        if enable:
            self.ign.setChecked(True)
        self._edited()

    def _method(self, m: str) -> None:
        self.abut.setChecked(m == "abutting")
        self.over.setChecked(m == "overlapping")
        self._edited()

    def _sync_visibility(self) -> None:
        bw = MODES[self.mode.currentIndex()][1] == "bw"
        self.threshold_row.setVisible(bw)
        self.palette_row.setVisible(not bw)
        self.colors_row.setVisible(not bw and self.palette.currentIndex() == 0)
        for w in (self.ign, self.ign_btn, self.eye):
            w.setEnabled(not bw)
        self.stroke_w.setEnabled(self.strokes.isChecked())

    def _edited(self, *_) -> None:
        if self._busy:
            return
        self._sync_visibility()
        if self.preset.currentText() != "[Custom]":
            if not self._custom_added:
                self.preset.addItem("[Custom]")
                self._custom_added = True
            with QSignalBlocker(self.preset):
                self.preset.setCurrentText("[Custom]")
        self.changed.emit()

    def _preset_chosen(self, name: str) -> None:
        if name not in PRESETS:
            return
        self.set_params(params_from_preset(name, self.params()))
        self.changed.emit()

    # --- public API -----------------------------------------------------------------
    def view_mode(self) -> str:
        return VIEWS[self.view.currentIndex()][1]

    def params(self) -> TraceParams:
        ic = self.ign_color
        return TraceParams(
            mode=MODES[self.mode.currentIndex()][1],
            palette=("limited", "automatic")[self.palette.currentIndex()],
            colors=self.colors.value(), threshold=self.threshold.value(),
            paths=self.paths.value(), corners=self.corners.value(), noise=self.noise.value(),
            method="overlapping" if self.over.isChecked() else "abutting",
            fills=self.fills.isChecked(), strokes=self.strokes.isChecked(),
            stroke_width=self.stroke_w.value(), smooth=self.smooth.value(),
            shapes=self.shapes.isChecked(), snap_lines=self.snap.isChecked(),
            outlines=self.outlines.isChecked(), outline_width=self.outline_w.value(),
            outline_fixed=((self.outline_color.red(), self.outline_color.green(), self.outline_color.blue())
                           if self.outline_fixed.isChecked() else None),
            ignore_color=(ic.red(), ic.green(), ic.blue()) if self.ign.isChecked() else None,
            auto_group=self.group.isChecked(), max_side=RES[self.res.currentIndex()][1])

    def set_params(self, p: TraceParams) -> None:
        self._busy = True
        try:
            self.mode.setCurrentIndex([m for _, m in MODES].index(p.mode))
            self.palette.setCurrentIndex(0 if p.palette == "limited" else 1)
            self.colors.set(p.colors)
            self.threshold.set(p.threshold)
            self.paths.set(p.paths)
            self.corners.set(p.corners)
            self.noise.set(p.noise)
            self.abut.setChecked(p.method == "abutting")
            self.over.setChecked(p.method == "overlapping")
            self.fills.setChecked(p.fills)
            self.strokes.setChecked(p.strokes)
            self.stroke_w.setValue(p.stroke_width)
            self.smooth.set(p.smooth)
            self.shapes.setChecked(p.shapes)
            self.snap.setChecked(p.snap_lines)
            self.outlines.setChecked(p.outlines)
            self.outline_w.setValue(p.outline_width)
        finally:
            self._busy = False
        self._sync_visibility()

    def set_info(self, paths: int, anchors: int, colors: int, seconds: Optional[float] = None) -> None:
        t = f"    {seconds:.2f} s" if seconds is not None else ""
        self.info.setText(f"Paths {paths:,}    Anchors {anchors:,}    Colors {colors}{t}")
