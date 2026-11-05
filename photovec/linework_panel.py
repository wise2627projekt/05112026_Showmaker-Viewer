"""Linework Snap panel (raster line drawing -> snapped vector paths)."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from .linework import PRESETS, LineworkParams
from .trace_panel import VIEWS, SliderRow, _row, _title

WIDTH_MODES = [("Match source (per path)", "match"), ("Uniform (median)", "uniform"), ("Fixed", "fixed")]
COLOR_MODES = [("Black", "black"), ("Source colours", "source")]
RES = [("Standard (4000 px)", 4000), ("Full resolution", 0), ("Fast (2400 px)", 2400)]


class LineworkPanel(QWidget):
    changed = Signal()
    view_changed = Signal(str)
    trace_clicked = Signal()
    paste_clicked = Signal()

    def __init__(self):
        super().__init__()
        self._busy = False
        self._custom = False
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        hint = QLabel("Paste (Ctrl+V) or drop a PNG that contains only lines. Paths follow the "
                      "centre of every line; junctions and loose ends are snapped together.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#777;font-size:11px")
        v.addWidget(hint)
        self.paste_btn = QPushButton("Paste image  (Ctrl+V)")
        self.paste_btn.clicked.connect(self.paste_clicked)
        v.addWidget(self.paste_btn)
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.view = QComboBox()
        for n, _ in VIEWS:
            self.view.addItem(n)
        v.addWidget(_row("Preset", self.preset))
        v.addWidget(_row("View", self.view))

        v.addWidget(_title("Line detection"))
        tr = QWidget()
        th = QHBoxLayout(tr)
        th.setContentsMargins(0, 0, 0, 0)
        self.auto = QCheckBox("Auto")
        self.auto.setChecked(True)
        self.threshold = SliderRow(10, 250, 90, "", "More", "Fewer")
        self.threshold.setEnabled(False)
        th.addWidget(self.auto)
        th.addWidget(self.threshold, 1)
        v.addWidget(_row("Threshold", tr))
        self.min_len = SliderRow(0, 80, 6, " px", "0", "80")
        self.min_len.setToolTip("Specks and dead-end spurs shorter than this are removed.")
        self.gap = SliderRow(0, 40, 3, " px", "0", "40")
        self.gap.setToolTip("Close gaps between collinear lines and snap loose ends onto nearby lines.")
        v.addWidget(_row("Min length", self.min_len))
        v.addWidget(_row("Gap snap", self.gap))

        v.addWidget(_title("Simplify paths"))
        self.simplify = SliderRow(0, 100, 30, " %", "Exact", "Simple")
        self.simplify.setToolTip("Higher = fewer nodes and looser fit to the drawn lines.")
        self.corners = SliderRow(0, 100, 60, " %", "Less", "More")
        v.addWidget(_row("Simplify", self.simplify))
        v.addWidget(_row("Corners", self.corners))
        self.join = QCheckBox("Join lines through junctions")
        self.join.setChecked(True)
        self.snap = QCheckBox("Snap curves to straight lines")
        self.snap.setChecked(True)
        v.addWidget(self.join)
        v.addWidget(self.snap)

        v.addWidget(_title("Output"))
        self.width_mode = QComboBox()
        for n, _ in WIDTH_MODES:
            self.width_mode.addItem(n)
        self.fixed = QDoubleSpinBox()
        self.fixed.setRange(0.1, 50)
        self.fixed.setValue(1.0)
        self.fixed.setSuffix(" px")
        self.fixed.setEnabled(False)
        wr = QWidget()
        wh = QHBoxLayout(wr)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.width_mode, 1)
        wh.addWidget(self.fixed)
        v.addWidget(_row("Stroke", wr))
        self.color_mode = QComboBox()
        for n, _ in COLOR_MODES:
            self.color_mode.addItem(n)
        v.addWidget(_row("Colour", self.color_mode))
        self.res = QComboBox()
        for n, _ in RES:
            self.res.addItem(n)
        v.addWidget(_row("Resolution", self.res))

        self.info = QLabel("Paths 0    Anchors 0")
        v.addWidget(self.info)
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
        self.auto.toggled.connect(lambda on: (self.threshold.setEnabled(not on), self._edited()))
        for w in (self.threshold, self.min_len, self.gap, self.simplify, self.corners):
            w.valueChanged.connect(self._edited)
        for w in (self.join, self.snap):
            w.toggled.connect(self._edited)
        for w in (self.width_mode, self.color_mode, self.res):
            w.currentIndexChanged.connect(self._edited)
        self.fixed.valueChanged.connect(self._edited)

    def view_mode(self) -> str:
        return VIEWS[self.view.currentIndex()][1]

    def _edited(self, *_) -> None:
        if self._busy:
            return
        self.fixed.setEnabled(WIDTH_MODES[self.width_mode.currentIndex()][1] == "fixed")
        if self.preset.currentText() != "[Custom]":
            if not self._custom:
                self.preset.addItem("[Custom]")
                self._custom = True
            with QSignalBlocker(self.preset):
                self.preset.setCurrentText("[Custom]")
        self.changed.emit()

    def _preset_chosen(self, name: str) -> None:
        if name in PRESETS:
            base = self.params()
            vals = {**{k: getattr(LineworkParams(), k) for k in ("simplify", "min_len", "gap", "join_through", "corners")},
                    **PRESETS[name]}
            self.set_params(LineworkParams(**{**base.__dict__, **vals}))
            self.changed.emit()

    def params(self) -> LineworkParams:
        return LineworkParams(
            threshold=0 if self.auto.isChecked() else self.threshold.value(),
            min_len=float(self.min_len.value()), gap=float(self.gap.value()),
            simplify=self.simplify.value(), corners=self.corners.value(),
            join_through=self.join.isChecked(), snap_lines=self.snap.isChecked(),
            width_mode=WIDTH_MODES[self.width_mode.currentIndex()][1], fixed_width=self.fixed.value(),
            color_mode=COLOR_MODES[self.color_mode.currentIndex()][1],
            max_side=RES[self.res.currentIndex()][1])

    def set_params(self, p: LineworkParams) -> None:
        self._busy = True
        try:
            self.min_len.set(int(p.min_len))
            self.gap.set(int(p.gap))
            self.simplify.set(p.simplify)
            self.corners.set(p.corners)
            self.join.setChecked(p.join_through)
            self.snap.setChecked(p.snap_lines)
        finally:
            self._busy = False

    def set_info(self, paths: int, anchors: int, raw: Optional[int] = None, seconds: Optional[float] = None) -> None:
        r = f"{raw:,} → " if raw is not None else ""
        t = f"    {seconds:.2f} s" if seconds is not None else ""
        self.info.setText(f"Paths {r}{paths:,}    Anchors {anchors:,}{t}")
