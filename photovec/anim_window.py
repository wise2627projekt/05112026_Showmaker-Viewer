"""Separate window: before/after slider(s), animation preview, GIF / MP4 export."""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QElapsedTimer, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressDialog,
                               QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from .animate import (SIDES_SINGLE, AnimSettings, compose, export_gif, export_mp4, frame_count,
                      slider_positions)

Provider = Callable[[str, str], Optional[Tuple[np.ndarray, np.ndarray]]]


def _qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    return QImage(np.ascontiguousarray(bgr).data, w, h, 3 * w, QImage.Format_BGR888).copy()


class CompareCanvas(QWidget):
    moved = Signal(float, float)          # px, py from dragging on the image

    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 420)
        self.before: Optional[np.ndarray] = None
        self.after: Optional[np.ndarray] = None
        self.s = AnimSettings()
        self.px, self.py = 0.5, 0.5
        self._img: Optional[QImage] = None

    def set_images(self, before: np.ndarray, after: np.ndarray, max_w: int = 1100) -> None:
        h, w = before.shape[:2]
        k = min(1.0, max_w / w)
        if k < 1.0:
            before = cv2.resize(before, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
        self.before, self.after = before, after
        self.redraw()

    def redraw(self) -> None:
        if self.before is not None:
            self._img = _qimage(compose(self.before, self.after, self.px, self.py, self.s))
        self.update()

    def _rect(self) -> QRect:
        if self._img is None:
            return QRect()
        k = min(self.width() / self._img.width(), self.height() / self._img.height())
        w, h = int(self._img.width() * k), int(self._img.height() * k)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self._img is None:
            p.setPen(QColor(170, 170, 175))
            p.drawText(self.rect(), Qt.AlignCenter, "Trace an image first,\nthen press “Refresh”.")
            return
        p.drawImage(self._rect(), self._img)

    def _drag(self, e) -> None:
        r = self._rect()
        if r.isNull() or r.width() == 0 or r.height() == 0:
            return
        pt = e.position().toPoint()
        fx = min(1.0, max(0.0, (pt.x() - r.left()) / r.width()))
        fy = min(1.0, max(0.0, (pt.y() - r.top()) / r.height()))
        s = self.s
        px = fx if s.side_x == "left" else 1.0 - fx
        py = fy if s.side_y == "top" else 1.0 - fy
        if s.mode == "single":
            px = px if s.axis == "x" else py
            py = self.py
        self.moved.emit(px, py)

    def mousePressEvent(self, e):
        self._drag(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._drag(e)


class AnimWindow(QWidget):
    def __init__(self, provider: Provider, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Compare & Animate — before / after")
        self.resize(1180, 760)
        self.provider = provider
        self.canvas = CompareCanvas()
        self.canvas.moved.connect(self._on_drag)
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._tick)
        self.clock = QElapsedTimer()
        self._busy = False

        side = QVBoxLayout()
        side.addWidget(self._view_box())
        side.addWidget(self._text_box())
        side.addWidget(self._anim_box())
        side.addWidget(self._export_box())
        side.addStretch(1)
        host = QWidget()
        host.setLayout(side)
        host.setFixedWidth(360)
        root = QHBoxLayout(self)
        root.addWidget(self.canvas, 1)
        root.addWidget(host)
        self._sync()
        self.refresh()

    # --- UI ---------------------------------------------------------------------------
    def _view_box(self) -> QGroupBox:
        g = QGroupBox("Slider")
        f = QFormLayout(g)
        self.mode = QComboBox()
        self.mode.addItems(["One slider", "Two sliders (4 quadrants)"])
        self.dir = QComboBox()
        self.dir.addItems(list(SIDES_SINGLE))
        self.dir.setCurrentIndex(0)
        self.side_x = QComboBox()
        self.side_x.addItems(["After enters from the right", "After enters from the left"])
        self.side_y = QComboBox()
        self.side_y.addItems(["After enters from the bottom", "After enters from the top"])
        self.sl_x = QSlider(Qt.Horizontal)
        self.sl_y = QSlider(Qt.Horizontal)
        for sl in (self.sl_x, self.sl_y):
            sl.setRange(0, 1000)
            sl.setValue(500)
            sl.valueChanged.connect(self._slider_moved)
        self.lbl_x = QLabel("Position")
        self.lbl_y = QLabel("Slider 2")
        f.addRow("Mode", self.mode)
        f.addRow("Direction", self.dir)
        f.addRow("Slider 1", self.side_x)
        f.addRow("Slider 2", self.side_y)
        f.addRow(self.lbl_x, self.sl_x)
        f.addRow(self.lbl_y, self.sl_y)
        self.divider = QCheckBox("Show divider line(s)")
        self.divider.setChecked(True)
        self.bg = QComboBox()
        self.bg.addItems(["White", "Checkerboard", "Dark"])
        self.after_view = QComboBox()
        self.after_view.addItems(["Tracing result", "Outlines only"])
        f.addRow(self.divider)
        f.addRow("Background", self.bg)
        f.addRow("After image", self.after_view)
        for w in (self.mode, self.dir, self.side_x, self.side_y, self.divider):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.toggled).connect(self._settings_changed)
        for w in (self.bg, self.after_view):
            w.currentIndexChanged.connect(self.refresh)
        return g

    def _text_box(self) -> QGroupBox:
        g = QGroupBox("Labels")
        f = QFormLayout(g)
        self.show_text = QCheckBox("Show BEFORE / AFTER text")
        self.show_text.setChecked(True)
        self.txt_before = QLineEdit("BEFORE")
        self.txt_after = QLineEdit("AFTER")
        f.addRow(self.show_text)
        f.addRow("Before", self.txt_before)
        f.addRow("After", self.txt_after)
        self.show_text.toggled.connect(self._settings_changed)
        self.txt_before.textChanged.connect(self._settings_changed)
        self.txt_after.textChanged.connect(self._settings_changed)
        return g

    def _anim_box(self) -> QGroupBox:
        g = QGroupBox("Animation")
        f = QFormLayout(g)
        self.duration = QDoubleSpinBox(); self.duration.setRange(0.3, 60); self.duration.setValue(3.0); self.duration.setSuffix(" s")
        self.hold_a = QDoubleSpinBox(); self.hold_a.setRange(0, 30); self.hold_a.setValue(0.5); self.hold_a.setSuffix(" s")
        self.hold_b = QDoubleSpinBox(); self.hold_b.setRange(0, 30); self.hold_b.setValue(0.8); self.hold_b.setSuffix(" s")
        self.fps = QSpinBox(); self.fps.setRange(5, 60); self.fps.setValue(25)
        self.pingpong = QCheckBox("Ping-pong (before → after → before)")
        self.easing = QComboBox(); self.easing.addItems(["Smooth (ease in/out)", "Linear"])
        self.ypeak = QDoubleSpinBox(); self.ypeak.setRange(0, 1); self.ypeak.setSingleStep(0.1); self.ypeak.setValue(0.5)
        self.ypeak.setToolTip("Two-slider mode: how far the second slider travels while the first one sweeps.")
        self.play = QPushButton("▶  Play preview")
        self.play.clicked.connect(self._toggle_play)
        for lbl, w in (("Motion", self.duration), ("Hold start", self.hold_a), ("Hold end", self.hold_b),
                       ("Frames / s", self.fps)):
            f.addRow(lbl, w)
        f.addRow(self.pingpong)
        f.addRow("Easing", self.easing)
        f.addRow("2nd slider travel", self.ypeak)
        f.addRow(self.play)
        for w in (self.duration, self.hold_a, self.hold_b, self.fps, self.ypeak):
            w.valueChanged.connect(self._settings_changed)
        self.pingpong.toggled.connect(self._settings_changed)
        self.easing.currentIndexChanged.connect(self._settings_changed)
        return g

    def _export_box(self) -> QGroupBox:
        g = QGroupBox("Export")
        v = QVBoxLayout(g)
        r = QHBoxLayout()
        r.addWidget(QLabel("Max width"))
        self.max_w = QSpinBox(); self.max_w.setRange(240, 3840); self.max_w.setValue(1080); self.max_w.setSuffix(" px")
        r.addWidget(self.max_w)
        v.addLayout(r)
        b = QHBoxLayout()
        self.gif_btn = QPushButton("Export GIF…")
        self.mp4_btn = QPushButton("Export MP4…")
        self.gif_btn.clicked.connect(lambda: self._export("gif"))
        self.mp4_btn.clicked.connect(lambda: self._export("mp4"))
        b.addWidget(self.gif_btn)
        b.addWidget(self.mp4_btn)
        v.addLayout(b)
        self.refresh_btn = QPushButton("Refresh from main window")
        self.refresh_btn.clicked.connect(self.refresh)
        v.addWidget(self.refresh_btn)
        return g

    # --- state ------------------------------------------------------------------------
    def settings(self) -> AnimSettings:
        quad = self.mode.currentIndex() == 1
        axis, side = list(SIDES_SINGLE.values())[self.dir.currentIndex()]
        sx = "right" if self.side_x.currentIndex() == 0 else "left"
        sy = "bottom" if self.side_y.currentIndex() == 0 else "top"
        if not quad:
            sx, sy = (side, sy) if axis == "x" else (sx, side)
        return AnimSettings(
            mode="quad" if quad else "single", axis=axis, side_x=sx, side_y=sy,
            show_text=self.show_text.isChecked(), text_before=self.txt_before.text(),
            text_after=self.txt_after.text(), divider=self.divider.isChecked(),
            duration=self.duration.value(), hold_start=self.hold_a.value(), hold_end=self.hold_b.value(),
            fps=self.fps.value(), pingpong=self.pingpong.isChecked(),
            easing="smooth" if self.easing.currentIndex() == 0 else "linear", ypeak=self.ypeak.value(),
            max_width=self.max_w.value())

    def _sync(self) -> None:
        quad = self.mode.currentIndex() == 1
        self.dir.setEnabled(not quad)
        self.side_x.setEnabled(quad)
        self.side_y.setEnabled(quad)
        self.sl_y.setEnabled(quad)
        self.lbl_x.setText("Slider 1" if quad else "Position")
        self.ypeak.setEnabled(quad)
        self.canvas.s = self.settings()

    def _settings_changed(self, *_) -> None:
        self._sync()
        self.canvas.redraw()

    def _slider_moved(self, *_) -> None:
        if self._busy:
            return
        self._stop()
        self.canvas.px, self.canvas.py = self.sl_x.value() / 1000.0, self.sl_y.value() / 1000.0
        self.canvas.redraw()

    def _on_drag(self, px: float, py: float) -> None:
        self._stop()
        self._set_positions(px, py)

    def _set_positions(self, px: float, py: float) -> None:
        self._busy = True
        self.sl_x.setValue(int(round(px * 1000)))
        self.sl_y.setValue(int(round(py * 1000)))
        self._busy = False
        self.canvas.px, self.canvas.py = px, py
        self.canvas.redraw()

    def refresh(self) -> None:
        pair = self.provider(("white", "checker", "dark")[self.bg.currentIndex()],
                             "result" if self.after_view.currentIndex() == 0 else "outline")
        if pair is None:
            return
        self.canvas.set_images(*pair)

    # --- playback ---------------------------------------------------------------------
    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self._stop()
        else:
            self.clock.start()
            self.timer.start()
            self.play.setText("■  Stop")

    def _stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("▶  Play preview")

    def _tick(self) -> None:
        s = self.settings()
        n = frame_count(s)
        i = int(self.clock.elapsed() / 1000.0 * s.fps) % (n + int(s.fps * 0.4))
        px, py = slider_positions(min(i, n - 1), n, s)
        self._set_positions(px, py)

    # --- export -----------------------------------------------------------------------
    def _export(self, kind: str) -> None:
        pair = self.provider(("white", "checker", "dark")[self.bg.currentIndex()],
                             "result" if self.after_view.currentIndex() == 0 else "outline")
        if pair is None:
            QMessageBox.information(self, "Nothing to export", "Trace an image first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, f"Save {kind.upper()}", os.path.expanduser(f"~/before_after.{kind}"),
                                             f"{kind.upper()} (*.{kind})")
        if not out:
            return
        s = self.settings()
        dlg = QProgressDialog(f"Rendering {kind.upper()}…", "Cancel", 0, frame_count(s), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(i: int, n: int) -> bool:
            dlg.setValue(i)
            QApplication.processEvents()
            return dlg.wasCanceled()

        try:
            (export_gif if kind == "gif" else export_mp4)(out, pair[0], pair[1], s, progress)
        except Exception as exc:
            dlg.close()
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        dlg.close()
        if os.path.exists(out) and not dlg.wasCanceled():
            QMessageBox.information(self, "Saved", f"{out}\n{os.path.getsize(out) / 1024:.0f} KB")

    def closeEvent(self, e):
        self._stop()
        e.accept()
