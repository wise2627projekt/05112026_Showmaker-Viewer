"""Scrub window (opened from Multi-Trace): all frames of ONE photo are prerendered once, then you slide through them freely.

    * GRAFFITI slider: the Threshold (From -> To) of the masked graffiti. Only the masked part is vectorised (the tracer never sees the wall).
    * FACADE slider (optional): a second, independent vectoriser for everything that is NOT masked, with its own Threshold range, Noise
      and number of frames. It is its own layer: the photo without the mask, vectorised on its own.
    * The background photo sits below both vectors. It can FADE WITH A SLIDER (from a level at one slider position to a level at another,
      following the graffiti or the facade slider) or STAY PERMANENTLY at a level of your choice.
    * Layers are laid over each other when a frame is shown, so moving a slider or changing the background is instant - nothing is
      traced again. Without a mask there is nothing to separate: the window scrubs the plain vector frames.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from .layers import composite_scrub, facade_region, traced_region
from .multi import MultiSettings, measure_source, render_preview, tile_colors


def background_level(position: float, mode: str, start: float, end: float, at_a: float, at_b: float, stay: float) -> float:
    """How much of the background photo is visible (0..1) at a slider position (0..1).
    mode "fade": `start` (%) up to slider position `at_a`, then moving to `end` (%) at `at_b` (linear), then staying there.
    mode "stay": always `stay` (%)."""
    if mode != "fade":
        return min(max(stay / 100.0, 0.0), 1.0)
    a, b = min(at_a, at_b), max(at_a, at_b)
    u = 0.0 if position <= a else 1.0 if position >= b or b <= a else (position - a) / (b - a)
    return min(max((start + (end - start) * u) / 100.0, 0.0), 1.0)


def scrub_values(thr_from: float, thr_to: float, n: int, k: int) -> float:
    return thr_from + (thr_to - thr_from) * (k / max(1, n - 1))


class _Sig(QObject):
    done = Signal(int, int, int)           # layer (0 graffiti, 1 facade), frame index, generation


def _spin(lo, hi, val=0, dec=None, suffix=""):
    w = QDoubleSpinBox() if dec is not None else QSpinBox()
    if dec is not None:
        w.setDecimals(dec)
    w.setRange(lo, hi)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class ScrubWindow(QWidget):
    def __init__(self, panels: list, st: MultiSettings, parent=None, autostart: bool = True):
        """panels: list of (title, SweepSource, PanelSpec, base_opaque or None) of the loaded photos."""
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Scrub - prerendered frames")
        self.resize(1100, 940)
        self.st = st
        self.entries = panels
        self.frames: List[Optional[tuple]] = []          # graffiti: (vector picture, ink uint8 or None)
        self.ffr: List[Optional[tuple]] = []             # facade:   (vector picture, ink uint8 or None)
        self.empty = (255, 255, 255)
        self.gen = 0
        self._prerenders: list = []
        self.src = None
        self.sig = _Sig()
        self.sig.done.connect(self._frame_done)
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.current: Optional[np.ndarray] = None
        self.stale = False

        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Photo:"))
        self.photo = QComboBox()
        for t, *_ in panels:
            self.photo.addItem(t)
        self.photo.currentIndexChanged.connect(self._photo_changed)
        top.addWidget(self.photo)
        top.addWidget(QLabel("<b>Graffiti</b> (masked)  Threshold from"))
        self.t0 = _spin(0, 255, 0, 1)
        self.t1 = _spin(0, 255, 0, 1)
        top.addWidget(self.t0)
        top.addWidget(QLabel("to"))
        top.addWidget(self.t1)
        top.addWidget(QLabel("Noise"))
        self.noise = _spin(0, 100)
        top.addWidget(self.noise)
        top.addWidget(QLabel("Frames"))
        self.count = _spin(2, 400, 60)
        top.addWidget(self.count)
        top.addWidget(QLabel("Size"))
        self.size = _spin(240, 2000, 720, suffix=" px")
        self.size.setSingleStep(80)
        self.size.setToolTip("Long side of the prerendered frames (memory: about frames x size² x 4 bytes, for both layers).")
        top.addWidget(self.size)
        self.go = QPushButton("Prerender")
        self.go.clicked.connect(self.start)
        top.addWidget(self.go)
        top.addStretch(1)
        v.addLayout(top)

        fr = QHBoxLayout()
        self.fac_on = QCheckBox("Vectorise the facade independently")
        self.fac_on.setToolTip("A second layer: everything that is NOT masked (the photo without the mask) is vectorised on its own, with its own\n"
                               "Threshold slider. The graffiti keeps its slider. Both are laid over the background photo.")
        self.fac_on.toggled.connect(self._facade_toggled)
        fr.addWidget(self.fac_on)
        self.f_lbl = [QLabel("Threshold from"), QLabel("to"), QLabel("Noise"), QLabel("Frames")]
        self.f0 = _spin(0, 255, 0, 1)
        self.f1 = _spin(0, 255, 0, 1)
        self.fnoise = _spin(0, 100)
        self.fcount = _spin(2, 400, 40)
        for lab, w in zip(self.f_lbl, (self.f0, self.f1, self.fnoise, self.fcount)):
            fr.addWidget(lab)
            fr.addWidget(w)
        fr.addStretch(1)
        v.addLayout(fr)
        for w in (self.t0, self.t1, self.noise, self.count, self.size, self.f0, self.f1, self.fnoise, self.fcount):
            w.valueChanged.connect(self._mark_stale)

        self.view = QLabel("")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(320, 320)
        self.view.setStyleSheet("background:#222;")
        v.addWidget(self.view, 1)
        self.prog = QProgressBar()
        v.addWidget(self.prog)

        row = QHBoxLayout()
        self.play = QPushButton("▶ Play")
        self.play.setCheckable(True)
        self.play.setToolTip("Moves both sliders from start to end together.")
        self.play.toggled.connect(self._toggle_play)
        row.addWidget(self.play)
        row.addWidget(QLabel("Graffiti"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 59)
        self.slider.valueChanged.connect(self._show)
        row.addWidget(self.slider, 1)
        self.info = QLabel("")
        self.info.setMinimumWidth(300)
        row.addWidget(self.info)
        v.addLayout(row)

        row2 = QHBoxLayout()
        self.fl_lbl = QLabel("Facade")
        row2.addSpacing(self.play.sizeHint().width() + 6)
        row2.addWidget(self.fl_lbl)
        self.fslider = QSlider(Qt.Horizontal)
        self.fslider.setRange(0, 39)
        self.fslider.valueChanged.connect(self._show)
        row2.addWidget(self.fslider, 1)
        self.finfo = QLabel("")
        self.finfo.setMinimumWidth(300)
        row2.addWidget(self.finfo)
        v.addLayout(row2)

        bg = QHBoxLayout()
        bg.addWidget(QLabel("Background (the photo):"))
        self.mode = QComboBox()
        self.mode.addItems(["Fades with a slider", "Stays permanently"])
        self.mode.setToolTip("Fades: the background photo goes from one level to another while you slide.\n"
                             "Stays permanently: it stays at the level you choose, whatever the sliders do.")
        self.mode.currentIndexChanged.connect(self._bg_changed)
        bg.addWidget(self.mode)
        self.follow = QComboBox()
        self.follow.addItems(["graffiti slider", "facade slider"])
        self.follow.setToolTip("Which slider the fade follows.")
        self.follow.currentIndexChanged.connect(self._bg_changed)
        self.lv0 = _spin(0, 100, 100, suffix=" %")
        self.lv1 = _spin(0, 100, 0, suffix=" %")
        self.pa = _spin(0, 100, 0, suffix=" %")
        self.pb = _spin(0, 100, 100, suffix=" %")
        self.l_fade = [self.follow, QLabel("from"), self.lv0, QLabel("to"), self.lv1, QLabel("between slider"), self.pa, QLabel("and"), self.pb]
        for w in self.l_fade:
            bg.addWidget(w)
        self.stay = QSlider(Qt.Horizontal)
        self.stay.setRange(0, 100)
        self.stay.setValue(100)
        self.stay.setMaximumWidth(260)
        self.stay_lbl = QLabel("100 %")
        bg.addWidget(self.stay)
        bg.addWidget(self.stay_lbl)
        bg.addStretch(1)
        self.save = QPushButton("Save this frame…")
        self.save.clicked.connect(self._save_frame)
        bg.addWidget(self.save)
        v.addLayout(bg)
        for w in (self.lv0, self.lv1, self.pa, self.pb):
            w.valueChanged.connect(self._bg_changed)
        self.stay.valueChanged.connect(self._bg_changed)
        self.note = QLabel("")
        self.note.setStyleSheet("color:#888;")
        v.addWidget(self.note)

        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._tick)
        self._photo_changed(0, start=False)
        if autostart and self.entries:
            self.start()

    # ------------------------------------------------------------------ setup
    def _entry(self):
        return self.entries[self.photo.currentIndex()] if self.entries else None

    def _has_mask(self) -> bool:
        e = self._entry()
        return e is not None and e[1].mask is not None

    def _photo_changed(self, i: int, start: bool = True) -> None:
        e = self._entry()
        if e is None:
            return
        _, src, spec, _ = e
        for w, val in ((self.t0, spec.thr[0]), (self.t1, spec.thr[-1]), (self.noise, spec.noise[0]),
                       (self.f0, spec.thr[0]), (self.f1, spec.thr[-1]), (self.fnoise, spec.noise[0])):
            w.blockSignals(True)
            w.setValue(int(round(val)) if w in (self.noise, self.fnoise) else float(val))
            w.blockSignals(False)
        self.fac_on.setEnabled(src.mask is not None)
        if src.mask is None:
            self.fac_on.setChecked(False)
        self._bg_changed()
        if start:
            self.start()

    def _facade_toggled(self, on: bool) -> None:
        self._mark_stale()
        self._bg_changed()

    def _mark_stale(self, *_) -> None:
        self.stale = True
        self.go.setText("Prerender (settings changed)")

    def _bg_changed(self, *_) -> None:
        fade = self.mode.currentIndex() == 0
        fac = self.fac_on.isChecked()
        for w in self.l_fade:
            w.setVisible(fade)
        self.follow.setEnabled(fac)
        if not fac and self.follow.currentIndex() == 1:
            self.follow.blockSignals(True)
            self.follow.setCurrentIndex(0)
            self.follow.blockSignals(False)
        self.stay.setVisible(not fade)
        self.stay_lbl.setVisible(not fade)
        self.stay_lbl.setText(f"{self.stay.value()} %")
        for w in (self.f0, self.f1, self.fnoise, self.fcount, self.fslider, self.fl_lbl, self.finfo, *self.f_lbl):
            w.setEnabled(fac)
            w.setVisible(True)
        has_mask = self._has_mask()
        for w in (self.mode, self.lv0, self.lv1, self.pa, self.pb, self.stay):
            w.setEnabled(has_mask)
        self.note.setText("" if has_mask else "This photo has no mask - draw one in Multi-Trace (Edit mask…) to keep a background photo and to "
                                              "vectorise graffiti and facade separately; the frames are plain vectors.")
        self._show()

    # ------------------------------------------------------------------ prerender
    def start(self) -> None:
        e = self._entry()
        if e is None:
            return
        self.gen += 1
        gen = self.gen
        _, src, spec, base = e
        long_side = int(self.size.value())
        fac = self.fac_on.isChecked() and src.mask is not None
        gsrc = replace(src, opaque=traced_region(base, src.mask, src.bgr.shape)) if src.mask is not None else src   # only the masked graffiti
        self.src = measure_source(gsrc, long_side)
        fsrc = measure_source(replace(src, opaque=facade_region(base, src.mask, src.bgr.shape)), long_side) if fac else None
        n, nf = int(self.count.value()), int(self.fcount.value()) if fac else 0
        self.frames = [None] * n
        self.ffr = [None] * nf
        for sl, cnt in ((self.slider, n), (self.fslider, max(2, nf))):
            sl.blockSignals(True)
            sl.setRange(0, cnt - 1)
            sl.setValue(min(sl.value(), cnt - 1))
            sl.blockSignals(False)
        self.prog.setRange(0, n + nf)
        self.prog.setValue(0)
        self.stale = False
        self.go.setText("Prerender")
        ink, paper = tile_colors(spec, self.st)
        paths = None if spec.paths is None else float(spec.paths[0])
        layers = [(0, self.src, self.frames, float(self.t0.value()), float(self.t1.value()), float(self.noise.value()))]
        if fac:
            layers.append((1, fsrc, self.ffr, float(self.f0.value()), float(self.f1.value()), float(self.fnoise.value())))

        def one(layer: int, lsrc, store: list, t0: float, t1: float, noise: float, cache: dict, k: int) -> None:
            if gen != self.gen:
                return
            vals = {"threshold": scrub_values(t0, t1, len(store), k), "noise": noise, "layer_p": 1.0, "layer_g": 1.0, "layer_f": 1.0}
            if paths is not None:
                vals["paths"] = paths
            try:
                vec, a, empty = render_preview(lsrc, self.st, vals, cache, spec.flip, ink, paper, spec.fill_over, parts=True)
            except Exception as exc:
                print("Scrub: frame failed", exc)
                return
            if gen != self.gen:
                return
            store[k] = (vec, None if a is None else np.clip(a * 255.0, 0, 255).astype(np.uint8))
            self.empty = empty
            self.sig.done.emit(layer, k, gen)

        self._close_prerenders()
        from .prerender import Prerender
        from .multi import styled_source
        for layer, lsrc, store, t0, t1, noise in layers:
            cache: dict = {}
            m = len(store)
            order = [0, m - 1] + sorted(range(1, m - 1), key=lambda k: (k * 7) % m)       # ends first, then spread out so the slider works early

            def values(k: int, t0=t0, t1=t1, noise=noise, n=m) -> dict:
                vals = {"threshold": scrub_values(t0, t1, n, k), "noise": noise, "layer_p": 1.0, "layer_g": 1.0, "layer_f": 1.0}
                if paths is not None:
                    vals["paths"] = paths
                return vals

            def got(k: int, vec, a, empty, layer=layer, store=store) -> None:
                if gen != self.gen:
                    return
                store[k] = (vec, None if a is None else np.clip(a * 255.0, 0, 255).astype(np.uint8))
                self.empty = empty
                self.sig.done.emit(layer, k, gen)

            def by_thread(ks, layer=layer, lsrc=lsrc, store=store, t0=t0, t1=t1, noise=noise, cache=cache) -> None:
                for k in ks:
                    self.pool.submit(one, layer, lsrc, store, t0, t1, noise, cache, k)

            pre = Prerender(styled_source(lsrc, self.st, spec.flip, ink, paper, spec.fill_over), self.st.bg)
            if pre.usable:                                       # worker processes: far faster than threads (the tracer is Python code)
                self._prerenders.append(pre)
                if pre.submit([(k, values(k)) for k in order], lambda k, vec, a, e, got=got: got(k, vec, a, e),
                              lambda ks, by_thread=by_thread: by_thread(ks)):
                    continue
            by_thread(order)

    def _close_prerenders(self) -> None:
        for p in self._prerenders:
            p.close()
        self._prerenders = []

    def _frame_done(self, layer: int, k: int, gen: int) -> None:
        if gen != self.gen:
            return
        self.prog.setValue(sum(f is not None for f in self.frames) + sum(f is not None for f in self.ffr))
        if self.current is None or (k == (self.slider.value() if layer == 0 else self.fslider.value())):
            self._show()

    def ready(self) -> bool:
        return bool(self.frames) and all(f is not None for f in self.frames) and all(f is not None for f in self.ffr)

    # ------------------------------------------------------------------ showing a frame
    def _pos(self, slider: QSlider, k: Optional[int] = None) -> float:
        return (slider.value() if k is None else k) / max(1, slider.maximum())

    def position(self) -> float:
        """Slider position (0..1) the background fade follows."""
        if self.follow.currentIndex() == 1 and self.fac_on.isChecked():
            return self._pos(self.fslider)
        return self._pos(self.slider)

    def level(self) -> float:
        return background_level(self.position(), "fade" if self.mode.currentIndex() == 0 else "stay", self.lv0.value(), self.lv1.value(),
                                self.pa.value() / 100.0, self.pb.value() / 100.0, self.stay.value())

    @staticmethod
    def _nearest(store: list, k: int):
        if not store:
            return None
        k = min(max(k, 0), len(store) - 1)
        if store[k] is not None:
            return store[k]
        done = [i for i, x in enumerate(store) if x is not None]
        return store[min(done, key=lambda i: abs(i - k))] if done else None

    def compose(self, k: Optional[int] = None, kf: Optional[int] = None) -> Optional[np.ndarray]:
        """The picture for graffiti frame k and facade frame kf (default: the sliders), the background at its current level (BGR)."""
        if not self.frames or self.src is None:
            return None
        k = self.slider.value() if k is None else k
        g = self._nearest(self.frames, k)
        if g is None:
            return None
        if self.src.mask is None:
            out = g[0]
        else:
            f = None
            if self.ffr:
                f = self._nearest(self.ffr, self.fslider.value() if kf is None else kf)
            if self.follow.currentIndex() == 1 and self.fac_on.isChecked():
                pos = self._pos(self.fslider, kf)
            else:
                pos = self._pos(self.slider, k)
            fv = background_level(pos, "fade" if self.mode.currentIndex() == 0 else "stay", self.lv0.value(), self.lv1.value(),
                                  self.pa.value() / 100.0, self.pb.value() / 100.0, self.stay.value())
            out = composite_scrub(self.src.bgr, self.src.mask, self.empty, fv, g, f)
        return 255 - out if self.st.invert else out

    def _show(self, *_) -> None:
        k = self.slider.value()
        n = len(self.frames)
        if n:
            thr = scrub_values(self.t0.value(), self.t1.value(), n, k)
            lv = "" if (self.src is None or self.src.mask is None) else f"   ·   background {self.level() * 100:.0f} %"
            self.info.setText(f"Threshold {thr:.1f}   ·   frame {k + 1}/{n}{lv}")
        nf = len(self.ffr)
        self.finfo.setText(f"Threshold {scrub_values(self.f0.value(), self.f1.value(), nf, self.fslider.value()):.1f}   ·   frame {self.fslider.value() + 1}/{nf}"
                           if nf and self.fac_on.isChecked() else "")
        img = self.compose()
        if img is None:
            return
        self.current = img
        h, w = img.shape[:2]
        qi = QImage(np.ascontiguousarray(img[..., ::-1]).data, w, h, 3 * w, QImage.Format_RGB888).copy()
        box = self.view.size()
        self.view.setPixmap(QPixmap.fromImage(qi).scaled(max(16, box.width() - 4), max(16, box.height() - 4), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.frames:
            self._show()

    def _toggle_play(self, on: bool) -> None:
        self.play.setText("⏸ Pause" if on else "▶ Play")
        self.timer.start() if on else self.timer.stop()

    def _tick(self) -> None:
        end = self.slider.value() >= self.slider.maximum()
        pos = 0.0 if end else self._pos(self.slider) + 1.0 / max(1, self.slider.maximum())
        if self.ffr and self.fac_on.isChecked():
            self.fslider.blockSignals(True)
            self.fslider.setValue(int(round(pos * self.fslider.maximum())))
            self.fslider.blockSignals(False)
        self.slider.setValue(0 if end else self.slider.value() + 1)
        self._show()

    def _save_frame(self) -> None:
        if self.current is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save this frame", "frame.png", "PNG (*.png)")
        if path:
            cv2.imwrite(path, self.current)

    def closeEvent(self, e):
        self.gen += 1
        self.timer.stop()
        self._close_prerenders()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(e)
