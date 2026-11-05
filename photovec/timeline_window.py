"""Timeline window: ONE picture, a video-editor style timeline.

The sliders of Multi-Trace (Threshold, Noise, Paths, and with a mask: Background photo, Graffiti vector, Background blur) are lanes with
keyframes. Put a keyframe where something should change, drag it to the value you want, choose how the lane gets to the next keyframe
(linear, smooth, ease in / out, hold). The preview shows the frame under the playhead; all frames are prerendered in the background, so
playing and scrubbing are smooth. Export (MP4, GIF, transparent video) uses the same engine as Multi-Trace.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QScrollBar, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .layers import LAYER_KEYS, composite, has_layers
from .multi import MultiSettings, PanelSpec, load_panel_source, mask_style, photo_background, render_preview, styled_source
from .multi_project import ProjectError, read_masks, read_project, write_project
from .sweep import values_key
from .timeline import (EASES, EASE_LABEL, LANE_DEFS, Timeline, default_timeline, from_multi, sequence_photo_to_graffiti,
                       sequence_vector_to_photo)
from .timeline_view import TimelineView
from .trace import PRESETS as TRACE_PRESETS

EXT = ".photovec-timeline"
APP = "photovec-timeline"
CACHE_MAX = 400
CACHE_BYTES = 640 * 1024 * 1024         # finished frames kept (the count limit alone lets a big preview size use gigabytes)
PARTS_BYTES = 256 * 1024 * 1024         # vector pictures kept (the layers are laid over them again and again)


class _Sig(QObject):
    rendered = Signal(int, object, object)        # frame, image (BGR or BGRA), cache key
    stats = Signal()


def checkerboard(h: int, w: int, s: int = 14) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    c = (((x // s + y // s) % 2) * 38 + 205).astype(np.float32)
    return np.dstack([c, c, c])


def over_checker(img: np.ndarray) -> np.ndarray:
    """BGRA -> BGR over a checkerboard (so transparency is visible); BGR stays as it is."""
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:].astype(np.float32) / 255.0
        return np.clip(img[..., :3].astype(np.float32) * a + checkerboard(*img.shape[:2]) * (1 - a), 0, 255).astype(np.uint8)
    return img


class TimelineWindow(QWidget):
    def __init__(self, parent=None, autostart: bool = True):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Timeline")
        self.resize(1240, 960)
        self.path = ""
        self.mask: Optional[np.ndarray] = None
        self.mask_rev = 0
        self.tl = default_timeline(10.0, 25)
        self.sig = _Sig()
        self.sig.rendered.connect(self._on_rendered)
        self.sig.stats.connect(self._stats_soon)
        self._stats_timer = QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.setInterval(120)
        self._stats_timer.timeout.connect(self._update_stats)
        self.autostart = autostart
        self.ai = None
        self.job = None
        self.project_file = ""
        self._temp_dirs = []
        self._src = None
        self._src_key = None
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        self._want = 0
        self._gen = 0
        self._pgen = 0
        self.live = ThreadPoolExecutor(max_workers=1)
        self.bulk = ThreadPoolExecutor(max_workers=3)
        self._undo: list = []
        self._redo: list = []
        self._shown: Optional[np.ndarray] = None
        self._tcache: dict = {}
        self._parts: "OrderedDict[tuple, tuple]" = OrderedDict()      # (slider values that matter, look) -> (vector, ink float16 or None, paper)
        self._parts_bytes = 0
        self._cache_bytes = 0
        self._pre = None
        self._pre_sig = None
        self._closed = False

        v = QVBoxLayout(self)
        v.addLayout(self._top_bar())
        v.addLayout(self._mask_bar())
        v.addLayout(self._second_bar())
        self.view = QLabel("Open a picture to begin.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(360, 300)
        self.view.setStyleSheet("background:#1a1b1e;color:#888;")
        v.addWidget(self.view, 1)
        v.addLayout(self._transport())
        self.tv = TimelineView(self.tl)
        self.tv.seek.connect(self._seek)
        self.tv.changed.connect(self._edited)
        self.tv.editing.connect(self._push_undo)
        self.tv.selection.connect(self._selection_changed)
        self.tv.viewed.connect(self._sync_scroll)
        v.addWidget(self.tv)
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.valueChanged.connect(self._scrolled)
        v.addWidget(self.scroll)
        v.addLayout(self._inspector())
        v.addLayout(self._export_bar())

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(30)
        self.render_timer.timeout.connect(self._show_current)
        self.prerender_timer = QTimer(self)
        self.prerender_timer.setSingleShot(True)
        self.prerender_timer.setInterval(450)
        self.prerender_timer.timeout.connect(self.prerender)
        for seq, fn in ((QKeySequence.Undo, self.undo), (QKeySequence.Redo, self.redo), ("Ctrl+Y", self.redo),
                        ("Space", lambda: self.play.toggle()), ("K", self.add_key_at_playhead)):
            QShortcut(QKeySequence(seq), self, activated=fn)
        self._apply_layers_ui()
        self._selection_changed()
        QTimer.singleShot(0, self.tv.fit)

    # ------------------------------------------------------------------ UI
    def _top_bar(self):
        h = QHBoxLayout()
        b = QPushButton("Open picture…")
        b.clicked.connect(self._open_picture_dialog)
        h.addWidget(b)
        self.mask_btn = QPushButton("Edit mask…")
        self.mask_btn.clicked.connect(self._edit_mask)
        h.addWidget(self.mask_btn)
        self.use_mask = QCheckBox("Use the mask")
        self.use_mask.setToolTip("Adds the lanes Background photo / Graffiti vector / Background blur. The mask splits the picture into graffiti and background.")
        self.use_mask.toggled.connect(self._mask_options_changed)
        h.addWidget(self.use_mask)
        self.only = QCheckBox("Vectorise only inside the mask")
        self.only.setToolTip("The tracer only sees the masked graffiti; the background is the original photo (lane 'Background photo' = how visible it is).")
        self.only.setChecked(True)
        self.only.toggled.connect(self._mask_options_changed)
        h.addWidget(self.only)
        h.addWidget(QLabel("Vectorise as:"))
        self.preset = QComboBox()
        self.preset.addItems(list(TRACE_PRESETS))
        self.preset.setCurrentText("Black and White Logo")
        self.preset.currentIndexChanged.connect(self._look_changed)
        h.addWidget(self.preset)
        self.flip = QCheckBox("Flip black/white")
        self.flip.toggled.connect(self._look_changed)
        h.addWidget(self.flip)
        from .multi_window import ColorButton
        self.ink_btn = ColorButton(" ink ", (0, 0, 0), "Colour of the drawing (black by default)")
        self.paper_btn = ColorButton(" paper ", (255, 255, 255), "Colour of the paper (white by default)")
        self.ink_btn.changed.connect(self._look_changed)
        self.paper_btn.changed.connect(self._look_changed)
        h.addWidget(self.ink_btn)
        h.addWidget(self.paper_btn)
        h.addWidget(QLabel("Paper:"))
        self.bg = QComboBox()
        self.bg.addItems(["White", "Dark"])
        self.bg.currentIndexChanged.connect(self._look_changed)
        h.addWidget(self.bg)
        h.addStretch(1)
        for text, fn in (("Open project…", self._open_project), ("Save project…", self._save_project)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        return h

    def _mask_bar(self):
        h = QHBoxLayout()
        h.addWidget(QLabel("Mask finish:"))
        self.m_vector = QCheckBox("Vector outline")
        self.m_vector.setToolTip("The mask's outline is rebuilt as smooth curves and drawn at the picture's own size: no mask pixels, sharp at any resolution.")
        self.m_smooth = QSlider(Qt.Horizontal); self.m_smooth.setRange(0, 100); self.m_smooth.setFixedWidth(130)
        self.m_smooth.setToolTip("Rounds the outline of the mask: jaggies and pixel steps disappear (more = rounder, fine details go).")
        self.m_soft = QSlider(Qt.Horizontal); self.m_soft.setRange(0, 100); self.m_soft.setFixedWidth(130)
        self.m_soft.setToolTip("A soft edge: the graffiti fades into the background over a few pixels instead of ending sharply.")
        self.m_lbl = QLabel("")
        self.m_lbl.setStyleSheet("color:#888")
        h.addWidget(self.m_vector)
        h.addWidget(QLabel("Smooth")); h.addWidget(self.m_smooth)
        h.addWidget(QLabel("Soft edge")); h.addWidget(self.m_soft)
        h.addWidget(self.m_lbl)
        h.addStretch(1)
        self.m_vector.toggled.connect(self._mask_finish_changed)
        self.m_smooth.valueChanged.connect(self._mask_finish_changed)
        self.m_soft.valueChanged.connect(self._mask_finish_changed)
        return h

    def _mask_finish_changed(self, *_) -> None:
        self.m_lbl.setText(f"smooth {self.m_smooth.value()}  ·  soft {self.m_soft.value()}")
        self._look_changed()

    def _second_bar(self):
        h = QHBoxLayout()
        h.addWidget(QLabel("Length"))
        self.duration = QDoubleSpinBox(); self.duration.setRange(0.2, 600); self.duration.setDecimals(2); self.duration.setSuffix(" s"); self.duration.setValue(self.tl.duration)
        self.duration.editingFinished.connect(self._duration_edited)
        h.addWidget(self.duration)
        self.stretch = QCheckBox("stretch keys")
        self.stretch.setToolTip("When the length changes, move all keyframes along with it (otherwise they stay where they are).")
        h.addWidget(self.stretch)
        h.addWidget(QLabel("FPS"))
        self.fps = QSpinBox(); self.fps.setRange(1, 120); self.fps.setValue(self.tl.fps)
        self.fps.editingFinished.connect(self._fps_edited)
        h.addWidget(self.fps)
        h.addWidget(QLabel("Preview"))
        self.pv_size = QSpinBox(); self.pv_size.setRange(240, 1600); self.pv_size.setSingleStep(80); self.pv_size.setValue(560); self.pv_size.setSuffix(" px")
        self.pv_size.setToolTip("Size the preview frames are traced at (memory: about frames × size² × 3 bytes). The export uses 'Quality'.")
        self.pv_size.editingFinished.connect(self._look_changed)
        h.addWidget(self.pv_size)
        self.transp = QCheckBox("Show transparency")
        self.transp.setToolTip("Shows the transparent parts of the picture (the paper of a vector in front of a faded-out photo) as a checkerboard.")
        self.transp.toggled.connect(self._look_changed)
        h.addWidget(self.transp)
        self.seq_btn = QToolButton()
        self.seq_btn.setText("Sequences ▾")
        self.seq_btn.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self.seq_btn)
        m.addAction("Vector → photo (transparent start; needs a mask)").triggered.connect(lambda: self._sequence("vp"))
        m.addAction("Photo → vector → only the graffiti (needs a mask)").triggered.connect(lambda: self._sequence("pg"))
        m.addSeparator()
        m.addAction("Reset all lanes").triggered.connect(self._reset_lanes)
        self.seq_btn.setMenu(m)
        h.addWidget(self.seq_btn)
        self.undo_btn = QPushButton("↶ Undo")
        self.redo_btn = QPushButton("↷ Redo")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        h.addWidget(self.undo_btn)
        h.addWidget(self.redo_btn)
        h.addStretch(1)
        self.stats = QLabel("")
        h.addWidget(self.stats)
        return h

    def _transport(self):
        h = QHBoxLayout()
        first = QPushButton("⏮")
        first.clicked.connect(lambda: self._goto(0))
        self.play = QPushButton("▶ Play")
        self.play.setCheckable(True)
        self.play.toggled.connect(self._toggle_play)
        last = QPushButton("⏭")
        last.clicked.connect(lambda: self._goto(self.tl.frames() - 1))
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        self.time_lbl = QLabel("")
        self.vals_lbl = QLabel("")
        for w in (first, self.play, last, self.loop, self.time_lbl):
            h.addWidget(w)
        h.addStretch(1)
        h.addWidget(self.vals_lbl)
        return h

    def _inspector(self):
        h = QHBoxLayout()
        self.sel_lbl = QLabel("")
        h.addWidget(self.sel_lbl)
        h.addWidget(QLabel("Time"))
        self.k_t = QDoubleSpinBox(); self.k_t.setRange(0, 600); self.k_t.setDecimals(2); self.k_t.setSuffix(" s"); self.k_t.setSingleStep(0.04)
        self.k_v = QDoubleSpinBox(); self.k_v.setRange(-1000, 1000); self.k_v.setDecimals(1)
        self.k_e = QComboBox()
        for e in EASES:
            self.k_e.addItem(EASE_LABEL[e], e)
        for lab, w in (("", self.k_t), ("Value", self.k_v), ("Then", self.k_e)):
            if lab:
                h.addWidget(QLabel(lab))
            h.addWidget(w)
        self.k_t.editingFinished.connect(self._key_fields_edited)
        self.k_v.editingFinished.connect(self._key_fields_edited)
        self.k_e.activated.connect(lambda *_: self._key_fields_edited())
        add = QPushButton("＋ Key at playhead (K)")
        add.setToolTip("Adds a keyframe to the selected lane at the playhead; it keeps the current value, so nothing changes until you move it.")
        add.clicked.connect(self.add_key_at_playhead)
        rm = QPushButton("Delete key")
        rm.clicked.connect(lambda: self.tv.delete_selected())
        fit = QPushButton("Fit")
        fit.setToolTip("Show the whole length. Ctrl + wheel zooms, the wheel scrolls.")
        fit.clicked.connect(self.tv.fit)
        for w in (add, rm, fit):
            h.addWidget(w)
        h.addStretch(1)
        return h

    def _export_bar(self):
        v = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Quality"))
        self.quality = QSpinBox(); self.quality.setRange(240, 4000); self.quality.setSingleStep(100); self.quality.setValue(1200); self.quality.setSuffix(" px")
        self.quality.setToolTip("Long side the picture is traced at for the export.")
        h.addWidget(self.quality)
        h.addWidget(QLabel("Width"))
        self.out_w = QSpinBox(); self.out_w.setRange(240, 7680); self.out_w.setSingleStep(120); self.out_w.setValue(1080); self.out_w.setSuffix(" px")
        h.addWidget(self.out_w)
        h.addStretch(1)
        self.mp4_btn = QPushButton("Export MP4…")
        self.gif_btn = QPushButton("Export GIF…")
        self.alpha_btn = QPushButton("Export transparent…")
        self.mp4_btn.clicked.connect(lambda: self._export("mp4"))
        self.gif_btn.clicked.connect(lambda: self._export("gif"))
        self.alpha_btn.clicked.connect(lambda: self._export("alpha"))
        for b in (self.mp4_btn, self.gif_btn, self.alpha_btn):
            h.addWidget(b)
        v.addLayout(h)
        h2 = QHBoxLayout()
        self.bar = QProgressBar(); self.bar.setVisible(False)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_export)
        h2.addWidget(self.bar, 1)
        h2.addWidget(self.cancel_btn)
        v.addLayout(h2)
        self.status = QLabel("")
        v.addWidget(self.status)
        return v

    # ------------------------------------------------------------------ settings / sources
    def mask_active(self) -> bool:
        return self.use_mask.isChecked() and self.mask is not None

    def settings(self, tile: Optional[int] = None, width: Optional[int] = None) -> MultiSettings:
        act = self.mask_active()
        return MultiSettings(duration=self.tl.frames() / float(self.tl.fps), fps=self.tl.fps, hold_start=0.0, hold_end=0.0,
                             preset=self.preset.currentText(), tile=tile or self.pv_size.value(), columns=1, gap=0,
                             width=width or self.out_w.value(), bg="white" if self.bg.currentIndex() == 0 else "dark", invert=False,
                             smooth=True, points=2, layers=act, only_mask=act and self.only.isChecked(), timeline=self.tl,
                             mask_smooth=float(self.m_smooth.value()), mask_soft=float(self.m_soft.value()), mask_vector=self.m_vector.isChecked())

    def spec(self) -> PanelSpec:
        return PanelSpec(self.path, (0, 255), (5, 5), self.flip.isChecked(), None, self.ink_btn.color(), self.paper_btn.color(), None, self.mask)

    def _source(self):
        """The picture at preview size, traced as the export will (reloaded when the picture, mask or preset changed)."""
        if not self.path or not os.path.exists(self.path):
            return None
        st = self.settings()
        key = (self.path, st.preset, st.tile, st.layers, st.only_mask, self.mask_rev, mask_style(st).key() if st.layers else 0)
        if key != self._src_key:
            self._src = load_panel_source(self.path, st, self.mask)
            self._src_key = key
            self._tcache = {}                                  # traces belong to this source
        return self._src

    def _sig(self) -> tuple:
        st = self.settings()
        return (self._src_key, self.flip.isChecked(), self.ink_btn.color(), self.paper_btn.color(), st.bg, bool(self.transp.isChecked() and self.mask_active()),
                self.tl.fps)

    def _vals(self, frame: int) -> dict:
        st = self.settings()
        layers = bool(st.layers and self.mask is not None)
        return self.tl.frame_values(frame, layers, bool(layers and photo_background(st)))

    def _key(self, frame: int) -> tuple:
        return (values_key(self._vals(frame)), self._sig())

    # ------------------------------------------------------------------ rendering
    def _psig(self) -> tuple:
        """What the vector pictures depend on (not the layers: those are laid over them afterwards)."""
        return (self._src_key, self.flip.isChecked(), self.ink_btn.color(), self.paper_btn.color(), self.settings().bg)

    @staticmethod
    def _pvals(vals: dict) -> dict:
        """The slider values the vector picture really depends on: the layers only decide what is laid over it."""
        out = {k: v for k, v in vals.items() if k not in LAYER_KEYS}
        if "layer_p" in vals:
            out["layer_p"] = 1.0
            out["layer_g"] = 0.0 if vals.get("layer_g", 1.0) <= 0.0 else 1.0
        return out

    def _parts_get(self, pkey: tuple):
        with self._lock:
            hit = self._parts.get(pkey)
            if hit is not None:
                self._parts.move_to_end(pkey)
            return hit

    def _parts_put(self, pkey: tuple, vec: np.ndarray, ink, paper) -> tuple:
        part = (vec, None if ink is None else np.asarray(ink, np.float16), tuple(int(x) for x in paper))
        n = vec.nbytes + (0 if part[1] is None else part[1].nbytes)
        with self._lock:
            old = self._parts.pop(pkey, None)
            if old is not None:
                self._parts_bytes -= old[0].nbytes + (0 if old[1] is None else old[1].nbytes)
            self._parts[pkey] = part
            self._parts_bytes += n
            while self._parts_bytes > PARTS_BYTES and len(self._parts) > 1:
                _, o = self._parts.popitem(last=False)
                self._parts_bytes -= o[0].nbytes + (0 if o[1] is None else o[1].nbytes)
        return part

    def _parts_for(self, src, st: MultiSettings, vals: dict, spec: PanelSpec, cache: dict, psig: tuple):
        pv = self._pvals(vals)
        pkey = (values_key(pv), psig)
        hit = self._parts_get(pkey)
        if hit is None:
            vec, ink, paper = render_preview(src, st, pv, cache, spec.flip, spec.ink, spec.paper, None, parts=True)
            hit = self._parts_put(pkey, vec, ink, paper)
        return hit

    def _compose(self, part: tuple, src, vals: dict, transparent: bool) -> np.ndarray:
        """The frame from its vector picture: the layers laid over it (the same step render_values ends with)."""
        vec, ink, paper = part
        ink = None if ink is None else ink.astype(np.float32)
        photo = src.bgr if (src.mask is not None and vals.get("layer_p", 0.0) >= 0.5) else None
        if transparent and src.mask is not None and "layer_p" in vals:
            from .alpha_export import matte
            b = composite(vec, src.mask, (0, 0, 0), vals, photo=src.bgr, ink=ink)
            w = composite(vec, src.mask, (255, 255, 255), vals, photo=src.bgr, ink=ink)
            return cv2.cvtColor(matte(b, w), cv2.COLOR_RGBA2BGRA)
        if src.mask is not None and has_layers(vals):
            return composite(vec, src.mask, paper, vals, photo=photo, ink=ink)
        return vec

    def _render(self, frame: int, key: tuple, src, st: MultiSettings, vals: dict, spec: PanelSpec, transparent: bool, cache: dict, psig: tuple) -> np.ndarray:
        return self._compose(self._parts_for(src, st, vals, spec, cache, psig), src, vals, transparent)

    def frame_image(self, frame: int) -> Optional[np.ndarray]:
        """The preview picture of a frame, computed right now (BGR, or BGRA with 'Show transparency')."""
        src = self._source()
        if src is None:
            return None
        key = self._key(frame)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        img = self._render(frame, key, src, self.settings(), self._vals(frame), self.spec(), bool(self.transp.isChecked()), self._src_cache(), self._psig())
        self._store(key, img)
        return img

    def _src_cache(self) -> dict:
        return self._tcache

    def _store(self, key, img) -> None:
        with self._lock:
            old = self._cache.pop(key, None)
            if old is not None:
                self._cache_bytes -= old.nbytes
            self._cache[key] = img
            self._cache_bytes += img.nbytes
            while (len(self._cache) > CACHE_MAX or self._cache_bytes > CACHE_BYTES) and len(self._cache) > 1:
                _, o = self._cache.popitem(last=False)
                self._cache_bytes -= o.nbytes

    def _submit(self, frame: int, live: bool) -> None:
        if self._closed:
            return
        src = self._source()
        if src is None:
            return
        st, spec, tr, cache, psig = self.settings(), self.spec(), bool(self.transp.isChecked()), self._src_cache(), self._psig()
        vals = self._vals(frame)
        key = (values_key(vals), self._sig())
        with self._lock:
            if key in self._cache:
                if live:
                    self.sig.rendered.emit(frame, self._cache[key], key)
                return
        gen = self._gen if live else self._pgen

        def run():
            if live and frame != self._want:
                return
            if not live and gen != self._pgen:
                return
            try:
                img = self._render(frame, key, src, st, vals, spec, tr, cache, psig)
            except Exception as exc:
                print("Timeline: frame failed", exc)
                return
            self._store(key, img)
            self.sig.rendered.emit(frame, img, key)
            self.sig.stats.emit()
        (self.live if live else self.bulk).submit(run)

    def _on_rendered(self, frame: int, img, key) -> None:
        if frame == self.tv.frame:
            self._display(img)

    def _display(self, img: np.ndarray) -> None:
        self._shown = img
        show = over_checker(img)
        h, w = show.shape[:2]
        qi = QImage(np.ascontiguousarray(show[..., ::-1]).data, w, h, 3 * w, QImage.Format_RGB888).copy()
        box = self.view.size()
        self.view.setPixmap(QPixmap.fromImage(qi).scaled(max(16, box.width() - 4), max(16, box.height() - 4), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_current(self) -> None:
        frame = self.tv.frame
        self._want = frame
        self._update_labels()
        if self._source() is None:
            return
        key = self._key(frame)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            self._display(hit)
        else:
            self._submit(frame, True)

    def prerender(self) -> None:
        """Render every frame in the background (the ones already rendered are kept). The vector pictures are made by the worker
        processes (the tracer is Python code: threads would wait for each other), the layers are laid over them in threads."""
        if self._source() is None or not self.autostart:
            return
        self._pgen += 1
        n = self.tl.frames()
        start = self.tv.frame
        order = [(start + i) % n for i in range(n)]
        if not self._prerender_processes(order):
            for f in order:
                self._submit(f, False)
        self._update_stats()

    def _close_prerender(self) -> None:
        if self._pre is not None:
            self._pre.close()
        self._pre, self._pre_sig = None, None

    def _prerender_processes(self, order: list) -> bool:
        from .prerender import Prerender
        src = self._source()
        st, spec, tr, cache, psig, sig = self.settings(), self.spec(), bool(self.transp.isChecked()), self._src_cache(), self._psig(), self._sig()
        if self._pre is None or self._pre_sig != psig:
            self._close_prerender()
            pre = Prerender(styled_source(src, st, spec.flip, spec.ink, spec.paper, None), st.bg)
            if not pre.usable:
                return False
            self._pre, self._pre_sig = pre, psig
        groups: "OrderedDict[tuple, tuple]" = OrderedDict()        # vector picture -> (its slider values, the frames that lay layers over it)
        for f in order:
            vals = self._vals(f)
            key = (values_key(vals), sig)
            with self._lock:
                if key in self._cache:
                    continue
            pv = self._pvals(vals)
            groups.setdefault((values_key(pv), psig), (pv, []))[1].append((f, vals, key))
        gen = self._pgen

        def finish(f: int, vals: dict, key: tuple, pkey: tuple, pv: dict) -> None:           # (a thread of the pool: layers over a vector picture)
            if gen != self._pgen or self._closed:
                return
            try:
                part = self._parts_get(pkey) or self._parts_for(src, st, vals, spec, cache, psig)
                img = self._compose(part, src, vals, tr)
            except Exception as exc:
                print("Timeline: frame failed", exc)
                return
            self._store(key, img)
            self.sig.rendered.emit(f, img, key)
            self.sig.stats.emit()

        def on_frame(pkey, vec, ink, paper) -> None:
            if gen != self._pgen or self._closed:
                return
            self._parts_put(pkey, vec, ink, paper)
            pv, frames = groups[pkey]
            for f, vals, key in frames:
                self.bulk.submit(finish, f, vals, key, pkey, pv)

        def on_error(pkeys) -> None:                                  # a worker died: those frames are made here
            for pk in pkeys:
                pv, frames = groups[pk]
                for f, vals, key in frames:
                    self.bulk.submit(finish, f, vals, key, pk, pv)

        todo = []
        for pkey, (pv, frames) in groups.items():
            if self._parts_get(pkey) is not None:
                for f, vals, key in frames:
                    self.bulk.submit(finish, f, vals, key, pkey, pv)
            else:
                todo.append((pkey, pv))
        if todo and not self._pre.submit(todo, on_frame, on_error):
            on_error([pk for pk, _ in todo])
        return True

    def ready_count(self) -> int:
        if self._source() is None:
            return 0
        sig = self._sig()
        with self._lock:
            return sum(1 for f in range(self.tl.frames()) if (values_key(self._vals(f)), sig) in self._cache)

    def _stats_soon(self) -> None:
        self._stats_timer.start()

    def _update_stats(self) -> None:
        if self._src is None:
            self.stats.setText("")
            return
        n = self.tl.frames()
        r = self.ready_count()
        self.stats.setText(f"Frames ready {r}/{n}" + ("" if r < n else "  ✓"))

    def _update_labels(self) -> None:
        f = self.tv.frame
        self.time_lbl.setText(f"{self.tl.time_of(f):6.2f} s of {self.tl.duration:.2f} s   ·   frame {f + 1}/{self.tl.frames()}")
        v = self._vals(f)
        parts = [f"Threshold {v['threshold']:.1f}", f"Noise {v['noise']:.0f}"]
        if "paths" in v:
            parts.append(f"Paths {v['paths']:.0f}")
        if "layer_f" in v:
            parts += [f"Background {v['layer_f'] * 100:.0f} %", f"Graffiti {v['layer_g'] * 100:.0f} %"]
        self.vals_lbl.setText("   ·   ".join(parts))

    # ------------------------------------------------------------------ playhead / playback
    def _seek(self, frame: int) -> None:
        self._update_labels()
        self.render_timer.start()

    def _goto(self, frame: int) -> None:
        self.tv.set_frame(frame)
        self._seek(self.tv.frame)

    def _toggle_play(self, on: bool) -> None:
        self.play.setText("⏸ Pause" if on else "▶ Play")
        if on:
            self.play_timer.start(max(5, int(round(1000.0 / self.tl.fps))))
        else:
            self.play_timer.stop()

    def _tick(self) -> None:
        f = self.tv.frame + 1
        if f >= self.tl.frames():
            if not self.loop.isChecked():
                self.play.setChecked(False)
                return
            f = 0
        self.tv.set_frame(f)
        self._want = f
        self._update_labels()
        key = self._key(f)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            self._display(hit)
        else:
            self._submit(f, True)

    # ------------------------------------------------------------------ edits
    def _push_undo(self) -> None:
        import json
        self._undo.append(json.dumps(self.tl.to_json()))
        del self._undo[:-100]
        self._redo.clear()

    def _restore(self, text: str) -> None:
        import json
        new = Timeline.from_json(json.loads(text))
        self.tl.duration, self.tl.fps, self.tl.lanes = new.duration, new.fps, new.lanes
        self.tv.key_sel = None
        self.duration.setValue(self.tl.duration)
        self.fps.setValue(self.tl.fps)
        self._selection_changed()
        self._edited()

    def undo(self) -> None:
        import json
        if self._undo:
            self._redo.append(json.dumps(self.tl.to_json()))
            self._restore(self._undo.pop())

    def redo(self) -> None:
        import json
        if self._redo:
            self._undo.append(json.dumps(self.tl.to_json()))
            self._restore(self._redo.pop())

    def _edited(self) -> None:
        self._update_labels()
        self._sync_inspector()
        self.tv.update()
        self.render_timer.start()
        self.prerender_timer.start()
        self._update_stats()

    def _selection_changed(self) -> None:
        self._sync_inspector()

    def _sync_inspector(self) -> None:
        k, lid = self.tv.key_sel, self.tv.lane_sel
        on = k is not None and lid is not None
        for w in (self.k_t, self.k_v, self.k_e):
            w.setEnabled(on)
        if lid is not None:
            lane = self.tl.lanes[lid]
            self.k_v.blockSignals(True)
            self.k_v.setRange(lane.lo, lane.hi)
            self.k_v.blockSignals(False)
            self.sel_lbl.setText(f"<b>{lane.label}</b>" + (" keyframe" if on else " (no keyframe selected: double-click the lane to add one)"))
        if on:
            for w, val in ((self.k_t, k.t), (self.k_v, k.v)):
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)
            self.k_e.blockSignals(True)
            self.k_e.setCurrentIndex(max(0, EASES.index(k.ease)))
            self.k_e.blockSignals(False)

    def _key_fields_edited(self) -> None:
        k, lid = self.tv.key_sel, self.tv.lane_sel
        if k is None or lid is None:
            return
        lane = self.tl.lanes[lid]
        t, v, e = self.k_t.value(), lane.clamp(self.k_v.value()), self.k_e.currentData()
        if (round(t, 4), round(v, 4), e) == (round(k.t, 4), round(k.v, 4), k.ease):
            return
        self._push_undo()
        k.t, k.v, k.ease = min(max(t, 0.0), self.tl.duration), v, e
        lane.sort()
        self._edited()

    def add_key_at_playhead(self) -> None:
        lid = self.tv.lane_sel or "threshold"
        self._push_undo()
        self.tv.key_sel = self.tl.lanes[lid].add(self.tl.time_of(self.tv.frame))
        self.tv.lane_sel = lid
        self._selection_changed()
        self._edited()

    def _duration_edited(self) -> None:
        if abs(self.duration.value() - self.tl.duration) < 1e-6:
            return
        self._push_undo()
        self.tl.set_duration(self.duration.value(), self.stretch.isChecked())
        self.tv.set_frame(self.tv.frame)
        self.tv.fit()
        self._edited()

    def _fps_edited(self) -> None:
        if self.fps.value() == self.tl.fps:
            return
        self._push_undo()
        self.tl.fps = self.fps.value()
        self.tv.set_frame(min(self.tv.frame, self.tl.frames() - 1))
        if self.play.isChecked():
            self.play_timer.start(max(5, int(round(1000.0 / self.tl.fps))))
        self._edited()

    def _sequence(self, which: str) -> None:
        if self.mask is None:
            self.status.setText("This sequence needs a mask: press 'Edit mask…' first.")
            return
        self._push_undo()
        thr = self.tl.lanes["threshold"].value(0.0)
        thr = thr if thr >= 20 else 150.0
        self.use_mask.blockSignals(True); self.use_mask.setChecked(True); self.use_mask.blockSignals(False)
        self.only.blockSignals(True); self.only.setChecked(True); self.only.blockSignals(False)
        (sequence_vector_to_photo if which == "vp" else sequence_photo_to_graffiti)(self.tl, thr)
        self._mask_options_changed()
        self.status.setText("Sequence set - move its keyframes to taste.")

    def _reset_lanes(self) -> None:
        self._push_undo()
        d = default_timeline(self.tl.duration, self.tl.fps)
        self.tl.lanes = d.lanes
        self.tv.key_sel = None
        self._selection_changed()
        self._edited()

    # ------------------------------------------------------------------ look changes (everything must be rendered again)
    def _mask_options_changed(self, *_) -> None:
        self._apply_layers_ui()
        self._look_changed()

    def _apply_layers_ui(self) -> None:
        self.use_mask.setEnabled(self.mask is not None)
        self.only.setEnabled(self.use_mask.isChecked() and self.mask is not None)
        for w_ in (self.m_vector, self.m_smooth, self.m_soft):
            w_.setEnabled(self.mask_active())
        self.tv.layers_on = self.mask_active()
        if not self.tv.layers_on and self.tv.lane_sel in ("photo", "graffiti", "blur"):
            self.tv.lane_sel, self.tv.key_sel = "threshold", None
        self.tv.sync_height()
        self.transp.setEnabled(self.mask_active())

    def _look_changed(self, *_) -> None:
        self._apply_layers_ui()
        self._sync_inspector()
        self._update_labels()
        self._pgen += 1
        self.render_timer.start()
        self.prerender_timer.start()

    # ------------------------------------------------------------------ picture / mask
    def _open_picture_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open picture", "", "Pictures (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.heic *.heif)")
        if path:
            self.set_picture(path)

    def set_picture(self, path: str, mask: Optional[np.ndarray] = None) -> None:
        self.path = path
        self.mask = mask
        self.mask_rev += 1
        self.use_mask.blockSignals(True)
        self.use_mask.setChecked(mask is not None)
        self.use_mask.blockSignals(False)
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0
            self._parts.clear()
            self._parts_bytes = 0
        self._close_prerender()
        self.setWindowTitle(f"Timeline – {os.path.basename(path)}")
        self._apply_layers_ui()
        self._look_changed()

    def set_mask(self, m: Optional[np.ndarray]) -> None:
        self.mask = m if (m is not None and m.any()) else None
        self.mask_rev += 1
        self.use_mask.blockSignals(True)
        self.use_mask.setChecked(self.mask is not None)
        self.use_mask.blockSignals(False)
        self._look_changed()

    def _get_ai(self):
        if self.ai is None:
            from .ai_client import AIClient
            self.ai = AIClient()
        return self.ai

    def _edit_mask(self) -> None:
        from .imageio_ import load_image_alpha
        from .mask_editor import MAX_EDIT, MaskEditor
        from .rows import fit_size, image_size
        if not self.path:
            self.status.setText("Open a picture first.")
            return
        try:
            iw, ih = image_size(self.path)
            tw, th = fit_size("fit_long", MAX_EDIT, MAX_EDIT, iw / max(1, ih))
            rgb, _ = load_image_alpha(self.path, (tw, th))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self.status.setText(f"Could not read the picture: {exc}")
            return
        dlg = MaskEditor(bgr, self.mask, "Timeline picture", self, ai_factory=self._get_ai)
        if dlg.exec() == QDialog.Accepted:
            self.set_mask(dlg.result())

    def load_from_multi(self, path: str, mask, spec: PanelSpec, st: MultiSettings) -> None:
        """Take over one photo of Multi-Trace: its picture, mask, look and its slider movement as keyframes."""
        self.tl = from_multi(spec, st)
        self.tv.tl = self.tl
        self.tv.key_sel = None
        self._undo.clear(); self._redo.clear()
        for w, val in ((self.duration, self.tl.duration), (self.fps, self.tl.fps)):
            w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        self.preset.blockSignals(True); self.preset.setCurrentText(st.preset); self.preset.blockSignals(False)
        self.flip.blockSignals(True); self.flip.setChecked(bool(spec.flip)); self.flip.blockSignals(False)
        self.ink_btn.set_color(spec.ink)
        self.paper_btn.set_color(spec.paper)
        self.bg.blockSignals(True); self.bg.setCurrentIndex(0 if st.bg == "white" else 1); self.bg.blockSignals(False)
        self.quality.setValue(int(st.tile))
        self.out_w.setValue(int(min(max(240, st.width // max(1, st.columns)), 7680)))
        self.set_picture(path, mask)
        on = bool(st.layers and mask is not None)
        self.use_mask.blockSignals(True); self.use_mask.setChecked(on); self.use_mask.blockSignals(False)
        self.only.blockSignals(True); self.only.setChecked(bool(st.only_mask or st.facade_photo)); self.only.blockSignals(False)
        for w, val in ((self.m_vector, bool(st.mask_vector)), (self.m_smooth, int(st.mask_smooth)), (self.m_soft, int(st.mask_soft))):
            w.blockSignals(True); (w.setChecked if isinstance(w, QCheckBox) else w.setValue)(val); w.blockSignals(False)
        self._apply_layers_ui()
        self.tv.fit()
        self._selection_changed()
        self._look_changed()

    # ------------------------------------------------------------------ scroll bar
    def _sync_scroll(self) -> None:
        span = self.tv.visible_span()
        total = max(self.tl.duration, span)
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, int(max(0.0, total - span * 0.2) * 100))
        self.scroll.setPageStep(int(span * 100))
        self.scroll.setValue(int(self.tv.t0 * 100))
        self.scroll.blockSignals(False)

    def _scrolled(self, v: int) -> None:
        self.tv.set_t0(v / 100.0)

    # ------------------------------------------------------------------ project
    def project_state(self) -> dict:
        return {"panels": [{"mask": None, "image": None}], "timeline": self.tl.to_json(),
                "settings": {"preset": self.preset.currentText(), "flip": self.flip.isChecked(), "ink": self.ink_btn.color(), "paper": self.paper_btn.color(),
                             "bg": self.bg.currentIndex(), "use_mask": self.use_mask.isChecked(), "only": self.only.isChecked(),
                             "mask_vector": self.m_vector.isChecked(), "mask_smooth": self.m_smooth.value(), "mask_soft": self.m_soft.value(),
                             "quality": self.quality.value(), "width": self.out_w.value(), "preview": self.pv_size.value()}}

    def _save_project(self, path: str = "") -> None:
        if not self.path:
            self.status.setText("Open a picture first.")
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save timeline project", self.project_file or f"timeline{EXT}", f"Timeline project (*{EXT})")
        if not path:
            return
        if not path.lower().endswith(EXT):
            path += EXT
        try:
            write_project(path, self.project_state(), [self.path], [self.mask], app=APP)
        except (ProjectError, OSError) as exc:
            self.status.setText("Could not save: " + str(exc))
            return
        self.project_file = path
        self.status.setText(f"Saved {os.path.basename(path)}")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open timeline project", "", f"Timeline project (*{EXT})")
        if path:
            self.load_project(path)

    def load_project(self, path: str) -> bool:
        d = tempfile.mkdtemp(prefix="photovec_timeline_")
        try:
            state, files = read_project(path, d, app=APP)
            masks = read_masks(path, state)
            tl = Timeline.from_json(state["timeline"])
            c = state.get("settings", {})
        except (ProjectError, OSError, KeyError, ValueError, TypeError) as exc:
            shutil.rmtree(d, ignore_errors=True)
            self.status.setText("Could not open the project: " + str(exc))
            return False
        self._temp_dirs.append(d)
        self.tl.duration, self.tl.fps, self.tl.lanes = tl.duration, tl.fps, tl.lanes
        self.tv.key_sel = None
        self._undo.clear(); self._redo.clear()
        for w, val in ((self.duration, tl.duration), (self.fps, tl.fps)):
            w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        if c.get("preset") in TRACE_PRESETS:
            self.preset.blockSignals(True); self.preset.setCurrentText(c["preset"]); self.preset.blockSignals(False)
        for w, val in ((self.flip, bool(c.get("flip", False))), (self.only, bool(c.get("only", True)))):
            w.blockSignals(True); w.setChecked(val); w.blockSignals(False)
        for w, val in ((self.m_vector, bool(c.get("mask_vector", False))), (self.m_smooth, int(c.get("mask_smooth", 0))), (self.m_soft, int(c.get("mask_soft", 0)))):
            w.blockSignals(True); (w.setChecked if isinstance(w, QCheckBox) else w.setValue)(val); w.blockSignals(False)
        self.ink_btn.set_color(tuple(c["ink"]) if c.get("ink") else None)
        self.paper_btn.set_color(tuple(c["paper"]) if c.get("paper") else None)
        self.bg.blockSignals(True); self.bg.setCurrentIndex(int(c.get("bg", 0))); self.bg.blockSignals(False)
        for w, k in ((self.quality, "quality"), (self.out_w, "width"), (self.pv_size, "preview")):
            if k in c:
                w.setValue(int(c[k]))
        self.set_picture(files[0] if files else "", masks[0] if masks else None)
        self.use_mask.blockSignals(True); self.use_mask.setChecked(bool(c.get("use_mask", False)) and self.mask is not None); self.use_mask.blockSignals(False)
        self._apply_layers_ui()
        self.tv.fit()
        self.project_file = path
        self._look_changed()
        self.status.setText(f"Project opened: {os.path.basename(path)}")
        return True

    # ------------------------------------------------------------------ export
    def _export(self, kind: str) -> None:
        from .multi_window import MultiExportJob
        if self.job is not None:
            return
        if not self.path:
            QMessageBox.information(self, "Export", "Open a picture first.")
            return
        if kind == "alpha":
            from .alpha_export import FORMATS
            path, flt = QFileDialog.getSaveFileName(self, "Save transparent clip", "timeline_transparent.mov", ";;".join(FORMATS.values()))
            if not path:
                return
            ext = next((k for k in FORMATS if path.lower().endswith("." + k)), next((k for k, v in FORMATS.items() if v == flt), "mov"))
            if not path.lower().endswith("." + ext):
                path += "." + ext
            kind = "alpha:" + ext
        else:
            ext = "mp4" if kind == "mp4" else "gif"
            path, _ = QFileDialog.getSaveFileName(self, "Save clip", f"timeline.{ext}", f"{ext.upper()} (*.{ext})")
            if not path:
                return
            if not path.lower().endswith("." + ext):
                path += "." + ext
        st = self.settings(tile=self.quality.value(), width=self.out_w.value())
        st = replace(st, timeline=self.tl.copy())                                  # the clip is made from the timeline as it is now
        self.job = MultiExportJob([self.spec()], st, path, kind)
        self.job.progress.connect(self._export_progress)
        self.job.done.connect(lambda p: self.status.setText(f"Saved {p}"))
        self.job.notice.connect(lambda m: QMessageBox.information(self, "Export note", m))
        self.job.failed.connect(self._export_failed)
        self.job.finished.connect(self._export_finished)
        for b in (self.mp4_btn, self.gif_btn, self.alpha_btn):
            b.setEnabled(False)
        self.bar.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.status.setText("Working…")
        self.job.start()

    def _export_progress(self, label: str, i: int, n: int) -> None:
        self.bar.setRange(0, max(1, n))
        self.bar.setValue(i)
        self.status.setText(label + "…")

    def _export_failed(self, msg: str) -> None:
        self.status.setText("Export failed: " + msg)
        QMessageBox.warning(self, "Export failed", msg)

    def _export_finished(self) -> None:
        self.job = None
        for b in (self.mp4_btn, self.gif_btn, self.alpha_btn):
            b.setEnabled(True)
        self.bar.setVisible(False)
        self.cancel_btn.setVisible(False)

    def _cancel_export(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.status.setText("Cancelling…")

    # ------------------------------------------------------------------ misc
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._shown is not None:
            self._display(self._shown)

    def closeEvent(self, e):
        self._closed = True
        for t in (self.play_timer, self.render_timer, self.prerender_timer):
            t.stop()
        self._pgen += 1
        self._close_prerender()
        self.live.shutdown(wait=False, cancel_futures=True)
        self.bulk.shutdown(wait=False, cancel_futures=True)
        if self.job is not None:
            self.job.cancel()
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(e)
