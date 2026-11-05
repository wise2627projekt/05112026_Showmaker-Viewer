"""The noise editor: a popup that shows how the fill of ONE photo looks at several moments of the fill time, the map of the noise
behind it, and the controls to adjust it so it fits that picture. The values are the clip's (all photos) unless "Own values for this
photo" is ticked."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

from .multi import MultiSettings, fill_preview
from .sweep import FILL_ORIGINS

ORIGIN_TEXT = {"drawing": "From the drawing (the lines grow)", "edges": "From the picture edges inwards", "centre": "From the centre outwards"}

# key -> (label, slider min, max, to value, from value, tooltip)
CONTROLS = [
    ("fill_noise", "Noise amount", 0, 100, lambda s: s / 100.0, lambda v: int(round(v * 100)),
     "0 = an even growth around the drawing, 100 = mostly noise: blobs appear away from the drawing, grow and merge."),
    ("fill_size", "Blob size", 25, 400, lambda s: s / 100.0, lambda v: int(round(v * 100)),
     "Size of the noise blobs. Small = fine speckles, large = a few big clouds. Compare against the picture's own detail."),
    ("fill_detail", "Detail", 0, 100, lambda s: s / 100.0, lambda v: int(round(v * 100)),
     "0 = only large soft blobs, 100 = many small ones on top."),
    ("fill_smooth", "Smooth front", 0, 100, lambda s: s / 100.0, lambda v: int(round(v * 100)),
     "0 = a ragged, jagged front, 100 = very round shapes."),
    ("fill_pace", "Pace", 40, 300, lambda s: s / 100.0, lambda v: int(round(v * 100)),
     "1 = even. Above 1 the fill starts slowly and rushes at the end, below 1 it starts fast and takes its time at the end."),
]


def _fmt(key: str, v) -> str:
    if key == "fill_pace" or key == "fill_size":
        return f"{v:.2f}" + ("" if key == "fill_pace" else "×")
    return f"{int(round(v * 100))} %"


class _Sig(QObject):
    done = Signal(int, object, object)      # generation, pictures, noise map
    failed = Signal(int, str)


class _Tile(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignCenter)
        self.cap = QLabel()
        self.cap.setWordWrap(True)
        self.cap.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.pic, 1)
        lay.addWidget(self.cap)


class FillNoiseDialog(QDialog):
    """Works directly on the MultiWindow `win`: its clip values (win.fill_cfg) and each panel's own values (panel.fill_over)."""
    changed = Signal()

    def __init__(self, win, panel_index: int = 0):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Noise of the fill")
        self.resize(1050, 760)
        self.gen = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.sig = _Sig()
        self.sig.done.connect(self._on_done)
        self.sig.failed.connect(self._on_failed)
        self._busy = False
        self._again = False
        self._building = False
        self._src_cache: Dict[str, object] = {}
        self.sliders: Dict[str, QSlider] = {}
        self.labels: Dict[str, QLabel] = {}
        self.tiles: List[_Tile] = []
        self.last = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._start)

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Photo:"))
        self.photo = QComboBox()
        for i, p in enumerate(win.panels):
            self.photo.addItem(f"{i + 1}: " + (p.name.text() if p.path else "– empty –"))
        top.addWidget(self.photo, 1)
        top.addWidget(QLabel("Pictures:"))
        self.n = QSpinBox(); self.n.setRange(2, 12); self.n.setValue(6)
        self.n.setToolTip("How many moments of the fill time are shown (the last one is the end of the fill).")
        top.addWidget(self.n)
        self.overlay = QCheckBox("Noise overlay")
        self.overlay.setChecked(True)
        self.overlay.setToolTip("Lays the noise over every picture: the colours show the part that is still to be filled (blue = soon, red = last), "
                                "the white line is the front at that moment. Only a display aid – it is not in the clip.")
        self.strength = QSlider(Qt.Horizontal); self.strength.setRange(0, 100); self.strength.setValue(55); self.strength.setFixedWidth(110)
        self.strength.setToolTip("How strongly the noise colours cover the picture")
        top.addWidget(self.overlay)
        top.addWidget(self.strength)
        root.addLayout(top)

        own = QHBoxLayout()
        self.own = QCheckBox("Own values for this photo")
        self.own.setToolTip("Off: the values below are the same for every photo. On: this photo gets its own noise, the others keep the clip's.")
        self.reset = QPushButton("Reset")
        self.reset.setToolTip("Back to the defaults (for this photo if 'Own values' is on, otherwise for the whole clip)")
        self.scope = QLabel("")
        self.scope.setStyleSheet("color:#888")
        own.addWidget(self.own)
        own.addWidget(self.reset)
        own.addWidget(self.scope, 1)
        root.addLayout(own)

        g = QGridLayout()
        for r, (key, text, lo, hi, _to, _from, tip) in enumerate(CONTROLS):
            lab = QLabel(text + ":")
            s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setMaximumWidth(520); s.setToolTip(tip); lab.setToolTip(tip)
            val = QLabel(); val.setMinimumWidth(48)
            g.addWidget(lab, r, 0); g.addWidget(s, r, 1); g.addWidget(val, r, 2)
            self.sliders[key] = s
            self.labels[key] = val
            s.valueChanged.connect(lambda v, k=key: self._slider(k, v))
        r = len(CONTROLS)
        g.addWidget(QLabel("Pattern (seed):"), r, 0)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999)
        self.seed.setToolTip("Another number = another noise pattern with the same character.")
        self.dice = QPushButton("🎲 Another")
        sh = QHBoxLayout(); sh.addWidget(self.seed); sh.addWidget(self.dice); sh.addStretch(1)
        g.addLayout(sh, r, 1)
        g.addWidget(QLabel("Starts:"), r + 1, 0)
        self.origin = QComboBox()
        for o in FILL_ORIGINS:
            self.origin.addItem(ORIGIN_TEXT[o], o)
        g.addWidget(self.origin, r + 1, 1)
        g.setColumnStretch(3, 1)
        root.addLayout(g)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#888")
        root.addWidget(self.status)
        self.area = QScrollArea(); self.area.setWidgetResizable(True)
        self.holder = QWidget(); self.grid = QGridLayout(self.holder)
        self.grid.setAlignment(Qt.AlignTop)
        self.area.setWidget(self.holder)
        root.addWidget(self.area, 1)

        self.photo.currentIndexChanged.connect(self._photo_changed)
        self.n.valueChanged.connect(self._schedule)
        self.overlay.toggled.connect(self._redraw)
        self.strength.valueChanged.connect(self._redraw)
        self.own.toggled.connect(self._own_toggled)
        self.reset.clicked.connect(self._reset)
        self.seed.valueChanged.connect(lambda v: self._set("fill_seed", int(v)))
        self.dice.clicked.connect(lambda: self.seed.setValue(int(np.random.randint(0, 1000000))))
        self.origin.currentIndexChanged.connect(lambda _=None: self._set("fill_origin", self.origin.currentData()))
        self.photo.setCurrentIndex(max(0, min(panel_index, self.photo.count() - 1)))
        self._photo_changed()

    # ------------------------------------------------------------------ values
    def index(self) -> int:
        return max(0, self.photo.currentIndex())

    def panel(self):
        return self.win.panels[self.index()]

    def values(self) -> dict:
        """What the shown photo uses now."""
        return {**self.win.fill_cfg, **(self.panel().fill_over or {})}

    def _photo_changed(self, *_) -> None:
        self._building = True
        self.own.setChecked(bool(self.panel().fill_over))
        self._building = False
        self._show_values()
        self._schedule()

    def _show_values(self) -> None:
        self._building = True
        v = self.values()
        for key, _t, _lo, _hi, _to, frm, _tip in CONTROLS:
            self.sliders[key].setValue(frm(v[key]))
            self.labels[key].setText(_fmt(key, v[key]))
        self.seed.setValue(int(v["fill_seed"]))
        self.origin.setCurrentIndex(max(0, list(FILL_ORIGINS).index(v["fill_origin"])) if v["fill_origin"] in FILL_ORIGINS else 0)
        self.scope.setText(f"These values are only used for photo {self.index() + 1}." if self.own.isChecked()
                           else "These values are used for every photo that has no own values.")
        self._building = False

    def _set(self, key: str, value) -> None:
        if self._building:
            return
        p = self.panel()
        if self.own.isChecked():
            p.fill_over = {**(p.fill_over or self.win.fill_cfg), key: value}
        else:
            self.win.fill_cfg[key] = value
        self.changed.emit()
        self._schedule()

    def _slider(self, key: str, s: int) -> None:
        to = next(c[4] for c in CONTROLS if c[0] == key)
        self.labels[key].setText(_fmt(key, to(s)))
        self._set(key, to(s))

    def _own_toggled(self, on: bool) -> None:
        if self._building:
            return
        p = self.panel()
        p.fill_over = dict(self.win.fill_cfg) if on else None       # starts from what the photo used so far
        self._show_values()
        self.changed.emit()
        self._schedule()

    def _reset(self) -> None:
        d = MultiSettings().fill_style()
        if self.own.isChecked():
            self.panel().fill_over = dict(d)
        else:
            self.win.fill_cfg.update(d)
        self._show_values()
        self.changed.emit()
        self._schedule()

    # ------------------------------------------------------------------ pictures
    def _schedule(self, *_) -> None:
        self.timer.start()

    def _start(self) -> None:
        if self._busy:
            self._again = True
            return
        p = self.panel()
        if not p.path or p.src is None:
            self.status.setText("This photo is not loaded yet.")
            self._clear_tiles()
            return
        self._busy, self._again = True, False
        self.gen += 1
        gen = self.gen
        st = self.win.settings()
        spec = p.spec()
        src = p.src
        n = self.n.value()
        self.status.setText("Rendering…")
        self.pool.submit(self._work, gen, src, st, spec, n)

    def _work(self, gen, src, st, spec, n) -> None:
        try:
            pics, heat = fill_preview(src, st, spec, n)
        except Exception as exc:                                      # shown in the window, never kills it
            self.sig.failed.emit(gen, str(exc))
            return
        self.sig.done.emit(gen, pics, heat)

    def _on_failed(self, gen: int, msg: str) -> None:
        self._busy = False
        self.status.setText("Could not render the preview: " + msg)

    def _on_done(self, gen: int, pics, heat) -> None:
        self._busy = False
        if self._again or gen != self.gen:
            self._start()
            return
        self.last = pics
        while len(self.tiles) < len(pics):
            self.tiles.append(_Tile())
        del self.tiles[len(pics):]
        self.status.setText(f"Photo {self.index() + 1}: {len(pics)} moments from the start of the fill ({self.win.fill_s.value():.1f} s) to the end of the movement"
                            + ("" if self.win.fill_on.isChecked() else "  (the fill is switched off in the main window – shown as if it were on)"))
        self._redraw()

    @staticmethod
    def compose(d: dict, strength: float) -> np.ndarray:
        """The picture of one moment with the noise laid over the part that is still to come, and the front as a white line."""
        img = d["image"].astype(np.float32)
        a = (d["ahead"].astype(np.float32) * strength)[..., None]
        out = img * (1 - a) + d["plain"].astype(np.float32) * a
        line = d["front"][..., None].astype(np.float32) / 255.0 * (1.0 if strength > 0 else 0.0)
        out = out * (1 - line) + 255.0 * line
        return np.clip(out, 0, 255).astype(np.uint8)

    def _redraw(self, *_) -> None:
        from .multi_window import _pix
        if self.last is None:
            return
        pics = self.last
        k = self.strength.value() / 100.0
        for t, d in zip(self.tiles, pics):
            t.pic.setPixmap(_pix(self.compose(d, k) if self.overlay.isChecked() else d["image"], 300))
            t.cap.setText(f"t = {d['seconds']:.1f} s  ·  {d['coverage'] * 100:.0f} % filled")
        self.strength.setEnabled(self.overlay.isChecked())
        self._layout_tiles()

    def _clear_tiles(self) -> None:
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().setParent(None)

    def _layout_tiles(self, *_) -> None:
        self._clear_tiles()
        for k, t in enumerate(self.tiles):
            self.grid.addWidget(t, k // 3, k % 3)

    def done(self, r: int) -> None:
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().done(r)
