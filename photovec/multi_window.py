"""Multi-Trace window: many photos side by side, each with its own Threshold / Noise range, one shared timeline, one clip.

    * Photos: choose how many panels you want, load a picture into each (button or drag & drop).
    * Every panel has a Threshold and a Noise slider with TWO handles: where the slider is at the start and at the end of the clip.
    * One time slider drives all panels. The duration (and FPS) you enter decides the number of frames; the step of every slider
      follows from its range and the duration.
    * Export: MP4 or GIF with all panels arranged in a grid."""
from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QApplication, QColorDialog, QLabel, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QSpinBox, QToolButton, QVBoxLayout, QWidget, QDialog, QLineEdit, QFormLayout, QDialogButtonBox)

from . import pool as pool_mod
from .animate import write_gif_stream, write_mp4
from .batch import resolve_workers, shared_pool
from .multi import (EASES, NOISE, WARHOL, palette_pairs, tile_colors, THRESHOLD, MultiSettings, PanelSpec, build_renderer, load_panel_source, measure_source, point_frame,
                    render_preview, steps_between_points, values_at)
from .layers import fit_mask
from .multi_project import EXT as PROJECT_EXT, ProjectError, read_masks, read_project, write_project
from .range_slider import RangeSlider
from .sweep import values_key
from .trace import PRESETS as TRACE_PRESETS

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif")


DEFAULT_NAMES = {"thr": "Threshold", "noise": "Noise", "paths": "Paths", "fac": "Facade", "gra": "Graffiti", "blur": "Facade blur"}


def _pix(bgr: np.ndarray, box: int) -> QPixmap:
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(box, box, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ColorButton(QPushButton):
    """A button that shows a colour and asks for a new one when clicked. `color()` is an (r, g, b) tuple or None (= the default)."""
    changed = Signal()

    def __init__(self, label: str, default: tuple, tip: str = ""):
        super().__init__(label)
        self._default = default
        self._rgb = None
        self.setFixedHeight(24)
        self.setToolTip(tip)
        self.clicked.connect(self._pick)
        self._paint()

    def color(self):
        return self._rgb

    def set_color(self, rgb, emit: bool = False) -> None:
        rgb = None if rgb is None else tuple(int(v) for v in rgb)
        if rgb != self._rgb:
            self._rgb = rgb
            self._paint()
            if emit:
                self.changed.emit()

    def _paint(self) -> None:
        r, g, b = self._rgb or self._default
        fg = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
        self.setStyleSheet(f"QPushButton{{background:rgb({r},{g},{b});color:{fg};border:1px solid #666;border-radius:3px;padding:0 8px}}")

    def _pick(self) -> None:
        r, g, b = self._rgb or self._default
        c = QColorDialog.getColor(QColor(r, g, b), self, "Choose a colour")
        if c.isValid():
            self.set_color((c.red(), c.green(), c.blue()), True)


class _Signals(QObject):
    sourced = Signal(int, object, str)             # panel, SweepSource | None, path
    rendered = Signal(int, object, object, int)    # panel, image, key, generation
    curved = Signal(int, object)                   # panel, BlackCurve | None


class Panel(QFrame):
    """One photo with its Threshold, Noise (and optionally Paths) sliders. Every slider has one handle per slider point."""
    edited = Signal(int)
    load_requested = Signal(int)
    file_dropped = Signal(int, str)
    black_requested = Signal(int)
    fit_requested = Signal(int)
    mask_requested = Signal(int)
    preview_point = Signal(int, int, bool)       # panel, slider point (-1 = back to the time slider), keep until the mouse is released

    def __init__(self, index: int, points: int = 5):
        super().__init__()
        self.index = index
        self.fill_over: Optional[dict] = None            # this photo's own noise settings for the fill (None = the clip's)
        self.mask: Optional[np.ndarray] = None           # layer mask: masked = graffiti, the rest = facade (uint8, any size)
        self.mask_rev = 0                                # counts changes, so previews know when to redraw
        self.points = max(2, points)
        self.path = ""
        self.src = None
        self.cache: dict = {}
        self.key = None
        self.curve = None
        self.setFrameShape(QFrame.StyledPanel)
        self.setAcceptDrops(True)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.title = QLabel(f"<b>Photo {index + 1}</b>")
        self.name = QLabel("– drop a picture here –")
        self.name.setStyleSheet("color:#888")
        load = QPushButton("Load…")
        load.clicked.connect(lambda: self.load_requested.emit(self.index))
        clear = QPushButton("×")
        clear.setFixedWidth(26)
        clear.setToolTip("Remove the picture")
        clear.clicked.connect(lambda: self.file_dropped.emit(self.index, ""))
        top.addWidget(self.title)
        top.addWidget(self.name, 1)
        top.addWidget(load)
        top.addWidget(clear)
        v.addLayout(top)
        self.view = QLabel()
        self.view.setMinimumSize(270, 200)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background:#1c1c20;color:#777")
        self.view.setText("no picture")
        v.addWidget(self.view, 1)
        self.thr = self._range(v, "Threshold", *THRESHOLD, (0, 255))
        self.noise = self._range(v, "Noise", *NOISE, (1, 25))
        self.paths = self._range(v, "Paths", 0, 100, (50, 50), optional=True)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#888;font-size:11px")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Colours:"))
        self.ink_btn = ColorButton("Drawing", (0, 0, 0), "Colour of the vector drawing (default black)")
        self.paper_btn = ColorButton("Paper", (255, 255, 255), "Colour of the paper behind the drawing (default white)")
        reset = QPushButton("↺")
        reset.setFixedSize(26, 24)
        reset.setToolTip("Back to black on white")
        reset.clicked.connect(lambda: self.set_colors(None, None))
        for b in (self.ink_btn, self.paper_btn):
            b.changed.connect(lambda: self.edited.emit(self.index))
            crow.addWidget(b)
        crow.addWidget(reset)
        crow.addStretch(1)
        v.addLayout(crow)
        self.flip = QCheckBox("Black → white")
        self.flip.setToolTip("Swap black and white for this photo only: the picture starts black and ends white "
                             "(useful for dark photos).")
        self.flip.toggled.connect(lambda _=None: self.edited.emit(self.index))
        v.addWidget(self.flip)
        mrow = QHBoxLayout()
        self.mask_btn = QPushButton("Mask…")
        self.mask_btn.setToolTip("Open the mask editor: mark what is graffiti (the mask) and what is facade. The layers then get their own "
                                 "visibility over time (row 'Layers' below the time slider).")
        self.mask_btn.clicked.connect(lambda: self.mask_requested.emit(self.index))
        self.mask_clear = QPushButton("×")
        self.mask_clear.setFixedWidth(26)
        self.mask_clear.setToolTip("Remove the mask of this photo")
        self.mask_clear.clicked.connect(lambda: self.set_mask(None))
        self.mask_lbl = QLabel("no mask")
        self.mask_lbl.setStyleSheet("color:#888")
        mrow.addWidget(self.mask_btn)
        mrow.addWidget(self.mask_clear)
        mrow.addWidget(self.mask_lbl, 1)
        v.addLayout(mrow)
        row = QHBoxLayout()
        bp = QPushButton("Threshold end → black")
        bp.setToolTip("Sets the last Threshold point to the value at which (almost) none of this picture is white any more.")
        bp.clicked.connect(lambda: self.black_requested.emit(self.index))
        self.black_btn = bp
        fp = QPushButton("Place Threshold points evenly")
        fp.setToolTip("Measures this picture and puts the Threshold points where its white area has shrunk by equal amounts: "
                      "point 1 = where it is now, last point = black. All photos placed this way lose their white together.")
        fp.clicked.connect(lambda: self.fit_requested.emit(self.index))
        self.fit_btn = fp
        row.addWidget(bp)
        row.addWidget(fp)
        v.addLayout(row)
        for d in (self.thr, self.noise, self.paths):
            self._build_spins(d)
        self.mask_clear.setEnabled(False)

    # -- one slider with `points` handles
    def _range(self, v, label: str, lo: int, hi: int, init, optional: bool = False) -> dict:
        row = QHBoxLayout()
        use = None
        if optional:
            use = QCheckBox(label)
            use.setToolTip("Animate the Image Trace 'Paths' slider (fit tightness) as well. Off = the preset's value is used.")
            row.addWidget(use)
            lab = use
        else:
            lab = QLabel(label)
            row.addWidget(lab)
        sl = RangeSlider(lo, hi, *init)
        sl.set_count(self.points)
        spin_row = QHBoxLayout()
        spin_row.setContentsMargins(0, 0, 0, 0)
        d = dict(slider=sl, spins=[], lo=lo, hi=hi, spin_row=spin_row, use=use, label=lab)
        row.addWidget(sl, 1)
        v.addLayout(row)
        v.addLayout(spin_row)
        sl.changed.connect(lambda: self._from_slider(d))
        sl.pressed.connect(lambda k: self.preview_point.emit(self.index, k, True))
        sl.released.connect(lambda: self.preview_point.emit(self.index, -1, True))
        if use is not None:
            def toggled(on):
                for w in (sl,):
                    w.setEnabled(on)
                for sp in d["spins"]:
                    sp.setEnabled(on)
                self.edited.emit(self.index)
            use.toggled.connect(toggled)
            sl.setEnabled(False)
        return d

    def _build_spins(self, d: dict) -> None:
        lay = d["spin_row"]
        while lay.count():
            it = lay.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        d["spins"] = []
        vals = d["slider"].values()
        enabled = d["use"].isChecked() if d["use"] is not None else True
        n = len(vals)
        lay.addStretch(1)
        for i, val in enumerate(vals):
            if n > 2 or i == 0:
                pass
            lab = QLabel(f"{i + 1}" if n > 2 else ("start" if i == 0 else "end"))
            lab.setStyleSheet("color:#888;font-size:10px")
            sp = QSpinBox()
            sp.setRange(d["lo"], d["hi"])
            sp.setValue(val)
            sp.setFixedWidth(54)
            sp.setEnabled(enabled)
            sp.valueChanged.connect(lambda _=None, dd=d, kk=i: self._from_spin(dd, kk))
            lay.addWidget(lab)
            lay.addWidget(sp)
            d["spins"].append(sp)
        lay.addStretch(1)

    def _show_values(self, d: dict) -> None:
        for sp, val in zip(d["spins"], d["slider"].values()):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

    def _from_slider(self, d: dict) -> None:
        self._show_values(d)
        self.edited.emit(self.index)

    def _from_spin(self, d: dict, k: int) -> None:
        d["slider"].set_values(*[sp.value() for sp in d["spins"]])
        self._show_values(d)
        self.edited.emit(self.index)
        self.preview_point.emit(self.index, k, False)

    def set_pinned(self, k: int) -> None:
        """Title shows which slider point the preview currently shows (-1 = the moment of the time slider)."""
        self.title.setText(f"<b>Photo {self.index + 1}</b>" + (f" · point {k + 1}" if k >= 0 else ""))

    def set_points(self, n: int) -> None:
        n = max(2, int(n))
        if n == self.points:
            return
        self.points = n
        for d in (self.thr, self.noise, self.paths):
            d["slider"].set_count(n)
            self._build_spins(d)
        self.edited.emit(self.index)

    # -- spec
    @staticmethod
    def _seq(d: dict) -> tuple:
        return tuple(d["slider"].values())

    @staticmethod
    def _put_seq(d: dict, seq) -> None:
        d["slider"].set_values(*seq)

    def spec(self) -> PanelSpec:
        paths = self._seq(self.paths) if self.paths["use"].isChecked() else None
        return PanelSpec(self.path, self._seq(self.thr), self._seq(self.noise), self.flip.isChecked(), paths,
                         self.ink_btn.color(), self.paper_btn.color(), dict(self.fill_over) if self.fill_over else None, self.mask)

    def set_mask(self, m: Optional[np.ndarray]) -> None:
        self.mask = m
        self.mask_rev += 1
        if m is None:
            self.mask_lbl.setText("no mask")
            self.mask_lbl.setStyleSheet("color:#888")
        else:
            self.mask_lbl.setText(f"mask: {100.0 * float((m > 127).mean()):.0f} % graffiti")
            self.mask_lbl.setStyleSheet("color:#4a4")
        self.mask_clear.setEnabled(m is not None)
        self.edited.emit(self.index)

    def set_names(self, names: dict) -> None:
        for key, d in (("thr", self.thr), ("noise", self.noise), ("paths", self.paths)):
            d["label"].setText(names.get(key, d["label"].text()))

    def set_colors(self, ink, paper) -> None:
        self.ink_btn.set_color(ink)
        self.paper_btn.set_color(paper)
        self.edited.emit(self.index)

    def set_seq(self, which: str, seq) -> None:
        d = {"thr": self.thr, "noise": self.noise, "paths": self.paths}[which]
        self._put_seq(d, [int(round(x)) for x in seq])
        self._show_values(d)
        self.edited.emit(self.index)

    def set_ends(self, which: str, start: int, end: int) -> None:
        """First and last point; the points in between stay where they are as far as the new ends allow."""
        d = {"thr": self.thr, "noise": self.noise, "paths": self.paths}[which]
        cur = list(self._seq(d))
        lo, hi = min(start, end), max(start, end)
        mid = sorted(min(max(x, lo), hi) for x in cur[1:-1])
        if start > end:
            mid.reverse()
        self.set_seq(which, [start] + mid + [end])

    def set_paths_default(self, value: int) -> None:
        if not self.paths["use"].isChecked():
            self.paths["slider"].set_values(*([int(value)] * self.points))
            self._show_values(self.paths)

    def set_marker(self, spec: PanelSpec, st: MultiSettings, frame: int) -> None:
        v = values_at(spec, st, frame)
        self.thr["slider"].set_marker(v["threshold"])
        self.noise["slider"].set_marker(v["noise"])
        self.paths["slider"].set_marker(v.get("paths"))

    def show_info(self, st: MultiSettings) -> None:
        sp = self.spec()

        def one(name: str, vals) -> str:
            steps = steps_between_points(vals, st)
            lo, hi = min(steps), max(steps)
            rate = f"{lo:.3f}" if abs(hi - lo) < 5e-4 else f"{lo:.3f}–{hi:.3f}"
            return f"{name} {vals[0]:g} → {vals[-1]:g}: {rate} per frame"
        parts = [one("Threshold", sp.thr), one("Noise", sp.noise)]
        if sp.paths is not None:
            parts.append(one("Paths", sp.paths))
        if sp.fill_over:
            parts.append("own fill noise")
        self.info.setText("   ·   ".join(parts))

    def set_picture(self, path: str) -> None:
        if path != self.path and self.mask is not None:          # a mask belongs to its picture
            self.set_mask(None)
        self.path = path
        self.src, self.key, self.cache, self.curve = None, None, {}, None
        self.mask_clear.setEnabled(self.mask is not None)
        self.name.setText(os.path.basename(path) if path else "– drop a picture here –")
        self.name.setStyleSheet("" if path else "color:#888")
        if not path:
            self.view.setPixmap(QPixmap())
            self.view.setText("no picture")

    def show_image(self, bgr: np.ndarray) -> None:
        self.view.setPixmap(_pix(bgr, max(200, min(self.view.width(), self.view.height()))))

    # -- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if f.lower().endswith(IMG_EXT):
                self.file_dropped.emit(self.index, f)
                break


class MultiExportJob(QThread):
    progress = Signal(str, int, int)
    done = Signal(str)
    failed = Signal(str)
    notice = Signal(str)                      # something the person should know although the export worked (a fallback encoder was used)

    def __init__(self, specs: List[PanelSpec], st: MultiSettings, path: str, kind: str):
        super().__init__()
        self.specs, self.st, self.path, self.kind = specs, st, path, kind
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _cb(self, label: str):
        def f(i: int, n: int) -> bool:
            self.progress.emit(label, i, n)
            return self._cancel
        return f

    def run(self) -> None:
        try:
            with pool_mod.busy():
                self._run()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        st = self.st
        nw = resolve_workers(0)
        pool = shared_pool(nw) if nw > 1 else None
        if self.kind.startswith("alpha"):
            from .alpha_export import export_transparent

            def prog(label: str, i: int, n: int) -> bool:
                self.progress.emit(label, i, n)
                return self._cancel
            if export_transparent(self.specs, st, self.path, self.kind.split(":", 1)[1], nw, pool, prog) and not self._cancel:
                from . import alpha_export
                if alpha_export.last_note:
                    self.notice.emit(alpha_export.last_note)
                self.done.emit(self.path)
            return
        r = build_renderer(self.specs, st, nw, self._cb("Tracing the photos"), pool)
        if r is None or self._cancel:
            return
        if not r.ensure_sweeps(nw, pool, self._cb("Rendering the frames")):
            return
        self.progress.emit("Encoding", 0, r.total)
        cb = self._cb("Encoding")
        if self.kind == "mp4":
            write_mp4(self.path, r.frames(cb, max(1, min(4, (os.cpu_count() or 2) - 1))), st.fps)
        else:
            write_gif_stream(self.path, r.total, r.frame, st.fps, cb, max(1, min(4, (os.cpu_count() or 2) - 1)))
        if self._cancel:
            if os.path.exists(self.path):
                os.remove(self.path)
            return
        self.done.emit(self.path)


class MultiWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Trace – many photos, one clip")
        self.project_file = ""
        self._temp_dirs: List[str] = []
        self.resize(1280, 860)
        self.panels: List[Panel] = []
        self.frame = 0
        self.gen = 0
        self._latest: dict = {}                                       # panel -> key of the newest preview asked for
        self.job: Optional[MultiExportJob] = None
        self.sig = _Signals()
        self.sig.sourced.connect(self._on_sourced)
        self.sig.rendered.connect(self._on_rendered)
        self.sig.curved.connect(self._on_curved)
        self.workers = ThreadPoolExecutor(max_workers=max(2, min(4, (os.cpu_count() or 2) - 1)))
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(40)
        self._render_timer.timeout.connect(self._render_all)
        self._dirty: set = set()
        self._mode: Dict[int, str] = {}
        self.fill_cfg: dict = MultiSettings().fill_style()     # the clip's noise settings (all photos; a photo can have its own)
        self.noise_dlg = None
        self.ai = None                           # one AI helper process for all mask editors (the model stays loaded)
        self.names = dict(DEFAULT_NAMES)         # what the sliders are called (editable: 'Rename sliders…')
        self.pin: Dict[int, int] = {}            # panel -> slider point its preview shows while a handle is held / a box is edited
        self._unpin_timer = QTimer(self)
        self._unpin_timer.setSingleShot(True)
        self._unpin_timer.setInterval(1500)
        self._unpin_timer.timeout.connect(self._unpin_all)
        self._src_key = None                     # (quality, preset, smoothing) the loaded pictures were prepared with
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(500)
        self._reload_timer.timeout.connect(self._reload_all)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)

        v = QVBoxLayout(self)
        v.addLayout(self._top_bar())
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        host = QWidget()
        self.grid = QGridLayout(host)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.area.setWidget(host)
        v.addWidget(self.area, 1)
        v.addLayout(self._time_bar())
        v.addLayout(self._output_bar())
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888")
        v.addWidget(self.status)
        self.seg_spins: List[QDoubleSpinBox] = []
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence.Save, self, activated=lambda: self._save_project(True))
        QShortcut(QKeySequence.Open, self, activated=self._open_project)
        self._rebuild_jumps()
        self._rebuild_segments()
        self._set_count(4)
        self._quality_changed()
        self._settings_changed()

    # ------------------------------------------------------------------ layout
    def _top_bar(self):
        h = QHBoxLayout()
        self.open_btn = QPushButton("Open project…")
        self.open_btn.setToolTip("Open a saved project: the photos and every setting come back exactly as they were (Ctrl+O)")
        self.open_btn.clicked.connect(self._open_project)
        self.save_btn = QPushButton("Save project…")
        self.save_btn.setToolTip("Save everything in ONE file, the photos included, so it opens even after the pictures were "
                                 "moved or deleted (Ctrl+S saves to the same file again)")
        self.save_btn.clicked.connect(lambda: self._save_project(False))
        h.addWidget(self.open_btn)
        h.addWidget(self.save_btn)
        h.addSpacing(12)
        h.addWidget(QLabel("Photos:"))
        self.count = QSpinBox(); self.count.setRange(1, 24); self.count.setValue(4)
        self.count.setToolTip("How many panels (one photo each)")
        self.count.valueChanged.connect(self._set_count)
        h.addWidget(self.count)
        h.addWidget(QLabel("Columns:"))
        self.cols = QSpinBox(); self.cols.setRange(1, 8); self.cols.setValue(2)
        self.cols.setToolTip("Panels per row (in this window and in the clip)")
        self.cols.valueChanged.connect(self._settings_changed)
        h.addWidget(self.cols)
        h.addWidget(QLabel("Vectorise as:"))
        self.preset = QComboBox()
        self.preset.addItems(list(TRACE_PRESETS))
        self.preset.setCurrentText("Black and White Logo")
        self.preset.setToolTip("Image Trace preset used for every photo. Threshold only works in the black-and-white presets.")
        self.preset.currentIndexChanged.connect(self._preset_changed)
        h.addWidget(self.preset)
        h.addWidget(QLabel("Quality:"))
        self.tile = QSlider(Qt.Horizontal); self.tile.setRange(160, 4000); self.tile.setSingleStep(40); self.tile.setPageStep(200)
        self.tile.setValue(1200); self.tile.setFixedWidth(200)
        self.tile.setToolTip("Resolution every photo is vectorised at (long side). Higher = smoother, more exact outlines; "
                             "the export takes longer (about 4x per doubling). Noise grows with it, so a Noise value looks the same at every quality.")
        self.tile.valueChanged.connect(self._quality_changed)
        h.addWidget(self.tile)
        self.quality_lbl = QLabel("")
        self.quality_lbl.setMinimumWidth(150)
        h.addWidget(self.quality_lbl)
        self.soften = QCheckBox("Smooth outlines while the Threshold changes")
        self.soften.setToolTip("Blurs the grey picture a little before the Threshold is applied, so edges move smoothly "
                               "from frame to frame instead of jumping, and outlines lose their pixel steps.")
        self.soften.toggled.connect(self._quality_changed)
        h.addWidget(self.soften)
        self.colors_btn = QToolButton()
        self.colors_btn.setText("Warhol colours ▾")
        self.colors_btn.setPopupMode(QToolButton.InstantPopup)
        self.colors_btn.setToolTip("Give every photo its own pair of drawing / paper colours from a pop-art palette. "
                                   "Fine-tune each photo with its Colours buttons.")
        menu = QMenu(self.colors_btn)
        for name in WARHOL:
            act = QAction(name, menu)
            act.triggered.connect(lambda _=False, n=name: self._apply_palette(n))
            menu.addAction(act)
        menu.addSeparator()
        menu.addAction("Shuffle the colours", self._shuffle_colors)
        menu.addAction("Back to black on white", self._reset_colors)
        self.colors_btn.setMenu(menu)
        h.addWidget(self.colors_btn)
        h.addWidget(QLabel("Panels show:"))
        self.show_what = QComboBox(); self.show_what.addItems(["Vector", "Photo"])
        self.show_what.currentIndexChanged.connect(lambda *_: self._render_all(force=True))
        h.addWidget(self.show_what)
        h.addStretch(1)
        return h

    def _time_bar(self):
        v = QVBoxLayout()
        h = QHBoxLayout()
        self.play = QPushButton("▶ Play")
        self.play.setCheckable(True)
        self.play.toggled.connect(self._toggle_play)
        h.addWidget(self.play)
        self.jump_lay = QHBoxLayout()
        self.jump_lay.setSpacing(2)
        h.addLayout(self.jump_lay)
        self.time = QSlider(Qt.Horizontal)
        self.time.valueChanged.connect(self._time_changed)
        h.addWidget(self.time, 1)
        self.time_lbl = QLabel("")
        self.time_lbl.setMinimumWidth(230)
        h.addWidget(self.time_lbl)
        v.addLayout(h)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Duration:"))
        self.duration = QDoubleSpinBox(); self.duration.setRange(0.2, 600); self.duration.setDecimals(1); self.duration.setValue(10.0)
        self.duration.setSuffix(" s"); self.duration.setToolTip("How long the sliders take from start to end (the sum of the times between the points). "
                                 "Changing it stretches or shrinks every part in proportion.")
        h2.addWidget(self.duration)
        h2.addWidget(QLabel("FPS:"))
        self.fps = QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(25)
        h2.addWidget(self.fps)
        h2.addWidget(QLabel("Slider points:"))
        self.points = QSpinBox(); self.points.setRange(2, 12); self.points.setValue(5)
        self.points.setToolTip("Handles on every slider: start, end and the points in between (5 = 3 extra points). "
                               "Every photo reaches its point 2, point 3 … in the same moment, spread evenly over the duration.")
        self.points.valueChanged.connect(self._points_changed)
        h2.addWidget(self.points)
        h2.addWidget(QLabel("Rest at start:"))
        self.hold_a = QDoubleSpinBox(); self.hold_a.setRange(0, 30); self.hold_a.setDecimals(1); self.hold_a.setValue(0.5); self.hold_a.setSuffix(" s")
        h2.addWidget(self.hold_a)
        h2.addWidget(QLabel("at end:"))
        self.hold_b = QDoubleSpinBox(); self.hold_b.setRange(0, 30); self.hold_b.setDecimals(1); self.hold_b.setValue(0.5); self.hold_b.setSuffix(" s")
        h2.addWidget(self.hold_b)
        h2.addWidget(QLabel("Easing:"))
        self.ease = QComboBox(); self.ease.addItems(["Linear", "Smooth (slows down at every point)"])
        h2.addWidget(self.ease)
        h2.addStretch(1)
        self.duration.valueChanged.connect(self._duration_edited)
        for w in (self.fps, self.hold_a, self.hold_b):
            w.valueChanged.connect(self._settings_changed)
        self.ease.currentIndexChanged.connect(self._settings_changed)
        v.addLayout(h2)
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Time between the points:"))
        self.seg_lay = QHBoxLayout()
        self.seg_lay.setSpacing(4)
        h3.addLayout(self.seg_lay)
        even = QPushButton("Even")
        even.setToolTip("Make all parts equally long again")
        even.clicked.connect(lambda: self._rebuild_segments(even=True))
        h3.addWidget(even)
        self.frames_lbl = QLabel("")
        self.frames_lbl.setStyleSheet("color:#888")
        h3.addWidget(self.frames_lbl)
        h3.addStretch(1)
        v.addLayout(h3)
        h4 = QHBoxLayout()
        self.fill_on = QCheckBox("Noise fill overlay")
        self.fill_on.setToolTip("Photos with a pure white background never turn black through the Threshold. With this on, the drawing "
                                "colour spreads over the whole picture like ink soaking in, as an overlay on top of the moving sliders. "
                                "It adds no time to the clip: it is complete at the end of the movement. Global: one time for all photos.")
        self.fill_s = QDoubleSpinBox(); self.fill_s.setRange(0.0, 600); self.fill_s.setDecimals(1); self.fill_s.setValue(0.0); self.fill_s.setPrefix("starts at "); self.fill_s.setSuffix(" s")
        self.fill_s.setToolTip("When the spreading starts, counted from the start of the movement (0 = together with the image, independent of the "
                                "Threshold sliders). It always ends where the movement ends; the same for every photo.")
        self.fill_noise = QSlider(Qt.Horizontal); self.fill_noise.setRange(0, 100); self.fill_noise.setValue(45); self.fill_noise.setFixedWidth(120)
        self.fill_noise.setToolTip("0 = an even growth around the drawing, 100 = mostly noise (blobs appear away from the drawing, grow and merge). The shapes are vectors with smooth curves.")
        self.fill_noise_lbl = QLabel("45 %")
        self.noise_edit = QPushButton("Noise editor…")
        self.noise_edit.setToolTip("A window that shows the fill of one photo at several moments, the noise map behind it and sliders to "
                                   "fit the noise to that picture (for all photos or just that one).")
        self.noise_edit.clicked.connect(self._open_noise_editor)
        h4.addWidget(self.fill_on)
        h4.addWidget(self.fill_s)
        h4.addWidget(QLabel("Noise:"))
        h4.addWidget(self.fill_noise)
        h4.addWidget(self.fill_noise_lbl)
        h4.addWidget(self.noise_edit)
        h4.addStretch(1)
        v.addLayout(h4)
        self.fill_on.toggled.connect(self._fill_changed)
        self.fill_s.valueChanged.connect(self._settings_changed)
        self.fill_s.valueChanged.connect(lambda *_: self.noise_dlg is not None and self.noise_dlg._schedule())
        self.fill_noise.valueChanged.connect(self._fill_slider)
        self._fill_ui()
        # layers: the photos' masks split every picture into facade and graffiti, each with its own visibility over time
        h5 = QHBoxLayout()
        self.layers_on = QCheckBox("Layers")
        self.layers_on.setToolTip("Photos with a mask (button 'Mask…' on the photo) are split into graffiti (the mask) and facade (the rest). "
                                  "Each layer gets its own visibility over time, the facade can also be blurred: the graffiti is "
                                  "'extracted' from its background while the clip plays. Photos without a mask are not changed.")

        def spin(v, tip):
            sp = QSpinBox(); sp.setRange(0, 100); sp.setValue(v); sp.setSuffix(" %"); sp.setToolTip(tip); sp.setFixedWidth(64)
            return sp
        self.fac0, self.fac1 = spin(100, "Facade visibility at the start"), spin(0, "Facade visibility at the end (0 % = only paper where the facade is)")
        self.blur0, self.blur1 = spin(0, "Facade blur at the start"), spin(40, "Facade blur at the end")
        self.gra0, self.gra1 = spin(100, "Graffiti visibility at the start"), spin(100, "Graffiti visibility at the end")
        self.lbl_fac, self.lbl_blur, self.lbl_gra = QLabel(), QLabel(), QLabel()
        h5.addWidget(self.layers_on)
        self.photo_on = QCheckBox("Facade = original photo")
        self.photo_on.setToolTip("The facade stays the ORIGINAL PHOTO (not a vector); only the masked graffiti is vectorized. "
                                 "'Facade' is then how much of the photo is left (it fades to paper) and 'Graffiti vector' how much of the "
                                 "masked part is vector instead of photo. Ticking it sets up the sequence: photo → vector of the graffiti → "
                                 "photo fades out → only the graffiti is left.")
        h5.addWidget(self.photo_on)
        self.only_on = QCheckBox("Vectorise only inside the mask")
        self.only_on.setToolTip("The vectoriser only ever sees the masked graffiti: no shapes, colours or Threshold measurements come from the "
                                "facade, the vector ends cleanly at the mask's edge, and it is faster. The background stays the original photo; fade it out "
                                "over time with the Facade slider (and the 'layers move from … to …' times). Always on with 'Facade = original photo'.")
        h5.addWidget(self.only_on)
        self.seq_btn = QPushButton("Sequence: vector → photo")
        self.seq_btn.setToolTip("Sets up: the graffiti is vectorised ONLY inside the mask at the first Threshold point, the background is transparent "
                                "(paper); while the Threshold goes to 0 at the last point, the background photo becomes more and more visible, "
                                "until it is 100 % opaque. Export it with 'Export transparent…' to keep the background really transparent.")
        self.seq_btn.clicked.connect(self._vector_to_photo)
        h5.addWidget(self.seq_btn)
        for lab, a, b in ((self.lbl_fac, self.fac0, self.fac1), (self.lbl_blur, self.blur0, self.blur1), (self.lbl_gra, self.gra0, self.gra1)):
            h5.addWidget(lab); h5.addWidget(a); h5.addWidget(QLabel("→")); h5.addWidget(b)
        h5.addStretch(1)
        v.addLayout(h5)
        h6 = QHBoxLayout()
        h6.addSpacing(24)
        h6.addWidget(QLabel("The layers move from"))
        self.layer_a = QDoubleSpinBox(); self.layer_a.setRange(0, 600); self.layer_a.setDecimals(1); self.layer_a.setSuffix(" s")
        self.layer_a.setToolTip("When the layer movement starts, counted from the start of the movement (0 = together with the image)")
        self.layer_b = QDoubleSpinBox(); self.layer_b.setRange(0, 600); self.layer_b.setDecimals(1); self.layer_b.setSuffix(" s")
        self.layer_b.setSpecialValueText("end of movement")
        self.layer_b.setToolTip("When it ends (0 = at the end of the movement)")
        rn = QPushButton("Rename sliders…")
        rn.setToolTip("Give the sliders your own names, e.g. 'Extract ornament' - they show in the photos and in this row.")
        rn.clicked.connect(self._rename_sliders)
        self.layer_lbl = QLabel("")
        self.layer_lbl.setStyleSheet("color:#888")
        h6.addWidget(self.layer_a); h6.addWidget(QLabel("to")); h6.addWidget(self.layer_b); h6.addWidget(rn); h6.addWidget(self.layer_lbl)
        h6.addStretch(1)
        v.addLayout(h6)
        h7 = QHBoxLayout()
        h7.addSpacing(24)
        self.gra_own = QCheckBox("The graffiti has its own timing: from")
        self.gra_own.setToolTip("Off: the graffiti moves in the same time window as the facade. On: it can start earlier or end sooner, "
                                "e.g. the vector appears first and only then the photo fades out.")
        self.gra_a = QDoubleSpinBox(); self.gra_a.setRange(0, 600); self.gra_a.setDecimals(1); self.gra_a.setSuffix(" s")
        self.gra_b = QDoubleSpinBox(); self.gra_b.setRange(0, 600); self.gra_b.setDecimals(1); self.gra_b.setSuffix(" s")
        self.gra_b.setSpecialValueText("end of movement")
        h7.addWidget(self.gra_own); h7.addWidget(self.gra_a); h7.addWidget(QLabel("to")); h7.addWidget(self.gra_b)
        h7.addStretch(1)
        v.addLayout(h7)
        h8 = QHBoxLayout()
        h8.addSpacing(24)
        h8.addWidget(QLabel("Mask finish:"))
        self.m_vector = QCheckBox("Vector outline")
        self.m_vector.setToolTip("The mask's outline is rebuilt as smooth curves and drawn at the picture's own size: no mask pixels, as sharp at 4000 px "
                                 "as at 400 px. The vectors end cleanly at that outline instead of at the stair steps of the pixel mask.")
        self.m_smooth = QSlider(Qt.Horizontal); self.m_smooth.setRange(0, 100); self.m_smooth.setFixedWidth(130)
        self.m_smooth.setToolTip("Rounds the outline of the mask: jaggies and pixel steps disappear (more = rounder, fine details go).")
        self.m_soft = QSlider(Qt.Horizontal); self.m_soft.setRange(0, 100); self.m_soft.setFixedWidth(130)
        self.m_soft.setToolTip("A soft edge: the graffiti fades into the background over a few pixels instead of ending sharply.")
        self.m_lbl = QLabel("")
        self.m_lbl.setStyleSheet("color:#888")
        h8.addWidget(self.m_vector)
        h8.addWidget(QLabel("Smooth")); h8.addWidget(self.m_smooth)
        h8.addWidget(QLabel("Soft edge")); h8.addWidget(self.m_soft)
        h8.addWidget(self.m_lbl)
        h8.addStretch(1)
        v.addLayout(h8)
        for w_ in (self.m_vector, self.m_smooth, self.m_soft):
            (w_.toggled if isinstance(w_, QCheckBox) else w_.valueChanged).connect(self._mask_finish_changed)
        self.layers_on.toggled.connect(self._layers_changed)
        self.photo_on.toggled.connect(self._layers_changed)
        self.photo_on.clicked.connect(self._photo_clicked)                       # only by the user: sets up the sequence
        self.only_on.toggled.connect(self._layers_changed)
        self.gra_own.toggled.connect(self._layers_changed)
        for w_ in (self.fac0, self.fac1, self.blur0, self.blur1, self.gra0, self.gra1, self.layer_a, self.layer_b, self.gra_a, self.gra_b):
            w_.valueChanged.connect(self._layers_changed)
        self._apply_names()
        self._layers_ui()
        return v

    def _output_bar(self):
        v = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Clip width:"))
        self.width_ = QSpinBox(); self.width_.setRange(320, 7680); self.width_.setSingleStep(160); self.width_.setValue(1920); self.width_.setSuffix(" px")
        self.width_.valueChanged.connect(lambda *_: self._quality_changed())
        h.addWidget(self.width_)
        h.addWidget(QLabel("Gap:"))
        self.gap = QSpinBox(); self.gap.setRange(0, 200); self.gap.setValue(12); self.gap.setSuffix(" px")
        h.addWidget(self.gap)
        h.addWidget(QLabel("Background:"))
        self.bg = QComboBox(); self.bg.addItems(["White", "Dark", "Colour…"])
        self.bg.setToolTip("Colour between the pictures. 'Colour…' lets you choose any colour.")
        self.bg.currentIndexChanged.connect(self._bg_changed)
        self.bg.activated.connect(lambda i: self.bg_btn._pick() if i == 2 else None)       # a click on 'Colour…' asks for the colour
        h.addWidget(self.bg)
        self.bg_btn = ColorButton("   ", (255, 255, 255), "Background colour")
        self.bg_btn.set_color((255, 214, 0))
        self.bg_btn.setVisible(False)
        self.bg_btn.changed.connect(lambda: self._render_all(force=True))
        h.addWidget(self.bg_btn)
        self.invert = QCheckBox("Invert colours (black → white)")
        self.invert.toggled.connect(lambda *_: self._render_all(force=True))
        h.addWidget(self.invert)
        self.smooth = QCheckBox("Smooth edges")
        self.smooth.setChecked(True)
        h.addWidget(self.smooth)
        h.addStretch(1)
        self.timeline_btn = QPushButton("Timeline window…")
        self.timeline_btn.setToolTip("One photo on a video-editor style timeline: keyframes for every slider (Threshold, Noise, Paths, background photo, "
                                     "graffiti, blur), a playhead, prerendered preview. Starts from this photo's current movement.")
        self.timeline_btn.clicked.connect(lambda: self._open_timeline())
        h.addWidget(self.timeline_btn)
        self.scrub_btn = QPushButton("Scrub window…")
        self.scrub_btn.setToolTip("Prerender the frames of one photo, then slide through the Threshold freely - and decide whether the "
                                  "background photo fades with the slider or stays.")
        self.scrub_btn.clicked.connect(self._open_scrub)
        h.addWidget(self.scrub_btn)
        self.mp4_btn = QPushButton("Export MP4…")
        self.gif_btn = QPushButton("Export GIF…")
        self.alpha_btn = QPushButton("Export transparent…")
        self.alpha_btn.setToolTip("Video with a really transparent background (alpha): ProRes 4444 .mov, WebM .webm or a PNG sequence. "
                                  "MP4 and GIF cannot be transparent - there the background is the paper colour.")
        self.alpha_btn.clicked.connect(lambda: self._export("alpha"))
        self.mp4_btn.clicked.connect(lambda: self._export("mp4"))
        self.gif_btn.clicked.connect(lambda: self._export("gif"))
        h.addWidget(self.mp4_btn)
        h.addWidget(self.gif_btn)
        h.addWidget(self.alpha_btn)
        v.addLayout(h)
        h3 = QHBoxLayout()
        self.bar = QProgressBar(); self.bar.setVisible(False)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_export)
        h3.addWidget(self.bar, 1)
        h3.addWidget(self.cancel_btn)
        v.addLayout(h3)
        return v

    # ------------------------------------------------------------------ settings
    def settings(self) -> MultiSettings:
        return MultiSettings(duration=self.duration.value(), seg_seconds=tuple(sp.value() for sp in self.seg_spins) or None, fps=self.fps.value(), hold_start=self.hold_a.value(),
                             hold_end=self.hold_b.value(), ease=EASES[self.ease.currentIndex()],
                             preset=self.preset.currentText(), tile=self.tile.value(), soften=self.soften.isChecked(), points=self.points.value(), fill=self.fill_on.isChecked(),
                             layers=self.layers_on.isChecked(), fac0=self.fac0.value(), fac1=self.fac1.value(), gra0=self.gra0.value(),
                             gra1=self.gra1.value(), blur0=self.blur0.value(), blur1=self.blur1.value(), layer_start=self.layer_a.value(),
                             layer_end=self.layer_b.value(), facade_photo=self.photo_on.isChecked(), only_mask=self._only_mask(), gra_own=self.gra_own.isChecked(),
                             gra_start=self.gra_a.value(), gra_end=self.gra_b.value(), fill_start=self.fill_s.value(), **{**self.fill_cfg, "fill_noise": self.fill_noise.value() / 100.0}, columns=self.cols.value(),
                             gap=self.gap.value(), width=self.width_.value(), bg=["white", "dark", "white"][self.bg.currentIndex()],
                             bg_color=self.bg_btn.color() if self.bg.currentIndex() == 2 else None,
                             invert=self.invert.isChecked(), smooth=self.smooth.isChecked(), mask_smooth=float(self.m_smooth.value()),
                             mask_soft=float(self.m_soft.value()), mask_vector=self.m_vector.isChecked())

    def specs(self) -> List[PanelSpec]:
        return [p.spec() for p in self.panels]

    def _settings_changed(self, *_) -> None:
        st = self.settings()
        total = st.total_frames()
        self.time.blockSignals(True)
        self.time.setRange(0, total - 1)
        self.frame = min(self.frame, total - 1)
        self.time.setValue(self.frame)
        self.time.blockSignals(False)
        parts = " + ".join(str(f) for f in st.segment_frames())
        self.frames_lbl.setText(f"→ {st.move_frames()} frames of movement ({parts}), {total} in the clip")
        self.play_timer.setInterval(max(10, int(1000 / st.fps)))
        for p in self.panels:
            p.show_info(st)
        self._arrange()
        self._update_time_label()
        self._request_all()

    def _bg_changed(self, i: int) -> None:
        self.bg_btn.setVisible(i == 2)
        self._render_all(force=True)

    # ------------------------------------------------------------------ colours
    def _apply_palette(self, name: str, shuffle: int = -1) -> None:
        pairs = palette_pairs(name, len(self.panels), shuffle)
        for p, (ink, paper) in zip(self.panels, pairs):
            p.ink_btn.set_color(ink)
            p.paper_btn.set_color(paper)
        self._last_palette = name
        self._request_all()
        self.status.setText(f"Colours: {name}" + (" (shuffled)" if shuffle >= 0 else ""))

    def _shuffle_colors(self) -> None:
        import random
        self._apply_palette(getattr(self, "_last_palette", next(iter(WARHOL))), random.randrange(10 ** 6))

    def _reset_colors(self) -> None:
        for p in self.panels:
            p.ink_btn.set_color(None)
            p.paper_btn.set_color(None)
        self._request_all()
        self.status.setText("Colours: black on white")

    def _preset_changed(self, *_) -> None:
        for p in self.panels:
            p.set_paths_default(self._preset_paths())
        self._reload_all()

    def _preset_paths(self) -> int:
        from .trace import params_from_preset
        return int(params_from_preset(self.preset.currentText()).paths)

    # ------------------------------------------------------------------ layers and masks
    def _photo_clicked(self, on: bool) -> None:
        """Ticking 'Facade = original photo': set up photo → vector of the graffiti → photo fades out → only the graffiti (editable after)."""
        if not on:
            return
        move = max(1.0, (self.settings().move_frames() - 1) / max(1, self.fps.value()))
        self.layers_on.setChecked(True)
        self.only_on.setChecked(True)                                           # the photo is the facade: only the graffiti is vectorised
        for sp, v in ((self.fac0, 100), (self.fac1, 0), (self.gra0, 0), (self.gra1, 100), (self.blur0, 0), (self.blur1, 0)):
            sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
        for sp, v in ((self.gra_a, 0.0), (self.gra_b, round(move * 0.4, 1)), (self.layer_a, round(move * 0.5, 1)), (self.layer_b, 0.0)):
            sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
        self.gra_own.blockSignals(True); self.gra_own.setChecked(True); self.gra_own.blockSignals(False)
        self._layers_changed()

    def _vector_to_photo(self, *_) -> None:
        """The sequence: vector ONLY in the mask on a transparent background (Threshold = point 1) -> the background photo becomes more
        visible as the Threshold goes to point 2 -> Threshold 0 and the photo 100 % opaque. Photos without a mask are left alone."""
        self.photo_on.blockSignals(True); self.photo_on.setChecked(False); self.photo_on.blockSignals(False)
        self.layers_on.setChecked(True)
        self.only_on.setChecked(True)                                           # (the background photo comes with it)
        for sp, v in ((self.fac0, 0), (self.fac1, 100), (self.gra0, 100), (self.gra1, 100), (self.blur0, 0), (self.blur1, 0)):
            sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
        for sp, v in ((self.layer_a, 0.0), (self.layer_b, 0.0)):
            sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False)
        self.gra_own.blockSignals(True); self.gra_own.setChecked(False); self.gra_own.blockSignals(False)
        n = 0
        for p in self.panels:
            if p.mask is None:
                continue
            first = int(p.spec().thr[0])
            p.set_ends("thr", first if first >= 20 else 150, 0)             # point 1 stays as you set it (150 if it was ~0); point 2 is 0
            n += 1
        self._layers_changed()
        self.status.setText(f"Sequence set for {n} photo(s) with a mask: vector in the mask on a transparent background → the photo fades in as the Threshold goes to 0."
                            if n else "Add a mask to a photo first (button 'Mask…'), then press the button again.")
        self._request_all()

    def _mask_finish_changed(self, *_) -> None:
        self.m_lbl.setText(f"smooth {self.m_smooth.value()}  ·  soft {self.m_soft.value()}")
        self._layers_ui()
        self._request_all()

    def _layers_ui(self) -> None:
        on = self.layers_on.isChecked()
        for w_ in (self.m_vector, self.m_smooth, self.m_soft):
            w_.setEnabled(on)
        for w_ in (self.fac0, self.fac1, self.blur0, self.blur1, self.gra0, self.gra1, self.layer_a, self.layer_b, self.photo_on, self.gra_own, self.only_on):
            w_.setEnabled(on)
        for w_ in (self.gra_a, self.gra_b):
            w_.setEnabled(on and self.gra_own.isChecked())
        self.only_on.setEnabled(on and not self.photo_on.isChecked())           # with the photo as facade it is always on
        self.lbl_gra.setText(self.names["gra"] + (" vector:" if self.photo_on.isChecked() else ":"))
        have = sum(1 for p in self.panels if p.mask is not None)
        self.layer_lbl.setText(f"{have} of {len(self.panels)} photos have a mask" if on else "")

    def _layers_changed(self, *_) -> None:
        self._layers_ui()
        self._settings_changed()
        self._dirty = set(range(len(self.panels)))
        self._render_timer.start()

    def _apply_names(self) -> None:
        for p in self.panels:
            p.set_names(self.names)
        self.lbl_fac.setText(self.names["fac"] + ":")
        self.lbl_blur.setText(self.names["blur"] + ":")
        self.lbl_gra.setText(self.names["gra"] + (" vector:" if self.photo_on.isChecked() else ":"))

    def _rename_sliders(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Rename sliders")
        form = QFormLayout(dlg)
        edits = {}
        for key, text in (("thr", "Threshold slider"), ("noise", "Noise slider"), ("paths", "Paths slider"),
                          ("fac", "Facade layer"), ("blur", "Facade blur"), ("gra", "Graffiti layer")):
            e = QLineEdit(self.names[key])
            e.setPlaceholderText(DEFAULT_NAMES[key])
            form.addRow(text, e)
            edits[key] = e
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(lambda: [edits[k].setText(DEFAULT_NAMES[k]) for k in edits])
        form.addRow(bb)
        if dlg.exec() == QDialog.Accepted:
            self.set_names({k: e.text().strip() or DEFAULT_NAMES[k] for k, e in edits.items()})

    def set_names(self, names: dict) -> None:
        self.names = {**DEFAULT_NAMES, **{k: str(v)[:60] for k, v in names.items() if k in DEFAULT_NAMES}}
        self._apply_names()

    def _edit_image(self, path: str) -> np.ndarray:
        from .imageio_ import load_image_alpha
        from .mask_editor import MAX_EDIT
        from .rows import fit_size, image_size
        iw, ih = image_size(path)
        tw, th = fit_size("fit_long", MAX_EDIT, MAX_EDIT, iw / max(1, ih))
        rgb, _ = load_image_alpha(path, (tw, th))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _get_ai(self):
        if self.ai is None:
            from .ai_client import AIClient
            self.ai = AIClient()
        return self.ai

    def _edit_mask(self, i: int) -> None:
        from .mask_editor import MaskEditor
        if i >= len(self.panels) or not self.panels[i].path:
            self.status.setText("Load a picture first.")
            return
        p = self.panels[i]
        try:
            bgr = self._edit_image(p.path)
        except Exception as exc:
            self.status.setText(f"Could not read the picture: {exc}")
            return
        dlg = MaskEditor(bgr, p.mask, f"Photo {i + 1}", self, ai_factory=self._get_ai)
        if dlg.exec() == QDialog.Accepted:
            m = dlg.result()
            self.set_panel_mask(i, m if m is not None and m.any() else None)
            if not self.layers_on.isChecked() and p.mask is not None:
                self.status.setText("Mask saved. Tick 'Layers' under the time slider to use it.")

    def set_panel_mask(self, i: int, m: Optional[np.ndarray]) -> None:
        if i < len(self.panels):
            self.panels[i].set_mask(m)
            self._layers_ui()

    def _open_scrub(self, autostart: bool = True):
        """A window with the prerendered frames of one photo (slide the Threshold freely; the background fades with it or stays)."""
        from .scrub_window import ScrubWindow
        entries = []
        for i, p in enumerate(self.panels):
            if p.src is None:
                continue
            self._attach_mask(p)
            entries.append((f"Photo {i + 1}", p.src, p.spec(), getattr(p, "base_opaque", p.src.opaque)))
        if not entries:
            self.status.setText("Load a picture first.")
            return None
        self._scrub = ScrubWindow(entries, self.settings(), self, autostart=bool(autostart))
        self._scrub.setAttribute(Qt.WA_DeleteOnClose, True)
        self._scrub.show()
        return self._scrub

    def _open_timeline(self, index: Optional[int] = None, autostart: bool = True):
        """A Timeline window for one photo, starting from this photo's movement here (its slider points and the layer settings)."""
        from .timeline_window import TimelineWindow
        loaded = [i for i, p in enumerate(self.panels) if p.path]
        if not loaded:
            self.status.setText("Load a picture first.")
            return None
        if index is None:
            index = loaded[0]
            if len(loaded) > 1:
                from PySide6.QtWidgets import QInputDialog
                item, ok = QInputDialog.getItem(self, "Timeline", "Which photo?", [f"Photo {i + 1}" for i in loaded], 0, False)
                if not ok:
                    return None
                index = loaded[[f"Photo {i + 1}" for i in loaded].index(item)]
        p = self.panels[index]
        st = self.settings()
        self._timeline = TimelineWindow(self, autostart=bool(autostart))
        self._timeline.setAttribute(Qt.WA_DeleteOnClose, True)
        self._timeline.load_from_multi(p.path, p.mask, p.spec(), st)
        self._timeline.show()
        return self._timeline

    def _mask_style(self):
        from .layers import MaskStyle
        return MaskStyle(float(self.m_smooth.value()), float(self.m_soft.value()), self.m_vector.isChecked())

    def _only_mask(self) -> bool:
        """Vectorise only inside the mask: its own box, and always when the photo itself is the facade."""
        return self.only_on.isChecked() or self.photo_on.isChecked()

    def _attach_mask(self, p) -> None:
        """The panel's source carries the mask at its own size (the sweep workers get it with the picture) and, with 'only inside the
        mask', the traced region: the tracer then never sees the facade."""
        only = self.layers_on.isChecked() and self._only_mask() and p.mask is not None
        style = self._mask_style()
        tag = (p.mask_rev, only, style.key())
        if p.src is not None and getattr(p, "_attached", None) != tag:
            from .layers import styled_mask, traced_region
            base = getattr(p, "base_opaque", p.src.opaque)
            m = None if p.mask is None else styled_mask(p.mask, p.src.bgr.shape[1], p.src.bgr.shape[0], style)
            p.src = replace(p.src, mask=m, opaque=traced_region(base, m, p.src.bgr.shape) if only else base)
            if getattr(p, "_attached", None) is not None:
                p.cache, p.curve = {}, None                       # traces and the measured black point belong to the old region
            p._attached = tag

    def _fill_slider(self, v: int) -> None:
        self.fill_cfg["fill_noise"] = v / 100.0
        self._fill_changed()

    def _open_noise_editor(self) -> None:
        from .fill_dialog import FillNoiseDialog
        if self.noise_dlg is not None:
            self.noise_dlg.close()
        first = next((i for i, p in enumerate(self.panels) if p.src is not None), 0)
        self.noise_dlg = FillNoiseDialog(self, first)
        self.noise_dlg.changed.connect(self._noise_edited)
        self.noise_dlg.finished.connect(lambda *_: setattr(self, "noise_dlg", None))
        self.noise_dlg.show()

    def _noise_edited(self) -> None:
        """The editor changed a value: the main window follows (slider, previews)."""
        v = int(round(self.fill_cfg["fill_noise"] * 100))
        if v != self.fill_noise.value():
            self.fill_noise.setValue(v)                  # also refreshes everything
            return
        self._fill_changed()
        for p in self.panels:
            p.show_info(self.settings())

    def _fill_ui(self) -> None:
        on = self.fill_on.isChecked()
        self.fill_s.setEnabled(on)
        self.fill_noise.setEnabled(on)
        self.fill_noise_lbl.setText(f"{self.fill_noise.value()} %")

    def _fill_changed(self, *_) -> None:
        self._fill_ui()
        for p in self.panels:
            p.cache.clear()
            p.key = None
        self._settings_changed()

    def _rebuild_segments(self, even: bool = False) -> None:
        """One box per part between two slider points. A change of the number of points (or 'Even') divides the duration equally."""
        n = self.points.value() - 1
        total = self.duration.value()
        old = [sp.value() for sp in self.seg_spins]
        while self.seg_lay.count():
            it = self.seg_lay.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        self.seg_spins = []
        vals = old if (len(old) == n and not even) else [total / n] * n
        for k in range(n):
            lab = QLabel(f"{k + 1}→{k + 2}")
            lab.setStyleSheet("color:#888;font-size:10px")
            sp = QDoubleSpinBox(); sp.setRange(0.04, 600); sp.setDecimals(2); sp.setSuffix(" s"); sp.setValue(vals[k])
            sp.setFixedWidth(74)
            sp.setToolTip(f"Time every photo takes from its slider point {k + 1} to point {k + 2}")
            sp.valueChanged.connect(self._segments_edited)
            self.seg_lay.addWidget(lab)
            self.seg_lay.addWidget(sp)
            self.seg_spins.append(sp)
        self._segments_edited()

    def _segments_edited(self, *_) -> None:
        total = sum(sp.value() for sp in self.seg_spins)
        self.duration.blockSignals(True)
        self.duration.setValue(total)
        self.duration.blockSignals(False)
        self._settings_changed()

    def _duration_edited(self, v: float) -> None:
        cur = sum(sp.value() for sp in self.seg_spins)
        k = v / cur if cur > 0 else 1.0
        for sp in self.seg_spins:
            sp.blockSignals(True)
            sp.setValue(max(0.04, sp.value() * k))
            sp.blockSignals(False)
        self._settings_changed()

    def _points_changed(self, n: int) -> None:
        for p in self.panels:
            p.set_points(n)
        self._rebuild_jumps()
        self._rebuild_segments()
        self._settings_changed()

    def _rebuild_jumps(self) -> None:
        while self.jump_lay.count():
            it = self.jump_lay.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        n = self.points.value()
        for k in range(n):
            b = QPushButton(str(k + 1))
            b.setFixedWidth(28)
            b.setToolTip("Jump to the moment where every photo is at its slider point " + str(k + 1) +
                         ". Set the same point of every photo while looking at them side by side.")
            b.clicked.connect(lambda _=None, kk=k: self._jump(kk))
            self.jump_lay.addWidget(b)

    def _jump(self, k: int) -> None:
        self.time.setValue(point_frame(self.settings(), k, self.points.value()))

    def _arrange(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        c = max(1, self.cols.value())
        for i, p in enumerate(self.panels):
            self.grid.addWidget(p, i // c, i % c)

    def _set_count(self, n: int) -> None:
        while len(self.panels) < n:
            p = Panel(len(self.panels), self.points.value())
            p.set_paths_default(self._preset_paths())
            p.mask_requested.connect(self._edit_mask)
            p.set_names(self.names)
            p.fit_requested.connect(lambda i: self._to_black(i, "fit"))
            p.preview_point.connect(self._preview_point)
            p.edited.connect(self._panel_edited)
            p.load_requested.connect(self._load_dialog)
            p.file_dropped.connect(self._set_path)
            p.black_requested.connect(lambda i: self._to_black(i, "end"))
            self.panels.append(p)
        while len(self.panels) > n:
            p = self.panels.pop()
            p.setParent(None)
            p.deleteLater()
        self._settings_changed()

    # ------------------------------------------------------------------ pictures
    def _load_dialog(self, i: int) -> None:
        f, _ = QFileDialog.getOpenFileName(self, f"Picture for photo {i + 1}", "",
                                           "Images (" + " ".join("*" + e for e in IMG_EXT) + ")")
        if f:
            self._set_path(i, f)

    def _set_path(self, i: int, path: str) -> None:
        if i >= len(self.panels):
            return
        p = self.panels[i]
        p.set_picture(path)
        if path:
            st = self.settings()
            self._src_key = self._src_sig(st)
            self.workers.submit(self._load_source, i, path, st)
        self.status.setText("")

    def _load_source(self, i: int, path: str, st: MultiSettings) -> None:
        try:
            src = load_panel_source(path, st)
        except Exception as exc:                         # unreadable picture
            print("Multi-Trace: could not load", path, exc)
            src = None
        self.sig.sourced.emit(i, src, path)

    def _on_sourced(self, i: int, src, path: str) -> None:
        if i >= len(self.panels) or self.panels[i].path != path:
            return
        p = self.panels[i]
        if src is None:
            p.view.setText("could not read this picture")
            return
        p.src = src
        p.base_opaque = src.opaque                       # the picture's own transparency, before the mask narrows the traced region
        p._attached = None
        self._attach_mask(p)
        self._dirty.add(i)
        self._render_timer.start()

    def _src_sig(self, st: MultiSettings):
        return (st.tile, st.preset, st.soften)

    def _quality_changed(self, *_) -> None:
        st = self.settings()
        cell = st.width / max(1, st.columns)
        self.quality_lbl.setText(f"{st.tile} px" + ("  (slow export)" if st.tile > 2400 else ""))
        self.quality_lbl.setToolTip(f"One picture of the clip is about {cell:.0f} px wide; higher qualities are scaled down to it, "
                                    "which makes the edges smoother.")
        self._reload_timer.start()

    def _reload_all(self, *_) -> None:
        st = self.settings()
        if self._src_sig(st) == self._src_key:
            return
        self._src_key = self._src_sig(st)
        for i, p in enumerate(self.panels):
            if p.path:
                path = p.path
                p.src, p.key, p.cache, p.curve = None, None, {}, None
                self.workers.submit(self._load_source, i, path, st)
        self._settings_changed()

    # ------------------------------------------------------------------ preview
    def _panel_edited(self, i: int) -> None:
        if i < len(self.panels):
            self._attach_mask(self.panels[i])
            self.panels[i].show_info(self.settings())
            self._dirty.add(i)
            self._render_timer.start()
            self._update_markers()

    def _preview_point(self, i: int, k: int, held: bool) -> None:
        """While a handle is held (or a number box was just edited) the panel shows the picture as it is at that slider point,
        with every slider of the photo at its value for that point; afterwards it goes back to the moment of the time slider."""
        if i >= len(self.panels):
            return
        if k < 0:
            self.pin.pop(i, None)
            self.panels[i].set_pinned(-1)
        else:
            self.pin[i] = k
            self.panels[i].set_pinned(k)
            if not held:
                self._unpin_timer.start()
        self._dirty.add(i)
        self._render_timer.start()

    def _unpin_all(self) -> None:
        for i in list(self.pin):
            self._preview_point(i, -1, True)

    def _pinned_values(self, spec: PanelSpec, k: int) -> dict:
        out = {"threshold": float(spec.thr[min(k, len(spec.thr) - 1)]), "noise": float(spec.noise[min(k, len(spec.noise) - 1)])}
        if spec.paths is not None:
            out["paths"] = float(spec.paths[min(k, len(spec.paths) - 1)])
        return out

    def _request_all(self) -> None:
        self._dirty |= set(range(len(self.panels)))
        self._render_timer.start()
        self._update_markers()

    def _update_markers(self) -> None:
        st = self.settings()
        for p in self.panels:
            p.set_marker(p.spec(), st, self.frame)

    def _render_all(self, force: bool = False) -> None:
        st = self.settings()
        self.gen += 1
        idx = set(range(len(self.panels))) if force else (self._dirty or set(range(len(self.panels))))
        self._dirty = set()
        for i in sorted(k for k in idx if k < len(self.panels)):
            p = self.panels[i]
            if p.src is None:
                continue
            self._attach_mask(p)                                        # (mask / 'only inside the mask' may have changed)
            if self.show_what.currentIndex() == 1:
                p.show_image(p.src.bgr)
                continue
            vals = self._pinned_values(p.spec(), self.pin[i]) if i in self.pin else values_at(p.spec(), st, self.frame)
            sp = p.spec()
            key = (values_key(vals), st.bg, st.invert, st.preset, st.tile, st.soften, tuple(st.fill_style().items()), tuple(sorted((sp.fill_over or {}).items())),
                   p.mask_rev if st.layers else 0, st.layers and st.only_mask, self._mask_style().key() if st.layers else 0, tile_colors(sp, st), sp.flip)
            if key == p.key and not force:
                continue
            self._latest[i] = key
            self.workers.submit(self._render_one, i, p.src, st, vals, p.cache, key, self.gen, sp.flip, *tile_colors(sp, st), sp.fill_over)

    def _render_one(self, i: int, src, st: MultiSettings, vals: dict, cache: dict, key, gen: int, flip: bool = False,
                    ink=None, paper=None, fill_over=None) -> None:
        if self._latest.get(i) != key:                                  # a newer picture of this photo is already queued: skip this one
            return
        try:
            img = render_preview(src, st, vals, cache, flip, ink, paper, fill_over)
        except Exception as exc:
            print("Multi-Trace: preview failed", exc)
            return
        self.sig.rendered.emit(i, img, key, gen)

    def _on_rendered(self, i: int, img, key, gen: int) -> None:
        if i < len(self.panels) and self.show_what.currentIndex() == 0:
            self.panels[i].key = key
            self.panels[i].show_image(img)

    # ------------------------------------------------------------------ time
    def _time_changed(self, v: int) -> None:
        self.frame = v
        self._update_time_label()
        self._update_markers()
        self._dirty = set(range(len(self.panels)))
        self._render_timer.start()

    def _update_time_label(self) -> None:
        st = self.settings()
        self.time_lbl.setText(f"{self.frame / st.fps:5.2f} s of {st.total_frames() / st.fps:.2f} s   ·   frame {self.frame + 1}/{st.total_frames()}")

    def _toggle_play(self, on: bool) -> None:
        self.play.setText("⏸ Pause" if on else "▶ Play")
        if on:
            self.play_timer.start()
        else:
            self.play_timer.stop()

    def _tick(self) -> None:
        if self.time.maximum() <= 0:
            return
        self.time.setValue(0 if self.frame >= self.time.maximum() else self.frame + 1)

    # ------------------------------------------------------------------ black point
    def _to_black(self, i: int, mode: str = "end") -> None:
        p = self.panels[i]
        if p.src is None:
            return
        if p.curve is not None:
            self._apply_curve(i, p.curve, mode)
            return
        self._mode[i] = mode
        self.status.setText(f"Photo {i + 1}: measuring where it turns black…")
        self.workers.submit(self._measure, i, p.src, p.path)

    def _measure(self, i: int, src, path: str) -> None:
        from .blackpoint import black_curve
        try:
            c = black_curve(measure_source(src))
        except Exception as exc:
            print("Multi-Trace: black point failed", exc)
            c = None
        self.sig.curved.emit(i, (path, c))

    def _on_curved(self, i: int, payload) -> None:
        path, c = payload
        if i >= len(self.panels) or self.panels[i].path != path:
            return
        if c is None:
            self.status.setText(f"Photo {i + 1}: this vectorising preset has no Threshold that turns the picture black.")
            return
        self.panels[i].curve = c
        self._apply_curve(i, c, self._mode.pop(i, "end"))

    def _apply_curve(self, i: int, c, mode: str = "end") -> None:
        p = self.panels[i]
        T = int(np.ceil(c.at(0.5)))
        start = int(p.spec().thr[0])
        if mode == "fit":
            # the white area shrinks by equal steps from point to point: every photo loses its white at the same pace
            n = p.points
            w0, w1 = c.white_at(start), 0.005
            seq = [start] + [int(round(c.threshold_for_white(w0 + (w1 - w0) * k / (n - 1)))) for k in range(1, n - 1)] + [max(T, start)]
            p.set_seq("thr", seq)
            self.status.setText(f"Photo {i + 1}: Threshold points placed at {', '.join(str(x) for x in seq)} (equal steps of white).")
            return
        p.set_ends("thr", start, T)
        self.status.setText(f"Photo {i + 1}: black (≤ 0.5 % white) at Threshold {T}.")

    # ------------------------------------------------------------------ project file
    def project_state(self) -> dict:
        panels = []
        for p in self.panels:
            sp = p.spec()
            panels.append({"name": os.path.basename(p.path), "thr": list(sp.thr), "noise": list(sp.noise),
                           "paths": list(p.paths["slider"].values()), "paths_on": p.paths["use"].isChecked(),
                           "flip": sp.flip, "ink": sp.ink, "paper": sp.paper, "fill_over": sp.fill_over, "mask": None})
        return {"settings": {
            "count": self.count.value(), "columns": self.cols.value(), "preset": self.preset.currentText(),
            "quality": self.tile.value(), "soften": self.soften.isChecked(), "points": self.points.value(),
            "seg_seconds": [sp.value() for sp in self.seg_spins], "duration": self.duration.value(), "fps": self.fps.value(),
            "hold_start": self.hold_a.value(), "hold_end": self.hold_b.value(), "ease": self.ease.currentIndex(),
            "width": self.width_.value(), "gap": self.gap.value(), "bg": self.bg.currentIndex(),
            "bg_color": self.bg_btn.color(), "invert": self.invert.isChecked(), "smooth": self.smooth.isChecked(),
            "show": self.show_what.currentIndex(), "fill": self.fill_on.isChecked(), "fill_start": self.fill_s.value(),
            "fill_noise": self.fill_noise.value(), "fill_style": dict(self.fill_cfg),
            "layers": {"on": self.layers_on.isChecked(), "fac": [self.fac0.value(), self.fac1.value()], "gra": [self.gra0.value(), self.gra1.value()],
                       "blur": [self.blur0.value(), self.blur1.value()], "start": self.layer_a.value(), "end": self.layer_b.value(),
                       "photo": self.photo_on.isChecked(), "only_mask": self.only_on.isChecked(), "mask_vector": self.m_vector.isChecked(), "mask_smooth": self.m_smooth.value(), "mask_soft": self.m_soft.value(), "gra_own": self.gra_own.isChecked(), "gra_start": self.gra_a.value(),
                       "gra_end": self.gra_b.value()},
            "names": dict(self.names)}, "frame": self.frame, "panels": panels}

    def _save_project(self, same_file: bool = False) -> None:
        path = self.project_file if (same_file and self.project_file) else ""
        if not path:
            start = self.project_file or "multi_trace" + PROJECT_EXT
            path, _ = QFileDialog.getSaveFileName(self, "Save project", start, f"Multi-Trace project (*{PROJECT_EXT})")
            if not path:
                return
            if not path.lower().endswith(PROJECT_EXT):
                path += PROJECT_EXT
        try:
            self.status.setText("Saving the project…")
            QApplication.processEvents()
            write_project(path, self.project_state(), [p.path for p in self.panels], [p.mask for p in self.panels])
        except (ProjectError, OSError) as exc:
            self.status.setText("Could not save the project: " + str(exc))
            QMessageBox.warning(self, "Save project", str(exc))
            return
        self.project_file = path
        self.setWindowTitle(f"Multi-Trace – {os.path.basename(path)}")
        self.status.setText(f"Project saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB, photos included)")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", f"Multi-Trace project (*{PROJECT_EXT})")
        if path:
            self.load_project(path)

    def load_project(self, path: str) -> bool:
        """Open a project file: photos and all settings come back. The photos are unpacked into a temporary folder that is removed
        when the window closes."""
        d = tempfile.mkdtemp(prefix="photovec_project_")
        try:
            state, files = read_project(path, d)
            self._apply_state(state, files, read_masks(path, state))
        except (ProjectError, OSError, KeyError, ValueError, TypeError) as exc:
            shutil.rmtree(d, ignore_errors=True)
            self.status.setText("Could not open the project: " + str(exc))
            if self.isVisible():
                QMessageBox.warning(self, "Open project", str(exc))
            return False
        self._temp_dirs.append(d)
        self.project_file = path
        self.setWindowTitle(f"Multi-Trace – {os.path.basename(path)}")
        self.status.setText(f"Project opened: {os.path.basename(path)}")
        return True

    def _apply_state(self, state: dict, files: List[str], masks: Optional[List[Optional[np.ndarray]]] = None) -> None:
        c = state["settings"]
        pan = state["panels"]
        if len(files) != len(pan):
            raise ProjectError("The project file is damaged (photos and settings do not match).")
        n = max(2, min(12, int(c["points"])))
        for d in pan:
            if not (len(d["thr"]) == len(d["noise"]) == len(d["paths"]) == n):
                raise ProjectError("The project file is damaged (slider points do not match).")
        self.play.setChecked(False)
        self.pin.clear()
        # the order matters: the number of panels and points first, then everything that depends on them
        self.count.setValue(max(1, min(24, int(c["count"]))))
        self.points.setValue(max(2, min(12, int(c["points"]))))
        self.cols.setValue(int(c["columns"]))
        self.preset.setCurrentText(c["preset"])
        self.tile.setValue(int(c["quality"]))
        self.soften.setChecked(bool(c["soften"]))
        self.fps.setValue(int(c["fps"]))
        self.hold_a.setValue(float(c["hold_start"]))
        self.hold_b.setValue(float(c["hold_end"]))
        self.ease.setCurrentIndex(max(0, min(len(EASES) - 1, int(c["ease"]))))
        self.width_.setValue(int(c["width"]))
        self.gap.setValue(int(c["gap"]))
        self.bg.setCurrentIndex(max(0, min(2, int(c["bg"]))))
        if c.get("bg_color"):
            self.bg_btn.set_color(c["bg_color"])
        self.invert.setChecked(bool(c["invert"]))
        self.smooth.setChecked(bool(c["smooth"]))
        self.show_what.setCurrentIndex(max(0, min(1, int(c["show"]))))
        self.fill_on.setChecked(bool(c.get("fill", False)))                     # older projects have no fill settings
        self.fill_s.setValue(float(c.get("fill_start", 0.0)))          # older projects had the fill after the movement: now an overlay from the start
        self.fill_cfg.update({k: v for k, v in (c.get("fill_style") or {}).items() if k in self.fill_cfg})
        self.fill_noise.setValue(int(c.get("fill_noise", 45)))
        ly = c.get("layers") or {}                                              # older projects have no layers
        self.layers_on.setChecked(bool(ly.get("on", False)))
        for (a, b), key, dflt in ((((self.fac0, self.fac1)), "fac", (100, 0)), ((self.gra0, self.gra1), "gra", (100, 100)), ((self.blur0, self.blur1), "blur", (0, 40))):
            v = ly.get(key) or dflt
            a.setValue(int(v[0])); b.setValue(int(v[1]))
        self.layer_a.setValue(float(ly.get("start", 0.0)))
        self.layer_b.setValue(float(ly.get("end", 0.0)))
        self.photo_on.setChecked(bool(ly.get("photo", False)))                  # older projects: facade is the vector
        self.only_on.setChecked(bool(ly.get("only_mask", False)))
        self.m_vector.setChecked(bool(ly.get("mask_vector", False)))
        self.m_smooth.setValue(int(ly.get("mask_smooth", 0)))
        self.m_soft.setValue(int(ly.get("mask_soft", 0)))
        self.gra_own.setChecked(bool(ly.get("gra_own", False)))
        self.gra_a.setValue(float(ly.get("gra_start", 0.0)))
        self.gra_b.setValue(float(ly.get("gra_end", 0.0)))
        self.set_names(c.get("names") or {})
        segs = c.get("seg_seconds") or []
        if len(segs) == len(self.seg_spins):
            for sp, v in zip(self.seg_spins, segs):
                sp.blockSignals(True)
                sp.setValue(float(v))
                sp.blockSignals(False)
            self._segments_edited()
        for p, d, f in zip(self.panels, pan, files):
            p.set_seq("thr", d["thr"])
            p.set_seq("noise", d["noise"])
            p.set_seq("paths", d["paths"])
            p.paths["use"].setChecked(bool(d["paths_on"]))
            p.flip.setChecked(bool(d["flip"]))
            p.ink_btn.set_color(d["ink"])
            p.paper_btn.set_color(d["paper"])
            fo = d.get("fill_over")                                             # older projects: no own noise values
            p.fill_over = {k: fo[k] for k in fo if k in self.fill_cfg} if isinstance(fo, dict) and fo else None
        self._src_key = None
        for i, f in enumerate(files):
            self._set_path(i, f)
        for i, p in enumerate(self.panels):                                     # masks after the pictures (a new picture clears its mask)
            p.set_mask(masks[i] if masks and i < len(masks) else None)
        self._layers_ui()
        self.time.setValue(max(0, min(self.time.maximum(), int(state.get("frame", 0)))))
        self._settings_changed()

    # ------------------------------------------------------------------ export
    def _export(self, kind: str) -> None:
        if self.job is not None:
            return
        specs = [s for s in self.specs() if s.path]
        if not specs:
            QMessageBox.information(self, "Export", "Load at least one picture first.")
            return
        if kind == "alpha":
            from .alpha_export import FORMATS
            path, flt = QFileDialog.getSaveFileName(self, "Save transparent clip", "multi_trace_transparent.mov", ";;".join(FORMATS.values()))
            if not path:
                return
            ext = next((k for k in FORMATS if path.lower().endswith("." + k)), next((k for k, v in FORMATS.items() if v == flt), "mov"))
            if not path.lower().endswith("." + ext):
                path += "." + ext
            kind = "alpha:" + ext
        else:
            ext = "mp4" if kind == "mp4" else "gif"
            path, _ = QFileDialog.getSaveFileName(self, "Save clip", f"multi_trace.{ext}", f"{ext.upper()} (*.{ext})")
            if not path:
                return
            if not path.lower().endswith("." + ext):
                path += "." + ext
        self.job = MultiExportJob(self.specs(), self.settings(), path, kind)
        self.job.progress.connect(self._export_progress)
        self.job.done.connect(self._export_done)
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

    def _export_done(self, path: str) -> None:
        self.status.setText(f"Saved {path}")

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

    def closeEvent(self, e):
        if self.noise_dlg is not None:
            self.noise_dlg.close()
        if self.ai is not None:
            self.ai.close()
        self.play_timer.stop()
        for t in (self._render_timer, self._reload_timer, self._unpin_timer):
            t.stop()
        if self.job is not None:
            self.job.cancel()
            self.job.wait(8000)
        self.workers.shutdown(wait=False, cancel_futures=True)
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(e)
