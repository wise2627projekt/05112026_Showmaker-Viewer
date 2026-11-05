"""Clip Studio: slider animations, morph into linework, grids of pictures, GIF / MP4 export."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QElapsedTimer, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QDialog, QDialogButtonBox, QTableWidgetItem, QScrollArea, QFrame, QHeaderView, QStackedWidget, QTableWidget, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QProgressBar, QPushButton, QRadioButton, QSlider, QSpinBox,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .animate import write_gif_stream, write_mp4
from . import memory, pool as pool_mod
from .batch import grid_jobs, job_key, plan_many, resolve_workers, shared_pool, trace_jobs
from .clip import (FIT_NAMES, PACE_NAMES, active_groups, attach_black, warm_curves, sweep_values, ClipSettings, GridLayout, GridRenderer, GridScene, Group, Renderer, Scene, Stage, area_of,
                   default_timeline, grid_timeline, scene_from_result, stage_timing, tile_at, tile_linework)
from .grid_workspace import CellInfo, GridWorkspace
from .linework import PRESETS as LW_PRESETS
from .rows import (SIDES, SIZE_LABEL, SIZE_MODES, STATE_LABEL, STATES, CellSpec, RowSpec, flat_cells, params_for_cell,
                   states_of)
from .sweep import SWEEP_PARAMS, Track, default_track
from .trace import PRESETS as TRACE_PRESETS
from .morph import MorphOptions, load_linework

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif")
DIRECTIONS = [("B enters from the right", "x", "right"), ("B enters from the left", "x", "left"),
              ("B enters from the bottom", "y", "bottom"), ("B enters from the top", "y", "top")]
IMAGES = [("Source photo", "source"), ("Vector", "vector"), ("Linework", "linework"), ("Blank (white)", "blank")]


def _qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    return QImage(np.ascontiguousarray(bgr).data, w, h, 3 * w, QImage.Format_BGR888).copy()


class Canvas(QWidget):
    tile_clicked = Signal(int, bool)           # picture, Ctrl held

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 380)
        self.img: Optional[QImage] = None
        self.msg = "Choose a source and press “Build”."
        self.tiles: list = []              # (picture index, (x0, y0, x1, y1)) in frame pixels
        self.states: Dict[int, str] = {}      # picture -> morph | stay | white
        self.selected: set = set()
        self.picking = False

    def show_frame(self, bgr: Optional[np.ndarray]) -> None:
        self.img = None if bgr is None else _qimage(bgr)
        self.update()

    def _geom(self):
        k = min(self.width() / self.img.width(), self.height() / self.img.height())
        w, h = int(self.img.width() * k), int(self.img.height() * k)
        return k, (self.width() - w) // 2, (self.height() - h) // 2, w, h

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self.img is None:
            p.setPen(QColor(170, 170, 175))
            p.drawText(self.rect(), Qt.AlignCenter, self.msg)
            return
        k, ox, oy, w, h = self._geom()
        p.drawImage(ox, oy, self.img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        if self.picking:                    # selection overlay: only on screen, never in exported clips
            cols = {"morph": QColor(46, 160, 90), "stay": QColor(64, 124, 214), "white": QColor(220, 40, 40)}
            names = {"morph": "morphs", "stay": "stays", "white": "→ white"}
            for i, (x0, y0, x1, y1) in self.tiles:
                r = QRectF(ox + x0 * k, oy + y0 * k, (x1 - x0) * k, (y1 - y0) * k)
                st = self.states.get(i, "morph")
                c = cols[st]
                p.fillRect(r, QColor(c.red(), c.green(), c.blue(), 95 if st == "white" else 28))
                sel = i in self.selected
                p.setPen(QPen(QColor(255, 255, 255) if sel else c, 4 if sel else 3))
                p.drawRect(r.adjusted(1, 1, -1, -1))
                p.setPen(QColor(255, 255, 255))
                p.drawText(r, Qt.AlignCenter, f"#{i + 1}  {names[st]}")

    def mousePressEvent(self, e):
        if not (self.picking and self.img is not None and self.tiles):
            return
        k, ox, oy, _, _ = self._geom()
        i = tile_at(self.tiles, (e.position().x() - ox) / k, (e.position().y() - oy) / k)
        if i is not None:
            self.tile_clicked.emit(i, bool(e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier)))


class TrackEditor(QWidget):
    """Table of sliders to animate: slider, from, to, step per frame, start delay (frames)."""
    changed = Signal()

    def __init__(self, info: Callable[[], dict]):
        super().__init__()
        self.info = info
        self._busy = False
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Slider", "From", "To", "Step", "After"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(150)
        v.addWidget(self.table)
        b = QHBoxLayout()
        for text, fn in (("+ Slider", self._add), ("Remove", self._remove), ("To = main window", self._to_current)):
            btn = QPushButton(text); btn.clicked.connect(fn); b.addWidget(btn)
        v.addLayout(b)

    def names(self) -> List[str]:
        return self.info().get("names") or list(SWEEP_PARAMS)

    def _row(self, t: Track) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        combo = QComboBox()
        names = self.names()
        if t.param not in names:
            names = names + [t.param]
        for n in names:
            combo.addItem(SWEEP_PARAMS[n][0], n)
        combo.setCurrentIndex(combo.findData(t.param))
        combo.currentIndexChanged.connect(lambda *_: self._param_changed(combo))
        self.table.setCellWidget(r, 0, combo)
        _, lo, hi, discrete, _ = SWEEP_PARAMS[t.param]
        for c, val in ((1, t.start), (2, t.end), (3, t.step)):
            sp = QDoubleSpinBox()
            sp.setMinimumWidth(46)
            if c == 3:
                sp.setDecimals(4)
                sp.setRange(0.005, max(1.0, hi - lo))
            else:
                sp.setDecimals(0 if discrete else 1)
                sp.setRange(lo, hi)              # BEFORE the value: a QDoubleSpinBox is capped at 99.99 until told otherwise
            sp.setValue(val)
            sp.valueChanged.connect(self._emit)
            self.table.setCellWidget(r, c, sp)
        d = QSpinBox(); d.setRange(0, 20000); d.setValue(t.delay); d.setMinimumWidth(46); d.valueChanged.connect(self._emit)
        self.table.setCellWidget(r, 4, d)
        self._ranges(r)

    def _ranges(self, r: int) -> None:
        name = self.table.cellWidget(r, 0).currentData()
        _, lo, hi, integer, _ = SWEEP_PARAMS[name]
        for c in (1, 2):
            sp = self.table.cellWidget(r, c)
            sp.blockSignals(True)
            sp.setRange(lo, hi)
            sp.setDecimals(0 if integer else 1)
            sp.blockSignals(False)
        st = self.table.cellWidget(r, 3)
        st.setRange(0.005, max(1.0, hi - lo))     # fractions allowed: 2000 steps over 0..128 is 0.064 per step
        st.setDecimals(4)
        st.setSingleStep(0.05)

    def _param_changed(self, combo) -> None:
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 0) is combo:
                self._ranges(r)
        self._emit()

    def _emit(self, *_) -> None:
        if not self._busy:
            self.changed.emit()

    def _add(self) -> None:
        info = self.info()
        used = {t.param for t in self.tracks()}
        free = [n for n in self.names() if n not in used] or self.names()
        n = free[0]
        cur = info.get("current", {}).get(n, SWEEP_PARAMS[n][1])
        self._row(default_track(n, cur))
        self._emit()

    def _remove(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._emit()

    def _to_current(self) -> None:
        cur = self.info().get("current", {})
        for r in range(self.table.rowCount()):
            n = self.table.cellWidget(r, 0).currentData()
            if n in cur:
                self.table.cellWidget(r, 2).setValue(cur[n])

    def tracks(self) -> List[Track]:
        out = []
        for r in range(self.table.rowCount()):
            out.append(Track(self.table.cellWidget(r, 0).currentData(), self.table.cellWidget(r, 1).value(),
                             self.table.cellWidget(r, 2).value(), self.table.cellWidget(r, 3).value(),
                             self.table.cellWidget(r, 4).value()))
        return out

    def set_tracks(self, tracks: List[Track]) -> None:
        self._busy = True
        self.table.setRowCount(0)
        for t in tracks:
            self._row(t)
        self._busy = False


class SweepJob(QThread):
    """Renders all slider-sweep frames in the background (several processes at once)."""
    progress = Signal(int, int)
    done = Signal(bool)
    failed = Signal(str)

    def __init__(self, renderer, workers: int):
        super().__init__()
        self.renderer, self.workers = renderer, workers
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        def cb(i: int, n: int) -> bool:
            self.progress.emit(i, n)
            return self._cancel
        try:
            nw = resolve_workers(self.workers)
            with pool_mod.busy():
                ok = self.renderer.ensure_sweeps(nw, shared_pool(nw) if nw > 1 else None, cb)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            ok = False
        self.done.emit(ok and not self._cancel)


class BuildJob(QThread):
    """Trace tiles and build the morph plan(s) off the GUI thread."""
    progress = Signal(str, int, int)
    done = Signal(object, object)       # scene, traced results (for caching)
    failed = Signal(str)

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _cb(self, label: str):
        def f(i: int, n: int) -> bool:
            self.progress.emit(label, i, n)
            return self._cancel
        return f

    def run(self) -> None:
        with pool_mod.busy():
            self._run()

    def _run(self) -> None:
        sp = self.spec
        try:
            lws = [load_linework(p) for p in sp["lw_paths"]]
            opts: MorphOptions = sp["opts"]
            if sp["mode"] == "single":
                self.progress.emit("Matching paths…", 0, 1)
                scene = scene_from_result(sp["result"], lws[0] if lws else None, opts)
                warm_curves(scene, 1, lambda: self._cancel)
                self.done.emit(scene, None)
                return
            rows: List[RowSpec] = sp["rows"]
            L: GridLayout = sp["layout"]
            jobs, keys = sp["jobs"], sp["keys"]
            have: Dict[tuple, object] = sp["cache"]
            todo = [i for i, kk in enumerate(keys) if kk not in have]
            fresh: Dict[tuple, object] = {}
            if todo:                                    # only squares whose picture / size / settings changed are traced again
                got = trace_jobs([jobs[i] for i in todo], sp["workers"], self._cb("Tracing pictures"))
                if self._cancel or len(got) != len(todo):
                    return
                for i, r in zip(todo, got):
                    fresh[keys[i]] = r
            results = [have.get(kk) or fresh[kk] for kk in keys]
            if self._cancel or len(results) != L.cells:
                return
            lw_mode = sp["lw_mode"]
            out = sp["dropped"] | sp["stay"]
            tile_lws = None
            plans = None
            if lw_mode != "span":
                cache: Dict[str, object] = {}
                tile_lws, k = [], 0
                for r in rows:
                    for i in range(len(r.paths)):
                        f = r.tile_linework(i)
                        own = cache.setdefault(f, load_linework(f)) if f else None
                        tile_lws.append(None if k in out else (own or tile_linework(lws, lw_mode, k)))
                        k += 1
                if any(t is not None for t in tile_lws):
                    plans = plan_many(results, tile_lws, opts, sp["workers"], self._cb("Matching paths"))
                else:
                    tile_lws = None
            if self._cancel:
                return
            self.progress.emit("Assembling grid…", 0, 1)
            nw = resolve_workers(sp["workers"])
            g = GridScene(results, L, lws, lw_mode, opts, "tile", plans, tile_lws, rows, nw,
                          shared_pool(nw) if nw > 1 else None, sp["dropped"], sp["stay"], sp["lw_area"])
            self.progress.emit("Measuring the white area of every picture…", 0, len(results))
            warm_curves(g.merged, min(4, nw + 1), lambda: self._cancel,
                        lambda i, n: self.progress.emit("Measuring the white area of every picture…", i, n))
            if self._cancel:
                return
            self.done.emit(g, fresh)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExportJob(QThread):
    progress = Signal(int, int)
    phase = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, scene, stages, cs: ClipSettings, width: int, path: str, kind: str, workers: int = 0):
        super().__init__()
        self.args = (scene, stages, cs, width, path, kind)
        self.workers = workers
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        scene, stages, cs, width, path, kind = self.args
        try:
            r = make_renderer(scene, stages, cs, width)

            def cb(i: int, n: int) -> bool:
                self.progress.emit(i, n)
                return self._cancel

            nw = max(1, min(4, (os.cpu_count() or 2) - 1))
            if any(st.kind == "sweep" for st in stages):
                self.phase.emit("Rendering slider frames")
                pw = resolve_workers(self.workers)
                with pool_mod.busy():
                    if not r.ensure_sweeps(pw, shared_pool(pw) if pw > 1 else None, cb):
                        return
            self.phase.emit("Encoding")
            if kind == "mp4":
                write_mp4(path, r.frames(cb, nw), cs.fps)
            else:
                write_gif_stream(path, r.total, r.frame, cs.fps, cb, max(1, min(4, (os.cpu_count() or 2) - 1)))           # frame by frame: never all frames in RAM
            if self._cancel:
                if os.path.exists(path):
                    os.remove(path)
                return
            self.done.emit(path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def make_renderer(scene, stages, cs: ClipSettings, width: int):
    """Renderer for a Scene (one picture) or GridScene at the given output width."""
    if isinstance(scene, GridScene):
        return GridRenderer(scene, stages, cs, width)
    W, H = scene.size
    ow = max(2, int(round(width))) // 2 * 2
    oh = max(2, int(round(H * ow / W))) // 2 * 2
    return Renderer(scene, stages, cs, (ow, oh))


class ClipStudio(QWidget):
    """sources() -> {'result': TraceResult|None, 'params': TraceParams|LineworkParams|None, 'note': str}"""

    def __init__(self, sources: Callable[[], dict], parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Clip Studio — sliders, grids, morph into linework")
        self.resize(1320, 820)
        self.sources = sources
        self.scene = None
        self.tile_cache: Dict[tuple, object] = {}        # job key -> traced result (only changed squares are traced again)
        self.stages: List[Stage] = default_timeline(False)
        self._user_timeline = False
        self.renderer = None
        self.job: Optional[BuildJob] = None
        self.exporter: Optional[ExportJob] = None
        self.sweep_job: Optional[SweepJob] = None
        self.sel: set = set()                           # selected squares (flat picture numbers)
        self.tool = "select"                             # select | morph | stay | white (paint)
        self._sq_loading = False
        self._rebuild_after_job = False
        self.sel_timer = QTimer(self)
        self.sel_timer.setSingleShot(True)
        self.sel_timer.setInterval(700)
        self.sel_timer.timeout.connect(self._build_after_selection)
        self.clock = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        self.rebuild = QTimer(self)
        self.rebuild.setSingleShot(True)
        self.rebuild.setInterval(200)
        self.rebuild.timeout.connect(self._make_renderer)
        self._loading = False

        self.canvas = Canvas()
        self.canvas.tile_clicked.connect(self._square_clicked)
        self.ws = GridWorkspace()
        self.ws.selection_changed.connect(self._ws_selection)
        self.ws.painted.connect(self._ws_painted)
        self.ws.activated.connect(lambda k: self.tabs.setCurrentIndex(1))
        self.stack = QStackedWidget()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.ws)
        left = QVBoxLayout()
        left.addLayout(self._view_bar())
        left.addWidget(self.stack, 1)
        left.addLayout(self._transport())
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(500)
        self.tabs.addTab(self._scroll(self._source_tab()), "1  Source")
        self.tabs.addTab(self._scroll(self._squares_tab()), "2  Squares")
        self.tabs.addTab(self._scroll(self._linework_tab()), "3  Linework")
        self.tabs.addTab(self._scroll(self._timeline_tab()), "4  Timeline")
        self.tabs.addTab(self._scroll(self._output_tab()), "5  Output")
        self.tabs.currentChanged.connect(self._tab_changed)
        root = QHBoxLayout(self)
        root.addLayout(left, 1)
        root.addWidget(self.tabs)
        self._refresh_stage_list()
        self._refresh_source_note()
        self._refresh_workspace()

    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        """A tab page that scrolls when the window is small."""
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(w)
        return sa

    # ------------------------------------------------------------------ view bar (preview / grid workspace, paint tools)
    def _view_bar(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self.view_prev = QToolButton(); self.view_prev.setText("Preview")
        self.view_ws = QToolButton(); self.view_ws.setText("Grid workspace")
        self.view_ws.setToolTip("Just the squares – no pictures, no tracing. Fast to select and paint, also for large grids.")
        grp = QButtonGroup(self)
        for b in (self.view_prev, self.view_ws):
            b.setCheckable(True); b.setAutoRaise(False); grp.addButton(b)
        self.view_prev.setChecked(True)
        self.view_prev.clicked.connect(lambda: self._set_view(0))
        self.view_ws.clicked.connect(lambda: self._set_view(1))
        h.addWidget(self.view_prev); h.addWidget(self.view_ws)
        h.addSpacing(14)
        h.addWidget(QLabel("Click does:"))
        self.tool_btns = {}
        tg = QButtonGroup(self)
        for key, text, tip in (("select", "Select", "Click / drag a frame to select squares (Ctrl or Shift adds)"),
                               ("morph", "Paint: morphs", "Squares you click or drag over will morph into the linework"),
                               ("stay", "Paint: stays", "Squares you click or drag over stay a vector while the others morph"),
                               ("white", "Paint: white", "Squares you click or drag over go back to white")):
            b = QToolButton(); b.setText(text); b.setCheckable(True); b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._set_tool(k))
            tg.addButton(b); h.addWidget(b); self.tool_btns[key] = b
        self.tool_btns["select"].setChecked(True)
        self.show_sq = QCheckBox("Squares on the preview")
        self.show_sq.setToolTip("Draws the squares (colour = morphs / stays / white) over the preview so you can click them there.")
        self.show_sq.toggled.connect(self._pick_toggled)
        h.addSpacing(10); h.addWidget(self.show_sq)
        h.addStretch(1)
        return h

    def _set_view(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        (self.view_ws if i else self.view_prev).setChecked(True)
        if i:
            self._refresh_workspace()

    def _set_tool(self, key: str) -> None:
        self.tool = key
        self.ws.tool = key
        self.tool_btns[key].setChecked(True)
        if key != "select" and self.stack.currentIndex() == 0:
            self.show_sq.setChecked(True)               # painting on the preview needs the squares on top of it

    def _tab_changed(self, i: int) -> None:
        if i == 1 and self.rb_grid.isChecked():         # Squares tab: show the grid workspace
            self._set_view(1)

    # ------------------------------------------------------------------ transport
    def _transport(self) -> QHBoxLayout:
        h = QHBoxLayout()
        self.play = QPushButton("▶")
        self.play.setFixedWidth(44)
        self.play.clicked.connect(self._toggle_play)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.valueChanged.connect(self._scrubbed)
        self.pos_lbl = QLabel("—")
        self.pos_lbl.setMinimumWidth(330)
        self.sweep_lbl = QLabel("")
        self.sweep_lbl.setStyleSheet("color:#777")
        h.addWidget(self.play)
        h.addWidget(self.loop)
        h.addWidget(self.scrub, 1)
        h.addWidget(self.sweep_lbl)
        h.addWidget(self.pos_lbl)
        return h

    # ------------------------------------------------------------------ tab 1: source
    def _source_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.rb_single = QRadioButton("One picture – the one open in the main window")
        self.rb_grid = QRadioButton("Grid of pictures – row by row, every row with its own settings")
        self.rb_single.setChecked(True)
        self.rb_single.toggled.connect(self._source_mode_changed)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#777;font-size:11px")
        v.addWidget(self.rb_single)
        v.addWidget(self.rb_grid)
        v.addWidget(self.note)

        self.row_specs: List[RowSpec] = []
        self._row_loading = False
        self.grid_box = QWidget()
        g = QVBoxLayout(self.grid_box)
        g.setContentsMargins(0, 0, 0, 0)
        self.row_list = QListWidget()
        self.row_list.setMaximumHeight(112)
        self.row_list.currentRowChanged.connect(self._row_selected)
        g.addWidget(self.row_list)
        b1, b2 = QHBoxLayout(), QHBoxLayout()
        for text, fn in (("+ Row", self._add_row), ("+ Folder row…", self._add_folder_row), ("Remove row", self._del_row),
                         ("▲", lambda: self._move_row(-1)), ("▼", lambda: self._move_row(1))):
            b = QPushButton(text); b.clicked.connect(fn); b1.addWidget(b)
        self.auto_cols = QSpinBox(); self.auto_cols.setRange(1, 30); self.auto_cols.setValue(3)
        arr = QPushButton("Re-arrange all pictures")
        arr.setToolTip("Pour every picture into rows of this many; each new row copies the settings of the row at the same position.")
        arr.clicked.connect(self._auto_layout)
        to_all = QPushButton("Settings → all rows")
        to_all.clicked.connect(self._row_to_all)
        b2.addWidget(arr); b2.addWidget(QLabel("per row")); b2.addWidget(self.auto_cols); b2.addWidget(to_all)
        g.addLayout(b1)
        g.addLayout(b2)

        self.row_box = QGroupBox("Selected row")
        f = QFormLayout(self.row_box)
        self.r_files = QListWidget()
        self.r_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.r_files.setMaximumHeight(84)
        fb = QHBoxLayout()
        for text, fn in (("Add pictures…", self._add_images), ("Add folder…", self._add_folder), ("Remove", self._remove_images)):
            b = QPushButton(text); b.clicked.connect(fn); fb.addWidget(b)
        f.addRow(self.r_files)
        f.addRow(fb)
        self.r_tw = QSpinBox(); self.r_tw.setRange(120, 2400); self.r_tw.setSuffix(" px")
        self.r_th = QSpinBox(); self.r_th.setRange(120, 2400); self.r_th.setSuffix(" px")
        sz = QHBoxLayout(); sz.addWidget(self.r_tw); sz.addWidget(QLabel("×")); sz.addWidget(self.r_th)
        self.r_size = QComboBox(); self.r_size.addItems([SIZE_LABEL[m] for m in SIZE_MODES])
        self.r_size.setToolTip("Fixed: every square is width × height.\nFit the picture: each square takes the shape of its picture; the number "
                               "above is the longest side / the height / the width, depending on the choice.")
        self.r_fit = QComboBox(); self.r_fit.addItems(["Cover (crop to fill)", "Contain (keep whole picture)"])
        self.r_engine = QComboBox(); self.r_engine.addItems(["Main window's current settings", "Image Trace preset", "Linework Snap preset"])
        self.r_preset = QComboBox()
        self.r_colors = QSpinBox(); self.r_colors.setRange(0, 64); self.r_colors.setSpecialValueText("keep")
        self.r_detail = QSpinBox(); self.r_detail.setRange(-1, 100); self.r_detail.setSpecialValueText("keep")
        self.r_detail.setToolTip("Image Trace: Paths.  Linework Snap: Simplify.")
        self.r_noise = QSpinBox(); self.r_noise.setRange(-1, 100); self.r_noise.setSpecialValueText("keep")
        self.r_side = QComboBox(); self.r_side.addItems(["As set in the timeline", "B enters from the right", "B enters from the left",
                                                          "B enters from the bottom", "B enters from the top"])
        self.r_delay = QSpinBox(); self.r_delay.setRange(0, 5000); self.r_delay.setSuffix(" frames")
        self.r_lw = QLineEdit(); self.r_lw.setReadOnly(True); self.r_lw.setPlaceholderText("none – uses the linework of tab 2")
        lwb = QPushButton("…"); lwb.setFixedWidth(28); lwb.clicked.connect(self._pick_row_lw)
        lwx = QPushButton("×"); lwx.setFixedWidth(28); lwx.clicked.connect(lambda: self.r_lw.setText(""))
        lwr = QHBoxLayout(); lwr.addWidget(self.r_lw, 1); lwr.addWidget(lwb); lwr.addWidget(lwx)
        f.addRow("Square size", self.r_size)
        f.addRow("Size (px)", sz)
        f.addRow("Fit", self.r_fit)
        f.addRow("Vectorise with", self.r_engine)
        f.addRow("Preset", self.r_preset)
        f.addRow("Colours", self.r_colors)
        f.addRow("Detail", self.r_detail)
        f.addRow("Noise", self.r_noise)
        f.addRow("Slider", self.r_side)
        f.addRow("Slider starts after", self.r_delay)
        f.addRow("Own linework", lwr)
        self.r_sweep = TrackEditor(self._sweep_info)
        f.addRow(QLabel("Own slider ranges (replace the timeline's sliders of the same name for this row):"))
        f.addRow(self.r_sweep)
        self.r_sweep.changed.connect(self._row_edited)
        g.addWidget(self.row_box)
        for wd in (self.r_tw, self.r_th, self.r_colors, self.r_detail, self.r_noise, self.r_delay):
            wd.valueChanged.connect(self._row_edited)
        for wd in (self.r_size, self.r_fit, self.r_preset, self.r_side):
            wd.currentIndexChanged.connect(self._row_edited)
        self.r_engine.currentIndexChanged.connect(self._engine_changed)
        self.r_lw.textChanged.connect(self._row_edited)

        gf = QFormLayout()
        self.gap = QSpinBox(); self.gap.setRange(0, 200); self.gap.setValue(12); self.gap.setSuffix(" px")
        self.workers = QSpinBox(); self.workers.setRange(0, 32); self.workers.setSpecialValueText("auto")
        self.workers.setToolTip("Auto = CPU cores − 1, but never more than fit into the memory budget (every process needs ~0.5 GB).")
        self.mem_gb = QSpinBox(); self.mem_gb.setRange(0, 512); self.mem_gb.setSpecialValueText("auto")
        self.mem_gb.setSuffix(" GB")
        self.mem_gb.setToolTip("How much RAM the program may use in total. Auto = 60 % of the installed RAM (about 9 GB on a 16 GB machine).\n"
                               "Slider frames are kept in a temporary file on disk, so long clips do not need RAM.")
        self.mem_gb.valueChanged.connect(self._memory_changed)
        gf.addRow("Gap", self.gap)
        gf.addRow("CPU processes", self.workers)
        gf.addRow("Memory budget", self.mem_gb)
        g.addLayout(gf)
        self.mem_lbl = QLabel(memory.describe())
        self.mem_lbl.setStyleSheet("color:#777;font-size:11px")
        self.mem_lbl.setWordWrap(True)
        g.addWidget(self.mem_lbl)
        self.gap.valueChanged.connect(lambda *_: (self._update_grid_info(), self._refresh_workspace()))
        self.grid_info = QLabel("")
        self.grid_info.setStyleSheet("color:#777;font-size:11px")
        self.grid_info.setWordWrap(True)
        g.addWidget(self.grid_info)
        v.addWidget(self.grid_box)

        self.build_btn = QPushButton("Build")
        self.build_btn.setStyleSheet("font-weight:600;padding:8px")
        self.build_btn.clicked.connect(self._build)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self.job and self.job.cancel())
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setFormat("%v / %m")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        row = QHBoxLayout(); row.addWidget(self.build_btn, 1); row.addWidget(self.cancel_btn)
        v.addLayout(row)
        v.addWidget(self.bar)
        v.addWidget(self.status)
        v.addStretch(1)
        self.grid_box.setVisible(False)
        self._engine_changed()
        return w

    # ------------------------------------------------------------------ tab 2: linework
    def _linework_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Linework the vectors morph into  (SVG, DXF or a PNG of lines):"))
        self.lw_list = QListWidget()
        self.lw_list.setMaximumHeight(100)
        b = QHBoxLayout()
        for text, fn in (("Add file…", self._add_lw), ("Remove", self._remove_lw), ("Clear", self.lw_list.clear)):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            b.addWidget(btn)
        v.addWidget(self.lw_list)
        v.addLayout(b)
        f = QFormLayout()
        self.lw_mode = QComboBox()
        self.lw_mode.addItems(["One drawing across the whole grid – all pictures morph into ONE image", "Same drawing in every picture", "One file per picture (in order)"])
        self.lw_area = QComboBox()
        self.lw_area.addItems(["The whole canvas", "Only the area of the morphing squares"])
        self.lw_area.setToolTip("Grids, one drawing across the grid: fit it to the whole canvas, or only to the smallest rectangle that holds\n"
                                "the squares that morph (squares that stay a vector keep their picture).")
        self.lw_area.currentIndexChanged.connect(lambda *_: self._refresh_workspace(form=False))
        self.fitmode = QComboBox()
        self.fitmode.addItems(["Contain (whole drawing)", "Cover (fill, crop)", "Stretch to frame", "Use the file's page"])
        self.margin = QDoubleSpinBox(); self.margin.setRange(0, 0.3); self.margin.setSingleStep(0.01); self.margin.setValue(0.03)
        self.use_w = QCheckBox("Override line width")
        self.line_w = QDoubleSpinBox(); self.line_w.setRange(0.3, 30); self.line_w.setValue(1.6); self.line_w.setSuffix(" px")
        self.stagger = QComboBox()
        self.stagger.addItems(["Left → right", "Right → left", "Top → bottom", "Bottom → top", "From the centre", "Random", "All at once"])
        self.spread = QDoubleSpinBox(); self.spread.setRange(0, 3); self.spread.setSingleStep(0.1); self.spread.setValue(0.6)
        self.spread.setToolTip("0 = every path moves at the same time. Larger = more staggered.")
        self.ease = QComboBox(); self.ease.addItems(["Smooth", "Linear", "Fast start"])
        f.addRow("Grids", self.lw_mode)
        f.addRow("Drawing covers", self.lw_area)
        f.addRow("Fit", self.fitmode)
        f.addRow("Margin", self.margin)
        f.addRow(self.use_w, self.line_w)
        f.addRow("Stagger", self.stagger)
        f.addRow("Stagger amount", self.spread)
        f.addRow("Easing", self.ease)
        v.addLayout(f)
        info = QLabel("What happens: fills fade out while their outlines slide onto the nearest line of your linework; "
                      "surplus paths merge into lines, surplus lines grow out of the picture. "
                      "Press “Build” on the Source tab after changing anything here. Which squares morph is chosen on the Squares tab.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#777;font-size:11px")
        v.addWidget(info)
        self.morph_stats = QLabel("")
        self.morph_stats.setWordWrap(True)
        v.addWidget(self.morph_stats)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ tab 3: timeline
    def _timeline_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.stage_list = QListWidget()
        self.stage_list.setMaximumHeight(150)
        self.stage_list.currentRowChanged.connect(self._stage_selected)
        v.addWidget(self.stage_list)
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        for text, fn in (("+ Hold", lambda: self._add_stage(Stage("hold", 1.0, "vector"))),
                         ("+ Wipe", lambda: self._add_stage(Stage("wipe", 2.5))),
                         ("+ Sliders", lambda: self._add_stage(self._new_sweep())),
                         ("+ Morph", lambda: self._add_stage(Stage("morph", 3.5)))):
            b = QPushButton(text); b.clicked.connect(fn); r1.addWidget(b)
        for text, fn in (("Duplicate", self._dup_stage), ("Remove", self._del_stage),
                         ("▲", lambda: self._move_stage(-1)), ("▼", lambda: self._move_stage(1))):
            b = QPushButton(text); b.clicked.connect(fn); r2.addWidget(b)
        v.addLayout(r1)
        v.addLayout(r2)
        story = QPushButton("Template: white → vectors appear → chosen pictures back to white → the rest morphs")
        story.setToolTip("Fills the timeline with the whole story. Choose which squares morph, stay or go back to white on the Squares tab.")
        story.clicked.connect(self._story_timeline)
        v.addWidget(story)
        sr = QHBoxLayout()
        sr.addWidget(QLabel("Steps per slider (1 step = 1 frame):"))
        self.story_steps = QSpinBox(); self.story_steps.setRange(10, 20000); self.story_steps.setValue(2000)
        self.story_steps.setToolTip("How many frames each slider sweep in the template runs through. More = the video lingers longer "
                                    "and the change between frames is finer. The video is this many frames divided by FPS long.")
        sr.addWidget(self.story_steps)
        sr.addWidget(QLabel("Threshold up to:"))
        self.story_thr = QSpinBox(); self.story_thr.setRange(-1, 255); self.story_thr.setSpecialValueText("main window"); self.story_thr.setValue(-1)
        self.story_thr.setToolTip("Where the Threshold slider ends in the template. 255 = the picture becomes completely black\n"
                                  "(like pushing the Threshold slider of the main window all the way up).")
        sr.addWidget(self.story_thr)
        tb = QPushButton("black"); tb.setFixedWidth(50); tb.setToolTip("Threshold 255: the picture ends fully black")
        tb.clicked.connect(lambda: self.story_thr.setValue(255))
        sr.addWidget(tb); sr.addStretch(1)
        v.addLayout(sr)
        sy = QHBoxLayout()
        sy.addWidget(QLabel("Finish together – black when white ≤"))
        self.story_sync = QDoubleSpinBox(); self.story_sync.setRange(-1, 50); self.story_sync.setDecimals(1); self.story_sync.setSingleStep(0.5)
        self.story_sync.setSpecialValueText("off"); self.story_sync.setSuffix(" % of the picture"); self.story_sync.setValue(0.5)
        self.story_sync.setToolTip("Some pictures turn black at a low Threshold, others only near 255. When this is on, the Threshold of every picture\n"
                                   "ends at ITS OWN black point (the lowest value where at most this much of it is still white; the last step then\n"
                                   "jumps to 255 = completely black) and all pictures are black in the same frame – see the box on the right for how.\n"
                                   "A picture with its own Threshold range on the Squares tab keeps that range.\n"
                                   "Needs the pictures to be built (Build button) and black-and-white mode.")
        sy.addWidget(self.story_sync)
        self.story_fit = QComboBox()
        self.story_fit.addItem("same white at every moment", "even")
        self.story_fit.addItem("each picture's own path", "natural")
        self.story_fit.setToolTip("How a picture goes from white to black:\n"
                                  "• same white at every moment: all pictures have the same share of white at the same time – 50 % black in the\n"
                                  "  same frame, finished in the same frame (flat areas such as a white background fill gradually from their edges).\n"
                                  "• each picture's own path: the Threshold just moves in equal steps; pictures with big bright areas stay white\n"
                                  "  longer and then flip.")
        sy.addWidget(self.story_fit)
        self.story_pace = QComboBox()
        self.story_pace.addItem("slower ones sped up", "fast")
        self.story_pace.addItem("as long as the slowest needs", "slow")
        self.story_pace.addItem("exactly the steps", "steps")
        self.story_pace.setToolTip("How long the sweep lasts (the steps above are the normal speed: a full 0 → 255 sweep takes that many):\n"
                                   "• slower ones sped up: as long as the picture that turns black first needs at normal speed; every slower picture\n"
                                   "  gets bigger steps and is black in the same frame.\n"
                                   "• as long as the slowest needs: normal speed for the slowest picture, the faster ones get finer steps.\n"
                                   "• exactly the steps: every picture takes exactly the number of steps above.")
        sy.addWidget(self.story_fit)
        sy.addWidget(self.story_pace)
        bpb = QPushButton("Black points…"); bpb.setToolTip("Table: at which Threshold every picture is black, and the step it gets")
        bpb.clicked.connect(self._show_black_points)
        sy.addWidget(bpb); sy.addStretch(1)
        v.addLayout(sy)

        self.edit = QGroupBox("Selected stage")
        f = QFormLayout(self.edit)
        self.e_seconds = QDoubleSpinBox(); self.e_seconds.setRange(0.1, 120); self.e_seconds.setSuffix(" s")
        self.e_a = QComboBox(); self.e_b = QComboBox()
        for n, k in IMAGES:
            self.e_a.addItem(n, k); self.e_b.addItem(n, k)
        self.e_mode = QComboBox(); self.e_mode.addItems(["One slider", "Two sliders (4 quadrants)"])
        self.e_dir = QComboBox()
        for n, _, _ in DIRECTIONS:
            self.e_dir.addItem(n)
        self.e_sx = QComboBox(); self.e_sx.addItems(["Slider 1: B from the right", "Slider 1: B from the left"])
        self.e_sy = QComboBox(); self.e_sy.addItems(["Slider 2: B from the bottom", "Slider 2: B from the top"])
        spins = {}
        for name in ("x0", "x1", "y0", "y1", "lag"):
            s = QDoubleSpinBox(); s.setRange(0, 1); s.setSingleStep(0.05); s.setDecimals(2); spins[name] = s
        self.e_x0, self.e_x1, self.e_y0, self.e_y1, self.e_lag = (spins[k] for k in ("x0", "x1", "y0", "y1", "lag"))
        self.e_ease = QComboBox(); self.e_ease.addItems(["Smooth", "Linear"])
        self.e_pp = QCheckBox("Ping-pong (A → B → A)")
        self.e_ta = QLineEdit(); self.e_tb = QLineEdit(); self.e_cap = QLineEdit()
        self.e_rev = QCheckBox("Reverse (morph: linework → vector, sliders: end → start)")
        self.e_scope = QComboBox(); self.e_scope.addItems(["One slider per row", "One slider per picture", "One slider over the whole grid"])
        self.e_step = QSpinBox(); self.e_step.setRange(0, 4000); self.e_step.setSpecialValueText("off – use seconds")
        self.e_step.setSuffix(" px per frame")
        self.e_step.setToolTip("1 step = 1 frame: the slider moves this many picture pixels in every frame.")
        auto = QPushButton("Auto"); auto.setFixedWidth(48); auto.setToolTip("Step size that makes the slider take about 4 seconds")
        auto.clicked.connect(self._auto_step)
        step_row = QHBoxLayout(); step_row.setContentsMargins(0, 0, 0, 0)
        step_w = QWidget(); step_w.setLayout(step_row); step_row.addWidget(self.e_step, 1); step_row.addWidget(auto)
        self.e_target = QComboBox(); self.e_target.addItems(["All pictures", "Only the pictures that do not go back to white", "Only the pictures that go back to white"])
        self.e_dropped = QComboBox(); self.e_dropped.addItems(["Automatic (white after the 'back to white' sliders)", "Show them", "Show them white"])
        self.e_tracks = TrackEditor(self._sweep_info)
        self.e_seq = QCheckBox("Sliders one after the other (not all at once)")
        self.e_fit = QComboBox()
        for k, label in (("even", "same white at every moment"), ("natural", "each picture's own path")):
            self.e_fit.addItem(label, k)
        self.e_pace = QComboBox()
        for k, label in (("fast", "slower ones sped up"), ("slow", "as long as the slowest needs"), ("steps", "exactly the steps of the slider")):
            self.e_pace.addItem(label, k)
        self.e_sync = QDoubleSpinBox(); self.e_sync.setRange(-1, 50); self.e_sync.setDecimals(1); self.e_sync.setSingleStep(0.5)
        self.e_sync.setSpecialValueText("off – same range for all"); self.e_sync.setSuffix(" % white left")
        self.e_sync.setToolTip("Every picture's Threshold slider ends (or, running down, starts) at its own black point – where at most this\n"
                               "much of it is white – in the same number of steps, so all pictures turn black in the same frame.")
        self.e_rowmode = QComboBox(); self.e_rowmode.addItems(["All rows at the same time", "One row after the other", "Rows start N frames apart"])
        self.e_stagger = QSpinBox(); self.e_stagger.setRange(0, 2000); self.e_stagger.setSuffix(" frames")
        self.e_alt = QCheckBox("Every second row slides in from the other side")
        self.rows_f = {}
        for label, wd, key in (("Duration", self.e_seconds, "sec"), ("Image A", self.e_a, "a"), ("Image B", self.e_b, "b"),
                               ("Sliders", self.e_tracks, "tracks"), ("Applies to", self.e_target, "target"), ("Dropped pictures", self.e_dropped, "dropped"), (None, self.e_seq, "seq"), ("Finish together", self.e_sync, "sync"), ("Path", self.e_fit, "fit"), ("Length", self.e_pace, "pace"), ("Step", step_w, "step"), ("Grid sliders", self.e_scope, "scope"),
                               ("Rows", self.e_rowmode, "rowmode"), ("Row offset", self.e_stagger, "stagger"), (None, self.e_alt, "alt"),
                               ("Sliders", self.e_mode, "mode"), ("Direction", self.e_dir, "dir"),
                               ("Slider 1 side", self.e_sx, "sx"), ("Slider 2 side", self.e_sy, "sy"),
                               ("Slider 1 start", self.e_x0, "x0"), ("Slider 1 end", self.e_x1, "x1"),
                               ("Slider 2 start", self.e_y0, "y0"), ("Slider 2 end", self.e_y1, "y1"),
                               ("Slider 2 starts late", self.e_lag, "lag"), ("Easing", self.e_ease, "ease"),
                               (None, self.e_pp, "pp"), ("Text A", self.e_ta, "ta"), ("Text B", self.e_tb, "tb"),
                               ("Caption", self.e_cap, "cap"), (None, self.e_rev, "rev")):
            if label is None:
                f.addRow(wd)
            else:
                f.addRow(label, wd)
            self.rows_f[key] = (label, wd)
        v.addWidget(self.edit)
        for wd in (self.e_seconds, self.e_x0, self.e_x1, self.e_y0, self.e_y1, self.e_lag, self.e_step, self.e_stagger, self.e_sync):
            wd.valueChanged.connect(self._stage_edited)
        for wd in (self.e_a, self.e_b, self.e_mode, self.e_dir, self.e_sx, self.e_sy, self.e_ease, self.e_scope, self.e_rowmode, self.e_target, self.e_dropped, self.e_fit, self.e_pace):
            wd.currentIndexChanged.connect(self._stage_edited)
        self.e_tracks.changed.connect(self._stage_edited)
        for wd in (self.e_pp, self.e_rev, self.e_alt, self.e_seq):
            wd.toggled.connect(self._stage_edited)
        for wd in (self.e_ta, self.e_tb, self.e_cap):
            wd.textChanged.connect(self._stage_edited)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ tab 4: output
    def _output_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.fps = QSpinBox(); self.fps.setRange(5, 60); self.fps.setValue(25)
        self.bg = QComboBox(); self.bg.addItems(["White", "Dark", "Checkerboard (images only)"])
        self.show_text = QCheckBox("Show BEFORE / AFTER texts and captions"); self.show_text.setChecked(True)
        self.divider = QCheckBox("Show divider lines"); self.divider.setChecked(True)
        self.out_w = QSpinBox(); self.out_w.setRange(240, 3840); self.out_w.setValue(960); self.out_w.setSuffix(" px")
        self.smooth = QCheckBox("Smooth morph (2× supersampling, slower)"); self.smooth.setChecked(True)
        self.invert = QCheckBox("Invert the vector colours (black paper, white drawing)")
        self.invert.setToolTip("Vectors, slider sweeps, morph and linework are drawn in negative colours: the clip goes from black to white\n"
                               "instead of from white to black. Photos (the source image) keep their colours.")
        self.prev_w = QComboBox(); self.prev_w.addItems(["Preview 640 px", "Preview 960 px", "Preview 480 px"])
        f.addRow("Frames / s", self.fps)
        f.addRow("Background", self.bg)
        f.addRow(self.show_text)
        f.addRow(self.divider)
        f.addRow("Output width", self.out_w)
        f.addRow(self.smooth)
        f.addRow(self.invert)
        f.addRow("Preview", self.prev_w)
        v.addLayout(f)
        self.total_lbl = QLabel("")
        v.addWidget(self.total_lbl)
        r = QHBoxLayout()
        self.gif_btn = QPushButton("Export GIF…"); self.mp4_btn = QPushButton("Export MP4…")
        self.gif_btn.clicked.connect(lambda: self._export("gif"))
        self.mp4_btn.clicked.connect(lambda: self._export("mp4"))
        r.addWidget(self.gif_btn); r.addWidget(self.mp4_btn)
        v.addLayout(r)
        self.exp_bar = QProgressBar()
        self.exp_phase = QLabel("")
        self.exp_cancel = QPushButton("Cancel export")
        self.exp_cancel.setEnabled(False)
        self.exp_cancel.clicked.connect(lambda: self.exporter and self.exporter.cancel())
        v.addWidget(self.exp_phase)
        v.addWidget(self.exp_bar)
        v.addWidget(self.exp_cancel)
        hint = QLabel("GIF: keep the width small (≈ 480–720 px) – files get large quickly. MP4 (H.264) plays everywhere.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#777;font-size:11px")
        v.addWidget(hint)
        v.addStretch(1)
        for wd in (self.fps, self.out_w):
            wd.valueChanged.connect(self._settings_changed)
        for wd in (self.bg, self.prev_w):
            wd.currentIndexChanged.connect(self._settings_changed)
        for wd in (self.show_text, self.divider, self.smooth, self.invert):
            wd.toggled.connect(self._settings_changed)
        return w

    # ------------------------------------------------------------------ source handling
    def _memory_changed(self, gb: int) -> None:
        memory.set_budget_gb(float(gb))
        self.mem_lbl.setText(memory.describe())

    def _source_mode_changed(self) -> None:
        self.grid_box.setVisible(self.rb_grid.isChecked())
        self._stage_selected(self.stage_list.currentRow())
        self._refresh_source_note()
        self.view_ws.setEnabled(self.rb_grid.isChecked())
        if not self.rb_grid.isChecked():
            self._set_view(0)
        self._refresh_workspace()

    def _refresh_source_note(self) -> None:
        s = self.sources()
        if self.rb_single.isChecked():
            self.note.setText("Open and trace a picture in the main window (Image Trace or Linework Snap), then press Build."
                              if s.get("result") is None else "Using the current trace of the main window.")
        else:
            self.note.setText(s.get("note", "") or "")

    def on_main_changed(self) -> None:
        """The main window produced a new trace: refresh the single-picture scene."""
        if self.isVisible() and self.rb_single.isChecked() and self.scene is not None and self.job is None:
            QTimer.singleShot(400, self._build)

    # --- rows ------------------------------------------------------------------------------------
    def _cur_row(self) -> Optional[RowSpec]:
        i = self.row_list.currentRow()
        return self.row_specs[i] if 0 <= i < len(self.row_specs) else None

    def _refresh_rows(self, select: Optional[int] = None) -> None:
        sel = self.row_list.currentRow() if select is None else select
        self._row_loading = True
        self.row_list.clear()
        for i, r in enumerate(self.row_specs):
            self.row_list.addItem(r.title(i))
        self.row_list.setCurrentRow(min(max(sel, 0), len(self.row_specs) - 1) if self.row_specs else -1)
        self._row_loading = False
        self._row_selected(self.row_list.currentRow())
        self._update_grid_info()
        self._refresh_workspace()

    def _update_grid_info(self) -> None:
        rows = [r for r in self.row_specs if r.paths]
        if not rows:
            self.grid_info.setText("Add pictures: each row can have its own number of pictures, size and settings.")
            return
        L = GridLayout.from_rows(rows, self.gap.value())
        self.grid_info.setText(f"{sum(len(r.paths) for r in rows)} pictures in {len(rows)} rows → canvas {L.canvas[0]} × {L.canvas[1]} px")

    def _new_row(self, paths=()) -> RowSpec:
        base = self._cur_row() or (self.row_specs[-1] if self.row_specs else None)
        r = (replace(base, paths=list(paths), delay=0, linework="", sweep=[replace(t) for t in base.sweep], cells=[])
             if base else RowSpec(paths=list(paths)))
        if base is None:
            r.tile_w = r.tile_h = 480
            r.size_mode = "fit_long"          # squares follow the shape of their picture unless told otherwise
        return r

    def _add_row(self) -> None:
        self.row_specs.append(self._new_row())
        self._refresh_rows(len(self.row_specs) - 1)

    def _add_folder_row(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder with the pictures of one row")
        if d:
            ps = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
            if ps:
                self.row_specs.append(self._new_row(ps))
                self._refresh_rows(len(self.row_specs) - 1)

    def _del_row(self) -> None:
        i = self.row_list.currentRow()
        if 0 <= i < len(self.row_specs):
            del self.row_specs[i]
            self._refresh_rows(max(0, i - 1))

    def _move_row(self, d: int) -> None:
        i = self.row_list.currentRow()
        j = i + d
        if 0 <= i < len(self.row_specs) and 0 <= j < len(self.row_specs):
            self.row_specs[i], self.row_specs[j] = self.row_specs[j], self.row_specs[i]
            self._refresh_rows(j)

    def _row_to_all(self) -> None:
        r = self._cur_row()
        if r is not None:
            self.row_specs = [replace(r, paths=list(o.paths), delay=o.delay if o is not r else r.delay, linework=o.linework,
                                      cells=o.cells, sweep=[replace(t) for t in r.sweep])
                              if o is not r else o for o in self.row_specs]
            self._refresh_rows()

    def _engine_changed(self, *_) -> None:
        eng = self.r_engine.currentIndex()
        self.r_preset.blockSignals(True)
        self.r_preset.clear()
        if eng == 1:
            self.r_preset.addItems(list(TRACE_PRESETS))
        elif eng == 2:
            self.r_preset.addItems(list(LW_PRESETS))
        self.r_preset.blockSignals(False)
        self.r_preset.setEnabled(eng != 0)
        self.r_colors.setEnabled(eng != 2)
        self.r_noise.setEnabled(eng != 2)
        self._row_edited()

    def _row_selected(self, _i: int) -> None:
        r = self._cur_row()
        self.row_box.setEnabled(r is not None)
        if r is None:
            self.r_files.clear()
            return
        self._row_loading = True
        self.r_files.clear()
        self.r_files.addItems(r.paths)
        self.r_tw.setValue(r.tile_w); self.r_th.setValue(r.tile_h)
        self.r_size.setCurrentIndex(SIZE_MODES.index(r.size_mode))
        self.r_fit.setCurrentIndex(0 if r.fit == "cover" else 1)
        eng = {"main": 0, "trace": 1, "linework": 2}[r.engine]
        self.r_engine.setCurrentIndex(eng)
        self._engine_changed_silent()
        if r.preset:
            k = self.r_preset.findText(r.preset)
            if k >= 0:
                self.r_preset.setCurrentIndex(k)
        self.r_colors.setValue(r.colors); self.r_detail.setValue(r.detail); self.r_noise.setValue(r.noise)
        self.r_side.setCurrentIndex(SIDES.index(r.side))
        self.r_delay.setValue(r.delay)
        self.r_lw.setText(r.linework)
        self.r_sweep.set_tracks(r.sweep)
        self._row_loading = False

    def _engine_changed_silent(self) -> None:
        was = self._row_loading
        self._row_loading = True
        self._engine_changed()
        self._row_loading = was

    def _row_edited(self, *_) -> None:
        if self._row_loading:
            return
        r = self._cur_row()
        if r is None:
            return
        r.tile_w, r.tile_h = self.r_tw.value(), self.r_th.value()
        r.size_mode = SIZE_MODES[self.r_size.currentIndex()]
        r.fit = "cover" if self.r_fit.currentIndex() == 0 else "contain"
        r.engine = ("main", "trace", "linework")[self.r_engine.currentIndex()]
        r.preset = self.r_preset.currentText() if r.engine != "main" else ""
        r.colors, r.detail, r.noise = self.r_colors.value(), self.r_detail.value(), self.r_noise.value()
        r.side = SIDES[self.r_side.currentIndex()]
        r.delay = self.r_delay.value()
        r.linework = self.r_lw.text()
        r.sweep = self.r_sweep.tracks()
        i = self.row_list.currentRow()
        self.row_list.item(i).setText(r.title(i))
        self._update_grid_info()
        self._refresh_workspace()

    def _add_images(self) -> None:
        ps, _ = QFileDialog.getOpenFileNames(self, "Add pictures", "", "Images (" + " ".join("*" + e for e in IMG_EXT) + ")")
        self._add_files(ps)

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder with pictures")
        if d:
            self._add_files(sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)))

    def _add_files(self, ps) -> None:
        if not ps:
            return
        if self._cur_row() is None:
            self.row_specs.append(self._new_row())
            self._refresh_rows(len(self.row_specs) - 1)
        r = self._cur_row()
        r.add_paths(ps)
        self._row_selected(self.row_list.currentRow())
        i = self.row_list.currentRow()
        self.row_list.item(i).setText(r.title(i))
        self._update_grid_info()

    def _remove_images(self) -> None:
        r = self._cur_row()
        if r is None:
            return
        gone = {it.row() for it in self.r_files.selectedIndexes()}
        r.remove_at(gone)
        self._row_selected(self.row_list.currentRow())
        i = self.row_list.currentRow()
        self.row_list.item(i).setText(r.title(i))
        self._update_grid_info()

    def _auto_layout(self) -> None:
        allp = [(p, c) for r in self.row_specs for p, c in zip(r.paths, r.sync())]       # squares keep their own settings
        if not allp:
            return
        n = self.auto_cols.value()
        chunks = [allp[i:i + n] for i in range(0, len(allp), n)]
        old = self.row_specs or [RowSpec()]
        self.row_specs = [replace(old[min(i, len(old) - 1)], paths=[p for p, _ in c], cells=[x for _, x in c],
                                  delay=old[i].delay if i < len(old) else 0,
                                  linework=old[i].linework if i < len(old) else "") for i, c in enumerate(chunks)]
        self._refresh_rows(0)

    def _pick_row_lw(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Linework of this row", "", "Linework (*.svg *.dxf *.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p:
            self.r_lw.setText(p)

    # --- squares: state (morphs / stays / white), selection, per-square settings ---------------------------------
    def _rows(self) -> List[RowSpec]:
        return [r for r in self.row_specs if r.paths]

    def _flat(self):
        return flat_cells(self._rows())

    def _n_tiles(self) -> int:
        return sum(len(r.paths) for r in self.row_specs)

    @property
    def dropped(self) -> set:
        """Pictures that go back to white (flat picture numbers)."""
        return states_of(self._rows())[0]

    @property
    def stay(self) -> set:
        return states_of(self._rows())[1]

    def _pick_toggled(self, on: bool) -> None:
        self.canvas.picking = on
        self._sync_overlay()

    def _cell_info(self, k: int, rows: List[RowSpec], flat) -> CellInfo:
        r, i, c = flat[k]
        row = rows[r]
        path = row.paths[i]
        w, h = row.tile_size(i)
        eng = c.engine or row.engine
        preset = c.preset if c.engine else row.preset
        tag = {"main": "main settings", "trace": preset or "trace", "linework": preset or "linework"}[eng]
        tip = [os.path.basename(path), f"{w} × {h} px  ·  {row.tile_fit(i)}", "Square " + STATE_LABEL[c.state], tag]
        if c.customised():
            tip.append("★ has its own settings")
        return CellInfo(os.path.basename(path), c.state, (w, h), tag, c.customised(), "\n".join(tip))

    def _lw_area(self) -> Optional[tuple]:
        """The part of the canvas the drawing is fitted to (None = the whole canvas)."""
        if not hasattr(self, "lw_area") or self.lw_area.currentIndex() != 1:
            return None
        rows = self._rows()
        if not rows:
            return None
        L = GridLayout.from_rows(rows, self.gap.value())
        white, stay = states_of(rows)
        ks = [k for k in range(L.cells) if k not in white and k not in stay]
        return area_of(L, ks) if ks else None

    def _refresh_workspace(self, form: bool = True) -> None:
        """Redraw the grid workspace from the rows (no tracing involved)."""
        rows = self._rows()
        if not rows or not self.rb_grid.isChecked():
            self.ws.set_grid(None, [])
            self._sync_overlay(form)
            return
        L = GridLayout.from_rows(rows, self.gap.value())
        flat = flat_cells(rows)
        self.ws.set_grid(L, [self._cell_info(k, rows, flat) for k in range(len(flat))],
                         [f"R{n + 1}" for n in range(len(rows))], self._lw_area())
        self.sel = {k for k in self.sel if k < len(flat)}
        self.ws.set_selection(self.sel)
        self._sync_overlay(form)

    def _sync_overlay(self, form: bool = True) -> None:
        r = self.renderer
        self.canvas.tiles = r.tile_boxes() if isinstance(r, GridRenderer) else []
        flat = self._flat() if self.rb_grid.isChecked() else []
        self.canvas.states = {k: c.state for k, (_, _, c) in enumerate(flat)}
        self.canvas.selected = set(self.sel)
        self.canvas.update()
        n = len(flat)
        if hasattr(self, "sq_summary"):
            cnt = {st: sum(1 for _, _, c in flat if c.state == st) for st in STATES}
            self.sq_summary.setText(f"{n} squares: {cnt['morph']} morph, {cnt['stay']} stay a vector, {cnt['white']} go back to white.   "
                                    f"{len(self.sel)} selected.")
            if form:
                self._load_square_form()

    def _square_clicked(self, k: int, extend: bool) -> None:
        """A click on a square of the preview."""
        if self.tool == "select":
            self.sel = (self.sel ^ {k}) if extend else {k}
            self._selection_only()
        else:
            self._set_states([k], self.tool)

    def _ws_selection(self) -> None:
        self.sel = set(self.ws.selected)
        self._selection_only()

    def _ws_painted(self, ks: list, state: str) -> None:
        self._set_states(ks, state)

    def _selection_only(self) -> None:
        self.ws.set_selection(self.sel)
        self._sync_overlay()

    def _select(self, ks) -> None:
        self.sel = set(ks)
        self.ws.set_selection(self.sel)
        self._sync_overlay()

    def _set_states(self, ks, state: str) -> None:
        flat = self._flat()
        for k in ks:
            if 0 <= k < len(flat):
                flat[k][2].state = state
        self._refresh_workspace()
        self._selection_changed()

    def _toggle_tile(self, k: int, *_) -> None:
        """Compatibility: toggle a square between 'morphs' and 'goes back to white'."""
        flat = self._flat()
        if 0 <= k < len(flat):
            self._set_states([k], "morph" if flat[k][2].state == "white" else "white")

    def _set_dropped(self, d: set) -> None:
        flat = self._flat()
        for k, (_, _, c) in enumerate(flat):
            c.state = "white" if k in d else ("morph" if c.state == "white" else c.state)
        self._refresh_workspace()
        self._selection_changed()

    def _selection_changed(self) -> None:
        if self.scene is not None and isinstance(self.scene, GridScene):
            self.sel_timer.start()                       # rebuild only the morph plan after a short pause

    def _build_after_selection(self) -> None:
        if self.job is not None:
            self._rebuild_after_job = True
        elif self.rb_grid.isChecked() and self.scene is not None:
            self._build()

    # -- quick selections
    def _sel_all(self) -> None:
        self._select(range(self._n_tiles()))

    def _sel_invert(self) -> None:
        self._select(set(range(self._n_tiles())) - self.sel)

    def _sel_rows(self) -> None:
        flat = self._flat()
        rs = {flat[k][0] for k in self.sel if k < len(flat)}
        self._select(k for k, (r, _, _) in enumerate(flat) if r in rs)

    def _sel_every(self, n: int) -> None:
        base = sorted(self.sel) if self.sel else list(range(self._n_tiles()))
        self._select(k for j, k in enumerate(base) if j % n == 0)

    def _sel_random(self) -> None:
        import random
        base = sorted(self.sel) if self.sel else list(range(self._n_tiles()))
        self._select(random.sample(base, len(base) // 2))

    def _sel_state(self, st: str) -> None:
        self._select(k for k, (_, _, c) in enumerate(self._flat()) if c.state == st)

    # -- tab 2: squares ---------------------------------------------------------------------------------
    def _squares_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.sq_summary = QLabel("")
        self.sq_summary.setWordWrap(True)
        self.sq_summary.setStyleSheet("color:#777;font-size:11px")
        v.addWidget(self.sq_summary)
        sg = QGroupBox("Select squares")
        sl = QVBoxLayout(sg)
        grid = QGridLayout()
        items = [("All", self._sel_all), ("None", lambda: self._select([])), ("Invert", self._sel_invert),
                 ("Whole rows", self._sel_rows), ("Every 2nd", lambda: self._sel_every(2)),
                 ("Every 3rd", lambda: self._sel_every(3)), ("Random half", self._sel_random)]
        for st, text in (("morph", "All morphing"), ("stay", "All staying"), ("white", "All white")):
            items.append((text, lambda k=st: self._sel_state(k)))
        for n, (text, fn) in enumerate(items):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=fn: f())
            grid.addWidget(b, n // 4, n % 4)
        sl.addLayout(grid)
        hint = QLabel("In the picture on the left: click or drag a frame to select; Ctrl / Shift adds. "
                      "Switch “Click does” to a Paint tool to set the state of the squares you touch.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#777;font-size:11px")
        sl.addWidget(hint)
        v.addWidget(sg)

        stg = QGroupBox("What the selected squares do")
        stl = QHBoxLayout(stg)
        for st, text in (("morph", "Morph"), ("stay", "Stay a vector"), ("white", "Go to white")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=st: self._set_states(sorted(self.sel), k))
            stl.addWidget(b)
        v.addWidget(stg)

        self.sq_box = QGroupBox("Settings of the selected squares  (empty / “row” = as in the row)")
        f = QFormLayout(self.sq_box)
        self.s_size = QComboBox(); self.s_size.addItems(["As in the row"] + [SIZE_LABEL[m] for m in SIZE_MODES])
        self.s_w = QSpinBox(); self.s_w.setRange(0, 4000); self.s_w.setSpecialValueText("row")
        self.s_h = QSpinBox(); self.s_h.setRange(0, 4000); self.s_h.setSpecialValueText("row")
        sz = QHBoxLayout(); sz.addWidget(self.s_w); sz.addWidget(QLabel("×")); sz.addWidget(self.s_h)
        self.s_fit = QComboBox(); self.s_fit.addItems(["As in the row", "Cover (crop to fill)", "Contain (keep whole picture)"])
        self.s_engine = QComboBox(); self.s_engine.addItems(["As in the row", "Main window's settings", "Image Trace preset", "Linework Snap preset"])
        self.s_preset = QComboBox()
        self.s_colors = QSpinBox(); self.s_colors.setRange(-1, 64); self.s_colors.setSpecialValueText("row")
        self.s_detail = QSpinBox(); self.s_detail.setRange(-1, 100); self.s_detail.setSpecialValueText("row")
        self.s_noise = QSpinBox(); self.s_noise.setRange(-1, 100); self.s_noise.setSpecialValueText("row")
        self.s_side = QComboBox(); self.s_side.addItems(["As in the row", "B enters from the right", "B enters from the left",
                                                          "B enters from the bottom", "B enters from the top"])
        self.s_delay = QSpinBox(); self.s_delay.setRange(-1, 20000); self.s_delay.setSpecialValueText("row"); self.s_delay.setSuffix(" frames")
        self.s_lw = QLineEdit(); self.s_lw.setReadOnly(True); self.s_lw.setPlaceholderText("as in the row")
        lwb = QPushButton("…"); lwb.setFixedWidth(28); lwb.clicked.connect(self._pick_square_lw)
        lwx = QPushButton("×"); lwx.setFixedWidth(28); lwx.clicked.connect(lambda: self._edit_cells(linework="", _lw=True))
        lwr = QHBoxLayout(); lwr.addWidget(self.s_lw, 1); lwr.addWidget(lwb); lwr.addWidget(lwx)
        f.addRow("Square size", self.s_size)
        f.addRow("Size (px)", sz)
        f.addRow("Fit", self.s_fit)
        f.addRow("Vectorise with", self.s_engine)
        f.addRow("Preset", self.s_preset)
        f.addRow("Colours", self.s_colors)
        f.addRow("Detail", self.s_detail)
        f.addRow("Noise", self.s_noise)
        f.addRow("Slider", self.s_side)
        f.addRow("Slider starts after", self.s_delay)
        f.addRow("Own linework", lwr)
        self.s_sweep = TrackEditor(self._sweep_info)
        f.addRow(QLabel("Own slider ranges for these squares:"))
        f.addRow(self.s_sweep)
        self.s_black = QLabel("")
        self.s_black.setWordWrap(True)
        self.s_black.setStyleSheet("color:#777;font-size:11px")
        bpb = QPushButton("Threshold: 0 → this picture's black point")
        bpb.setToolTip("Writes a Threshold range that ends where this picture is (almost) black, using the steps and the % set on the Timeline tab.\n"
                       "A square with its own Threshold range is not touched by 'Finish together'.")
        bpb.clicked.connect(self._square_black)
        f.addRow(self.s_black)
        f.addRow(bpb)
        self.s_size.currentIndexChanged.connect(lambda i: self._edit_cells(size_mode="" if i == 0 else SIZE_MODES[i - 1]))
        self.s_w.valueChanged.connect(lambda x: self._edit_cells(w=x))
        self.s_h.valueChanged.connect(lambda x: self._edit_cells(h=x))
        self.s_fit.currentIndexChanged.connect(lambda i: self._edit_cells(fit=["", "cover", "contain"][i]))
        self.s_engine.currentIndexChanged.connect(self._square_engine)
        self.s_preset.currentIndexChanged.connect(lambda *_: self._edit_cells(preset=self.s_preset.currentText() if self.s_engine.currentIndex() > 1 else ""))
        self.s_colors.valueChanged.connect(lambda x: self._edit_cells(colors=None if x < 0 else x))
        self.s_detail.valueChanged.connect(lambda x: self._edit_cells(detail=None if x < 0 else x))
        self.s_noise.valueChanged.connect(lambda x: self._edit_cells(noise=None if x < 0 else x))
        self.s_side.currentIndexChanged.connect(lambda i: self._edit_cells(side="" if i == 0 else SIDES[i]))
        self.s_delay.valueChanged.connect(lambda x: self._edit_cells(delay=None if x < 0 else x))
        self.s_sweep.changed.connect(lambda: self._edit_cells(sweep=self.s_sweep.tracks()))
        v.addWidget(self.sq_box)

        ab = QHBoxLayout()
        ag = QGridLayout()
        for n, (text, fn) in enumerate((("Replace picture…", self._square_replace), ("◀ move", lambda: self._square_move(-1)),
                                        ("move ▶", lambda: self._square_move(1)), ("Remove square", self._square_remove),
                                        ("Reset to row's settings", self._square_reset))):
            b = QPushButton(text); b.clicked.connect(lambda _=False, f=fn: f()); ag.addWidget(b, n // 3, n % 3)
        v.addLayout(ag)
        self.sq_note = QLabel("")
        self.sq_note.setWordWrap(True)
        self.sq_note.setStyleSheet("color:#777;font-size:11px")
        v.addWidget(self.sq_note)
        v.addStretch(1)
        self.sq_box.setEnabled(False)
        return w

    def _selected_cells(self):
        flat = self._flat()
        return [flat[k] for k in sorted(self.sel) if k < len(flat)]

    def _square_engine(self, i: int) -> None:
        self.s_preset.blockSignals(True)
        self.s_preset.clear()
        if i == 2:
            self.s_preset.addItems(list(TRACE_PRESETS))
        elif i == 3:
            self.s_preset.addItems(list(LW_PRESETS))
        self.s_preset.blockSignals(False)
        self.s_preset.setEnabled(i > 1)
        self.s_colors.setEnabled(i != 3)
        self.s_noise.setEnabled(i != 3)
        self._edit_cells(engine=("", "main", "trace", "linework")[i],
                         preset=self.s_preset.currentText() if i > 1 else "")

    def _load_square_form(self) -> None:
        """Show the settings of the first selected square (edits go to all selected squares)."""
        sel = self._selected_cells()
        self.sq_box.setEnabled(bool(sel))
        if not sel:
            return
        c = sel[0][2]
        self._sq_loading = True
        try:
            self.s_size.setCurrentIndex(0 if not c.size_mode else SIZE_MODES.index(c.size_mode) + 1)
            self.s_w.setValue(c.w); self.s_h.setValue(c.h)
            self.s_fit.setCurrentIndex({"": 0, "cover": 1, "contain": 2}[c.fit])
            ei = ("", "main", "trace", "linework").index(c.engine)
            self.s_engine.setCurrentIndex(ei)
            self.s_preset.blockSignals(True)
            self.s_preset.clear()
            if ei == 2:
                self.s_preset.addItems(list(TRACE_PRESETS))
            elif ei == 3:
                self.s_preset.addItems(list(LW_PRESETS))
            if c.preset:
                j = self.s_preset.findText(c.preset)
                if j >= 0:
                    self.s_preset.setCurrentIndex(j)
            self.s_preset.blockSignals(False)
            self.s_preset.setEnabled(ei > 1)
            self.s_colors.setValue(-1 if c.colors is None else c.colors)
            self.s_detail.setValue(-1 if c.detail is None else c.detail)
            self.s_noise.setValue(-1 if c.noise is None else c.noise)
            self.s_side.setCurrentIndex(0 if not c.side else SIDES.index(c.side))
            self.s_delay.setValue(-1 if c.delay is None else c.delay)
            self.s_lw.setText(c.linework)
            self.s_sweep.set_tracks(c.sweep)
        finally:
            self._sq_loading = False
        self._update_black_label(sel)
        n = len(sel)
        self.sq_note.setText("Editing 1 square: #%d" % (sorted(self.sel)[0] + 1) if n == 1 else
                             f"Editing {n} squares – the values shown are those of #{sorted(self.sel)[0] + 1}; "
                             "whatever you change is applied to all selected squares.")

    def _update_black_label(self, sel) -> None:
        sc = self._black_scene() if self.scene is not None else None
        if sc is None or not sel:
            self.s_black.setText("Black point: shown after Build.")
            return
        k = next((n for n, (r, i, _) in enumerate(self._flat()) if (r, i) == sel[0][:2]), None)
        cv = sc.black_curve(k) if k is not None else None
        if cv is None:
            self.s_black.setText("Black point: – (only black-and-white Image Trace pictures have one).")
            return
        pct = self._sync_pct()
        T = cv.at(pct)
        self.s_black.setText(f"Black point of #{k + 1}: Threshold {T:.1f} (white left {cv.white_at(T) * 100:.2f} %, limit {pct:g} %). "
                             f"At 255 it would be {cv.white_at(255) * 100:.2f} % white.")

    def _edit_cells(self, _lw: bool = False, **kw) -> None:
        if self._sq_loading:
            return
        sel = self._selected_cells()
        if not sel:
            return
        for _, _, c in sel:
            for name, val in kw.items():
                setattr(c, name, [replace(t) for t in val] if name == "sweep" else val)
        if _lw:
            self.s_lw.setText("")
        self._refresh_workspace(form=False)
        if self.scene is not None:
            self.status.setText("Squares changed – press Build to apply (only the changed squares are traced again).")

    def _pick_square_lw(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Linework of the selected squares", "", "Linework (*.svg *.dxf *.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if p:
            self.s_lw.setText(p)
            self._edit_cells(linework=p)

    def _square_reset(self) -> None:
        for _, _, c in self._selected_cells():
            st = c.state
            c.__dict__.update(CellSpec(state=st).__dict__)
        self._refresh_workspace()

    def _square_replace(self) -> None:
        sel = self._selected_cells()
        if len(sel) != 1:
            QMessageBox.information(self, "Replace picture", "Select exactly one square.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "New picture for this square", "", "Images (" + " ".join("*" + e for e in IMG_EXT) + ")")
        if p:
            r, i, _c = sel[0]
            self._rows()[r].paths[i] = p
            self._refresh_rows()
            self._refresh_workspace()

    def _square_move(self, d: int) -> None:
        sel = self._selected_cells()
        if len(sel) != 1:
            return
        r, i, _c = sel[0]
        row = self._rows()[r]
        j = i + d
        if 0 <= j < len(row.paths):
            row.swap(i, j)
            self._select({min(self.sel) + d})
            self._refresh_rows()
            self._refresh_workspace()

    def _square_remove(self) -> None:
        rows = self._rows()
        by_row: Dict[int, list] = {}
        for r, i, _c in self._selected_cells():
            by_row.setdefault(r, []).append(i)
        for r, idx in by_row.items():
            rows[r].remove_at(idx)
        self.sel = set()
        self._refresh_rows()
        self._refresh_workspace()

    def _add_lw(self) -> None:
        ps, _ = QFileDialog.getOpenFileNames(self, "Linework", "", "Linework (*.svg *.dxf *.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        for p in ps:
            self.lw_list.addItem(p)

    def _remove_lw(self) -> None:
        for it in self.lw_list.selectedItems():
            self.lw_list.takeItem(self.lw_list.row(it))

    def morph_options(self) -> MorphOptions:
        st = ["left_right", "right_left", "top_bottom", "bottom_top", "radial", "random", "none"][self.stagger.currentIndex()]
        fit = ["contain", "cover", "stretch", "page"][self.fitmode.currentIndex()]
        return MorphOptions(stagger=st, spread=self.spread.value(), ease=["smooth", "linear", "out"][self.ease.currentIndex()],
                            width_override=self.line_w.value() if self.use_w.isChecked() else None,
                            fit=fit, margin=self.margin.value())

    def _build(self) -> None:
        if self.job is not None:
            return
        s = self.sources()
        lw_paths = [self.lw_list.item(i).text() for i in range(self.lw_list.count())]
        spec: dict = dict(lw_paths=lw_paths, opts=self.morph_options(), dropped=set(),
                          workers=self.workers.value() or 0)
        if self.rb_single.isChecked():
            if s.get("result") is None:
                QMessageBox.information(self, "No picture", s.get("note") or "Trace a picture in the main window first.")
                return
            spec.update(mode="single", result=s["result"])
        else:
            rows = [r for r in self.row_specs if r.paths]
            if not rows:
                QMessageBox.information(self, "No pictures", "Add the pictures for the grid first.")
                return
            if s.get("params") is None:
                QMessageBox.information(self, "Trace settings", "Switch the main window to Image Trace or Linework Snap – "
                                        "rows set to “main window settings” are traced with those.")
                return
            L = GridLayout.from_rows(rows, self.gap.value())
            jobs = grid_jobs(rows, s["params"])
            keys = [job_key(j) for j in jobs]
            white, stay = states_of(rows)
            spec.update(mode="grid", rows=[replace(r, cells=[c.copy() for c in r.sync()]) for r in rows], params=s["params"],
                        layout=L, jobs=jobs, keys=keys, cache=dict(self.tile_cache),
                        lw_mode=["span", "same", "per_tile"][self.lw_mode.currentIndex()],
                        lw_area="morph" if self.lw_area.currentIndex() == 1 else "all", dropped=white, stay=stay)
            self._pending_keys = keys
        self.job = BuildJob(spec)
        self.job.progress.connect(self._on_progress)
        self.job.done.connect(self._on_built)
        self.job.failed.connect(self._on_failed)
        self.job.finished.connect(self._job_finished)
        self.build_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Working…")
        memory.set_budget_gb(float(self.mem_gb.value()))
        self.job.start()

    def _on_progress(self, label: str, i: int, n: int) -> None:
        self.bar.setRange(0, max(1, n))
        self.bar.setValue(i)
        self.status.setText(label + "…")

    def _on_built(self, scene, fresh) -> None:
        self._release_scene(scene)
        self.scene = scene
        if self.rb_grid.isChecked() and self.sel:
            self._update_black_label(self._selected_cells())
        if fresh is not None:                           # keep the traced squares that are still on the grid, drop the rest
            self.tile_cache.update(fresh)
            keep = set(getattr(self, "_pending_keys", ()))
            self.tile_cache = {k: v for k, v in self.tile_cache.items() if k in keep}
        st: dict = {}
        scenes = [scene.merged] if isinstance(scene, GridScene) else [scene]
        for sc in scenes:
            for k, v in (sc.plan.stats if sc.plan else {}).items():
                st[k] = st.get(k, 0) + v
        if st:
            self.morph_stats.setText(f"Matched {st.get('matched', 0):,} paths 1:1, merged {st.get('merged', 0):,} "
                                     f"surplus paths onto lines, {st.get('born', 0):,} lines grow out of the picture.")
        has_lw = self._has_linework()
        if not self._user_timeline:
            self.stages = (grid_timeline(has_lw, scene.layout.canvas[0], self.fps.value()) if isinstance(scene, GridScene)
                           else default_timeline(has_lw))
            self._refresh_stage_list()
        elif has_lw and not any(s.kind == "morph" for s in self.stages):
            self.stages = default_timeline(True)
            self._refresh_stage_list()
        elif not has_lw:
            self.stages = [s for s in self.stages if s.kind != "morph" and "linework" not in (s.img_a, s.img_b)] or default_timeline(False)
            self._refresh_stage_list()
        self.status.setText("Ready." if has_lw else "Ready (no linework loaded: morph stages are unavailable).")
        self._make_renderer()

    @staticmethod
    def _scenes_of(sc) -> list:
        return [] if sc is None else ([sc.merged] if isinstance(sc, GridScene) else [sc])

    def _release_scene(self, new=None) -> None:
        """Free the temporary slider-frame files of the scene that is being replaced. When the new scene shows the very same
        traced pictures (only the morph selection changed), its slider frames are the old ones: they are handed over."""
        if self.sweep_job is not None:
            self.sweep_job.cancel()
            self.sweep_job.wait(5000)
        olds, news = self._scenes_of(self.scene), self._scenes_of(new)
        for o in olds:
            st = getattr(o, "_store", None)
            if st is None:
                continue
            same = next((n for n in news if len(n.sources) == len(o.sources) and all(a is b for a, b in zip(n.sources, o.sources))), None)
            if same is not None:
                same._store = st
            else:
                st.close()
            o._store = None

    def _has_linework(self) -> bool:
        s = self.scene
        if s is None:
            return False
        if isinstance(s, Scene):
            return s.has_linework
        return s.merged.has_linework

    def _on_failed(self, msg: str) -> None:
        if "BrokenProcessPool" in msg or "MemoryError" in msg:
            msg += ("\n\nThe computer ran out of memory. Close other programs (browsers, games, Steam) and press Build again, "
                    "or lower “CPU processes” / “Memory budget” on the Source tab, or use smaller squares.")
        self.status.setText("Failed: " + msg)
        QMessageBox.critical(self, "Build failed", msg)

    def _show_pool_notices(self) -> None:
        for m in pool_mod.take_notices():
            self.status.setText((self.status.text() + "   " if self.status.text() else "") + m)

    def _job_finished(self) -> None:
        self.job = None
        self._show_pool_notices()
        self.build_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._rebuild_after_job:
            self._rebuild_after_job = False
            self.sel_timer.start()

    # ------------------------------------------------------------------ timeline editing
    def _refresh_stage_list(self, select: int = 0) -> None:
        self._loading = True
        self.stage_list.clear()
        for i, s in enumerate(self.stages):
            self.stage_list.addItem(f"{i + 1}.  {s.describe()}")
        self.stage_list.setCurrentRow(min(select, len(self.stages) - 1))
        self._loading = False
        self._stage_selected(self.stage_list.currentRow())
        self._settings_changed()

    def _cur(self) -> Optional[Stage]:
        i = self.stage_list.currentRow()
        return self.stages[i] if 0 <= i < len(self.stages) else None

    def _stage_selected(self, _row: int) -> None:
        st = self._cur()
        self.edit.setEnabled(st is not None)
        if st is None:
            return
        self._loading = True
        self.e_seconds.setValue(st.seconds)
        self.e_a.setCurrentIndex(max(0, [k for _, k in IMAGES].index(st.img_a)))
        self.e_b.setCurrentIndex(max(0, [k for _, k in IMAGES].index(st.img_b)))
        self.e_mode.setCurrentIndex(1 if st.mode == "quad" else 0)
        self.e_dir.setCurrentIndex(max(0, [(a, s) for _, a, s in DIRECTIONS].index((st.axis, st.side_x if st.axis == "x" else st.side_y))
                                       if (st.axis, st.side_x if st.axis == "x" else st.side_y) in [(a, s) for _, a, s in DIRECTIONS] else 0))
        self.e_sx.setCurrentIndex(0 if st.side_x == "right" else 1)
        self.e_sy.setCurrentIndex(0 if st.side_y == "bottom" else 1)
        for wd, val in ((self.e_x0, st.x0), (self.e_x1, st.x1), (self.e_y0, st.y0), (self.e_y1, st.y1), (self.e_lag, st.y_lag)):
            wd.setValue(val)
        self.e_ease.setCurrentIndex(0 if st.easing == "smooth" else 1)
        self.e_pp.setChecked(st.pingpong)
        self.e_ta.setText(st.text_a); self.e_tb.setText(st.text_b); self.e_cap.setText(st.caption)
        self.e_rev.setChecked(st.reverse)
        self.e_scope.setCurrentIndex(["row", "tile", "grid"].index(st.scope))
        self.e_step.setValue(st.step_px)
        self.e_rowmode.setCurrentIndex(["together", "sequence", "stagger"].index(st.row_mode))
        self.e_stagger.setValue(st.row_stagger)
        self.e_alt.setChecked(st.alternate)
        self.e_seq.setChecked(st.track_seq)
        self.e_sync.setValue(st.sync)
        self.e_fit.setCurrentIndex(max(0, self.e_fit.findData(st.fit)))
        self.e_pace.setCurrentIndex(max(0, self.e_pace.findData(st.pace)))
        self.e_tracks.set_tracks(st.tracks)
        self.e_target.setCurrentIndex(["all", "kept", "dropped"].index(st.target))
        self.e_dropped.setCurrentIndex(["auto", "show", "white"].index(st.dropped))
        self.e_seconds.setEnabled(st.step_px == 0)
        grid = self.rb_grid.isChecked()
        wipe, hold, morph, quad = st.kind == "wipe", st.kind == "hold", st.kind == "morph", st.mode == "quad"
        sweep = st.kind == "sweep"
        vis = {"target": sweep and grid, "dropped": (hold or wipe or sweep) and grid, "tracks": sweep, "seq": sweep, "sync": sweep, "fit": sweep and st.sync >= 0, "pace": sweep and st.sync >= 0, "sec": not sweep, "a": wipe or hold, "b": wipe, "mode": wipe, "dir": wipe and not quad,
               "sx": wipe and quad, "sy": wipe and quad, "x0": wipe, "x1": wipe, "y0": wipe and quad, "y1": wipe and quad,
               "lag": wipe and quad, "ease": wipe and st.step_px == 0, "pp": wipe or sweep,
               "step": wipe, "scope": wipe and grid,
               "rowmode": (wipe and grid and st.scope != "grid") or (sweep and grid),
               "stagger": ((wipe and grid and st.scope != "grid") or (sweep and grid)) and st.row_mode == "stagger",
               "alt": wipe and grid and st.scope != "grid", "ta": wipe, "tb": wipe, "cap": hold or morph, "rev": morph or sweep}
        for key, (label, wd) in self.rows_f.items():
            wd.setVisible(vis[key])
            lab = self.edit.layout().labelForField(wd)
            if lab is not None:
                lab.setVisible(vis[key])
        self._loading = False

    def _stage_edited(self, *_) -> None:
        if self._loading:
            return
        st = self._cur()
        if st is None:
            return
        st.seconds = self.e_seconds.value()
        st.img_a, st.img_b = self.e_a.currentData(), self.e_b.currentData()
        st.mode = "quad" if self.e_mode.currentIndex() == 1 else "single"
        _, axis, side = DIRECTIONS[self.e_dir.currentIndex()]
        if st.mode == "single":
            st.axis = axis
            if axis == "x":
                st.side_x = side
            else:
                st.side_y = side
        else:
            st.side_x = "right" if self.e_sx.currentIndex() == 0 else "left"
            st.side_y = "bottom" if self.e_sy.currentIndex() == 0 else "top"
        st.x0, st.x1, st.y0, st.y1, st.y_lag = (self.e_x0.value(), self.e_x1.value(), self.e_y0.value(),
                                                self.e_y1.value(), self.e_lag.value())
        st.easing = "smooth" if self.e_ease.currentIndex() == 0 else "linear"
        st.pingpong = self.e_pp.isChecked()
        st.text_a, st.text_b, st.caption, st.reverse = self.e_ta.text(), self.e_tb.text(), self.e_cap.text(), self.e_rev.isChecked()
        st.scope = ["row", "tile", "grid"][self.e_scope.currentIndex()]
        st.step_px = self.e_step.value()
        st.row_mode = ["together", "sequence", "stagger"][self.e_rowmode.currentIndex()]
        st.row_stagger, st.alternate = self.e_stagger.value(), self.e_alt.isChecked()
        st.tracks, st.track_seq = self.e_tracks.tracks(), self.e_seq.isChecked()
        st.sync, st.fit, st.pace = self.e_sync.value(), self.e_fit.currentData(), self.e_pace.currentData()
        st.target = ["all", "kept", "dropped"][self.e_target.currentIndex()]
        st.dropped = ["auto", "show", "white"][self.e_dropped.currentIndex()]
        self._user_timeline = True
        row = self.stage_list.currentRow()
        self.stage_list.item(row).setText(f"{row + 1}.  {st.describe()}")
        self._stage_selected(row)
        self._settings_changed()

    def _add_stage(self, st: Stage) -> None:
        if (st.kind == "morph") and not self._has_linework():
            QMessageBox.information(self, "No linework", "Add a linework file (tab 2) and press Build first.")
            return
        row = self.stage_list.currentRow()
        self.stages.insert(row + 1, st)
        self._user_timeline = True
        self._refresh_stage_list(row + 1)

    def _dup_stage(self) -> None:
        st = self._cur()
        if st:
            row = self.stage_list.currentRow()
            self.stages.insert(row + 1, replace(st))
            self._user_timeline = True
            self._refresh_stage_list(row + 1)

    def _del_stage(self) -> None:
        row = self.stage_list.currentRow()
        if len(self.stages) > 1 and row >= 0:
            del self.stages[row]
            self._user_timeline = True
            self._refresh_stage_list(max(0, row - 1))

    def _move_stage(self, d: int) -> None:
        row = self.stage_list.currentRow()
        j = row + d
        if 0 <= row < len(self.stages) and 0 <= j < len(self.stages):
            self.stages[row], self.stages[j] = self.stages[j], self.stages[row]
            self._user_timeline = True
            self._refresh_stage_list(j)

    # ------------------------------------------------------------------ settings / renderer
    def clip_settings(self, export: bool = False) -> ClipSettings:
        return ClipSettings(fps=self.fps.value(), bg=["white", "dark", "checker"][self.bg.currentIndex()],
                            show_text=self.show_text.isChecked(), divider=self.divider.isChecked(),
                            invert=self.invert.isChecked(),
                            ss=2 if (export and self.smooth.isChecked()) else 1)

    def _groups_of(self, st: Stage) -> List[Group]:
        sc = self.scene
        if isinstance(sc, GridScene):
            gl = sc.groups("tile" if st.kind == "sweep" else st.scope)        # (sweeps: every square has its own slider)
        elif isinstance(sc, Scene):
            gl = [Group((0, 0) + sc.size)]
        else:
            return [Group((0, 0, 1000, 1000))]
        if st.kind == "sweep" and st.sync >= 0:
            attach_black(gl, self._black_scene())
        return gl

    def _count(self) -> int:
        cs = self.clip_settings()
        drop = self.dropped if isinstance(self.scene, GridScene) else set()
        total = 0
        for st in self.stages:
            gl = self._groups_of(st)
            total += stage_timing(st, cs.fps, gl, cs, active_groups(st, gl, drop)).n
        return total

    def _settings_changed(self, *_) -> None:
        n = self._count()
        self.total_lbl.setText(f"{n} frames · {n / self.fps.value():.1f} s")
        self.rebuild.start()

    def _sweep_info(self) -> dict:
        """Which sliders the main window's engine has, and where they are set right now."""
        p = self.sources().get("params")
        names = [n for n in SWEEP_PARAMS if p is None or hasattr(p, n)]
        if p is not None and getattr(p, "mode", "bw") != "bw" and "threshold" in names and hasattr(p, "colors"):
            names.remove("threshold")                       # colour / grey modes have no threshold slider
        cur = {n: float(getattr(p, n)) for n in names if p is not None and hasattr(p, n)}
        return dict(names=names, current=cur)

    def _story_timeline(self) -> None:
        info = self._sweep_info()
        cur, fps = info["current"], self.fps.value()

        n = self.story_steps.value()
        thr, paths = cur.get("threshold", 128.0), cur.get("paths", 50.0)
        sync, fit, pace = self.story_sync.value(), self.story_fit.currentData(), self.story_pace.currentData()
        if self.story_thr.value() >= 0:
            thr = float(self.story_thr.value())
        if sync >= 0:
            thr = 255.0                                   # each picture's own black point replaces this end value
        appear = []
        if "threshold" in info["names"]:
            appear.append(Track.with_steps("threshold", 0, thr, n))
        if "paths" in info["names"]:
            appear.append(Track.with_steps("paths", 0, paths, n))
        elif "simplify" in info["names"]:
            appear.append(Track.with_steps("simplify", 0, cur.get("simplify", 30.0), n))
        back = ([Track.with_steps("threshold", thr, 0, n)] if "threshold" in info["names"]
                else [Track.with_steps("paths", paths, 0, n)])
        st = [Stage("hold", 0.6, "blank"), Stage("sweep", 0, tracks=appear, sync=sync, fit=fit, pace=pace), Stage("hold", 0.6, "vector"),
              Stage("sweep", 0, tracks=back, target="dropped", sync=sync, fit=fit, pace=pace)]
        if self._has_linework():
            st += [Stage("morph", 3.5), Stage("hold", 1.2, "linework")]
        else:
            st += [Stage("hold", 1.0, "vector")]
        self.stages = st
        self._user_timeline = True
        self._refresh_stage_list(0)
        mode = getattr(self.sources().get("params"), "mode", "bw")
        secs = (2 * n) / max(1, fps)
        self.status.setText(f"Template set: {n} steps per sweep, about {secs:.0f} s of sweeping at {fps} fps."
                            + (f" Every picture is black (white ≤ {sync:g} %) in the same frame: {FIT_NAMES[fit]}, {PACE_NAMES[pace]}." if sync >= 0 else "")
                            + ("" if mode == "bw" else
                            "  Note: the Threshold slider only works in black-and-white mode – use a B/W preset for the rows."))

    def _black_scene(self):
        sc = self.scene
        return sc.merged if isinstance(sc, GridScene) else sc

    def _sync_pct(self) -> float:
        st = self._cur()
        if st is not None and st.kind == "sweep" and st.sync >= 0:
            return st.sync
        return self.story_sync.value() if self.story_sync.value() >= 0 else 0.5

    def _show_black_points(self) -> None:
        """Table of every picture's black point: where its Threshold has to end and how big its step becomes."""
        sc = self._black_scene()
        if sc is None or not getattr(sc, "sources", None):
            QMessageBox.information(self, "Black points", "Press Build first – the pictures have to be loaded.")
            return
        pct, n = self._sync_pct(), self.story_steps.value()
        names = {}
        if self.rb_grid.isChecked():
            for k, (r, i, _c) in enumerate(self._flat()):
                names[k] = os.path.basename(self._rows()[r].paths[i])
        rows = []
        for k in range(len(sc.sources)):
            cv = sc.black_curve(k)
            rows.append((k, names.get(k, f"picture {k + 1}"), None if cv is None else cv.at(pct),
                         None if cv is None else cv.white_at(cv.at(pct)) * 100.0))
        dlg = QDialog(self)
        dlg.setWindowTitle("Black points")
        lay = QVBoxLayout(dlg)
        have = [r for r in rows if r[2] is not None]
        slow = max(have, key=lambda r: r[2], default=None)
        fast = min(have, key=lambda r: r[2], default=None)
        st = self._cur()
        fit, pace = ((st.fit, st.pace) if st is not None and st.kind == "sweep" and st.sync >= 0
                     else (self.story_fit.currentData(), self.story_pace.currentData()))
        v = 255.0 / max(1, n)                                   # normal speed: a full 0 → 255 sweep takes `n` steps
        frames = n if pace == "steps" or not have else int(round((slow if pace == "slow" else fast)[2] / v))
        rows = [(k, nm, T, w) for k, nm, T, w in rows]
        lay.addWidget(QLabel(f"Threshold at which at most {pct:g} % of a picture is white, and the step per frame when all pictures finish "
                             f"together ({FIT_NAMES[fit]}, {PACE_NAMES[pace]}: {frames} frames)."
                             + (f"\nFastest picture: #{fast[0] + 1} (black at {fast[2]:.1f}), slowest: #{slow[0] + 1} (black at {slow[2]:.1f})." if have else "")))
        t = QTableWidget(len(rows), 5)
        t.setHorizontalHeaderLabels(["#", "Picture", "Black at Threshold", "White left", "Step per frame"])
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for r, (k, nm, T, w) in enumerate(rows):
            vals = [str(k + 1), nm, "–" if T is None else f"{T:.2f}", "–" if w is None else f"{w:.2f} %",
                    "–" if T is None else f"{T / max(1, frames):.4f}"]
            for c, v_ in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(v_))
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        t.resizeColumnsToContents()
        lay.addWidget(t)
        if any(r[2] is None for r in rows):
            lay.addWidget(QLabel("– = not a black-and-white Image Trace picture (no Threshold that turns it black)."))
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.resize(620, 420)
        dlg.exec()

    def _square_black(self) -> None:
        """Give each selected square its own Threshold range 0 → its black point (so you can fine-tune it afterwards)."""
        sc = self._black_scene()
        sel = self._selected_cells()
        if sc is None or not sel:
            QMessageBox.information(self, "Black point", "Build first and select the squares.")
            return
        pct, n, done = self._sync_pct(), self.story_steps.value(), 0
        flat = self._flat()
        index = {(r, i): k for k, (r, i, _) in enumerate(flat)}
        for r, i, c in sel:
            cv = sc.black_curve(index[(r, i)])
            if cv is None:
                continue
            c.sweep = [t for t in c.sweep if t.param != "threshold"] + [Track.with_steps("threshold", 0, max(1.0, cv.at(pct)), n)]
            done += 1
        self._load_square_form()
        self._refresh_workspace(form=False)
        self.status.setText(f"Threshold range 0 → black point (white ≤ {pct:g} %, {n} steps) set for {done} square(s). "
                            "Change it in the table below; press Build to apply." if done else
                            "These squares have no black-and-white Image Trace picture, so there is no black point.")

    def _new_sweep(self) -> Stage:
        info = self._sweep_info()
        pick = [n for n in ("threshold", "paths", "noise", "simplify") if n in info["names"]][:2]
        tracks = [default_track(n, info["current"].get(n, SWEEP_PARAMS[n][1])) for n in pick]
        return Stage("sweep", 0, tracks=tracks)

    def _start_sweep_fill(self) -> None:
        if self.sweep_job is not None:
            self.sweep_job.cancel()
        r = self.renderer
        if r is None or not any(st.kind == "sweep" for st in r.stages):
            self.sweep_lbl.setText("")
            return
        job = SweepJob(r, self.workers.value() or 0)
        job.progress.connect(lambda i, n, j=job: self._sweep_progress(j, i, n))
        job.done.connect(lambda ok, j=job: self._sweep_done(j, ok))
        job.failed.connect(lambda m: (self.sweep_lbl.setText("slider frames failed"), self.status.setText("Slider frames: " + m),
                                      QMessageBox.warning(self, "Slider frames", m)))
        self.sweep_job = job
        job.start()

    def _sweep_progress(self, job, i: int, n: int) -> None:
        if job is self.sweep_job:
            self.sweep_lbl.setText(f"Rendering slider frames {i}/{n}…")

    def _sweep_done(self, job, ok: bool) -> None:
        if job is self.sweep_job:
            self.sweep_job = None
            self.sweep_lbl.setText("Slider frames ready." if ok else "")
            self._show_pool_notices()
            if ok:
                self._show(self.scrub.value())

    def _auto_step(self) -> None:
        st = self._cur()
        if st is None:
            return
        groups = self._groups_of(st)
        ext = max((g.box[2] if (st.axis == "x" or st.mode == "quad") else g.box[3]) for g in groups)
        self.e_step.setValue(max(1, int(np.ceil(ext * abs(st.x1 - st.x0) / (4.0 * self.fps.value())))))

    def _make_renderer(self) -> None:
        if self.scene is None:
            return
        try:
            w = [640, 960, 480][self.prev_w.currentIndex()]
            self.renderer = make_renderer(self.scene, self.stages, self.clip_settings(), w)
        except ValueError as exc:
            self.renderer = None
            self.canvas.msg = str(exc)
            self.canvas.show_frame(None)
            return
        cur = self.scrub.value()
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, self.renderer.total - 1)
        self.scrub.setValue(min(cur, self.renderer.total - 1))
        self.scrub.blockSignals(False)
        self._sync_overlay()
        self._start_sweep_fill()
        self._show(self.scrub.value())

    def _show(self, i: int) -> None:
        if self.renderer is None:
            return
        i = min(max(i, 0), self.renderer.total - 1)
        self.canvas.show_frame(self.renderer.frame(i))
        si, _ = self.renderer.locate(i)
        fps = self.fps.value()
        extra = ""
        r = self.renderer
        st = r.stages[si]
        if st.kind == "sweep":                       # the slider values shown in this frame (of the selected square, else the first)
            tm = r.timings[si]
            gi = min(sorted(self.sel)[0] if self.sel else 0, len(tm.groups) - 1)
            vals = sweep_values(st, tm.groups[gi], i - int(r._starts[si]))
            extra = "  ·  " + ", ".join(f"{SWEEP_PARAMS[k][0]} {v:.1f}" for k, v in vals.items())
        self.pos_lbl.setText(f"{i + 1}/{r.total}  ·  {i / fps:.1f} s  ·  stage {si + 1}{extra}")

    def _scrubbed(self, i: int) -> None:
        if self.timer.isActive() and getattr(self, "_from_timer", False):
            return
        self._show(i)

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("▶")
        elif self.renderer is not None:
            self.clock.start()
            self._t0 = self.scrub.value() / self.fps.value()
            self.timer.start()
            self.play.setText("■")

    def _tick(self) -> None:
        r = self.renderer
        if r is None:
            self._toggle_play()
            return
        i = int(self._t0 * self.fps.value() + self.clock.elapsed() / 1000.0 * self.fps.value())
        if i >= r.total:
            if self.loop.isChecked():
                self.clock.start()
                self._t0 = 0.0
                i = 0
            else:
                self._toggle_play()
                i = r.total - 1
        self._from_timer = True
        self.scrub.setValue(i)
        self._from_timer = False
        self._show(i)

    # ------------------------------------------------------------------ export
    def _export(self, kind: str) -> None:
        if self.scene is None:
            QMessageBox.information(self, "Nothing to export", "Press Build first.")
            return
        if self.exporter is not None:
            return
        out, _ = QFileDialog.getSaveFileName(self, f"Save {kind.upper()}", os.path.expanduser(f"~/clip.{kind}"), f"{kind.upper()} (*.{kind})")
        if not out:
            return
        try:
            make_renderer(self.scene, self.stages, self.clip_settings(), 64)      # validates the timeline
        except ValueError as exc:
            QMessageBox.warning(self, "Timeline", str(exc))
            return
        self.exporter = ExportJob(self.scene, list(self.stages), self.clip_settings(True), self.out_w.value(), out, kind, self.workers.value() or 0)
        self.exporter.phase.connect(lambda t: self.exp_phase.setText(t))
        self.exporter.progress.connect(lambda i, n: (self.exp_bar.setRange(0, n), self.exp_bar.setValue(i)))
        self.exporter.done.connect(lambda p: QMessageBox.information(self, "Saved", f"{p}\n{os.path.getsize(p) / 1024:.0f} KB"))
        self.exporter.failed.connect(lambda m: QMessageBox.critical(self, "Export failed", m))
        self.exporter.finished.connect(self._export_finished)
        self.gif_btn.setEnabled(False); self.mp4_btn.setEnabled(False); self.exp_cancel.setEnabled(True)
        self.exporter.start()

    def _export_finished(self) -> None:
        self.exporter = None
        self.gif_btn.setEnabled(True); self.mp4_btn.setEnabled(True); self.exp_cancel.setEnabled(False)
        self.exp_bar.setValue(0)
        self.exp_phase.setText("")

    def closeEvent(self, e):
        self.timer.stop()
        for j in (self.job, self.exporter, self.sweep_job):
            if j is not None:
                j.cancel()
                j.wait(5000)
        self._release_scene()
        e.accept()
