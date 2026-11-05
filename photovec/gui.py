"""PySide6 desktop GUI: drag-and-drop, live progress, split before/after view, exports."""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
import tempfile
import time
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QStackedWidget,
                               QVBoxLayout, QWidget)

from .config import PROFILES, STAGES, RunConfig
from .export import path_stats, render_png, write_dxf, write_pdf, write_svg
from .pipeline import Result, run_pipeline
from .scheduler import Cancelled, Scheduler
from . import trace_export as tx
from .imageio_ import load_image, load_image_alpha
from .trace import TraceResult, composite_checker, prepare_source, recolor, render_trace, trace_bgr, warm_pool
from .trace_panel import TracePanel
from .anim_window import AnimWindow
from .clip_window import ClipStudio
from .multi_window import MultiWindow
from .animate import make_pair
from .linework import LineworkParams, linework_bgr, looks_like_linework
from .linework_panel import LineworkPanel

IMG_FILTER = "Images (*.jpg *.jpeg *.png *.tif *.tiff *.heic *.heif)"


def to_qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    return QImage(np.ascontiguousarray(bgr).data, w, h, 3 * w, QImage.Format_BGR888).copy()


class Worker(QObject):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, path: str, cfg: RunConfig):
        super().__init__()
        self.path, self.cfg = path, cfg
        self.sched = Scheduler(PROFILES[cfg.budget], lambda s, t: self.progress.emit(s, t))

    def run(self) -> None:
        try:
            self.done.emit(run_pipeline(self.path, self.cfg, self.sched))
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # shown to the user, never silently swallowed
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class TraceWorker(QObject):
    """Lives in its own thread; newer requests supersede queued older ones."""
    done = Signal(int, object, object, float)    # id, TraceResult, preview BGR, seconds
    rendered = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self):
        super().__init__()
        self.latest = 0

    @Slot(object, object, str, object, int)
    def run(self, src, params, view, overrides, rid):
        if rid != self.latest:
            return
        try:
            bgr, opaque = src
            t = time.perf_counter()
            res = linework_bgr(bgr, params, opaque) if isinstance(params, LineworkParams) \
                else trace_bgr(bgr, params, opaque)
            dt = time.perf_counter() - t
            if rid != self.latest:
                return
            shown = recolor(res, overrides) if overrides else res
            self.done.emit(rid, res, render_trace(shown, view), dt)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(rid, f"{type(exc).__name__}: {exc}")

    @Slot(object, str, int)
    def render(self, res, view, rid):
        if rid == self.latest:
            self.rendered.emit(rid, render_trace(res, view))


class SplitView(QWidget):
    """Before (left) / after (right) viewer with a draggable divider."""
    picked = Signal(float, float)   # eyedropper: fractional image coordinates
    pick_mode = False

    def __init__(self):
        super().__init__()
        self.before: Optional[QImage] = None
        self.after: Optional[QImage] = None
        self.split = 0.5
        self.setMinimumSize(640, 480)
        self.setAcceptDrops(False)
        self.setMouseTracking(True)

    def set_images(self, before: Optional[QImage], after: Optional[QImage]) -> None:
        self.before, self.after = before, after
        self.update()

    def _target(self) -> QRect:
        img = self.before or self.after
        if img is None:
            return QRect()
        s = min(self.width() / img.width(), self.height() / img.height())
        w, h = int(img.width() * s), int(img.height() * s)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _font(self, size: int, bold: bool = False) -> QFont:
        f = QFont(self.font())                         # the system's own UI font (Segoe UI on Windows, San Francisco on macOS)
        f.setPointSize(size)
        f.setBold(bold)
        return f

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(38, 38, 42))
        r = self._target()
        if r.isNull():
            p.setPen(QColor(170, 170, 175))
            p.setFont(self._font(14))
            p.drawText(self.rect(), Qt.AlignCenter, "Drop a photo here\nor click “Open photo…”")
            return
        x = r.left() + int(r.width() * self.split)
        if self.before is not None:
            p.setClipRect(QRect(r.left(), r.top(), x - r.left(), r.height()))
            p.drawImage(r, self.before)
        if self.after is not None:
            p.setClipRect(QRect(x, r.top(), r.right() - x + 1, r.height()))
            p.drawImage(r, self.after)
        p.setClipping(False)
        if self.before is not None and self.after is not None:
            p.setPen(QPen(QColor(255, 190, 0), 2))
            p.drawLine(x, r.top(), x, r.bottom())
            p.setFont(self._font(9, True))
            p.setPen(QColor(255, 255, 255))
            p.drawText(r.left() + 8, r.top() + 18, "BEFORE")
            p.drawText(r.right() - 52, r.top() + 18, "AFTER")

    def mousePressEvent(self, e):
        if self.pick_mode:
            r = self._target()
            if not r.isNull():
                pt = e.position().toPoint()
                self.pick_mode = False
                self.unsetCursor()
                self.picked.emit(min(1, max(0, (pt.x() - r.left()) / r.width())),
                                 min(1, max(0, (pt.y() - r.top()) / r.height())))
            return
        self._drag(e.position().toPoint())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._drag(e.position().toPoint())

    def _drag(self, pt: QPoint) -> None:
        r = self._target()
        if not r.isNull():
            self.split = min(1.0, max(0.0, (pt.x() - r.left()) / max(1, r.width())))
            self.update()


class MainWindow(QMainWindow):
    trace_request = Signal(object, object, str, object, int)
    render_request = Signal(object, str, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoVec — photo to vector line drawing")
        self.setAcceptDrops(True)
        self.resize(1280, 780)
        self.path: Optional[str] = None
        self.result: Optional[Result] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.stage_name, self.stage_idx = "", 0

        self.view = SplitView()
        self.view.picked.connect(self._on_pick)
        self.rgb: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.src_cache = {}
        self.trace_res: Optional[TraceResult] = None
        self.trace_id = 0
        self.tthread = QThread(self)
        self.tworker = TraceWorker()
        self.tworker.moveToThread(self.tthread)
        self.trace_request.connect(self.tworker.run)
        self.render_request.connect(self.tworker.render)
        self.tworker.done.connect(self._on_trace_done)
        self.tworker.rendered.connect(self._on_rendered)
        self.tworker.failed.connect(self._on_trace_failed)
        self.tthread.start()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(180)
        self.debounce.timeout.connect(self._do_trace)

        self.engine = QComboBox()
        self.engine.addItems(["Image Trace (Illustrator-style)", "Linework Snap (raster lines → vector)",
                              "Line Art (photo → drawing)"])
        self.engine.currentIndexChanged.connect(self._engine_changed)
        self.tpanel = TracePanel()
        self.tpanel.changed.connect(self._trace_param_changed)
        self.tpanel.view_changed.connect(self._trace_view_changed)
        self.tpanel.trace_clicked.connect(self._do_trace)
        self.tpanel.pick_clicked.connect(self._start_pick)
        self.tpanel.palette_changed.connect(self._palette_changed)
        self.lpanel = LineworkPanel()
        self.lpanel.changed.connect(self._trace_param_changed)
        self.lpanel.view_changed.connect(self._trace_view_changed)
        self.lpanel.trace_clicked.connect(self._do_trace)
        self.lpanel.paste_clicked.connect(self._paste)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tpanel)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setWidget(self.lpanel)
        scroll2.setFrameShape(QScrollArea.NoFrame)
        for sc in (scroll, scroll2):
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        QShortcut(QKeySequence.Paste, self, activated=self._paste)
        self.export_base: Optional[str] = None
        line = QWidget()
        lv = QVBoxLayout(line)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._settings_box())
        lv.addWidget(self._run_box())
        lv.addStretch(1)
        self.stack = QStackedWidget()
        self.stack.addWidget(scroll)
        self.stack.addWidget(scroll2)
        self.stack.addWidget(line)
        side = QVBoxLayout()
        side.addWidget(self.engine)
        side.addWidget(self.stack, 1)
        side.addWidget(self._export_box())
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(360)
        root = QHBoxLayout()
        root.addWidget(self.view, 1)
        root.addWidget(side_w)
        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)
        self.statusBar().showMessage("Ready. Drop a photo to begin.")
        self._set_export_enabled(False)

    # ---- panels -------------------------------------------------------------
    def _settings_box(self) -> QGroupBox:
        g = QGroupBox("Settings")
        f = QFormLayout(g)
        self.mode = QComboBox()
        self.mode.addItems(["Auto (proposes General)", "General", "Facade (coming in stage 3)"])
        self.mode.model().item(2).setEnabled(False)
        self.mode.setCurrentIndex(1)
        self.line_aware = QComboBox()
        self.line_aware.addItems(["Line-aware: Auto", "Line-aware: On", "Line-aware: Off"])
        self.budget = QComboBox()
        self.budget.addItems(["15 s  Fast", "30 s  Balanced", "Unlimited  Max quality"])
        self.budget.setCurrentIndex(1)
        self.foliage = QComboBox()
        self.foliage.addItems(["Stylized", "Simplify", "Remove", "Keep raw"])
        self.detail = QComboBox()
        self.detail.addItems(["Clean", "PhotoCAD-like", "Maximum detail"])
        self.detail.setCurrentIndex(1)
        self.simplify = QSlider(Qt.Horizontal)
        self.simplify.setRange(5, 40)
        self.simplify.setValue(10)
        self.simplify.setToolTip("Curve/line fitting tolerance. Higher = fewer nodes.")
        self.stroke = QSlider(Qt.Horizontal)
        self.stroke.setRange(2, 30)
        self.stroke.setValue(6)
        self.stroke.setToolTip("Stroke width, 0.01% of drawing width per step (default 0.058%).")
        self.stroke.valueChanged.connect(self._restyle)
        for lbl, w in (("Mode", self.mode), ("", self.line_aware), ("Time budget", self.budget),
                       ("Foliage", self.foliage), ("Detail", self.detail),
                       ("Simplification", self.simplify), ("Stroke width", self.stroke)):
            f.addRow(lbl, w)
        self.show_overlay = QCheckBox("Draw lines over photo")
        self.show_overlay.toggled.connect(self._restyle)
        self.show_fol = QCheckBox("Show foliage mask")
        self.show_fol.toggled.connect(self._restyle)
        f.addRow(self.show_overlay)
        f.addRow(self.show_fol)
        return g

    def _run_box(self) -> QGroupBox:
        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        self.open_btn = QPushButton("Open photo…")
        self.open_btn.clicked.connect(self._open)
        self.run_btn = QPushButton("Convert")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("font-weight:600;padding:8px")
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.bar = QProgressBar()
        self.bar.setRange(0, len(STAGES))
        self.stage_lbl = QLabel("Idle")
        self.info = QLabel("")
        self.info.setWordWrap(True)
        for w in (self.open_btn, self.run_btn, self.cancel_btn, self.bar, self.stage_lbl, self.info):
            v.addWidget(w)
        return g

    def _export_box(self) -> QGroupBox:
        g = QGroupBox("Export")
        v = QVBoxLayout(g)
        h = QHBoxLayout()
        self.exp_btns = []
        for fmt in ("SVG", "DXF", "PDF", "PNG"):
            b = QPushButton(fmt)
            b.clicked.connect(lambda _=False, f=fmt.lower(): self._export(f))
            h.addWidget(b)
            self.exp_btns.append(b)
        v.addLayout(h)
        self.anim_btn = QPushButton("Compare / Animate…  (slider, GIF, MP4)")
        self.anim_btn.clicked.connect(self._open_anim)
        v.addWidget(self.anim_btn)
        self.clip_btn = QPushButton("Clip Studio…  (sliders, grids, morph into linework)")
        self.clip_btn.clicked.connect(self._open_clip)
        v.addWidget(self.clip_btn)
        self.multi_btn = QPushButton("Multi-Trace…  (many photos, own Threshold / Noise ranges, one clip)")
        self.multi_btn.clicked.connect(self._open_multi)
        v.addWidget(self.multi_btn)
        return g

    def _set_export_enabled(self, on: bool) -> None:
        for b in self.exp_btns:
            b.setEnabled(on)
        self.anim_btn.setEnabled(on)

    # ---- compare / animate window --------------------------------------------------------
    def _anim_provider(self, bg: str, view: str):
        if self.engine.currentIndex() in (0, 1):
            res = self.display_res()
            return None if res is None else make_pair(res, bg, view)
        r = self.result
        if r is None:
            return None
        cfg = replace(self.cfg, stroke_frac=self.stroke.value() * 0.0001)
        return r.image_bgr, render_png(r.paths, r.size, cfg, None, 1.0)

    def _open_anim(self) -> None:
        if not hasattr(self, "anim"):
            self.anim = AnimWindow(self._anim_provider)
        self.anim.refresh()
        self.anim.show()
        self.anim.raise_()
        self.anim.activateWindow()

    def _refresh_anim(self) -> None:
        if hasattr(self, "anim") and self.anim.isVisible():
            self.anim.refresh()
        if hasattr(self, "clip"):
            self.clip.on_main_changed()

    def _clip_sources(self) -> dict:
        """What Clip Studio may use from this window."""
        if self.engine.currentIndex() in (0, 1):
            panel = self.active_panel()
            return dict(result=self.display_res(), params=panel.params(),
                        note="Grid pictures are traced with the current " +
                             ("Image Trace" if self.engine.currentIndex() == 0 else "Linework Snap") + " settings.")
        return dict(result=None, params=None,
                    note="Switch the main window to Image Trace or Linework Snap to use Clip Studio.")

    def _open_multi(self) -> None:
        if not hasattr(self, "multi"):
            self.multi = MultiWindow()
        self.multi.show()
        self.multi.raise_()
        self.multi.activateWindow()

    def _open_clip(self) -> None:
        if not hasattr(self, "clip"):
            self.clip = ClipStudio(self._clip_sources)
        self.clip._refresh_source_note()
        self.clip.show()
        self.clip.raise_()
        self.clip.activateWindow()

    # ---- file handling ----------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.load(u.toLocalFile())
                break

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open photo", "", IMG_FILTER)
        if p:
            self.load(p)

    def load(self, path: str, auto_trace: bool = True) -> None:
        try:
            self.rgb, self.alpha = load_image_alpha(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open", str(exc))
            return
        img = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        self.path, self.result, self.trace_res, self.src_cache = path, None, None, {}
        self.export_base = None
        prev = composite_checker(*prepare_source(self.rgb, 1600, self.alpha))
        self.view.set_images(to_qimage(prev), None)
        self.run_btn.setEnabled(True)
        self._set_export_enabled(False)
        self.info.setText(os.path.basename(path))
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
        if auto_trace and self.engine.currentIndex() in (0, 1) and self.active_panel().preview.isChecked():
            self._do_trace()

    # ---- clipboard ---------------------------------------------------------------------
    def _paste(self) -> None:
        cb = QApplication.clipboard()
        md = cb.mimeData()
        if md.hasImage() and not cb.image().isNull():
            img = cb.image().convertToFormat(QImage.Format_RGBA8888)
            w, h, bpl = img.width(), img.height(), img.bytesPerLine()
            arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, :w * 4]
            arr = arr.reshape(h, w, 4).copy()
            alpha = arr[..., 3]
            tmp = os.path.join(tempfile.gettempdir(), "photovec_pasted.png")
            cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))[1].tofile(tmp)
            self.load(tmp, auto_trace=False)
            self.export_base = os.path.join(os.path.expanduser("~"), "pasted")
            bgr, op = prepare_source(self.rgb, 1600, self.alpha)
            if self.engine.currentIndex() != 1 and looks_like_linework(bgr, op):
                self.engine.setCurrentIndex(1)       # announced, never silent
                self.statusBar().showMessage("Pasted image looks like line art → switched to Linework Snap.")
            else:
                self.statusBar().showMessage("Pasted image from clipboard.")
            if self.engine.currentIndex() in (0, 1) and self.active_panel().preview.isChecked():
                self._do_trace()
        elif md.hasUrls() and md.urls() and md.urls()[0].isLocalFile():
            self.load(md.urls()[0].toLocalFile())
        else:
            self.statusBar().showMessage("Clipboard has no image.")

    def active_panel(self):
        return self.lpanel if self.engine.currentIndex() == 1 else self.tpanel

    # ---- image trace ------------------------------------------------------------------
    def _engine_changed(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.trace_res = None            # results of the other engine are not comparable
        if i in (0, 1):
            self._show_trace()
        else:
            self._show_lineart()

    def _source(self, max_side: int) -> np.ndarray:
        if max_side not in self.src_cache:
            self.src_cache[max_side] = prepare_source(self.rgb, max_side, self.alpha)
        return self.src_cache[max_side]

    def _trace_param_changed(self) -> None:
        if self.active_panel().preview.isChecked() and self.rgb is not None:
            self.debounce.start()

    def _do_trace(self) -> None:
        if self.rgb is None or self.engine.currentIndex() not in (0, 1):
            return
        panel = self.active_panel()
        p = panel.params()
        self.trace_id += 1
        self.tworker.latest = self.trace_id
        self.statusBar().showMessage("Tracing…")
        self.trace_request.emit(self._source(p.max_side), p, panel.view_mode(), self._overrides(), self.trace_id)

    def _on_trace_done(self, rid: int, res: TraceResult, preview, secs: float) -> None:
        if rid != self.trace_id:
            return
        self.trace_res = res
        if self.engine.currentIndex() == 0:
            self.tpanel.set_palette(res)
        if self.engine.currentIndex() == 1:
            self.lpanel.set_info(res.path_count, res.anchor_count, int(res.meta.get("raw_paths", 0)) or None, secs)
        else:
            self.tpanel.set_info(res.path_count, res.anchor_count, res.color_count, secs)
        self._show_after(res, preview)
        self._set_export_enabled(True)
        self._refresh_anim()
        self.statusBar().showMessage(f"Traced in {secs:.2f} s")

    def _on_rendered(self, rid: int, preview) -> None:
        if rid == self.trace_id and self.trace_res is not None:
            self._show_after(self.trace_res, preview)

    def _on_trace_failed(self, rid: int, msg: str) -> None:
        if rid == self.trace_id:
            self.statusBar().showMessage("Trace failed")
            QMessageBox.critical(self, "Trace failed", msg)

    def _show_after(self, res: TraceResult, preview) -> None:
        self.view.set_images(to_qimage(composite_checker(res.image_bgr, res.opaque)), to_qimage(preview))

    def _overrides(self):
        return self.tpanel.palette_overrides() if self.engine.currentIndex() == 0 else {}

    def display_res(self):
        """The trace result with the user's replacement colours applied (what is shown/exported)."""
        if self.trace_res is None:
            return None
        ov = self._overrides()
        return recolor(self.trace_res, ov) if ov else self.trace_res

    def _palette_changed(self) -> None:
        self._trace_view_changed(self.tpanel.view_mode())

    def _trace_view_changed(self, view: str) -> None:
        if self.trace_res is not None:
            self.trace_id += 1
            self.tworker.latest = self.trace_id
            self.render_request.emit(self.display_res(), view, self.trace_id)
        self._refresh_anim()

    def _show_trace(self) -> None:
        if self.trace_res is not None:
            self._trace_view_changed(self.active_panel().view_mode())
            self._set_export_enabled(True)
        else:
            self._set_export_enabled(False)
            if self.rgb is not None and self.active_panel().preview.isChecked():
                self._do_trace()

    def _show_lineart(self) -> None:
        if self.result is not None:
            self._restyle()
            self._set_export_enabled(True)
        elif self.rgb is not None:
            self.view.set_images(to_qimage(composite_checker(*prepare_source(self.rgb, 1600, self.alpha))), None)
            self._set_export_enabled(False)

    def _start_pick(self) -> None:
        self.view.pick_mode = True
        self.view.setCursor(Qt.CrossCursor)
        self.statusBar().showMessage("Click a colour in the image…")

    def _on_pick(self, fx: float, fy: float) -> None:
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        r, g, b = self.rgb[min(h - 1, int(fy * h)), min(w - 1, int(fx * w))]
        self.tpanel.set_ignore_color((int(r), int(g), int(b)), enable=True)

    # ---- run ---------------------------------------------------------------------------
    def _config(self) -> RunConfig:
        cfg = RunConfig(
            budget=("15", "30", "unlimited")[self.budget.currentIndex()],
            mode=("auto", "general", "facade")[self.mode.currentIndex()],
            line_aware=(None, True, False)[self.line_aware.currentIndex()],
            foliage=("stylized", "simplify", "remove", "keep_raw")[self.foliage.currentIndex()],
            detail=("clean", "photocad", "maximum")[self.detail.currentIndex()],
            simplify=self.simplify.value() / 10.0,
            stroke_frac=self.stroke.value() * 0.0001,
        )
        cfg.validate()
        return cfg

    def _start(self):
        if not self.path or self.thread:
            return
        cfg = self._config()
        self.cfg = cfg
        self.worker = Worker(self.path, cfg)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(lambda: self.statusBar().showMessage("Cancelled."))
        self.worker.finished.connect(self._cleanup)
        self.run_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._set_export_enabled(False)
        self.bar.setValue(0)
        self.info.setText("")
        self.stage_name, self.stage_idx = "starting", 0
        self.limit = PROFILES[cfg.budget].limit_s
        self.timer.start(100)
        self.thread.start()

    def _cancel(self):
        if self.worker:
            self.worker.sched.cancel()
            self.stage_lbl.setText("Cancelling…")

    def _on_progress(self, stage: str, _t: float):
        self.stage_name = stage
        self.stage_idx = STAGES.index(stage) if stage in STAGES else self.stage_idx
        self.bar.setValue(self.stage_idx)

    def _tick(self):
        if self.worker:
            e = self.worker.sched.elapsed()
            lim = f" / {self.limit:.0f} s" if self.limit else ""
            self.stage_lbl.setText(f"{self.stage_name}…  {e:.1f} s{lim}")

    def _on_done(self, res: Result):
        self.result = res
        self.bar.setValue(len(STAGES))
        total = sum(res.timings.values())
        st = path_stats(res.paths)
        txt = (f"Done in {total:.1f} s — {st['paths']} paths, {st['nodes']} nodes, "
               f"{st['straight_ratio']:.0%} straight. Line-aware "
               f"{'on' if res.line_aware else 'off'}.")
        if res.fallbacks:
            txt += "\nFallbacks used: " + ", ".join(res.fallbacks)
        self.info.setText(txt)
        self.stage_lbl.setText("Finished")
        if self.engine.currentIndex() == 2:
            self._restyle()
            self._set_export_enabled(True)

    def _on_failed(self, msg: str):
        self.stage_lbl.setText("Failed")
        QMessageBox.critical(self, "Conversion failed", msg)

    def _cleanup(self):
        self.timer.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None
        self.run_btn.setEnabled(bool(self.path))
        self.open_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    # ---- view / export --------------------------------------------------------------------
    def _restyle(self):
        r = self.result
        if r is None:
            return
        cfg = replace(self.cfg, stroke_frac=self.stroke.value() * 0.0001) if hasattr(self, "cfg") \
            else self._config()
        before = r.image_bgr.copy()
        if self.show_fol.isChecked() and r.foliage_mask.any():
            tint = before.copy()
            tint[r.foliage_mask] = (0, 200, 0)
            before = cv2.addWeighted(before, 0.6, tint, 0.4, 0)
        photo = r.image_bgr if self.show_overlay.isChecked() else None
        after = render_png(r.paths, r.size, cfg, photo, 1.0 if photo is not None else 1.0)
        self.view.set_images(to_qimage(before), to_qimage(after))

    def _export(self, fmt: str):
        if not self.path:
            return
        trace = self.engine.currentIndex() in (0, 1)
        if (trace and self.trace_res is None) or (not trace and self.result is None):
            return
        base = (self.export_base or os.path.splitext(self.path)[0]) + f".{fmt}"
        out, _ = QFileDialog.getSaveFileName(self, f"Save {fmt.upper()}", base, f"{fmt.upper()} (*.{fmt})")
        if not out:
            return
        try:
            if trace:
                t = self.display_res()
                {"svg": lambda: tx.write_svg(out, t), "dxf": lambda: tx.write_dxf(out, t),
                 "pdf": lambda: tx.write_pdf(out, t), "png": lambda: tx.write_png(out, t)}[fmt]()
            else:
                r = self.result
                cfg = replace(self.cfg, stroke_frac=self.stroke.value() * 0.0001)
                if fmt == "svg":
                    write_svg(out, r.paths, r.size, cfg)
                elif fmt == "dxf":
                    write_dxf(out, r.paths, r.size, cfg)
                elif fmt == "pdf":
                    write_pdf(out, r.paths, r.size, cfg)
                else:
                    cv2.imencode(".png", render_png(r.paths, r.size, cfg, None, 2.0))[1].tofile(out)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {out}")

    def closeEvent(self, e):
        if hasattr(self, "anim"):
            self.anim.close()
        if hasattr(self, "clip"):
            self.clip.close()
        self.tworker.latest = -1
        self.tthread.quit()
        self.tthread.wait(3000)
        if self.worker:
            self.worker.sched.cancel()
        if self.thread:
            self.thread.quit()
            self.thread.wait(5000)
        e.accept()


def selftest(report: str = "") -> int:
    """Headless check of the PACKAGED program (the build machine runs it on the finished exe / app): every window module imports,
    the main window opens, a photo is vectorised through the worker pool and the vector can be saved. Writes the outcome to `report`
    (a windowed exe has no console). Exit code 0 = everything worked."""
    import importlib
    import tempfile
    lines, ok = [], True

    def step(name, fn):
        nonlocal ok
        try:
            fn()
            lines.append("ok    " + name)
        except Exception as exc:
            ok = False
            lines.append(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    step("imports", lambda: [importlib.import_module("photovec." + m) for m in
                             ("multi_window", "clip_window", "mask_editor", "smartsel", "layers", "ai_client", "trace", "export")])
    folder = tempfile.mkdtemp(prefix="photovec_selftest_")
    img = np.full((360, 480, 3), 235, np.uint8)
    cv2.putText(img, "PV", (40, 250), cv2.FONT_HERSHEY_DUPLEX, 7, (30, 30, 30), 18, cv2.LINE_AA)
    cv2.circle(img, (380, 120), 60, (40, 40, 40), 14, cv2.LINE_AA)
    src = os.path.join(folder, "test.png")
    cv2.imwrite(src, img)
    holder = {}

    def open_and_trace():
        w = MainWindow()
        holder["w"] = w
        w.show()
        w.load(src)
        t0 = time.time()
        while w.trace_res is None and time.time() - t0 < 180:
            app.processEvents()
            time.sleep(0.05)
        if w.trace_res is None:
            raise RuntimeError("no result after 180 s: " + w.statusBar().currentMessage())
        if w.trace_res.path_count < 1:
            raise RuntimeError("the vectorisation found no paths")
    step("main window opens, photo is vectorised (worker pool)", open_and_trace)

    def save_svg():
        out = os.path.join(folder, "out.svg")
        tx.write_svg(out, holder["w"].trace_res)
        if os.path.getsize(out) < 200:
            raise RuntimeError("the SVG is empty")
    if "w" in holder and holder["w"].trace_res is not None:
        step("SVG export", save_svg)

    def smart_brush():
        from .smartsel import SmartSelector
        sel = SmartSelector(img)
        sel.add_stroke([(80, 200)], 8, True)
        sel.solve()
        if not sel.alpha(matte=False).any():
            raise RuntimeError("empty selection")
    step("smart brush", smart_brush)
    lines.append("SELFTEST OK" if ok else "SELFTEST FAILED")
    text = "\n".join(lines)
    if report:
        with open(report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    if "w" in holder:
        holder["w"].close()
        app.processEvents()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)                    # Qt and the worker pool sometimes abort while the interpreter shuts down; the report is written


def main(argv=None) -> int:
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        return selftest(sys.argv[i + 1] if len(sys.argv) > i + 1 else "")
    app = QApplication(argv if argv is not None else sys.argv)
    w = MainWindow()
    w.show()
    QTimer.singleShot(500, lambda: __import__('threading').Thread(target=warm_pool, daemon=True).start())
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[-1]):
        w.load(sys.argv[-1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
