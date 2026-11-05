"""The mask editor: a window in which you tell PhotoVec what is graffiti (the mask) and what is facade (the rest).

Tools
  Smart brush   paint a few strokes on what you want (left button) and on what you do not want (right button or Alt): the selection
                snaps to the edges of the picture, like Photoshop's Quick Selection (see smartsel.py).
  Brush/Eraser  plain painting with an adjustable size and soft edge, for the last touches.
  Magic wand    one click selects the connected area of a similar colour.
  AI click      click on an object (or drag a box around it): the SAM model cuts it out precisely (needs the AI helper).
  AI find       type what to look for ("graffiti", "ornament" ...): every matching object is found (SAM 3 / YOLOE, AI helper).
  Colour guess  one click proposes everything that differs in colour from the main wall colour (no AI needed).

Every proposal (AI, colour) is shown in blue first; "Apply" puts it into the mask as Replace / Add / Subtract. Everything can be undone.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from .smartsel import SmartSelector, grow, magic_wand, paint_disc, remove_specks, smooth_edges

MAX_EDIT = 1600          # the picture is edited at this size at most (the mask is scaled to whatever size it is used at)


def colour_proposal(bgr: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """Everything that differs in colour from the main (wall) colour of the picture, as a mask (uint8 0/255). A starting point
    for easy photos - plain wall, clearly coloured graffiti - that the other tools then correct."""
    from skimage.filters import threshold_otsu
    h, w = bgr.shape[:2]
    s = min(1.0, 500.0 / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    pts = np.ascontiguousarray(lab.reshape(-1, 3)[::7])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, li, cen = cv2.kmeans(pts, 4, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    wall = cen[np.bincount(li.ravel()).argmax()]
    de = np.linalg.norm(lab - wall, axis=2)
    t = threshold_otsu(de) / max(0.3, sensitivity)
    m = (de > t).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 40
    m = keep[lb].astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else m


def preview_result(bgr: np.ndarray, mask: np.ndarray, fade: float = 0.55, blur: float = 6.0) -> np.ndarray:
    """What the layers will look like: the masked part sharp, the rest washed out and blurred."""
    bg = cv2.GaussianBlur(bgr, (0, 0), blur)
    bg = cv2.addWeighted(bg, 1 - fade, np.full_like(bg, 255), fade, 0)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    return (bgr * a + bg * (1 - a)).astype(np.uint8)


def tint_array(m: np.ndarray, rgb: Tuple[int, int, int], opacity: float, outline: Tuple[int, int, int]) -> np.ndarray:
    """The mask as a premultiplied RGBA overlay (colour `rgb`, `opacity`) with a 1 px outline: lookup tables, no float pictures."""
    ramp = np.arange(256, dtype=np.float32)
    lut_a = (ramp * opacity).astype(np.uint8)
    lut_c = [((lut_a.astype(np.uint16) * c) // 255).astype(np.uint8) for c in rgb]                # premultiplied colour of every mask value
    arr = cv2.merge([cv2.LUT(m, lut_c[0]), cv2.LUT(m, lut_c[1]), cv2.LUT(m, lut_c[2]), cv2.LUT(m, lut_a)])
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if cs:
        cv2.drawContours(arr, cs, -1, (int(outline[0]), int(outline[1]), int(outline[2]), 255), 1)
    return arr


def _qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    return QImage(np.ascontiguousarray(bgr).data, w, h, 3 * w, QImage.Format_BGR888).copy()


class Canvas(QWidget):
    """The picture with the mask over it: zoom with the wheel, pan with the middle button or Space + drag."""
    stroke = Signal(str, list, bool)           # tool, points (image coordinates), positive?
    click = Signal(str, float, float, bool)    # tool, x, y, positive?
    box = Signal(float, float, float, float)
    zoom_changed = Signal(float)

    def __init__(self, bgr: np.ndarray):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(520, 420)
        self.bgr = bgr
        self.h, self.w = bgr.shape[:2]
        self._base = _qimage(bgr)
        self.mask = np.zeros((self.h, self.w), np.uint8)
        self.cand: Optional[np.ndarray] = None      # a proposal, drawn in blue
        self.points: List[Tuple[float, float, int]] = []   # AI click points (x, y, label)
        self.mode = 0                               # 0 overlay, 1 mask only, 2 result
        self.opacity = 0.5
        self.tool = "smart"
        self.radius = 18.0
        self.zoom, self.off = 1.0, QPointF(0, 0)
        self.cursor_pos: Optional[QPointF] = None
        self._stroke: List[Tuple[float, float]] = []
        self._neg = False
        self._pan_from: Optional[QPointF] = None
        self._space = False
        self._box_from: Optional[QPointF] = None
        self._box_to: Optional[QPointF] = None
        self._ov: Optional[QImage] = None
        self._cand_img: Optional[QImage] = None
        self._res: Optional[QImage] = None
        self._first = True

    # ------------------------------------------------------------------ data
    def set_mask(self, m: np.ndarray) -> None:
        self.mask = m
        self._ov = self._res = None
        self.update()

    def set_candidate(self, m: Optional[np.ndarray]) -> None:
        self.cand = m
        self._cand_img = None
        self.update()

    def _tint(self, m: np.ndarray, rgb: Tuple[int, int, int], opacity: float, outline: Tuple[int, int, int]) -> QImage:
        arr = tint_array(m, rgb, opacity, outline)
        return QImage(arr.data, self.w, self.h, 4 * self.w, QImage.Format_RGBA8888_Premultiplied).copy()

    def fit(self) -> None:
        s = min(self.width() / self.w, self.height() / self.h) * 0.96
        self.zoom = max(0.05, s)
        self.off = QPointF((self.width() - self.w * self.zoom) / 2, (self.height() - self.h * self.zoom) / 2)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def set_zoom(self, z: float, anchor: Optional[QPointF] = None) -> None:
        z = min(max(z, 0.05), 24.0)
        a = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        img = (a - self.off) / self.zoom
        self.zoom = z
        self.off = a - img * z
        self.zoom_changed.emit(self.zoom)
        self.update()

    def to_image(self, p: QPointF) -> QPointF:
        return (p - self.off) / self.zoom

    def resizeEvent(self, e):
        if self._first:
            self._first = False
            self.fit()
        super().resizeEvent(e)

    # ------------------------------------------------------------------ drawing
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(38, 38, 42))
        p.translate(self.off)
        p.scale(self.zoom, self.zoom)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 1.0)
        if self.mode == 2:
            if self._res is None:
                self._res = _qimage(preview_result(self.bgr, self.mask))
            p.drawImage(0, 0, self._res)
        else:
            p.drawImage(0, 0, self._base)
            if self.mode == 1:
                p.fillRect(QRectF(0, 0, self.w, self.h), QColor(0, 0, 0))
            if self._ov is None:
                self._ov = self._tint(self.mask, (255, 40, 90) if self.mode == 0 else (255, 255, 255), self.opacity if self.mode == 0 else 1.0,
                                      (255, 255, 0))
            p.drawImage(0, 0, self._ov)
        if self.cand is not None:
            if self._cand_img is None:
                self._cand_img = self._tint(self.cand, (40, 120, 255), 0.45, (0, 255, 255))
            p.drawImage(0, 0, self._cand_img)
        if self._stroke:
            col = QColor(255, 60, 60, 170) if self._neg else (QColor(60, 255, 90, 170) if self.tool == "smart" else QColor(255, 255, 255, 120))
            pen = QPen(col, 2 * self.radius, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            pts = [QPointF(x, y) for x, y in self._stroke]
            if len(pts) == 1:
                p.drawPoint(pts[0])
            else:
                p.drawPolyline(pts)
        for x, y, lab in self.points:
            p.setPen(QPen(QColor(0, 0, 0), 2 / self.zoom))
            p.setBrush(QColor(60, 255, 90) if lab else QColor(255, 60, 60))
            p.drawEllipse(QPointF(x, y), 6 / self.zoom, 6 / self.zoom)
        if self._box_from is not None and self._box_to is not None:
            p.setPen(QPen(QColor(0, 255, 255), 2 / self.zoom, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            a, b = self._box_from, self._box_to
            p.drawRect(QRectF(a, b).normalized())
        p.resetTransform()
        if self.cursor_pos is not None and self.tool in ("smart", "brush", "eraser") and self._pan_from is None:
            p.setRenderHint(QPainter.Antialiasing, True)
            r = self.radius * self.zoom
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self.cursor_pos, r, r)
            p.setPen(QPen(QColor(255, 255, 255, 230), 1))
            p.drawEllipse(self.cursor_pos, r, r)

    # ------------------------------------------------------------------ mouse
    def wheelEvent(self, e):
        f = 1.18 ** (e.angleDelta().y() / 120.0)
        self.set_zoom(self.zoom * f, e.position())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._space = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(e)

    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._pan_from = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        neg = e.button() == Qt.RightButton or bool(e.modifiers() & Qt.AltModifier)
        ip = self.to_image(pos)
        if self.tool in ("smart", "brush", "eraser"):
            self._neg = neg or self.tool == "eraser"
            self._stroke = [(ip.x(), ip.y())]
        elif self.tool == "wand":
            self.click.emit("wand", ip.x(), ip.y(), not neg)
        elif self.tool == "ai":
            self._neg = neg
            self._box_from, self._box_to = ip, ip
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        self.cursor_pos = pos
        if self._pan_from is not None:
            self.off += pos - self._pan_from
            self._pan_from = pos
        elif self._stroke:
            ip = self.to_image(pos)
            last = self._stroke[-1]
            if math.hypot(ip.x() - last[0], ip.y() - last[1]) >= max(1.0, self.radius * 0.18):
                self._stroke.append((ip.x(), ip.y()))
        elif self._box_from is not None:
            self._box_to = self.to_image(pos)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._pan_from is not None and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan_from = None
            if self._space:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            return
        if self._stroke:
            pts, self._stroke = self._stroke, []
            self.stroke.emit(self.tool, pts, not self._neg)
        elif self._box_from is not None:
            a, b = self._box_from, self._box_to or self._box_from
            self._box_from = self._box_to = None
            if math.hypot((b.x() - a.x()) * self.zoom, (b.y() - a.y()) * self.zoom) > 8:
                self.box.emit(min(a.x(), b.x()), min(a.y(), b.y()), max(a.x(), b.x()), max(a.y(), b.y()))
            else:
                self.click.emit("ai", a.x(), a.y(), not self._neg)
        self.update()

    def leaveEvent(self, e):
        self.cursor_pos = None
        self.update()


class _Sig(QObject):
    done = Signal(int, str, object)      # generation, what, payload
    failed = Signal(int, str, str)


class MaskEditor(QDialog):
    """Edit the mask of one picture. `bgr` is the picture, `mask` an existing mask (any size) or None. After exec(), `result()`
    gives the mask at the size of the picture shown here (uint8 0..255) or None if the window was cancelled."""

    def __init__(self, bgr: np.ndarray, mask: Optional[np.ndarray] = None, title: str = "", parent=None,
                 ai_factory: Optional[Callable[[], object]] = None, work: int = 640):
        super().__init__(parent)
        self.setWindowTitle("Mask editor" + (f" – {title}" if title else ""))
        self.resize(1250, 820)
        h, w = bgr.shape[:2]
        s = min(1.0, MAX_EDIT / float(max(h, w)))
        self.bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
        self.h, self.w = self.bgr.shape[:2]
        m = np.zeros((self.h, self.w), np.uint8) if mask is None else cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        self.ai_factory = ai_factory
        self.ai = None
        self.sel = SmartSelector(self.bgr, work=work)
        self.undo: List[np.ndarray] = []
        self.redo: List[np.ndarray] = []
        self.sel_synced = False
        self.cands: List[dict] = []
        self.ai_points: List[Tuple[float, float, int]] = []
        self.ai_box: Optional[List[float]] = None
        self._gen = 0
        self._accepted = False
        self.sig = _Sig()
        self.sig.done.connect(self._on_done)
        self.sig.failed.connect(self._on_failed)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.ai_pool = ThreadPoolExecutor(max_workers=1)
        self._final_timer = QTimer(self)
        self._final_timer.setSingleShot(True)
        self._final_timer.setInterval(350)
        self._final_timer.timeout.connect(self._finalise)
        self.canvas = Canvas(self.bgr)
        self.canvas.set_mask(m)
        self._build()
        self.canvas.stroke.connect(self._on_stroke)
        self.canvas.click.connect(self._on_click)
        self.canvas.box.connect(self._on_box)
        self.canvas.zoom_changed.connect(lambda z: self.zoom_lbl.setText(f"{z * 100:.0f} %"))
        self._shortcuts()
        self._tool("smart")

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QHBoxLayout(self)
        side = QVBoxLayout()
        side.setSpacing(5)
        side.addWidget(QLabel("<b>Tools</b>"))
        self.tool_btns = {}
        for key, text, tip in (
                ("smart", "Smart brush (B)", "Paint on what you want (left button). Paint on what you do NOT want with the right button or Alt. "
                                             "The selection snaps to the edges of the picture."),
                ("brush", "Brush (P)", "Paints the mask exactly where you paint."),
                ("eraser", "Eraser (E)", "Removes the mask where you paint."),
                ("wand", "Magic wand (W)", "Click: selects the connected area of a similar colour. Right click subtracts."),
                ("ai", "AI click / box (K)", "Click on an object (left = this, right = not this) or drag a box around it: the SAM model cuts it out. "
                                             "Needs the AI helper.")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._tool(k))
            side.addWidget(b)
            self.tool_btns[key] = b
        opt = QGridLayout()
        self.size = QSlider(Qt.Horizontal); self.size.setRange(2, 200); self.size.setValue(18)
        self.size_lbl = QLabel("18 px")
        self.soft = QSlider(Qt.Horizontal); self.soft.setRange(0, 100); self.soft.setValue(20)
        self.tol = QSlider(Qt.Horizontal); self.tol.setRange(0, 100); self.tol.setValue(50)
        self.tol.setToolTip("Smart brush / wand: how readily similar colours are taken along.")
        self.conn = QCheckBox("Only areas touched by a stroke")
        self.conn.setChecked(True)
        self.conn.setToolTip("Smart brush: off = every similar-coloured area inside the working region is taken, even if no stroke touched it.")
        opt.addWidget(QLabel("Brush size"), 0, 0); opt.addWidget(self.size, 0, 1); opt.addWidget(self.size_lbl, 0, 2)
        opt.addWidget(QLabel("Softness"), 1, 0); opt.addWidget(self.soft, 1, 1)
        opt.addWidget(QLabel("Tolerance"), 2, 0); opt.addWidget(self.tol, 2, 1)
        opt.addWidget(self.conn, 3, 0, 1, 3)
        side.addLayout(opt)
        self.size.valueChanged.connect(self._size_changed)
        side.addSpacing(6)
        side.addWidget(QLabel("<b>Mask</b>"))
        row = QHBoxLayout()
        for text, fn, tip in (("Undo", self.do_undo, "Ctrl+Z"), ("Redo", self.do_redo, "Ctrl+Y"), ("Invert", self.invert, "Swap graffiti and facade"),
                              ("Clear", self.clear, "Empty the mask")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        side.addLayout(row)
        row = QHBoxLayout()
        for text, fn, tip in (("Grow", lambda: self.morph("grow"), "Grow the mask by 1 px"), ("Shrink", lambda: self.morph("shrink"), "Shrink by 1 px"),
                              ("Smooth", lambda: self.morph("smooth"), "Calmer, rounder outline"),
                              ("Specks", lambda: self.morph("specks"), "Remove tiny islands and fill tiny holes")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        side.addLayout(row)
        side.addSpacing(6)
        side.addWidget(QLabel("<b>Proposals</b>"))
        cg = QPushButton("Colour guess")
        cg.setToolTip("Proposes everything that differs in colour from the main wall colour. Works on plain walls, no AI needed.")
        cg.clicked.connect(self.colour_guess)
        side.addWidget(cg)
        arow = QHBoxLayout()
        self.prompt = QLineEdit("graffiti")
        self.prompt.setToolTip("What to look for, in English works best: graffiti, street art, ornament, window, poster ... (several words with commas)")
        self.find_btn = QPushButton("AI find")
        self.find_btn.clicked.connect(self.ai_find)
        arow.addWidget(self.prompt, 1)
        arow.addWidget(self.find_btn)
        side.addLayout(arow)
        erow = QHBoxLayout()
        self.engine = QComboBox()
        self.engine.addItem("Auto", "auto")
        self.engine.addItem("SAM 3 (best)", "sam3")
        self.engine.addItem("YOLOE (light)", "yoloe")
        self.engine.setToolTip("SAM 3 needs its weights file (see README) and an NVIDIA card. YOLOE is small and also runs on the CPU.")
        self.ai_status = QPushButton("Check AI")
        self.ai_status.clicked.connect(self.ai_check)
        erow.addWidget(self.engine, 1)
        erow.addWidget(self.ai_status)
        side.addLayout(erow)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Confidence"))
        self.conf = QSlider(Qt.Horizontal); self.conf.setRange(1, 99); self.conf.setValue(30)
        self.conf.valueChanged.connect(self._refilter)
        crow.addWidget(self.conf, 1)
        side.addLayout(crow)
        zrow = QHBoxLayout()
        zrow.addWidget(QLabel("Largest object"))
        self.maxarea = QSlider(Qt.Horizontal); self.maxarea.setRange(5, 100); self.maxarea.setValue(80)
        self.maxarea.setToolTip("Objects covering more of the picture than this are ignored: for 'graffiti' the AI often answers with the whole wall.")
        self.maxarea.valueChanged.connect(self._refilter)
        zrow.addWidget(self.maxarea, 1)
        side.addLayout(zrow)
        self.cand_lbl = QLabel("No proposal.")
        self.cand_lbl.setStyleSheet("color:#888")
        self.cand_lbl.setWordWrap(True)
        side.addWidget(self.cand_lbl)
        brow = QHBoxLayout()
        self.how = QComboBox()
        self.how.addItems(["Add", "Replace", "Subtract"])
        ap = QPushButton("Apply proposal (Enter)")
        ap.clicked.connect(self.apply_candidate)
        di = QPushButton("Discard")
        di.clicked.connect(self.discard_candidate)
        brow.addWidget(self.how)
        brow.addWidget(ap, 1)
        brow.addWidget(di)
        side.addLayout(brow)
        side.addSpacing(6)
        side.addWidget(QLabel("<b>View</b>"))
        vrow = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems(["Mask over photo", "Mask only", "Result: graffiti sharp, facade faded"])
        self.view.currentIndexChanged.connect(self._view_changed)
        self.op = QSlider(Qt.Horizontal); self.op.setRange(5, 100); self.op.setValue(50)
        self.op.valueChanged.connect(self._opacity_changed)
        vrow.addWidget(self.view, 1)
        vrow.addWidget(self.op)
        side.addLayout(vrow)
        zrow = QHBoxLayout()
        fit = QPushButton("Fit (0)")
        fit.clicked.connect(self.canvas.fit)
        one = QPushButton("100 % (1)")
        one.clicked.connect(lambda: self.canvas.set_zoom(1.0))
        self.zoom_lbl = QLabel("")
        zrow.addWidget(fit); zrow.addWidget(one); zrow.addWidget(self.zoom_lbl)
        side.addLayout(zrow)
        side.addStretch(1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#8a8")
        side.addWidget(self.status)
        brow2 = QHBoxLayout()
        ok = QPushButton("Use this mask")
        ok.setDefault(True)
        ok.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        brow2.addWidget(ok, 1)
        brow2.addWidget(cancel)
        side.addLayout(brow2)
        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(360)
        root.addWidget(sw)
        root.addWidget(self.canvas, 1)

    def _shortcuts(self) -> None:
        def sc(keys, fn):
            QShortcut(QKeySequence(keys), self, activated=fn)
        for k, t in (("B", "smart"), ("P", "brush"), ("E", "eraser"), ("W", "wand"), ("K", "ai")):
            sc(k, lambda t=t: self._tool(t))
        sc("Ctrl+Z", self.do_undo)
        sc("Ctrl+Y", self.do_redo)
        sc("Ctrl+Shift+Z", self.do_redo)
        sc("[", lambda: self.size.setValue(max(2, int(self.size.value() * 0.85))))
        sc("]", lambda: self.size.setValue(min(200, int(self.size.value() * 1.18) + 1)))
        sc("0", self.canvas.fit)
        sc("1", lambda: self.canvas.set_zoom(1.0))
        sc("Return", self.apply_candidate)
        sc("Escape", self.discard_candidate)

    # ------------------------------------------------------------------ small things
    def _tool(self, key: str) -> None:
        self.canvas.tool = key
        for k, b in self.tool_btns.items():
            b.setChecked(k == key)
        self.canvas.update()

    def _size_changed(self, v: int) -> None:
        self.canvas.radius = float(v)
        self.size_lbl.setText(f"{v} px")
        self.canvas.update()

    def _view_changed(self, i: int) -> None:
        self.canvas.mode = i
        self.canvas._ov = self.canvas._res = None
        self.canvas.update()

    def _opacity_changed(self, v: int) -> None:
        self.canvas.opacity = v / 100.0
        self.canvas._ov = None
        self.canvas.update()

    def say(self, text: str, error: bool = False) -> None:
        self.status.setStyleSheet("color:#e66" if error else "color:#8a8")
        self.status.setText(text)

    # ------------------------------------------------------------------ mask history
    def mask(self) -> np.ndarray:
        return self.canvas.mask

    def push_undo(self) -> None:
        self.undo.append(self.canvas.mask.copy())
        del self.undo[:-40]
        self.redo.clear()

    def set_mask(self, m: np.ndarray, sync: bool = False) -> None:
        self.canvas.set_mask(m)
        if not sync:
            self.sel_synced = False             # the smart brush must continue from this mask next time

    def do_undo(self) -> None:
        if self.undo:
            self.redo.append(self.canvas.mask.copy())
            self.set_mask(self.undo.pop())

    def do_redo(self) -> None:
        if self.redo:
            self.undo.append(self.canvas.mask.copy())
            self.set_mask(self.redo.pop())

    def invert(self) -> None:
        self.push_undo()
        self.set_mask(255 - self.canvas.mask)

    def clear(self) -> None:
        self.push_undo()
        self.set_mask(np.zeros_like(self.canvas.mask))

    def morph(self, what: str) -> None:
        m = self.canvas.mask
        self.push_undo()
        if what == "grow":
            m = grow(m, 1)
        elif what == "shrink":
            m = grow(m, -1)
        elif what == "smooth":
            m = smooth_edges(m, 2.0)
        else:
            m = remove_specks(m, max(12, int(self.w * self.h * 0.00005)))
        self.set_mask(m)

    # ------------------------------------------------------------------ strokes
    def _on_stroke(self, tool: str, pts: list, positive: bool) -> None:
        r = self.canvas.radius
        if tool == "smart":
            self._smart(pts, r, positive)
        elif tool in ("brush", "eraser"):
            self.push_undo()
            m = self.canvas.mask.copy()
            paint_disc(m, pts, r, 255 if (tool == "brush" and positive) else 0, self.soft.value() / 100.0)
            self.set_mask(m)

    def _smart(self, pts, r, positive) -> None:
        if not self.sel_synced:
            self.sel.reset()
            self.sel.set_start(self.canvas.mask)
            self.sel_synced = True
        self.push_undo()
        self.sel.add_stroke(pts, r, positive)
        self._gen += 1
        gen = self._gen
        conn, tol = self.conn.isChecked(), self.tol.value() / 100.0
        self.say("Thinking …")

        def job():
            if gen != self._gen:                         # a newer stroke already replaced this one: don't waste the time
                return
            try:
                res = self.sel.solve(connected=conn, tolerance=tol)
                self.sig.done.emit(gen, "smart", self.sel.alpha(res, matte=False))
            except Exception as exc:
                self.sig.failed.emit(gen, "smart", str(exc))
        self.pool.submit(job)

    def _finalise(self) -> None:
        """Quiet moment after the last stroke: the exact edge (a little slower than the quick result)."""
        gen = self._gen

        def job():
            if gen != self._gen:
                return
            try:
                self.sig.done.emit(gen, "final", self.sel.alpha())
            except Exception as exc:
                self.sig.failed.emit(gen, "final", str(exc))
        self.pool.submit(job)

    def _on_done(self, gen: int, what: str, payload) -> None:
        if what in ("smart", "final"):
            if gen != self._gen:
                return
            self.set_mask(payload, sync=True)
            if what == "smart":
                self.say("Edges are being refined …")
                self._final_timer.start()
            else:
                self.say("Done.")
        elif what == "ai":
            self.find_btn.setEnabled(True)
            self.cands = payload["masks"]
            self._refilter()
            self.say(f"{len(self.cands)} object(s) found ({payload.get('engine', '')}).")
        elif what == "aiclick":
            self.find_btn.setEnabled(True)
            m = payload
            if m is None:
                self.say("The model found nothing at that spot.", True)
                return
            self.cands = [{"mask": m, "score": 1.0}]
            self._refilter()
            self.say("Left click = this, right click = not this; Enter applies the object.")
        elif what == "status":
            st = payload
            lines = [f"Device: {st.get('device')} {st.get('gpu', '')}", "SAM 3: " + ("found" if st.get("sam3") else "weights not installed"),
                     "Text engines: " + ", ".join(st.get("text_engines", []) or ["none"])]
            if st.get("errors"):
                lines += st["errors"]
            self.say("\n".join(lines), bool(st.get("errors")))

    def _on_failed(self, gen: int, what: str, msg: str) -> None:
        self.find_btn.setEnabled(True)
        self.say(f"{what}: {msg}", True)

    # ------------------------------------------------------------------ wand, colour guess
    def _on_click(self, tool: str, x: float, y: float, positive: bool) -> None:
        if tool == "wand":
            sel = magic_wand(self.bgr, int(x), int(y), 6 + 60 * self.tol.value() / 100.0, True)
            m = self.canvas.mask.copy()
            self.push_undo()
            if positive:
                m[sel] = 255
            else:
                m[sel] = 0
            self.set_mask(m)
        elif tool == "ai":
            self.ai_points.append((x, y, 1 if positive else 0))
            self.ai_box = None
            self._ai_click()

    def _on_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ai_box = [x0, y0, x1, y1]
        self.ai_points = []
        self._ai_click()

    def colour_guess(self) -> None:
        self.cands = [{"mask": colour_proposal(self.bgr), "score": 1.0}]
        self._refilter()
        self.say("Colour guess shown in blue. Apply it or correct the mask with the other tools.")

    # ------------------------------------------------------------------ AI
    def _ai(self):
        if self.ai is None:
            if self.ai_factory is None:
                from .ai_client import AIClient
                self.ai = AIClient()
            else:
                self.ai = self.ai_factory()
        return self.ai

    def ai_check(self) -> None:
        self.say("Asking the AI helper …")

        def job():
            try:
                self.sig.done.emit(0, "status", self._ai().status(refresh=True))
            except Exception as exc:
                self.sig.failed.emit(0, "AI", str(exc))
        self.ai_pool.submit(job)

    def ai_find(self) -> None:
        words, engine = self.prompt.text(), self.engine.currentData()
        self.find_btn.setEnabled(False)
        self.say("The AI is searching … (the first time it loads the model)")

        def job():
            try:
                self.sig.done.emit(0, "ai", self._ai().find(self.bgr, words, engine, 0.05, key="edit"))
            except Exception as exc:
                self.sig.failed.emit(0, "AI", str(exc))
        self.ai_pool.submit(job)

    def _ai_click(self) -> None:
        pts, labs, box = [(x, y) for x, y, _ in self.ai_points], [l for _, _, l in self.ai_points], self.ai_box
        self.canvas.points = list(self.ai_points)
        self.canvas.update()
        self.say("The AI is cutting out the object …")

        def job():
            try:
                self.sig.done.emit(0, "aiclick", self._ai().segment(self.bgr, pts, labs, box, key="edit"))
            except Exception as exc:
                self.sig.failed.emit(0, "AI", str(exc))
        self.ai_pool.submit(job)

    # ------------------------------------------------------------------ proposals
    def _refilter(self, *_) -> None:
        if not self.cands:
            self.canvas.set_candidate(None)
            self.cand_lbl.setText("No proposal.")
            return
        t = self.conf.value() / 100.0
        big = self.maxarea.value() / 100.0
        use = [c for c in self.cands if c["score"] >= t and float((c["mask"] > 127).mean()) <= big]
        if not use:
            self.canvas.set_candidate(None)
            self.cand_lbl.setText(f"{len(self.cands)} object(s), none above confidence {t:.2f} and smaller than {int(big * 100)} % of the picture.")
            return
        m = np.zeros((self.h, self.w), np.uint8)
        for c in use:
            np.maximum(m, c["mask"], out=m)
        self.canvas.set_candidate(m)
        self.cand_lbl.setText(f"Proposal (blue): {len(use)} of {len(self.cands)} object(s) at confidence ≥ {t:.2f}.")

    def apply_candidate(self) -> None:
        c = self.canvas.cand
        if c is None:
            return
        self.push_undo()
        how = self.how.currentIndex()
        m = self.canvas.mask
        if how == 0:
            m = np.maximum(m, c)
        elif how == 1:
            m = c.copy()
        else:
            m = np.minimum(m, 255 - c)
        self.set_mask(m)
        self.discard_candidate()
        self.say("Proposal applied.")

    def discard_candidate(self) -> None:
        self.cands = []
        self.ai_points, self.ai_box = [], None
        self.canvas.points = []
        self.canvas.set_candidate(None)
        self.cand_lbl.setText("No proposal.")

    # ------------------------------------------------------------------ finish
    def _accept(self) -> None:
        self._accepted = True
        self.accept()

    def result(self) -> Optional[np.ndarray]:
        return self.canvas.mask.copy() if self._accepted else None

    def done(self, r: int) -> None:
        self._final_timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.ai_pool.shutdown(wait=False, cancel_futures=True)
        if self.ai is not None and self.ai_factory is None and hasattr(self.ai, "close"):          # a shared helper stays loaded
            try:
                self.ai.close()
            except Exception:
                pass
        super().done(r)
