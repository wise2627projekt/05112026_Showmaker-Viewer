"""Letter editor: mark every letter of a tag with a skeleton drawn with a pen.

Click = a corner anchor, click and drag = an anchor with bezier handles, click on the first anchor of an open path = close it.
Every segment between two anchors is either straight or curved (buttons, or drag its handles). Enter / right click finishes a path (a letter may
have several: the dot of an "i", the bar of an "A"). Each letter gets the character it is (type it in the box): the Show Maker later fits the
skeleton onto that character in the chosen font.

The pen maths (`segment_points`, `split_segment`, `make_curved`, ...) are plain functions, so they are tested without a window."""
from __future__ import annotations

import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from .letters import Letter, Stroke

ANCHOR_R = 6.5          # screen px
HANDLE_R = 5.0
HIT = 9.0
SEG_HIT = 6.0
DRAG_START = 3.0        # px of mouse travel before a click becomes a drag that pulls handles out
MAX_UNDO = 120


# ---------------------------------------------------------------------------------------------------- pen maths
def _pt(q) -> np.ndarray:
    return np.array([float(q[0]), float(q[1])])


def segment_ctrl(s: Stroke, seg: int):
    """(p0, c0, c1, p1) of segment `seg` (the handles default to the anchors: a straight segment)."""
    i, j = s.segments()[seg]
    p0, p1 = _pt(s.pts[i]), _pt(s.pts[j])
    if s.hout[i] is None and s.hin[j] is None:                          # straight: handles at the thirds, so the parameter runs evenly along the line
        return p0, p0 + (p1 - p0) / 3.0, p1 - (p1 - p0) / 3.0, p1
    return p0, (_pt(s.hout[i]) if s.hout[i] is not None else p0), (_pt(s.hin[j]) if s.hin[j] is not None else p1), p1


def segment_points(s: Stroke, seg: int, n: int = 24) -> np.ndarray:
    """n + 1 points along the segment."""
    p0, c0, c1, p1 = segment_ctrl(s, seg)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    g = 1.0 - t
    return g ** 3 * p0 + 3 * g * g * t * c0 + 3 * g * t * t * c1 + t ** 3 * p1


def nearest_on_stroke(s: Stroke, pt, tol: float) -> Optional[Tuple[int, float, float]]:
    """(segment, t, distance) of the point of the path nearest to `pt` if it is within `tol`, else None."""
    best = None
    q = _pt(pt)
    for seg in range(len(s.segments())):
        P = segment_points(s, seg, 32)
        d = np.hypot(P[:, 0] - q[0], P[:, 1] - q[1])
        k = int(d.argmin())
        if d[k] <= tol and (best is None or d[k] < best[2]):
            best = (seg, k / 32.0, float(d[k]))
    return best


def split_segment(s: Stroke, seg: int, t: float) -> int:
    """Insert an anchor at parameter t of segment `seg` without changing the shape (de Casteljau). Returns the new anchor's index."""
    i, j = s.segments()[seg]
    p0, c0, c1, p1 = segment_ctrl(s, seg)
    straight = s.hout[i] is None and s.hin[j] is None
    a, b, c = p0 + (c0 - p0) * t, c0 + (c1 - c0) * t, c1 + (p1 - c1) * t
    d, e = a + (b - a) * t, b + (c - b) * t
    m = d + (e - d) * t
    at = i + 1 if j != 0 or not s.closed else len(s.pts)
    s.pts.insert(at, tuple(m))
    s.hin.insert(at, None if straight else tuple(d))
    s.hout.insert(at, None if straight else tuple(e))
    if not straight:
        s.hout[i] = tuple(a)
        jj = (j + 1) if j >= at else j
        s.hin[jj] = tuple(c)
    return at


def make_curved(s: Stroke, seg: int) -> None:
    """Give a straight segment handles at a third of the way (they lie on the line, so the shape stays until you pull them)."""
    i, j = s.segments()[seg]
    p0, p1 = _pt(s.pts[i]), _pt(s.pts[j])
    if s.hout[i] is None:
        s.hout[i] = tuple(p0 + (p1 - p0) / 3.0)
    if s.hin[j] is None:
        s.hin[j] = tuple(p1 - (p1 - p0) / 3.0)


def make_straight(s: Stroke, seg: int) -> None:
    i, j = s.segments()[seg]
    s.hout[i] = None
    s.hin[j] = None


def is_smooth(s: Stroke, i: int, tol: float = 0.12) -> bool:
    """Both handles exist and point in opposite directions (a smooth point)."""
    if s.hin[i] is None or s.hout[i] is None:
        return False
    a, b = _pt(s.hin[i]) - _pt(s.pts[i]), _pt(s.hout[i]) - _pt(s.pts[i])
    la, lb = np.hypot(*a), np.hypot(*b)
    return la > 1e-6 and lb > 1e-6 and float(a @ b) / (la * lb) < -1.0 + tol


def toggle_smooth(s: Stroke, i: int) -> None:
    """Smooth point -> corner (handles removed); corner -> smooth (handles along the line between its neighbours)."""
    n = len(s.pts)
    if is_smooth(s, i):
        s.hin[i] = s.hout[i] = None
        return
    prev = s.pts[i - 1] if (i > 0 or s.closed) else None
    nxt = s.pts[(i + 1) % n] if (i < n - 1 or s.closed) else None
    p = _pt(s.pts[i])
    if prev is not None and nxt is not None:
        d = _pt(nxt) - _pt(prev)
    elif nxt is not None:
        d = _pt(nxt) - p
    elif prev is not None:
        d = p - _pt(prev)
    else:
        return
    ln = math.hypot(*d)
    if ln < 1e-6:
        return
    d = d / ln
    reach = [math.hypot(*(_pt(q) - p)) / 3.0 for q in (prev, nxt) if q is not None]
    r = float(min(reach)) if reach else 10.0
    s.hin[i] = tuple(p - d * r) if (prev is not None) else None
    s.hout[i] = tuple(p + d * r) if (nxt is not None) else None


def delete_anchor(s: Stroke, i: int) -> None:
    for lst in (s.pts, s.hin, s.hout):
        del lst[i]
    if len(s.pts) < 3:
        s.closed = False


def move_anchor(s: Stroke, i: int, delta) -> None:
    d = _pt(delta)
    s.pts[i] = tuple(_pt(s.pts[i]) + d)
    for lst in (s.hin, s.hout):
        if lst[i] is not None:
            lst[i] = tuple(_pt(lst[i]) + d)


def move_handle(s: Stroke, i: int, which: str, to, mirror: bool) -> None:
    """Move the 'in' or 'out' handle of anchor i to `to`; with `mirror` the other handle follows on the opposite side (same length when it had one)."""
    p = _pt(s.pts[i])
    q = _pt(to)
    this, other = (s.hin, s.hout) if which == "in" else (s.hout, s.hin)
    this[i] = tuple(q)
    if mirror:
        v = q - p
        ln = math.hypot(*v)
        if ln > 1e-9 and other[i] is not None:
            keep = math.hypot(*(_pt(other[i]) - p))
            other[i] = tuple(p - v / ln * keep)
        elif other[i] is None:
            other[i] = tuple(p - v)


def letter_to_state(letters: List[Letter], cur: int) -> str:
    return json.dumps({"letters": [l.to_dict() for l in letters], "cur": cur})


def state_to_letters(s: str) -> Tuple[List[Letter], int]:
    d = json.loads(s)
    return [Letter.from_dict(x) for x in d["letters"]], int(d["cur"])


# ---------------------------------------------------------------------------------------------------- the canvas
LETTER_COLORS = [QColor(255, 99, 71), QColor(0, 200, 255), QColor(255, 196, 0), QColor(120, 220, 120), QColor(220, 130, 255), QColor(255, 150, 60)]


class SkeletonCanvas(QWidget):
    """The photo with the skeletons on top. Coordinates of the letters are pixels of the ORIGINAL photo (size `orig`)."""
    changed = Signal()              # the letters were edited
    cur_changed = Signal(int)       # another letter became current (by clicking in the canvas)
    char_typed = Signal(str)        # a character key was pressed: that is the character of the current letter

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(520, 400)
        self.orig = (1, 1)
        self.photo: Optional[QImage] = None
        self.drawing: Optional[QImage] = None
        self.show_drawing = True
        self.letters: List[Letter] = []
        self.cur = -1
        self.active = -1                      # the stroke of the current letter that new anchors are added to (-1 = none)
        self.sel: Optional[Tuple[int, int]] = None      # (stroke, anchor)
        self.sel_seg: Optional[Tuple[int, int]] = None  # (stroke, segment)
        self.zoom = 1.0
        self.off = QPointF(0, 0)
        self._undo: List[str] = []
        self._redo: List[str] = []
        self._op: Optional[dict] = None       # what the mouse is doing right now
        self._pushed = False
        self._pan_from: Optional[QPointF] = None
        self._space = False
        self.sealed = False                   # the current letter got its character: the next click on empty ground starts a new letter

    # ---- setup
    def set_images(self, photo: Optional[QImage], orig: Tuple[int, int]) -> None:
        self.photo, self.orig = photo, (max(1, orig[0]), max(1, orig[1]))
        self.fit()

    def set_drawing(self, img: Optional[QImage]) -> None:
        self.drawing = img
        self.update()

    def set_letters(self, letters: List[Letter], cur: int = -1) -> None:
        self.letters = letters
        self.cur = cur if 0 <= cur < len(letters) else (0 if letters else -1)
        self.active, self.sel, self.sel_seg = -1, None, None
        self._undo.clear()
        self._redo.clear()
        self.update()

    def fit(self) -> None:
        w, h = self.orig
        self.zoom = min((self.width() - 20) / w, (self.height() - 20) / h)
        self.off = QPointF((self.width() - w * self.zoom) / 2, (self.height() - h * self.zoom) / 2)
        self.update()

    def resizeEvent(self, e):
        if self._undo == [] and self.zoom == 1.0:
            self.fit()
        super().resizeEvent(e)

    # ---- coordinates
    def to_img(self, p: QPointF) -> np.ndarray:
        return np.array([(p.x() - self.off.x()) / self.zoom, (p.y() - self.off.y()) / self.zoom])

    def to_scr(self, q) -> QPointF:
        return QPointF(q[0] * self.zoom + self.off.x(), q[1] * self.zoom + self.off.y())

    # ---- undo
    def _push(self) -> None:
        if self._pushed:
            return
        self._pushed = True
        self._undo.append(letter_to_state(self.letters, self.cur))
        del self._undo[:-MAX_UNDO]
        self._redo.clear()

    def undo(self) -> None:
        if self._undo:
            self._redo.append(letter_to_state(self.letters, self.cur))
            self.letters, self.cur = state_to_letters(self._undo.pop())
            self._after_state()

    def redo(self) -> None:
        if self._redo:
            self._undo.append(letter_to_state(self.letters, self.cur))
            self.letters, self.cur = state_to_letters(self._redo.pop())
            self._after_state()

    def _after_state(self) -> None:
        self.active, self.sel, self.sel_seg = -1, None, None
        self.update()
        self.changed.emit()
        self.cur_changed.emit(self.cur)

    # ---- helpers on the current letter
    def letter(self) -> Optional[Letter]:
        return self.letters[self.cur] if 0 <= self.cur < len(self.letters) else None

    def _hit_handle(self, L: Letter, scr: QPointF):
        if self.sel is None:
            return None
        si, ai = self.sel
        s = L.strokes[si]
        n = len(s.pts)
        cands = [ai] + ([(ai - 1) % n, (ai + 1) % n] if s.closed else [k for k in (ai - 1, ai + 1) if 0 <= k < n])
        for k in cands:
            for which, h in (("in", s.hin[k]), ("out", s.hout[k])):
                if h is not None and math.hypot(*(np.array([self.to_scr(h).x(), self.to_scr(h).y()]) - [scr.x(), scr.y()])) <= HIT:
                    return si, k, which
        return None

    def _hit_anchor(self, scr: QPointF, letters_only_current: bool = True):
        L = self.letter()
        if L is None:
            return None
        best = None
        for si, s in enumerate(L.strokes):
            for ai, p in enumerate(s.pts):
                q = self.to_scr(p)
                d = math.hypot(q.x() - scr.x(), q.y() - scr.y())
                if d <= HIT and (best is None or d < best[0]):
                    best = (d, si, ai)
        return None if best is None else (best[1], best[2])

    def _hit_segment(self, L: Letter, scr: QPointF):
        q = self.to_img(scr)
        tol = SEG_HIT / self.zoom
        for si, s in enumerate(L.strokes):
            r = nearest_on_stroke(s, q, tol)
            if r is not None:
                return si, r[0], r[1]
        return None

    def _letter_at(self, scr: QPointF) -> int:
        q = self.to_img(scr)
        tol = HIT / self.zoom
        for li, L in enumerate(self.letters):
            if li == self.cur:
                continue
            for s in L.strokes:
                if nearest_on_stroke(s, q, tol) is not None or any(math.hypot(*(_pt(p) - q)) <= tol for p in s.pts):
                    return li
        return -1

    # ---- mouse
    def mousePressEvent(self, e):
        self.setFocus()
        scr = e.position()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._pan_from = scr
            return
        if e.button() == Qt.RightButton:
            self.finish_stroke()
            return
        if e.button() != Qt.LeftButton:
            return
        self._pushed = False
        L = self.letter()
        if L is not None:
            h = self._hit_handle(L, scr)
            if h is not None:
                self._op = {"kind": "handle", "stroke": h[0], "anchor": h[1], "which": h[2], "alt": bool(e.modifiers() & Qt.AltModifier)}
                return
            a = self._hit_anchor(scr)
            if a is not None:
                si, ai = a
                s = L.strokes[si]
                if si == self.active and ai == 0 and len(s.pts) >= 3 and not s.closed:        # back on the first anchor: close the path
                    self._push()
                    s.closed = True
                    self.active, self.sel = -1, (si, ai)
                    self.changed.emit()
                    self.update()
                    return
                self.sel, self.sel_seg = (si, ai), None
                self._op = {"kind": "move", "stroke": si, "anchor": ai, "last": self.to_img(scr), "moved": False}
                self.update()
                return
            seg = self._hit_segment(L, scr)
            if seg is not None:
                si, k, t = seg
                if e.modifiers() & Qt.ControlModifier:                                            # Ctrl + click on a path: a new anchor there
                    self._push()
                    ai = split_segment(L.strokes[si], k, t)
                    self.sel, self.sel_seg = (si, ai), None
                    self.changed.emit()
                else:
                    self.sel, self.sel_seg = None, (si, k)
                self.update()
                return
        other = self._letter_at(scr)
        if other >= 0 and not (e.modifiers() & Qt.ShiftModifier):                                # click on another letter's path: switch to it
            self.cur, self.active, self.sel, self.sel_seg = other, -1, None, None
            self.cur_changed.emit(other)
            self.update()
            return
        # empty place: a new anchor
        self._push()
        if L is not None and self.sealed and L.char:
            L = None
        self.sealed = False
        if L is None:
            self.letters.append(Letter("", []))
            self.cur = len(self.letters) - 1
            L = self.letters[self.cur]
            self.cur_changed.emit(self.cur)
        if not (0 <= self.active < len(L.strokes)) or L.strokes[self.active].closed:
            L.strokes.append(Stroke([], [], [], False))
            self.active = len(L.strokes) - 1
        s = L.strokes[self.active]
        s.pts.append(tuple(self.to_img(scr)))
        s.hin.append(None)
        s.hout.append(None)
        self.sel, self.sel_seg = (self.active, len(s.pts) - 1), None
        self._op = {"kind": "create", "stroke": self.active, "anchor": len(s.pts) - 1, "start": scr}
        self.changed.emit()
        self.update()

    def mouseMoveEvent(self, e):
        scr = e.position()
        if self._pan_from is not None:
            self.off += scr - self._pan_from
            self._pan_from = scr
            self.update()
            return
        op, L = self._op, self.letter()
        if op is None or L is None:
            self.setCursor(Qt.OpenHandCursor if self._space else Qt.CrossCursor)
            return
        s = L.strokes[op["stroke"]]
        k, q = op["anchor"], self.to_img(scr)
        if op["kind"] == "create":
            st = op["start"]
            if math.hypot(scr.x() - st.x(), scr.y() - st.y()) >= DRAG_START:                    # dragged: pull symmetric handles out
                p = _pt(s.pts[k])
                s.hout[k] = tuple(q)
                s.hin[k] = tuple(2 * p - q)
        elif op["kind"] == "move":
            d = q - op["last"]
            if d.any():
                self._push()
                move_anchor(s, k, d)
                op["last"] = q
                op["moved"] = True
        elif op["kind"] == "handle":
            self._push()
            move_handle(s, k, op["which"], q, mirror=not bool(e.modifiers() & Qt.AltModifier) and is_smooth(s, k, 0.5))
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.LeftButton) and self._pan_from is not None:
            self._pan_from = None
        self._op = None

    def mouseDoubleClickEvent(self, e):
        L = self.letter()
        if L is None or e.button() != Qt.LeftButton:
            return
        a = self._hit_anchor(e.position())
        if a is not None:                                                                     # double click on an anchor: corner <-> smooth
            self._push()
            toggle_smooth(L.strokes[a[0]], a[1])
            self.sel = a
            self.changed.emit()
        else:
            self.finish_stroke()
        self.update()

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if not d:
            return
        f = 1.15 if d > 0 else 1 / 1.15
        p = e.position()
        before = self.to_img(p)
        self.zoom = min(40.0, max(0.05, self.zoom * f))
        self.off = QPointF(p.x() - before[0] * self.zoom, p.y() - before[1] * self.zoom)
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space:
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        elif k in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            self.finish_stroke()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif k == Qt.Key_Z and e.modifiers() & Qt.ControlModifier:
            self.redo() if e.modifiers() & Qt.ShiftModifier else self.undo()
        elif k == Qt.Key_Y and e.modifiers() & Qt.ControlModifier:
            self.redo()
        else:
            t = e.text()
            if len(t) == 1 and t.isprintable() and not t.isspace() and not (e.modifiers() & (Qt.ControlModifier | Qt.AltModifier)) and self.letter() is not None:
                self.finish_stroke()
                self.char_typed.emit(t)
            else:
                super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.setCursor(Qt.CrossCursor)
        super().keyReleaseEvent(e)

    # ---- actions (also wired to the dialog's buttons)
    def finish_stroke(self) -> None:
        L = self.letter()
        if L is not None and 0 <= self.active < len(L.strokes):
            if len(L.strokes[self.active].pts) < 2:                     # a lonely anchor is not a path: drop it
                self._push()
                del L.strokes[self.active]
                self.sel = None
                self.changed.emit()
        self.active = -1
        self.update()

    def delete_selected(self) -> None:
        L = self.letter()
        if L is None:
            return
        self._push()
        if self.sel is not None and self.sel[0] < len(L.strokes):
            si, ai = self.sel
            s = L.strokes[si]
            delete_anchor(s, ai)
            if not s.pts:
                del L.strokes[si]
                self.active = -1
            self.sel = None
        elif self.sel_seg is not None:                                  # a selected segment: remove the stroke's anchors on both ends? no - the whole stroke
            del L.strokes[self.sel_seg[0]]
            self.sel_seg, self.active = None, -1
        self.changed.emit()
        self.update()

    def set_segment(self, curved: bool) -> None:
        L = self.letter()
        if L is None or self.sel_seg is None:
            return
        si, k = self.sel_seg
        self._push()
        (make_curved if curved else make_straight)(L.strokes[si], k)
        self.changed.emit()
        self.update()

    def toggle_closed(self) -> None:
        L = self.letter()
        if L is None:
            return
        si = self.sel[0] if self.sel is not None else (self.active if self.active >= 0 else len(L.strokes) - 1)
        if 0 <= si < len(L.strokes) and len(L.strokes[si].pts) >= 3:
            self._push()
            L.strokes[si].closed = not L.strokes[si].closed
            self.changed.emit()
            self.update()

    def toggle_smooth_selected(self) -> None:
        L = self.letter()
        if L is not None and self.sel is not None:
            self._push()
            toggle_smooth(L.strokes[self.sel[0]], self.sel[1])
            self.changed.emit()
            self.update()

    def new_stroke(self) -> None:
        self.active, self.sel, self.sel_seg = -1, None, None
        self.update()

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(40, 40, 44))
        w, h = self.orig
        r = QRectF(self.off.x(), self.off.y(), w * self.zoom, h * self.zoom)
        img = self.drawing if (self.show_drawing and self.drawing is not None) else self.photo
        if img is not None:
            p.drawImage(r, img)
        else:
            p.fillRect(r, QColor(90, 90, 95))
        for li, L in enumerate(self.letters):
            self._paint_letter(p, L, li, li == self.cur)

    def _paint_letter(self, p: QPainter, L: Letter, li: int, current: bool) -> None:
        col = LETTER_COLORS[li % len(LETTER_COLORS)]
        for si, s in enumerate(L.strokes):
            pen = QPen(col, 2.4 if current else 1.4)
            pen.setCosmetic(True)
            for seg in range(len(s.segments())):
                poly = QPolygonF([self.to_scr(q) for q in segment_points(s, seg, 28)])
                hot = current and self.sel_seg == (si, seg)
                p.setPen(QPen(QColor(255, 255, 255), 3.4) if hot else pen)
                p.setBrush(Qt.NoBrush)
                p.drawPolyline(poly)
            if not current:
                continue
            show_h = set()
            if self.sel is not None and self.sel[0] == si:
                n = len(s.pts)
                a = self.sel[1]
                show_h = {a, (a - 1) % n, (a + 1) % n} if s.closed else {k for k in (a - 1, a, a + 1) if 0 <= k < n}
            for k in show_h:
                for h in (s.hin[k], s.hout[k]):
                    if h is None:
                        continue
                    a_scr, h_scr = self.to_scr(s.pts[k]), self.to_scr(h)
                    p.setPen(QPen(QColor(255, 255, 255, 200), 1))
                    p.drawLine(a_scr, h_scr)
                    p.setBrush(QBrush(col))
                    p.setPen(QPen(QColor(20, 20, 20), 1))
                    p.drawEllipse(h_scr, HANDLE_R * 0.8, HANDLE_R * 0.8)
            for k, q in enumerate(s.pts):
                c = self.to_scr(q)
                chosen = self.sel == (si, k)
                p.setPen(QPen(QColor(20, 20, 20), 1.4))
                p.setBrush(QBrush(col if chosen else QColor(255, 255, 255)))
                p.drawRect(QRectF(c.x() - ANCHOR_R / 1.3, c.y() - ANCHOR_R / 1.3, ANCHOR_R * 1.54, ANCHOR_R * 1.54))
                if k == 0 and si == self.active and len(s.pts) >= 3 and not s.closed:           # the first anchor of an open path: a ring says "click to close"
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(col, 1.6))
                    p.drawEllipse(c, ANCHOR_R * 1.6, ANCHOR_R * 1.6)
        x0, y0, x1, y1 = L.bbox()
        if L.strokes and (L.char or current):
            tag = QPointF(self.to_scr((x0, y0)).x(), self.to_scr((x0, y0)).y() - 6)
            f = QFont(p.font())
            f.setPointSizeF(13 if current else 11)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QPen(QColor(0, 0, 0), 4))
            for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                p.drawText(tag + QPointF(dx, dy), L.char or "?")
            p.setPen(QPen(col, 1))
            p.drawText(tag, L.char or "?")


# ---------------------------------------------------------------------------------------------------- the dialog
class LetterEditor(QDialog):
    """Mark the letters of one photo. `letters` are in pixels of the original photo; `drawing_provider()` (optional) returns the traced picture
    (BGR) shown on request in the background thread."""
    _drawing_ready = Signal(object)

    def __init__(self, parent, photo_bgr: np.ndarray, orig_size: Tuple[int, int], letters: List[Letter], name: str = "",
                 font_name: str = "Futura", drawing_provider: Optional[Callable[[], Optional[np.ndarray]]] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Mark the letters — {name}")
        self.resize(1180, 780)
        self.font_name = font_name
        root = QHBoxLayout(self)
        self.canvas = SkeletonCanvas()
        self.canvas.set_images(_qimage(photo_bgr), orig_size)
        self.canvas.set_letters(copy.deepcopy(letters), 0)
        root.addWidget(self.canvas, 1)

        side = QVBoxLayout()
        side.addWidget(QLabel("Letters"))
        self.list = QListWidget()
        self.list.setFixedWidth(250)
        self.list.currentRowChanged.connect(self._row_changed)
        side.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.new_btn, self.del_btn = QPushButton("New letter"), QPushButton("Delete letter")
        self.new_btn.clicked.connect(self.new_letter)
        self.del_btn.clicked.connect(self.delete_letter)
        row.addWidget(self.new_btn)
        row.addWidget(self.del_btn)
        side.addLayout(row)
        side.addWidget(QLabel("Which character is it?"))
        crow = QHBoxLayout()
        self.char = QLineEdit()
        self.char.setMaxLength(1)
        self.char.setFixedWidth(60)
        f = QFont(self.char.font())
        f.setPointSize(20)
        self.char.setFont(f)
        self.char.textEdited.connect(self._char_edited)
        self.glyph = QLabel("")
        self.glyph.setFixedSize(90, 60)
        self.glyph.setAlignment(Qt.AlignCenter)
        self.glyph.setStyleSheet("background: white; color: black; border: 1px solid #888")
        self.glyph.setToolTip("The character in the font you chose")
        crow.addWidget(self.char)
        crow.addWidget(self.glyph)
        crow.addStretch(1)
        side.addLayout(crow)

        side.addWidget(QLabel("Path"))
        grid = QHBoxLayout()
        self.b_straight, self.b_curved = QPushButton("Straight"), QPushButton("Curved")
        self.b_straight.setToolTip("Make the selected segment (click a line between two anchors) straight")
        self.b_curved.setToolTip("Give the selected segment bezier handles, then drag them")
        self.b_straight.clicked.connect(lambda: self.canvas.set_segment(False))
        self.b_curved.clicked.connect(lambda: self.canvas.set_segment(True))
        grid.addWidget(self.b_straight)
        grid.addWidget(self.b_curved)
        side.addLayout(grid)
        grid2 = QHBoxLayout()
        self.b_smooth = QPushButton("Corner ↔ smooth")
        self.b_smooth.setToolTip("Selected anchor: remove its handles / add handles (also: double click the anchor)")
        self.b_smooth.clicked.connect(self.canvas.toggle_smooth_selected)
        self.b_close = QPushButton("Close / open path")
        self.b_close.clicked.connect(self.canvas.toggle_closed)
        grid2.addWidget(self.b_smooth)
        grid2.addWidget(self.b_close)
        side.addLayout(grid2)
        grid3 = QHBoxLayout()
        self.b_finish = QPushButton("Finish path (Enter)")
        self.b_finish.setToolTip("Next click starts another path of the same letter (the dot of an i, the bar of an A)")
        self.b_finish.clicked.connect(self.canvas.finish_stroke)
        self.b_delanchor = QPushButton("Delete anchor")
        self.b_delanchor.clicked.connect(self.canvas.delete_selected)
        grid3.addWidget(self.b_finish)
        grid3.addWidget(self.b_delanchor)
        side.addLayout(grid3)
        grid4 = QHBoxLayout()
        self.b_undo, self.b_redo = QPushButton("Undo"), QPushButton("Redo")
        self.b_undo.clicked.connect(self.canvas.undo)
        self.b_redo.clicked.connect(self.canvas.redo)
        grid4.addWidget(self.b_undo)
        grid4.addWidget(self.b_redo)
        side.addLayout(grid4)

        side.addWidget(QLabel("Show"))
        vrow = QHBoxLayout()
        self.r_draw, self.r_photo = QRadioButton("Drawing"), QRadioButton("Photo")
        self.r_draw.setChecked(True)
        self.r_draw.setEnabled(False)
        self.r_photo.setChecked(True)
        grp = QButtonGroup(self)
        for rb in (self.r_draw, self.r_photo):
            grp.addButton(rb)
            vrow.addWidget(rb)
        self.r_draw.toggled.connect(lambda on: (setattr(self.canvas, "show_drawing", on), self.canvas.update()))
        vrow.addStretch(1)
        side.addLayout(vrow)
        self.fit_btn = QPushButton("Fit to window")
        self.fit_btn.clicked.connect(self.canvas.fit)
        side.addWidget(self.fit_btn)

        help_ = QLabel("Click = corner anchor.  Click + drag = curve handles.\nClick the first anchor again = close.\n"
                       "Enter / right click = finish the path.\nType the character (e.g. M) = name the letter, then mark the next one.\nCtrl + click on a line = add an anchor.\nDouble click an anchor = corner ↔ smooth.\n"
                       "Wheel = zoom, Space + drag (or middle mouse) = move.")
        help_.setStyleSheet("color: gray")
        help_.setWordWrap(True)
        side.addWidget(help_)
        brow = QHBoxLayout()
        self.ok_btn, self.cancel_btn = QPushButton("Done"), QPushButton("Cancel")
        self.ok_btn.setDefault(False)
        self.ok_btn.setAutoDefault(False)
        self.cancel_btn.setAutoDefault(False)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        brow.addWidget(self.cancel_btn)
        brow.addWidget(self.ok_btn)
        side.addLayout(brow)
        root.addLayout(side)

        self.canvas.changed.connect(self._refresh_list)
        self.canvas.cur_changed.connect(self._select_row)
        self.canvas.char_typed.connect(self._char_typed)
        QShortcut(QKeySequence.Undo, self, activated=self.canvas.undo)
        QShortcut(QKeySequence.Redo, self, activated=self.canvas.redo)
        self._drawing_ready.connect(self._on_drawing)
        self._exec: Optional[ThreadPoolExecutor] = None
        if drawing_provider is not None:
            self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="letters-drawing")
            self._exec.submit(lambda: self._drawing_ready.emit(self._safe(drawing_provider)))
        self._refresh_list()
        self._set_glyph_font()

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception as exc:
            print("Letter editor: could not make the drawing:", exc)
            return None

    # ---- the list of letters
    def _label(self, L: Letter) -> str:
        n = sum(len(s.pts) for s in L.strokes)
        return f"{L.char or '?'}    {n} anchor{'s' if n != 1 else ''}" + ("" if L.char else "   (type the character)")

    def _refresh_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for L in self.canvas.letters:
            self.list.addItem(QListWidgetItem(self._label(L)))
        if 0 <= self.canvas.cur < self.list.count():
            self.list.setCurrentRow(self.canvas.cur)
        self.list.blockSignals(False)
        self._sync_char()

    def _select_row(self, row: int) -> None:
        self._refresh_list()

    def _row_changed(self, row: int) -> None:
        if row != self.canvas.cur and 0 <= row < len(self.canvas.letters):
            self.canvas.cur = row
            self.canvas.active, self.canvas.sel, self.canvas.sel_seg, self.canvas.sealed = -1, None, None, False
            self.canvas.update()
        self._sync_char()

    def _sync_char(self) -> None:
        L = self.canvas.letter()
        self.char.blockSignals(True)
        self.char.setText(L.char if L is not None else "")
        self.char.setEnabled(L is not None)
        self.char.blockSignals(False)
        self.glyph.setText(L.char if L is not None else "")
        self.del_btn.setEnabled(L is not None)

    def _char_edited(self, text: str) -> None:
        L = self.canvas.letter()
        if L is None:
            return
        self.canvas._pushed = False
        self.canvas._push()
        L.char = text[:1]
        self._refresh_list()
        self.glyph.setText(L.char)
        self.canvas.update()

    def _char_typed(self, t: str) -> None:
        self._char_edited(t)
        self.canvas.sealed = True
        self.canvas.setFocus()

    def new_letter(self) -> None:
        self.canvas._pushed = False
        self.canvas._push()
        self.canvas.letters.append(Letter("", []))
        self.canvas.cur = len(self.canvas.letters) - 1
        self.canvas.active, self.canvas.sel, self.canvas.sel_seg = -1, None, None
        self.canvas.sealed = False
        self._refresh_list()
        self.canvas.setFocus()
        self.canvas.update()

    def delete_letter(self) -> None:
        if self.canvas.letter() is None:
            return
        self.canvas._pushed = False
        self.canvas._push()
        del self.canvas.letters[self.canvas.cur]
        self.canvas.cur = min(self.canvas.cur, len(self.canvas.letters) - 1)
        self.canvas.active, self.canvas.sel, self.canvas.sel_seg = -1, None, None
        self._refresh_list()
        self.canvas.update()

    def _set_glyph_font(self) -> None:
        try:
            from .letterfit import resolve_font
            face, _ = resolve_font(self.font_name)
            f = QFont(face.family)
            f.setPixelSize(40)
            self.glyph.setFont(f)
        except Exception:
            pass

    def _on_drawing(self, bgr) -> None:
        if bgr is None:
            return
        self.canvas.set_drawing(_qimage(bgr))
        self.r_draw.setEnabled(True)
        self.r_draw.setChecked(True)

    # ---- result
    def result_letters(self) -> List[Letter]:
        """The finished letters (empty ones are left out)."""
        return [copy.deepcopy(L) for L in self.canvas.letters if L.strokes or L.char]

    def done(self, r):
        if self._exec is not None:
            self._exec.shutdown(wait=False, cancel_futures=True)
        super().done(r)


def _qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()
