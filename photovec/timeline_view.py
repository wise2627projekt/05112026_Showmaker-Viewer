"""The timeline widget: lanes with keyframes, a ruler and a playhead, like a video editor.

    * click / drag in the ruler: move the playhead       * click an empty place in a lane: playhead there + lane selected
    * double-click in a lane: a keyframe there           * drag a keyframe: change its time (snaps to frames) and value
    * right-click a keyframe: easing, value, delete      * Delete: remove the selected keyframe (a lane keeps at least one)
    * Ctrl + wheel: zoom (around the mouse)              * wheel / middle-drag: scroll in time
    * Left / Right: one frame back / forward (Shift: ten)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QInputDialog, QMenu, QWidget

from .timeline import EASE_LABEL, EASES, LANE_DEFS, LAYER_LANES, ORDER, Key, Lane, Timeline

BG = QColor("#26282c")
LANE_BG = QColor("#30333a")
LANE_ALT = QColor("#2c2f35")
GRID = QColor("#454a53")
TEXT = QColor("#d8dce2")
DIM = QColor("#7c828c")
HEAD = QColor("#ff3b30")


def nice_step(span: float) -> float:
    """A round tick distance (seconds) that gives roughly 6-12 ticks over `span` seconds."""
    for s in (0.04, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
        if span / s <= 12:
            return s
    return 600.0


class TimelineView(QWidget):
    changed = Signal()            # a keyframe or a lane switch was edited
    editing = Signal()            # about to edit (the window saves an undo step)
    seek = Signal(int)            # the playhead was moved to this frame
    selection = Signal()          # another keyframe / lane was selected
    viewed = Signal()             # zoom / scroll changed

    LW, RH, LH = 176, 28, 68
    PAD = 12                                   # space between the label column and time 0, so the first keyframe is fully visible

    def __init__(self, tl: Timeline, parent=None):
        super().__init__(parent)
        self.tl = tl
        self.frame = 0
        self.t0 = 0.0
        self.pps = 60.0
        self.layers_on = False
        self.lane_sel: Optional[str] = "threshold"
        self.key_sel: Optional[Key] = None
        self._drag: Optional[tuple] = None
        self._seeking = False
        self._pan: Optional[tuple] = None
        self._moved = False
        self._auto_fit = True                       # follows the window width until you zoom or scroll yourself
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.sync_height()

    # ------------------------------------------------------------------ geometry
    def lane_ids(self) -> List[str]:
        return [i for i in ORDER if self.layers_on or i not in LAYER_LANES]

    def sync_height(self) -> None:
        self.setMinimumHeight(self.RH + self.LH * len(self.lane_ids()) + 6)
        self.update()

    def x_of(self, t: float) -> float:
        return self.LW + self.PAD + (t - self.t0) * self.pps

    def t_of(self, x: float) -> float:
        return self.t0 + (x - self.LW - self.PAD) / self.pps

    def lane_rect(self, idx: int) -> QRectF:
        return QRectF(self.LW, self.RH + idx * self.LH, max(10, self.width() - self.LW), self.LH)

    def y_of(self, lane: Lane, v: float, r: QRectF) -> float:
        pad = 9.0
        return r.bottom() - pad - (v - lane.lo) / max(1e-9, lane.hi - lane.lo) * (r.height() - 2 * pad)

    def v_of(self, lane: Lane, y: float, r: QRectF) -> float:
        pad = 9.0
        return lane.lo + (r.bottom() - pad - y) / max(1e-9, r.height() - 2 * pad) * (lane.hi - lane.lo)

    def visible_span(self) -> float:
        return max(1e-6, (self.width() - self.LW - self.PAD) / self.pps)

    def fit(self) -> None:
        self.pps = max(4.0, (self.width() - self.LW - self.PAD - 24) / max(0.2, self.tl.duration))
        self.t0 = 0.0
        self._auto_fit = True
        self.viewed.emit()
        self.update()

    def set_frame(self, frame: int) -> None:
        self.frame = int(min(max(frame, 0), self.tl.frames() - 1))
        t = self.tl.time_of(self.frame)
        if t < self.t0 or t > self.t0 + self.visible_span():       # keep the playhead on screen
            self.t0 = max(0.0, t - self.visible_span() * 0.1)
            self.viewed.emit()
        self.update()

    def set_t0(self, t0: float) -> None:
        self.t0 = max(0.0, min(t0, max(0.0, self.tl.duration - self.visible_span() * 0.2)))
        self.update()

    def lane_at(self, y: float) -> int:
        i = int((y - self.RH) // self.LH)
        return i if 0 <= i < len(self.lane_ids()) else -1

    def hit_key(self, pos) -> Optional[Tuple[Lane, Key]]:
        ids = self.lane_ids()
        i = self.lane_at(pos.y())
        if i < 0 or pos.x() < self.LW:
            return None
        lane = self.tl.lanes[ids[i]]
        r = self.lane_rect(i)
        best, bd = None, 11.0
        for k in lane.keys:
            d = ((self.x_of(k.t) - pos.x()) ** 2 + (self.y_of(lane, k.v, r) - pos.y()) ** 2) ** 0.5
            if d < bd:
                best, bd = k, d
        return (lane, best) if best is not None else None

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        font = QFont(self.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 0.5))
        p.setFont(font)
        ids = self.lane_ids()
        w = self.width()
        # lanes
        for i, lid in enumerate(ids):
            lane = self.tl.lanes[lid]
            r = self.lane_rect(i)
            full = QRectF(0, r.top(), w, r.height())
            p.fillRect(full, LANE_BG if i % 2 == 0 else LANE_ALT)
            if lid == self.lane_sel:
                p.fillRect(QRectF(0, r.top(), self.LW, r.height()), QColor(255, 255, 255, 18))
            col = QColor(lane.color)
            p.fillRect(QRectF(0, r.top(), 4, r.height()), col)
            p.setPen(TEXT if lane.enabled else DIM)
            p.drawText(QRectF(14, r.top() + 6, self.LW - 20, 18), Qt.AlignLeft | Qt.AlignVCenter, lane.label)
            v = lane.value(self.tl.time_of(self.frame))
            p.setPen(col if lane.enabled else DIM)
            p.drawText(QRectF(14, r.top() + 24, self.LW - 20, 18), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{v:.0f}{lane.unit}" if lane.discrete or lane.unit else f"{v:.1f}")
            if lid == "paths":
                box = QRectF(14, r.top() + 44, 14, 14)
                p.setPen(QPen(TEXT, 1))
                p.setBrush(col if lane.enabled else Qt.NoBrush)
                p.drawRect(box)
                p.setPen(DIM)
                p.drawText(QRectF(34, r.top() + 42, self.LW - 40, 18), Qt.AlignLeft | Qt.AlignVCenter, "animate")
            p.setPen(QPen(GRID, 1))
            p.drawLine(QPointF(0, r.bottom()), QPointF(w, r.bottom()))
            p.drawLine(QPointF(self.LW, r.top()), QPointF(self.LW, r.bottom()))
            # guide lines (min / middle / max)
            p.setPen(QPen(GRID, 1, Qt.DotLine))
            for frac in (0.0, 0.5, 1.0):
                y = self.y_of(lane, lane.lo + frac * (lane.hi - lane.lo), r)
                p.drawLine(QPointF(self.LW, y), QPointF(w, y))
            # the curve
            p.save()
            p.setClipRect(QRectF(self.LW, r.top(), r.width(), r.height()))
            path = QPainterPath()
            x = float(self.LW)
            first = True
            while x <= w + 2:
                y = self.y_of(lane, lane.value(self.t_of(x)), r)
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
                x += 2.0
            fill = QPainterPath(path)
            fill.lineTo(w + 2, r.bottom())
            fill.lineTo(self.LW, r.bottom())
            fill.closeSubpath()
            fc = QColor(col)
            fc.setAlpha(40 if lane.enabled else 14)
            p.fillPath(fill, fc)
            lc = QColor(col) if lane.enabled else QColor(DIM)
            p.setPen(QPen(lc, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            # the end of the clip
            xe = self.x_of(self.tl.duration)
            p.fillRect(QRectF(xe, r.top(), max(0.0, w - xe), r.height()), QColor(0, 0, 0, 90))
            # keyframes
            for k in lane.keys:
                kx, ky = self.x_of(k.t), self.y_of(lane, k.v, r)
                sel = k is self.key_sel
                s = 7.0 if sel else 5.5
                d = QPolygonF([QPointF(kx, ky - s), QPointF(kx + s, ky), QPointF(kx, ky + s), QPointF(kx - s, ky)])
                p.setPen(QPen(QColor("#ffffff") if sel else QColor("#10121a"), 1.4))
                p.setBrush(QColor("#ffffff") if sel else lc)
                p.drawPolygon(d)
                if k.ease == "hold":
                    p.setPen(QPen(lc, 1))
                    p.drawLine(QPointF(kx + s + 1, ky), QPointF(kx + s + 6, ky))
                if sel:
                    p.setPen(TEXT)
                    p.drawText(QPointF(kx + 10, ky - 9), f"{k.v:.1f}{lane.unit}  @ {k.t:.2f}s")
            p.restore()
        # ruler
        p.fillRect(QRectF(0, 0, w, self.RH), QColor("#1d1f23"))
        p.setPen(QPen(GRID, 1))
        p.drawLine(QPointF(0, self.RH), QPointF(w, self.RH))
        step = nice_step(self.visible_span())
        t = (int(self.t0 / step)) * step
        while t <= self.t0 + self.visible_span() + step:
            x = self.x_of(t)
            if x >= self.LW:
                p.setPen(QPen(DIM, 1))
                p.drawLine(QPointF(x, self.RH - 9), QPointF(x, self.RH))
                p.setPen(TEXT)
                p.drawText(QPointF(x + 3, self.RH - 11), f"{t:g}s" if step >= 1 else f"{t:.2f}s")
            t += step
        if self.pps >= 16 * self.tl.fps / 2:                       # zoomed in far enough: a tick for every frame
            f = max(0, int(self.t0 * self.tl.fps))
            while self.x_of(f / self.tl.fps) < w and f <= self.tl.frames():
                x = self.x_of(f / self.tl.fps)
                if x >= self.LW:
                    p.setPen(QPen(GRID, 1))
                    p.drawLine(QPointF(x, self.RH - 4), QPointF(x, self.RH))
                f += 1
        p.fillRect(QRectF(0, 0, self.LW, self.RH), QColor("#1d1f23"))
        p.setPen(DIM)
        p.drawText(QRectF(14, 0, self.LW - 20, self.RH), Qt.AlignLeft | Qt.AlignVCenter,
                   f"{self.tl.time_of(self.frame):.2f} s · frame {self.frame + 1}")
        # playhead
        hx = self.x_of(self.tl.time_of(self.frame))
        if hx >= self.LW - 1:
            p.setPen(QPen(HEAD, 1.5))
            p.drawLine(QPointF(hx, 0), QPointF(hx, self.height()))
            p.setBrush(HEAD)
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF([QPointF(hx - 7, 0), QPointF(hx + 7, 0), QPointF(hx, 12)]))
        p.end()

    # ------------------------------------------------------------------ mouse
    def _snap_t(self, t: float) -> float:
        f = round(t * self.tl.fps) / float(self.tl.fps)
        return min(max(f, 0.0), self.tl.duration)

    def _seek_to(self, x: float) -> None:
        f = int(round(max(0.0, self.t_of(x)) * self.tl.fps))
        f = min(max(f, 0), self.tl.frames() - 1)
        if f != self.frame:
            self.frame = f
            self.seek.emit(f)
            self.update()

    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        if e.button() == Qt.MiddleButton:
            self._pan = (pos.x(), self.t0)
            return
        if e.button() == Qt.RightButton:
            self._context(e)
            return
        if e.button() != Qt.LeftButton:
            return
        if pos.y() < self.RH and pos.x() >= self.LW:
            self._seeking = True
            self._seek_to(pos.x())
            return
        ids = self.lane_ids()
        i = self.lane_at(pos.y())
        if i < 0:
            return
        lane = self.tl.lanes[ids[i]]
        if pos.x() < self.LW:                                           # the label column
            self.lane_sel = lane.id
            if lane.id == "paths" and QRectF(10, self.RH + i * self.LH + 40, 80, 22).contains(pos):
                self.editing.emit()
                lane.enabled = not lane.enabled
                self.changed.emit()
            self.selection.emit()
            self.update()
            return
        hit = self.hit_key(pos)
        self.lane_sel = lane.id
        if hit is not None:
            self.key_sel = hit[1]
            self._drag = (hit[0], hit[1], hit[1].t, hit[1].v, False)
        else:
            self.key_sel = None
            self._seeking = True
            self._seek_to(pos.x())
        self.selection.emit()
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._pan is not None:
            self._auto_fit = False
            self.set_t0(self._pan[1] - (pos.x() - self._pan[0]) / self.pps)
            self.viewed.emit()
            return
        if self._seeking:
            self._seek_to(pos.x())
            return
        if self._drag is not None:
            lane, key, t0, v0, started = self._drag
            if not started:
                self.editing.emit()
                self._drag = (lane, key, t0, v0, True)
            ids = self.lane_ids()
            r = self.lane_rect(ids.index(lane.id))
            if not (e.modifiers() & Qt.ShiftModifier):
                key.t = self._snap_t(self.t_of(pos.x()))
            else:
                key.t = t0
            if not (e.modifiers() & Qt.ControlModifier):
                key.v = lane.clamp(round(self.v_of(lane, pos.y(), r), 1))
            else:
                key.v = v0
            lane.sort()
            self.changed.emit()
            self.update()
            return
        self.setCursor(Qt.PointingHandCursor if self.hit_key(pos) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._seeking = False
        self._pan = None
        if self._drag is not None:
            self._drag = None
            self.changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, e):
        pos = e.position()
        if e.button() != Qt.LeftButton or pos.x() < self.LW or pos.y() < self.RH:
            return
        i = self.lane_at(pos.y())
        if i < 0 or self.hit_key(pos) is not None:
            return
        lane = self.tl.lanes[self.lane_ids()[i]]
        self.editing.emit()
        self.key_sel = lane.add(self._snap_t(self.t_of(pos.x())), round(self.v_of(lane, pos.y(), self.lane_rect(i)), 1))
        self.lane_sel = lane.id
        self._seeking = False
        self.selection.emit()
        self.changed.emit()
        self.update()

    def _context(self, e) -> None:
        pos = e.position()
        hit = self.hit_key(pos)
        ids = self.lane_ids()
        i = self.lane_at(pos.y())
        menu = QMenu(self)
        if hit is not None:
            lane, key = hit
            self.key_sel, self.lane_sel = key, lane.id
            self.selection.emit()
            self.update()
            for kind in EASES:
                a = menu.addAction(("● " if key.ease == kind else "    ") + EASE_LABEL[kind])
                a.triggered.connect(lambda _=False, k=kind: self.set_ease(k))
            menu.addSeparator()
            menu.addAction("Set value…").triggered.connect(lambda: self._ask_value(lane, key))
            menu.addAction("Delete keyframe").triggered.connect(self.delete_selected)
        elif i >= 0 and pos.x() >= self.LW:
            lane = self.tl.lanes[ids[i]]
            t = self._snap_t(self.t_of(pos.x()))
            menu.addAction("Add keyframe here (keeps the curve)").triggered.connect(lambda: self._add(lane, t, None))
            menu.addAction("Add keyframe here at this height").triggered.connect(
                lambda: self._add(lane, t, round(self.v_of(lane, pos.y(), self.lane_rect(i)), 1)))
        else:
            return
        menu.exec(e.globalPosition().toPoint())

    def _add(self, lane: Lane, t: float, v: Optional[float]) -> None:
        self.editing.emit()
        self.key_sel = lane.add(t, v)
        self.lane_sel = lane.id
        self.selection.emit()
        self.changed.emit()
        self.update()

    def _ask_value(self, lane: Lane, key: Key) -> None:
        v, ok = QInputDialog.getDouble(self, "Value", f"{lane.label}{lane.unit}:", key.v, lane.lo, lane.hi, 1)
        if ok:
            self.editing.emit()
            key.v = lane.clamp(v)
            self.changed.emit()
            self.update()

    def set_ease(self, kind: str) -> None:
        if self.key_sel is not None and self.key_sel.ease != kind:
            self.editing.emit()
            self.key_sel.ease = kind
            self.changed.emit()
            self.update()

    def delete_selected(self) -> bool:
        if self.key_sel is None or self.lane_sel is None:
            return False
        lane = self.tl.lanes[self.lane_sel]
        if len(lane.keys) <= 1 or self.key_sel not in lane.keys:
            return False
        self.editing.emit()
        lane.remove(self.key_sel)
        self.key_sel = None
        self.selection.emit()
        self.changed.emit()
        self.update()
        return True

    def wheelEvent(self, e):
        dy = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.ControlModifier:
            x = e.position().x()
            t = self.t_of(max(x, self.LW))
            self._auto_fit = False
            self.pps = min(2000.0, max(4.0, self.pps * (1.15 if dy > 0 else 1 / 1.15)))
            self.t0 = max(0.0, t - (max(x, self.LW) - self.LW) / self.pps)
        else:
            self._auto_fit = False
            self.set_t0(self.t0 - dy / 120.0 * self.visible_span() * 0.12)
        self.viewed.emit()
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif e.key() in (Qt.Key_Left, Qt.Key_Right):
            n = (10 if e.modifiers() & Qt.ShiftModifier else 1) * (1 if e.key() == Qt.Key_Right else -1)
            self.set_frame(self.frame + n)
            self.seek.emit(self.frame)
        elif e.key() == Qt.Key_Home:
            self.set_frame(0)
            self.seek.emit(0)
        elif e.key() == Qt.Key_End:
            self.set_frame(self.tl.frames() - 1)
            self.seek.emit(self.frame)
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._auto_fit:
            self.fit()
        self.viewed.emit()
