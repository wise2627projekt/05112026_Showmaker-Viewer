"""A slider with several handles (the start, the slider points in between, and the end) and an optional marker for the current value.

The handles are numbered in time order (1 = start, last = end) and each one moves freely, so a slider can run up, down or both.
`pressed(k)` / `released()` tell the window which handle the mouse holds (the panel then shows that point)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

START, END, MIDDLE = QColor(46, 160, 90), QColor(214, 120, 40), QColor(90, 120, 225)


class RangeSlider(QWidget):
    changed = Signal()
    pressed = Signal(int)
    released = Signal()

    def __init__(self, minimum: int = 0, maximum: int = 100, *values: int):
        super().__init__()
        self._min, self._max = minimum, maximum
        self._vals: List[int] = self._fit(values if values else (minimum, maximum))
        self._drag: Optional[int] = None
        self._marker: Optional[float] = None
        self.setMinimumSize(120, 30)
        self.setFocusPolicy(Qt.NoFocus)

    def _fit(self, vals: Sequence[float]) -> List[int]:
        v = [min(max(int(round(x)), self._min), self._max) for x in vals]
        if not v:
            return [self._min, self._max]
        return v if len(v) >= 2 else v * 2

    # -- values
    def values(self) -> Tuple[int, ...]:
        return tuple(self._vals)

    def count(self) -> int:
        return len(self._vals)

    def set_values(self, *vals: int, emit: bool = False) -> None:
        if len(vals) == 1 and not isinstance(vals[0], (int, float)):
            vals = tuple(vals[0])
        new = self._fit(vals)
        if new != self._vals:
            self._vals = new
            self.update()
            if emit:
                self.changed.emit()

    def set_count(self, n: int) -> None:
        """More or fewer handles; the others stay where they are as far as possible (the line through them is resampled)."""
        n = max(2, int(n))
        old = self._vals
        if n == len(old):
            return
        out = []
        for i in range(n):
            x = i / (n - 1) * (len(old) - 1)
            j = min(int(x), len(old) - 2)
            out.append(old[j] + (old[j + 1] - old[j]) * (x - j))
        self._vals = self._fit(out)
        self.update()

    def set_marker(self, v: Optional[float]) -> None:
        self._marker = v
        self.update()

    # -- geometry
    def _x(self, v: float) -> float:
        w = max(1, self.width() - 20)
        return 10 + w * (v - self._min) / max(1, self._max - self._min)

    def _val(self, x: float) -> int:
        w = max(1, self.width() - 20)
        return int(round(self._min + (x - 10) / w * (self._max - self._min)))

    # -- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2
        p.setPen(QPen(QColor(120, 120, 128), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(self._x(self._min), cy), QPointF(self._x(self._max), cy))
        p.setPen(QPen(QColor(70, 140, 230), 4, Qt.SolidLine, Qt.RoundCap))
        for a, b in zip(self._vals, self._vals[1:]):                       # the stretches between consecutive points
            p.drawLine(QPointF(self._x(a), cy), QPointF(self._x(b), cy))
        if self._marker is not None:
            p.setPen(QPen(QColor(235, 235, 240), 2))
            x = self._x(min(max(self._marker, self._min), self._max))
            p.drawLine(QPointF(x, cy - 11), QPointF(x, cy + 11))
        f = QFont(p.font())
        f.setPixelSize(9)
        f.setBold(True)
        p.setFont(f)
        n = len(self._vals)
        for i, v in enumerate(self._vals):
            col = START if i == 0 else END if i == n - 1 else MIDDLE
            p.setPen(QPen(QColor(25, 25, 30), 1.5))
            p.setBrush(col)
            c = QPointF(self._x(v), cy)
            p.drawEllipse(c, 8, 8)
            if n > 2:
                p.setPen(QColor(255, 255, 255))
                p.drawText(int(c.x()) - 8, int(cy) - 8, 16, 16, Qt.AlignCenter, str(i + 1))

    # -- mouse
    def _nearest(self, x: float) -> int:
        d = [abs(x - self._x(v)) for v in self._vals]
        best = min(d)
        ties = [i for i, di in enumerate(d) if di == best]
        if len(ties) == 1:
            return ties[0]
        return ties[0] if x < self._x(self._vals[ties[0]]) else ties[-1]      # on stacked handles: pull the one on the side of the mouse

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = self._nearest(e.position().x())
            self.pressed.emit(self._drag)
            self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        v = min(max(self._val(e.position().x()), self._min), self._max)
        if v != self._vals[self._drag]:
            self._vals[self._drag] = v
            self.update()
            self.changed.emit()

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.released.emit()
