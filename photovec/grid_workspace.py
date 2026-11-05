"""Grid workspace: the layout of the grid as plain squares – no pictures, no tracing – so squares can be selected,
painted (morphs / stays / back to white) and inspected quickly, also for grids of hundreds of pictures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

STATE_COLORS = {"morph": QColor(46, 160, 90), "stay": QColor(64, 124, 214), "white": QColor(208, 70, 70)}
STATE_SHORT = {"morph": "morphs", "stay": "stays", "white": "→ white"}


@dataclass
class CellInfo:
    label: str = ""          # file name
    state: str = "morph"
    size: Tuple[int, int] = (0, 0)
    tag: str = ""            # short note, e.g. the vectorisation preset
    custom: bool = False     # the square overrides settings of its row
    tip: str = ""


class GridWorkspace(QWidget):
    """Squares drawn to scale. tool: "select" selects; "morph" / "stay" / "white" paint that state onto the squares you touch."""
    selection_changed = Signal()
    painted = Signal(list, str)            # flat indices, state
    activated = Signal(int)                # double click on a square

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 380)
        self.setMouseTracking(True)
        self.layout_ = None
        self.cells: List[CellInfo] = []
        self.selected: Set[int] = set()
        self.tool = "select"
        self.area: Optional[Tuple[int, int, int, int]] = None      # x, y, w, h: where the linework will be drawn (canvas pixels)
        self.row_labels: List[str] = []
        self._band: Optional[Tuple[QPointF, QPointF]] = None
        self._band_add = False
        self._band_base: Set[int] = set()
        self._paint: Optional[Tuple[str, List[int]]] = None
        self._anchor: Optional[int] = None

    # -- data
    def set_grid(self, layout, cells: Sequence[CellInfo], row_labels: Sequence[str] = (),
                 area: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.layout_ = layout
        self.cells = list(cells)
        self.row_labels = list(row_labels)
        self.area = area
        self.selected = {k for k in self.selected if k < len(self.cells)}
        self.update()

    def set_selection(self, sel: Set[int]) -> None:
        self.selected = {k for k in sel if k < len(self.cells)}
        self.update()

    # -- geometry
    def _view(self):
        CW, CH = self.layout_.canvas
        m = 34                                     # room for row labels on the left
        k = min((self.width() - m - 10) / max(1, CW), (self.height() - 20) / max(1, CH))
        w, h = CW * k, CH * k
        return k, m + (self.width() - m - 10 - w) / 2, 10 + (self.height() - 20 - h) / 2

    def _rect(self, i: int) -> QRectF:
        k, ox, oy = self._view()
        x, y, w, h = self.layout_.box(i)
        return QRectF(ox + x * k, oy + y * k, w * k, h * k)

    def cell_at(self, pt: QPointF) -> Optional[int]:
        if self.layout_ is None:
            return None
        for i in range(len(self.cells)):
            if self._rect(i).contains(pt):
                return i
        return None

    # -- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self.layout_ is None or not self.cells:
            p.setPen(QColor(170, 170, 175))
            p.drawText(self.rect(), Qt.AlignCenter, "Add pictures on the Source tab:\nevery picture becomes a square here.")
            return
        k, ox, oy = self._view()
        CW, CH = self.layout_.canvas
        p.setPen(QPen(QColor(70, 70, 76), 1, Qt.DashLine))
        p.drawRect(QRectF(ox, oy, CW * k, CH * k))
        if self.area is not None:                              # where the drawing will appear
            ax, ay, aw, ah = self.area
            p.setPen(QPen(QColor(230, 190, 60), 2, Qt.DashLine))
            p.drawRect(QRectF(ox + ax * k, oy + ay * k, aw * k, ah * k))
        # row labels
        p.setPen(QColor(150, 150, 156))
        f = QFont(self.font())
        f.setPointSizeF(8)
        p.setFont(f)
        seen = set()
        for i in range(len(self.cells)):
            r = self.layout_.row_of(i)
            if r in seen:
                continue
            seen.add(r)
            rc = self._rect(i)
            p.drawText(QRectF(2, rc.top(), 30, rc.height()), Qt.AlignVCenter | Qt.AlignLeft,
                       self.row_labels[r] if r < len(self.row_labels) else f"R{r + 1}")
        for i, c in enumerate(self.cells):
            self._cell(p, i, c)
        if self._band is not None:
            a, b = self._band
            r = QRectF(a, b).normalized()
            p.fillRect(r, QColor(255, 255, 255, 30))
            p.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine))
            p.drawRect(r)

    def _cell(self, p: QPainter, i: int, c: CellInfo) -> None:
        r = self._rect(i)
        col = STATE_COLORS.get(c.state, STATE_COLORS["morph"])
        fill = QColor(col)
        fill.setAlpha(150 if c.state != "white" else 40)
        p.fillRect(r, fill)
        sel = i in self.selected
        p.setPen(QPen(QColor(255, 255, 255) if sel else col, 3 if sel else 1.5))
        p.drawRect(r.adjusted(1, 1, -1, -1))
        if c.state == "white":                                 # crossed out: it is white in the end
            p.setPen(QPen(QColor(208, 70, 70, 130), 1.5))
            p.drawLine(r.topLeft() + QPointF(3, 3), r.bottomRight() - QPointF(3, 3))
            p.drawLine(r.topRight() + QPointF(-3, 3), r.bottomLeft() + QPointF(3, -3))
        # text, scaled to what fits
        s = max(7.0, min(20.0, r.height() / 6.0, r.width() / 5.0))
        if r.width() < 26 or r.height() < 18:
            return
        f = QFont(self.font())
        f.setPointSizeF(s)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        p.drawText(r.adjusted(4, 2, -4, -2), Qt.AlignTop | Qt.AlignLeft, f"#{i + 1}" + (" ★" if c.custom else ""))
        f.setBold(False)
        f.setPointSizeF(max(6.5, s * 0.62))
        p.setFont(f)
        fm = QFontMetrics(f)
        lines = [STATE_SHORT.get(c.state, "")]
        if c.size[0]:
            lines.append(f"{c.size[0]}×{c.size[1]}")
        if c.tag:
            lines.append(c.tag)
        if c.label:
            lines.append(c.label)
        y = r.top() + s * 2.1
        for t in lines:
            if y + fm.height() > r.bottom() - 2:
                break
            p.setPen(QColor(235, 235, 240))
            p.drawText(QRectF(r.left() + 4, y, r.width() - 8, fm.height()), Qt.AlignLeft | Qt.AlignVCenter,
                       fm.elidedText(t, Qt.ElideMiddle, int(r.width() - 8)))
            y += fm.height()

    # -- mouse
    def mousePressEvent(self, e):
        if self.layout_ is None or e.button() != Qt.LeftButton:
            return
        pt = e.position()
        i = self.cell_at(pt)
        ctrl = bool(e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))
        shift = bool(e.modifiers() & Qt.ShiftModifier)
        if self.tool != "select":
            self._paint = (self.tool, [])
            if i is not None:
                self._paint_cell(i)
            return
        if i is None:
            self._band, self._band_add = (pt, pt), ctrl or shift
            self._band_base = set(self.selected) if self._band_add else set()
            if not self._band_add:
                self.selected = set()
                self.selection_changed.emit()
            self.update()
            return
        if shift and self._anchor is not None:
            a, b = sorted((self._anchor, i))
            self.selected |= set(range(a, b + 1))
        elif ctrl:
            self.selected ^= {i}
            self._anchor = i
        else:
            self.selected = {i}
            self._anchor = i
        self.selection_changed.emit()
        self.update()

    def _paint_cell(self, i: int) -> None:
        state, done = self._paint
        if i not in done:
            done.append(i)
            self.cells[i].state = state                    # instant feedback; the owner stores it on release
            self.update()

    def mouseMoveEvent(self, e):
        pt = e.position()
        if self._paint is not None:
            i = self.cell_at(pt)
            if i is not None:
                self._paint_cell(i)
            return
        if self._band is not None:
            self._band = (self._band[0], pt)
            r = QRectF(self._band[0], self._band[1]).normalized()
            self.selected = self._band_base | {i for i in range(len(self.cells)) if self._rect(i).intersects(r)}
            self.update()
            return

    def mouseReleaseEvent(self, e):
        if self._paint is not None:
            state, done = self._paint
            self._paint = None
            if done:
                self.painted.emit(done, state)
            return
        if self._band is not None:
            self._band = None
            self.selection_changed.emit()
            self.update()

    def mouseDoubleClickEvent(self, e):
        i = self.cell_at(e.position())
        if i is not None and self.tool == "select":
            self.activated.emit(i)

    def event(self, e):
        if e.type() == e.Type.ToolTip and self.layout_ is not None:
            i = self.cell_at(QPointF(e.pos()))
            if i is not None and self.cells[i].tip:
                QToolTip.showText(e.globalPos(), self.cells[i].tip, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(e)
