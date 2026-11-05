"""Palette editor: generated colour (circle) -> replacement colour (circle, normal colour picker)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (QColorDialog, QGridLayout, QHBoxLayout, QLabel, QMenu, QPushButton,
                               QVBoxLayout, QWidget)

RGB = Tuple[int, int, int]


def _hex(c: RGB) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


class ColorDot(QPushButton):
    """Round colour swatch."""
    rightClicked = Signal()

    def __init__(self, rgb: RGB, size: int = 26, editable: bool = False):
        super().__init__()
        self.setFixedSize(size, size)
        self._size = size
        self.editable = editable
        self.set_rgb(rgb)
        self.setCursor(Qt.PointingHandCursor if editable else Qt.ArrowCursor)

    def set_rgb(self, rgb: RGB, changed: bool = False) -> None:
        self.rgb = tuple(int(v) for v in rgb)
        border = "#ffb000" if changed else "#888"
        width = 3 if changed else 1
        self.setStyleSheet(f"QPushButton{{background:{_hex(self.rgb)};border:{width}px solid {border};"
                           f"border-radius:{self._size // 2}px;}}")
        self.setToolTip(_hex(self.rgb) + ("  (click to choose a replacement)" if self.editable else ""))

    def contextMenuEvent(self, e):
        if self.editable:
            self.rightClicked.emit()


class PaletteEditor(QWidget):
    """Shows each generated palette colour next to the colour it should be replaced with."""
    changed = Signal()

    COLUMNS = 3

    def __init__(self):
        super().__init__()
        self._signature = None
        self._items: List[Tuple[int, RGB]] = []
        self._over: Dict[int, RGB] = {}
        self._dots: Dict[int, ColorDot] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        self.title = QLabel("Replace colours")
        self.title.setStyleSheet("font-weight:600")
        self.reset_btn = QPushButton("Reset all")
        self.reset_btn.clicked.connect(self.reset)
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.reset_btn)
        v.addLayout(head)
        self.hint = QLabel("Left: colour found in the image.  Right: click to choose what it becomes.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#777;font-size:10px")
        v.addWidget(self.hint)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        v.addWidget(self.grid_host)
        self.setVisible(False)

    # --- API --------------------------------------------------------------------------
    def set_palette(self, palette: Sequence[RGB], used: Sequence[int], signature) -> None:
        items = [(i, tuple(palette[i])) for i in used]
        if signature != self._signature:
            self._over.clear()
            self._signature = signature
        else:                       # same structure: keep overrides, drop ones that no longer exist
            self._over = {i: c for i, c in self._over.items() if i < len(palette)}
        if items != self._items:
            self._items = items
            self._rebuild()
        self.setVisible(bool(items))

    def overrides(self) -> Dict[int, RGB]:
        return {i: c for i, c in self._over.items() if any(i == j and c != rgb for j, rgb in self._items)}

    def reset(self) -> None:
        if self._over:
            self._over.clear()
            for i, d in self._dots.items():
                d.set_rgb(dict(self._items)[i])
            self.changed.emit()

    # --- internals --------------------------------------------------------------------
    def _rebuild(self) -> None:
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._dots.clear()
        for n, (idx, rgb) in enumerate(self._items):
            cell = QWidget()
            h = QHBoxLayout(cell)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            src = ColorDot(rgb)
            dst = ColorDot(self._over.get(idx, rgb), editable=True, size=26)
            dst.set_rgb(self._over.get(idx, rgb), changed=idx in self._over)
            dst.clicked.connect(lambda _=False, i=idx: self._pick(i))
            dst.rightClicked.connect(lambda i=idx: self._reset_one(i))
            arrow = QLabel("→")
            h.addWidget(src)
            h.addWidget(arrow)
            h.addWidget(dst)
            self._dots[idx] = dst
            self.grid.addWidget(cell, n // self.COLUMNS, n % self.COLUMNS)

    def _pick(self, idx: int) -> None:
        cur = self._over.get(idx, dict(self._items)[idx])
        dlg = QColorDialog(QColor(*cur), self)
        dlg.setWindowTitle("Replacement colour")
        dlg.setOption(QColorDialog.DontUseNativeDialog, True)      # same dialog on every platform
        if dlg.exec():
            c = dlg.currentColor()
            self.set_override(idx, (c.red(), c.green(), c.blue()))

    def set_override(self, idx: int, rgb: RGB) -> None:
        base = dict(self._items).get(idx)
        if base is None:
            return
        if tuple(rgb) == base:
            self._over.pop(idx, None)
        else:
            self._over[idx] = tuple(rgb)
        self._dots[idx].set_rgb(self._over.get(idx, base), changed=idx in self._over)
        self.changed.emit()

    def _reset_one(self, idx: int) -> None:
        if idx in self._over:
            self.set_override(idx, dict(self._items)[idx])
