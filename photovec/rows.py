"""Rows and squares of a picture grid.

A row holds pictures and the settings shared by its squares (tile size, vectorisation, slider direction / delay, linework).
Every square (CellSpec) can override those settings and says whether it morphs into the linework, stays a vector or goes
back to white."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .linework import PRESETS as LW_PRESETS, LineworkParams
from .sweep import Track
from .trace import PRESETS as TRACE_PRESETS, TraceParams, params_from_preset

SIDES = ("auto", "right", "left", "bottom", "top")      # where image B enters from ("auto" = the stage's setting)
STATES = ("morph", "stay", "white")                      # what a square does in the story
STATE_LABEL = {"morph": "morphs into the linework", "stay": "stays a vector", "white": "goes back to white"}
SIZE_MODES = ("fixed", "fit_long", "fit_h", "fit_w")
SIZE_LABEL = {"fixed": "Fixed width × height", "fit_long": "Fit the picture – longest side",
              "fit_h": "Fit the picture – height", "fit_w": "Fit the picture – width"}
MIN_TILE, MAX_TILE = 16, 4000


@dataclass
class CellSpec:
    """One square. Empty / None = use the row's setting."""
    state: str = "morph"
    size_mode: str = ""
    w: int = 0
    h: int = 0
    fit: str = ""
    engine: str = ""
    preset: str = ""
    colors: Optional[int] = None
    detail: Optional[int] = None
    noise: Optional[int] = None
    side: str = ""
    delay: Optional[int] = None
    linework: str = ""
    sweep: List[Track] = field(default_factory=list)

    def copy(self) -> "CellSpec":
        return replace(self, sweep=[replace(t) for t in self.sweep])

    def customised(self) -> bool:
        """True when this square overrides anything of its row (the state does not count)."""
        d = CellSpec()
        return any(getattr(self, f) != getattr(d, f) for f in
                   ("size_mode", "w", "h", "fit", "engine", "preset", "colors", "detail", "noise", "side", "delay", "linework")) \
            or bool(self.sweep)


def fit_size(mode: str, w: int, h: int, aspect: float) -> Tuple[int, int]:
    """Tile size that follows the picture's shape. `aspect` = picture width / height; w, h are the row's numbers."""
    aspect = max(0.05, min(20.0, aspect or 1.0))
    if mode == "fit_long":
        n = max(w, h)
        tw, th = (n, n / aspect) if aspect >= 1 else (n * aspect, n)
    elif mode == "fit_h":
        tw, th = h * aspect, h
    elif mode == "fit_w":
        tw, th = w, w / aspect
    else:
        tw, th = w, h
    c = lambda v: int(min(MAX_TILE, max(MIN_TILE, round(v))))
    return c(tw), c(th)


_SIZES: Dict[tuple, Tuple[int, int]] = {}


def image_size(path: str) -> Tuple[int, int]:
    """(width, height) of a picture as it will be shown (EXIF orientation applied), read from the file header only."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return (1, 1)
    hit = _SIZES.get(key)
    if hit is None:
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
                try:
                    if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                        w, h = h, w
                except Exception:
                    pass
            hit = (max(1, w), max(1, h))
        except Exception:
            hit = (1, 1)
        if len(_SIZES) > 4000:
            _SIZES.clear()
        _SIZES[key] = hit
    return hit


@dataclass
class RowSpec:
    paths: List[str] = field(default_factory=list)
    tile_w: int = 480
    tile_h: int = 480
    fit: str = "cover"            # cover (centre crop) | contain (whole picture, padded)
    # --- vectorisation of this row ---------------------------------------------------------
    engine: str = "main"          # main = settings of the main window | trace = Image Trace | linework = Linework Snap
    preset: str = ""              # preset name of that engine ("" = keep what the main window has)
    colors: int = 0               # Image Trace: 0 = keep, else max colours
    detail: int = -1              # Image Trace "Paths" / Linework "Simplify" (-1 = keep)
    noise: int = -1               # Image Trace "Noise" (-1 = keep)
    # --- animation of this row -------------------------------------------------------------
    side: str = "auto"            # SIDES
    delay: int = 0                # frames this row's sliders start later
    linework: str = ""            # own linework file (only when the morph is NOT one picture across the grid)
    sweep: List[Track] = field(default_factory=list)   # this row's own slider ranges (replace the stage's track of the same slider)
    size_mode: str = "fixed"      # SIZE_MODES: fixed tile size, or tiles that follow the shape of their picture
    cells: List[CellSpec] = field(default_factory=list)   # one per picture (kept in step with `paths` by sync())

    # --- squares ---------------------------------------------------------------------------------
    def sync(self) -> List[CellSpec]:
        while len(self.cells) < len(self.paths):
            self.cells.append(CellSpec())
        del self.cells[len(self.paths):]
        return self.cells

    def cell(self, i: int) -> CellSpec:
        return self.sync()[i]

    def add_paths(self, ps: Sequence[str]) -> None:
        self.sync()
        for p in ps:
            if p not in self.paths:
                self.paths.append(p)
        self.sync()

    def remove_at(self, idx: Sequence[int]) -> None:
        self.sync()
        gone = set(idx)
        keep = [i for i in range(len(self.paths)) if i not in gone]
        self.paths = [self.paths[i] for i in keep]
        self.cells = [self.cells[i] for i in keep]

    def swap(self, i: int, j: int) -> None:
        self.sync()
        self.paths[i], self.paths[j] = self.paths[j], self.paths[i]
        self.cells[i], self.cells[j] = self.cells[j], self.cells[i]

    # --- resolved values of square i -------------------------------------------------------------
    def tile_size(self, i: int) -> Tuple[int, int]:
        c = self.cell(i)
        mode = c.size_mode or self.size_mode
        w, h = (c.w or self.tile_w), (c.h or self.tile_h)
        if mode == "fixed":
            return w, h
        iw, ih = image_size(self.paths[i])
        return fit_size(mode, w, h, iw / ih)

    def tile_fit(self, i: int) -> str:
        return self.cell(i).fit or self.fit

    def tile_side(self, i: int) -> str:
        return self.cell(i).side or self.side

    def tile_delay(self, i: int) -> int:
        d = self.cell(i).delay
        return self.delay if d is None else d

    def tile_linework(self, i: int) -> str:
        return self.cell(i).linework or self.linework

    def tile_sweep(self, i: int) -> List[Track]:
        """Row sliders, with this square's own ranges replacing the ones of the same name."""
        own = self.cell(i).sweep
        names = {t.param for t in own}
        return [t for t in self.sweep if t.param not in names] + list(own)

    def row_for_cell(self, i: int) -> "RowSpec":
        """The row with this square's trace settings folded in (what params_for_row needs)."""
        c = self.cell(i)
        return replace(self, engine=c.engine or self.engine,
                       preset=c.preset if c.engine else self.preset,
                       colors=self.colors if c.colors is None else c.colors,
                       detail=self.detail if c.detail is None else c.detail,
                       noise=self.noise if c.noise is None else c.noise, cells=[])

    def title(self, i: int) -> str:
        eng = {"main": "main settings", "trace": f"trace · {self.preset or 'default'}",
               "linework": f"linework · {self.preset or 'default'}"}[self.engine]
        size = (f"{self.tile_w}×{self.tile_h}" if self.size_mode == "fixed"
                else {"fit_long": f"fit · long side {max(self.tile_w, self.tile_h)}", "fit_h": f"fit · height {self.tile_h}",
                      "fit_w": f"fit · width {self.tile_w}"}[self.size_mode])
        return f"Row {i + 1}  ·  {len(self.paths)} pictures  ·  {size}  ·  {eng}"


def params_for_row(row: RowSpec, base):
    """The trace parameters of one row: the main window's settings, a preset, plus the row's overrides."""
    if row.engine == "linework" or (row.engine == "main" and isinstance(base, LineworkParams)):
        p = base if isinstance(base, LineworkParams) and row.engine == "main" else LineworkParams()
        if row.preset in LW_PRESETS:
            p = replace(LineworkParams(), **LW_PRESETS[row.preset])
        if row.detail >= 0:
            p = replace(p, simplify=int(row.detail))
        return p
    p = base if isinstance(base, TraceParams) else TraceParams()
    if row.engine == "trace" and row.preset in TRACE_PRESETS:
        p = params_from_preset(row.preset, p)
    if row.colors > 0:
        p = replace(p, colors=int(row.colors))
    if row.detail >= 0:
        p = replace(p, paths=int(row.detail))
    if row.noise >= 0:
        p = replace(p, noise=int(row.noise))
    return p


def params_for_cell(row: RowSpec, i: int, base):
    return params_for_row(row.row_for_cell(i) if row.cells else row, base)


def flat_cells(rows: Sequence[RowSpec]) -> List[Tuple[int, int, CellSpec]]:
    """(row, index in row, square) for every picture of the grid, in picture order."""
    out = []
    for r, row in enumerate(rows):
        row.sync()
        for i in range(len(row.paths)):
            out.append((r, i, row.cells[i]))
    return out


def states_of(rows: Sequence[RowSpec]) -> Tuple[Set[int], Set[int]]:
    """(pictures that go back to white, pictures that stay a vector) as flat picture numbers."""
    white, stay = set(), set()
    for k, (_, _, c) in enumerate(flat_cells(rows)):
        if c.state == "white":
            white.add(k)
        elif c.state == "stay":
            stay.add(k)
    return white, stay
