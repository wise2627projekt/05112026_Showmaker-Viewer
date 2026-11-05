"""Show Maker side of the letter morph: fonts, the text line, and fitting a marked skeleton onto a glyph (needs Qt, scipy, skimage).

Nothing in here is needed to play a show: `letters.LetterMorph` carries the result."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .letters import Letter, LetterMorph, fill_nonzero, normals, tps_eval, tps_fit

# the four fonts the user asked for, with the free look-alike that stands in when the real one is not installed
FONT_CHOICES = ("Futura", "Consolas", "OCR-B", "MS Gothic")
_ALIASES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    # name -> (family names it may be installed as, bundled substitute)
    "futura": (("Futura", "Futura PT", "Futura Std", "Futura Medium", "Futura Book"), "Jost"),
    "consolas": (("Consolas",), "Inconsolata"),
    "ocr-b": (("OCR-B", "OCR B", "OCRB", "OCR-B 10 BT", "OCR B Std", "OCR-B Regular"), "Inconsolata"),
    "ms gothic": (("MS Gothic", "MS ゴシック", "MS Gothic Regular"), "M PLUS 1 Code"),
}
_BUNDLED = ("Jost-Book.ttf", "Inconsolata-Regular.ttf", "MPLUS1Code-Regular.ttf")


def font_dirs() -> List[Path]:
    """Folders searched for extra .ttf / .otf files: the bundled look-alikes and a `fonts` folder next to the program."""
    here = Path(__file__).resolve().parent
    out = [here / "fonts"]
    exe = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else here.parent
    out += [exe / "fonts", Path.home() / "PhotoVec fonts"]
    return out


class FontFace:
    """One font: outlines and advances in em units (y grows downwards, the baseline is y = 0)."""

    def __init__(self, family: str, size: int = 1000):
        from PySide6.QtGui import QFont, QRawFont
        f = QFont(family)
        f.setPixelSize(size)
        self.raw = QRawFont.fromFont(f)
        self.size = float(size)
        self.family = self.raw.familyName() or family
        self.ascent = self.raw.ascent() / self.size
        self.descent = self.raw.descent() / self.size
        self.cap = (self.raw.capHeight() if hasattr(self.raw, "capHeight") else 0) / self.size or 0.7
        self._cache: Dict[str, Optional[Tuple[List[np.ndarray], float]]] = {}

    def valid(self) -> bool:
        return bool(self.raw.isValid())

    def glyph(self, ch: str) -> Optional[Tuple[List[np.ndarray], float]]:
        """(contours, advance) or None when the font has no such character."""
        if ch in self._cache:
            return self._cache[ch]
        res = None
        idx = self.raw.glyphIndexesForString(ch)
        if idx and idx[0] != 0:
            adv = self.raw.advancesForGlyphIndexes(idx)[0].x() / self.size
            polys = [np.array([(q.x(), q.y()) for q in poly], np.float64) / self.size
                     for poly in self.raw.pathForGlyph(idx[0]).toSubpathPolygons()]
            res = ([p for p in polys if len(p) >= 3], float(adv))
        self._cache[ch] = res
        return res


_loaded: Dict[str, str] = {}


def register_font_file(path: str) -> Optional[str]:
    """Make a font file known to Qt; returns its family name (None if Qt cannot read it)."""
    from PySide6.QtGui import QFontDatabase
    path = os.path.abspath(path)
    if path in _loaded:
        return _loaded[path]
    i = QFontDatabase.addApplicationFont(path)
    fams = QFontDatabase.applicationFontFamilies(i) if i >= 0 else []
    if not fams:
        return None
    _loaded[path] = fams[0]
    return fams[0]


def register_bundled() -> None:
    for d in font_dirs():
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    register_font_file(str(f))


def installed_families() -> List[str]:
    from PySide6.QtGui import QFontDatabase
    register_bundled()
    return sorted(set(QFontDatabase.families()), key=str.lower)


def resolve_font(name: str) -> Tuple[FontFace, str]:
    """(font, note). `name` is a family name, one of FONT_CHOICES or a path to a .ttf/.otf/.ttc. `note` says when something else was used."""
    from PySide6.QtGui import QFontDatabase
    register_bundled()
    fams = {f.lower(): f for f in QFontDatabase.families()}
    if os.path.isfile(name):
        fam = register_font_file(name)
        if not fam:
            raise ValueError(f"Cannot read the font file {os.path.basename(name)}")
        return FontFace(fam), ""
    key = name.strip().lower()
    if key in _ALIASES:
        names, sub = _ALIASES[key]
        for n in names:
            if n.lower() in fams:
                return FontFace(fams[n.lower()]), ""
        face = FontFace(fams.get(sub.lower(), sub))
        if face.family.lower() != sub.lower():                   # Qt would quietly use some other font: better to say so
            raise ValueError(f"{name} is not installed and its free look-alike {sub} was not found (the folder “fonts” next to the program is missing).")
        extra = " (put the real font into the fonts folder or load it with the Load button to use it)"
        return face, f"{name} is not installed, so the free look-alike {sub} is used{extra}."
    if key in fams:
        return FontFace(fams[key]), ""
    raise ValueError(f"No font called {name}")


# ---------------------------------------------------------------------------------------------------- the text line

@dataclass
class Placement:
    """Where a glyph of the finished line of text goes: outline contours in photo pixels."""
    contours: List[np.ndarray]
    scale: float                     # pixels per em
    origin: Tuple[float, float]      # baseline start (x, y) in photo pixels
    advance: float                   # em


def order_letters(letters: Sequence[Letter]) -> List[List[int]]:
    """Reading order: lines top to bottom (letters whose centres lie on about the same height), each line left to right."""
    if not letters:
        return []
    boxes = [l.bbox() for l in letters]
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    cy = [(b[1] + b[3]) / 2 for b in boxes]
    hs = [max(1.0, b[3] - b[1]) for b in boxes]
    med_h = float(np.median(hs))
    idx = sorted(range(len(letters)), key=lambda i: cy[i])
    lines: List[List[int]] = []
    for i in idx:
        for line in lines:
            if abs(np.mean([cy[k] for k in line]) - cy[i]) <= 0.6 * med_h:
                line.append(i)
                break
        else:
            lines.append([i])
    lines.sort(key=lambda ln: np.mean([cy[k] for k in ln]))
    return [sorted(ln, key=lambda k: cx[k]) for ln in lines]


def layout_text(letters: Sequence[Letter], font: FontFace, lines: Optional[List[List[int]]] = None,
                space_gap: float = 0.9) -> Dict[int, Placement]:
    """Put the letters in `lines` as normal text, fitted into the area they cover now. A gap between two letters wider than
    `space_gap` x the typical letter width becomes a space."""
    lines = lines if lines is not None else order_letters(letters)
    boxes = [l.bbox() for l in letters]
    if not lines:
        return {}
    ax0, ay0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    ax1, ay1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    widths = [max(1.0, b[2] - b[0]) for b in boxes]
    med_w = float(np.median(widths))
    glyphs = {}
    for i, l in enumerate(letters):
        if len(l.char) > 1:
            raise ValueError(f"A letter is one character (“{l.char}” is {len(l.char)})")
        g = font.glyph(l.char) if l.char else None
        if g is None:
            raise ValueError(f"The font {font.family} has no “{l.char or '?'}”")
        glyphs[i] = g
    # advance positions (em) of every letter on its line
    pos: Dict[int, Tuple[int, float]] = {}
    line_w = []
    for li, ln in enumerate(lines):
        x = 0.0
        for n, i in enumerate(ln):
            if n and boxes[i][0] - boxes[ln[n - 1]][2] > space_gap * med_w:
                x += 0.3                                          # (em)
            pos[i] = (li, x)
            x += glyphs[i][1]
        line_w.append(x)
    # vertical extent of every line = what its glyphs really cover (caps only, or with descenders), stacked with a gap of 0.25 em
    ext = []
    for ln in lines:
        top = min(float(glyphs[i][0][k][:, 1].min()) for i in ln for k in range(len(glyphs[i][0])))
        bot = max(float(glyphs[i][0][k][:, 1].max()) for i in ln for k in range(len(glyphs[i][0])))
        ext.append((top, max(bot, 0.0)))
    gap = 0.25
    block = sum(b - t for t, b in ext) + gap * (len(lines) - 1)
    area_w, area_h = max(1.0, ax1 - ax0), max(1.0, ay1 - ay0)
    scale = min(area_w / max(line_w), area_h / max(block, 1e-6))
    y = ay0 + (area_h - block * scale) / 2
    out: Dict[int, Placement] = {}
    for li, ln in enumerate(lines):
        top, bot = ext[li]
        base_y = y - top * scale
        x_off = ax0 + (area_w - line_w[li] * scale) / 2
        y += (bot - top + gap) * scale
        for i in ln:
            polys, adv = glyphs[i]
            ox = x_off + pos[i][1] * scale
            out[i] = Placement([p * scale + [ox, base_y] for p in polys], scale, (ox, base_y), adv)
    return out


# ---------------------------------------------------------------------------------------------------- skeletons

def _raster(contours: Sequence[np.ndarray], ppu: float, pad: int = 6):
    """Filled glyph on a pixel grid. Returns (mask uint8, (x0, y0) of pixel (0,0) in contour units, ppu)."""
    allp = np.concatenate(contours)
    x0, y0 = allp.min(0) - pad / ppu
    w = int(np.ceil((allp[:, 0].max() - x0) * ppu)) + pad + 1
    h = int(np.ceil((allp[:, 1].max() - y0) * ppu)) + pad + 1
    m = fill_nonzero((h, w), [(c - [x0, y0]) * ppu for c in contours])
    return m, (float(x0), float(y0)), ppu


def glyph_skeleton(contours: Sequence[np.ndarray], height_px: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    """Centre line of a glyph: (points (K,2), half width (K,)) in the units of `contours`."""
    from skimage.morphology import skeletonize
    allp = np.concatenate(contours)
    ext = max(1e-6, float(allp[:, 1].max() - allp[:, 1].min()), float(allp[:, 0].max() - allp[:, 0].min()))
    m, (x0, y0), ppu = _raster(contours, height_px / ext)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    sk = skeletonize(m > 0)
    ys, xs = np.nonzero(sk)
    if not len(xs):
        ys, xs = np.nonzero(dt >= dt.max() * 0.9)
    pts = np.stack([xs / ppu + x0, ys / ppu + y0], 1)
    return pts, dt[ys, xs] / ppu


def stroke_widths(ink: np.ndarray, P: np.ndarray, runs, smooth: int = 2) -> np.ndarray:
    """Half stroke width of the drawing around every skeleton point (ink: HxW float 0..1)."""
    mask = (ink > 0.5).astype(np.uint8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    H, W = dt.shape
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, H - 1)
    w = np.zeros(len(P))
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            w = np.maximum(w, dt[np.clip(yi + dy, 0, H - 1), np.clip(xi + dx, 0, W - 1)])
    good = w >= 1.0
    fill = float(np.median(w[good])) if good.any() else 3.0
    w = np.where(good, w, fill)
    return _smooth_runs(w, runs, smooth)


def _smooth_runs(v: np.ndarray, runs, n: int) -> np.ndarray:
    out = v.copy()
    k = np.ones(2 * n + 1) / (2 * n + 1)
    for a, b, closed in runs:
        x = v[a:b]
        if len(x) < 3:
            continue
        pad = np.concatenate([x[-n:], x, x[:n]]) if closed else np.concatenate([np.full(n, x[0]), x, np.full(n, x[-1])])
        out[a:b] = np.convolve(pad, k, mode="valid")[:len(x)]
    return out


def _second_diff(n: int, runs) -> np.ndarray:
    """Matrix D with D @ X = second differences along every stroke (zero rows elsewhere)."""
    rows = []
    for a, b, closed in runs:
        m = b - a
        rng = range(m) if closed else range(1, m - 1)
        for i in rng:
            r = np.zeros(n)
            r[a + (i - 1) % m] += 1
            r[a + i] -= 2
            r[a + (i + 1) % m] += 1
            rows.append(r)
    return np.array(rows) if rows else np.zeros((0, n))


def fit_to_glyph(S0: np.ndarray, runs, G: np.ndarray, iters: int = 60) -> np.ndarray:
    """Move the marked skeleton S0 (N,2) onto the glyph's centre line G (K,2) (same units), keeping its order and its general shape."""
    from scipy.spatial import cKDTree
    N = len(S0)
    gmin, gmax = G.min(0), G.max(0)
    smin, smax = S0.min(0), S0.max(0)
    gs, ss = np.maximum(gmax - gmin, 1e-6), np.maximum(smax - smin, 1e-6)
    ratio = ss / ss.max()
    sc = gs / ss
    if ratio[0] < 0.12:                      # a vertical stroke ("I"): do not blow its width up
        sc[0] = sc[1]
    if ratio[1] < 0.12:
        sc[1] = sc[0]
    S_init = (S0 - (smin + smax) / 2) * sc + (gmin + gmax) / 2
    S = S_init.copy()
    D2 = _second_diff(N, runs)
    D2tD2 = D2.T @ D2
    D2S0 = D2 @ S_init
    tree = cKDTree(G)
    scale = float(np.linalg.norm(gs))
    for it in range(iters):
        f = it / max(1, iters - 1)
        w_init = 0.30 * (1 - f) + 0.01
        w_bend = 6.0 * (1 - f) + 0.6
        d, nn = tree.query(S)
        T = G[nn]
        tree_s = cKDTree(S)
        _, back = tree_s.query(G)                                   # which skeleton point is nearest to every glyph point
        cnt = np.bincount(back, minlength=N).astype(float)
        R = np.zeros_like(S)
        np.add.at(R, back, G)
        has = cnt > 0
        R[has] /= cnt[has, None]
        wc = 0.8 * np.minimum(cnt, 6) / 6.0
        A = np.diag(1.0 + wc + w_init) + w_bend * D2tD2
        b = T + wc[:, None] * R + w_init * S_init + w_bend * (D2.T @ D2S0)
        S = np.linalg.solve(A, b)
    return S


def make_morph(letter: Letter, place: Placement, font: FontFace, ink: Optional[np.ndarray], step: float = 4.0,
               reach_factor: float = 1.0) -> LetterMorph:
    """Fit one marked letter onto its glyph."""
    S0, runs = letter.flatten(step)
    if len(S0) < 2:
        raise ValueError(f"The letter “{letter.char}” has no path yet")
    G_em, hw_em = glyph_skeleton(_to_em(place))
    G = G_em * place.scale + place.origin
    hw = hw_em * place.scale
    S = fit_to_glyph(S0, runs, G)
    # glyph stroke half width at every fitted point
    from scipy.spatial import cKDTree
    _, nn = cKDTree(G).query(S)
    w_dst = _smooth_runs(hw[nn], runs, 2)
    w_src = stroke_widths(ink, S0, runs) if ink is not None else np.full(len(S0), float(np.median(w_dst)))
    contours1 = [_densify(c, 2.0) for c in merge_overlaps(place.contours)]
    contours0 = [pull_back(c, S, S0, runs, w_dst, w_src) for c in contours1]
    reach = max(float(w_src.max()) * 1.7 + 3.0, 4.0) * reach_factor
    return LetterMorph(char=letter.char, src=S0, dst=S, w_src=w_src, w_dst=w_dst, runs=runs, glyph0=contours0, glyph1=contours1, reach=reach)


def merge_overlaps(contours: Sequence[np.ndarray], max_px: int = 3000) -> List[np.ndarray]:
    """The outline of a glyph without overlapping parts. Many fonts draw a crossbar or a join as a contour that crosses the rest; filled
    with the non-zero rule that is solid, but the quick even-odd fill would punch a hole in it. Such glyphs are merged once here (by
    tracing the filled shape), so playing them back stays fast. Glyphs without overlaps come back untouched."""
    contours = [np.asarray(c, np.float64) for c in contours if len(c) >= 3]
    if not contours:
        return contours
    allp = np.concatenate(contours)
    x0, y0 = np.floor(allp.min(0)) - 2
    ext = max(allp[:, 0].max() - x0, allp[:, 1].max() - y0) + 3
    k = float(min(8.0, max(1.0, max_px / ext)))
    w, h = int(np.ceil((allp[:, 0].max() - x0 + 3) * k)), int(np.ceil((allp[:, 1].max() - y0 + 3) * k))
    q = [(c - [x0, y0]) * k + (k - 1) / 2.0 for c in contours]
    a = fill_nonzero((h, w), q, evenodd=True)
    b = fill_nonzero((h, w), q)
    if np.count_nonzero(a != b) <= 0.0015 * max(1, np.count_nonzero(b)):
        return contours
    found, _ = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in found:
        c = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(np.float64)
        if len(c) >= 3:
            out.append((c - (k - 1) / 2.0) / k + [x0, y0])
    return out or contours


def _to_em(place: Placement) -> List[np.ndarray]:
    return [(c - place.origin) / place.scale for c in place.contours]


def pull_back(poly: np.ndarray, S: np.ndarray, S0: np.ndarray, runs, w_dst: np.ndarray, w_src: np.ndarray) -> np.ndarray:
    """Where the points of the final glyph outline sit on the marked letter: the same spline the Viewer plays with, run backwards."""
    n1, n0 = normals(S, runs), normals(S0, runs)
    sel = np.unique(np.round(np.linspace(0, len(S) - 1, min(len(S), 44))).astype(int))
    cp = np.concatenate([S[sel], (S + n1 * w_dst[:, None])[sel], (S - n1 * w_dst[:, None])[sel]])
    vp = np.concatenate([S0[sel], (S0 + n0 * w_src[:, None])[sel], (S0 - n0 * w_src[:, None])[sel]])
    coef = tps_fit(cp, vp)
    return tps_eval(cp, coef, poly)


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        for j in range(k):
            out.append(a + (b - a) * (j / k))
    return np.asarray(out)
