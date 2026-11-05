"""Illustrator-style Image Trace engine.

Controls mirror Adobe Illustrator's Image Trace panel (Preset, View, Mode, Palette,
Threshold, Paths, Corners, Noise, Method, Fills, Strokes, Smooth, Shapes, Snap Curves To
Lines, Ignore Color, Auto Grouping).  Adobe's tracer is proprietary, so results are
*similar in character*, not identical; preset values below are approximations.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import astuple, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import skeletonize

from .extract import trace_skeleton
from .geometry import Chain, Path, bezier_points, fit_fast, fit_tiny, join_chains

RGB = Tuple[int, int, int]


@dataclass
class TraceParams:
    mode: str = "bw"                 # bw | gray | color
    palette: str = "limited"         # limited | automatic
    colors: int = 6                  # max colours (gray/color, Limited)
    threshold: int = 128             # bw: pixels darker than this are black
    paths: int = 50                  # 0..100  fit tightness (high = closer, more anchors)
    corners: int = 75                # 0..100  (high = more corners)
    noise: int = 25                  # 1..100 px: areas smaller than this are ignored
    method: str = "abutting"         # abutting | overlapping
    fills: bool = True
    strokes: bool = False
    stroke_width: float = 10.0       # max width (px) of features turned into strokes
    smooth: int = 1                  # 0..100 %
    shapes: bool = True              # recognise ellipses
    snap_lines: bool = True          # snap curves to lines
    outlines: bool = False           # stroke the outline of every fill (no inside colour needed)
    outline_width: float = 1.5       # px
    outline_fixed: Optional[RGB] = None   # None = outline takes the fill's colour
    ignore_color: Optional[RGB] = None
    ignore_tol: float = 28.0
    auto_group: bool = False
    max_side: int = 2400             # working resolution cap (0 = full)
    scale: float = 1.0               # px-based settings are tuned for ~1000 px images; clips trace smaller tiles
    grey_blur: float = 0.7           # bw: Gaussian sigma (px) of the grey picture the Threshold is applied to
    soften: float = 0.0              # bw: extra sigma per 1000 px of picture size (smoother outlines that move without jumps)
    scale_cap: float = 1.0           # largest `scale` a tile may get (clips of big pictures raise it so Noise etc. grow with the resolution)
    ink: Optional[RGB] = None        # colour of the drawing in slider clips (None = black); with `paper` for two-colour, Warhol-style pictures
    paper: Optional[RGB] = None      # colour of the paper (None = white)
    fill: float = 0.0                # 0..1: how far the drawing colour has spread over the whole picture (slider clips, see sweep.fill_ink)
    fill_noise: float = 0.45         # 0..1: how irregular that spreading is (0 = an even growth from the drawing)
    fill_size: float = 1.0           # size of the noise blobs (bigger = fewer, larger blobs)
    fill_detail: float = 0.5         # 0..1: 0 = only large soft blobs, 1 = lots of small ones as well
    fill_smooth: float = 0.25        # 0..1: smoothing of the front (0 = ragged, 1 = very round)
    fill_pace: float = 1.0           # progress ** pace: above 1 = slow start, fast end; below 1 = fast start, slow end
    fill_seed: int = 7               # a different seed = a different noise pattern
    fill_origin: str = "drawing"     # where the spreading starts: drawing | edges | centre (always from the drawing as well)
    flip: bool = False               # draw the finished vector with black and white swapped (per picture, Multi-Trace)
    spread: bool = False             # bw: pixels of a flat area darken one after the other (edge-in), so the Threshold can split them (slider sweeps)

    @property
    def blur_sigma(self) -> float:
        return self.grey_blur * max(1.0, self.scale) + self.soften * self.scale   # grain is blurred in proportion to the resolution

    # --- derived fitting parameters ---------------------------------------------------
    @property
    def fit_tol(self) -> float:      # Paths 0 -> 3.0 px ... 100 -> 0.35 px (exponential)
        return max(0.12, 3.0 * (0.35 / 3.0) ** (self.paths / 100.0) * self.scale)

    @property
    def corner_angle(self) -> float:  # Corners 0 -> 120 deg ... 100 -> 30 deg
        return 120.0 - 90.0 * self.corners / 100.0

    @property
    def smooth_window(self) -> int:
        return 3 + 2 * int(round(self.smooth / 100.0 * 8))


_C = dict  # brevity
PRESETS: Dict[str, dict] = {
    "[Default]": {},
    "High Fidelity Photo": _C(mode="color", palette="limited", colors=30, paths=90, corners=25,
                              noise=5, method="abutting", smooth=1),
    "Low Fidelity Photo": _C(mode="color", palette="limited", colors=12, paths=40, corners=20,
                             noise=20, method="abutting", smooth=15),
    "3 Colors": _C(mode="color", colors=3, paths=50, corners=75, noise=25),
    "6 Colors": _C(mode="color", colors=6, paths=50, corners=75, noise=25),
    "16 Colors": _C(mode="color", colors=16, paths=60, corners=75, noise=20),
    "Shades of Gray": _C(mode="gray", colors=16, paths=60, corners=40, noise=20),
    "Black and White Logo": _C(mode="bw", threshold=128, paths=50, corners=75, noise=25),
    "Sketched Art": _C(mode="bw", threshold=140, fills=False, strokes=True, stroke_width=5,
                       paths=60, corners=60, noise=10),
    "Silhouettes": _C(mode="bw", threshold=110, paths=45, corners=50, noise=40, smooth=10),
    "Line Art": _C(mode="bw", threshold=150, fills=False, strokes=True, stroke_width=4,
                   paths=55, corners=85, noise=8),
    "Technical Drawing": _C(mode="bw", threshold=160, fills=False, strokes=True, stroke_width=3,
                            paths=70, corners=90, noise=6, snap_lines=True),
}


def params_from_preset(name: str, base: Optional[TraceParams] = None) -> TraceParams:
    p = TraceParams()
    if base is not None:   # keep view-independent user choices
        p = replace(base, **{k: getattr(TraceParams(), k) for k in
                             ("mode", "palette", "colors", "threshold", "paths", "corners", "noise",
                              "method", "fills", "strokes", "stroke_width", "smooth")})
    return replace(p, **PRESETS[name])


@dataclass
class Shape:
    rings: List[Path]
    rgb: RGB


@dataclass
class TraceResult:
    shapes: List[Shape]
    strokes: List[Path]
    size: Tuple[int, int]
    image_bgr: np.ndarray
    palette: List[RGB]
    timings: Dict[str, float]
    params: TraceParams
    opaque: Optional[np.ndarray] = None   # source alpha >= 128, or None if fully opaque
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def path_count(self) -> int:
        return sum(len(s.rings) for s in self.shapes) + len(self.strokes)

    @property
    def anchor_count(self) -> int:
        return sum(r.node_count() for s in self.shapes for r in s.rings) + \
            sum(p.node_count() for p in self.strokes)

    @property
    def color_count(self) -> int:
        return len({s.rgb for s in self.shapes} | {p.rgb for p in self.strokes})


# --------------------------------------------------------------------------- recolouring
def used_palette_indices(res: "TraceResult") -> List[int]:
    """Palette indices that actually appear in the result (bw: just the ink colour)."""
    used = {s.rgb for s in res.shapes} | {p.rgb for p in res.strokes if p.layer != "OUTLINE_FIXED"}
    return [i for i, c in enumerate(res.palette) if c in used]


def recolor(res: "TraceResult", overrides: Dict[int, RGB]) -> "TraceResult":
    """Copy of `res` with palette entries replaced (index -> new RGB). Geometry is shared."""
    import copy
    mapping = {res.palette[i]: tuple(rgb) for i, rgb in overrides.items()
               if 0 <= i < len(res.palette) and tuple(rgb) != res.palette[i]}
    if not mapping:
        return res
    shapes = [Shape(s.rings, mapping.get(s.rgb, s.rgb)) for s in res.shapes]
    strokes = []
    for st in res.strokes:
        c = mapping.get(st.rgb)
        if c is None or st.layer == "OUTLINE_FIXED":
            strokes.append(st)
        else:
            q = copy.copy(st)       # keeps the cached polyline
            q.rgb = c
            strokes.append(q)
    return replace(res, shapes=shapes, strokes=strokes)


def outline_strokes(shapes: List["Shape"], p: TraceParams) -> List[Path]:
    fixed = p.outline_fixed
    out = []
    for sh in shapes:
        for r in sh.rings:
            out.append(Path(r.segs, True, "OUTLINE_FIXED" if fixed else "OUTLINE", fixed or sh.rgb,
                            p.outline_width * p.scale))
    return out


# --------------------------------------------------------------------------- source
def prepare_source(rgb: np.ndarray, max_side: int, alpha: Optional[np.ndarray] = None):
    """-> (bgr, opaque_mask or None) at working resolution.

    Hidden RGB under transparent pixels is junk (it causes smeared colours), so it is replaced
    by the nearest opaque colour before any blurring / quantisation.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = max(bgr.shape[:2])
    if max_side and m > max_side:
        f = max_side / m
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
    if alpha is None:
        return bgr, None
    opaque = alpha >= 128
    return fill_transparent_colors(bgr, opaque), opaque


def fill_transparent_colors(bgr: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Replace colours under transparent pixels by the nearest opaque colour."""
    if not opaque.any() or opaque.all():
        return bgr
    return nearest_fill(bgr, opaque)


def checkerboard(h: int, w: int, size: int = 12) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // size) + (xx // size)) % 2).astype(np.uint8)
    return np.where(c[..., None] == 0, 255, 222).astype(np.uint8).repeat(3, 2)


def composite_checker(bgr: np.ndarray, opaque: Optional[np.ndarray]) -> np.ndarray:
    if opaque is None:
        return bgr
    out = checkerboard(*bgr.shape[:2])
    out[opaque] = bgr[opaque]
    return out


# --------------------------------------------------------------------------- labels
def _assign(X: np.ndarray, C: np.ndarray, chunk: int = 262144) -> np.ndarray:
    out = np.empty(len(X), np.int16)
    c2 = (C * C).sum(1)[None, :]
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        d = (x * x).sum(1)[:, None] - 2.0 * x @ C.T + c2
        out[s:s + chunk] = np.argmin(d, 1)
    return out


def _kmeans(samples: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(np.unique(samples, axis=0))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 0.5)
    _, _, c = cv2.kmeans(samples, k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
    return c


def _auto_k(samples: np.ndarray, kmax: int = 24) -> int:
    for k in (3, 4, 6, 8, 12, 16, 24):
        if k > kmax:
            break
        c = _kmeans(samples, k)
        err = np.sqrt(((samples[:, None, :] - c[None]) ** 2).sum(2).min(1)).mean()
        if err < 7.0:
            return k
    return kmax


_SPREAD_CACHE: "dict" = {}


def spread_field(g: np.ndarray) -> np.ndarray:
    """0..1 per pixel: how late a pixel of a flat area turns dark. Flat areas (a white background, a plain sky) have thousands of
    pixels with exactly the same grey, so a Threshold can only flip them all at once. This orders them: pixels close to an edge
    of the picture or to an outline go first, the middle of the area last, so the area fills with black from its edges inward."""
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    free = np.pad((cv2.magnitude(gx, gy) <= 16.0).astype(np.uint8), 1)         # 0 = an edge, or outside the picture
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 3)[1:-1, 1:-1]
    h, w = g.shape
    pos = (np.arange(w, dtype=np.float32)[None, :] / max(1, w - 1) + np.arange(h, dtype=np.float32)[:, None] / max(1, h - 1)) / 2.0
    return 0.9 * dt / max(float(dt.max()), 1e-6) + 0.1 * pos            # (the diagonal term separates pixels of equal distance)


def bw_grey(bgr: np.ndarray, spread: bool = False, blur: float = 0.7) -> np.ndarray:
    """The float grey picture the black-and-white threshold is applied to (dark = grey < threshold).
    With `spread` every pixel is lowered by 0.001 .. 1 grey level (see spread_field): below the Threshold of 255 nothing changes
    visibly, but no two neighbours of a flat area are tied any more, and 255 still makes everything black."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), max(0.1, float(blur)))
    if not spread:
        return g
    key = (bgr.shape, round(float(blur), 3), hashlib.blake2b(bgr.tobytes(), digest_size=10).digest())
    f = _SPREAD_CACHE.get(key)
    if f is None:
        f = spread_field(g)
        while len(_SPREAD_CACHE) >= 6:
            _SPREAD_CACHE.pop(next(iter(_SPREAD_CACHE)))
        _SPREAD_CACHE[key] = f
    return g - (0.001 + 0.999 * (1.0 - f))


def make_labels(bgr: np.ndarray, p: TraceParams, opaque: Optional[np.ndarray] = None):
    """Return (labels int16 [-1 = ignored], palette list[RGB], background label or None)."""
    h, w = bgr.shape[:2]
    cv2.setRNGSeed(1234)
    if p.mode == "bw":
        # float grey values: a threshold of 100.0 and 100.1 differ, so a sweep can use thousands of tiny steps
        g = bw_grey(bgr, p.spread, p.blur_sigma)
        dark = g < p.threshold
        if opaque is not None:
            dark &= opaque                     # transparent = paper
        return dark.astype(np.int16), [(255, 255, 255), (0, 0, 0)], 0
    src = cv2.GaussianBlur(bgr, (0, 0), 0.8)
    rng = np.random.default_rng(1234)
    pool_idx = np.flatnonzero(opaque.ravel()) if opaque is not None and opaque.any() else np.arange(h * w)
    sel = rng.choice(pool_idx, size=min(30000, len(pool_idx)), replace=False)
    if p.mode == "gray":
        g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        s = g.reshape(-1)[sel].astype(np.float32)[:, None]
        k = _auto_k(s) if p.palette == "automatic" else p.colors
        c = np.sort(_kmeans(s, k)[:, 0])
        lab = np.searchsorted((c[1:] + c[:-1]) / 2.0, g.reshape(-1)).astype(np.int16).reshape(h, w)
        pal = [(int(v),) * 3 for v in np.rint(c)]
    else:
        lab_img = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        X = lab_img.reshape(-1, 3)
        s = X[sel]
        k = _auto_k(s) if p.palette == "automatic" else p.colors
        C = _kmeans(s, k)
        lab = _assign(X, C).reshape(h, w)
        cnt = np.bincount(lab.ravel(), minlength=len(C)).astype(np.float64)
        # mean RGB per label (one pass per channel)
        sums = [np.bincount(lab.ravel(), weights=src[..., ch].ravel().astype(np.float64),
                            minlength=len(C)) for ch in (2, 1, 0)]
        pal = [tuple(int(round(sums[ch][j] / max(cnt[j], 1))) for ch in range(3)) for j in range(len(C))]
    if opaque is not None:
        lab[~opaque] = -1                      # transparent: never traced, no base layer
    if p.ignore_color is not None:
        r, g_, b = p.ignore_color
        d = np.sqrt(((bgr.astype(np.float32) - np.array([b, g_, r], np.float32)) ** 2).sum(2))
        lab[d <= p.ignore_tol] = -1
    return lab, pal, None


def clean_labels(labels: np.ndarray, noise: int, n: int) -> np.ndarray:
    """Remove regions smaller than `noise` px (both polarities) by nearest-label fill."""
    if noise <= 1:
        return labels
    ign = labels < 0
    bad = np.zeros(labels.shape, bool)
    for k in range(n):
        m = (labels == k).astype(np.uint8)
        if not m.any():
            continue
        _, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < noise
        small[0] = False
        if small.any():
            bad |= small[lab]
    if bad.any():
        keep = ~bad & ~ign
        if keep.any():
            labels = nearest_fill(labels, keep)
            labels[ign] = -1
    return labels


def nearest_fill(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """`values` with every pixel replaced by the value of the nearest pixel where `keep` is True (Euclidean distance; pixels that are
    kept stay as they are). Same result as scipy's distance_transform_edt indices, but about five times faster."""
    _, lb = cv2.distanceTransformWithLabels((~keep).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros((int(lb.max()) + 1,) + values.shape[2:], values.dtype)
    lut[lb[keep]] = values[keep]
    return lut[lb]


# --------------------------------------------------------------------------- strokes
def extract_strokes(labels: np.ndarray, pal: List[RGB], bg: Optional[int], p: TraceParams):
    """Thin features (narrower than `stroke_width`) -> centre-line stroke paths."""
    d = max(3, int(round(p.stroke_width * p.scale)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    k3 = np.ones((3, 3), np.uint8)
    strokes: List[Path] = []
    remove = np.zeros(labels.shape, bool)
    for k in range(len(pal)):
        if k == bg:
            continue
        m = (labels == k).astype(np.uint8)
        if not m.any():
            continue
        thin = m & ~cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, ker), k3)
        if not thin.any():
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        chains = trace_skeleton(skeletonize(thin > 0), max(4.0, 0.8 * d))
        if not chains:
            continue
        chains = join_chains(chains, gap=3.5, ang_tol=40.0)
        _, cc = cv2.connectedComponents(thin, connectivity=8)
        keep_ids = set()
        for ch in chains:
            pts = ch.pts
            if len(pts) < 4 and not ch.closed:
                continue
            xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
            width = float(max(1.0, 2.0 * dt[yi, xi].mean()))
            keep_ids.add(int(cc[yi[len(yi) // 2], xi[len(xi) // 2]]))
            path = fit_fast(pts + 0.5, ch.closed, tol=p.fit_tol, corner_thr=p.corner_angle,
                            smooth_w=max(5, p.smooth_window), snap_lines=p.snap_lines,
                            line_tol=0.4 * p.fit_tol, layer="STROKE")
            if path is not None:
                path.rgb, path.width = pal[k], width
                strokes.append(path)
        if keep_ids:
            remove |= np.isin(cc, list(keep_ids)) & (thin > 0)
    if remove.any():
        labels = labels.copy()
        labels[remove] = -2
        valid = labels >= 0
        fill = nearest_fill(labels, valid)
        ign = labels == -1
        labels = np.where(labels == -2, fill, labels)
        labels[ign] = -1
    return labels, strokes


# --------------------------------------------------------------------------- shapes
def mask_rings(mask: np.ndarray) -> List[np.ndarray]:
    """Marching-squares iso-contours at the 0.5 level: sub-pixel, exactly shared between
    neighbouring regions.  Coordinates are in a corner-based frame (pixel i spans i..i+1)."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0 or h == 0:
        return []
    sub = np.pad(mask[y:y + h, x:x + w].astype(np.float32), 1)
    out = []
    for c in find_contours(sub, 0.5):
        xy = c[:, ::-1] - 0.5 + np.array([x, y])
        out.append(xy[:-1] if abs(xy[0, 0] - xy[-1, 0]) < 1e-8 and abs(xy[0, 1] - xy[-1, 1]) < 1e-8 else xy)
    return out


def _ellipse_path(ring: np.ndarray, tol: float) -> Optional[Path]:
    if len(ring) < 20:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(ring.astype(np.float32).reshape(-1, 1, 2))
    a, b = d1 / 2.0, d2 / 2.0
    if min(a, b) < 3:
        return None
    th = np.radians(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = (ring - [cx, cy]) @ R
    resid = np.abs(np.sqrt((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2) - 1.0) * (a + b) / 2.0
    if resid.max() > max(0.6, 0.8 * tol):
        return None
    k = 0.5522847498
    pts = np.array([[a, 0], [a, k * b], [k * a, b], [0, b], [-k * a, b], [-a, k * b], [-a, 0],
                    [-a, -k * b], [-k * a, -b], [0, -b], [k * a, -b], [a, -k * b], [a, 0]])
    pts = pts @ R.T + [cx, cy]
    return Path([("B", pts[i:i + 4]) for i in (0, 3, 6, 9)], True)


def get_pool():
    """The shared process pool for ring fitting (None inside workers, or when only one process fits)."""
    import multiprocessing
    if multiprocessing.parent_process() is not None:     # inside a worker process: never start a pool of our own
        return None
    from . import memory, pool
    n = memory.auto_workers()
    return pool.get(n) if n >= 2 else None


def _warm_worker(_) -> int:
    """Runs in a worker: load everything the preview / export jobs need now, so the first real job does not wait for the imports."""
    import time as _t
    from . import prerender  # noqa: F401  (imports the sweep renderer, the tracer, scipy and skimage)
    _t.sleep(0.25)                                     # (keeps this worker busy so the next task starts another process)
    return 0


def warm_pool() -> None:
    """Spawn the workers ahead of time (call once at application start) and let them import the heavy modules; an idle pool is
    shut down again by itself."""
    pool = get_pool()
    if pool is not None:
        list(pool.map(_warm_worker, range(pool._max_workers)))


def _fit_worker(args):
    rings, p, W, H = args
    return _rings_to_paths_serial(rings, p, W, H)


def _rings_to_paths(rings: List[np.ndarray], p: TraceParams, W: int, H: int) -> List[Path]:
    if len(rings) >= 1500:
        pool = get_pool()
        if pool is not None:
            try:
                from . import pool as _pool
                order = sorted(range(len(rings)), key=lambda i: -len(rings[i]))
                nch = 3 * pool._max_workers
                chunks = [[rings[i] for i in order[c::nch]] for c in range(nch)]
                with _pool.busy():
                    res = pool.map(_fit_worker, [(c, p, W, H) for c in chunks])
                    return [q for part in res for q in part]
            except Exception:       # broken pool (e.g. out of memory): fall back to serial
                pass
    return _rings_to_paths_serial(rings, p, W, H)


def _rings_to_paths_serial(rings: List[np.ndarray], p: TraceParams, W: int, H: int) -> List[Path]:
    out = []
    tol = p.fit_tol
    hi = np.array([W, H], float)
    for r in rings:
        r = np.minimum(np.maximum(r, 0.0), hi)
        path = None
        small = len(r) <= 64
        if small:
            path = fit_tiny(r, tol, smooth=p.corners < 95)   # degenerate specks are dropped
        elif p.shapes:
            path = _ellipse_path(r, tol)
        if path is None and not small:
            path = fit_fast(r, True, tol=tol, corner_thr=p.corner_angle, smooth_w=p.smooth_window,
                            snap_lines=p.snap_lines, line_tol=0.4 * tol, layer="FILL")
        if path is not None:
            out.append(path)
    return out


def build_shapes(labels: np.ndarray, pal: List[RGB], bg: Optional[int], p: TraceParams) -> List[Shape]:
    H, W = labels.shape
    counts = np.bincount(labels[labels >= 0].ravel(), minlength=len(pal))
    order = [k for k in np.argsort(-counts) if counts[k] > 0 and k != bg]
    shapes: List[Shape] = []
    has_ignored = bool((labels < 0).any())
    if p.method == "overlapping" and bg is None and p.fills:
        acc = np.zeros(labels.shape, bool)
        built = []
        for k in reversed(order):       # smallest region = painted last = built first
            acc |= labels == k
            built.append((k, mask_rings(acc)))
        for k, rings in reversed(built):
            paths = _rings_to_paths(rings, p, W, H)
            for q in paths:
                q.rgb = pal[k]
            if paths:
                shapes.append(Shape(paths, pal[k]))
        return shapes
    under = order[0] if (bg is None and not has_ignored and order and p.fills) else None
    if under is not None:    # dominant colour becomes a base rectangle: no hairline gaps
        rect = Path([("L", np.array([[0, 0], [W, 0]], float)), ("L", np.array([[W, 0], [W, H]], float)),
                     ("L", np.array([[W, H], [0, H]], float)), ("L", np.array([[0, H], [0, 0]], float))],
                    True, "FILL", pal[under])
        shapes.append(Shape([rect], pal[under]))
    skip = None
    if not p.fills and bg is None and order:      # outlines only: the page-colour region is not an object
        border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        if (border == order[0]).mean() >= 0.5:
            skip = order[0]
    for k in order:
        if k == under or k == skip:
            continue
        paths = _rings_to_paths(mask_rings(labels == k), p, W, H)
        for q in paths:
            q.rgb = pal[k]
        if paths:
            shapes.append(Shape(paths, pal[k]))
    return shapes


# --------------------------------------------------------------------------- driver
def _label_key(p: TraceParams):
    bw = p.mode == "bw"
    return (p.mode, p.palette, 0 if bw else p.colors, p.threshold if bw else 0, p.ignore_color, p.ignore_tol, p.spread and bw, round(p.blur_sigma, 3) if bw else 0)


def _remember(cache: dict, key, value):
    if len(cache) >= 10:                     # tiny session cache: keep the 10 most recent stages
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def trace_bgr(bgr: np.ndarray, p: TraceParams, opaque: Optional[np.ndarray] = None,
              cache: Optional[dict] = None) -> TraceResult:
    """`cache` (a dict kept by the caller for one image) lets parameter sweeps skip stages whose inputs did not change:
    Paths / Corners / Smooth sweeps reuse the colour quantisation and the noise clean-up, Noise sweeps reuse the quantisation."""
    t = {}
    t0 = time.perf_counter()
    lk = ("lab", _label_key(p))
    hit = cache.get(lk) if cache is not None else None
    if hit is None:
        hit = make_labels(bgr, p, opaque)
        if cache is not None:
            _remember(cache, lk, hit)
    labels, pal, bg = hit
    t["quantise"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    noise = max(1, int(round(p.noise * p.scale ** 2)))
    ck = ("cln", _label_key(p), noise)
    cl = cache.get(ck) if cache is not None else None
    if cl is None:
        cl = clean_labels(labels, noise, len(pal))
        if cache is not None:
            _remember(cache, ck, cl)
    labels = cl
    t["noise"] = time.perf_counter() - t0
    sk = None
    if cache is not None:          # same cleaned labels + same fitting settings = same result (common in fine sweeps)
        sk = ("res", hashlib.blake2b(labels.tobytes(), digest_size=12).digest(),
              astuple(replace(p, threshold=0, noise=0, colors=0)), tuple(pal))
        done = cache.get(sk)
        if done is not None:
            return done
    strokes: List[Path] = []
    t0 = time.perf_counter()
    if p.strokes:
        labels, strokes = extract_strokes(labels, pal, bg, p)
    t["strokes"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    shapes = build_shapes(labels, pal, bg, p) if (p.fills or p.outlines) else []
    if p.outlines:
        strokes = strokes + outline_strokes(shapes, p)
    if not p.fills:
        shapes = []
    t["fills"] = time.perf_counter() - t0
    h, w = bgr.shape[:2]
    res = TraceResult(shapes, strokes, (w, h), bgr, pal, t, p, opaque)
    if sk is not None:
        for old in [k for k in cache if isinstance(k, tuple) and k[0] == "res"][:-2]:
            cache.pop(old)         # keep the two latest results only (they hold the vector paths)
        cache[sk] = res
    return res


# --------------------------------------------------------------------------- rendering
def _poly(path: Path, n: int = 8) -> np.ndarray:
    poly = getattr(path, "_poly", None)
    if poly is None:
        poly = path.flatten(n)
        path._poly = poly
    return poly


def render_trace(res: TraceResult, view: str = "result", ss: int = 2) -> np.ndarray:
    """BGR preview. view: result | outline | result_outline | outline_source | source."""
    W, H = res.size
    if view == "source":
        return composite_checker(res.image_bgr, res.opaque).copy()
    S = ss if view in ("result", "result_outline") else 1
    if view == "outline_source":
        src = composite_checker(res.image_bgr, res.opaque)
        canvas = cv2.addWeighted(src, 0.4, np.full_like(src, 255), 0.6, 0)
    elif res.opaque is not None:
        canvas = checkerboard(H * S, W * S, 12 * S)
    else:
        canvas = np.full((H * S, W * S, 3), 255, np.uint8)
    if view in ("result", "result_outline"):
        for sh in res.shapes:
            polys = [np.rint(_poly(r) * S).astype(np.int32) for r in sh.rings]
            cv2.fillPoly(canvas, polys, (int(sh.rgb[2]), int(sh.rgb[1]), int(sh.rgb[0])))      # even-odd: holes preserved
        for st in res.strokes:
            cv2.polylines(canvas, [np.rint(_poly(st) * S).astype(np.int32).reshape(-1, 1, 2)], st.closed,
                          st.rgb[::-1], max(1, int(round(st.width * S))), cv2.LINE_AA)
    if view in ("outline", "outline_source", "result_outline"):
        col = (0, 0, 0) if view == "outline" else (0, 0, 220)
        for sh in res.shapes:
            for r in sh.rings:
                cv2.polylines(canvas, [np.rint(_poly(r) * S).astype(np.int32).reshape(-1, 1, 2)], True, col, 1, cv2.LINE_AA)
        for st in res.strokes:
            cv2.polylines(canvas, [np.rint(_poly(st) * S).astype(np.int32).reshape(-1, 1, 2)], st.closed, col, 1, cv2.LINE_AA)
    if S > 1:
        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    return canvas


def _trace_layers(res: TraceResult, ss: int = 2):
    """(colour BGR uint8, coverage uint8 0..255) of the tracing result, both at the picture's size. The shapes are drawn straight onto
    the supersampled canvases (even-odd fill: holes stay holes), no picture per shape."""
    W, H = res.size
    S = ss
    col = np.full((H * S, W * S, 3), 255, np.uint8)
    cov = np.zeros((H * S, W * S), np.uint8)
    for sh in res.shapes:
        polys = [np.rint(_poly(r) * S).astype(np.int32) for r in sh.rings]
        rgb = (int(sh.rgb[2]), int(sh.rgb[1]), int(sh.rgb[0]))
        cv2.fillPoly(col, polys, rgb)
        cv2.fillPoly(cov, polys, 255)
    for st in res.strokes:
        q = np.rint(_poly(st) * S).astype(np.int32).reshape(-1, 1, 2)
        wpx = max(1, int(round(st.width * S)))
        cv2.polylines(col, [q], st.closed, st.rgb[::-1], wpx, cv2.LINE_AA)
        cv2.polylines(cov, [q], st.closed, 255, wpx, cv2.LINE_AA)
    if S > 1:
        col = cv2.resize(col, (W, H), interpolation=cv2.INTER_AREA)
        cov = cv2.resize(cov, (W, H), interpolation=cv2.INTER_AREA)
    return col, cov


def render_trace_rgba(res: TraceResult, ss: int = 2) -> np.ndarray:
    """BGRA image of the tracing result with true transparency (coverage from the vector shapes)."""
    col, cov = _trace_layers(res, ss)
    return cv2.merge([col[..., 0], col[..., 1], col[..., 2], cov])
