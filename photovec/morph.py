"""Morph traced vectors into user-provided linework.

Linework can be an SVG, a DXF or a raster line drawing (PNG/JPG -> Linework Snap).  The traced
vector (fills, strokes, outlines) is converted to polylines; source and target paths are paired
(optimal assignment on position / length), resampled to equal point counts, aligned, and
interpolated frame by frame.  Surplus source paths collapse onto the nearest target line, surplus
target lines grow out of the nearest source path, fills fade out while strokes grow in.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

RGB = Tuple[int, int, int]


# --------------------------------------------------------------------------- data
@dataclass
class DstLine:
    pts: np.ndarray               # (n,2)
    closed: bool = False
    rgb: RGB = (0, 0, 0)
    width: float = 1.5            # in the units of its drawing until fitted, then canvas px


@dataclass
class Linework:
    lines: List[DstLine]
    page: Optional[Tuple[float, float, float, float]] = None   # x0, y0, x1, y1 of the drawing's page
    source: str = ""

    def bounds(self) -> Tuple[float, float, float, float]:
        allp = np.vstack([l.pts for l in self.lines])
        return (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max()))


@dataclass
class SrcItem:
    pts: np.ndarray
    closed: bool
    rgb: RGB
    width: float                  # stroke width (0 for fills)
    group: int = -1               # fill group (shape index) or -1 for strokes
    fade_only: bool = False       # page-sized background rectangle: fades, never morphs


@dataclass
class MorphOptions:
    stagger: str = "left_right"   # none | left_right | right_left | top_bottom | bottom_top | radial | random
    spread: float = 0.6           # 0 = all paths move together; larger = more staggered
    ease: str = "smooth"          # smooth | linear | out
    seed: int = 7
    width_override: Optional[float] = None   # px; None = use widths of the linework
    min_width: float = 0.8
    max_points: int = 500
    fit: str = "contain"          # contain | cover | stretch | page
    margin: float = 0.03          # fraction of the canvas left free in contain/cover/stretch


# --------------------------------------------------------------------------- loaders
RASTER_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def load_linework(path: str) -> Linework:
    """Load SVG, DXF or a raster line drawing. Raises ValueError with a clear message."""
    ext = os.path.splitext(path)[1].lower()
    if not os.path.isfile(path):
        raise ValueError(f"linework file not found: {path}")
    if ext == ".svg":
        lw = _load_svg(path)
    elif ext == ".dxf":
        lw = _load_dxf(path)
    elif ext in RASTER_EXT:
        lw = _load_raster(path)
    elif ext == ".pdf":
        raise ValueError("PDF is not read directly: export it to SVG or PNG first")
    else:
        raise ValueError(f"unsupported linework format '{ext}' (use SVG, DXF or PNG)")
    lw.lines = [l for l in lw.lines if len(l.pts) >= 2 and np.isfinite(l.pts).all()]
    if not lw.lines:
        raise ValueError(f"no lines found in {os.path.basename(path)}")
    lw.source = path
    return lw


def _is_white(c) -> bool:
    return c is not None and c.red > 240 and c.green > 240 and c.blue > 240


def _load_svg(path: str) -> Linework:
    import svgelements as se
    svg = se.SVG.parse(path, reify=True)
    lines: List[DstLine] = []
    for el in svg.elements():
        if not isinstance(el, se.Shape) or isinstance(el, se.Group):
            continue
        stroke = el.stroke if getattr(el.stroke, "value", None) is not None else None
        fill = el.fill if getattr(el.fill, "value", None) is not None else None
        if stroke is None and (fill is None or _is_white(fill)):
            continue                        # invisible, or the white page rectangle
        col = stroke if stroke is not None else fill
        rgb = (int(col.red), int(col.green), int(col.blue))
        width = float(getattr(el, "implicit_stroke_width", None) or el.stroke_width or 1.0) if stroke is not None else 1.0
        try:
            sp_all = list(se.Path(el).as_subpaths())
        except Exception:
            continue
        for sp in sp_all:
            pts = _sample_subpath(se, se.Path(sp))
            if pts is None:
                continue
            closed = bool(isinstance(sp.segments(False)[-1], se.Close)) if len(sp.segments(False)) else False
            if not closed and len(pts) > 3 and np.hypot(*(pts[0] - pts[-1])) < 1e-6 * max(1.0, float(np.ptp(pts))):
                closed = True
            if closed and np.hypot(*(pts[0] - pts[-1])) < 1e-9:
                pts = pts[:-1]
            lines.append(DstLine(pts, closed, rgb, width))
    page = None
    try:
        page = (0.0, 0.0, float(svg.width), float(svg.height))
    except Exception:
        pass
    return Linework(lines, page)


def _sample_subpath(se, sp) -> Optional[np.ndarray]:
    out: List[np.ndarray] = []
    for seg in sp:
        if isinstance(seg, (se.Move, se.Close)):
            if isinstance(seg, se.Move) and seg.end is not None and not out:
                out.append(np.array([[seg.end.x, seg.end.y]], float))
            elif isinstance(seg, se.Close) and out:
                pass
            continue
        if isinstance(seg, se.Line):
            out.append(np.array([[seg.end.x, seg.end.y]], float))
            continue
        try:
            L = float(seg.length())
        except Exception:
            L = 20.0
        k = int(min(96, max(6, math.ceil(L / 2.0))))
        pts = np.asarray(seg.npoint(np.linspace(0, 1, k)), float).reshape(-1, 2)
        out.append(pts[1:])
    if not out:
        return None
    pts = np.vstack(out)
    return pts if len(pts) >= 2 else None


def _load_dxf(path: str) -> Linework:
    import ezdxf
    from ezdxf import path as ezpath
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    ents = []

    def walk(entities, depth=0):
        for e in entities:
            if e.dxftype() == "INSERT" and depth < 4:
                try:
                    walk(list(e.virtual_entities()), depth + 1)
                except Exception:
                    pass
            else:
                ents.append(e)

    walk(msp)
    paths = []
    for e in ents:
        if e.dxftype() not in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "LWPOLYLINE", "POLYLINE", "SPLINE"):
            continue
        try:
            paths.append(ezpath.make_path(e))
        except Exception:
            continue
    if not paths:
        return Linework([])
    ext = np.array([[v.x, v.y] for p in paths for v in (p.start, p.end)])
    diag = float(np.hypot(*np.ptp(ext, axis=0))) or 1.0
    lines: List[DstLine] = []
    for p in paths:
        for sp in p.sub_paths():
            v = np.array([[q.x, -q.y] for q in sp.flattening(diag / 3000.0, segments=8)], float)
            if len(v) < 2:
                continue
            closed = bool(sp.is_closed) and len(v) > 3
            if closed and np.hypot(*(v[0] - v[-1])) < 1e-9 * diag:
                v = v[:-1]
            lines.append(DstLine(v, closed, (0, 0, 0), diag / 800.0))
    return Linework(lines)


def _load_raster(path: str) -> Linework:
    from .imageio_ import load_image_alpha
    from .linework import LineworkParams, linework_bgr
    from .trace import prepare_source
    rgb, alpha = load_image_alpha(path)
    bgr, op = prepare_source(rgb, 3000, alpha)
    res = linework_bgr(bgr, LineworkParams(simplify=25, min_len=6, gap=3), op)
    lines = []
    for p in res.strokes:
        pts = p.flatten(8)
        closed = p.closed
        if closed and len(pts) > 3 and np.hypot(*(pts[0] - pts[-1])) < 1e-6:
            pts = pts[:-1]
        lines.append(DstLine(pts, closed, (0, 0, 0), float(p.width or 1.0)))
    h, w = bgr.shape[:2]
    return Linework(lines, (0.0, 0.0, float(w), float(h)))


# --------------------------------------------------------------------------- fitting linework into a canvas
def fit_linework(lw: Linework, W: int, H: int, o: MorphOptions) -> List[DstLine]:
    """Map the drawing into canvas pixels. Returns new lines (input is not modified)."""
    if o.fit == "page" and lw.page is not None:
        x0, y0, x1, y1 = lw.page
    else:
        x0, y0, x1, y1 = lw.bounds()
    bw, bh = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    aw, ah = W * (1 - 2 * o.margin), H * (1 - 2 * o.margin)
    if o.fit == "page":
        aw, ah = W, H
    if o.fit == "stretch":
        sx, sy = aw / bw, ah / bh
        s = math.sqrt(sx * sy)
    else:
        s = max(aw / bw, ah / bh) if o.fit == "cover" else min(aw / bw, ah / bh)
        sx = sy = s
    ox = (W - bw * sx) / 2.0 - x0 * sx
    oy = (H - bh * sy) / 2.0 - y0 * sy
    out = []
    for l in lw.lines:
        pts = l.pts * np.array([sx, sy]) + np.array([ox, oy])
        w = o.width_override if o.width_override else max(o.min_width, l.width * s)
        out.append(DstLine(pts, l.closed, l.rgb, float(w)))
    return out


# --------------------------------------------------------------------------- source items from a trace
def _flat(path) -> np.ndarray:
    """8 points per curve segment, shared with the trace renderer's cache (never modified in place)."""
    poly = getattr(path, "_poly", None)
    if poly is None:
        poly = path._poly = path.flatten(8)
    return poly


def items_from_result(res) -> List[SrcItem]:
    """Convert a TraceResult (shapes + strokes) into polyline items in working-pixel coordinates.
    The result is cached on `res` (grid tiles compute it inside the tracing worker processes)."""
    cached = getattr(res, "_items", None)
    if cached is not None:
        return cached
    W, H = res.size
    items: List[SrcItem] = []
    for gi, sh in enumerate(res.shapes):
        for ring in sh.rings:
            pts = _flat(ring)
            if len(pts) > 3 and np.hypot(*(pts[0] - pts[-1])) < 1e-6:
                pts = pts[:-1]
            if len(pts) < 3:
                continue
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            page = (x1 - x0) >= 0.98 * W and (y1 - y0) >= 0.98 * H
            items.append(SrcItem(pts, True, tuple(int(v) for v in sh.rgb), 0.0, gi, page))
    for st in res.strokes:
        pts = _flat(st)
        if len(pts) < 2:
            continue
        closed = bool(st.closed)
        if closed and len(pts) > 3 and np.hypot(*(pts[0] - pts[-1])) < 1e-6:
            pts = pts[:-1]
        items.append(SrcItem(pts, closed, tuple(int(v) for v in (st.rgb or (0, 0, 0))),
                             float(st.width or 1.0), -1))
    res._items = items
    return items


def offset_items(items: Sequence[SrcItem], dx: float, dy: float, group_base: int = 0) -> List[SrcItem]:
    return [SrcItem(i.pts + [dx, dy], i.closed, i.rgb, i.width,
                    (i.group + group_base) if i.group >= 0 else -1, i.fade_only) for i in items]


def offset_lines(lines: Sequence[DstLine], dx: float, dy: float) -> List[DstLine]:
    return [DstLine(l.pts + [dx, dy], l.closed, l.rgb, l.width) for l in lines]


# --------------------------------------------------------------------------- geometry helpers
def _cum(pts: np.ndarray, closed: bool) -> np.ndarray:
    P = np.vstack([pts, pts[:1]]) if closed else pts
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def path_length(pts: np.ndarray, closed: bool) -> float:
    return float(_cum(pts, closed)[-1])


def resample(pts: np.ndarray, closed: bool, n: int) -> np.ndarray:
    """n points equally spaced by arc length (closed: cyclic, no repeated end point)."""
    P = np.concatenate([pts, pts[:1]]) if closed else pts
    d = np.diff(P, axis=0)
    c = np.empty(len(P))
    c[0] = 0.0
    np.cumsum(np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2), out=c[1:])
    L = c[-1]
    if L < 1e-9:
        return np.repeat(P[:1], n, axis=0)
    t = np.arange(n) * (L / (n if closed else max(n - 1, 1)))
    out = np.empty((n, 2))
    out[:, 0] = np.interp(t, c, P[:, 0])
    out[:, 1] = np.interp(t, c, P[:, 1])
    return out


def _best_roll(S: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, float]:
    """Cyclically shift (and maybe reverse) T to best match S. Returns (T_aligned, cost)."""
    n = len(S)
    best = (None, np.inf)
    for cand in (T, T[::-1]):
        step = max(1, n // 24)
        sh = np.arange(0, n, step)
        idx = (np.arange(n)[None, :] + sh[:, None]) % n
        cost = ((S[None] - cand[idx]) ** 2).sum((1, 2))
        k = int(sh[np.argmin(cost)])
        if step == 1:                       # every shift was tried already
            if cost.min() < best[1]:
                best = (cand[idx[k]], float(cost.min()))
            continue
        sh2 = np.arange(k - step, k + step + 1) % n
        idx2 = (np.arange(n)[None, :] + sh2[:, None]) % n
        cost2 = ((S[None] - cand[idx2]) ** 2).sum((1, 2))
        j = int(np.argmin(cost2))
        if cost2[j] < best[1]:
            best = (cand[idx2[j]], float(cost2[j]))
    return best


def _roll_fast(S: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Cut the ring S where it is nearest to the start of the open line T and pick the better direction.
    Two cost evaluations instead of the full search of _best_roll: used for the many tiny paths that merge onto a line."""
    i0 = int(np.argmin(((S - T[0]) ** 2).sum(1)))
    A = np.roll(S, -i0, axis=0)
    B = np.concatenate([A[:1], A[:0:-1]])
    return A if ((A - T) ** 2).sum() <= ((B - T) ** 2).sum() else B


def _align(S: np.ndarray, sc: bool, T: np.ndarray, tc: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned (S, T) so that point i of S travels to point i of T."""
    if sc and tc:
        T2, _ = _best_roll(S, T)
        return S, T2
    if sc and not tc:                       # ring -> open line: cut the ring where it fits best
        S2, _ = _best_roll(T, S)
        return S2, T
    if not sc and tc:                       # open -> ring: cut the ring
        T2, _ = _best_roll(S, T)
        return S, T2
    a = ((S - T) ** 2).sum()
    b = ((S - T[::-1]) ** 2).sum()
    return (S, T) if a <= b else (S, T[::-1])


def _subline(pts: np.ndarray, closed: bool, centre: np.ndarray, length: float,
             cache: Dict[int, Tuple[np.ndarray, np.ndarray]], key: int) -> np.ndarray:
    """Piece of a target line of about `length` px around the point nearest to `centre`."""
    if key not in cache:
        L = path_length(pts, closed)
        m = int(min(2000, max(16, L / 1.5)))
        d = resample(pts, closed, m)
        cache[key] = (d, _cum(d, closed))
    d, c = cache[key]
    m = len(d)
    i = int(np.argmin(((d - centre) ** 2).sum(1)))
    total = float(c[-1])
    length = min(max(length, 1.0), total)
    step = total / max(m - (0 if closed else 1), 1)
    half = max(1, int(round(length / step / 2.0)))
    if closed:
        idx = np.arange(i - half, i + half + 1) % m
    else:
        lo, hi = max(0, i - half), min(m - 1, i + half)
        if hi - lo < 2:
            lo, hi = max(0, hi - 2), min(m - 1, lo + 2)
        idx = np.arange(lo, hi + 1)
    return d[idx]


# --------------------------------------------------------------------------- plan
@dataclass
class MorphPlan:
    size: Tuple[int, int]
    S: np.ndarray                 # (P,2) start points, all paths concatenated
    T: np.ndarray                 # (P,2) end points
    off: np.ndarray               # (K+1,) offsets
    s_rgb: np.ndarray             # (K,3)
    t_rgb: np.ndarray
    s_w: np.ndarray               # (K,)
    t_w: np.ndarray
    closed_t: np.ndarray          # (K,) bool: draw closed once the morph has finished
    group: np.ndarray             # (K,) fill group id or -1
    delay: np.ndarray             # (K,) 0..1
    fades: List[Tuple[np.ndarray, RGB]]           # page-sized rectangles (fade only)
    opts: MorphOptions
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.off) - 1


def _ease(u: np.ndarray, kind: str) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    if kind == "linear":
        return u
    if kind == "out":
        return 1 - (1 - u) ** 3
    return u * u * (3 - 2 * u)


def _delays(cent: np.ndarray, W: int, H: int, o: MorphOptions) -> np.ndarray:
    n = len(cent)
    if n == 0 or o.stagger == "none":
        return np.zeros(n)
    x, y = cent[:, 0] / max(W, 1), cent[:, 1] / max(H, 1)
    if o.stagger == "left_right":
        d = x
    elif o.stagger == "right_left":
        d = 1 - x
    elif o.stagger == "top_bottom":
        d = y
    elif o.stagger == "bottom_top":
        d = 1 - y
    elif o.stagger == "radial":
        d = np.hypot(x - 0.5, y - 0.5) / 0.7071
    else:
        d = np.random.default_rng(o.seed).random(n)
    d = np.clip(d, 0, 1)
    lo, hi = float(d.min()), float(d.max())
    return (d - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros(n)


def _chunks(n: int, parts: int) -> List[Tuple[int, int]]:
    parts = max(1, min(parts, n or 1))
    cuts = np.linspace(0, n, parts + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a] or [(0, 0)]


def _pair_chunk(args):
    """Align a slice of the source items with their target lines (runs in a worker process for big grids)."""
    A, cs, La, match, B, spacing, o = args
    ct = np.array([b.pts.mean(0) for b in B])
    treeB = cKDTree(ct)
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    S_l, T_l, meta = [], [], []
    nm = ng = 0

    def count_pts(L: float) -> int:
        return int(min(o.max_points, max(8, math.ceil(L / spacing))))

    for i, a in enumerate(A):
        j = match[i]
        if j >= 0:
            b = B[j]
            k = count_pts(max(La[i], path_length(b.pts, b.closed)))
            S, T = _align(resample(a.pts, a.closed, k), a.closed, resample(b.pts, b.closed, k), b.closed)
            tw, closed_t = b.width, b.closed
            nm += 1
        else:
            j = int(treeB.query(cs[i])[1])
            b = B[j]
            k = count_pts(La[i])
            piece = _subline(b.pts, b.closed, cs[i], max(La[i], 4 * spacing), cache, j)
            S, T = resample(a.pts, a.closed, k), resample(piece, False, k)
            S = _roll_fast(S, T) if a.closed else _align(S, False, T, False)[0]
            tw, closed_t = 0.0, False
            ng += 1
        S_l.append(S)
        T_l.append(T)
        meta.append((a.rgb, b.rgb, a.width, tw, closed_t, a.group, S.mean(0)))
    return S_l, T_l, meta, nm, ng


def build_plan(items: Sequence[SrcItem], lines: Sequence[DstLine], size: Tuple[int, int],
               opts: Optional[MorphOptions] = None, workers: int = 1, pool=None) -> MorphPlan:
    """Pair source items with target lines and prepare everything needed to render any frame.
    With `pool` (a process pool) and many items the pairing runs in `workers` chunks at once."""
    o = opts or MorphOptions()
    W, H = size
    diag = math.hypot(W, H)
    spacing = max(1.5, diag / 700.0)
    fades = [(i.pts, i.rgb) for i in items if i.fade_only]
    A = [i for i in items if not i.fade_only]
    B = list(lines)
    n, m = len(A), len(B)
    S_l: List[np.ndarray] = []
    T_l: List[np.ndarray] = []
    meta: List[tuple] = []            # (s_rgb, t_rgb, s_w, t_w, closed_t, group, start_centroid)
    stats = dict(sources=n, targets=m, matched=0, merged=0, born=0)

    if n and m:
        cs = np.array([a.pts.mean(0) for a in A])
        ct = np.array([b.pts.mean(0) for b in B])
        La = np.array([path_length(a.pts, a.closed) for a in A])
        Lb = np.array([path_length(b.pts, b.closed) for b in B])
        match: Dict[int, int] = {}
        if n * m <= 2_500_000:
            cost = cdist(cs, ct) / diag + 0.25 * np.abs(np.log((La[:, None] + 1) / (Lb[None, :] + 1)))
            cost += 0.08 * (np.array([a.closed for a in A])[:, None] != np.array([b.closed for b in B])[None, :])
            r, c = linear_sum_assignment(cost)
            match = {int(i): int(j) for i, j in zip(r, c)}
        else:
            tree = cKDTree(ct)
            kq = min(24, m)
            _, nn = tree.query(cs, k=kq)
            nn = nn.reshape(n, -1)
            used = np.zeros(m, bool)
            for i in np.argsort(-La):
                for j in nn[i]:
                    if not used[j]:
                        used[j] = True
                        match[int(i)] = int(j)
                        break
        matched_b = set(match.values())
        mlist = [match.get(i, -1) for i in range(n)]
        chunks = _chunks(n, workers if n >= 4000 else 1)
        jobs = [(A[lo:hi], cs[lo:hi], La[lo:hi], mlist[lo:hi], B, spacing, o) for lo, hi in chunks]
        parts = None
        if len(jobs) > 1 and pool is not None:
            try:
                parts = list(pool.map(_pair_chunk, jobs))
            except Exception:            # a worker died (out of memory): do it here
                parts = None
        if parts is None:
            parts = [_pair_chunk(j) for j in jobs]
        for pS, pT, pm, nm, ng in parts:
            S_l += pS
            T_l += pT
            meta += pm
            stats["matched"] += nm
            stats["merged"] += ng
        matched_b = set(match.values())

        def count_pts(L: float) -> int:
            return int(min(o.max_points, max(8, math.ceil(L / spacing))))

        treeA = cKDTree(cs)
        for j, b in enumerate(B):
            if j in matched_b:
                continue
            i = int(treeA.query(ct[j])[1])
            src = A[i].pts
            p0 = src[int(np.argmin(((src - ct[j]) ** 2).sum(1)))]
            k = count_pts(Lb[j])
            T = resample(b.pts, b.closed, k)
            S = np.repeat(p0[None, :], k, axis=0)
            S_l.append(S)
            T_l.append(T)
            meta.append((b.rgb, b.rgb, 0.0, b.width, b.closed, -1, p0))
            stats["born"] += 1
    elif n:                              # no target: everything just dissolves in place
        for a in A:
            k = int(min(o.max_points, max(8, math.ceil(path_length(a.pts, a.closed) / spacing))))
            S = resample(a.pts, a.closed, k)
            S_l.append(S)
            T_l.append(S.copy())
            meta.append((a.rgb, a.rgb, a.width, 0.0, False, a.group, S.mean(0)))
    else:                                # no source: lines grow from their own centre
        for b in B:
            k = int(min(o.max_points, max(8, math.ceil(path_length(b.pts, b.closed) / spacing))))
            T = resample(b.pts, b.closed, k)
            S_l.append(np.repeat(T.mean(0)[None, :], k, axis=0))
            T_l.append(T)
            meta.append((b.rgb, b.rgb, 0.0, b.width, b.closed, -1, T.mean(0)))
            stats["born"] += 1

    K = len(S_l)
    off = np.r_[0, np.cumsum([len(s) for s in S_l])].astype(np.int64)
    cent = np.array([mm[6] for mm in meta]) if K else np.zeros((0, 2))
    return MorphPlan(
        size=size,
        S=np.vstack(S_l) if K else np.zeros((0, 2)), T=np.vstack(T_l) if K else np.zeros((0, 2)),
        off=off,
        s_rgb=np.array([mm[0] for mm in meta], float).reshape(-1, 3),
        t_rgb=np.array([mm[1] for mm in meta], float).reshape(-1, 3),
        s_w=np.array([mm[2] for mm in meta], float), t_w=np.array([mm[3] for mm in meta], float),
        closed_t=np.array([mm[4] for mm in meta], bool), group=np.array([mm[5] for mm in meta], int),
        delay=_delays(cent, W, H, o), fades=fades, opts=o, stats=stats)


def merge_plans(plans: Sequence[MorphPlan], offsets: Sequence[Tuple[float, float]],
                size: Tuple[int, int]) -> MorphPlan:
    """Combine tile plans into one grid-wide plan (tile coordinates -> grid coordinates)."""
    S, T, off, s_rgb, t_rgb, s_w, t_w, ct, grp, dly, fades = [], [], [], [], [], [], [], [], [], [], []
    base_pt, base_g = 0, 0
    stats: Dict[str, int] = {}
    for p, (dx, dy) in zip(plans, offsets):
        S.append(p.S + [dx, dy])
        T.append(p.T + [dx, dy])
        off.append(p.off[:-1] + base_pt)
        base_pt += int(p.off[-1])
        s_rgb.append(p.s_rgb); t_rgb.append(p.t_rgb); s_w.append(p.s_w); t_w.append(p.t_w)
        ct.append(p.closed_t)
        grp.append(np.where(p.group >= 0, p.group + base_g, -1))
        base_g += int(p.group.max()) + 1 if len(p.group) and p.group.max() >= 0 else 0
        dly.append(p.delay)
        fades += [(f + [dx, dy], c) for f, c in p.fades]
        for k, v in p.stats.items():
            stats[k] = stats.get(k, 0) + v
    o = plans[0].opts
    return MorphPlan(size, np.vstack(S), np.vstack(T), np.r_[np.concatenate(off), base_pt].astype(np.int64),
                     np.vstack(s_rgb), np.vstack(t_rgb), np.concatenate(s_w), np.concatenate(t_w),
                     np.concatenate(ct), np.concatenate(grp), np.concatenate(dly), fades, o, stats)


# --------------------------------------------------------------------------- rendering
def progress_of(plan: MorphPlan, m: float) -> np.ndarray:
    sp = plan.opts.spread
    u = (m * (1.0 + sp) - sp * plan.delay)
    return _ease(u, plan.opts.ease)


def render_morph(plan: MorphPlan, m: float, out_size: Tuple[int, int], bg: RGB = (255, 255, 255),
                 ss: int = 1) -> np.ndarray:
    """One morph frame (BGR) at motion fraction m in [0,1], drawn at out_size (w,h)."""
    ow, oh = out_size
    k = ow / plan.size[0] * ss
    SH = 4
    canvas = np.empty((oh * ss, ow * ss, 3), np.uint8)
    canvas[:] = np.array(bg[::-1], np.uint8)
    bgf = np.array(bg, float)
    if plan.count:
        p = progress_of(plan, m)
        cnt = np.diff(plan.off)
        pp = np.repeat(p, cnt)[:, None]
        P = np.rint((plan.S + (plan.T - plan.S) * pp) * (k * (1 << SH))).astype(np.int32)
    gm = _ease(np.array([m * (1 + plan.opts.spread)]), plan.opts.ease)[0]
    # page-sized backgrounds fade first
    for pts, rgb in plan.fades:
        a = max(0.0, 1.0 - gm * 1.25)
        if a > 1 / 255:
            col = bgf + (np.array(rgb, float) - bgf) * a
            q = np.rint(pts * (k * (1 << SH))).astype(np.int32)
            cv2.fillPoly(canvas, [q], tuple(float(v) for v in col[::-1]), cv2.LINE_AA, SH)
    if not plan.count:
        return canvas if ss == 1 else cv2.resize(canvas, (ow, oh), interpolation=cv2.INTER_AREA)
    # fills: grouped (even-odd holes), faded by the group's leading ring
    gids = plan.group
    fill_idx = np.nonzero(gids >= 0)[0]
    if len(fill_idx):
        order = fill_idx[np.argsort(gids[fill_idx], kind="stable")]
        g_sorted = gids[order]
        bounds = np.r_[0, np.nonzero(np.diff(g_sorted))[0] + 1, len(order)]
        cnt_all = np.diff(plan.off)
        for a0, a1 in zip(bounds[:-1], bounds[1:]):
            members = order[a0:a1]
            lead = members[int(np.argmax(cnt_all[members]))]
            alpha = max(0.0, 1.0 - 1.25 * p[lead])
            if alpha <= 1 / 255:
                continue
            col = bgf + (plan.s_rgb[lead] - bgf) * alpha
            polys = [P[plan.off[i]:plan.off[i + 1]] for i in members]
            cv2.fillPoly(canvas, polys, tuple(float(v) for v in col[::-1]), cv2.LINE_AA, SH)
    # strokes (outlines, centre lines, growing target lines)
    w = (plan.s_w + (plan.t_w - plan.s_w) * p) * k
    cols = plan.s_rgb + (plan.t_rgb - plan.s_rgb) * p[:, None]
    for i in np.nonzero(w > 0.04)[0]:
        wi = float(w[i])
        c = cols[i]
        if wi < 1.0:
            c = bgf + (c - bgf) * wi
        closed = bool(plan.closed_t[i] and p[i] >= 0.999)
        cv2.polylines(canvas, [P[plan.off[i]:plan.off[i + 1]]], closed,
                      (float(c[2]), float(c[1]), float(c[0])), max(1, int(round(wi))), cv2.LINE_AA, SH)
    return canvas if ss == 1 else cv2.resize(canvas, (ow, oh), interpolation=cv2.INTER_AREA)


def render_lines(lines: Sequence[DstLine], size: Tuple[int, int], out_size: Tuple[int, int],
                 bg: RGB = (255, 255, 255), ss: int = 1) -> np.ndarray:
    """Draw the fitted target linework (used as the 'linework' image in wipes)."""
    ow, oh = out_size
    k = ow / size[0] * ss
    canvas = np.empty((oh * ss, ow * ss, 3), np.uint8)
    canvas[:] = np.array(bg[::-1], np.uint8)
    for l in lines:
        q = np.rint(l.pts * k * 16).astype(np.int32)
        wi = max(1, int(round(l.width * k)))
        cv2.polylines(canvas, [q], l.closed, (l.rgb[2], l.rgb[1], l.rgb[0]), wi, cv2.LINE_AA, 4)
    return canvas if ss == 1 else cv2.resize(canvas, (ow, oh), interpolation=cv2.INTER_AREA)
