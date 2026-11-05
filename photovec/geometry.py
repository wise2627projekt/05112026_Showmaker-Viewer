"""Chains, paths, hybrid straight/Bezier fitting, joining and de-duplication."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.spatial import cKDTree

Seg = Tuple[str, np.ndarray]  # ("L", (2,2)) or ("B", (4,2))


# --------------------------------------------------------------------- types
@dataclass
class Chain:
    """Ordered pixel/point chain. Closed chains do NOT repeat the first point."""
    pts: np.ndarray
    closed: bool = False
    layer: str = "EDGES"
    fixed: bool = False  # detector-straight: never smoothed, joined or refit

    def length(self) -> float:
        return polyline_length(self.pts, self.closed)


@dataclass
class Path:
    segs: List[Seg]
    closed: bool = False
    layer: str = "EDGES"
    rgb: "tuple | None" = None      # trace engine: fill / stroke colour
    width: "float | None" = None    # trace engine: stroke width (px)

    def length(self) -> float:
        return polyline_length(self.flatten(), self.closed)

    def node_count(self) -> int:
        return 1 + sum(1 if k == "L" else 3 for k, _ in self.segs)

    def is_straight(self) -> bool:
        return len(self.segs) == 1 and self.segs[0][0] == "L"

    def flatten(self, n: int = 16) -> np.ndarray:
        segs = self.segs
        ci = [i for i, (k, _) in enumerate(segs) if k != "L"]
        if len(ci) == len(segs):                              # all curves (the usual case): one vectorised evaluation
            C = np.stack([c for _, c in segs])                # (m, 4, 2)
            t = np.linspace(0.0, 1.0, n)[1:, None, None]
            m = 1.0 - t
            pts = (m ** 3 * C[None, :, 0] + 3 * m ** 2 * t * C[None, :, 1]
                   + 3 * m * t ** 2 * C[None, :, 2] + t ** 3 * C[None, :, 3])        # (n-1, m, 2)
            out = np.empty((len(segs) * (n - 1) + 1, 2))
            out[0] = C[0, 0]
            out[1:] = pts.transpose(1, 0, 2).reshape(-1, 2)
            return out
        # lines and curves mixed: the curves are evaluated together, then everything is put in order
        sizes = np.array([1 if k == "L" else n - 1 for k, _ in segs])
        ends = np.cumsum(sizes)
        out = np.empty((int(ends[-1]) + 1, 2))
        out[0] = segs[0][1][0]
        if ci:
            C = np.stack([segs[i][1] for i in ci])
            t = np.linspace(0.0, 1.0, n)[1:, None, None]
            m = 1.0 - t
            pts = (m ** 3 * C[None, :, 0] + 3 * m ** 2 * t * C[None, :, 1]
                   + 3 * m * t ** 2 * C[None, :, 2] + t ** 3 * C[None, :, 3])        # (n-1, curves, 2)
        j = 0
        for i, (kind, c) in enumerate(segs):
            e = int(ends[i]) + 1
            if kind == "L":
                out[e - 1] = c[1]
            else:
                out[e - (n - 1):e] = pts[:, j]
                j += 1
        return out


def polyline_length(pts: np.ndarray, closed: bool = False) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.hypot(*np.diff(pts, axis=0).T).sum()
    if closed:
        d += float(np.hypot(*(pts[0] - pts[-1])))
    return float(d)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


# --------------------------------------------------------------- primitives
def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Douglas-Peucker; returns kept indices."""
    n = len(pts)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        a, b = pts[s], pts[e]
        d = b - a
        L = float(np.hypot(*d))
        seg = pts[s + 1:e]
        if L < 1e-9:
            dist = np.hypot(*(seg - a).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / L
        i = int(np.argmax(dist))
        if dist[i] > eps:
            k = s + 1 + i
            keep[k] = True
            stack.append((s, k))
            stack.append((k, e))
    return np.nonzero(keep)[0]


def fit_line(pts: np.ndarray):
    """PCA line. Returns (centre, direction, max perpendicular residual, p_start, p_end)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    t = (pts - c) @ d
    res = np.abs((pts - c) @ np.array([-d[1], d[0]]))
    return c, d, float(res.max()), c + d * t.min(), c + d * t.max()


def classify_straight(pts: np.ndarray, tol: float) -> bool:
    """True if all points lie within `tol` px of their best-fit line."""
    if len(pts) < 2:
        return False
    if len(pts) == 2:
        return True
    return fit_line(pts)[2] <= tol


def bezier_points(c: np.ndarray, n: int = 16) -> np.ndarray:
    return _bez(c, np.linspace(0, 1, n))


def _bez(c, t):
    t = np.asarray(t)[:, None]
    m = 1 - t
    return m**3 * c[0] + 3 * m**2 * t * c[1] + 3 * m * t**2 * c[2] + t**3 * c[3]


def _bez_d1(c, t):
    t = t[:, None]
    m = 1 - t
    return 3 * (m**2 * (c[1] - c[0]) + 2 * m * t * (c[2] - c[1]) + t**2 * (c[3] - c[2]))


def _bez_d2(c, t):
    t = t[:, None]
    m = 1 - t
    return 6 * (m * (c[2] - 2 * c[1] + c[0]) + t * (c[3] - 2 * c[2] + c[1]))


def _chord_param(pts):
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    return d / d[-1] if d[-1] > 1e-12 else np.linspace(0, 1, len(pts))


def _gen_bezier(pts, u, t1, t2):
    p0, p3 = pts[0], pts[-1]
    b1 = 3 * (1 - u) ** 2 * u
    b2 = 3 * (1 - u) * u**2
    A1, A2 = t1[None, :] * b1[:, None], t2[None, :] * b2[:, None]
    C = np.array([[(A1 * A1).sum(), (A1 * A2).sum()], [(A1 * A2).sum(), (A2 * A2).sum()]])
    tmp = pts - (p0 * ((1 - u) ** 3 + b1)[:, None] + p3 * (b2 + u**3)[:, None])
    X = np.array([(A1 * tmp).sum(), (A2 * tmp).sum()])
    det = C[0, 0] * C[1, 1] - C[0, 1] ** 2
    seg = float(np.hypot(*(p3 - p0)))
    if abs(det) > 1e-12:
        a1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        a2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
    else:
        a1 = a2 = seg / 3
    if a1 < 1e-6 * seg or a2 < 1e-6 * seg:
        a1 = a2 = seg / 3
    return np.array([p0, p0 + t1 * a1, p3 + t2 * a2, p3])


def _max_err(pts, ctrl, u):
    d = np.sum((_bez(ctrl, u) - pts) ** 2, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), min(max(i, 1), len(pts) - 2)


def _reparam(pts, u, ctrl):
    d = _bez(ctrl, u) - pts
    d1, d2 = _bez_d1(ctrl, u), _bez_d2(ctrl, u)
    num = (d * d1).sum(1)
    den = (d1 * d1).sum(1) + (d * d2).sum(1)
    den = np.where(np.abs(den) < 1e-12, 1.0, den)
    u2 = np.clip(u - num / den, 0, 1)
    u2[0], u2[-1] = 0.0, 1.0
    return np.maximum.accumulate(u2)


def fit_curve(pts: np.ndarray, tol: float, t1=None, t2=None, depth: int = 0) -> List[np.ndarray]:
    """Schneider piecewise cubic Bezier fit. Returns a list of (4,2) control arrays."""
    n = len(pts)
    if n == 2:
        a, b = pts
        return [np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])]
    t1 = unit(pts[min(3, n - 1)] - pts[0]) if t1 is None else t1
    t2 = unit(pts[max(n - 4, 0)] - pts[-1]) if t2 is None else t2
    u = _chord_param(pts)
    ctrl = _gen_bezier(pts, u, t1, t2)
    err, split = _max_err(pts, ctrl, u)
    if err < tol * tol:
        return [ctrl]
    if err < (tol * 6) ** 2:
        for _ in range(5):
            u = _reparam(pts, u, ctrl)
            ctrl = _gen_bezier(pts, u, t1, t2)
            err, split = _max_err(pts, ctrl, u)
            if err < tol * tol:
                return [ctrl]
    if n <= 4 or depth > 14:
        return [ctrl]
    c = unit(pts[split - 1] - pts[split + 1])
    return (fit_curve(pts[:split + 1], tol, t1, c, depth + 1)
            + fit_curve(pts[split:], tol, -c, t2, depth + 1))


def smooth_pts(pts: np.ndarray, closed: bool, w: int = 5) -> np.ndarray:
    """Moving-average de-jaggying; open chains keep their end points."""
    if len(pts) < w + 2:
        return pts.astype(np.float64)
    pad = w // 2
    p = np.pad(pts.astype(np.float64), ((pad, pad), (0, 0)), mode="wrap" if closed else "edge")
    k = np.ones(w) / w
    out = np.stack([np.convolve(p[:, i], k, mode="valid") for i in (0, 1)], 1)
    if not closed:
        out[0], out[-1] = pts[0], pts[-1]
    return out


def find_corners(pts: np.ndarray, closed: bool, k: int = 4, thr: float = 55.0) -> List[int]:
    """Indices where the turning angle over +-k samples is a local maximum above thr."""
    n = len(pts)
    if n < 2 * k + 1:
        return []
    if closed:
        a, b = pts - np.roll(pts, k, 0), np.roll(pts, -k, 0) - pts
    else:
        a = np.zeros_like(pts)
        b = np.zeros_like(pts)
        a[k:] = pts[k:] - pts[:-k]      # a[i] = pts[i]   - pts[i-k]
        b[:-k] = pts[k:] - pts[:-k]     # b[i] = pts[i+k] - pts[i]
    na, nb = np.hypot(*a.T), np.hypot(*b.T)
    cos = (a * b).sum(1) / np.maximum(na * nb, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    if not closed:
        ang[:k] = 0
        ang[n - k:] = 0
    peak = ang == maximum_filter1d(ang, size=2 * k + 1, mode="wrap" if closed else "nearest")
    idx = np.nonzero(peak & (ang > thr))[0]
    out: List[int] = []
    for i in idx:
        if not out or i - out[-1] > k:
            out.append(int(i))
    return out


def _central(P: np.ndarray, i: int, cyclic: bool, k: int = 3) -> np.ndarray:
    n = len(P)
    if cyclic:
        return unit(P[(i + k) % n] - P[(i - k) % n])
    return unit(P[min(i + k, n - 1)] - P[max(i - k, 0)])


def vectorise_chain(ch: Chain, tol: float = 1.5, line_tol: float = 1.2,
                    mode: str = "bezier") -> "Path | None":
    """Chain -> Path of exact straight segments and smooth cubic Beziers."""
    pts = ch.pts.astype(np.float64)
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-6] if len(pts) > 1 else np.array([True])
    pts = pts[keep]
    if len(pts) < 2:
        return None
    if ch.fixed or (len(pts) == 2 and not ch.closed):
        return Path([("L", np.array([pts[0], pts[-1]]))], False, ch.layer)
    if mode == "fast":
        return fit_fast(pts, ch.closed and len(pts) >= 4, tol=tol, line_tol=line_tol, layer=ch.layer)
    closed = ch.closed and len(pts) >= 4
    if mode == "polyline":
        P = np.vstack([pts, pts[:1]]) if closed else pts
        idx = rdp(P, tol)
        return Path([("L", np.array([P[a], P[b]])) for a, b in zip(idx[:-1], idx[1:])],
                    closed, ch.layer)
    pts = smooth_pts(pts, closed)
    n = len(pts)
    corners = find_corners(pts, closed)
    if closed:
        if corners:
            pts = np.roll(pts, -corners[0], axis=0)
            cuts = [c - corners[0] for c in corners] + [n]
            corner_set = set(cuts)
        else:
            cuts, corner_set = [0, n // 2, n], set()
        P = np.vstack([pts, pts[:1]])
    else:
        cuts = [0] + [c for c in corners if 0 < c < n - 1] + [n - 1]
        corner_set = set(cuts)
        P = pts
    segs: List[Seg] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        sec = P[a:b + 1]
        if len(sec) < 2:
            continue
        if len(sec) <= 3 or classify_straight(sec, line_tol):
            if len(sec) > 2:
                pass
            p0, p1 = sec[0], sec[-1]  # keep joints exact
            segs.append(("L", np.array([p0, p1])))
            continue
        t1 = unit(sec[min(3, len(sec) - 1)] - sec[0]) if a in corner_set or not closed \
            else _central(pts, a % n, True)
        t2 = unit(sec[max(len(sec) - 4, 0)] - sec[-1]) if b in corner_set or not closed \
            else -_central(pts, b % n, True)
        for c in fit_curve(sec, tol, t1, t2):
            segs.append(("B", c))
    if not segs:
        return None
    return Path(segs, closed, ch.layer)


# ------------------------------------------------------ straight-line merging
def merge_collinear(segs: np.ndarray, ang_tol: float = 2.0, off_tol: float = 2.0,
                    gap_tol: float = 20.0) -> np.ndarray:
    """Greedily merge near-collinear, near-touching segments (N,4)=x0,y0,x1,y1."""
    if len(segs) == 0:
        return segs
    S = segs.astype(np.float64).copy()
    alive = np.ones(len(S), bool)
    cos_tol = np.cos(np.radians(ang_tol))
    changed = True
    while changed:
        changed = False
        order = np.argsort(-np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]))
        for i in order:
            if not alive[i]:
                continue
            while True:
                p0, p1 = S[i, :2], S[i, 2:]
                d = unit(p1 - p0)
                nrm = np.array([-d[1], d[0]])
                idx = np.nonzero(alive)[0]
                idx = idx[idx != i]
                if len(idx) == 0:
                    break
                q0, q1 = S[idx, :2], S[idx, 2:]
                dq = q1 - q0
                lq = np.maximum(np.hypot(*dq.T), 1e-9)
                par = np.abs((dq @ d) / lq) >= cos_tol
                o0 = np.abs((q0 - p0) @ nrm)
                o1 = np.abs((q1 - p0) @ nrm)
                near = par & (o0 <= off_tol) & (o1 <= off_tol)
                ti = np.sort([0.0, (p1 - p0) @ d])
                tq0, tq1 = (q0 - p0) @ d, (q1 - p0) @ d
                lo, hi = np.minimum(tq0, tq1), np.maximum(tq0, tq1)
                gap = np.maximum(lo - ti[1], ti[0] - hi)
                sel = idx[near & (gap <= gap_tol)]
                if len(sel) == 0:
                    break
                allp = np.vstack([p0, p1, S[sel, :2], S[sel, 2:]])
                t = (allp - p0) @ d
                S[i] = np.r_[allp[np.argmin(t)], allp[np.argmax(t)]]
                alive[sel] = False
                changed = True
    return S[alive]


# ------------------------------------------------------------ chain operations
def _endpoint_tangents(c: Chain, k: int = 6):
    p = c.pts
    k = min(k, len(p) - 1)
    return unit(p[0] - p[k]), unit(p[-1] - p[-1 - k])


def join_chains(chains: Sequence[Chain], gap: float = 4.0, ang_tol: float = 35.0,
                max_rounds: int = 30) -> List[Chain]:
    """Link end-to-end chains with compatible tangents; close self-closing chains."""
    rest = [c for c in chains if c.fixed or c.closed or len(c.pts) < 2]
    cur = [c for c in chains if not (c.fixed or c.closed or len(c.pts) < 2)]
    cos_lim = -np.cos(np.radians(ang_tol))
    for _ in range(max_rounds):
        m = len(cur)
        if m < 2:
            break
        ends = np.empty((2 * m, 2))
        tans = np.empty((2 * m, 2))
        for i, c in enumerate(cur):
            ends[2 * i], ends[2 * i + 1] = c.pts[0], c.pts[-1]
            tans[2 * i], tans[2 * i + 1] = _endpoint_tangents(c)
        pairs = cKDTree(ends).query_pairs(gap, output_type="ndarray")
        cands = []
        for a, b in pairs:
            if a // 2 == b // 2:
                continue
            cos = float(tans[a] @ tans[b])
            if cos > cos_lim:
                continue
            cands.append((float(np.hypot(*(ends[a] - ends[b]))) + 5 * (1 + cos), int(a), int(b)))
        if not cands:
            break
        cands.sort()
        used, merged, merges = set(), [], []
        for _, a, b in cands:
            if a // 2 in used or b // 2 in used:
                continue
            used.update((a // 2, b // 2))
            merges.append((a, b))
        for a, b in merges:
            A, B = cur[a // 2], cur[b // 2]
            pa = A.pts[::-1] if a % 2 == 0 else A.pts
            pb = B.pts[::-1] if b % 2 == 1 else B.pts
            layer = A.layer if A.length() >= B.length() else B.layer
            merged.append(Chain(np.vstack([pa, pb]), False, layer))
        cur = [c for i, c in enumerate(cur) if i not in used] + merged
    out = list(rest)
    for c in cur:
        if len(c.pts) >= 10 and np.hypot(*(c.pts[0] - c.pts[-1])) <= gap * 1.5 \
                and c.length() > 6 * gap:
            t0, t1 = _endpoint_tangents(c)
            if float(t0 @ t1) < cos_lim:
                out.append(Chain(c.pts, True, c.layer))
                continue
        out.append(c)
    return out


def _densify(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return pts
    steps = np.hypot(*np.diff(pts, axis=0).T)
    if steps.max() <= 1.5:
        return pts
    out = [pts[:1]]
    for a, b, s in zip(pts[:-1], pts[1:], steps):
        k = max(int(np.ceil(s)), 1)
        out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.vstack(out)


def dedup_groups(groups: Sequence[Tuple[Sequence[Chain], int]], shape: Tuple[int, int],
                 min_len: float) -> List[Chain]:
    """Priority de-duplication.

    `groups` is ordered high -> low priority; each is (chains, tolerance_px).
    Pixels of a chain already covered (within tolerance) by an accepted chain are cut
    out, so exact straight lines win over edge pixels, and no line is drawn twice.
    """
    H, W = shape
    occ = np.zeros((H, W), np.uint8)
    out: List[Chain] = []
    for chains, r in groups:
        thick = 2 * r + 1
        for ch in chains:
            P = _densify(ch.pts.astype(np.float64))
            xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
            free = occ[yi, xi] == 0
            pieces: List[Chain] = []
            if free.all():
                pieces.append(ch)
            elif free.any():
                edges = np.diff(np.r_[0, free.view(np.int8), 0])
                for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
                    if e - s >= 2:
                        pieces.append(Chain(P[s:e], False, ch.layer, ch.fixed))
            for pc in pieces:
                if pc.length() < min_len:
                    continue
                out.append(pc)
                q = np.rint(_densify(pc.pts.astype(np.float64))).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(occ, [q], pc.closed, 255, thick)
    return out


# ------------------------------------------------------------------ fast fitter
def _unitv(v: np.ndarray) -> np.ndarray:
    n = np.hypot(v[:, 0], v[:, 1])[:, None]
    return v / np.maximum(n, 1e-12)


def _key(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32).astype(np.float64)
    return np.rint(a[:, 0] * 8).astype(np.int64) * 1000003 + np.rint(a[:, 1] * 8).astype(np.int64)


def fit_fast(P: np.ndarray, closed: bool, tol: float = 1.2, corner_thr: float = 55.0,
             smooth_w: int = 3, line_tol: "float | None" = None, snap_lines: bool = True,
             dp_scale: float = 10.0, layer: str = "EDGES") -> "Path | None":
    """Vectorised hybrid fitter: coarse Douglas-Peucker vertices + adaptive refinement.

    Corners are forced in; tangents come from the original samples; a segment within
    `line_tol` of its chord becomes an exact line, otherwise a cubic Bezier with
    arc-length handles.  Vertices are inserted where the curve strays more than `tol`
    (a few vectorised rounds, no per-segment Python recursion), so anchor counts stay
    low while running ~5-10x faster than `fit_curve` on contours with many features.
    """
    P = np.asarray(P, np.float64)
    if len(P) > 1:
        P = P[np.r_[True, np.abs(np.diff(P, axis=0)).sum(1) > 1e-9]]
        if closed and len(P) > 1 and np.abs(P[0] - P[-1]).sum() < 1e-9:
            P = P[:-1]
    n = len(P)
    if n < 2 or (closed and n < 3):
        return None
    if n == 2:
        return Path([("L", P.copy())], False, layer)
    line_tol = 0.4 * tol if line_tol is None else line_tol
    Ps = smooth_pts(P, closed, smooth_w) if smooth_w > 1 and n > smooth_w + 2 else P
    corners = np.asarray(find_corners(Ps, closed, 4, corner_thr), dtype=np.int64)
    ap = cv2.approxPolyDP(Ps.astype(np.float32).reshape(-1, 1, 2), max(tol * dp_scale, 0.05),
                          closed)[:, 0, :]
    idx = np.union1d(np.nonzero(np.isin(_key(Ps), _key(ap)))[0], corners)
    if not closed:
        idx = np.union1d(idx, [0, n - 1])
    if len(idx) < (3 if closed else 2):
        idx = np.unique(np.linspace(0, n - 1, 4 if closed else 2).astype(np.int64))
    if closed:                      # work in a frame whose first vertex is index 0
        r = int(idx[0])
        Q, idx, corners = np.roll(Ps, -r, 0), idx - r, (corners - r) % n
    else:
        Q = Ps
    ext = np.vstack([Q, Q[:1]]) if closed else Q
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ext, axis=0).T))]
    cmask = np.zeros(n, bool)
    cmask[corners] = True
    pos = np.arange(n)
    t = 3

    for _ in range(7):
        m = len(idx)
        ns = m if closed else m - 1
        V = Q[idx]
        j = (idx[:, None] + np.array([t, -t])[None, :]) % n if closed else \
            np.clip(idx[:, None] + np.array([t, -t])[None, :], 0, n - 1)
        fw, bk = Q[j[:, 0]], Q[j[:, 1]]
        fwd, bwd, cen = _unitv(fw - V), _unitv(V - bk), _unitv(fw - bk)
        is_c = cmask[idx].copy()
        if not closed:
            is_c[0] = is_c[-1] = True
        t_out = np.where(is_c[:, None], fwd, cen)
        t_in = np.where(is_c[:, None], bwd, cen)
        ends = np.r_[idx[1:], n] if closed else idx[1:]
        s0 = idx[:ns]
        arc = cum[ends] - cum[s0]
        j1 = (np.arange(ns) + 1) % m
        A, B = V[:ns], V[j1]
        chord = np.hypot(*(B - A).T)
        h = (np.maximum(arc, chord) / 3.0)[:, None]
        C1, C2 = A + t_out[:ns] * h, B - t_in[j1] * h

        seg = np.minimum(np.searchsorted(idx, pos, side="right") - 1, ns - 1)
        a, b = A[seg], B[seg]
        ab = b - a
        L = np.hypot(ab[:, 0], ab[:, 1])
        dline = np.where(L > 1e-9, np.abs(ab[:, 0] * (Q[:, 1] - a[:, 1]) - ab[:, 1] * (Q[:, 0] - a[:, 0]))
                         / np.maximum(L, 1e-9), np.hypot(Q[:, 0] - a[:, 0], Q[:, 1] - a[:, 1]))
        devl = np.maximum.reduceat(dline, idx[:ns])
        straight = (devl <= line_tol) if snap_lines else np.zeros(ns, bool)
        u = np.clip((cum[:n] - cum[idx[seg]]) / np.maximum(arc[seg], 1e-9), 0, 1)[:, None]
        mu = 1 - u
        curve = mu**3 * a + 3 * mu**2 * u * C1[seg] + 3 * mu * u**2 * C2[seg] + u**3 * b
        ec = np.hypot(*(curve - Q).T)
        segerr = np.where(straight, 0.0, np.maximum.reduceat(ec, idx[:ns]))
        if snap_lines is False:
            segerr = np.maximum.reduceat(ec, idx[:ns])
        bad = np.nonzero((segerr > tol * 1.7) & (np.diff(np.r_[idx[:ns], n if closed else idx[-1]]) >= 4)
                         if closed else (segerr > tol * 1.7) & (np.diff(idx) >= 4))[0]
        if len(bad) == 0:
            break
        worst = ec.copy()
        worst[~np.isin(seg, bad)] = -1
        worst[idx] = -1
        pick = []
        for sgi in bad:                 # one new vertex per offending segment
            lo = idx[sgi]
            hi = ends[sgi]
            w = worst[lo:hi] if hi > lo else worst[lo:]
            pick.append(lo + int(np.argmax(w)))
        idx = np.union1d(idx, np.asarray(pick, np.int64))

    segs: List[Seg] = []
    for k in range(ns):
        if straight[k]:
            segs.append(("L", np.array([A[k], B[k]])))
        else:
            segs.append(("B", np.array([A[k], C1[k], C2[k], B[k]])))
    return Path(segs, closed, layer)


def fit_tiny(P: np.ndarray, tol: float, layer: str = "FILL", smooth: bool = True) -> "Path | None":
    """Closed-form fit for small closed rings (<= ~24 samples): Catmull-Rom spline through
    Douglas-Peucker vertices.  No iteration, ~1/10 the cost of `fit_fast`."""
    P = np.asarray(P, np.float64)
    if len(P) > 1 and np.abs(P[0] - P[-1]).sum() < 1e-9:
        P = P[:-1]
    if len(P) < 3:
        return None
    ap = cv2.approxPolyDP(P.astype(np.float32).reshape(-1, 1, 2), max(tol * 0.6, 0.05), True)[:, 0, :]
    V = ap.astype(np.float64)
    m = len(V)
    if m < 3:
        return None
    if m <= 3 or not smooth:
        return Path([("L", np.array([V[i], V[(i + 1) % m]])) for i in range(m)], True, layer)
    idx = np.arange(m)
    nxt, prv = V[(idx + 1) % m], V[(idx - 1) % m]
    T = (nxt - prv) / 6.0
    ctrl = np.empty((m, 4, 2))                     # all Bezier segments in one array (the segments are views)
    ctrl[:, 0] = V
    ctrl[:, 1] = V + T
    ctrl[:, 2] = nxt - T[(idx + 1) % m]
    ctrl[:, 3] = nxt
    segs: List[Seg] = [("B", ctrl[i]) for i in range(m)]
    return Path(segs, True, layer)
