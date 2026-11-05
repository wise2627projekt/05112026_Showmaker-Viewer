"""Linework Snap: raster line drawings (e.g. a rasterised PDF) -> connected vector paths.

Pipeline: ink map (distance from the paper colour) -> mask -> skeleton -> *graph* with exact
junction nodes -> sub-pixel centre-line refinement -> spur/speck removal -> join through
junctions -> gap closing + end snapping onto neighbouring lines -> hybrid line/Bezier fit
with a user-controlled simplification level.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .geometry import Chain, Path, fit_fast, join_chains, polyline_length
from .trace import TraceParams, TraceResult


@dataclass
class LineworkParams:
    threshold: int = 0              # 0 = automatic (Otsu); else ink level 10..250
    min_len: float = 6.0            # px: drop specks and dead-end spurs shorter than this
    gap: float = 3.0                # px: close gaps / snap loose ends onto nearby lines
    simplify: int = 30              # 0..100 %  (path tolerance)
    corners: int = 60               # 0..100 %
    join_through: bool = True       # merge collinear lines through junctions
    snap_lines: bool = True         # straight within tolerance -> exact LINE
    width_mode: str = "match"       # match | uniform | fixed
    fixed_width: float = 1.0
    color_mode: str = "black"       # black | source
    max_side: int = 4000            # 0 = full resolution
    scale: float = 1.0              # px-based settings shrink for small tiles (see batch.tile_params)

    @property
    def fit_tol(self) -> float:     # 0 % -> 0.35 px ... 100 % -> 6 px
        return max(0.12, (0.35 + 5.65 * (self.simplify / 100.0) ** 1.6) * self.scale)

    @property
    def corner_angle(self) -> float:
        return 120.0 - 90.0 * self.corners / 100.0


PRESETS: Dict[str, dict] = {
    "[Default]": {},
    "Exact (keep detail)": dict(simplify=5, min_len=3, gap=2, join_through=True, corners=80),
    "Clean drawing": dict(simplify=35, min_len=8, gap=4, join_through=True, corners=60),
    "Simplified (few nodes)": dict(simplify=75, min_len=14, gap=6, join_through=True, corners=40),
}


# --------------------------------------------------------------------------- ink / mask
def paper_color(bgr: np.ndarray, opaque: Optional[np.ndarray]) -> np.ndarray:
    px = bgr[opaque] if opaque is not None and opaque.any() else bgr.reshape(-1, 3)
    if len(px) > 200000:
        px = px[:: len(px) // 200000]
    return np.median(px, axis=0)


def ink_map(bgr: np.ndarray, opaque: Optional[np.ndarray]) -> np.ndarray:
    """0 = paper, 255 = full-strength line (max channel distance from the paper colour)."""
    if opaque is not None and opaque.mean() < 0.5:
        # lines-only PNG on a transparent page: the opaque pixels *are* the ink
        return opaque.astype(np.uint8) * 255
    bg = paper_color(bgr, opaque)
    ink = np.abs(bgr.astype(np.int16) - bg.astype(np.int16)).max(2).astype(np.uint8)
    if opaque is not None:
        ink[~opaque] = 0
    return ink


def looks_like_linework(bgr: np.ndarray, opaque: Optional[np.ndarray] = None) -> bool:
    """Mostly paper with thin marks: used to suggest the Linework engine after a paste."""
    ink = ink_map(bgr, opaque)
    frac = float((ink > 60).mean())
    return 0.0005 < frac < 0.12


# --------------------------------------------------------------------------- skeleton graph
@dataclass
class _Edge:
    pts: np.ndarray                 # (n,2) float, corner-frame coords (pixel centre = i + .5)
    a: int = -1                     # node id at start (-1 = free end)
    b: int = -1                     # node id at end
    closed: bool = False


def skeleton_graph(skel: np.ndarray, ink: np.ndarray):
    """Skeleton -> (edges, node_xy).  Junction pixel clusters become nodes (their centroid);
    every edge that touches a cluster ends *exactly* on its node, so lines meet precisely."""
    H, W = skel.shape
    sk = skel.astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    J = (sk > 0) & (nb >= 3)
    E = (sk > 0) & ~J
    nj, Jlab = cv2.connectedComponents(J.astype(np.uint8), connectivity=8)
    node_xy = np.zeros((max(nj, 1), 2))
    if nj > 1:
        ys, xs = np.nonzero(Jlab)
        lab = Jlab[ys, xs]
        cnt = np.maximum(np.bincount(lab, minlength=nj), 1)
        node_xy[:, 0] = np.bincount(lab, weights=xs, minlength=nj) / cnt + 0.5
        node_xy[:, 1] = np.bincount(lab, weights=ys, minlength=nj) / cnt + 0.5
    near = ndi.maximum_filter(Jlab, size=3)

    ys, xs = np.nonzero(E)
    N = len(ys)
    if N == 0:
        return [], node_xy
    idx = np.full((H + 2, W + 2), -1, np.int32)
    idx[ys + 1, xs + 1] = np.arange(N)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    cand = np.stack([idx[ys + 1 + dy, xs + 1 + dx] for dy, dx in offs], 1)
    valid = cand >= 0
    order = np.argsort(~valid, axis=1, kind="stable")[:, :2]
    nbr = np.take_along_axis(cand, order, 1)
    nbr[~np.take_along_axis(valid, order, 1)] = -1
    deg = valid.sum(1)

    # sub-pixel centre refinement: ink-weighted centroid over the 3x3 window
    padi = np.pad(ink.astype(np.float32), 1)
    wsum = np.zeros(N, np.float32)
    cx = np.zeros(N, np.float32)
    cy = np.zeros(N, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = padi[ys + 1 + dy, xs + 1 + dx]
            wsum += w
            cx += w * dx
            cy += w * dy
    wsum = np.maximum(wsum, 1e-6)
    R = np.stack([xs + 0.5 + np.clip(cx / wsum, -0.5, 0.5), ys + 0.5 + np.clip(cy / wsum, -0.5, 0.5)], 1)

    nc, comp = cv2.connectedComponents(E.astype(np.uint8), connectivity=8)
    lp = comp[ys, xs]
    srt = np.argsort(lp, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(lp[srt]))[0] + 1, N]
    n1, n2, degl = nbr[:, 0].tolist(), nbr[:, 1].tolist(), deg.tolist()
    visited = bytearray(N)
    edges: List[_Edge] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ids = srt[a:b].tolist()
        start = next((i for i in ids if degl[i] <= 1), None)
        closed = start is None
        cur = ids[0] if closed else start
        path, prev = [cur], -1
        visited[cur] = 1
        while True:
            x, y = n1[cur], n2[cur]
            nxt = x if (x != -1 and x != prev) else (y if (y != -1 and y != prev) else -1)
            if nxt == -1 or visited[nxt]:
                break
            visited[nxt] = 1
            prev, cur = cur, nxt
            path.append(cur)
        if len(path) < 2 and not (nc > 0 and nj > 1):
            continue
        pts = R[path]
        e = _Edge(pts, closed=closed and len(path) >= 6)
        if not closed:
            for end in (0, 1):
                px, py = (xs[path[0]], ys[path[0]]) if end == 0 else (xs[path[-1]], ys[path[-1]])
                lab = int(near[py, px])
                if lab > 0:
                    if end == 0:
                        e.a = lab
                        e.pts = np.vstack([node_xy[lab], e.pts])
                    else:
                        e.b = lab
                        e.pts = np.vstack([e.pts, node_xy[lab]])
        edges.append(e)
    return edges, node_xy


def _edge_len(e: _Edge) -> float:
    return polyline_length(e.pts, e.closed)


def prune(edges: List[_Edge], min_len: float) -> List[_Edge]:
    """Drop isolated specks and dead-end spurs shorter than min_len."""
    out = []
    for e in edges:
        L = _edge_len(e)
        dead_end = (e.a < 0) != (e.b < 0)          # exactly one end free
        isolated = e.a < 0 and e.b < 0
        if (isolated and L < min_len) or (dead_end and L < min_len):
            continue
        out.append(e)
    return out


def _leave_dir(e: _Edge, end: int, k: int = 8) -> np.ndarray:
    p = e.pts
    k = min(k, len(p) - 1)
    v = (p[k] - p[0]) if end == 0 else (p[-1 - k] - p[-1])
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def join_through_nodes(edges: List[_Edge], angle: float = 25.0) -> List[Chain]:
    """Merge edges that continue straight through a junction; return chains (pts, closed)."""
    cos_lim = -np.cos(np.radians(angle))
    at: Dict[int, List[Tuple[int, int]]] = {}
    for i, e in enumerate(edges):
        if e.closed:
            continue
        if e.a > 0:
            at.setdefault(e.a, []).append((i, 0))
        if e.b > 0:
            at.setdefault(e.b, []).append((i, 1))
    partner: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for node, ends in at.items():
        dirs = [_leave_dir(edges[i], s) for i, s in ends]
        cands = []
        for p in range(len(ends)):
            for q in range(p + 1, len(ends)):
                if ends[p][0] == ends[q][0]:
                    continue
                d = float(dirs[p] @ dirs[q])
                if d <= (cos_lim if len(ends) > 2 else -0.3):
                    cands.append((d, p, q))
        cands.sort()
        used = set()
        for d, p, q in cands:
            if p in used or q in used:
                continue
            used.update((p, q))
            partner[ends[p]] = ends[q]
            partner[ends[q]] = ends[p]
    seen = set()
    chains: List[Chain] = []

    def walk(e0: int, s0: int):
        parts, cur, closed = [], (e0, s0), False
        while True:
            e, s = cur
            if e in seen:
                closed = (e == e0)
                break
            seen.add(e)
            P = edges[e].pts if s == 0 else edges[e].pts[::-1]
            parts.append(P)
            nxt = partner.get((e, 1 - s))
            if nxt is None:
                break
            cur = nxt
        pts = parts[0]
        for P in parts[1:]:
            pts = np.vstack([pts, P[1:] if np.allclose(pts[-1], P[0], atol=1e-6) else P])
        return pts, closed

    for i, e in enumerate(edges):           # chains that start at a free end first
        if i in seen or e.closed:
            continue
        free = [s for s in (0, 1) if (i, s) not in partner]
        if free:
            pts, closed = walk(i, free[0])
            chains.append(Chain(pts, False))
    for i, e in enumerate(edges):           # closed loops through junctions / plain rings
        if i in seen:
            continue
        if e.closed:
            seen.add(i)
            chains.append(Chain(e.pts, True))
        else:
            pts, closed = walk(i, 0)
            if closed and np.allclose(pts[0], pts[-1], atol=1e-6):
                pts = pts[:-1]
            chains.append(Chain(pts, closed))
    return chains


def snap_ends(chains: List[Chain], gap: float) -> List[Chain]:
    """Extend loose ends that nearly touch another line onto that line (T-junction closure)."""
    open_ch = [c for c in chains if not c.closed and len(c.pts) >= 2]
    if gap <= 0 or not chains:
        return chains
    allpts = np.vstack([c.pts for c in chains])
    owner = np.concatenate([np.full(len(c.pts), i) for i, c in enumerate(chains)])
    tree = cKDTree(allpts)
    pos = {id(c): i for i, c in enumerate(chains)}
    for c in open_ch:
        me = pos[id(c)]
        for end in (0, 1):
            p = c.pts[0] if end == 0 else c.pts[-1]
            k = min(6, len(c.pts) - 1)
            t = (p - c.pts[k]) if end == 0 else (p - c.pts[-1 - k])
            tn = float(np.hypot(*t))
            if tn < 1e-9:
                continue
            t = t / tn
            best, bd = None, gap
            for j in tree.query_ball_point(p, gap):
                if owner[j] == me:
                    continue
                v = allpts[j] - p
                d = float(np.hypot(*v))
                if 0.5 < d <= bd and float(v @ t) / d > 0.5:   # ahead of the end, within the gap
                    best, bd = j, d
            if best is not None:
                c.pts = np.vstack([allpts[best], c.pts]) if end == 0 else np.vstack([c.pts, allpts[best]])
    return chains


def _quick_line(c: Chain, line_tol: float) -> Optional[Path]:
    """Cheap exact-line test (most hatch/wall segments are straight): one vector pass."""
    P = c.pts
    a, b = P[0], P[-1]
    ab = b - a
    L = float(np.hypot(*ab))
    if L < 1e-6:
        return None
    dev = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
    if dev.max() <= line_tol and float(((P - a) @ ab).min()) >= -line_tol * L:
        return Path([("L", np.array([a, b]))], False, "LINES")
    return None


# --------------------------------------------------------------------------- driver
def linework_bgr(bgr: np.ndarray, p: LineworkParams, opaque: Optional[np.ndarray] = None,
                 cache: Optional[dict] = None) -> TraceResult:
    """`cache`: per-image dict for sweeps (ink map and skeleton are reused while threshold / min length stay the same)."""
    t: Dict[str, float] = {}
    t0 = time.perf_counter()
    p_in = p
    p = replace(p, min_len=p.min_len * p.scale, gap=p.gap * p.scale)
    if cache is not None and "ink" in cache:
        ink = cache["ink"]
    else:
        ink = ink_map(bgr, opaque)
        if cache is not None:
            cache["ink"] = ink
    thr = p.threshold
    if thr <= 0:
        otsu, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(otsu * 0.7, 25, 200))
    mask = ink > thr
    if p.min_len > 1:      # specks smaller than a short line are paper dust
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        small = st[:, cv2.CC_STAT_AREA] < max(2, p.min_len * 0.5)
        small[0] = False
        mask &= ~small[lab]
    t["mask"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk_key = ("skel", thr, round(p.min_len, 4))
    if cache is not None and sk_key in cache:
        skel = cache[sk_key]
    else:
        skel = skeletonize(mask)
        if cache is not None:
            for old in [k for k in cache if isinstance(k, tuple) and k[0] == "skel"][:-2]:
                cache.pop(old)
            cache[sk_key] = skel
    t["skeleton"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    edges, _ = skeleton_graph(skel, ink)
    raw = len(edges)
    edges = prune(edges, p.min_len)
    t["graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if p.join_through:
        chains = join_through_nodes(edges)
    else:
        chains = [Chain(e.pts, e.closed) for e in edges]
    chains = join_chains(chains, gap=max(p.gap, 1.0), ang_tol=40.0) if p.gap > 0 else chains
    chains = snap_ends(chains, p.gap)
    t["join"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    H, W = mask.shape
    tol = p.fit_tol
    line_tol = max(0.7, 0.9 * tol) if p.snap_lines else 0.0
    sw = 3 if p.simplify < 34 else 5 if p.simplify < 67 else 7
    paths: List[Path] = []
    widths: List[float] = []
    for c in chains:
        if not c.closed and polyline_length(c.pts) < p.min_len:
            continue
        path = _quick_line(c, line_tol) if p.snap_lines and not c.closed else None
        if path is None:
            path = fit_fast(c.pts, c.closed, tol=tol, corner_thr=p.corner_angle, smooth_w=sw,
                            snap_lines=p.snap_lines, line_tol=line_tol, layer="LINES")
        if path is None:
            continue
        xi = np.clip(c.pts[:, 0].astype(int), 0, W - 1)
        yi = np.clip(c.pts[:, 1].astype(int), 0, H - 1)
        path.width = float(max(1.0, 2.0 * np.median(dt[yi, xi]) - 1.0))
        if p.color_mode == "source":
            col = np.median(bgr[yi, xi], axis=0)
            col = (np.clip(np.rint(col / 24.0), 0, 10) * 24 + 12).clip(0, 255).astype(int)
            path.rgb = (int(col[2]), int(col[1]), int(col[0]))
        else:
            path.rgb = (0, 0, 0)
        widths.append(path.width)
        paths.append(path)
    if widths:
        if p.width_mode == "uniform":
            u = float(np.median(widths))
            for q in paths:
                q.width = u
        elif p.width_mode == "fixed":
            for q in paths:
                q.width = p.fixed_width
    t["fit"] = time.perf_counter() - t0
    h, w = bgr.shape[:2]
    meta = {"raw_paths": raw, "threshold": thr}
    res = TraceResult([], paths, (w, h), bgr, [], t, TraceParams(), opaque, meta)
    res.lw_params = p_in          # the settings that made this result (for parameter sweeps)
    return res
