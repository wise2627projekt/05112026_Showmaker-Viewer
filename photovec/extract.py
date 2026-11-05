"""Cue extractors: foliage mask, straight lines, edges, tonal contours, centre lines."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import skeletonize

from .geometry import Chain, merge_collinear, polyline_length


def local_std(gray: np.ndarray, k: int = 7) -> np.ndarray:
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0))


def _grad_mag(gs: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def split_runs(pts: np.ndarray, good: np.ndarray, closed: bool, min_pts: int = 3) -> List[Chain]:
    """Split a point sequence into runs where `good` is True."""
    if good.all():
        return [Chain(pts, closed)] if len(pts) >= min_pts else []
    if not good.any():
        return []
    d = np.diff(np.r_[0, good.view(np.int8), 0])
    out = []
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if e - s >= min_pts:
            out.append(Chain(pts[s:e], False))
    return out


# --------------------------------------------------------------- skeleton trace
def trace_skeleton(skel: np.ndarray, min_len: float, layer: str = "EDGES") -> List[Chain]:
    """Trace a 1-px skeleton into ordered chains (junction pixels removed)."""
    sk = skel.astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    seg = (sk > 0) & (nb < 3)
    n, lab = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    ys, xs = np.nonzero(lab)
    ls = lab[ys, xs]
    order = np.argsort(ls, kind="stable")
    ys, xs, ls = ys[order], xs[order], ls[order]
    bounds = np.r_[0, np.nonzero(np.diff(ls))[0] + 1, len(ls)]
    out: List[Chain] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 3:
            continue
        pts = {(int(x), int(y)) for x, y in zip(xs[a:b], ys[a:b])}
        adj = {}
        for (x, y) in pts:
            adj[(x, y)] = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                           if (dx or dy) and (x + dx, y + dy) in pts]
        ends = [p for p, v in adj.items() if len(v) <= 1]
        closed = not ends
        cur = ends[0] if ends else next(iter(pts))
        path, seen = [cur], {cur}
        while True:
            nxt = [q for q in adj[cur] if q not in seen]
            if not nxt:
                break
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))  # prefer 4-neighbours
            cur = nxt[0]
            path.append(cur)
            seen.add(cur)
        arr = np.asarray(path, np.float32)
        if closed and len(arr) < 8:
            continue
        if polyline_length(arr) >= min_len:
            out.append(Chain(arr, closed, layer))
    return out


# --------------------------------------------------------------------- foliage
def foliage_mask(bgr: np.ndarray) -> np.ndarray:
    """Vegetation cue: ExG + yellow/green hue, AND local texture. bool (H,W).

    Stage 2 fuses this with SegFormer / open-vocabulary votes.
    """
    h, w = bgr.shape[:2]
    f = bgr.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    exg = 2 * g - r - b
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_ok = (hsv[..., 0] >= 22) & (hsv[..., 0] <= 95) & (hsv[..., 1] > 80) & (hsv[..., 2] > 40)
    color = (exg > 0.06) | hue_ok
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tex = cv2.GaussianBlur(local_std(gray, 7), (0, 0), 3) > 5.0
    m = (color & tex).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 6e-4 * h * w
    m = keep[lab]
    return ndi.binary_fill_holes(m) if m.any() else m


def mask_contours(mask: np.ndarray, min_area: float, layer: str = "FOLIAGE") -> List[Chain]:
    """Silhouette chains of a binary mask (frame border pixels removed)."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out: List[Chain] = []
    for c in cnts:
        if cv2.contourArea(c) < min_area or len(c) < 8:
            continue
        p = c[:, 0, :].astype(np.float32)
        good = (p[:, 0] > 1) & (p[:, 1] > 1) & (p[:, 0] < w - 2) & (p[:, 1] < h - 2)
        for ch in split_runs(p, good, True):
            ch.layer = layer
            out.append(ch)
    return out


def fraction_inside(pts: np.ndarray, mask: np.ndarray) -> float:
    h, w = mask.shape
    x = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return float(mask[y, x].mean())


# ------------------------------------------------------------ straight lines
def detect_lines(gray: np.ndarray, min_len: float, foliage: Optional[np.ndarray],
                 ang_tol: float = 2.0, off_tol: float = 2.0, gap_tol: float = 25.0) -> np.ndarray:
    """LSD -> merge collinear -> drop foliage-borne and short segments. (N,4)."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    s = np.asarray(lines, dtype=np.float64).reshape(-1, 4)   # (N,1,4) in OpenCV 4, (N,4) in 5
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= max(6.0, 0.35 * min_len)]
    if foliage is not None and len(s):
        # cheap vectorised test BEFORE the quadratic merge: foliage yields thousands of
        # tiny segments that would otherwise dominate the run time.
        h, w = foliage.shape
        inside = np.zeros(len(s))
        for t in (0.15, 0.5, 0.85):
            x = np.clip(np.rint(s[:, 0] + (s[:, 2] - s[:, 0]) * t).astype(int), 0, w - 1)
            y = np.clip(np.rint(s[:, 1] + (s[:, 3] - s[:, 1]) * t).astype(int), 0, h - 1)
            inside += foliage[y, x]
        keep = inside < 2          # at most one of three samples inside the mask
        s = s[keep]
    if len(s) > 3000:  # pathological texture: keep the longest, bound the O(N^2) merge
        L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
        s = s[np.argsort(-L)[:3000]]
    s = merge_collinear(s, ang_tol, off_tol, gap_tol)
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= min_len]
    return s


def line_structure_strength(segs: np.ndarray, long_side: int) -> float:
    """Total length of long straight segments, in units of the image long side."""
    if len(segs) == 0:
        return 0.0
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return float(L[L >= 0.05 * long_side].sum() / long_side)


# ----------------------------------------------------------------- edge chains
def edge_chains(gray: np.ndarray, scales: Sequence[float], pct: float, smooth_std: float,
                min_len: float) -> List[Chain]:
    """Multi-scale Canny (percentile thresholds) -> thin -> suppress smooth areas -> trace."""
    sd = local_std(gray, 7)
    acc = np.zeros(gray.shape, np.uint8)
    for i, s in enumerate(scales):
        g = cv2.GaussianBlur(gray, (0, 0), s)
        hi = max(float(np.percentile(_grad_mag(g.astype(np.float32)) * 8.0, pct + 2.0 * (i == 0))), 24.0)
        acc |= cv2.Canny(g, 0.4 * hi, hi, L2gradient=True)
    acc[sd < smooth_std] = 0
    acc = cv2.dilate(acc, np.ones((3, 3), np.uint8))
    skel = skeletonize(acc > 0)
    return trace_skeleton(skel, min_len, "EDGES")


def ridge_chains(gray: np.ndarray, sigmas: Sequence[float], pct: float, min_len: float,
                 smooth_std: float, bright_too: bool) -> List[Chain]:
    """Centre lines of thin dark (and optionally bright) ridges: cracks, twigs, joints."""
    g = gray.astype(np.float32) / 255.0
    r = sato(g, sigmas=list(sigmas), black_ridges=True)
    if bright_too:
        r = np.maximum(r, sato(g, sigmas=list(sigmas), black_ridges=False))
    hi = max(float(np.percentile(r, pct)), 0.012)
    m = apply_hysteresis_threshold(r, 0.5 * hi, hi)
    m &= local_std(gray, 7) >= smooth_std
    skel = skeletonize(m)
    return trace_skeleton(skel, min_len, "CRACKS")


def tonal_chains(gray: np.ndarray, levels: int, sigma: float, min_area: float,
                 grad_thr: float = 1.5, region: Optional[np.ndarray] = None,
                 layer: str = "TONAL") -> List[Chain]:
    """Iso-contours of the smoothed image at quantile levels (stains, shadows, masses).

    Contour pixels are kept only where real contrast exists, so smooth gradients
    (sky) never produce artificial iso-lines.
    """
    h, w = gray.shape
    gs = cv2.GaussianBlur(gray, (0, 0), sigma)
    mag = _grad_mag(gs.astype(np.float32))
    vals = gs[region] if region is not None and region.any() else gs
    thr = np.unique(np.percentile(vals, np.linspace(0, 100, levels + 2)[1:-1]))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out: List[Chain] = []
    for t in thr:
        b = (gs < t).astype(np.uint8)
        if region is not None:
            b &= region.astype(np.uint8)
        b = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
        cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 12 or cv2.contourArea(c) < min_area:
                continue
            p = c[:, 0, :].astype(np.float32)
            x, y = p[:, 0].astype(int), p[:, 1].astype(int)
            good = (mag[y, x] > grad_thr) & (x > 1) & (y > 1) & (x < w - 2) & (y < h - 2)
            for ch in split_runs(p, good, True, 6):
                ch.layer = layer
                out.append(ch)
    return out
