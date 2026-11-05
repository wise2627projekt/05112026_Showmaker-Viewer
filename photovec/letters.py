"""Letters that morph into a font (the part the Viewer needs: only numpy + cv2, nothing heavy).

The Show Maker lets you mark every letter of a tag with a skeleton (`Stroke` = anchors with optional bezier handles) and
`letterfit.py` fits that skeleton onto the matching glyph of the chosen font. What this file keeps of that work is a `LetterMorph`:
the skeleton as it is drawn (`src`), as it lies on the glyph in the finished line of text (`dst`), the stroke half widths at both
ends and the glyph outline at both ends. `LetterSet.apply` then moves the ink of every letter from `src` to `dst` with a thin-plate
spline (the stroke edges are control points too, so the strokes also get as thick as the font's) and fades the warped ink into the crisp glyph
at the end. Slider 0 returns the picture untouched, slider 1 draws the exact glyph outlines."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Pt = Tuple[float, float]
FADE_FROM = 0.55            # the warped ink starts to give way to the crisp glyph at this slider position (0..1)
GRID = 12                   # the spline is evaluated on a grid this many px apart and interpolated in between
MAX_CTRL = 28               # skeleton samples used as spline control points per letter (x3 with the two stroke edges)


# ---------------------------------------------------------------------------------------------------- the user's skeleton

@dataclass
class Stroke:
    """One pen path: anchors `pts`; `hout[i]` / `hin[i]` are the absolute bezier handle positions next to anchor i (None = no handle).
    A segment with no handle at either end is straight."""
    pts: List[Pt] = field(default_factory=list)
    hin: List[Optional[Pt]] = field(default_factory=list)
    hout: List[Optional[Pt]] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self):
        self.pts = [(float(x), float(y)) for x, y in self.pts]
        n = len(self.pts)
        self.hin = [None if h is None else (float(h[0]), float(h[1])) for h in list(self.hin)[:n]] + [None] * max(0, n - len(self.hin))
        self.hout = [None if h is None else (float(h[0]), float(h[1])) for h in list(self.hout)[:n]] + [None] * max(0, n - len(self.hout))

    def segments(self) -> List[Tuple[int, int]]:
        n = len(self.pts)
        return [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if self.closed and n > 2 else [])

    def is_straight(self, seg: int) -> bool:
        i, j = self.segments()[seg]
        return self.hout[i] is None and self.hin[j] is None

    def flatten(self, step: float = 4.0) -> np.ndarray:
        """Points along the path, about `step` px apart (anchors included; a closed path does not repeat its first point)."""
        out: List[Pt] = []
        segs = self.segments()
        for i, j in segs:
            p0, p1 = self.pts[i], self.pts[j]
            c0, c1 = self.hout[i], self.hin[j]
            if c0 is None and c1 is None:
                n = max(1, int(math.ceil(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step)))
                for k in range(n):
                    f = k / n
                    out.append((p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f))
            else:
                c0 = c0 or p0
                c1 = c1 or p1
                ln = math.hypot(c0[0] - p0[0], c0[1] - p0[1]) + math.hypot(c1[0] - c0[0], c1[1] - c0[1]) + math.hypot(p1[0] - c1[0], p1[1] - c1[1])
                n = max(2, int(math.ceil(ln / step)))
                for k in range(n):
                    f = k / n
                    g = 1.0 - f
                    out.append((g ** 3 * p0[0] + 3 * g * g * f * c0[0] + 3 * g * f * f * c1[0] + f ** 3 * p1[0],
                                g ** 3 * p0[1] + 3 * g * g * f * c0[1] + 3 * g * f * f * c1[1] + f ** 3 * p1[1]))
        if not self.closed and self.pts:
            out.append(self.pts[-1])
        return np.asarray(out, np.float64).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {"pts": [list(p) for p in self.pts], "hin": [None if h is None else list(h) for h in self.hin],
                "hout": [None if h is None else list(h) for h in self.hout], "closed": bool(self.closed)}

    @staticmethod
    def from_dict(d: dict) -> "Stroke":
        return Stroke(pts=[tuple(p) for p in d.get("pts", [])], hin=[None if h is None else tuple(h) for h in d.get("hin", [])],
                      hout=[None if h is None else tuple(h) for h in d.get("hout", [])], closed=bool(d.get("closed", False)))


@dataclass
class Letter:
    """One marked letter: which character it is and its skeleton (one or several strokes, e.g. an "i" with its dot)."""
    char: str = ""
    strokes: List[Stroke] = field(default_factory=list)

    def flatten(self, step: float = 4.0):
        """(points (N,2), runs [(start, stop, closed)]) of all strokes."""
        pts, runs, at = [], [], 0
        for s in self.strokes:
            a = s.flatten(step)
            if len(a) < 2:
                continue
            pts.append(a)
            runs.append((at, at + len(a), bool(s.closed)))
            at += len(a)
        return (np.concatenate(pts) if pts else np.zeros((0, 2))), runs

    def bbox(self) -> Tuple[float, float, float, float]:
        p, _ = self.flatten(8.0)
        if not len(p):
            return 0.0, 0.0, 0.0, 0.0
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    def to_dict(self) -> dict:
        return {"char": self.char, "strokes": [s.to_dict() for s in self.strokes]}

    @staticmethod
    def from_dict(d: dict) -> "Letter":
        return Letter(char=str(d.get("char", "")), strokes=[Stroke.from_dict(s) for s in d.get("strokes", [])])


# ---------------------------------------------------------------------------------------------------- what the Viewer plays

def _arr(a, shape_last=None) -> np.ndarray:
    return np.asarray(a, np.float32).reshape((-1, shape_last) if shape_last else (-1,))


@dataclass
class LetterMorph:
    """One letter ready to play: all coordinates are pixels of the photo as stored in the show."""
    char: str
    src: np.ndarray                       # (N,2) skeleton as marked
    dst: np.ndarray                       # (N,2) the same points on the glyph in the finished line of text
    w_src: np.ndarray                     # (N,) half stroke width of the marked letter
    w_dst: np.ndarray                     # (N,) half stroke width of the glyph
    runs: List[Tuple[int, int, bool]]     # strokes inside src / dst
    glyph0: List[np.ndarray]              # glyph outline pulled back onto the marked letter (polygons, N x 2 each)
    glyph1: List[np.ndarray]              # glyph outline in the finished line of text
    reach: float                          # radius around the skeleton whose ink belongs to this letter

    def to_dict(self) -> dict:
        r = lambda a, n=2: np.round(np.asarray(a, np.float64), n).tolist()
        return {"char": self.char, "src": r(self.src), "dst": r(self.dst), "w_src": r(self.w_src), "w_dst": r(self.w_dst),
                "runs": [[int(a), int(b), bool(c)] for a, b, c in self.runs], "glyph0": [r(p) for p in self.glyph0],
                "glyph1": [r(p) for p in self.glyph1], "reach": round(float(self.reach), 2)}

    @staticmethod
    def from_dict(d: dict) -> "LetterMorph":
        n = len(d["src"])
        runs = [(int(a), int(b), bool(c)) for a, b, c in d["runs"]]
        if (not n or not (len(d["dst"]) == len(d["w_src"]) == len(d["w_dst"]) == n) or not runs or runs[0][0] != 0
                or runs[-1][1] != n or any(b - a < 2 or a >= n or b > n for a, b, _ in runs)):
            raise ValueError("inconsistent letter data")
        return LetterMorph(char=str(d["char"]), src=_arr(d["src"], 2), dst=_arr(d["dst"], 2), w_src=_arr(d["w_src"]),
                           w_dst=_arr(d["w_dst"]), runs=runs, glyph0=[_arr(p, 2) for p in d["glyph0"]],
                           glyph1=[_arr(p, 2) for p in d["glyph1"]], reach=float(d["reach"]))


def morphs_to_list(ms: Sequence[LetterMorph]) -> list:
    return [m.to_dict() for m in ms]


def morphs_from_list(a) -> List[LetterMorph]:
    return [LetterMorph.from_dict(d) for d in a]


# ---------------------------------------------------------------------------------------------------- thin-plate spline

def _kernel(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 0.5 * r2 * np.log(r2)
    return np.where(r2 > 1e-6, k, 0.0)


def tps_fit(cp: np.ndarray, val: np.ndarray, smooth: float = 1e-3) -> np.ndarray:
    """Coefficients of the spline that sends control points `cp` (n,2) to `val` (n,2) (a little smoothing keeps it calm)."""
    n = len(cp)
    d = cp[:, None, :] - cp[None, :, :]
    r2 = (d * d).sum(-1)
    scale = max(1.0, float(r2.max()))
    K = _kernel(r2) + np.eye(n) * smooth * scale
    P = np.concatenate([np.ones((n, 1)), cp], axis=1)
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    b = np.zeros((n + 3, 2))
    b[:n] = val
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def tps_eval(cp: np.ndarray, coef: np.ndarray, xy: np.ndarray) -> np.ndarray:
    c = cp.astype(np.float32)
    x = xy.astype(np.float32)
    r2 = (x * x).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * (x @ c.T)
    U = _kernel(np.maximum(r2, 0.0))
    return (U @ coef[:len(cp)].astype(np.float32) + coef[len(cp)].astype(np.float32) + x @ coef[len(cp) + 1:].astype(np.float32)).astype(np.float64)


# ---------------------------------------------------------------------------------------------------- geometry helpers

def normals(P: np.ndarray, runs) -> np.ndarray:
    """Unit normals of the polylines inside P (left of the walking direction)."""
    N = np.zeros_like(P, dtype=np.float64)
    for a, b, closed in runs:
        Q = P[a:b].astype(np.float64)
        if closed:
            T = np.roll(Q, -1, 0) - np.roll(Q, 1, 0)
        else:
            T = np.gradient(Q, axis=0)
        L = np.hypot(T[:, 0], T[:, 1])
        L[L < 1e-9] = 1.0
        N[a:b, 0] = -T[:, 1] / L
        N[a:b, 1] = T[:, 0] / L
    return N


def _pick(n: int, runs, cap: int) -> np.ndarray:
    """Indices of at most ~cap samples spread over all runs (every run keeps its two ends)."""
    if n <= cap:
        return np.arange(n)
    keep = []
    for a, b, closed in runs:
        m = max(2, int(round(cap * (b - a) / n)))
        keep.append(np.unique(np.round(np.linspace(a, b - 1, m)).astype(int)))
    return np.unique(np.concatenate(keep))


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------------------------------------------- playing

_POOL = None


def _pool():
    """A few worker threads shared by all letter sets (None on a single-core machine)."""
    global _POOL
    if _POOL is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        n = min(6, os.cpu_count() or 1)
        _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="letters") if n > 1 else False
    return _POOL or None

def _ink_proxy(vec: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """How much of every pixel is drawing when the picture has no ink map (the drawing is on plain paper)."""
    d = np.abs(vec.astype(np.float32) - paper.astype(np.float32)[None, None, :]).max(axis=2)
    return np.clip(d / 96.0, 0.0, 1.0)


class LetterSet:
    """The letters of one photo at one working size. `apply(vec, ink, paper, t)` returns the picture with the letters t (0..1) of the way
    to the font. Slider 0 returns the inputs themselves (nothing is copied)."""

    def __init__(self, morphs: Sequence[LetterMorph], stored_size: Tuple[int, int], work_size: Tuple[int, int]):
        self.W, self.H = int(work_size[0]), int(work_size[1])
        f = self.W / max(1, stored_size[0])
        fy = self.H / max(1, stored_size[1])
        self.ms = [m for m in morphs if len(m.src) >= 2]
        sc = np.array([f, fy], np.float32)
        self.src = [m.src * sc for m in self.ms]
        self.dst = [m.dst * sc for m in self.ms]
        self.ws = [m.w_src * f for m in self.ms]
        self.wd = [m.w_dst * f for m in self.ms]
        self.g0 = [[p * sc for p in m.glyph0] for m in self.ms]
        self.g1 = [[p * sc for p in m.glyph1] for m in self.ms]
        self.reach = [max(2.0, m.reach * f) for m in self.ms]
        self._labels: Optional[np.ndarray] = None
        self._stg = None                                       # (vec, ink, paper, stage) of the last picture
        self._gfinal: dict = {}
        self._simple_ok: dict = {}
        self._boxes: List[Tuple[int, int, int, int]] = []
        order = sorted(range(len(self.ms)), key=lambda i: float(self.src[i][:, 0].mean()))
        self._rank: Optional[List[int]] = [order.index(i) for i in range(len(self.ms))]       # left to right

    def __len__(self) -> int:
        return len(self.ms)

    # which letter owns which pixel: the nearest skeleton within `reach`
    def _label_map(self) -> np.ndarray:
        if self._labels is not None:
            return self._labels
        H, W = self.H, self.W
        lab = np.full((H, W), -1, np.int16)
        best = np.full((H, W), np.inf, np.float32)
        self._boxes = []
        for j, m in enumerate(self.ms):
            P, r = self.src[j], self.reach[j]
            x0, y0 = max(0, int(P[:, 0].min() - r - 2)), max(0, int(P[:, 1].min() - r - 2))
            x1, y1 = min(W, int(P[:, 0].max() + r + 3)), min(H, int(P[:, 1].max() + r + 3))
            self._boxes.append((x0, y0, x1, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            canvas = np.full((y1 - y0, x1 - x0), 255, np.uint8)
            for a, b, closed in m.runs:
                q = np.round((P[a:b] - [x0, y0]) * 16).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [q], bool(closed), 0, 1, cv2.LINE_8, 4)
            D = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
            sub_b, sub_l = best[y0:y1, x0:x1], lab[y0:y1, x0:x1]
            take = (D < sub_b) & (D <= r)
            sub_b[take] = D[take]
            sub_l[take] = j
        self._labels = lab
        return lab

    def _control(self, j: int, e: float, thin: float = 0.0):
        """Spline control points: where the skeleton and its two stroke edges are at slider position e (dest) and where they came from (src)."""
        m = self.ms[j]
        sel = _pick(len(self.src[j]), m.runs, MAX_CTRL)
        P0, P1 = self.src[j].astype(np.float64), self.dst[j].astype(np.float64)
        Pt_ = P0 + (P1 - P0) * e
        w = (self.ws[j].astype(np.float64) + (self.wd[j].astype(np.float64) - self.ws[j]) * e) * (1.0 - thin)
        n0, nt = normals(P0, m.runs), normals(Pt_, m.runs)
        w0 = self.ws[j].astype(np.float64)
        dest = np.concatenate([Pt_[sel], (Pt_ + nt * w[:, None])[sel], (Pt_ - nt * w[:, None])[sel]])
        srcp = np.concatenate([P0[sel], (P0 + n0 * w0[:, None])[sel], (P0 - n0 * w0[:, None])[sel]])
        return dest, srcp

    def _dest_box(self, dest: np.ndarray, extra: float) -> Tuple[int, int, int, int]:
        x0, y0 = int(max(0, math.floor(dest[:, 0].min() - extra))), int(max(0, math.floor(dest[:, 1].min() - extra)))
        x1, y1 = int(min(self.W, math.ceil(dest[:, 0].max() + extra))), int(min(self.H, math.ceil(dest[:, 1].max() + extra)))
        return x0, y0, x1, y1

    def _flow(self, dest: np.ndarray, srcp: np.ndarray, box) -> Tuple[np.ndarray, np.ndarray]:
        """For every pixel of `box` (destination) the position it is taken from (source): (mapx, mapy)."""
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        gx = np.arange(0, w + GRID, GRID, dtype=np.float64) + x0
        gy = np.arange(0, h + GRID, GRID, dtype=np.float64) + y0
        XX, YY = np.meshgrid(gx, gy)
        pts = np.stack([XX.ravel(), YY.ravel()], 1)
        coef = tps_fit(dest, srcp)
        out = tps_eval(dest, coef, pts).astype(np.float32).reshape(len(gy), len(gx), 2)
        mx = np.broadcast_to((np.arange(w, dtype=np.float32) / GRID)[None, :], (h, w)).copy()
        my = np.broadcast_to((np.arange(h, dtype=np.float32) / GRID)[:, None], (h, w)).copy()
        full = cv2.remap(out, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return full[..., 0], full[..., 1]

    def prepare(self) -> None:
        """Do the one-time work (which pixel belongs to which letter) now, e.g. in a background thread."""
        self._label_map()

    def timing(self, j: int, t: float, style: str = "melt") -> Tuple[float, float, float]:
        """(move e, melt s, thin h) of letter j at slider position t. e: 0 = where it was marked, 1 = in the line of text; s: how far the ink has
        melted into the glyph shape; h: how much the stroke is squeezed towards its skeleton on the way (style "thin")."""
        n = max(1, len(self.ms))
        lt = t
        if style == "wave":                                  # the letters leave one after the other, left to right
            rank = self._rank[j] if self._rank is not None else j
            lag = 0.45 * rank / max(1, n - 1)
            lt = min(1.0, max(0.0, (t - lag) / (1.0 - 0.45)))
        e = lt
        s = _smoothstep((lt - FADE_FROM) / (1.0 - FADE_FROM)) if lt < 1.0 else 1.0
        h = 0.7 * math.sin(math.pi * min(1.0, lt)) ** 2 if style == "thin" else 0.0
        return e, s, h

    def _stage(self, vec: np.ndarray, ink: Optional[np.ndarray], paper_a: np.ndarray):
        """The part of the work that does not depend on the slider: every letter pulled out of the picture (its place becomes empty paper) and
        its pieces kept. Remembered for the last picture, so dragging the slider only pays for the moving letters."""
        c = self._stg
        if c is not None and c[0] is vec and c[1] is ink and np.array_equal(c[2], paper_a):
            return c[3]
        lab = self._label_map()
        orig_v = vec
        orig_a = ink if ink is not None else np.clip(_ink_proxy(vec, paper_a) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out_a = orig_a.copy()
        out_v = vec.copy()
        full = 255.0 if out_a.dtype == np.uint8 else 1.0
        paper_u8 = paper_a.astype(np.uint8)
        pieces = []
        for j in range(len(self.ms)):
            x0, y0, x1, y1 = self._boxes[j]
            own = (lab[y0:y1, x0:x1] == j) if (x1 > x0 and y1 > y0) else None
            if own is None or not own.any():
                pieces.append(None)
                continue
            aj = np.where(own, out_a[y0:y1, x0:x1].astype(np.float32) / full, 0.0).astype(np.float32)
            vj = (out_v[y0:y1, x0:x1].astype(np.float32) * aj[..., None]).astype(np.float32)          # premultiplied colour
            tot = float(aj.sum())
            col = (vj.reshape(-1, 3).sum(0) / tot) if tot > 1e-3 else np.zeros(3, np.float32)
            pieces.append((aj, vj, col, own))
            out_a[y0:y1, x0:x1][own] = 0
            out_v[y0:y1, x0:x1][own] = paper_u8
        res = (out_v, out_a, pieces, full, orig_v, orig_a)
        self._stg = (vec, ink, paper_a.copy(), res)
        return res

    def _final_glyph(self, j: int, box) -> np.ndarray:
        """The finished glyph (slider 100) inside `box`; it never changes, so it is drawn once."""
        hit = self._gfinal.get(j)
        if hit is not None and hit[0] == box:
            return hit[1]
        A = self._glyph(j, 1.0, box)
        self._gfinal[j] = (box, A)
        return A

    def apply(self, vec: np.ndarray, ink: Optional[np.ndarray], paper, t: float, style: str = "melt",
              mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """vec: HxWx3 uint8 BGR picture, ink: HxW uint8 (or float 0..1) or None, paper: BGR colour, mask: HxW uint8 or None (the area where the
        drawing is shown). -> (vec, ink, mask), same dtypes; the inputs themselves are never changed. The returned mask also covers the
        letters wherever they have travelled to."""
        t = float(min(1.0, max(0.0, t)))
        if t <= 0.0 or not self.ms or vec.shape[1] != self.W or vec.shape[0] != self.H:
            return vec, ink, mask
        paper_a = np.asarray(paper, np.float32).reshape(3)
        base_v, base_a, pieces, full, orig_v, orig_a = self._stage(vec, ink, paper_a)
        out_v = base_v.copy()
        out_a = base_a.copy()
        out_m = None if mask is None else mask.copy()
        todo = []
        for j, piece in enumerate(pieces):
            if piece is None:
                continue
            e, s, h = self.timing(j, t, style)
            if e <= 0.0:                                           # not started yet: it simply stays where it is
                sx0, sy0, sx1, sy1 = self._boxes[j]
                out_a[sy0:sy1, sx0:sx1][piece[3]] = orig_a[sy0:sy1, sx0:sx1][piece[3]]
                out_v[sy0:sy1, sx0:sx1][piece[3]] = orig_v[sy0:sy1, sx0:sx1][piece[3]]
                continue
            todo.append((j, piece, e, s, h))
        pool = _pool() if len(todo) >= 3 else None                 # (the letters are independent: cv2 and numpy let go of the GIL)
        if pool is not None:
            done = list(pool.map(lambda a: self._render(*a), todo))
        else:
            done = [self._render(*a) for a in todo]
        for r in done:                                             # (put together in order, so overlapping letters stack as before)
            if r is None:
                continue
            (x0, y0, x1, y1), A, V = r
            sa = out_a[y0:y1, x0:x1].astype(np.float32) / full
            sv = out_v[y0:y1, x0:x1].astype(np.float32)
            sv = sv * (1.0 - A[..., None]) + V
            out_v[y0:y1, x0:x1] = np.clip(sv + 0.5, 0, 255).astype(np.uint8)
            sa = 1.0 - (1.0 - sa) * (1.0 - A)
            out_a[y0:y1, x0:x1] = np.clip(sa * full + 0.5, 0, full).astype(out_a.dtype) if full == 255.0 else sa
            if out_m is not None:
                np.maximum(out_m[y0:y1, x0:x1], np.clip(A * 255.0 + 0.5, 0, 255).astype(out_m.dtype), out=out_m[y0:y1, x0:x1])
        return out_v, (out_a if ink is not None else None), out_m

    def _render(self, j: int, piece, e: float, s: float, h: float):
        """One letter at its place: (box, coverage A, premultiplied colour V) or None."""
        aj, vj, col, own = piece
        sx0, sy0, _, _ = self._boxes[j]
        dest, srcp = self._control(j, e, h)
        wmax = float(max(self.ws[j].max(), self.wd[j].max()))
        box = self._dest_box(dest, wmax * 1.6 + 4)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return None
        if s >= 1.0:
            A = self._final_glyph(j, box)
            return box, A, A[..., None] * col
        mx, my = self._flow(dest, srcp, box)
        wa = cv2.remap(aj, mx - sx0, my - sy0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wv = cv2.remap(vj, mx - sx0, my - sy0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if s <= 0.0:
            return box, wa, wv
        ga = self._glyph(j, e, box)                                # melt: blend the two shapes' signed distances, so it stays solid
        A = _melt(wa, ga, s)
        cw = wv / np.maximum(wa, 0.05)[..., None]
        cw = np.where((wa > 0.05)[..., None], cw, col)
        return box, A, A[..., None] * (cw * (1.0 - s) + col * s)

    def _simple(self, j: int) -> bool:
        """True when the glyph's outline has no overlapping parts, so a plain even-odd fill gives the same shape as the non-zero fill."""
        v = self._simple_ok.get(j)
        if v is None:
            polys = self.g1[j]
            allp = np.concatenate(polys)
            x0, y0 = np.floor(allp.min(0)) - 1
            w, h = int(np.ceil(allp[:, 0].max() - x0)) + 2, int(np.ceil(allp[:, 1].max() - y0)) + 2
            SS = 2
            q = [((p - [x0, y0]) * SS + (SS - 1) / 2.0) for p in polys]
            a = fill_nonzero((h * SS, w * SS), q, evenodd=True)
            b = fill_nonzero((h * SS, w * SS), q)
            v = bool(np.count_nonzero(a != b) <= 0.002 * max(1, np.count_nonzero(b)))
            self._simple_ok[j] = v
        return v

    def _glyph(self, j: int, e: float, box) -> np.ndarray:
        """The glyph at slider position e as coverage 0..1 inside `box` (outline polygons morph from the pulled-back to the final position)."""
        x0, y0, x1, y1 = box
        SS = 2 if e < 0.8 else (3 if e < 1.0 else 8)   # supersampling: coarse while moving (it only shapes the melt), crisp at the end
        h, w = (y1 - y0) * SS, (x1 - x0) * SS
        canvas = np.zeros((h, w), np.uint8)
        polys = []
        for p0, p1 in zip(self.g0[j], self.g1[j]):
            p = p0 + (p1 - p0) * e
            polys.append(np.round(((p - [x0, y0]) * SS + (SS - 1) / 2.0) * 16).astype(np.int32).reshape(-1, 1, 2))      # (pixel centres sit on whole numbers)
        if self._simple(j):
            cv2.fillPoly(canvas, polys, 255, cv2.LINE_8, 4)        # (even-odd is right for this glyph, and several times faster)
        else:
            _fill_evenodd(canvas, polys)
        small = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0


def fill_nonzero(shape, polys, evenodd: bool = False) -> np.ndarray:
    """Non-zero winding fill of closed polygons (float x, y; whole numbers are pixel centres) -> uint8 0/255.
    Opposite-direction contours cancel (the hole of an "O"); overlapping or self-overlapping contours of the
    same direction stay solid (fonts draw crossbars and joins that way). cv2.fillPoly cannot do this."""
    h, w = int(shape[0]), int(shape[1])
    rows, xs, dirs = [], [], []
    for p in polys:
        a = np.asarray(p, np.float64).reshape(-1, 2)
        if len(a) < 3:
            continue
        b = np.roll(a, -1, axis=0)
        x0, y0, x1, y1 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        up = y1 > y0
        lo = np.where(up, y0, y1)
        hi = np.where(up, y1, y0)
        r0 = np.ceil(lo).astype(np.int64)                  # rows whose centre lies in [lo, hi)
        r1 = np.ceil(hi).astype(np.int64)
        r0c = np.clip(r0, 0, h)
        r1c = np.clip(r1, 0, h)
        n = np.maximum(r1c - r0c, 0)
        n[y0 == y1] = 0
        tot = int(n.sum())
        if tot == 0:
            continue
        edge = np.repeat(np.arange(len(a)), n)
        start = np.cumsum(n) - n
        r = r0c[edge] + (np.arange(tot) - start[edge])
        t = (r - y0[edge]) / (y1[edge] - y0[edge])
        rows.append(r)
        xs.append(x0[edge] + t * (x1[edge] - x0[edge]))
        dirs.append(np.where(up[edge], 1, -1))
    out = np.zeros((h, w), np.uint8)
    if not rows:
        return out
    r = np.concatenate(rows)
    x = np.concatenate(xs)
    d = np.concatenate(dirs)
    order = np.argsort(r * float(w + 4) + np.clip(x, -1.0, w + 1.0) + 1.0, kind="stable")
    r, x, d = r[order], x[order], d[order]
    wind = np.cumsum(np.ones_like(d) if evenodd else d)    # (a row of a closed outline balances back to 0 / an even count, so no reset is needed)
    if evenodd:
        wind = wind & 1
    inside = (wind[:-1] != 0) & (r[1:] == r[:-1])
    ia = np.nonzero(inside)[0]
    if len(ia) == 0:
        return out
    xa = np.clip(np.ceil(x[ia]).astype(np.int64), 0, w)
    xb = np.clip(np.ceil(x[ia + 1]).astype(np.int64), 0, w)
    n = h * (w + 1)
    diff = (np.bincount(r[ia] * (w + 1) + xa, minlength=n) - np.bincount(r[ia] * (w + 1) + xb, minlength=n)).reshape(h, w + 1)
    out[:] = (np.cumsum(diff, axis=1)[:, :w] > 0) * 255
    return out


def _fill_evenodd(canvas: np.ndarray, polys) -> None:
    """Fill `polys` (int32 points in 1/16 pixel units, as cv2.fillPoly takes them) into canvas, non-zero winding."""
    canvas[:] = np.maximum(canvas, fill_nonzero(canvas.shape, [p.reshape(-1, 2) / 16.0 for p in polys]))


def _signed(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the edge of a 0/1 mask (positive inside)."""
    m = mask.astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)


def _melt(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Coverage of the shape that is the fraction s of the way from shape a to shape b (both 0..1 coverage maps)."""
    ma = cv2.morphologyEx((a >= 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))      # (closes pin holes the warp can leave)
    sd = (1.0 - s) * _signed(ma) + s * _signed(b >= 0.5)
    return np.clip(sd + 0.5, 0.0, 1.0).astype(np.float32)
