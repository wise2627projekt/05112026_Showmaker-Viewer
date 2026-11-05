"""Context-aware selection for the mask editor ("smart brush"): paint a few strokes on what you want (green) and on what you do not
want (red) and the selection snaps to the edges of the picture, like Photoshop's Quick Selection.

How it works (all classic image processing, no model, runs everywhere):
  1. The strokes are turned into hard seeds. A working copy of the picture (long side `work`, default 640 px) is used for speed.
  2. GrabCut (a colour model of the seeds plus an edge-aware graph cut) decides every other pixel. The result of the previous
     stroke is the starting point, so every further stroke only corrects it: one stroke takes a fraction of a second.
  3. The result is lifted to the full resolution and snapped to the real edges with a guided filter, so the outline is as sharp as
     the photo itself, not as coarse as the working copy.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter (grey guide, float 0..1): smooths `src` but keeps the edges of `guide`."""
    r = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, r, normalize=True, borderType=cv2.BORDER_REFLECT)
    mg, ms = box(guide), box(src)
    cov = box(guide * src) - mg * ms
    var = box(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return box(a) * guide + box(b)


class SmartSelector:
    """Interactive selection of one picture. Strokes are in the coordinates of the full picture."""

    def __init__(self, bgr: np.ndarray, work: int = 640):
        self.full = bgr
        self.h, self.w = bgr.shape[:2]
        self.scale = min(1.0, work / float(max(self.h, self.w)))
        ww, wh = max(8, int(round(self.w * self.scale))), max(8, int(round(self.h * self.scale)))
        small = cv2.resize(bgr, (ww, wh), interpolation=cv2.INTER_AREA) if self.scale < 1 else bgr.copy()
        self.work = cv2.GaussianBlur(small, (0, 0), 0.8)            # the wall's grain would otherwise cost the graph cut its edges
        self.wh, self.ww = self.work.shape[:2]
        self.lab = cv2.cvtColor(self.work, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.fg = np.zeros((self.wh, self.ww), np.uint8)             # seeds: 255 where the user painted "this is it"
        self.bg = np.zeros((self.wh, self.ww), np.uint8)             # seeds: 255 where the user painted "this is not it"
        self.prev: Optional[np.ndarray] = None                       # last result (bool, working size)

    def set_start(self, mask: Optional[np.ndarray]) -> None:
        """Continue from an existing mask (any size, 0..255): the next strokes correct it."""
        if mask is None or not mask.any():
            self.prev = None
            return
        self.prev = cv2.resize(mask, (self.ww, self.wh), interpolation=cv2.INTER_AREA) > 127

    def _raster(self, target: np.ndarray, pts: Sequence[Point], radius: float) -> None:
        p = np.array([[x * self.scale, y * self.scale] for x, y in pts], np.float32)
        th = max(1, int(round(2 * radius * self.scale)))
        if len(p) == 1:
            cv2.circle(target, (int(round(p[0][0])), int(round(p[0][1]))), max(1, th // 2), 255, -1)
        else:
            cv2.polylines(target, [np.round(p).astype(np.int32)], False, 255, th, cv2.LINE_8)
            for q in (p[0], p[-1]):
                cv2.circle(target, (int(round(q[0])), int(round(q[1]))), max(1, th // 2), 255, -1)

    def add_stroke(self, pts: Sequence[Point], radius: float, positive: bool = True) -> None:
        """A stroke of the brush. A stroke on the other kind removes the seeds it covers (a correction wins over an old stroke)."""
        if not pts:
            return
        a, b = (self.fg, self.bg) if positive else (self.bg, self.fg)
        self._raster(a, pts, radius)
        tmp = np.zeros_like(a)
        self._raster(tmp, pts, radius)
        b[tmp > 0] = 0

    # ------------------------------------------------------------------ solving
    def _colour_prior(self, fg: np.ndarray, tolerance: float, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Pixels whose colour is like the colour under the green strokes (bool). It gives GrabCut a tight starting model instead
        of a blob of mixed colours, which is what makes the result follow the stroke's colour and not the whole wall.
        A stroke that is wider than the thing it paints on also covers some wall: colours of the stroke that are just as common in
        the surroundings (or under the red strokes) are recognised as "wall" and left out."""
        x0, y0, x1, y1 = roi
        pts = self.lab[fg]
        rng = np.random.default_rng(0)
        if len(pts) > 6000:
            pts = pts[rng.choice(len(pts), 6000, replace=False)]
        # what the surroundings look like: the red strokes, or else everything in the area that is not near a green stroke
        if (self.bg > 0).sum() >= 50:
            around = self.lab[self.bg > 0]
        else:
            far = np.zeros((self.wh, self.ww), np.uint8)
            far[y0:y1, x0:x1] = 255
            far[cv2.dilate(self.fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 6 + 1,) * 2)) > 0] = 0
            around = self.lab[far > 0]
        if len(around) > 6000:
            around = around[rng.choice(len(around), 6000, replace=False)]
        k = int(min(3, max(1, len(pts) // 40)))
        if k > 1:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, lab_k, cen = cv2.kmeans(np.ascontiguousarray(pts), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
            lab_k = lab_k.ravel()
        else:
            cen, lab_k = pts.mean(axis=0, keepdims=True), np.zeros(len(pts), int)
        tol = 0.5 + min(max(tolerance, 0.0), 1.0)
        kept = []
        for i, c in enumerate(cen):
            mine = pts[lab_k == i]
            if len(mine) < max(8, 0.08 * len(pts)) and len(cen) > 1:
                continue
            tc = max(9.0, 1.7 * float(np.percentile(np.linalg.norm(mine - c, axis=1), 90)))
            wall = float((np.linalg.norm(around - c, axis=1) < tc).mean()) if len(around) else 0.0
            kept.append((wall, tc, c))
        good = [x for x in kept if x[0] <= 0.5] or ([min(kept, key=lambda x: x[0])] if kept else [(0.0, 12.0, pts.mean(axis=0))])
        score = np.min([np.linalg.norm(self.lab - c, axis=2) / (tc * tol) for _, tc, c in good], axis=0)
        return score < 1.0

    def solve(self, connected: bool = True, iterations: int = 2, tolerance: float = 0.5) -> np.ndarray:
        """Decide every pixel from the seeds (bool, working size). `tolerance` 0..1: how readily similar colours are taken along."""
        fg, bg = self.fg > 0, self.bg > 0
        if not fg.any():
            res = np.zeros((self.wh, self.ww), bool) if self.prev is None else self.prev & ~bg
            self.prev = res
            return res
        # the area GrabCut works on: everything the strokes and the last result touch, with a margin (faster, and it keeps a
        # first stroke from selecting colours of the whole picture)
        # parts of the existing mask that the new strokes do not touch are left exactly as they are (otherwise GrabCut would judge a
        # region painted earlier, with other tools, by the colours of a stroke somewhere else and throw it away)
        lock = np.zeros((self.wh, self.ww), bool)
        if self.prev is not None and self.prev.any():
            n_p, lab_p = cv2.connectedComponents(self.prev.astype(np.uint8), connectivity=8)
            touch = cv2.dilate(((fg | bg).astype(np.uint8)), np.ones((9, 9), np.uint8)) > 0
            hit_p = np.unique(lab_p[touch & self.prev])
            lock = self.prev & ~np.isin(lab_p, hit_p[hit_p > 0])
        prev_live = None if self.prev is None else self.prev & ~lock
        cover = fg | bg | (prev_live if prev_live is not None else False)
        ys, xs = np.nonzero(cover)
        m = int(0.25 * max(ys.max() - ys.min(), xs.max() - xs.min(), 24)) + 12
        x0, x1 = max(0, xs.min() - m), min(self.ww, xs.max() + m + 1)
        y0, y1 = max(0, ys.min() - m), min(self.wh, ys.max() + m + 1)
        mask = np.full((self.wh, self.ww), cv2.GC_PR_BGD, np.uint8)
        prior = self._colour_prior(fg, tolerance, (x0, y0, x1, y1))
        near = cv2.dilate(self.fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1,) * 2)) > 0
        guess = prior & near                                          # colours like the strokes, in the neighbourhood of the strokes
        if prev_live is not None:
            guess = guess | prev_live
        mask[guess] = cv2.GC_PR_FGD
        mask[fg] = cv2.GC_FGD
        mask[bg] = cv2.GC_BGD
        sub = np.ascontiguousarray(self.work[y0:y1, x0:x1])
        sm = np.ascontiguousarray(mask[y0:y1, x0:x1])
        if not ((sm == cv2.GC_FGD) | (sm == cv2.GC_PR_FGD)).any() or not ((sm == cv2.GC_BGD) | (sm == cv2.GC_PR_BGD)).any():
            res = (self.prev.copy() if self.prev is not None else np.zeros((self.wh, self.ww), bool)) | fg
            self.prev = res & ~bg
            return self.prev
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(sub, sm, None, bgd, fgd, max(1, int(iterations)), cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            sm = np.where(fg[y0:y1, x0:x1], cv2.GC_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        part = (sm == cv2.GC_FGD) | (sm == cv2.GC_PR_FGD)
        res = self.prev.copy() if self.prev is not None else np.zeros((self.wh, self.ww), bool)
        res[y0:y1, x0:x1] = part
        res |= lock
        res &= ~bg
        res |= fg
        if connected:
            keep = fg | (self.prev if self.prev is not None else False)
            n, lab = cv2.connectedComponents(res.astype(np.uint8), connectivity=8)
            if n > 2:
                hit = np.unique(lab[keep & res])
                res = np.isin(lab, hit[hit > 0])
        self.prev = res
        return res

    # ------------------------------------------------------------------ result at full size
    def alpha(self, result: Optional[np.ndarray] = None, snap: float = 1.0, matte: bool = True) -> np.ndarray:
        """The selection at the full resolution, snapped to the edges of the photo (uint8 0..255). Only the part of the picture that
        the selection touches is computed, so a small selection on a big photo is quick."""
        r = self.prev if result is None else result
        out = np.zeros((self.h, self.w), np.uint8)
        if r is None or not r.any():
            return out
        ys, xs = np.nonzero(r)
        rad = min(24, max(2, int(round(1.5 / self.scale)))) if self.scale < 1 else 2
        m = int(4 * rad + 12 + (0.05 * max(self.h, self.w) if matte else 0))
        x0, x1 = max(0, int(xs.min() / self.scale) - m), min(self.w, int((xs.max() + 1) / self.scale) + m + 1)
        y0, y1 = max(0, int(ys.min() / self.scale) - m), min(self.h, int((ys.max() + 1) / self.scale) + m + 1)
        cw, ch = x1 - x0, y1 - y0
        soft = cv2.GaussianBlur(r.astype(np.float32), (0, 0), 1.2)
        # to full size (pixel centres line up with the area-averaged working copy), smooth steps instead of staircases
        M = np.array([[self.scale, 0, self.scale * x0 + 0.5 * self.scale - 0.5], [0, self.scale, self.scale * y0 + 0.5 * self.scale - 0.5]], np.float64)
        up = np.clip(cv2.warpAffine(soft, M, (cw, ch), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE), 0, 1)
        guide = self.guide[y0:y1, x0:x1]
        if snap > 0:
            # the guided filter pulls the outline onto the real edges (radius ~ the coarseness of the working copy)
            up = np.clip(guided_filter(guide, up, rad, 2e-4 + 4e-3 * (1.0 - min(max(snap, 0), 1))), 0, 1)
        # at the very edge the photo itself decides: every pixel in a narrow band is a mix of the paint colour and the wall colour
        # next to it, so its share of paint is its position between the two (a small matting step). That puts the outline where
        # the picture's own 50 % edge is, not one pixel inside or outside.
        if matte:
            up = self._matte(up, self.full[y0:y1, x0:x1])
        t = np.clip((up - 0.5) / 0.35 + 0.5, 0, 1)
        out[y0:y1, x0:x1] = np.round(t * t * (3 - 2 * t) * 255).astype(np.uint8)
        return out

    def _matte(self, base: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        inside, outside = base >= 0.97, base <= 0.03
        band = ~(inside | outside)
        if not band.any() or not inside.any() or not outside.any():
            return base
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        band = cv2.dilate(band.astype(np.uint8), k) > 0
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        H, W = base.shape
        solid_in = (cv2.erode(inside.astype(np.uint8), k) > 0) & ~band
        solid_out = (cv2.erode(outside.astype(np.uint8), k) > 0) & ~band
        sig = max(4.0, 0.015 * max(self.h, self.w))

        f = max(1, int(min(H, W) // 400))                 # the surface colour varies slowly: estimate it on a small copy
        sm = cv2.resize(img, (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA) if f > 1 else img

        def local(valid: np.ndarray) -> np.ndarray:                # colour of the surface nearby (normalised convolution)
            wgt = valid.astype(np.float32)
            if f > 1:
                wgt = cv2.resize(wgt, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_AREA)
            num = cv2.GaussianBlur(sm * wgt[..., None], (0, 0), sig / f)
            den = cv2.GaussianBlur(wgt, (0, 0), sig / f)
            out = num / np.maximum(den, 1e-4)[..., None]
            return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR) if f > 1 else out
        cf, cb = local(solid_in), local(solid_out)
        d = cf - cb
        den = (d * d).sum(axis=2)
        alpha = ((img - cb) * d).sum(axis=2) / np.maximum(den, 1e-3)
        alpha = np.clip(alpha, 0, 1)
        ok = band & (den > 12.0 ** 2)                               # a colour difference too small to measure: keep the shape
        out = base.copy()
        out[ok] = alpha[ok]
        return out


# ------------------------------------------------------------------ helpers shared by the editor and the tests
def magic_wand(bgr: np.ndarray, x: int, y: int, tolerance: float = 18.0, contiguous: bool = True) -> np.ndarray:
    """Pixels whose colour is close to the clicked one (Lab distance <= tolerance), by default only the connected area (bool)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    x, y = min(max(int(x), 0), bgr.shape[1] - 1), min(max(int(y), 0), bgr.shape[0] - 1)
    ref = np.median(lab[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3), axis=0)
    near = np.linalg.norm(lab - ref, axis=2) <= tolerance
    if not contiguous:
        return near
    n, lb = cv2.connectedComponents(near.astype(np.uint8), connectivity=4)
    return lb == lb[y, x] if lb[y, x] > 0 else np.zeros_like(near)


def paint_disc(mask: np.ndarray, pts: Sequence[Point], radius: float, value: int, softness: float = 0.0) -> None:
    """Plain brush: paints value (0 or 255) along the points, optionally with a soft edge."""
    layer = np.zeros(mask.shape[:2], np.uint8)
    p = np.round(np.array(pts, np.float32)).astype(np.int32)
    th = max(1, int(round(2 * radius)))
    if len(p) == 1:
        cv2.circle(layer, tuple(int(v) for v in p[0]), max(1, th // 2), 255, -1, cv2.LINE_AA)
    else:
        cv2.polylines(layer, [p], False, 255, th, cv2.LINE_AA)
        for q in (p[0], p[-1]):
            cv2.circle(layer, tuple(int(v) for v in q), max(1, th // 2), 255, -1, cv2.LINE_AA)
    if softness > 0:
        layer = cv2.GaussianBlur(layer, (0, 0), max(0.5, radius * softness * 0.5))
    if value > 127:
        np.maximum(mask, layer, out=mask)
    else:
        mask[:] = (mask.astype(np.int32) * (255 - layer.astype(np.int32)) // 255).astype(np.uint8)


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow (px > 0) or shrink (px < 0) a mask by whole pixels."""
    if px == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(px) + 1,) * 2)
    return cv2.dilate(mask, k) if px > 0 else cv2.erode(mask, k)


def smooth_edges(mask: np.ndarray, radius: float) -> np.ndarray:
    """Rounder, calmer outline: blur then re-sharpen the edge."""
    if radius <= 0:
        return mask
    b = cv2.GaussianBlur(mask, (0, 0), radius).astype(np.float32) / 255.0
    t = np.clip((b - 0.4) / 0.2, 0, 1)
    return np.round(t * t * (3 - 2 * t) * 255).astype(np.uint8)


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop selected islands smaller than `min_area` px and fill holes smaller than that."""
    if min_area <= 0:
        return mask
    out = (mask > 127).astype(np.uint8)
    for val in (1, 0):
        n, lb, st, _ = cv2.connectedComponentsWithStats(out if val else 1 - out, connectivity=8)
        small = np.zeros(n, bool)
        small[1:] = st[1:, cv2.CC_STAT_AREA] < min_area
        sel = small[lb]
        if val:
            out[sel] = 0
        else:
            out[sel] = 1
    return (out * 255).astype(np.uint8)
