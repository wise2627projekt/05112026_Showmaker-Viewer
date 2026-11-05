"""Layers: a mask splits a photo into two layers - the part that is masked (the graffiti) and the rest (the facade). Each layer has
its own visibility over time, and the facade can also be blurred, so the sliders can "extract" the graffiti from its background.

The layers are applied to the finished frame of one photo (after tracing, colours and the fill), so they work with every other
setting. Values (all optional, 1 / 1 / 0 when missing):
    layer_f   visibility of the facade   0..1   (0 = only paper where the facade is)
    layer_g   visibility of the graffiti 0..1
    blur_f    blur of the facade         0..100 (percent of 3 % of the picture's long side)
    layer_p   1 = "facade = original photo" mode: the facade is the photo itself (not a vector), only the masked part is vectorized.
              Then layer_f fades the photo to paper and layer_g is the share of the VECTOR in the masked part (0 = still the photo).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

LAYER_KEYS = ("layer_f", "layer_g", "blur_f", "layer_p")
BLUR_MAX = 0.03                 # blur at 100 %: sigma = 3 % of the long side


def fit_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """The mask (uint8 0..255) at the size of a picture."""
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA if mask.shape[0] > h else cv2.INTER_LINEAR)


# The same mask is used for every frame: its float version and the eroded copy are kept (a few, newest last) instead of being made again.
_MASK_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_MASK_LOCK = threading.Lock()
_MASK_KEEP = 6                       # entries
_MASK_BYTES = 160 * 1024 * 1024      # ... and bytes (a float mask of a 4000 px picture is 55 MB)


def _mask_cached(mask: np.ndarray, key: tuple, make):
    k = (id(mask),) + key
    with _MASK_LOCK:
        hit = _MASK_CACHE.get(k)
        if hit is not None and hit[0] is mask:
            _MASK_CACHE.move_to_end(k)
            return hit[1]
    val = make()
    with _MASK_LOCK:
        _MASK_CACHE[k] = (mask, val)                      # (holding the mask itself keeps its id from being reused)
        while len(_MASK_CACHE) > 1 and (len(_MASK_CACHE) > _MASK_KEEP or sum(v[1].nbytes for v in _MASK_CACHE.values()) > _MASK_BYTES):
            _MASK_CACHE.popitem(last=False)
    return val


def mask_f32(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """The mask at w x h as float 0..1 (shared, read-only: do not change it)."""
    return _mask_cached(mask, ("f", w, h), lambda: fit_mask(mask, w, h).astype(np.float32) * np.float32(1.0 / 255.0))


def _mask_eroded_f32(mask: np.ndarray, w: int, h: int, k: int) -> np.ndarray:
    return _mask_cached(mask, ("e", w, h, k), lambda: cv2.erode(fit_mask(mask, w, h), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(np.float32) * np.float32(1.0 / 255.0))


def _alpha(ink: Optional[np.ndarray], w: int, h: int):
    """Ink coverage (0..1) at w x h; None = everything is drawing."""
    if ink is None:
        return None
    if ink.dtype == np.uint8:                           # (the windows keep the ink as bytes: 0..255)
        a = ink.astype(np.float32)
        a *= np.float32(1.0 / 255.0)
    else:
        a = ink if ink.dtype == np.float32 else ink.astype(np.float32)
    return a if a.shape[:2] == (h, w) else cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if a.shape[0] > h else cv2.INTER_LINEAR)


def _to_u8(d: np.ndarray) -> np.ndarray:
    """float picture -> uint8, rounded to the nearest and clipped (one pass)."""
    return cv2.add(d, 0.0, dtype=cv2.CV_8U) if d.ndim == 2 else cv2.add(d, (0.0, 0.0, 0.0, 0.0), dtype=cv2.CV_8U)


def _fade(pic: np.ndarray, empty: np.ndarray, v: float) -> np.ndarray:
    """pic * v + empty * (1 - v) as a new float32 picture (empty: shape (1, 1, 3))."""
    out = pic * np.float32(v)
    out += empty * np.float32(1.0 - v)
    return out


def _lerp(base: np.ndarray, target: np.ndarray, wgt) -> np.ndarray:
    """base + (target - base) * wgt as a new float32 picture; wgt is (h, w) or a number."""
    d = np.subtract(target, base, dtype=np.float32)
    d *= (wgt[..., None] if isinstance(wgt, np.ndarray) else np.float32(wgt))
    d += base
    return d


@dataclass(frozen=True)
class MaskStyle:
    """How the mask is finished before it is used (the drawn mask itself is never changed).
    smooth: 0..100, rounds the outline - the pixel steps and jaggies of a painted / smart-selected mask disappear.
    soft:   0..100, a soft edge (feather): the graffiti fades into the background over a few pixels instead of ending sharply.
    vector: the outline is rebuilt as smooth curves (sub-pixel contour of the mask, smoothed, drawn anti-aliased at the picture's own
            size), so it is as sharp at 4000 px as at 400 px and never shows the pixels of the mask."""
    smooth: float = 0.0
    soft: float = 0.0
    vector: bool = False

    def plain(self) -> bool:
        return not (self.smooth > 0 or self.soft > 0 or self.vector)

    def key(self) -> tuple:
        return (round(float(self.smooth), 2), round(float(self.soft), 2), bool(self.vector))


def _smooth_alpha(m: np.ndarray, amount: float, upscale: float = 1.0) -> np.ndarray:
    """Raster smoothing: blur, then a steep ramp around the half-way level (rounds the outline, edge stays crisp, ~1.5 px anti-aliasing)."""
    h, w = m.shape
    sigma = max(0.6, amount / 100.0 * 0.012 * max(h, w), amount / 100.0 * 1.5 * upscale)      # (a small mask enlarged a lot has big steps)
    f = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sigma)
    half = min(0.5, 0.75 / (2.507 * sigma))                           # the blurred edge is about 2.5 sigma wide; keep 1.5 px of it
    return np.clip((f - (0.5 - half)) / (2 * half), 0.0, 1.0)


def _smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a closed polyline (about one point per pixel), then a small push along the normals that gives back the area
    the smoothing took (a plain Gaussian shrinks rounded shapes)."""
    n = len(pts)
    r = int(min(n // 2 - 1, max(1, np.ceil(3 * sigma))))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([pts[-r:], pts, pts[:r]]) if r > 0 else pts
    sm = np.stack([np.convolve(ext[:, c], k, mode="valid") for c in (0, 1)], axis=1)

    def area(q):
        return 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    a0, a1 = area(pts), area(sm)
    t = np.roll(sm, -1, 0) - np.roll(sm, 1, 0)
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1) / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None] * (1.0 if a1 >= 0 else -1.0)
    per = float(np.hypot(*(np.roll(sm, -1, 0) - sm).T).sum())
    return sm + nrm * ((abs(a0) - abs(a1)) / max(per, 1e-9))


def vector_mask(mask: np.ndarray, w: int, h: int, smooth: float = 0.0) -> np.ndarray:
    """The mask as smooth curves, drawn anti-aliased at w x h (uint8 0..255). The mask is enlarged (so its pixel steps become tiny), its
    outlines are traced, smoothed (`smooth` 0..100 = how round) and filled with the even-odd rule, so holes stay holes. The result does
    not depend on the size of the mask: asking for a bigger picture gives the same curves, only drawn finer."""
    mh, mw = mask.shape[:2]
    long = max(mh, mw)
    u = int(min(6, max(2, np.ceil(2400.0 / long))))                    # enlarge the mask so that its steps are far below a pixel
    up = cv2.resize(mask, (mw * u, mh * u), interpolation=cv2.INTER_CUBIC)
    binary = cv2.copyMakeBorder((up >= 128).astype(np.uint8), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    cs, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ss = 4 if w * h <= 2.0e6 else 3 if w * h <= 5.0e6 else 2
    cw, ch = w * ss, h * ss
    canvas = np.zeros((ch, cw), np.uint8)
    sigma = (1.7 + smooth / 100.0 * 0.06 * long * 0.34) * u            # native pixels -> enlarged pixels (smooth is relative to the mask size)
    sx, sy = w / float(mw), h / float(mh)
    polys = []
    for c in cs:
        if len(c) < 8:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        sg = max(1.0, min(sigma, len(pts) / 8.0))                      # a tiny shape is not smoothed away
        polys.append(_smooth_closed(pts, sg))
    if not polys:
        return np.zeros((h, w), np.uint8)
    areas = [0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])) for q in polys]
    outer_sign = np.sign(areas[int(np.argmax(np.abs(areas)))]) or 1.0
    for q, a in zip(polys, areas):
        t = np.roll(q, -1, 0) - np.roll(q, 1, 0)
        nrm = np.stack([t[:, 1], -t[:, 0]], axis=1) / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None] * (1.0 if a >= 0 else -1.0)
        q = q + nrm * (0.5 if np.sign(a) == outer_sign else -0.5)       # the traced line runs through the centres of the edge pixels
        xs = ((q[:, 0] - 2 + 0.5) / u * sx) * ss                       # (border 2) -> mask pixels -> picture pixels -> supersampled canvas
        ys = ((q[:, 1] - 2 + 0.5) / u * sy) * ss
        poly = np.stack([xs, ys], axis=1)
        x0, y0 = np.floor(poly.min(axis=0)).astype(int) - 1
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 2
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(loc, [np.rint((poly - (x0, y0)) * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        canvas[y0:y1, x0:x1] ^= loc                                    # even-odd: a hole inside a shape is cut out again
    return cv2.resize(canvas * 255, (w, h), interpolation=cv2.INTER_AREA)


def styled_mask(mask: Optional[np.ndarray], w: int, h: int, style: Optional[MaskStyle] = None) -> Optional[np.ndarray]:
    """The finished mask at w x h: plain (just resized), smoothed, vector, and/or with a soft edge."""
    if mask is None:
        return None
    style = style or MaskStyle()
    if style.plain():
        return fit_mask(mask, w, h)
    if style.vector:
        m = vector_mask(mask, w, h, style.smooth)
    elif style.smooth > 0:
        m = np.rint(_smooth_alpha(fit_mask(mask, w, h), style.smooth, max(1.0, w / float(mask.shape[1]), h / float(mask.shape[0]))) * 255.0).astype(np.uint8)
    else:
        m = fit_mask(mask, w, h)
    if style.soft > 0:
        sigma = style.soft / 100.0 * 0.02 * max(h, w)
        if sigma >= 0.4:
            m = np.clip(np.rint(cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)), 0, 255).astype(np.uint8)
    return m


def traced_region(opaque: Optional[np.ndarray], mask: Optional[np.ndarray], shape) -> Optional[np.ndarray]:
    """The part of a picture that is vectorised when "only inside the mask" is on: the masked area (a pixel wider, so the edge of a soft
    mask is reached), and - as before - only where the picture is opaque. The tracer never looks at anything outside it: no shapes,
    no palette colours and no Threshold measurement from the facade. A missing or empty mask leaves the picture as it is."""
    if mask is None:
        return opaque
    h, w = shape[:2]
    m = fit_mask(mask, w, h) >= 8
    if not m.any():
        return opaque
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0       # a little wider than the mask: its smooth edge does the cutting, not the pixels
    return m if opaque is None else (opaque & m)


def _seam(h: int, w: int) -> int:
    """How far (pixels) the facade reaches under the edge of the mask, so no light ring is left between graffiti and facade."""
    return max(2, int(round(max(h, w) / 250.0)))


def facade_region(opaque: Optional[np.ndarray], mask: Optional[np.ndarray], shape) -> Optional[np.ndarray]:
    """The part of a picture that is NOT masked (the facade): what the independent facade vectoriser traces. It reaches a few pixels
    under the mask's edge (the graffiti is laid over it), and - as always - only where the picture is opaque."""
    if mask is None:
        return opaque
    h, w = shape[:2]
    k = _seam(h, w)
    m = fit_mask(mask, w, h) >= 128
    f = ~(cv2.erode(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0)
    return f if opaque is None else (opaque & f)


def composite_scrub(photo: np.ndarray, mask: np.ndarray, empty_bgr, fv: float, gra: Tuple[np.ndarray, Optional[np.ndarray]],
                    fac: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None) -> np.ndarray:
    """Layers, from the bottom: paper -> the facade photo (strength fv) -> the facade vector (`fac` = vector picture, ink 0..1 or None;
    outside the mask, reaching a little under its edge) -> the graffiti vector (`gra`, inside the mask). Only the ink of a vector is
    laid over what is below it."""
    vec, ink = gra
    h, w = vec.shape[:2]
    if photo.shape[:2] != (h, w):
        photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
    m = mask_f32(mask, w, h)
    e = np.array(empty_bgr, np.float32).reshape(1, 1, 3)
    look = _fade(photo, e, fv) if fv < 1.0 else photo.astype(np.float32)
    if fac is not None:
        fvec, fink = fac
        a = _mask_eroded_f32(mask, w, h, _seam(h, w))
        a = 1.0 - a if fink is None else _alpha(fink, w, h) * (1.0 - a)
        look = _lerp(look, fvec, a)
    ga = _alpha(ink, w, h)
    return _to_u8(_lerp(look, vec, m if ga is None else ga * m))


def has_layers(values: Dict[str, float]) -> bool:
    return any(k in values for k in LAYER_KEYS)


def _blur_facade(img: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Blur only the facade: its colours spread among themselves, the graffiti's colours do not bleed into it."""
    h, w = img.shape[:2]
    f = max(1, int(sigma // 3))                         # a big blur is (all but) the same on a small copy, and much quicker
    if f > 1:
        sw, sh = max(2, w // f), max(2, h // f)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        wf = cv2.resize(1.0 - m, (sw, sh), interpolation=cv2.INTER_AREA)
        s = sigma / f
    else:
        small, wf, s = img, 1.0 - m, sigma
    num = cv2.GaussianBlur(small * wf[..., None], (0, 0), s)
    den = cv2.GaussianBlur(wf, (0, 0), s)
    out = num / np.maximum(den, 1e-3)[..., None]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) if f > 1 else out


def _composite_photo(vec: np.ndarray, photo: np.ndarray, mask: np.ndarray, empty_bgr, fv: float, gv: float, bl: float,
                     ink: Optional[np.ndarray] = None) -> np.ndarray:
    """The facade is the original photo; only the masked part is the vectorization (`vec`, drawn on paper). gv = how much of the masked
    part is vector instead of photo, fv = how much of the photo facade is still there (the rest is paper), bl = facade blur.
    `ink` (0..1, how much of every pixel of `vec` is drawing): only the drawing is laid over the picture, the paper of the vector
    is the facade as it looks at that moment (so no white patches appear where the mask also covers a bit of wall).

    With look = the facade as it looks now (photo, blurred, faded to paper) the picture is
        look + m (1 - gv) (photo - look) + m gv ink (vec - look)            (m = the mask), i.e. two weights, no per-layer pictures."""
    h, w = vec.shape[:2]
    if photo.shape[:2] != (h, w):
        photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
    m = mask_f32(mask, w, h)
    a = _alpha(ink, w, h)
    sigma = bl / 100.0 * BLUR_MAX * max(h, w) if (bl > 0 and fv > 0) else 0.0
    if sigma < 0.5 and fv >= 1.0:
        # the usual case, the facade is the photo as it is: picture = photo + (vec - photo) * (m gv ink), and only where that is not 0
        wgt = m * np.float32(gv) if a is None else m * a * np.float32(gv)
        x, y, bw, bh = cv2.boundingRect((wgt > 0).view(np.uint8))
        out = photo.copy()
        if bw and bh:
            sl = (slice(y, y + bh), slice(x, x + bw))
            out[sl] = _to_u8(_lerp(photo[sl].astype(np.float32), vec[sl], wgt[sl]))
        return out
    ph = photo.astype(np.float32)
    if sigma >= 0.5:
        b = _blur_facade(ph, m, sigma)
        ph = _lerp(b, ph, m)                                       # the graffiti keeps its own colours, the facade is blurred
    e = np.array(empty_bgr, np.float32).reshape(1, 1, 3)
    look = _fade(ph, e, fv) if fv < 1.0 else ph     # the facade as it looks now (also behind the vector's paper)
    wv = m * np.float32(gv) if a is None else m * a * np.float32(gv)
    out = _lerp(look, vec, wv)
    if gv < 1.0 and fv < 1.0:                                      # (the photo part of the mask is brighter than the faded facade around it)
        d = np.subtract(ph, look)
        d *= (m * np.float32(1.0 - gv))[..., None]
        out += d
    return _to_u8(out)


def composite(img: np.ndarray, mask: Optional[np.ndarray], empty_bgr, values: Dict[str, float],
              photo: Optional[np.ndarray] = None, ink: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the layer values to a finished picture (BGR uint8). `empty_bgr` is the colour of a layer that is not visible.
    `photo` (the original picture) is needed for the "facade = original photo" mode (layer_p)."""
    if mask is None or not has_layers(values):
        return img
    if values.get("layer_p", 0.0) >= 0.5 and photo is not None:
        return _composite_photo(img, photo, mask, empty_bgr, float(min(max(values.get("layer_f", 1.0), 0.0), 1.0)),
                                float(min(max(values.get("layer_g", 1.0), 0.0), 1.0)), float(min(max(values.get("blur_f", 0.0), 0.0), 100.0)), ink)
    fv = float(min(max(values.get("layer_f", 1.0), 0.0), 1.0))
    gv = float(min(max(values.get("layer_g", 1.0), 0.0), 1.0))
    bl = float(min(max(values.get("blur_f", 0.0), 0.0), 100.0))
    if fv >= 1.0 and gv >= 1.0 and bl <= 0.0:
        return img
    h, w = img.shape[:2]
    m = mask_f32(mask, w, h)
    pic = img.astype(np.float32)
    if bl > 0 and fv > 0:
        sigma = bl / 100.0 * BLUR_MAX * max(h, w)
        if sigma >= 0.5:
            b = _blur_facade(pic, m, sigma)
            pic = _lerp(b, pic, m)
    vis = m * np.float32(gv - fv) + np.float32(fv)                  # gv m + fv (1 - m)
    empty = np.array(empty_bgr, np.float32).reshape(1, 1, 3)
    return _to_u8(_lerp(np.broadcast_to(empty, pic.shape), pic, vis))
