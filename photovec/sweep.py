"""Parameter sweeps: animate Image Trace / Linework Snap sliders (Threshold, Paths, Corners, Noise …).

One *step* of a slider is one frame: every frame is a real re-trace of the picture with the slider values of that frame.
Frames are cached as small PNGs and rendered in parallel (several processes, stage caches inside each process)."""
from __future__ import annotations

import atexit
import hashlib
import math
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .layers import LAYER_KEYS, composite, has_layers

# name -> (label, minimum, maximum, discrete, engines that have it)
# discrete = only whole numbers make sense (pixels, colours); the other sliders are continuous, so a sweep can use
# thousands of tiny steps and every frame still looks a little different.
SWEEP_PARAMS: Dict[str, Tuple[str, float, float, bool, Tuple[str, ...]]] = {
    "threshold": ("Threshold", 0, 255, False, ("trace", "linework")),
    "paths": ("Paths", 0, 100, False, ("trace",)),
    "corners": ("Corners", 0, 100, False, ("trace", "linework")),
    "noise": ("Noise", 1, 100, True, ("trace",)),
    "colors": ("Colors", 2, 64, True, ("trace",)),
    "fill": ("Fill", 0, 1, False, ("trace",)),
    "layer_f": ("Facade", 0, 1, False, ("trace", "linework")),       # layers: see layers.py (only used when the picture has a mask)
    "layer_g": ("Graffiti", 0, 1, False, ("trace", "linework")),
    "blur_f": ("Facade blur", 0, 100, False, ("trace", "linework")),
    "layer_p": ("Facade = photo", 0, 1, False, ("trace", "linework")),
    "smooth": ("Smooth", 0, 100, False, ("trace",)),
    "simplify": ("Simplify", 0, 100, False, ("linework",)),
    "min_len": ("Min. length", 0, 200, False, ("linework",)),
    "gap": ("Gap", 0, 60, False, ("linework",)),
}


@dataclass
class Track:
    """One slider: moves from `start` to `end` by `step` per frame (1 step = 1 frame), after `delay` frames."""
    param: str = "paths"
    start: float = 0.0
    end: float = 50.0
    step: float = 1.0
    delay: int = 0
    warp: Optional[Callable[[float], float]] = None     # progress 0..1 -> value, instead of equal steps (made by "finish together")

    def steps(self) -> int:
        return max(1, int(math.ceil(abs(self.end - self.start) / max(self.step, 1e-6) - 1e-9)))

    def value(self, i: int) -> float:
        """Slider value after i steps (i = 0 .. steps), stopping exactly on `end`."""
        if self.warp is not None:
            u = min(max(i, 0) / max(1, self.steps()), 1.0)
            _, lo, hi, discrete, _ = SWEEP_PARAMS[self.param]
            v = self.start if u <= 0 else self.end if u >= 1 else self.warp(u)
            v = min(max(v, lo), hi)
            return float(int(round(v))) if discrete else round(v, 4)
        d = min(max(i, 0) * max(self.step, 1e-6), abs(self.end - self.start))
        v = self.start + math.copysign(d, self.end - self.start)
        _, lo, hi, discrete, _ = SWEEP_PARAMS[self.param]
        v = min(max(v, lo), hi)
        return float(int(round(v))) if discrete else round(v, 4)

    @classmethod
    def with_steps(cls, param: str, start: float, end: float, steps: int, delay: int = 0) -> "Track":
        """A slider that needs exactly `steps` frames to get from start to end."""
        return cls(param, float(start), float(end), abs(end - start) / max(1, steps), delay)

    def label(self) -> str:
        return f"{SWEEP_PARAMS[self.param][0]} {self.start:g}→{self.end:g}"


def default_track(param: str, current: float) -> Track:
    """A new slider that ends on the value currently set in the main window (so the next stage joins seamlessly)."""
    label, lo, hi, discrete, _ = SWEEP_PARAMS[param]
    start = lo if current - lo >= hi - current else hi
    return Track(param, float(start), float(current), 1.0 if hi - lo >= 50 else 0.5)


def merge_tracks(base: Sequence[Track], override: Sequence[Track]) -> List[Track]:
    """A row's own tracks replace the stage's track of the same slider; other stage tracks stay."""
    if not override:
        return list(base)
    own = {t.param: t for t in override}
    out = [own.pop(t.param, t) for t in base]
    return out + list(own.values())


def apply_values(params, values: Dict[str, float]):
    """Trace / linework parameters with the slider values of one frame (sliders the engine doesn't have are ignored)."""
    kw = {}
    for k, v in values.items():
        if k in ("spread", "flip"):
            if hasattr(params, k):
                kw[k] = bool(v)
        elif hasattr(params, k):
            kw[k] = int(round(v)) if SWEEP_PARAMS[k][3] else float(v)
    return replace(params, **kw)


def norm_values(src: "SweepSource", values: Dict[str, float]) -> Dict[str, float]:
    """Values as the engine can actually use them (Linework Snap works on an 8-bit ink map: its threshold is whole),
    so frames that would look identical share one cache entry."""
    if src.linework and "threshold" in values:
        values = dict(values)
        values["threshold"] = float(int(round(values["threshold"])))
    return values


def values_key(values: Dict[str, float]) -> tuple:
    return tuple(sorted((k, round(float(v), 4)) for k, v in values.items()))


@dataclass
class SweepSource:
    """What is needed to re-trace one picture."""
    bgr: np.ndarray
    opaque: Optional[np.ndarray]
    params: object                      # TraceParams or LineworkParams
    linework: bool = False
    mask: Optional[np.ndarray] = None   # layer mask (uint8, picture size): masked = graffiti, the rest = facade

    @classmethod
    def from_result(cls, res) -> "SweepSource":
        lw = getattr(res, "lw_params", None)
        return cls(res.image_bgr, res.opaque, lw if lw is not None else res.params, lw is not None, getattr(res, "mask", None))


_NOISE: dict = {}
FILL_ORIGINS = ("drawing", "edges", "centre")


def _style(p) -> tuple:
    """The fill settings of a parameter set (works for anything with the fill_* fields, with defaults for the rest)."""
    return (round(float(getattr(p, "fill_noise", 0.45)), 4), round(float(getattr(p, "fill_size", 1.0)), 4),
            round(float(getattr(p, "fill_detail", 0.5)), 4), round(float(getattr(p, "fill_smooth", 0.25)), 4),
            round(float(getattr(p, "fill_pace", 1.0)), 4), int(getattr(p, "fill_seed", 7)), str(getattr(p, "fill_origin", "drawing")))


def _noise_field(h: int, w: int, size: float = 1.0, detail: float = 0.5, smooth: float = 0.25, seed: int = 7) -> np.ndarray:
    """Smooth, blobby noise (0..1), the same for every call with the same settings. Only soft blobs and no pixel grain, so the
    fronts it draws are made of smooth curves that the vector fit can follow. `size` scales the blobs, `detail` mixes in smaller
    ones, `smooth` rounds the result, `seed` picks the pattern."""
    key = (h, w, round(size, 3), round(detail, 3), round(smooth, 3), int(seed))
    f = _NOISE.get(key)
    if f is None:
        rng = np.random.default_rng(int(seed))
        d = min(max(detail, 0.0), 1.0)
        weights = np.array([0.55 * (1 - d) + 0.15, 0.3, 0.55 * d + 0.15], np.float64)
        acc = np.zeros((h, w), np.float32)
        for cells, wt in zip((4, 9, 20), weights / weights.sum()):
            c = max(1.2, cells / max(size, 0.1))
            ch = max(2, int(round(c * h / max(h, w))))
            cw = max(2, int(round(c * w / max(h, w))))
            acc += float(wt) * cv2.resize(rng.random((ch, cw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        sigma = max(1.0, (0.0005 + 0.006 * min(max(smooth, 0.0), 1.0)) * max(h, w))
        acc = cv2.GaussianBlur(acc, (0, 0), sigma)
        lo, hi = np.percentile(acc, (2, 98))
        f = np.clip((acc - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)
        while len(_NOISE) >= 4:
            _NOISE.pop(next(iter(_NOISE)))
        _NOISE[key] = f
    return f


def _fill_field(dark: np.ndarray, p=None) -> np.ndarray:
    """Score (0..1) of every pixel: small = filled early. A mix of the distance from the drawing (and from the picture's edges or
    centre, if chosen) and smooth noise."""
    amount, size, detail, smooth, _pace, seed, origin = _style(p)
    h, w = dark.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if dark.any():
        d = cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3)
        d = d / max(float(d.max()), 1e-6)
    else:
        d = np.ones((h, w), np.float32)
    if origin == "edges":
        e = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        d = np.minimum(d, e / max(float(e.max()), 1.0))
    elif origin == "centre":
        c = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
        d = np.minimum(d, c / max(float(c.max()), 1.0))
    elif not dark.any():                                      # nothing drawn: start from the middle
        c = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
        d = c / max(float(c.max()), 1.0)
    a = min(max(amount, 0.0), 1.0)
    return ((1.0 - a) * d + a * _noise_field(h, w, size, detail, smooth, seed)).astype(np.float32)


def fill_front(p) -> float:
    """How far the front has got in score terms: the progress with the 'pace' curve applied."""
    return float(min(max(p.fill, 0.0), 1.0) ** max(float(getattr(p, "fill_pace", 1.0)), 0.05))


def fill_state(src: "SweepSource", p, cache: Optional[dict], vkey):
    """(dark, field) of a picture at its current slider values: the pixels that are drawing (after the colour swap) and the score
    of every pixel for the spreading (small = filled early)."""
    from .trace import _label_key, _remember, clean_labels, make_labels
    noise = max(1, int(round(p.noise * p.scale ** 2)))
    ck = ("cln", _label_key(p), noise)
    cl = cache.get(ck) if cache is not None else None
    if cl is None:
        lk = ("lab", _label_key(p))
        hit = cache.get(lk) if cache is not None else None
        if hit is None:
            hit = make_labels(src.bgr, p, src.opaque)
            if cache is not None:
                _remember(cache, lk, hit)
        cl = clean_labels(hit[0], noise, len(hit[1]))
        if cache is not None:
            _remember(cache, ck, cl)
    dark = cl == 1
    if getattr(p, "flip", False):
        dark = ~dark
    fk = ("fillfield", vkey, _style(p), bool(getattr(p, "flip", False)))
    field = cache.get(fk) if cache is not None else None
    if field is None:
        field = _fill_field(dark, p)
        if cache is not None:
            for k in [k for k in cache if isinstance(k, tuple) and k and k[0] == "fillfield" and k != fk]:
                cache.pop(k)
            cache[fk] = field
    return dark, field


def fill_vector(src: "SweepSource", p, bg: str, cache: Optional[dict], vkey) -> np.ndarray:
    """The drawing colour spreading over the picture, drawn as VECTORS: the region that is filled so far (the drawing plus the
    spreading front) is traced like any other picture, so its outline is made of the same smooth, fitted curves as the rest of
    the drawing, and small stray spots are dropped by the Noise setting exactly as they are elsewhere. No pixel filter."""
    from dataclasses import replace as rep
    from .animate import vector_image
    from .trace import _label_key, trace_bgr
    h, w = src.bgr.shape[:2]
    if p.fill >= 1:
        return np.zeros((h, w, 3), np.uint8)
    dark, field = fill_state(src, p, cache, vkey)
    mask = dark | (field <= fill_front(p))
    p2 = rep(p, fill=0.0, flip=False)
    res = trace_bgr(src.bgr, p2, src.opaque, {("lab", _label_key(p2)): (mask.astype(np.int16), [(255, 255, 255), (0, 0, 0)], 0)})
    return vector_image(res, bg)


def fill_map(src: "SweepSource", p, levels, cache: Optional[dict] = None, vkey=None) -> np.ndarray:
    """A picture of the noise: the score of every pixel as colours (dark blue = filled first, red = last), the drawing in black
    and a white line where the front is at each of `levels` (fill progress 0..1, the pace curve is applied here)."""
    from dataclasses import replace as rep
    dark, field = fill_state(src, p, cache, vkey)
    heat = cv2.applyColorMap(np.clip(field * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[dark] = (20, 20, 20)
    for lv in levels:
        m = ((dark | (field <= fill_front(rep(p, fill=float(lv))))).astype(np.uint8)) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heat, cs, -1, (255, 255, 255), max(1, int(round(max(heat.shape[:2]) / 400))), cv2.LINE_AA)
    return heat


def fill_ink(img: np.ndarray, progress: float, amount: float = 0.45, cache: Optional[dict] = None, key=None, p=None) -> np.ndarray:
    """Spread the drawing colour (black) from the existing drawing over the whole picture, with an uneven, noisy front:
    `progress` 0 = unchanged, 1 = all black. Pictures with a pure-white background never turn black through the Threshold
    (a pixel that is 255 is never darker than 255), so this finishes them. The front moves with the distance from the drawing
    mixed with noise (`amount` 0..1), so it grows like ink soaking in instead of as a clean outline."""
    if progress <= 0:
        return img
    if progress >= 1:
        return np.zeros_like(img)
    h, w = img.shape[:2]
    from dataclasses import replace as _rep
    from .trace import TraceParams
    sp = _rep(p if p is not None else TraceParams(), fill_noise=float(amount))
    progress = float(min(max(progress, 0.0), 1.0) ** max(float(getattr(sp, "fill_pace", 1.0)), 0.05))
    ck = ("fillfield", key, _style(sp))
    field = cache.get(ck) if cache is not None else None
    if field is None:
        field = _fill_field(img.mean(axis=2) < 128, sp)
        if cache is not None:
            for k in [k for k in cache if isinstance(k, tuple) and k and k[0] == "fillfield" and k != ck]:
                cache.pop(k)
            cache[ck] = field
    soft = 0.012
    front = progress * (1.0 + soft) - soft / 2
    alpha = np.clip((front - field) / soft + 0.5, 0.0, 1.0)
    dark = np.maximum(1.0 - img.mean(axis=2) / 255.0, alpha)
    return np.repeat(np.rint((1.0 - dark) * 255.0).astype(np.uint8)[..., None], 3, axis=2)


def recolor(img: np.ndarray, ink=None, paper=None) -> np.ndarray:
    """A black-and-white drawing (BGR) with its black replaced by `ink` and its white by `paper` (RGB tuples); the grey of
    anti-aliased edges becomes the matching mix."""
    ink = (0, 0, 0) if ink is None else ink
    paper = (255, 255, 255) if paper is None else paper
    g = (img.astype(np.float32).mean(axis=2, keepdims=True)) / 255.0
    a = np.array(ink[::-1], np.float32)
    b = np.array(paper[::-1], np.float32)
    return np.rint(a + (b - a) * g).astype(np.uint8)


def render_values(src: SweepSource, bg: str, values: Dict[str, float], cache: Optional[dict] = None, parts: bool = False):
    """Re-trace with the slider values and draw the vector on `bg` (BGR, picture size).
    parts=True: do not apply the layers; return (vector picture, ink 0..1 or None, colour of paper (BGR)) so that a window can lay
    the facade photo behind the drawing at any strength without tracing again."""
    from .animate import vector_image
    from .linework import linework_bgr
    from .trace import trace_bgr
    p = apply_values(src.params, values)
    ink, paper = getattr(p, "ink", None), getattr(p, "paper", None)
    bgk = "white" if (ink or paper) else bg                           # coloured pictures: transparent areas are paper
    fill = getattr(p, "fill", 0.0) > 0
    vkey = tuple(sorted((k, round(float(v), 4)) for k, v in values.items() if k != "fill" and k not in LAYER_KEYS)) if fill else None
    photo_mode = src.mask is not None and values.get("layer_p", 0.0) >= 0.5
    drawn = True
    if photo_mode and values.get("layer_g", 1.0) <= 0.0:
        img = src.bgr                                                  # nothing is vectorized yet: the picture is the photo (no tracing needed)
        drawn = False
    elif fill and not src.linework and getattr(p, "mode", "") == "bw":
        img = fill_vector(src, p, bgk, cache, vkey)                    # the spreading drawing colour, traced as vectors
    else:
        res = linework_bgr(src.bgr, p, src.opaque, cache) if src.linework else trace_bgr(src.bgr, p, src.opaque, cache)
        img = vector_image(res, bgk)
        if getattr(p, "flip", False):
            img = 255 - img
        if fill:                                                       # colour / grey presets: no labels to trace, pixel fill
            img = fill_ink(img, p.fill, getattr(p, "fill_noise", 0.45), cache, vkey, p)
    ink_a = None
    if photo_mode and drawn and getattr(p, "mode", "") == "bw":
        ink_a = 1.0 - img.astype(np.float32).mean(axis=2) / 255.0       # how much of every pixel is drawing (before the colours are applied)
    img = recolor(img, ink, paper) if (drawn and (ink or paper)) else img
    if parts or (src.mask is not None and has_layers(values)):
        from .animate import background
        e = background(1, 1, bgk)
        if getattr(p, "flip", False):
            e = 255 - e
        if ink or paper:
            e = recolor(e, ink, paper)
        if parts:
            return img, ink_a, tuple(int(x) for x in e[0, 0])
        img = composite(img, src.mask, e[0, 0], values, photo=src.bgr if photo_mode else None, ink=ink_a)
    return img


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()


# --------------------------------------------------------------------------- worker side
_SRC_CACHE: "OrderedDict[str, SweepSource]" = OrderedDict()


def _load_source(path: str, params, linework: bool) -> SweepSource:
    """Worker: the picture of a sweep source from its temp file (a few are kept, so chunks of one picture load it once)."""
    src = _SRC_CACHE.get(path)
    if src is None:
        with np.load(path) as z:
            op = z["opaque"] if "opaque" in z.files else None
            src = SweepSource(z["bgr"], op, params, linework, z["mask"] if "mask" in z.files else None)
        _SRC_CACHE[path] = src
        while len(_SRC_CACHE) > 3:
            _SRC_CACHE.popitem(last=False)
    else:
        _SRC_CACHE.move_to_end(path)
    return src


def _sweep_chunk(args) -> List[bytes]:
    """Worker process: one picture, several frames in sweep order (the stage cache lives for the whole chunk)."""
    path, params, linework, bg, vlist = args
    src = _load_source(path, params, linework)
    cache: dict = {}
    return [_png(render_values(src, bg, dict(v), cache)) for v in vlist]


# --------------------------------------------------------------------------- frame file
_TEMP_FILES: set = set()


def _cleanup_temp() -> None:
    for pth in list(_TEMP_FILES):
        try:
            os.remove(pth)
        except OSError:
            pass
        _TEMP_FILES.discard(pth)


atexit.register(_cleanup_temp)


class FrameFile:
    """Append-only temporary file that holds the compressed frames, so thousands of sweep frames do not fill the RAM.
    Identical frames (common when a slider moves in tiny steps) are stored once."""

    def __init__(self, directory: Optional[str] = None):
        fd, self.path = tempfile.mkstemp(prefix="photovec_sweep_", suffix=".bin", dir=directory)
        self._fh = os.fdopen(fd, "w+b")
        _TEMP_FILES.add(self.path)
        self._lock = threading.Lock()
        self._dedupe: Dict[bytes, Tuple[int, int]] = {}
        self.size = 0

    def put(self, data: bytes) -> Tuple[int, int]:
        h = hashlib.blake2b(data, digest_size=10).digest()
        with self._lock:
            hit = self._dedupe.get(h)
            if hit is not None:
                return hit
            self._fh.seek(0, os.SEEK_END)
            off = self._fh.tell()
            self._fh.write(data)
            self.size = off + len(data)
            self._dedupe[h] = (off, len(data))
            return off, len(data)

    def read(self, loc: Tuple[int, int]) -> bytes:
        with self._lock:
            self._fh.seek(loc[0])
            return self._fh.read(loc[1])

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            try:
                os.remove(self.path)
            except OSError:
                pass
            _TEMP_FILES.discard(self.path)


class SweepStore:
    """Compressed sweep frames of all pictures of a scene, keyed by (picture, background, slider values).
    The frames live in a temporary file (see FrameFile); RAM only holds the index and a small cache of decoded frames."""

    CACHE_BYTES = 96 * 1024 * 1024

    def __init__(self, sources: Sequence[SweepSource]):
        self.sources = list(sources)
        self.frames: Dict[tuple, Tuple[int, int]] = {}          # key -> (offset, length) in the frame file
        self._file: Optional[FrameFile] = None
        self._sess: Dict[int, dict] = {}
        self._srcfiles: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._lru: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._lru_bytes = 0

    # -- storage
    def _ff(self) -> FrameFile:
        if self._file is None:
            self._file = FrameFile()
        return self._file

    def _put(self, key: tuple, data: bytes) -> None:
        self.frames[key] = self._ff().put(data)

    def read(self, key: tuple) -> bytes:
        return self._ff().read(self.frames[key])

    @property
    def disk_bytes(self) -> int:
        return self._file.size if self._file else 0

    def close(self) -> None:
        """Delete the temporary files."""
        if self._file is not None:
            self._file.close()
            self._file = None
        for pth in self._srcfiles.values():
            try:
                os.remove(pth)
            except OSError:
                pass
            _TEMP_FILES.discard(pth)
        self._srcfiles.clear()
        self.frames.clear()
        self._lru.clear()
        self._lru_bytes = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _src_path(self, k: int) -> str:
        pth = self._srcfiles.get(k)
        if pth is None:
            fd, pth = tempfile.mkstemp(prefix="photovec_src_", suffix=".npz")
            os.close(fd)
            _TEMP_FILES.add(pth)
            src = self.sources[k]
            arrs = {"bgr": src.bgr}
            if src.opaque is not None:
                arrs["opaque"] = src.opaque
            if src.mask is not None:
                arrs["mask"] = src.mask
            np.savez(pth, **arrs)
            self._srcfiles[k] = pth
        return pth

    # -- access
    def _key(self, k: int, bg: str, values: Dict[str, float]) -> tuple:
        return (k, bg, values_key(norm_values(self.sources[k], values)))

    def has(self, key: tuple) -> bool:
        return key in self.frames

    def get(self, k: int, bg: str, values: Dict[str, float]) -> np.ndarray:
        key = self._key(k, bg, values)
        with self._lock:
            img = self._lru.get(key)
            if img is not None:
                self._lru.move_to_end(key)
                return img.copy()
            if key not in self.frames:                  # not rendered yet: do it here (slow path, one at a time)
                sess = self._sess.setdefault(k, {})
                self._put(key, _png(render_values(self.sources[k], bg, values, sess)))
            data = self.read(key)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        with self._lock:
            self._lru[key] = img
            self._lru_bytes += img.nbytes
            while self._lru_bytes > self.CACHE_BYTES and len(self._lru) > 1:
                _, old = self._lru.popitem(last=False)
                self._lru_bytes -= old.nbytes
        return img.copy()

    def ensure(self, needs: Iterable[Tuple[int, str, tuple]], workers: int = 1, pool=None,
               progress: Optional[Callable[[int, int], bool]] = None, chunk: int = 48) -> bool:
        """Render every missing (picture, bg, values) frame, several processes at once. Returns False when cancelled.
        Work is handed out in small chunks and the results are written to disk as they arrive, so memory stays flat."""
        from . import pool as pool_mod
        todo: Dict[int, List[Tuple[tuple, str, tuple]]] = {}
        seen = set()
        for k, bg, vk in needs:
            key = (k, bg, vk)
            if key in self.frames or key in seen:
                continue
            seen.add(key)
            todo.setdefault(k, []).append((key, bg, vk))
        del seen
        total = sum(len(v) for v in todo.values())
        if not total:
            return True
        done = 0
        if progress and progress(0, total):
            return False
        if pool is None or workers <= 1:
            for k, items in todo.items():
                sess = self._sess.setdefault(k, {})
                for key, bg, vk in items:
                    self._put(key, _png(render_values(self.sources[k], bg, dict(vk), sess)))
                    done += 1
                    if progress and progress(done, total):
                        return False
            return True
        # chunks of consecutive frames of one picture; pictures take turns so every picture's first frames come first
        per_pic = {k: [items[i:i + chunk] for i in range(0, len(items), chunk)] for k, items in todo.items()}
        order: List[Tuple[int, List]] = []
        for r in range(max(len(v) for v in per_pic.values())):
            for k, parts in per_pic.items():
                if r < len(parts):
                    order.append((k, parts[r]))
        paths = {k: self._src_path(k) for k in per_pic}
        jobs = []
        for k, part in order:
            src = self.sources[k]
            jobs.append((paths[k], src.params, src.linework, part[0][1], [p[2] for p in part]))
        fin = [0]

        def sink(i: int, data: List[bytes]) -> None:          # frames go straight to disk, nothing piles up in RAM
            for (key, _, _), d in zip(order[i][1], data):
                self._put(key, d)
            fin[0] += len(order[i][1])

        def prog(_d: int, _n: int) -> bool:
            return bool(progress and progress(fin[0], total))

        _, cancelled = pool_mod.map_jobs(_sweep_chunk, jobs, workers, prog, None, sink)
        if cancelled:
            return False
        if progress:
            progress(total, total)
        return True
