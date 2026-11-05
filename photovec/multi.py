"""Multi-Trace: several photos, each with its own Threshold / Noise range, one shared timeline, one clip.

This module has no GUI: it turns the settings of the window into per-photo slider tracks (`make_tracks`), the value of a slider at
a frame (`values_at`, used by the live preview and by the export, so both always show the same), and the renderer of the
finished clip (`build_renderer`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .sweep import SweepSource, Track

THRESHOLD = (0, 255)
NOISE = (1, 100)
EASES = ("linear", "smooth")

# Pop-art colour pairs (ink, paper), after the flat, loud colour blocks of Andy Warhol's screen prints.
FILL_KEYS = ("fill_noise", "fill_size", "fill_detail", "fill_smooth", "fill_pace", "fill_seed", "fill_origin")
HEX = lambda h: (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
WARHOL: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    name: [(HEX(a), HEX(b)) for a, b in pairs] for name, pairs in {
        "Marilyn": [("#1b2a8a", "#ff4fa3"), ("#111111", "#ffe600"), ("#d6007f", "#00c2c7"), ("#3b1f78", "#ff7a00"),
                    ("#e3001b", "#8fd3ff"), ("#0a8f3c", "#ffb3d9"), ("#102a6e", "#c6f000"), ("#7a0f7a", "#ffd0e8")],
        "Soup cans": [("#c8102e", "#ffffff"), ("#ffffff", "#c8102e"), ("#111111", "#f4c20d"), ("#f4c20d", "#c8102e"),
                      ("#c8102e", "#f4e7c3"), ("#1d3c8f", "#f4c20d")],
        "Electric": [("#00e5ff", "#1a0033"), ("#ff2bd6", "#0b1d4d"), ("#c6ff00", "#2a0a4a"), ("#ff9100", "#10216b"),
                     ("#ffffff", "#e0004d"), ("#ffe600", "#4a00b0")],
        "Pastel pop": [("#5b4a8a", "#ffd6e7"), ("#2f6f73", "#fff3b0"), ("#8a3b52", "#c9f0e1"), ("#3d5a99", "#ffe0c2"),
                       ("#6b3fa0", "#d4f1ff"), ("#9a3d20", "#e6f5c8")],
        "Black and one colour": [("#000000", "#ff2d95"), ("#000000", "#ffe600"), ("#000000", "#00d1ff"),
                                 ("#000000", "#ff7a00"), ("#000000", "#7cff3a"), ("#000000", "#b36bff")],
    }.items()}


def palette_pairs(name: str, n: int, shuffle: int = -1):
    """n (ink, paper) pairs from a Warhol palette, repeating if the palette is shorter; shuffle >= 0 = a seeded random order."""
    pairs = list(WARHOL[name])
    if shuffle >= 0:
        import random
        random.Random(shuffle).shuffle(pairs)
    return [pairs[i % len(pairs)] for i in range(n)]


@dataclass
class PanelSpec:
    """One photo: where its sliders are at the start, at every slider point in between, and at the end of the clip.
    `thr`, `noise` (and `paths`) hold one value per slider point, in time order. The points are spread evenly over the movement,
    so every photo reaches its point 2, its point 3 ... in the same frame, whatever the values are."""
    path: str = ""
    thr: Tuple[float, ...] = (0, 255)
    noise: Tuple[float, ...] = (1, 25)
    flip: bool = False                # black ↔ white for this picture only (the clip's own Invert still applies on top)
    paths: Optional[Tuple[float, ...]] = None   # Image Trace "Paths" slider; None = not animated (the preset's value)
    ink: Optional[Tuple[int, int, int]] = None  # colour of the drawing (RGB); None = black
    paper: Optional[Tuple[int, int, int]] = None  # colour of the paper (RGB); None = white (or the clip's custom background)
    fill_over: Optional[dict] = None   # this photo's own noise settings for the fill (keys of FILL_KEYS); None = the clip's
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)   # layer mask (uint8, any size): masked = graffiti, rest = facade


@dataclass
class MultiSettings:
    duration: float = 10.0            # seconds the sliders take from start to end
    fps: int = 25
    hold_start: float = 0.5           # seconds the clip rests at the start values / at the end values
    hold_end: float = 0.5
    ease: str = "linear"              # linear | smooth (slow start and end)
    preset: str = "Black and White Logo"
    tile: int = 1200                  # Quality: long side (px) every photo is traced at (the clip is scaled down from it)
    columns: int = 3
    gap: int = 12
    width: int = 1920                 # width of the finished clip
    bg: str = "white"
    invert: bool = False
    smooth: bool = True               # export: 2x supersampling
    points: int = 2                   # slider points per slider (start, end and those in between); the movement is cut into equal parts
    seg_seconds: Optional[Tuple[float, ...]] = None   # seconds of every part between two slider points (None = equal parts of `duration`)
    bg_color: Optional[Tuple[int, int, int]] = None   # custom colour between the pictures (and behind photos that are not square); None = bg
    fill: bool = False                # an OVERLAY: the drawing colour spreads over the whole picture while the sliders move and is complete
                                      # at the end of the movement (white backgrounds never turn black). It adds no frames.
    layers: bool = False              # use the photos' masks: facade and graffiti get their own visibility (and the facade a blur) over time
    fac0: float = 100.0               # facade visibility at the start / end of the layer movement (percent)
    fac1: float = 0.0
    gra0: float = 100.0               # graffiti visibility
    gra1: float = 100.0
    blur0: float = 0.0                # facade blur (percent of 3 % of the picture's long side)
    blur1: float = 40.0
    layer_start: float = 0.0          # when the layer movement starts, in seconds after the start of the movement (0 = with the image)
    layer_end: float = 0.0            # when it ends (0 = at the end of the movement)
    facade_photo: bool = False        # the facade is the ORIGINAL PHOTO (not a vector); only the masked part is vectorized.
                                      # Facade = how much of the photo is left (fades to paper), Graffiti = share of vector in the mask.
    only_mask: bool = False           # vectorise ONLY inside the photo's mask (the tracer never sees the facade); the background stays the
                                      # original photo, and the Facade slider (timed) fades it out
    gra_own: bool = False             # the graffiti has its own timing (below) instead of the layer window above
    gra_start: float = 0.0
    gra_end: float = 0.0
    fill_start: float = 0.0           # when it starts, in seconds after the start of the movement (global; 0 = together with the image)
    fill_noise: float = 0.45          # 0 = even growth from the drawing, 1 = mostly noise
    fill_size: float = 1.0            # size of the noise blobs
    fill_detail: float = 0.5          # 0 = only large soft blobs, 1 = many small ones too
    fill_smooth: float = 0.25         # 0 = ragged front, 1 = very round
    fill_pace: float = 1.0            # progress ** pace (above 1: slow start, fast end)
    fill_seed: int = 7                # another number = another noise pattern
    fill_origin: str = "drawing"      # drawing | edges | centre
    soften: bool = False              # blur the grey picture before the Threshold: outlines move smoothly instead of in jumps
    mask_smooth: float = 0.0          # finishing of the masks: round the outline (0..100), see layers.MaskStyle
    mask_soft: float = 0.0            # ... a soft edge (0..100)
    mask_vector: bool = False         # ... rebuild the outline as smooth curves (no mask pixels, sharp at any size)
    timeline: object = field(default=None, compare=False, repr=False)   # Timeline window: keyframed lanes replace the slider points (see timeline.py)

    def fill_style(self) -> dict:
        """The noise settings of the fill as TraceParams fields."""
        return {k: getattr(self, k) for k in FILL_KEYS}

    def base_params(self):
        """Image Trace settings of the preset, set up for the clip's quality: Noise and the fitting tolerances grow with the
        resolution (a Noise of 25 means the same share of the picture at 600 px and at 3000 px)."""
        from dataclasses import replace as rep
        from .trace import params_from_preset
        return rep(params_from_preset(self.preset), scale_cap=8.0, soften=1.0 if self.soften else 0.0, **self.fill_style())

    def segment_frames(self) -> List[int]:
        """Frames of every part between two slider points (the same for all photos). Each part is a whole number of frames, so
        every point falls on a real frame; the clip may be a few frames longer or shorter than duration × fps."""
        n = max(2, int(self.points))
        if self.seg_seconds is not None and len(self.seg_seconds) == n - 1:
            return [max(1, int(round(float(t) * self.fps))) for t in self.seg_seconds]
        base = max(2, int(round(self.duration * self.fps)))
        return [max(1, int(round((base - 1) / (n - 1))))] * (n - 1)

    def move_frames(self) -> int:
        return sum(self.segment_frames()) + 1

    def point_offsets(self) -> List[int]:
        """Frame (counted from the start of the movement) of every slider point."""
        out = [0]
        for f in self.segment_frames():
            out.append(out[-1] + f)
        return out

    def holds(self) -> Tuple[int, int]:
        return max(0, int(round(self.hold_start * self.fps))), max(0, int(round(self.hold_end * self.fps)))

    def fill_window(self) -> Tuple[float, float]:
        """First and last frame of the spreading, counted from the start of the movement (it ends where the movement ends)."""
        end = float(self.move_frames() - 1)
        return min(max(0.0, self.fill_start * self.fps), max(0.0, end - 1.0)), end

    def total_frames(self) -> int:
        hs, he = self.holds()
        return hs + self.move_frames() + he


def _ease(p: float, kind: str) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p) if kind == "smooth" else p


def progress_at(st: MultiSettings, frame: float) -> float:
    """How far the sliders have moved (0 = start values, 1 = end values) at a frame of the clip."""
    hs, _ = st.holds()
    return _ease((frame - hs) / max(1, st.move_frames() - 1), st.ease)


def interp(vals: Sequence[float], p: float) -> float:
    """Value at progress p (0..1) of a slider that passes through `vals` at evenly spaced moments."""
    n = len(vals)
    if n == 1:
        return float(vals[0])
    x = min(max(p, 0.0), 1.0) * (n - 1)
    i = min(int(x), n - 2)
    return float(vals[i]) + (float(vals[i + 1]) - float(vals[i])) * (x - i)


def value_between(a: float, b: float, st: MultiSettings, frame: float) -> float:
    return a + (b - a) * progress_at(st, frame)


def value_over(vals: Sequence[float], st: MultiSettings, frame: float) -> float:
    """Value of a slider with several points at a frame: straight (or eased, per part) from point to point, every part
    taking as long as the global 'time between points' says."""
    n = len(vals)
    if n == 1:
        return float(vals[0])
    hs, _ = st.holds()
    x = frame - hs
    offs = st.point_offsets()
    if len(offs) != n:                                            # values for a different number of points: equal parts
        offs = [round(k * (offs[-1]) / (n - 1), 6) for k in range(n)]
    if x <= 0:
        return float(vals[0])
    if x >= offs[-1]:
        return float(vals[-1])
    i = max(k for k in range(n - 1) if offs[k] <= x)
    u = _ease((x - offs[i]) / max(1e-9, offs[i + 1] - offs[i]), st.ease)
    return float(vals[i]) + (float(vals[i + 1]) - float(vals[i])) * u


def point_frame(st: MultiSettings, k: int, n: int) -> int:
    """The frame at which every photo is at its slider point k (0-based) of n."""
    hs, _ = st.holds()
    offs = st.point_offsets()
    if len(offs) == n:
        return hs + offs[min(max(k, 0), n - 1)]
    return hs + int(round(k / max(1, n - 1) * (offs[-1])))


def fill_progress(st: MultiSettings, frame: float) -> float:
    """How far the drawing colour has spread at a frame: 0 until the fill starts, 1 at the end of the movement (and after it)."""
    if not st.fill:
        return 0.0
    hs, _ = st.holds()
    a, b = st.fill_window()
    return _ease((frame - hs - a) / max(1e-9, b - a), st.ease)


def layer_progress(st: MultiSettings, frame: float, graffiti: bool = False) -> float:
    """How far the layer movement has got at a frame (0..1): from `layer_start` to `layer_end`, eased like the sliders.
    `graffiti=True`: the graffiti's own window when it has one."""
    hs, _ = st.holds()
    end = float(st.move_frames() - 1)
    s0, e0 = (st.gra_start, st.gra_end) if (graffiti and st.gra_own) else (st.layer_start, st.layer_end)
    a = min(max(0.0, s0 * st.fps), max(0.0, end - 1.0))
    b = e0 * st.fps if e0 > 0 else end
    b = min(max(b, a + 1.0), max(end, a + 1.0))
    return _ease((frame - hs - a) / max(1e-9, b - a), st.ease)


def mask_style(st: MultiSettings):
    from .layers import MaskStyle
    return MaskStyle(float(st.mask_smooth), float(st.mask_soft), bool(st.mask_vector))


def photo_background(st: MultiSettings) -> bool:
    """The facade is the original photo: with 'Facade = original photo', and also whenever only the mask is vectorised (the background
    stays; the Facade slider fades it out over time)."""
    return bool(st.facade_photo or (st.layers and st.only_mask))


def layer_values(st: MultiSettings, frame: float) -> Dict[str, float]:
    u = layer_progress(st, frame)
    ug = layer_progress(st, frame, graffiti=True)
    out = {"layer_f": round((st.fac0 + (st.fac1 - st.fac0) * u) / 100.0, 4), "layer_g": round((st.gra0 + (st.gra1 - st.gra0) * ug) / 100.0, 4),
           "blur_f": round(st.blur0 + (st.blur1 - st.blur0) * u, 3)}
    if photo_background(st):
        out["layer_p"] = 1.0
    return out


FILL_THRESHOLD_CAP = 254.9      # with the fill on a Threshold of 255 means "just below 255": a pure white background stays white until the fill


def _cap(param: str, v: float, st: MultiSettings) -> float:
    return min(v, FILL_THRESHOLD_CAP) if (st.fill and param == "threshold") else v


def values_at(spec: PanelSpec, st: MultiSettings, frame: int) -> Dict[str, float]:
    """Slider values of one photo at a frame (Noise is whole pixels)."""
    out = {"threshold": round(_cap("threshold", value_over(spec.thr, st, frame), st), 4),
           "noise": float(int(round(value_over(spec.noise, st, frame))))}
    if spec.paths is not None:
        out["paths"] = round(value_over(spec.paths, st, frame), 4)
    if st.fill:
        out["fill"] = round(fill_progress(st, frame), 4)
    if st.layers and spec.mask is not None:
        out.update(layer_values(st, frame))
    return out


def step_per_frame(a: float, b: float, st: MultiSettings, segments: int = 1) -> float:
    """How much a slider moves per frame over a stretch of the movement (before easing)."""
    return abs(b - a) / max(1.0, (st.move_frames() - 1) / max(1, segments))


def steps_between_points(vals: Sequence[float], st: MultiSettings) -> List[float]:
    """Movement per frame in every part between two points."""
    frames = st.segment_frames()
    if len(frames) != len(vals) - 1:
        frames = [max(1.0, (st.move_frames() - 1) / max(1, len(vals) - 1))] * (len(vals) - 1)
    return [abs(vals[i + 1] - vals[i]) / frames[i] for i in range(len(vals) - 1)]


def make_tracks(spec: PanelSpec, st: MultiSettings) -> List[Track]:
    """The photo's sliders as tracks of a 'Sliders' stage: one step per frame, resting at the start / end values."""
    if getattr(st, "timeline", None) is not None:
        return st.timeline.make_tracks(spec, st)
    steps = max(1, st.total_frames() - 1)
    out = []
    items = [("threshold", spec.thr), ("noise", spec.noise)] + ([("paths", spec.paths)] if spec.paths is not None else [])
    for param, vals in items:
        vals = [_cap(param, float(v), st) for v in vals] or [0.0]
        if len(vals) == 1:
            vals = vals * 2
        if len(set(vals)) == 1:
            if param == "noise":
                continue                                           # static: set on the square instead
            vals[-1] = vals[0] + (0.01 if vals[0] < (255 if param == "threshold" else 100) else -0.01)   # keeps the clip as long as the others
        t = Track.with_steps(param, vals[0], vals[-1], steps)
        t.warp = lambda u, v=tuple(vals): value_over(v, st, u * steps)
        out.append(t)
    if st.fill:
        t = Track.with_steps("fill", 0.0, 1.0, steps)
        t.warp = lambda u: fill_progress(st, u * steps)
        out.append(t)
    if st.layers and spec.mask is not None:
        hs, _ = st.holds()
        for key, a, b in (("layer_f", st.fac0 / 100.0, st.fac1 / 100.0), ("layer_g", st.gra0 / 100.0, st.gra1 / 100.0),
                          ("blur_f", st.blur0, st.blur1)) + ((("layer_p", 1.0, 1.0),) if photo_background(st) else ()):
            t = Track.with_steps(key, a, b, steps)
            t.warp = lambda u, k=key: layer_values(st, u * steps)[k]
            out.append(t)
    return out


def load_panel_source(path: str, st: MultiSettings, mask: Optional[np.ndarray] = None) -> SweepSource:
    """The picture of one panel exactly as the export will trace it (same size, same parameters)."""
    from .batch import tile_params
    from .clip import fit_tile
    from .imageio_ import load_image_alpha
    from .rows import fit_size, image_size
    iw, ih = image_size(path)
    tw, th = fit_size("fit_long", st.tile, st.tile, iw / max(1, ih))
    rgb, alpha = load_image_alpha(path, (tw, th))
    bgr, opaque = fit_tile(rgb, alpha, tw, th, "contain")
    from .layers import styled_mask, traced_region
    m = None if mask is None else styled_mask(mask, bgr.shape[1], bgr.shape[0], mask_style(st))
    if st.layers and st.only_mask and m is not None:
        opaque = traced_region(opaque, m, bgr.shape)
    return SweepSource(bgr, opaque, tile_params(st.base_params(), tw, th), False, m)


def measure_source(src: SweepSource, long_side: int = 1000) -> SweepSource:
    """A smaller copy of a photo for measuring its black point (the curve is a Threshold per share of white, which does not
    depend on the resolution; measuring at 4000 px would take minutes)."""
    import cv2
    from dataclasses import replace as rep
    h, w = src.bgr.shape[:2]
    f = long_side / float(max(h, w))
    if f >= 1.0:
        return src
    size = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    bgr = cv2.resize(src.bgr, size, interpolation=cv2.INTER_AREA)
    op = None if src.opaque is None else cv2.resize(src.opaque.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    from .layers import fit_mask
    mk = None if src.mask is None else fit_mask(src.mask, *size)
    return SweepSource(bgr, op, rep(src.params, scale=max(0.25, src.params.scale * f)), src.linework, mk)


def tile_colors(spec: PanelSpec, st: MultiSettings):
    """(ink, paper) of one photo's picture; None for what is not set. A custom clip background is the default paper, so
    pictures that do not fill their square blend into it."""
    return spec.ink, (spec.paper if spec.paper is not None else st.bg_color)


def styled_source(src: SweepSource, st: MultiSettings, flip: bool = False, ink=None, paper=None, fill_over: Optional[dict] = None) -> SweepSource:
    """The source with this photo's look (flip, colours, noise/fill settings) in its parameters, as render_preview uses it."""
    from dataclasses import replace as rep
    style = {**st.fill_style(), **(fill_over or {})}                  # the clip's noise settings, this photo's own on top
    cur = {k: getattr(src.params, k, None) for k in style}
    if (flip, ink, paper) != (bool(getattr(src.params, "flip", False)), getattr(src.params, "ink", None), getattr(src.params, "paper", None)) \
            or style != cur:
        return rep(src, params=rep(src.params, flip=flip, ink=ink, paper=paper, **style))
    return src


def render_preview(src: SweepSource, st: MultiSettings, values: Dict[str, float], cache: Optional[dict] = None,
                   flip: bool = False, ink=None, paper=None, fill_over: Optional[dict] = None, parts: bool = False):
    """One picture of a photo at slider `values`. parts=True: (vector picture, ink 0..1 or None, paper colour) without the layers."""
    from .sweep import render_values
    src = styled_source(src, st, flip, ink, paper, fill_over)
    if parts:
        return render_values(src, st.bg, values, cache, parts=True)
    img = render_values(src, st.bg, values, cache)
    return 255 - img if st.invert else img


def plan_rows(specs: Sequence[PanelSpec], st: MultiSettings):
    """Rows of squares for the loaded photos (panel order, `columns` per row) with every square's own slider tracks."""
    from .rows import CellSpec, RowSpec
    loaded = [s for s in specs if s.path and os.path.exists(s.path)]
    rows = []
    for i in range(0, len(loaded), max(1, st.columns)):
        chunk = loaded[i:i + max(1, st.columns)]
        r = RowSpec([s.path for s in chunk], st.tile, st.tile, fit="contain", size_mode="fit_long")
        r.cells = [CellSpec(noise=int(round(s.noise[0])), sweep=make_tracks(s, st)) for s in chunk]
        rows.append(r)
    return rows, loaded


def build_renderer(specs: Sequence[PanelSpec], st: MultiSettings, workers: int = 0, progress: Optional[Callable] = None,
                   pool=None):
    """Trace every photo at the clip's size and return the GridRenderer of the whole clip (None if nothing is loaded
    or the work was cancelled by `progress` returning True)."""
    from .batch import grid_jobs, resolve_workers, trace_jobs
    from .clip import ClipSettings, GridLayout, GridRenderer, GridScene, Stage
    from .morph import MorphOptions
    from dataclasses import replace as rep
    rows, loaded = plan_rows(specs, st)
    if not rows:
        return None
    nw = resolve_workers(workers)
    results = trace_jobs(grid_jobs(rows, st.base_params()), nw, progress)
    if len(results) < len(loaded):
        return None
    for res, spec in zip(results, loaded):
        if st.layers and spec.mask is not None:
            from .layers import styled_mask
            res.mask = styled_mask(spec.mask, res.image_bgr.shape[1], res.image_bgr.shape[0], mask_style(st))   # the layers of this photo
            if st.only_mask:
                from .layers import traced_region
                res.opaque = traced_region(res.opaque, res.mask, res.image_bgr.shape)         # ... and the tracer sees only the masked part
        ink, paper = tile_colors(spec, st)
        if spec.flip or ink or paper or spec.fill_over:            # per photo: black ↔ white, its own colours and fill noise
            res.params = rep(res.params, flip=spec.flip, ink=ink, paper=paper, **(spec.fill_over or {}))
    bg = st.bg
    if st.bg_color is not None:
        from .animate import CUSTOM_BG
        from .clip import BG_RGB
        BG_RGB["custom"] = CUSTOM_BG["custom"] = tuple(st.bg_color)   # the colour between the pictures
        bg = "custom"
    g = GridScene(results, GridLayout.from_rows(rows, st.gap), [], "span", MorphOptions(), "tile", None, None, rows, nw, pool)
    cs = ClipSettings(fps=st.fps, bg=bg, show_text=False, divider=False, invert=st.invert, ss=2 if st.smooth else 1)
    return GridRenderer(g, [Stage("sweep", 0, tracks=[])], cs, st.width)


def fill_preview(src: SweepSource, st: MultiSettings, spec: PanelSpec, n: int = 6, long_side: int = 480):
    """What the fill of one photo looks like: n pictures at equal moments between the start of the fill and the end of the movement
    (the last one is that end), with the slider values of that moment, and for the overlay the plain noise colours, the pixels still
    to be filled and the line of the front in each. Returns (list of dicts with seconds / coverage / image / plain / ahead / front,
    a map of the noise with the front at all those moments)."""
    from dataclasses import replace as rep
    from .sweep import apply_values, fill_front, fill_map, fill_state
    small = measure_source(src, long_side)
    hs, _ = st.holds()
    st2 = rep(st, fill=True)                                   # also when the fill is still switched off: show what it would do
    a, b = st2.fill_window()
    ink, paper = tile_colors(spec, st)
    c_ink = np.array((ink or (0, 0, 0))[::-1], np.float32)
    c_paper = np.array((paper or (255, 255, 255))[::-1], np.float32)
    cache: dict = {}
    out, levels = [], []
    style = {**st.fill_style(), **(spec.fill_over or {})}
    for k in range(max(1, n)):
        frame = hs + a + (k + 1) / max(1, n) * (b - a)
        vals = values_at(spec, st2, int(round(frame)))
        vals["fill"] = round(fill_progress(st2, frame), 4)
        img = render_preview(small, st2, vals, cache, spec.flip, ink, paper, spec.fill_over)
        im = img.astype(np.float32)
        cover = float((np.linalg.norm(im - c_ink, axis=2) < np.linalg.norm(im - c_paper, axis=2)).mean())
        # the noise at this moment (its distance part follows the drawing as it is at this moment)
        p = rep(apply_values(small.params, {x: y for x, y in vals.items() if x != "fill"}), flip=spec.flip, fill=float(vals["fill"]), **style)
        vkey = tuple(sorted((x, round(float(y), 4)) for x, y in vals.items() if x != "fill"))
        dark, field = fill_state(small, p, {}, vkey)
        front = dark | (field <= fill_front(p))
        cs, _ = cv2.findContours(front.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        line = np.zeros(front.shape, np.uint8)
        cv2.drawContours(line, cs, -1, 255, max(1, int(round(max(front.shape) / 400))), cv2.LINE_AA)
        out.append({"seconds": frame / st.fps, "progress": vals["fill"], "coverage": cover, "image": img,
                    "plain": fill_map(small, p, [], {}, vkey), "ahead": ~front if vals["fill"] < 1 else np.zeros_like(front), "front": line})
        levels.append(p)
    last = levels[-1]
    return out, fill_map(small, last, [x.fill for x in levels[:-1]], {}, tuple(sorted((x, round(float(y), 4)) for x, y in values_at(spec, st2, int(round(hs + b))).items() if x != "fill")))
