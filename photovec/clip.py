"""Clip timeline: stages (hold / slider wipe / morph) rendered for one picture or a grid of tiles."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .animate import AnimSettings, _text, background, compose, source_image, vector_image
from .sweep import SweepSource, SweepStore, Track, merge_tracks
from .morph import (Linework, MorphOptions, MorphPlan, build_plan, fit_linework, items_from_result,
                    merge_plans, offset_items, offset_lines, render_lines, render_morph, DstLine, SrcItem)

BG_RGB = {"white": (255, 255, 255), "dark": (40, 40, 44), "checker": (255, 255, 255)}
IMG_KEYS = ("source", "vector", "linework", "blank")


# --------------------------------------------------------------------------- stages
FIT_NAMES = {"even": "same white at every moment", "natural": "own path"}
PACE_NAMES = {"fast": "slower ones sped up", "slow": "faster ones slowed", "steps": "exact steps"}


@dataclass
class Stage:
    kind: str = "hold"            # hold | wipe | morph | sweep
    seconds: float = 2.0
    img_a: str = "source"         # hold: shown image; wipe: image before the slider passes
    img_b: str = "vector"         # wipe: image after the slider passes
    # wipe: one slider or two (four quadrants); every slider has its own start / end position
    mode: str = "single"          # single | quad
    axis: str = "x"               # single: x = vertical divider moving sideways, y = horizontal divider
    side_x: str = "right"         # image B enters from this side (vertical divider)
    side_y: str = "bottom"        # ... (horizontal divider)
    x0: float = 0.0
    x1: float = 1.0
    y0: float = 0.0
    y1: float = 1.0
    y_lag: float = 0.0            # slider 2 starts this fraction of the stage later
    easing: str = "smooth"        # smooth | linear
    pingpong: bool = False
    text_a: str = "BEFORE"
    text_b: str = "AFTER"
    caption: str = ""             # hold / morph: optional label
    reverse: bool = False         # morph: linework -> vector instead of vector -> linework
    # --- grids (ignored for a single picture, except step_px) --------------------------------
    scope: str = "row"            # tile: every picture has its own slider | row: one slider per row | grid: one slider
    step_px: int = 0              # >0: slider moves this many picture-pixels per frame (1 step = 1 frame); 0 = seconds
    row_mode: str = "together"    # together | sequence (row after row) | stagger (row_stagger frames apart)
    row_stagger: int = 10         # frames between rows in 'stagger' mode
    alternate: bool = False       # every second row slides in from the opposite side
    # --- sweep: animate Image Trace / Linework Snap sliders (1 step = 1 frame) -----------------------
    tracks: List[Track] = field(default_factory=list)
    track_seq: bool = False       # sliders one after the other instead of all at once
    target: str = "all"           # sweep: which pictures move: all | kept | dropped (pictures chosen to go back to white)
    dropped: str = "auto"         # dropped pictures are shown: auto (white once a 'dropped' sweep has run) | show | white
    sync: float = -1.0            # >= 0: finish together – every picture's Threshold slider ends at its own black point (white area <= sync %) ...
    fit: str = "even"             # ... path: even = same share of white in every picture at every moment | natural = each picture's own Threshold path
    pace: str = "fast"            # ... length: fast = as long as the fastest picture needs at the slider's speed (the slower ones are sped up) |
                                  # slow = as long as the slowest needs (the faster ones get finer steps) | steps = exactly the slider's number of steps

    def describe(self) -> str:
        if self.kind == "hold":
            return f"Hold  {'blank (white)' if self.img_a == 'blank' else self.img_a}  {self.seconds:g} s"
        if self.kind == "sweep":
            who = {"all": "", "kept": "  [kept pictures]", "dropped": "  [dropped pictures]"}[self.target]
            n = max([t.delay + t.steps() for t in self.tracks] or [0]) * (2 if self.pingpong else 1)
            return ("Sweep  " + (", ".join(t.label() for t in self.tracks) or "no sliders") + who +
                    (f"  [black at ≤{self.sync:g}% white, {FIT_NAMES.get(self.fit, self.fit)}, {PACE_NAMES.get(self.pace, self.pace)}]" if self.sync >= 0 else "") +
                    ("  (back and forth)" if self.pingpong else "") + f"  · {n} steps")
        if self.kind == "morph":
            return f"Morph  {'linework → vector' if self.reverse else 'vector → linework'}  {self.seconds:g} s"
        sl = "2 sliders" if self.mode == "quad" else f"slider {self.axis}"
        tm = f"{self.step_px} px per frame" if self.step_px > 0 else f"{self.seconds:g} s"
        return f"Wipe  {self.img_a} → {self.img_b}  ({sl}, per {self.scope})  {tm}"


@dataclass
class ClipSettings:
    fps: int = 25
    bg: str = "white"             # white | dark | checker (checker only affects images, not the morph)
    show_text: bool = True
    divider: bool = True
    ss: int = 1                   # supersampling of morph frames (2 = smoother, 4x slower)
    invert: bool = False          # inverted vector colours: black paper with white drawing (photos are left alone)


def default_timeline(has_linework: bool = True) -> List[Stage]:
    st = [Stage("hold", 0.8, "source"),
          Stage("wipe", 2.5, "source", "vector", mode="single", axis="x", side_x="right"),
          Stage("hold", 0.8, "vector")]
    if has_linework:
        st += [Stage("morph", 3.5), Stage("hold", 1.2, "linework")]
    return st


def grid_timeline(has_linework: bool, canvas_w: int, fps: int = 25, secs: float = 4.0) -> List[Stage]:
    """Default clip for a grid: row sliders that move in steps (1 step = 1 frame), then the morph."""
    step = max(1, int(math.ceil(canvas_w / (secs * fps))))
    st = [Stage("hold", 0.6, "source"),
          Stage("wipe", 2.5, "source", "vector", mode="single", axis="x", side_x="right", step_px=step,
                scope="row", row_mode="stagger", row_stagger=8),
          Stage("hold", 0.6, "vector")]
    if has_linework:
        st += [Stage("morph", 3.5), Stage("hold", 1.2, "linework")]
    return st


def stage_frames(st: Stage, fps: int) -> int:
    """Frames of a time-based stage (step-based wipes depend on the picture size: see stage_timing)."""
    return max(1, int(round(st.seconds * fps)))


def _ease(u: float, kind: str) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u) if kind == "smooth" else u


def wipe_positions(st: Stage, m: float) -> Tuple[float, float]:
    """Slider positions (fraction covered by image B) for stage progress m in 0..1."""
    lin = (1.0 - abs(2.0 * m - 1.0)) if st.pingpong else m
    px = st.x0 + (st.x1 - st.x0) * _ease(lin, st.easing)
    lag = min(max(st.y_lag, 0.0), 0.95)
    my = min(1.0, max(0.0, (lin - lag) / (1.0 - lag)))
    py = st.y0 + (st.y1 - st.y0) * _ease(my, st.easing)
    return px, py


# --------------------------------------------------------------------------- groups & timing
@dataclass
class Group:
    """A region with its own slider(s): a tile, a row or the whole grid (scene pixels)."""
    box: Tuple[int, int, int, int]      # x, y, w, h
    row: int = 0
    side: str = "auto"                  # auto | left | right | top | bottom  (where image B enters)
    delay: int = 0                      # frames
    tiles: List[Tuple[int, Tuple[int, int, int, int]]] = field(default_factory=list)   # (picture index, box) inside this group
    tracks: List[Track] = field(default_factory=list)       # this row's own sweep sliders
    black: Optional[Callable[[float], Optional[float]]] = None   # white-% -> Threshold at which the pictures are (almost) black


def is_target(st: "Stage", k: int, dropped) -> bool:
    """Does a sweep stage move picture k (pictures that go back to white can be left out or picked)?"""
    return not ((st.target == "kept" and k in dropped) or (st.target == "dropped" and k not in dropped))


def active_groups(st: "Stage", groups: Sequence[Group], dropped) -> List[bool]:
    return [any(is_target(st, k, dropped) for k, _ in (g.tiles or [(0, g.box)])) for g in groups]


def attach_black(groups: Sequence[Group], scene) -> None:
    """Give every slider region the black point function of its pictures (needs the traced pictures of `scene`)."""
    from .blackpoint import group_black
    for g in groups:
        if g.black is None:
            g.black = group_black([scene.black_curve(k) for k, _ in (g.tiles or [(0, g.box)])])


def fit_groups(st: "Stage", groups: Sequence[Group], merged: List[List[Track]],
               active: Optional[Sequence[bool]] = None) -> List[List[Track]]:
    """Finish together (`st.sync` >= 0): every picture's Threshold slider ends – or, running down, starts – at its black point
    (the lowest Threshold with at most `sync` % white), so the picture is black exactly when its slider ends.
    Path (`st.fit`): "even" = every picture follows its own white-area curve, so at every moment all pictures have the same share
    of white left; "natural" = equal Threshold steps from 0 to the black point.
    Length (`st.pace`): the stage's slider gives the normal speed (threshold per frame). "fast" = as long as the picture that is
    black first needs at that speed – every slower picture gets bigger steps; "slow" = as long as the slowest needs, the faster
    ones get finer steps; "steps" = exactly the number of steps of the slider.
    On its last step the slider jumps to 255 (completely black), so no white is left over. Pictures whose square has its own
    Threshold range keep it and do not set the pace."""
    out = [list(t) for t in merged]
    plan = []                                              # (group, track index, track, black point)
    for gi, (g, tr) in enumerate(zip(groups, merged)):
        mine = {t.param for t in g.tracks}
        for ti, t in enumerate(tr):
            if t.param == "threshold" and "threshold" not in mine and g.black is not None:
                T = g.black(st.sync)
                if T is not None:
                    plan.append((gi, ti, t, max(1.0, min(255.0, T))))
    if not plan:
        return out
    dist = lambda t, T: abs((T - t.start) if t.end >= t.start else (t.end - T))
    if st.pace == "steps":
        n_of = lambda t, T: t.steps()
    else:
        lead = [p for p in plan if active is None or active[p[0]]] or plan
        ns = [max(1, int(math.ceil(dist(t, T) / max(t.step, 1e-6) - 1e-9))) for _, _, t, T in lead]
        n_ref = max(ns) if st.pace == "slow" else min(ns)
        n_of = lambda t, T: n_ref
    for gi, ti, t, T in plan:
        up = t.end >= t.start
        n = n_of(t, T)
        nt = Track.with_steps("threshold", t.start if up else 255.0, 255.0 if up else t.end, n, t.delay)
        cv = getattr(groups[gi].black, "curve", None)
        if st.fit == "even" and cv is not None:
            a, b = (cv.white_at(t.start), cv.white_at(T)) if up else (cv.white_at(T), cv.white_at(t.end))
            nt.warp = lambda u, cv=cv, a=a, b=b: cv.threshold_for_white(a + (b - a) * u)
        else:
            lo, hi = (t.start, T) if up else (T, t.end)
            nt.warp = lambda u, lo=lo, hi=hi: lo + (hi - lo) * u
        out[gi][ti] = nt
    return out


def warm_curves(scene, workers: int = 4, cancelled: Optional[Callable[[], bool]] = None,
                progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Measure every picture's white-area curve now (threads; numpy / OpenCV release the GIL), so the first change of the
    timeline does not have to wait for it."""
    from concurrent.futures import ThreadPoolExecutor
    n = len(scene.sources)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _ in ex.map(lambda k: None if (cancelled and cancelled()) else scene.black_curve(k), range(n)):
            done += 1
            if progress is not None:
                progress(done, n)


@dataclass
class GroupTiming:
    delay: int
    dur: int
    anim: AnimSettings
    sx: int = 1                         # step mode: steps of slider 1 / slider 2 and slider 2's late start
    sy: int = 1
    lag: int = 0
    tracks: List[Track] = field(default_factory=list)      # sweep: sliders of this group and their start frames
    tdelay: List[int] = field(default_factory=list)


@dataclass
class StageTiming:
    n: int
    groups: List[GroupTiming]


def _group_anim(st: Stage, g: Group, cs: "ClipSettings") -> AnimSettings:
    side_x, side_y, axis = st.side_x, st.side_y, st.axis
    if g.side in ("left", "right"):
        side_x = g.side
        if st.mode == "single":
            axis = "x"
    elif g.side in ("top", "bottom"):
        side_y = g.side
        if st.mode == "single":
            axis = "y"
    if st.alternate and g.row % 2 == 1:
        side_x = "left" if side_x == "right" else "right"
        side_y = "top" if side_y == "bottom" else "bottom"
    return AnimSettings(mode=st.mode, axis=axis, side_x=side_x, side_y=side_y, show_text=cs.show_text,
                        text_before=st.text_a, text_after=st.text_b, divider=cs.divider)


def _steps(extent: float, step: int) -> int:
    return max(1, int(math.ceil(max(extent, 0.0) / max(1, step) - 1e-9)))


def stage_timing(st: Stage, fps: int, groups: Sequence[Group], cs: "ClipSettings",
                 active: Optional[Sequence[bool]] = None) -> StageTiming:
    """Frames of a stage and the start / length of every group's slider.
    Step mode (step_px > 0): the slider moves `step_px` picture pixels per frame, so 1 step = 1 frame."""
    animes = [_group_anim(st, g, cs) for g in groups]
    if st.kind not in ("wipe", "sweep"):
        n = stage_frames(st, fps)
        return StageTiming(n, [GroupTiming(0, max(1, n - 1), a) for a in animes])
    nrows = max(g.row for g in groups) + 1
    pp = 2 if st.pingpong else 1
    if st.kind == "sweep":
        info = []
        merged = [merge_tracks(st.tracks, g.tracks) for g in groups]
        if st.sync >= 0:
            merged = fit_groups(st, groups, merged, active)
        for g, tr in zip(groups, merged):
            td, acc = [], 0
            for t in tr:
                td.append(t.delay + acc)
                if st.track_seq:
                    acc += t.steps() * pp + t.delay
            dur = max([d + t.steps() * pp for d, t in zip(td, tr)] or [1])
            info.append((tr, td, dur))
        rowdur = [max([i[2] for g, i in zip(groups, info) if g.row == r] or [1]) for r in range(nrows)]
        rd = _row_delays(st, nrows, rowdur)
        gts = [GroupTiming(rd[g.row] + g.delay, i[2], a, tracks=i[0], tdelay=i[1]) for g, a, i in zip(groups, animes, info)]
        return StageTiming(max(t.delay + t.dur for t in gts) + 1, gts)
    if st.step_px > 0:
        info = []
        for g, a in zip(groups, animes):
            w, h = g.box[2], g.box[3]
            if a.mode == "quad":
                sx, sy = _steps(abs(st.x1 - st.x0) * w, st.step_px), _steps(abs(st.y1 - st.y0) * h, st.step_px)
                lag = int(round(min(max(st.y_lag, 0.0), 0.95) * max(sx, sy)))
                dur = max(sx, lag + sy) * pp
            else:
                sx = _steps(abs(st.x1 - st.x0) * (w if a.axis == "x" else h), st.step_px)
                sy, lag, dur = 1, 0, sx * pp
            info.append((sx, sy, lag, dur))
        rowdur = [max([i[3] for g, i in zip(groups, info) if g.row == r] or [1]) for r in range(nrows)]
        rd = _row_delays(st, nrows, rowdur)
        gts = [GroupTiming(rd[g.row] + g.delay, i[3], a, i[0], i[1], i[2]) for g, a, i in zip(groups, animes, info)]
        return StageTiming(max(t.delay + t.dur for t in gts) + 1, gts)
    n = stage_frames(st, fps)
    if st.row_mode == "sequence":
        slot = max(1, int(round((n - 1) / nrows)))
        rd = [r * slot for r in range(nrows)]
        gts = [GroupTiming(rd[g.row] + g.delay, slot, a) for g, a in zip(groups, animes)]
    else:
        rd = _row_delays(st, nrows, [1] * nrows)
        gts = []
        for g, a in zip(groups, animes):
            d = rd[g.row] + g.delay
            gts.append(GroupTiming(d, max(1, n - 1 - d), a))
    return StageTiming(n, gts)


def _row_delays(st: Stage, nrows: int, rowdur: Sequence[int]) -> List[int]:
    if st.row_mode == "sequence":
        out, acc = [], 0
        for d in rowdur:
            out.append(acc)
            acc += d
        return out
    if st.row_mode == "stagger":
        return [r * max(0, st.row_stagger) for r in range(nrows)]
    return [0] * nrows


def _lin(jj: int, steps: int, pp: bool) -> float:
    if jj <= 0:
        return 0.0
    if pp:
        return 1.0 - abs(2.0 * min(1.0, jj / (2.0 * steps)) - 1.0)
    return min(1.0, jj / steps)


def group_positions(st: Stage, gt: GroupTiming, j: int) -> Tuple[float, float]:
    """Slider positions of one group at stage frame j."""
    jj = j - gt.delay
    if st.step_px > 0:
        px = st.x0 + (st.x1 - st.x0) * _lin(jj, gt.sx, st.pingpong)
        py = st.y0 + (st.y1 - st.y0) * _lin(jj - gt.lag, gt.sy, st.pingpong)
        return px, py
    t = min(1.0, max(0.0, jj / gt.dur))
    return wipe_positions(st, t)


def sweep_values(st: Stage, gt: GroupTiming, j: int) -> Dict[str, float]:
    """Slider values of one group at stage frame j (each track: one step per frame)."""
    out: Dict[str, float] = {}
    jj = j - gt.delay
    for t, d in zip(gt.tracks, gt.tdelay):
        n = t.steps()
        local = min(max(jj - d, 0), 2 * n if st.pingpong else n)
        i = n - abs(local - n) if st.pingpong else local
        out[t.param] = t.value(n - i if st.reverse else i)
    if st.sync >= 0 and "threshold" in out:
        out["spread"] = 1.0                    # flat areas fill gradually, so the sweep can follow every picture's white-area curve
    return out


def clip_frame_count(stages: Sequence[Stage], fps: int, groups: Optional[Sequence[Group]] = None) -> int:
    groups = list(groups) if groups else [Group((0, 0, 1000, 1000))]
    cs = ClipSettings(fps=fps)
    return sum(stage_timing(s, fps, groups, cs).n for s in stages)


# --------------------------------------------------------------------------- scene
@dataclass
class Scene:
    """Everything needed to render one picture: images, fitted linework and the morph plan."""
    size: Tuple[int, int]
    plan: Optional[MorphPlan]
    lines: List[DstLine]
    provider: Callable[[str], Tuple[np.ndarray, np.ndarray]]   # bg key -> (source, vector)
    transparent: bool = False      # False: opaque photo (canvas is always white)
    _cache: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = field(default_factory=dict)
    sources: List = field(default_factory=list)       # TraceResults of the pictures (for slider sweeps)
    dropped: frozenset = frozenset()                  # grids: pictures chosen to go back to white
    stay: frozenset = frozenset()                     # grids: pictures that stay a vector (do not morph)
    _store: Optional[SweepStore] = None
    _black: Dict[int, object] = field(default_factory=dict)

    def black_curve(self, k: int):
        """Grey-value curve of picture k (see blackpoint.py), cached; None when the picture has no black point."""
        if k not in self._black:
            from .blackpoint import black_curve
            self._black[k] = black_curve(SweepSource.from_result(self.sources[k])) if 0 <= k < len(self.sources) else None
        return self._black[k]

    def sweep_store(self) -> SweepStore:
        if self._store is None:
            self._store = SweepStore([SweepSource.from_result(r) for r in self.sources])
        return self._store

    def bg_rgb(self, bg: str) -> Tuple[int, int, int]:
        return BG_RGB[bg] if self.transparent else (255, 255, 255)

    def images(self, bg: str):
        if bg not in self._cache:
            src, vec = self.provider(bg)
            lw = render_lines(self.lines, self.size, self.size, self.bg_rgb(bg)) if self.lines else None
            self._cache = {bg: (src, vec, lw)}
        return self._cache[bg]

    @property
    def has_linework(self) -> bool:
        return self.plan is not None


def scene_from_result(res, lw: Optional[Linework], opts: MorphOptions,
                      plan: Optional[MorphPlan] = None) -> Scene:
    W, H = res.size
    lines = fit_linework(lw, W, H, opts) if lw is not None else []
    if plan is None and lines:
        plan = build_plan(items_from_result(res), lines, (W, H), opts)
    return Scene((W, H), plan, lines,
                 lambda bg, r=res: (source_image(r, bg), vector_image(r, bg)),
                 transparent=res.opaque is not None, sources=[res])


# --------------------------------------------------------------------------- renderer
class Renderer:
    """Frame i of a clip. One scene (picture or merged grid canvas), any number of slider groups."""

    def __init__(self, scene: Scene, stages: Sequence[Stage], cs: ClipSettings, size: Tuple[int, int],
                 groups: Optional[Sequence[Group]] = None):
        if not stages:
            raise ValueError("the timeline is empty")
        needs = any(s.kind == "morph" or (s.kind != "sweep" and "linework" in (s.img_a, s.img_b)) for s in stages)
        if needs and not scene.has_linework:
            raise ValueError("the timeline uses linework, but no linework file is loaded")
        self.scene, self.stages, self.cs, self.size = scene, list(stages), cs, size
        W, H = scene.size
        ow, oh = size
        fx, fy = ow / W, oh / H
        # slider regions per scope ("tile" | "row" | "grid"); a plain list serves every scope
        sets = dict(groups) if isinstance(groups, dict) else {None: list(groups) if groups else [Group((0, 0, W, H))]}
        self._sets = {}
        for key, gl in sets.items():
            boxes = []
            for g in gl:
                x, y, w, h = g.box
                x0, y0 = min(ow - 1, max(0, int(round(x * fx)))), min(oh - 1, max(0, int(round(y * fy))))
                x1, y1 = min(ow, max(x0 + 1, int(round((x + w) * fx)))), min(oh, max(y0 + 1, int(round((y + h) * fy))))
                boxes.append((x0, y0, x1, y1))
            self._sets[key] = (list(gl), boxes)
        self._tboxes: Dict[tuple, List[Tuple[int, Tuple[int, int, int, int]]]] = {}
        self._lboxes: Dict[tuple, List[Tuple[int, Tuple[int, int, int, int]]]] = {}      # same, in scene pixels (x, y, w, h)
        for key, (gl, _) in self._sets.items():
            for gi, g in enumerate(gl):
                tl = g.tiles or [(0, g.box)]
                lst = []
                for k, (x, y, w, h) in tl:
                    x0, y0 = min(ow - 1, max(0, int(round(x * fx)))), min(oh - 1, max(0, int(round(y * fy))))
                    x1, y1 = min(ow, max(x0 + 1, int(round((x + w) * fx)))), min(oh, max(y0 + 1, int(round((y + h) * fy))))
                    lst.append((k, (x0, y0, x1, y1)))
                self._tboxes[(key, gi)] = lst
                self._lboxes[(key, gi)] = [(k, tuple(int(round(v)) for v in b)) for k, b in tl]
        drop = set(scene.dropped)
        if any(st.kind == "sweep" and st.sync >= 0 for st in self.stages):        # finish-together needs every picture's black point
            for gl, _ in self._sets.values():
                attach_black(gl, scene)
        self.groups = next(iter(self._sets.values()))[0]
        self.timings = [stage_timing(st, cs.fps, self._set(st)[0], cs, active_groups(st, self._set(st)[0], drop)) for st in self.stages]
        self.counts = [t.n for t in self.timings]
        self.total = sum(self.counts)
        self._starts = np.r_[0, np.cumsum(self.counts)]
        src, vec, lw = scene.images(cs.bg)
        self.invert = bool(cs.invert)
        if self.invert:                                  # vectors and linework in negative colours; the photos stay as they are
            vec = 255 - vec
            lw = None if lw is None else 255 - lw
        rs = lambda im: None if im is None else cv2.resize(im, (ow, oh), interpolation=cv2.INTER_AREA)
        self.imgs = {"source": rs(src), "vector": rs(vec), "linework": rs(lw)}
        blank = np.empty((oh, ow, 3), np.uint8)
        blank[:] = self._paper()
        self.imgs["blank"] = blank
        self.dropped = set(scene.dropped)
        first = next((i for i, st in enumerate(self.stages) if st.kind == "sweep" and st.target == "dropped"), None)
        self._white = [st.dropped == "white" or (st.dropped == "auto" and first is not None and i > first)
                       for i, st in enumerate(self.stages)]
        tset = self._tboxes.get(("tile", 0)) and "tile" or next(iter(self._sets))
        pick = (lambda d: [d[(tset, gi)][0] for gi in range(len(self._sets[tset][0]))]) if tset == "tile" else \
            (lambda d: [t for gi in range(len(self._sets[tset][0])) for t in d[(tset, gi)]])
        self._all_tiles = pick(self._tboxes)
        self._all_lboxes = pick(self._lboxes)
        self.stay = set(getattr(scene, "stay", ()))
        if self.stay and lw is not None:                 # squares that stay a vector keep their vector in the linework picture
            lw = lw.copy()
            for k, (x, y, w, h) in self._all_lboxes:
                if k in self.stay:
                    lw[y:y + h, x:x + w] = vec[y:y + h, x:x + w]
            self.imgs["linework"] = rs(lw)
        self._layout = (src, vec, lw)
        self._rs = rs
        self._masked: Dict[tuple, np.ndarray] = {}
        self._static: Dict[int, np.ndarray] = {}

    def _paper(self) -> np.ndarray:
        """Colour of the empty canvas (BGR): the background, or its negative when the colours are inverted."""
        c = np.array(self.scene.bg_rgb(self.cs.bg)[::-1], np.uint8)
        return 255 - c if self.invert else c

    def _layout_img(self, si: int, key: str) -> np.ndarray:
        """Source / vector canvas in scene pixels; pictures that went back to white stay white in later stages."""
        im = {"source": self._layout[0], "vector": self._layout[1]}[key]
        if not (self._white[si] and self.dropped):
            return im
        out = im.copy()
        bgc = self._paper()
        for k, (x, y, w, h) in self._all_lboxes:
            if k in self.dropped:
                out[y:y + h, x:x + w] = bgc
        return out

    def _img(self, si: int, key: str) -> np.ndarray:
        im = self.imgs[key]
        if not (self._white[si] and self.dropped) or key not in ("source", "vector"):
            return im
        if (si, key) not in self._masked:
            self._masked[(si, key)] = self._rs(self._layout_img(si, key))
        return self._masked[(si, key)]

    def tile_boxes(self) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """(picture index, x0, y0, x1, y1 in output pixels) of every picture."""
        return list(self._all_tiles)

    def _is_target(self, st: Stage, k: int) -> bool:
        return not ((st.target == "kept" and k in self.dropped) or (st.target == "dropped" and k not in self.dropped))

    def _set(self, st: Stage):
        if st.kind == "sweep" and "tile" in self._sets:
            return self._sets["tile"]                        # every square has its own sliders (row values unless overridden)
        if st.scope in self._sets:
            return self._sets[st.scope]
        return self._sets.get(None) or self._sets.get("grid") or next(iter(self._sets.values()))

    def locate(self, i: int) -> Tuple[int, int]:
        i = min(max(i, 0), self.total - 1)
        si = int(np.searchsorted(self._starts, i, side="right") - 1)
        return si, i - int(self._starts[si])

    def _caption(self, img: np.ndarray, text: str) -> np.ndarray:
        if not (self.cs.show_text and text):
            return img
        out = img.copy()
        scale = max(0.5, out.shape[1] / 900.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, max(1, int(round(scale * 1.6))))
        m = int(14 * scale)
        _text(out, text, (m, m + th), scale)
        return out

    def frame(self, i: int) -> np.ndarray:
        si, j = self.locate(i)
        st, tm = self.stages[si], self.timings[si]
        if st.kind == "hold":
            if si not in self._static:
                self._static[si] = self._caption(self._img(si, st.img_a), st.caption)
            return self._static[si]
        if st.kind == "wipe":
            A, B = self._img(si, st.img_a), self._img(si, st.img_b)
            _, boxes = self._set(st)
            if len(boxes) == 1 and boxes[0] == (0, 0, A.shape[1], A.shape[0]):
                return compose(A, B, *group_positions(st, tm.groups[0], j), tm.groups[0].anim)
            out = A.copy()
            for (x0, y0, x1, y1), gt in zip(boxes, tm.groups):
                px, py = group_positions(st, gt, j)
                out[y0:y1, x0:x1] = compose(A[y0:y1, x0:x1], B[y0:y1, x0:x1], px, py, gt.anim)
            return out
        if st.kind == "sweep":
            return self._sweep_frame(si, st, tm, j)
        m = j / (tm.n - 1) if tm.n > 1 else 1.0
        mm = 1.0 - m if st.reverse else m
        img = render_morph(self.scene.plan, mm, self.size, self.scene.bg_rgb(self.cs.bg), self.cs.ss)
        if self.invert:
            img = 255 - img
        if self.stay:                                   # squares that do not morph stay as they are
            vec = self.imgs["vector"]
            for k, (x0, y0, x1, y1) in self._all_tiles:
                if k in self.stay:
                    img[y0:y1, x0:x1] = vec[y0:y1, x0:x1]
        return self._caption(img, st.caption)

    def _set_key(self, st: Stage):
        for key, v in self._sets.items():
            if v is self._set(st):
                return key
        return None

    def _bg_key(self) -> str:
        return self.cs.bg

    def _sweep_frame(self, si: int, st: Stage, tm: StageTiming, j: int) -> np.ndarray:
        store = self.scene.sweep_store()
        key = self._set_key(st)
        base = self._layout_img(si, "vector").copy()         # scene pixels: the same pixels (and the same single resize)
        for gi, gt in enumerate(tm.groups):                  # as the hold stages, so a sweep that ends on the main window's
            vals = sweep_values(st, gt, j)                   # values joins the next stage without a flicker
            for k, (x, y, w, h) in self._lboxes[(key, gi)]:
                if not self._is_target(st, k):
                    continue
                img = store.get(k, self._bg_key(), vals)
                if self.invert:
                    img = 255 - img
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                base[y:y + h, x:x + w] = img
        ow, oh = self.size
        return base if base.shape[:2] == (oh, ow) else cv2.resize(base, (ow, oh), interpolation=cv2.INTER_AREA)

    def sweep_needs(self) -> Iterator[Tuple[int, str, tuple]]:
        """Every (picture, background, slider values) a sweep stage will show, in playing order."""
        from .sweep import norm_values, values_key
        for si, st in enumerate(self.stages):
            if st.kind != "sweep":
                continue
            tm, key = self.timings[si], self._set_key(st)
            for j in range(tm.n):
                for gi, gt in enumerate(tm.groups):
                    vals = sweep_values(st, gt, j)
                    for k, _ in self._tboxes[(key, gi)]:
                        if self._is_target(st, k):
                            yield k, self._bg_key(), values_key(norm_values(self.scene.sweep_store().sources[k], vals))

    def ensure_sweeps(self, workers: int = 1, pool=None, progress: Optional[Callable[[int, int], bool]] = None) -> bool:
        """Render all slider-sweep frames up front (parallel). False = cancelled."""
        if not any(s.kind == "sweep" for s in self.stages):
            return True
        return self.scene.sweep_store().ensure(self.sweep_needs(), workers, pool, progress)

    def frames(self, progress: Optional[Callable[[int, int], bool]] = None, workers: int = 1) -> Iterator[np.ndarray]:
        """All frames in order. workers > 1 renders several frames at once (cv2 / numpy release the GIL)."""
        if workers <= 1:
            for i in range(self.total):
                if progress is not None and progress(i, self.total):
                    return
                yield self.frame(i)
            return
        from concurrent.futures import ThreadPoolExecutor
        for si, st in enumerate(self.stages):             # fill the lazy caches before threads start
            if st.kind == "hold":
                self.frame(int(self._starts[si]))
        win = workers * 3
        with ThreadPoolExecutor(max_workers=workers) as ex:
            pending = []
            nxt = 0
            for i in range(self.total):
                while nxt < self.total and len(pending) < win:
                    pending.append(ex.submit(self.frame, nxt))
                    nxt += 1
                if progress is not None and progress(i, self.total):
                    for f in pending:
                        f.cancel()
                    return
                yield pending.pop(0).result()


# --------------------------------------------------------------------------- grid
@dataclass
class GridLayout:
    """Where every picture sits on the canvas. Rows are centred; inside a row the squares are placed left to right and
    centred vertically, so every square can have its own size (fixed, or following the shape of its picture)."""
    rows: int = 2
    cols: int = 3
    tile_w: int = 480
    tile_h: int = 480
    gap: int = 12
    rowdefs: Optional[List[Tuple[int, int, int]]] = None      # (pictures, tile_w, tile_h) per row: ragged / mixed sizes
    sizes: Optional[List[List[Tuple[int, int]]]] = None       # (w, h) of every square, row by row (wins over rowdefs)
    _boxes: Optional[List[Tuple[int, int, int, int]]] = field(default=None, init=False, repr=False, compare=False)
    _rowof: Optional[List[int]] = field(default=None, init=False, repr=False, compare=False)
    _canvas: Optional[Tuple[int, int]] = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_rows(cls, rows: Sequence, gap: int = 12) -> "GridLayout":
        sizes = []
        for r in rows:
            r.sync()
            sizes.append([r.tile_size(i) for i in range(len(r.paths))])
        first = next((s[0] for s in sizes if s), (480, 480))
        return cls(len(sizes), max(len(s) for s in sizes), first[0], first[1], gap, None, sizes)

    @property
    def grid_sizes(self) -> List[List[Tuple[int, int]]]:
        if self.sizes is not None:
            return self.sizes
        return [[(tw, th)] * n for n, tw, th in self.geom]

    @property
    def geom(self) -> List[Tuple[int, int, int]]:
        if self.sizes is not None:
            return [(len(r), r[0][0] if r else self.tile_w, r[0][1] if r else self.tile_h) for r in self.sizes]
        return list(self.rowdefs) if self.rowdefs else [(self.cols, self.tile_w, self.tile_h)] * self.rows

    def _row_w(self, sizes: Sequence[Tuple[int, int]]) -> int:
        return sum(w for w, _ in sizes) + max(0, len(sizes) - 1) * self.gap

    def _layout(self) -> None:
        rows = self.grid_sizes
        CW = max([self._row_w(r) for r in rows] or [1])
        boxes, rowof, y = [], [], 0
        for ri, r in enumerate(rows):
            rh = max([h for _, h in r] or [0])
            x = (CW - self._row_w(r)) // 2
            for w, h in r:
                boxes.append((x, y + (rh - h) // 2, w, h))
                rowof.append(ri)
                x += w + self.gap
            y += rh + self.gap
        self._boxes, self._rowof = boxes, rowof
        self._canvas = (CW, max(1, y - self.gap))

    @property
    def canvas(self) -> Tuple[int, int]:
        if self._canvas is None:
            self._layout()
        return self._canvas

    @property
    def cells(self) -> int:
        return sum(len(r) for r in self.grid_sizes)

    def row_of(self, k: int) -> int:
        if self._rowof is None:
            self._layout()
        return self._rowof[min(k, len(self._rowof) - 1)] if self._rowof else 0

    def box(self, k: int) -> Tuple[int, int, int, int]:
        """x, y, w, h of picture k."""
        if self._boxes is None:
            self._layout()
        if not 0 <= k < len(self._boxes):
            raise IndexError("picture index outside the grid")
        return self._boxes[k]

    def origin(self, k: int) -> Tuple[int, int]:
        return self.box(k)[:2]

    def size(self, k: int) -> Tuple[int, int]:
        return self.box(k)[2:]


def fit_tile(rgb: np.ndarray, alpha: Optional[np.ndarray], tw: int, th: int, fit: str):
    """Image -> exactly (tw, th). cover = centre crop; contain = padded with transparency.
    Returns (bgr, opaque mask) – the mask is never None so tiles share one background."""
    h, w = rgb.shape[:2]
    s = max(tw / w, th / h) if fit == "cover" else min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (nw, nh), interpolation=interp)
    op = np.ones((nh, nw), bool) if alpha is None else cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA) >= 128
    out = np.zeros((th, tw, 3), np.uint8)
    mask = np.zeros((th, tw), bool)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw, ch = min(nw - sx0, tw - dx0), min(nh - sy0, th - dy0)
    out[dy0:dy0 + ch, dx0:dx0 + cw] = bgr[sy0:sy0 + ch, sx0:sx0 + cw]
    mask[dy0:dy0 + ch, dx0:dx0 + cw] = op[sy0:sy0 + ch, sx0:sx0 + cw]
    return out, mask


def tile_linework(lws: Sequence[Linework], lw_mode: str, k: int) -> Optional[Linework]:
    """Which linework file belongs to tile k (None for 'span': that one is built for the whole grid)."""
    if not lws or lw_mode == "span":
        return None
    return lws[0] if lw_mode == "same" else lws[k % len(lws)]


def area_of(layout: "GridLayout", ks: Sequence[int]) -> Tuple[int, int, int, int]:
    """x, y, w, h of the smallest rectangle that holds the squares `ks` (the whole canvas when `ks` is empty)."""
    bx = [layout.box(k) for k in ks] or [(0, 0) + layout.canvas]
    x0, y0 = min(b[0] for b in bx), min(b[1] for b in bx)
    x1, y1 = max(b[0] + b[2] for b in bx), max(b[1] + b[3] for b in bx)
    return x0, y0, x1 - x0, y1 - y0


def lines_outside(lines: Sequence[DstLine], boxes: Sequence[Tuple[int, int, int, int]], step: float = 3.0) -> List[DstLine]:
    """The parts of the lines that lie outside all boxes (x, y, w, h): lines crossing a box are cut at its edges, so the
    rest of a long outline still reaches the morphing squares."""
    if not boxes:
        return list(lines)
    B = [(x, y, x + w, y + h) for x, y, w, h in boxes]
    out: List[DstLine] = []
    for l in lines:
        P = l.pts
        lo, hi = P.min(axis=0), P.max(axis=0)
        if not any(lo[0] < x1 and hi[0] >= x0 and lo[1] < y1 and hi[1] >= y0 for x0, y0, x1, y1 in B):
            out.append(l)
            continue
        Q = np.vstack([P, P[:1]]) if l.closed else P
        seg = np.hypot(*np.diff(Q, axis=0).T)
        t = np.r_[0.0, np.cumsum(seg)]
        if t[-1] <= 0:
            continue
        s_ = np.linspace(0.0, t[-1], max(2, int(np.ceil(t[-1] / step)) + 1))
        D = np.stack([np.interp(s_, t, Q[:, 0]), np.interp(s_, t, Q[:, 1])], axis=1)
        inside = np.zeros(len(D), bool)
        for x0, y0, x1, y1 in B:
            inside |= (D[:, 0] >= x0) & (D[:, 0] < x1) & (D[:, 1] >= y0) & (D[:, 1] < y1)
        if not inside.any():
            out.append(l)
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, (~inside).astype(np.int8), 0]))
        for a_, b_ in zip(edges[::2], edges[1::2]):               # runs of points outside the boxes
            if b_ - a_ >= 2:
                out.append(DstLine(D[a_:b_].copy(), False, l.rgb, l.width))
    return out


class GridScene:
    """All pictures on one canvas.

    Morph: `lw_mode` "span" = every picture dissolves into ONE drawing that covers the whole canvas;
           "same" / "per_tile" = every picture morphs into its own copy of a drawing.
    Sliders: `scope` "tile" = one slider per picture, "row" = one per row, "grid" = one over the whole canvas."""

    def __init__(self, results: Sequence, layout: GridLayout, linework: Sequence[Linework],
                 lw_mode: str, opts: MorphOptions, scope: str,
                 plans: Optional[Sequence[Optional[MorphPlan]]] = None,
                 tile_lws: Optional[Sequence[Optional[Linework]]] = None, row_specs: Optional[Sequence] = None,
                 workers: int = 1, pool=None, dropped: Optional[Sequence[int]] = None,
                 stay: Optional[Sequence[int]] = None, lw_area: str = "all"):
        if not results:
            raise ValueError("the grid has no images")
        if len(results) > layout.cells:
            raise ValueError("more pictures than cells in the grid")
        self.layout, self.scope, self.lw_mode, self.opts = layout, scope, lw_mode, opts
        self.row_specs = list(row_specs) if row_specs else []
        self._workers, self._pool = workers, pool
        self.dropped = {k for k in (dropped or ()) if 0 <= k < len(results)}
        self.stay = {k for k in (stay or ()) if 0 <= k < len(results)} - self.dropped
        self.lw_area = lw_area                       # all = the drawing covers the canvas | morph = only the morphing squares
        self.results = list(results)
        self.lw = list(linework)
        self.tiles: List[Scene] = []
        for k, res in enumerate(self.results):
            lw = None
            out = k in self.dropped or k in self.stay
            if lw_mode != "span" and not out:
                lw = tile_lws[k] if tile_lws is not None else tile_linework(self.lw, lw_mode, k)
            self.tiles.append(scene_from_result(res, lw, opts, plans[k] if plans and not out else None))
        self.merged: Scene = self._merge()

    def morphing(self) -> List[int]:
        """Pictures that take part in the morph."""
        return [k for k in range(len(self.results)) if k not in self.dropped and k not in self.stay]

    def _merge(self) -> Scene:
        L = self.layout
        CW, CH = L.canvas
        offs = [L.origin(k) for k in range(len(self.results))]
        lines: List[DstLine] = []
        plan: Optional[MorphPlan] = None
        morph = self.morphing()
        if self.lw_mode == "span" and self.lw and morph:
            items: List[SrcItem] = []
            bases, base = [], 0
            for res in self.results:
                bases.append(base)
                base += len(res.shapes)
            for k in morph:                           # white and staying pictures do not take part in the morph
                dx, dy = offs[k]
                items += offset_items(items_from_result(self.results[k]), dx, dy, bases[k])
            if self.lw_area == "morph":               # the drawing fits the area of the morphing squares only
                bx, by, bw, bh = self.morph_area()
                lines = offset_lines(fit_linework(self.lw[0], bw, bh, self.opts), bx, by)
            else:
                lines = fit_linework(self.lw[0], CW, CH, self.opts)
            lines = lines_outside(lines, [L.box(k) for k in self.stay])     # no target lines inside squares that stay
            plan = build_plan(items, lines, (CW, CH), self.opts, self._workers, self._pool) if lines and items else None
            if plan is None:
                lines = []
        else:
            have = [(t, o) for k, (t, o) in enumerate(zip(self.tiles, offs)) if t.plan is not None and k in morph]
            if have:
                plan = merge_plans([t.plan for t, _ in have], [o for _, o in have], (CW, CH))
                for k, (t, (dx, dy)) in enumerate(zip(self.tiles, offs)):
                    if k in morph:
                        lines += offset_lines(t.lines, dx, dy)
        results = self.results

        def provider(bg: str):
            def canvas():
                return background(CH, CW, bg if bg != "checker" else "white")
            s, v = canvas(), canvas()
            for k, res in enumerate(results):
                x, y = offs[k]
                a, b = source_image(res, bg), vector_image(res, bg)
                s[y:y + a.shape[0], x:x + a.shape[1]] = a
                v[y:y + b.shape[0], x:x + b.shape[1]] = b
            return s, v

        return Scene((CW, CH), plan, lines, provider, transparent=True, sources=list(results),
                     dropped=frozenset(self.dropped), stay=frozenset(self.stay))

    def morph_area(self) -> Tuple[int, int, int, int]:
        """x, y, w, h of the smallest rectangle that holds all morphing squares."""
        return area_of(self.layout, self.morphing())

    def group_sets(self) -> Dict[str, List[Group]]:
        return {sc: self.groups(sc) for sc in ("tile", "row", "grid")}

    def groups(self, scope: Optional[str] = None) -> List[Group]:
        """The slider regions for a scope, each with its row's side / delay / own sweep sliders and its pictures."""
        scope = scope or self.scope
        L = self.layout
        n = len(self.results)

        starts = np.r_[0, np.cumsum([len(r) for r in L.grid_sizes])]

        def mk(box, r, tiles):
            """A slider region. A single square takes its own side / delay / sliders, a row or the grid the row's."""
            sp = self.row_specs[r] if r < len(self.row_specs) else None
            if sp is None:
                return Group(box, r, "auto", 0, tiles, [])
            if len(tiles) == 1 and scope == "tile":
                i = tiles[0][0] - int(starts[r])
                if 0 <= i < len(sp.paths):
                    return Group(box, r, sp.tile_side(i), sp.tile_delay(i), tiles, sp.tile_sweep(i))
            return Group(box, r, sp.side, sp.delay, tiles, list(sp.sweep))
        if scope == "grid":
            g = Group((0, 0) + L.canvas, 0, tiles=[(k, L.box(k)) for k in range(n)])
            return [g]
        out: List[Group] = []
        if scope == "row":
            for r in range(len(L.geom)):
                ks = [k for k in range(n) if L.row_of(k) == r]
                if not ks:
                    continue
                bx = [L.box(k) for k in ks]
                x0, y0 = min(b[0] for b in bx), min(b[1] for b in bx)
                x1, y1 = max(b[0] + b[2] for b in bx), max(b[1] + b[3] for b in bx)
                out.append(mk((x0, y0, x1 - x0, y1 - y0), r, [(k, L.box(k)) for k in ks]))
            return out
        for k in range(n):
            out.append(mk(L.box(k), L.row_of(k), [(k, L.box(k))]))
        return out


def tile_at(tiles: Sequence[Tuple[int, Tuple[int, int, int, int]]], x: float, y: float) -> Optional[int]:
    """Picture under the point (x, y) (output pixels) or None."""
    for k, (x0, y0, x1, y1) in tiles:
        if x0 <= x < x1 and y0 <= y < y1:
            return k
    return None


class GridRenderer(Renderer):
    """The merged grid canvas with one slider group per tile / row / grid."""

    def __init__(self, g: GridScene, stages: Sequence[Stage], cs: ClipSettings, out_w: int):
        CW, CH = g.layout.canvas
        f = out_w / CW
        out = (max(2, int(round(CW * f))) // 2 * 2, max(2, int(round(CH * f))) // 2 * 2)
        super().__init__(g.merged, stages, cs, out, g.group_sets())
        self.out = out
        self.grid = g
