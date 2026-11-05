"""The data of the Timeline window: keyframes on lanes, like in a video editor.

A LANE is one slider over time (Threshold, Noise, Paths, Background photo, Graffiti, Background blur). It holds KEYFRAMES
(time in seconds, value, and how to get to the next keyframe). Before the first keyframe the lane keeps the first value, after the
last one the last value. Nothing here knows about Qt: the Timeline renders through the same tracks as Multi-Trace (`make_tracks`).
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .sweep import SWEEP_PARAMS, Track

EASES = ("linear", "smooth", "ease_in", "ease_out", "hold")
EASE_LABEL = {"linear": "Linear", "smooth": "Smooth (slow in and out)", "ease_in": "Ease in (slow start)", "ease_out": "Ease out (slow end)",
              "hold": "Hold (jump at the next key)"}


def ease_fn(kind: str, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    if kind == "smooth":
        return u * u * (3.0 - 2.0 * u)
    if kind == "ease_in":
        return u * u
    if kind == "ease_out":
        return 1.0 - (1.0 - u) * (1.0 - u)
    if kind == "hold":
        return 0.0 if u < 1.0 else 1.0
    return u


@dataclass
class Key:
    t: float
    v: float
    ease: str = "linear"                 # how the lane moves from this key to the next one


# lane id -> (label, minimum, maximum, discrete, unit, sweep parameter it drives, colour)
LANE_DEFS = {
    "threshold": ("Threshold", 0.0, 255.0, False, "", "threshold", "#ff9f1a"),
    "noise": ("Noise", 1.0, 100.0, True, "", "noise", "#4fc3f7"),
    "paths": ("Paths", 0.0, 100.0, False, "", "paths", "#b388ff"),
    "photo": ("Background photo", 0.0, 100.0, False, " %", "layer_f", "#81c784"),
    "graffiti": ("Graffiti vector", 0.0, 100.0, False, " %", "layer_g", "#ef5350"),
    "blur": ("Background blur", 0.0, 100.0, False, " %", "blur_f", "#90a4ae"),
}
ORDER = ("threshold", "noise", "paths", "photo", "graffiti", "blur")
LAYER_LANES = ("photo", "graffiti", "blur")


@dataclass
class Lane:
    id: str
    keys: List[Key] = field(default_factory=list)
    enabled: bool = True                 # Paths is optional (off = the preset's own value)

    @property
    def label(self) -> str:
        return LANE_DEFS[self.id][0]

    @property
    def lo(self) -> float:
        return LANE_DEFS[self.id][1]

    @property
    def hi(self) -> float:
        return LANE_DEFS[self.id][2]

    @property
    def discrete(self) -> bool:
        return LANE_DEFS[self.id][3]

    @property
    def unit(self) -> str:
        return LANE_DEFS[self.id][4]

    @property
    def color(self) -> str:
        return LANE_DEFS[self.id][6]

    def clamp(self, v: float) -> float:
        v = min(max(float(v), self.lo), self.hi)
        return float(int(round(v))) if self.discrete else v

    def sort(self) -> None:
        self.keys.sort(key=lambda k: k.t)

    def value(self, t: float) -> float:
        ks = self.keys
        if not ks:
            return self.lo
        if t <= ks[0].t:
            return ks[0].v
        if t >= ks[-1].t:
            return ks[-1].v
        i = bisect.bisect_right([k.t for k in ks], t) - 1
        a, b = ks[i], ks[i + 1]
        u = (t - a.t) / max(1e-9, b.t - a.t)
        return a.v + (b.v - a.v) * ease_fn(a.ease, u)

    def add(self, t: float, v: Optional[float] = None, ease: str = "linear") -> Key:
        """A keyframe at time t (an existing one within a millisecond is replaced); without a value the curve's own value is used,
        so adding a key never changes the look."""
        v = self.value(t) if v is None else v
        for k in self.keys:
            if abs(k.t - t) < 1e-3:
                k.v, k.ease = self.clamp(v), ease
                return k
        k = Key(float(t), self.clamp(v), ease)
        self.keys.append(k)
        self.sort()
        return k

    def remove(self, key: Key) -> bool:
        if len(self.keys) <= 1 or key not in self.keys:
            return False                              # a lane always keeps at least one keyframe
        self.keys.remove(key)
        return True

    def to_json(self) -> dict:
        return {"id": self.id, "enabled": self.enabled, "keys": [[round(k.t, 4), round(k.v, 4), k.ease] for k in self.keys]}

    @classmethod
    def from_json(cls, d: dict) -> "Lane":
        lane = cls(str(d["id"]), enabled=bool(d.get("enabled", True)))
        if lane.id not in LANE_DEFS:
            raise ValueError(f"unknown lane {lane.id!r}")
        for t, v, e in d.get("keys", []):
            lane.keys.append(Key(float(t), lane.clamp(v), e if e in EASES else "linear"))
        lane.sort()
        if not lane.keys:
            lane.keys.append(Key(0.0, lane.lo))
        return lane


@dataclass
class FrameTrack(Track):
    """A slider that has one value per frame (taken from the timeline) instead of equal steps between two ends."""
    n: int = 1
    fn: Optional[Callable[[int], float]] = None

    def steps(self) -> int:
        return max(1, self.n)

    def value(self, i: int) -> float:
        _, lo, hi, discrete, _ = SWEEP_PARAMS[self.param]
        v = min(max(self.fn(min(max(int(i), 0), self.n)), lo), hi)
        return float(int(round(v))) if discrete else round(v, 4)


@dataclass
class Timeline:
    duration: float = 10.0
    fps: int = 25
    lanes: Dict[str, Lane] = field(default_factory=dict)

    def __post_init__(self):
        for lid in ORDER:
            if lid not in self.lanes:
                self.lanes[lid] = Lane(lid, enabled=(lid != "paths"))
        for lane in self.lanes.values():
            if not lane.keys:
                lane.keys.append(Key(0.0, lane.lo))

    # ------------------------------------------------------------------ frames
    def frames(self) -> int:
        """Number of frames: one every 1/fps, the last one exactly at the end."""
        return max(2, int(round(self.duration * self.fps)) + 1)

    def time_of(self, frame: int) -> float:
        return min(max(frame, 0) / float(self.fps), self.duration)

    def frame_values(self, frame: float, layers: bool = False, photo_bg: bool = False) -> Dict[str, float]:
        """Slider values of one frame as the engine takes them (the same for the preview and for the export)."""
        t = min(max(frame, 0.0) / float(self.fps), self.duration)
        out = {"threshold": round(self.lanes["threshold"].value(t), 4), "noise": float(int(round(self.lanes["noise"].value(t))))}
        if self.lanes["paths"].enabled:
            out["paths"] = round(self.lanes["paths"].value(t), 4)
        if layers:
            out["layer_f"] = round(self.lanes["photo"].value(t) / 100.0, 4)
            out["layer_g"] = round(self.lanes["graffiti"].value(t) / 100.0, 4)
            out["blur_f"] = round(self.lanes["blur"].value(t), 3)
            if photo_bg:
                out["layer_p"] = 1.0
        return out

    def make_tracks(self, spec, st) -> List[Track]:
        """The slider tracks of the (one) picture for the Multi-Trace renderer: one value per frame."""
        from .multi import photo_background
        n = max(1, st.total_frames() - 1)
        layers = bool(st.layers and spec.mask is not None)
        photo = bool(layers and photo_background(st))
        keys = list(self.frame_values(0, layers, photo).keys())
        out = []
        for k in keys:
            out.append(FrameTrack(k, 0.0, 0.0, 1.0, 0, None, n, lambda i, k=k: self.frame_values(i, layers, photo)[k]))
        return out

    # ------------------------------------------------------------------ project
    def to_json(self) -> dict:
        return {"duration": self.duration, "fps": self.fps, "lanes": [self.lanes[i].to_json() for i in ORDER]}

    @classmethod
    def from_json(cls, d: dict) -> "Timeline":
        tl = cls(float(d.get("duration", 10.0)), int(d.get("fps", 25)), {})
        for ld in d.get("lanes", []):
            lane = Lane.from_json(ld)
            tl.lanes[lane.id] = lane
        tl.__post_init__()
        tl.duration = min(max(tl.duration, 0.2), 600.0)
        tl.fps = min(max(tl.fps, 1), 120)
        return tl

    def copy(self) -> "Timeline":
        return Timeline.from_json(json.loads(json.dumps(self.to_json())))

    def set_duration(self, seconds: float, scale_keys: bool = False) -> None:
        """A new length. The keys stay where they are (keys beyond the end are pulled in), or all keys stretch with the length."""
        seconds = min(max(float(seconds), 0.2), 600.0)
        f = seconds / max(1e-9, self.duration)
        for lane in self.lanes.values():
            for k in lane.keys:
                k.t = k.t * f if scale_keys else min(k.t, seconds)
            lane.sort()
        self.duration = seconds


def default_timeline(duration: float = 10.0, fps: int = 25) -> Timeline:
    """A starting point: Threshold from 60 to 200, a constant Noise, the background photo and graffiti fully there."""
    tl = Timeline(duration, fps, {})
    tl.lanes["threshold"].keys[:] = [Key(0.0, 60.0), Key(duration, 200.0)]
    tl.lanes["noise"].keys[:] = [Key(0.0, 5.0)]
    tl.lanes["paths"].keys[:] = [Key(0.0, 50.0)]
    tl.lanes["photo"].keys[:] = [Key(0.0, 100.0)]
    tl.lanes["graffiti"].keys[:] = [Key(0.0, 100.0)]
    tl.lanes["blur"].keys[:] = [Key(0.0, 0.0)]
    return tl


def sequence_vector_to_photo(tl: Timeline, threshold: float = 150.0) -> None:
    """Vector only in the mask on a transparent background -> the background photo fades in as the Threshold goes to 0."""
    d = tl.duration
    tl.lanes["threshold"].keys[:] = [Key(0.0, tl.lanes["threshold"].clamp(threshold)), Key(d, 0.0)]
    tl.lanes["photo"].keys[:] = [Key(0.0, 0.0), Key(d, 100.0)]
    tl.lanes["graffiti"].keys[:] = [Key(0.0, 100.0)]
    tl.lanes["blur"].keys[:] = [Key(0.0, 0.0)]


def sequence_photo_to_graffiti(tl: Timeline, threshold: float = 150.0) -> None:
    """The original photo -> the graffiti turns into vector -> the photo fades out -> only the vector graffiti is left."""
    d = tl.duration
    tl.lanes["threshold"].keys[:] = [Key(0.0, tl.lanes["threshold"].clamp(threshold))]
    tl.lanes["graffiti"].keys[:] = [Key(0.0, 0.0), Key(d * 0.4, 100.0)]
    tl.lanes["photo"].keys[:] = [Key(0.0, 100.0), Key(d * 0.5, 100.0), Key(d, 0.0)]
    tl.lanes["blur"].keys[:] = [Key(0.0, 0.0)]


def from_multi(spec, st) -> Timeline:
    """The movement of one Multi-Trace photo as keyframes: its Threshold / Noise / Paths points and the layer movement of the clip."""
    from .multi import value_over
    hs, _ = st.holds()
    fps = st.fps
    n = st.total_frames()
    tl = Timeline(max(0.2, (n - 1) / fps), fps, {})
    offs = st.point_offsets()
    for lid, vals in (("threshold", spec.thr), ("noise", spec.noise), ("paths", spec.paths)):
        if vals is None:
            tl.lanes[lid].enabled = False
            continue
        vals = list(vals)
        keys = [Key(0.0, tl.lanes[lid].clamp(vals[0]))]
        if len(vals) > 1:
            for j, v in enumerate(vals):
                off = offs[j] if len(offs) == len(vals) else round(j * offs[-1] / (len(vals) - 1))
                keys.append(Key((hs + off) / fps, tl.lanes[lid].clamp(v), "smooth" if st.ease == "smooth" else "linear"))
            keys.append(Key((n - 1) / fps, tl.lanes[lid].clamp(vals[-1])))
        tl.lanes[lid].keys[:] = _dedupe(keys)
    from .multi import layer_values
    end_s = (n - 1) / fps
    b0 = hs / fps                                                              # where the movement starts (seconds)
    s0 = b0 + max(0.0, st.layer_start)
    s1 = end_s if st.layer_end <= 0 else min(end_s, b0 + st.layer_end)
    own = bool(st.gra_own)
    g0 = b0 + max(0.0, st.gra_start)
    g1 = end_s if st.gra_end <= 0 else min(end_s, b0 + st.gra_end)
    ease = "smooth" if st.ease == "smooth" else "linear"
    for lid, k, scale in (("photo", "layer_f", 100.0), ("graffiti", "layer_g", 100.0), ("blur", "blur_f", 1.0)):
        if not st.layers:
            continue                                                           # layers were off: the lane keeps its default
        w0, w1 = (g0, g1) if (lid == "graffiti" and own) else (s0, s1)
        pts = sorted({0.0, round(w0, 4), round(w1, 4), round(end_s, 4)})
        keys = [Key(t, tl.lanes[lid].clamp(layer_values(st, int(round(t * fps)))[k] * scale), ease if w0 - 1e-6 <= t < w1 - 1e-6 else "linear")
                for t in pts]
        tl.lanes[lid].keys[:] = _dedupe(keys)
    return tl


def _dedupe(keys: List[Key]) -> List[Key]:
    out: List[Key] = []
    for k in sorted(keys, key=lambda k: k.t):
        if out and abs(out[-1].t - k.t) < 1e-3:
            out[-1] = k
        else:
            out.append(k)
    return out
