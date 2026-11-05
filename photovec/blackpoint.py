"""Black points: at which Threshold does a picture turn (almost) completely black?

In black-and-white mode a pixel is black when its grey value is below the Threshold, so the white area of a picture only
shrinks while the slider goes up. Dark photos are black at a low Threshold, bright ones only near 255. To let a whole grid
finish its Threshold sweep at the same moment, every picture needs its own end value: the lowest Threshold at which no more
than x % of the picture is still white. That is measured on the thresholded and cleaned-up picture (no tracing of paths), a fraction of a second per picture."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MAX_PCT = 50.0
POINTS = 4097                               # the grey values are first sorted into 4097 evenly spaced ranks
SAMPLES = 49                                # then the real result (with noise clean-up) is measured at 49 thresholds
_FRAC = np.linspace(0.0, 1.0, POINTS)


class BlackCurve:
    """How the dark area of one picture grows with the Threshold, measured on what is really drawn.

    A first pass sorts the grey values (cheap), which tells where the area changes. Then the picture is thresholded and cleaned
    up exactly like the trace does it (areas smaller than the Noise setting disappear) at 49 thresholds placed at equal steps of
    that first curve, and the dark share of the cleaned result is stored. That matters for grainy photos: tiny speckles are
    removed by the trace, so the drawn white share differs from the raw grey values by several percentage points."""

    def __init__(self, ts: np.ndarray, dark: np.ndarray):
        self.ts = np.asarray(ts, np.float64)                     # increasing Thresholds
        self.dark = np.asarray(dark, np.float64)                 # dark share (0..1), non-decreasing
        self._strict = self.dark + 1e-9 * np.arange(len(self.dark))

    def white_at(self, threshold: float) -> float:
        """Share (0..1) of the picture that is still white at this Threshold."""
        return 1.0 - float(np.interp(threshold, self.ts, self.dark))

    def threshold_for_white(self, white: float) -> float:
        """The Threshold at which exactly this share (0..1) of the picture is white (continuous, at most 255)."""
        return float(min(255.0, np.interp(1.0 - min(max(white, 0.0), 1.0), self._strict, self.ts)))

    def at(self, white_pct: float) -> float:
        """Lowest Threshold at which at most `white_pct` % of the picture is white (255 at most: the slider's end)."""
        return self.threshold_for_white(min(max(white_pct, 0.0), MAX_PCT) / 100.0)


def black_curve(src) -> Optional[BlackCurve]:
    """The curve of a sweep source, or None when it has no Threshold that turns it black (colour / grey modes, Linework Snap)."""
    if getattr(src, "linework", False) or getattr(src.params, "mode", "bw") != "bw":
        return None
    from dataclasses import replace
    from .trace import bw_grey, clean_labels, make_labels
    g = bw_grey(src.bgr, True, getattr(src.params, "blur_sigma", 0.7))
    vis = src.opaque if src.opaque is not None else None
    gv = np.sort((g[vis] if vis is not None else g).ravel())
    if gv.size == 0:
        return None
    first = BlackCurve(gv[np.rint(_FRAC * (gv.size - 1)).astype(np.int64)], _FRAC)       # grey value at every rank = raw curve
    p = src.params
    noise = max(1, int(round(p.noise * p.scale ** 2)))
    ts = [0.0] + [first.threshold_for_white(1.0 - j / (SAMPLES - 1)) for j in range(1, SAMPLES)] + [255.0]
    ts = sorted(set(round(t, 4) for t in ts))
    dark = []
    for t in ts:
        labels, pal, _ = make_labels(src.bgr, replace(p, threshold=t, spread=True), src.opaque)
        lab = clean_labels(labels, noise, len(pal))
        d = lab == 1
        dark.append(float(d[vis].mean() if vis is not None else d.mean()))
    dark = np.maximum.accumulate(np.array(dark))               # a Threshold never makes a picture whiter
    dark[0] = 0.0
    return BlackCurve(np.array(ts), dark)


class GroupBlack:
    """White-% -> Threshold for a region with one or more pictures (the slowest picture decides). With exactly one picture
    `curve` is available too, which lets the sweep follow that picture's own white-area curve."""

    def __init__(self, curves: Sequence[BlackCurve]):
        self.curves = list(curves)
        self.curve: Optional[BlackCurve] = self.curves[0] if len(self.curves) == 1 else None

    def __call__(self, pct: float) -> float:
        return max(c.at(pct) for c in self.curves)


def group_black(curves: Sequence[Optional[BlackCurve]]) -> Optional[GroupBlack]:
    cs = [c for c in curves if c is not None]
    return GroupBlack(cs) if cs else None
