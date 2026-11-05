"""Rendering a show (no windows in here, so it can be tested and used from any thread).

`ShowPhoto` holds what the Show Maker knows about one photo, `render_show` traces every photo at all its Threshold steps and writes
the `.photovecshow` file (showfile.py). The photos are traced exactly like in the other windows (same tracer, same mask finishing)."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .letters import Letter, Stroke
from .showfile import DEFAULT_LABELS, CaptionStyle, PhotoInfo, ShowWriter, threshold_of


@dataclass
class ShowPhoto:
    path: str = ""
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)      # uint8 0..255, any size; None = no mask
    thr_min: float = 40.0             # Threshold of the first / last prerendered picture
    thr_max: float = 220.0
    steps: int = 40                   # how many pictures (= positions of slider 1)
    noise: float = 25.0               # Image Trace "Noise"
    preset: str = "Black and White Logo"
    bg_min: float = 0.0               # slider 2: percent of the background photo that is visible at its left / right end
    bg_max: float = 100.0
    blur: float = 0.0                 # background blur at the left end of slider 2
    start1: float = 0.5               # where the sliders rest when the photo is opened in the Viewer
    start2: float = 0.0               # (without a mask slider 2 fades the drawing into the photo, so it starts at the drawing)
    letters: List[Letter] = field(default_factory=list, compare=False, repr=False)   # marked letters, in pixels of the ORIGINAL photo
    font: str = "Futura"              # the font slider 3 leads to (a family name or the path of a .ttf / .otf)
    lmotion: str = "melt"             # "melt" = all letters at once, "wave" = one after the other
    start3: float = 0.0               # where slider 3 rests when the photo is opened
    reach: float = 1.0                # how far around a skeleton the ink belongs to its letter (1 = automatic)
    caption: str = ""                 # the photo's name, shown on the picture in the Viewer
    letters_on: bool = False          # the Show Maker's checkbox: while it is off the marked letters wait in `letters_keep` and the photo has no slider 3
    letters_keep: List[Letter] = field(default_factory=list, compare=False, repr=False)
    rev: int = field(default=0, compare=False, repr=False)      # counts changes of the mask / style (the Show Maker's preview cache)


@dataclass
class ShowSettings:
    title: str = ""
    labels: Tuple[str, ...] = DEFAULT_LABELS
    quality: int = 1600               # long side (px) the photos are traced and stored at
    mask_smooth: float = 0.0
    mask_soft: float = 0.0
    mask_vector: bool = True
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)      # how the names of the photos look in the Viewer
    caption_font_file: str = ""       # a font file to put into the show (so the names look the same on other computers); "" = none


def _settings(p: ShowPhoto, ss: ShowSettings, tile: int):
    from .multi import MultiSettings
    masked = p.mask is not None
    return MultiSettings(preset=p.preset, tile=int(tile), layers=masked, only_mask=masked, mask_smooth=ss.mask_smooth,
                         mask_soft=ss.mask_soft, mask_vector=ss.mask_vector)


def build_source(p: ShowPhoto, ss: ShowSettings, tile: int):
    """The photo as it is traced: size, preset, finished mask, and (with a mask) only the masked part is traced."""
    from .multi import load_panel_source
    from .rows import image_size
    iw, ih = image_size(p.path)
    return load_panel_source(p.path, _settings(p, ss, min(int(tile), max(iw, ih))), p.mask)      # (a small photo is not enlarged)


def step_values(p: ShowPhoto, k: int) -> Dict[str, float]:
    v = {"threshold": float(threshold_of(photo_info(p, 1, 1), k)), "noise": float(p.noise)}
    if p.mask is not None:
        v.update(layer_p=1.0, layer_g=1.0, layer_f=1.0)                  # (so the tracer also returns how much of every pixel is drawing)
    return v


def photo_info(p: ShowPhoto, w: int, h: int, paper=(255, 255, 255), has_ink: bool = False, letters: bool = False) -> PhotoInfo:
    return PhotoInfo(w=w, h=h, steps=p.steps, thr_min=p.thr_min, thr_max=p.thr_max, bg_min=p.bg_min, bg_max=p.bg_max, blur=p.blur,
                     start1=p.start1, start2=p.start2, masked=p.mask is not None, has_ink=has_ink, paper=tuple(paper),
                     name=p.path.replace("\\", "/").rsplit("/", 1)[-1], caption=p.caption, letters=letters, start3=p.start3, lmotion=p.lmotion).clamp()


def scaled_letters(letters: List[Letter], f: float) -> List[Letter]:
    """Copies of the letters with every coordinate multiplied by f."""
    out = []
    for l in letters:
        strokes = []
        for st in l.strokes:
            sc = lambda q: None if q is None else (q[0] * f, q[1] * f)
            strokes.append(Stroke(pts=[sc(q) for q in st.pts], hin=[sc(q) for q in st.hin], hout=[sc(q) for q in st.hout], closed=st.closed))
        out.append(Letter(char=l.char, strokes=strokes))
    return out


def check_letters(letters: List[Letter]) -> Optional[str]:
    """None when every letter is finished, otherwise what is missing (shown to the user before rendering)."""
    for n, l in enumerate(letters, 1):
        if not l.char.strip():
            return f"Letter {n} has no character yet."
        if not any(len(s.pts) >= 2 for s in l.strokes):
            return f"Letter “{l.char}” ({n}) has no path yet (a path needs at least two anchors)."
    return None


def fit_photo_letters(p: ShowPhoto, size: Tuple[int, int], orig: Tuple[int, int], ink: Optional[np.ndarray]):
    """(LetterMorph list, note) for the marked letters of `p`. size = (w, h) of the stored photo, orig = (w, h) of the original photo,
    ink = how much of every pixel is drawing (HxW float 0..1, stored size) or None."""
    from .letterfit import layout_text, make_morph, order_letters, resolve_font
    bad = check_letters(p.letters)
    if bad:
        raise ValueError(bad)
    font, note = resolve_font(p.font)
    f = size[0] / float(max(1, orig[0]))
    letters = scaled_letters(p.letters, f)
    place = layout_text(letters, font, order_letters(letters))
    return [make_morph(l, place[i], font, ink, reach_factor=p.reach) for i, l in enumerate(letters)], note


def reference_ink(vec: np.ndarray, ink: Optional[np.ndarray], paper) -> np.ndarray:
    """The drawing as a 0..1 map (the ink map when there is one, else how far the picture differs from the paper)."""
    if ink is not None:
        return ink.astype(np.float32) if ink.dtype != np.uint8 else ink.astype(np.float32) / 255.0
    d = np.abs(vec.astype(np.float32) - np.asarray(paper, np.float32).reshape(1, 1, 3)).max(axis=2)
    return np.clip(d / 96.0, 0.0, 1.0)


def render_step(src, p: ShowPhoto, k: int, cache: Optional[dict] = None):
    """(vector picture, ink 0..1 or None, paper colour) of step k."""
    from .sweep import render_values
    return render_values(src, "white", step_values(p, k), cache if cache is not None else {}, parts=True)


def _steps_serial(src, p, ks, cache, on_frame) -> None:
    for k in ks:
        on_frame(k, *render_step(src, p, k, cache))


def render_show(photos: List[ShowPhoto], ss: ShowSettings, path: str,
                progress: Optional[Callable[[str, int, int], bool]] = None, workers: int = 0, pool=None,
                notes: Optional[List[str]] = None) -> bool:
    """Trace every photo at every step and write the show to `path`. progress(text, done, total) may return True to cancel
    (then False is returned and nothing is written). `pool` = a process pool to spread the steps over (None: this thread only).
    Things worth telling the user (a font that had to be replaced) are appended to `notes`."""
    from .prerender import Prerender
    if not photos:
        raise ValueError("no photos")
    writer = ShowWriter(path)
    infos: List[PhotoInfo] = []
    total = sum(max(2, int(p.steps)) for p in photos)
    done = 0

    def tick(text: str) -> bool:
        return bool(progress(text, done, total)) if progress else False
    try:
        for i, p in enumerate(photos):
            if tick(f"Photo {i + 1} of {len(photos)}: preparing"):
                writer.abort()
                return False
            src = build_source(p, ss, ss.quality)
            writer.add_photo(i, src.bgr, src.mask)
            steps = max(2, int(p.steps))
            p.steps = steps
            results: Dict[int, tuple] = {}
            errors: List[int] = []
            cv = threading.Condition()

            def on_frame(k, vec, ink, paper):
                with cv:
                    results[k] = (vec, ink, paper)
                    cv.notify_all()

            def on_error(tags):
                with cv:
                    errors.extend(tags)
                    cv.notify_all()
            ks = list(range(steps))
            pr = Prerender(src, "white", pool=pool) if pool is not None else None
            started = pr is not None and pr.submit([(k, step_values(p, k)) for k in ks], on_frame, on_error, chunk=4)
            if not started:
                pr = None
            cache: dict = {}
            serial_ks = [] if started else list(ks)
            written = 0
            paper, has_ink = (255, 255, 255), False
            kref = int(round(min(max(p.start1, 0.0), 1.0) * (steps - 1)))
            ref = None
            while written < steps:
                if serial_ks:                                                       # no pool (or a chunk failed): this thread does it
                    k = serial_ks.pop(0)
                    on_frame(k, *render_step(src, p, k, cache))
                with cv:
                    while not results and not serial_ks and not errors:
                        cv.wait(0.2)
                        if tick(f"Photo {i + 1} of {len(photos)}: step {written + 1} of {steps}"):
                            break
                    if errors:
                        serial_ks.extend(sorted(set(errors) - set(results) - set(serial_ks)))
                        errors.clear()
                    ready = sorted(results)
                    batch = [(k, results.pop(k)) for k in ready]
                if tick(f"Photo {i + 1} of {len(photos)}: step {written + 1} of {steps}"):
                    if pr is not None:
                        pr.close()
                    writer.abort()
                    return False
                for k, (vec, ink, pp) in batch:
                    writer.add_step(i, k, vec, ink)
                    paper, has_ink = pp, ink is not None
                    if k == kref:
                        ref = reference_ink(vec, ink, pp)
                    written += 1
                    done += 1
            if pr is not None:
                pr.close()
            has_letters = bool(p.letters)
            if has_letters:
                from .rows import image_size
                tick(f"Photo {i + 1} of {len(photos)}: fitting the letters to {p.font}")
                morphs, note = fit_photo_letters(p, (src.bgr.shape[1], src.bgr.shape[0]), image_size(p.path), ref)
                writer.add_letters(i, morphs)
                if note and notes is not None:
                    notes.append(note)
            infos.append(photo_info(p, src.bgr.shape[1], src.bgr.shape[0], paper, has_ink, has_letters))
        font_bytes = None
        if ss.caption_font_file and os.path.isfile(ss.caption_font_file):
            with open(ss.caption_font_file, "rb") as fh:
                font_bytes = fh.read()
        writer.finish(infos, ss.title, ss.labels, ss.caption_style, font_bytes)
        tick("Done")
        return True
    except BaseException:
        writer.abort()
        raise
