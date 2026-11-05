"""Before/after compositing (one slider or two), animation timeline, GIF and MP4 export."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .trace import TraceResult, checkerboard, render_trace, render_trace_rgba

SIDES_SINGLE = {
    "Right → Left (after enters from the right)": ("x", "right"),
    "Left → Right (after enters from the left)": ("x", "left"),
    "Bottom → Top (after enters from the bottom)": ("y", "bottom"),
    "Top → Bottom (after enters from the top)": ("y", "top"),
}


@dataclass
class AnimSettings:
    mode: str = "single"           # single | quad
    axis: str = "x"                # single mode: x = vertical divider, y = horizontal divider
    side_x: str = "right"          # after image enters from this side (vertical divider)
    side_y: str = "bottom"         # ... (horizontal divider)
    show_text: bool = True
    text_before: str = "BEFORE"
    text_after: str = "AFTER"
    divider: bool = True
    duration: float = 3.0          # seconds of motion (a ping-pong cycle counts as one duration)
    hold_start: float = 0.5
    hold_end: float = 0.8
    fps: int = 25
    pingpong: bool = False
    easing: str = "smooth"         # smooth | linear
    ypeak: float = 0.5             # quad animation: how far the second slider travels (0..1)
    max_width: int = 1080


# --------------------------------------------------------------------------- images
CUSTOM_BG: dict = {}                 # name -> RGB of colours chosen by the user (Multi-Trace's own background)


def background(h: int, w: int, kind: str = "white") -> np.ndarray:
    if kind in CUSTOM_BG:
        return np.full((h, w, 3), CUSTOM_BG[kind][::-1], np.uint8)
    if kind == "checker":
        return checkerboard(h, w)
    if kind == "dark":
        return np.full((h, w, 3), (44, 40, 40), np.uint8)
    return np.full((h, w, 3), 255, np.uint8)


def vector_image(res: TraceResult, bg: str = "white", view: str = "result") -> np.ndarray:
    """The traced vector rendered on `bg` (transparent sources keep their transparency)."""
    h, w = res.image_bgr.shape[:2]
    if res.opaque is None or view == "outline":
        return render_trace(res, "outline" if view == "outline" else "result")
    from .trace import _trace_layers
    col, cov = _trace_layers(res)
    out = background(h, w, bg)                                           # (a fresh picture)
    x, y, bw, bh = cv2.boundingRect(cov)                                 # only the part that has any drawing is blended
    if bw and bh:
        sl = (slice(y, y + bh), slice(x, x + bw))
        a = cov[sl]
        if a.min() == 255:
            out[sl] = col[sl]
        else:
            base = out[sl].astype(np.float32)
            d = np.subtract(col[sl], base, dtype=np.float32)
            d *= (a.astype(np.float32) * np.float32(1.0 / 255.0))[..., None]
            d += base
            out[sl] = cv2.add(d, (0.0, 0.0, 0.0, 0.0), dtype=cv2.CV_8U)
    return out


def source_image(res: TraceResult, bg: str = "white") -> np.ndarray:
    if res.opaque is None:
        return res.image_bgr.copy()
    out = background(*res.image_bgr.shape[:2], bg)
    out[res.opaque] = res.image_bgr[res.opaque]
    return out


def make_pair(res: TraceResult, bg: str = "white", view: str = "result") -> Tuple[np.ndarray, np.ndarray]:
    """(before, after) BGR images of identical size; transparent sources are put on `bg`."""
    return source_image(res, bg), vector_image(res, bg, view)


# --------------------------------------------------------------------------- compositing
def _after_mask_axis(n: int, p: float, side_first: bool) -> Tuple[np.ndarray, float]:
    """Boolean mask over one axis and the divider coordinate. p = fraction covered by 'after'."""
    xs = np.arange(n) + 0.5
    if side_first:                 # after enters from the left / top
        edge = p * n
        return xs < edge, edge
    edge = (1.0 - p) * n           # after enters from the right / bottom
    return xs >= edge, edge


def _text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float) -> int:
    """Outlined label (readable on any background). Returns text width in px."""
    font = cv2.FONT_HERSHEY_DUPLEX
    th = max(1, int(round(scale * 1.6)))
    (tw, _), _b = cv2.getTextSize(text, font, scale, th)
    cv2.putText(img, text, org, font, scale, (20, 20, 20), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), th, cv2.LINE_AA)
    return tw


def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    """Constant runs (start, stop, value) of a 1-D boolean mask."""
    cuts = [0] + (np.flatnonzero(np.diff(mask.astype(np.int8))) + 1).tolist() + [len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(cuts[:-1], cuts[1:])]


def compose(before: np.ndarray, after: np.ndarray, px: float, py: float, s: AnimSettings) -> np.ndarray:
    """One frame. px/py are slider positions 0..1 (0 = all before in single mode).
    Rectangles are copied with slice assignment (no per-pixel masks): a frame costs one array copy."""
    H, W = before.shape[:2]
    first_x, first_y = s.side_x == "left", s.side_y == "top"
    out = before.copy()
    ax = ay = None
    ex, ey = W, H
    if s.mode == "quad":
        ax, ex = _after_mask_axis(W, px, first_x)
        ay, ey = _after_mask_axis(H, py, first_y)
        for ya, yb, vy in _runs(ay):
            for xa, xb, vx in _runs(ax):
                if vx != vy:
                    out[ya:yb, xa:xb] = after[ya:yb, xa:xb]
    elif s.axis == "x":
        ax, ex = _after_mask_axis(W, px, first_x)
        for xa, xb, v in _runs(ax):
            if v:
                out[:, xa:xb] = after[:, xa:xb]
    else:
        ay, ey = _after_mask_axis(H, px, first_y)      # single slider: px drives either axis
        for ya, yb, v in _runs(ay):
            if v:
                out[ya:yb] = after[ya:yb]

    def is_after(cy: int, cx: int) -> bool:
        if s.mode == "quad":
            return bool(ay[cy] ^ ax[cx])
        return bool(ax[cx]) if s.axis == "x" else bool(ay[cy])

    xs = [0.0, W]
    ys = [0.0, H]
    if s.mode == "quad" or s.axis == "x":
        xs = [0.0] + ([ex] if 0 < ex < W else []) + [W]
    if s.mode == "quad" or s.axis == "y":
        ys = [0.0] + ([ey] if 0 < ey < H else []) + [H]
    if s.divider:
        t = max(2, W // 400)
        for x in xs[1:-1]:
            cv2.line(out, (int(x), 0), (int(x), H), (255, 255, 255), t + 2, cv2.LINE_AA)
            cv2.line(out, (int(x), 0), (int(x), H), (60, 60, 60), max(1, t - 1), cv2.LINE_AA)
        for y in ys[1:-1]:
            cv2.line(out, (0, int(y)), (W, int(y)), (255, 255, 255), t + 2, cv2.LINE_AA)
            cv2.line(out, (0, int(y)), (W, int(y)), (60, 60, 60), max(1, t - 1), cv2.LINE_AA)
    if s.show_text:
        scale = max(0.5, W / 900.0)
        m = int(14 * scale)
        for xa, xb in zip(xs[:-1], xs[1:]):
            for ya, yb in zip(ys[:-1], ys[1:]):
                cx, cy = int((xa + xb) / 2), int((ya + yb) / 2)
                label = s.text_after if is_after(min(cy, H - 1), min(cx, W - 1)) else s.text_before
                if not label:
                    continue
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, max(1, int(round(scale * 1.6))))
                if xb - xa < tw + 2 * m or yb - ya < th + 2 * m:
                    continue                      # region too small for its label
                _text(out, label, (int(xa) + m, int(ya) + m + th), scale)
    return out


# --------------------------------------------------------------------------- timeline
def _ease(u: float, kind: str) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u) if kind == "smooth" else u


def frame_count(s: AnimSettings) -> int:
    return max(2, int(math.ceil((s.hold_start + s.duration + s.hold_end) * s.fps)))


def slider_positions(i: int, n: int, s: AnimSettings) -> Tuple[float, float]:
    """(px, py) for frame i of n."""
    t = i / max(1, s.fps)
    if t < s.hold_start:
        m = 0.0
    elif t < s.hold_start + s.duration:
        m = (t - s.hold_start) / max(s.duration, 1e-6)
    else:
        m = 1.0
    return slider_at(m, s)


def slider_at(m: float, s: AnimSettings) -> Tuple[float, float]:
    """(px, py) for a motion fraction m in 0..1 (holds are handled by the caller)."""
    m = min(1.0, max(0.0, m))
    if s.pingpong:
        lin = 1.0 - abs(2.0 * m - 1.0)        # 0 -> 1 -> 0
        lin = 0.0 if m <= 0 or m >= 1 else lin
    else:
        lin = m
    px = _ease(lin, s.easing)
    py = 0.0
    if s.mode == "quad":
        py = s.ypeak * math.sin(math.pi * lin)
    return px, py


def frames(before: np.ndarray, after: np.ndarray, s: AnimSettings,
           progress: Optional[Callable[[int, int], bool]] = None) -> Iterator[np.ndarray]:
    """Yield composed frames scaled to s.max_width (even dimensions)."""
    H, W = before.shape[:2]
    if s.max_width and W > s.max_width:
        k = s.max_width / W
        size = (int(round(W * k)) // 2 * 2, int(round(H * k)) // 2 * 2)
    else:
        size = (W // 2 * 2, H // 2 * 2)
    b = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(after, size, interpolation=cv2.INTER_AREA)
    n = frame_count(s)
    for i in range(n):
        if progress is not None and progress(i, n):
            return
        px, py = slider_positions(i, n, s)
        yield compose(b, a, px, py, s)


# --------------------------------------------------------------------------- export
def export_gif(path: str, before, after, s: AnimSettings, progress=None) -> int:
    from PIL import Image
    gen = frames(before, after, s, progress)
    first = next(gen, None)
    if first is None:
        return 0
    # one shared palette built from both images: no colour flicker between frames
    pal_src = np.vstack([cv2.resize(first, (first.shape[1] // 2, first.shape[0] // 2)),
                         cv2.resize(compose(first, first, 1.0, 0.0, AnimSettings(show_text=False, divider=False)),
                                    (first.shape[1] // 2, first.shape[0] // 2))])
    b_small = cv2.resize(before, (first.shape[1] // 2, first.shape[0] // 2), interpolation=cv2.INTER_AREA)
    a_small = cv2.resize(after, (first.shape[1] // 2, first.shape[0] // 2), interpolation=cv2.INTER_AREA)
    pal = Image.fromarray(cv2.cvtColor(np.vstack([b_small, a_small]), cv2.COLOR_BGR2RGB)).quantize(
        colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    imgs = [Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)).quantize(palette=pal, dither=Image.Dither.NONE)]
    for fr in gen:
        imgs.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).quantize(palette=pal, dither=Image.Dither.NONE))
    dur = max(20, int(round(1000.0 / s.fps / 10.0)) * 10)     # GIF delays are 10 ms steps
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=False, disposal=1)
    return len(imgs)


def export_mp4(path: str, before, after, s: AnimSettings, progress=None) -> int:
    """H.264 via imageio-ffmpeg when available (plays everywhere), else OpenCV mp4v."""
    gen = frames(before, after, s, progress)
    first = next(gen, None)
    if first is None:
        return 0
    h, w = first.shape[:2]
    count = 0
    try:
        import imageio_ffmpeg
        wr = imageio_ffmpeg.write_frames(path, (w, h), fps=s.fps, codec="libx264", quality=7,
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2,
                                         output_params=["-preset", "veryfast"])     # same quality (crf), ~2x faster, ~2 % bigger file
        wr.send(None)
        try:
            for fr in [first] + list(gen):
                wr.send(np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).tobytes())
                count += 1
        finally:
            wr.close()
        return count
    except ImportError:
        pass
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), s.fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("no MP4 encoder available (install imageio-ffmpeg)")
    try:
        for fr in [first] + list(gen):
            vw.write(fr)
            count += 1
    finally:
        vw.release()
    return count


# --------------------------------------------------------------------------- generic writers
class Cancelled(Exception):
    pass


def write_gif_stream(path: str, count: int, frame_at, fps: int, progress=None, threads: int = 1) -> int:
    """GIF from `count` frames produced on demand by frame_at(i) -> BGR. Memory stays small: frames are
    quantised to one shared palette (sampled over the whole clip -> no flicker) and Pillow stores only the
    changed rectangle of every frame, so a moving slider costs a few KB per frame.
    progress(i, n) may return True to cancel (the half-written file is removed). threads > 1: that many frames are made and quantised
    at once (frame_at must then be safe to call from several threads)."""
    from PIL import Image
    if count <= 0:
        return 0
    pick = np.unique(np.linspace(0, count - 1, min(12, count)).astype(int))
    smalls = []
    for i in pick:
        f = frame_at(int(i))
        h, w = f.shape[:2]
        smalls.append(cv2.resize(f, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA))
    pal = Image.fromarray(cv2.cvtColor(np.vstack(smalls), cv2.COLOR_BGR2RGB)).quantize(
        colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    del smalls

    def make(i: int):
        return Image.fromarray(cv2.cvtColor(frame_at(i), cv2.COLOR_BGR2RGB)).quantize(palette=pal, dither=Image.Dither.NONE)

    ex = None
    futs: list = []
    if threads > 1 and count > 2:
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=threads)
        nxt = [0]

        def fill(upto: int) -> None:
            while nxt[0] < count and nxt[0] < upto + threads * 3:
                futs.append(ex.submit(make, nxt[0]))
                nxt[0] += 1

    def img(i: int):
        if progress is not None and progress(i, count):
            raise Cancelled()
        if ex is None:
            return make(i)
        fill(i)
        out, futs[i] = futs[i].result(), None                 # (no reference kept: a long clip must not pile up in memory)
        return out

    dur = max(20, int(round(1000.0 / fps / 10.0)) * 10)
    try:
        first = img(0)
        first.save(path, save_all=True, append_images=(img(i) for i in range(1, count)), duration=dur, loop=0,
                   optimize=False, disposal=1)
    except Cancelled:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
        import os
        if os.path.exists(path):
            os.remove(path)
        return 0
    finally:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
    return count


def write_gif(path: str, frames_bgr: List[np.ndarray], fps: int) -> int:
    """GIF from a list of rendered frames (see write_gif_stream for long clips)."""
    return write_gif_stream(path, len(frames_bgr), lambda i: frames_bgr[i], fps)


def prefetch(it, depth: int = 6):
    """Run a (render) generator in a thread so rendering overlaps with encoding / disk writes."""
    import queue
    import threading
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    END = object()
    stop = threading.Event()

    def work():
        try:
            for x in it:
                while not stop.is_set():
                    try:
                        q.put(x, timeout=0.2)
                        break
                    except queue.Full:
                        pass
                if stop.is_set():
                    return
            q.put(END)
        except BaseException as exc:          # surface errors in the consumer
            q.put(exc)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while True:
            x = q.get()
            if x is END:
                return
            if isinstance(x, BaseException):
                raise x
            yield x
    finally:
        stop.set()


def write_mp4(path: str, frame_iter, fps: int) -> int:
    """Stream frames into H.264 MP4 (imageio-ffmpeg) or OpenCV mp4v fallback. Even sizes enforced."""
    it = iter(frame_iter)
    first = next(it, None)
    if first is None:
        return 0

    def even(f):
        h, w = f.shape[:2]
        return f if (h % 2 == 0 and w % 2 == 0) else f[: h // 2 * 2, : w // 2 * 2]

    first = even(first)
    h, w = first.shape[:2]
    count = 0
    try:
        import imageio_ffmpeg
        wr = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec="libx264", quality=7,
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2,
                                         output_params=["-preset", "veryfast"])     # same quality (crf), ~2x faster, ~2 % bigger file
        wr.send(None)
        try:
            wr.send(np.ascontiguousarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)).tobytes())
            count = 1
            for f in it:
                wr.send(np.ascontiguousarray(cv2.cvtColor(even(f), cv2.COLOR_BGR2RGB)).tobytes())
                count += 1
        finally:
            wr.close()
        return count
    except ImportError:
        pass
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("no MP4 encoder available (install imageio-ffmpeg)")
    try:
        vw.write(first)
        count = 1
        for f in it:
            vw.write(even(f))
            count += 1
    finally:
        vw.release()
    return count
