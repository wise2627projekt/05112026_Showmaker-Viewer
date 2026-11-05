"""Export with a TRANSPARENT background (MP4 and GIF cannot hold transparency).

The clip is rendered twice, once on pure black and once on pure white. Whatever is see-through shows the background, so
    alpha = 1 - (white - black) / 255        and        colour = black / alpha
for every pixel (the usual two-backgrounds matte). It works through the whole pipeline (traces, layers, the facade photo fading in,
gaps between photos) without that pipeline knowing about transparency.

Formats: ProRes 4444 (.mov: Premiere, After Effects, DaVinci, Final Cut), WebM VP9 with alpha (.webm: browsers, OBS) and a PNG
sequence (name_00001.png ...).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import replace
from typing import Callable, Iterator, List, Optional

import cv2
import numpy as np

FORMATS = {"mov": "ProRes 4444 (*.mov)", "webm": "WebM VP9 with alpha (*.webm)", "png": "PNG sequence (*.png)"}


def matte(black: np.ndarray, white: np.ndarray) -> np.ndarray:
    """The picture rendered on black and on white (BGR uint8) -> straight RGBA (uint8)."""
    b = black.astype(np.float32)
    w = white.astype(np.float32)
    a = np.clip(1.0 - (w - b).mean(axis=2) / 255.0, 0.0, 1.0)
    rgb = np.where(a[..., None] > 1.0 / 255.0, b / np.maximum(a, 1.0 / 255.0)[..., None], 0.0)
    out = np.empty(black.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(np.rint(rgb[..., ::-1]), 0, 255).astype(np.uint8)         # BGR -> RGB
    out[..., 3] = np.rint(a * 255.0).astype(np.uint8)
    return out


# Encoders tried one after the other until one works (the first is the best quality and the usual one). Some ffmpeg builds / machines
# make the ProRes encoder quit without a word, so the frames are kept on disk and a failure only costs the next try, not the render.
_ENCODERS = {
    "mov": [
        ("ProRes 4444", ["-c:v", "prores_ks", "-pix_fmt", "yuva444p10le", "-profile:v", "4444", "-vendor", "apl0", "-alpha_bits", "16"]),
        ("ProRes 4444 (8-bit alpha, one thread)", ["-c:v", "prores_ks", "-pix_fmt", "yuva444p10le", "-profile:v", "4444", "-alpha_bits", "8", "-threads", "1"]),
        ("QuickTime PNG (lossless, larger file)", ["-c:v", "png", "-pix_fmt", "rgba"]),
    ],
    "webm": [
        ("WebM VP9", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-crf", "18", "-b:v", "0"]),
        ("WebM VP9 (one thread)", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-crf", "24", "-b:v", "0", "-threads", "1", "-row-mt", "0"]),
    ],
}


class EncodeError(RuntimeError):
    pass


def _encode(frames: List[str], out: str, size: tuple, fps: int, args: List[str]) -> None:
    """Pipe the staged RGBA frames (PNG files) into ffmpeg. Raises EncodeError with the exit code and everything ffmpeg said."""
    import subprocess
    import imageio_ffmpeg
    w, h = size
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    log = tempfile.TemporaryFile()
    cmd = [exe, "-y", "-v", "warning", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", f"{fps:.2f}", "-i", "-", "-an", *args, out]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, creationflags=flags)
    err = None
    try:
        for fn in frames:
            bgra = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            p.stdin.write(np.ascontiguousarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)).tobytes())
    except OSError as exc:                                                                   # broken pipe: ffmpeg has quit
        err = exc
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass
    rc = p.wait()
    log.seek(0)
    text = log.read().decode("utf-8", "replace").strip()
    log.close()
    if err is not None or rc != 0:
        raise EncodeError(f"ffmpeg exited with code {rc & 0xFFFFFFFF:#x} ({err or 'error'}). " + (text[-600:] or "It printed no message."))


def encode_video(frames: List[str], path: str, fmt: str, fps: int) -> str:
    """Write the staged frames as a video with alpha; falls back to the next encoder if one fails. Returns a note ('' = first choice
    worked). If no encoder works the frames are saved as a PNG sequence next to `path` instead (the render is never lost)."""
    first = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
    size = (first.shape[1], first.shape[0])
    ext = os.path.splitext(path)[1] or "." + fmt
    problems: List[str] = []
    work = tempfile.mkdtemp(prefix="photovec_enc_")                                           # an ASCII path nothing else touches
    try:
        for label, args in _ENCODERS[fmt]:
            tmp_out = os.path.join(work, "clip" + ext)
            try:
                _encode(frames, tmp_out, size, fps, args)
                if os.path.exists(path):
                    os.remove(path)
                shutil.move(tmp_out, path)
                return "" if not problems else f"Saved with {label}: the better encoder failed on this computer ({problems[0]})"
            except (EncodeError, OSError) as exc:
                problems.append(f"{label}: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stem = os.path.splitext(path)[0]
    for i, fn in enumerate(frames):
        shutil.copyfile(fn, f"{stem}_{i + 1:05d}.png")
    return "The video encoder failed, so the frames were saved as a PNG sequence instead (name_00001.png ...). " + " | ".join(problems)


last_note = ""                 # set by export_transparent: what to tell the person when a fallback encoder had to be used


def export_transparent(specs, st, path: str, fmt: str, workers: int, pool=None,
                       progress: Optional[Callable[[str, int, int], bool]] = None) -> bool:
    """Render the clip of `specs` / `st` on black and on white and write it with an alpha channel. False = cancelled.
    progress(label, i, n) may return True to cancel."""
    from .multi import build_renderer
    global last_note
    last_note = ""
    tmp = tempfile.mkdtemp(prefix="photovec_alpha_")
    try:
        def cb(label):
            return (lambda i, n: bool(progress(label, i, n))) if progress else None
        st2 = replace(st, invert=False)
        black_files: List[str] = []
        total = 0
        threads = max(1, min(4, (os.cpu_count() or 2) - 1))
        for pass_i, col in enumerate(((0, 0, 0), (255, 255, 255))):
            tag = ("on black", "on white")[pass_i]
            r = build_renderer(specs, replace(st2, bg_color=col), workers, cb(f"Tracing the photos {tag}"), pool)
            if r is None or not r.ensure_sweeps(workers, pool, cb(f"Rendering the frames {tag}")):
                return False
            total = r.total
            n_done = 0
            stream: Iterator[np.ndarray] = r.frames(cb(f"Frames {tag}"), threads)
            if pass_i == 0:
                for i, f in enumerate(stream):
                    fn = os.path.join(tmp, f"b{i:06d}.png")
                    cv2.imwrite(fn, f, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    black_files.append(fn)
                if len(black_files) < total:
                    return False                                                        # cancelled
                continue
            stem = os.path.splitext(path)[0]
            staged: List[str] = []
            for i, f in enumerate(stream):
                rgba = matte(cv2.imread(black_files[i]), f)
                if fmt == "png":
                    cv2.imwrite(f"{stem}_{i + 1:05d}.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                else:
                    h, w = rgba.shape[:2]
                    rgba = rgba[: h // 2 * 2, : w // 2 * 2]
                    fn = os.path.join(tmp, f"m{i:06d}.png")
                    cv2.imwrite(fn, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    staged.append(fn)
                n_done = i + 1
            if n_done < total:
                return False                                                            # cancelled
            if fmt != "png":
                if progress and progress("Encoding the video", 0, 1):
                    return False
                last_note = encode_video(staged, path, fmt, st.fps)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
