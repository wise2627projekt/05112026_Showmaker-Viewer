"""Prerender slider frames in the worker processes (the preview windows used to do this in threads).

The tracer is mostly Python code, so threads wait for each other (the GIL): three threads were hardly faster than one. Here the
frames of a picture are handed to the shared process pool (photovec.pool) in small chunks and come back as they are finished:
every frame is the vector picture, how much of each pixel is drawing (0..1, or None) and the paper colour - the same three
things `render_values(..., parts=True)` returns - and the window lays the layers over them itself.

If no pool is available (one-core machine, inside a worker) `Prerender.pool()` is None and the caller keeps using threads; if a
chunk fails (a worker died) `on_error` is called with its tags and the caller renders those frames itself.
"""
from __future__ import annotations

import os
import tempfile
import threading
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import pool as pool_mod
from .sweep import _TEMP_FILES, SweepSource, _load_source, render_values

CHUNK = 2                      # frames per job: small, so results appear early and a cancel does not wait long


def _parts_chunk(args):
    """Worker process: a few frames of one picture -> [(vector picture, ink 0..1 float32 or None, paper colour)]."""
    path, params, linework, bg, vlist = args
    src = _load_source(path, params, linework)
    cache: dict = {}
    out = []
    for v in vlist:
        vec, ink, paper = render_values(src, bg, dict(v), cache, parts=True)
        out.append((vec, None if ink is None else np.ascontiguousarray(ink, np.float32), paper))
    return out


MIN_PROCESSES = 3              # with fewer cores the threads do as well (the main program needs a core of its own for the layers)


def pool_for_previews():
    """The shared process pool when it pays off (a few worker processes), else None (the windows then render in threads)."""
    from .trace import get_pool
    pool = get_pool()
    return pool if pool is not None and getattr(pool, "_max_workers", 0) >= MIN_PROCESSES else None


class Prerender:
    """The frames of ONE picture (with its style already in `src.params`) rendered by the worker processes."""

    def __init__(self, src: SweepSource, bg: str, pool=None):
        self.src, self.bg = src, bg
        self.pool = pool if pool is not None else pool_for_previews()
        self.path: Optional[str] = None
        self._futs: list = []
        self._lock = threading.Lock()
        self._pending = 0
        self._busy = None
        self._closed = False

    @property
    def usable(self) -> bool:
        return self.pool is not None and not self._closed

    def _file(self) -> str:
        if self.path is None:
            fd, self.path = tempfile.mkstemp(prefix="photovec_prev_", suffix=".npz")
            os.close(fd)
            _TEMP_FILES.add(self.path)
            arrs = {"bgr": self.src.bgr}
            if self.src.opaque is not None:
                arrs["opaque"] = self.src.opaque
            if self.src.mask is not None:
                arrs["mask"] = self.src.mask
            np.savez(self.path, **arrs)
        return self.path

    def submit(self, items: Sequence[Tuple[object, dict]], on_frame: Callable, on_error: Callable[[List[object]], None], chunk: int = CHUNK) -> bool:
        """items: (tag, slider values). on_frame(tag, vec, ink, paper) is called from a pool thread for every frame, in no fixed order.
        False = could not start (the caller renders them itself)."""
        if not self.usable or not items:
            return False
        try:
            path = self._file()
            jobs = [list(items[i:i + chunk]) for i in range(0, len(items), chunk)]
            with self._lock:
                if self._busy is None:
                    self._busy = pool_mod.busy()
                    self._busy.__enter__()
                self._pending += len(jobs)
            for part in jobs:
                fut = self.pool.submit(_parts_chunk, (path, self.src.params, self.src.linework, self.bg, [dict(v) for _, v in part]))
                self._futs.append(fut)
                fut.add_done_callback(lambda f, part=part: self._done(f, part, on_frame, on_error))
            return True
        except Exception:
            self._release(len(jobs) if "jobs" in locals() else 0)
            return False

    def _done(self, fut, part, on_frame, on_error) -> None:
        try:
            if fut.cancelled() or self._closed:
                return
            try:
                res = fut.result()
            except BaseException:
                on_error([t for t, _ in part])
                return
            for (tag, _), (vec, ink, paper) in zip(part, res):
                on_frame(tag, vec, ink, paper)
        finally:
            self._release(1)

    def _release(self, n: int) -> None:
        with self._lock:
            self._pending = max(0, self._pending - n)
            if self._pending == 0 and self._busy is not None:
                busy, self._busy = self._busy, None
                busy.__exit__(None, None, None)

    def cancel(self) -> None:
        for f in self._futs:
            f.cancel()
        self._futs = [f for f in self._futs if not f.done()]

    def close(self) -> None:
        """Stop what has not started and remove the picture's temporary file."""
        self._closed = True
        self.cancel()
        if self.path is not None:
            try:
                os.remove(self.path)
            except OSError:
                pass
            _TEMP_FILES.discard(self.path)
            self.path = None
