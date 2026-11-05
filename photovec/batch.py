"""Trace many pictures at once (grid tiles). Process pool when several cores are available."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, List, Optional, Sequence

from . import memory, pool
from .clip import fit_tile
from .morph import MorphOptions, build_plan, fit_linework, items_from_result
from .imageio_ import load_image_alpha
from .linework import LineworkParams, linework_bgr
from .rows import RowSpec, params_for_cell
from .trace import TraceParams, trace_bgr


def resolve_workers(workers: int) -> int:
    """Worker processes: the user's number (reduced if the memory budget / free memory cannot hold that many) or the
    automatic choice (CPU cores - 1, limited by free memory and machine size)."""
    return memory.limit_workers(workers) if workers > 0 else memory.auto_workers()


def shared_pool(workers: int):
    """The shared process pool (see photovec.pool): created on first use, shut down when idle."""
    return pool.get(resolve_workers(workers))


def shutdown_pool() -> None:
    pool.shutdown()


def tile_params(params, tw: int, th: int):
    """Pixel-based settings are tuned for ~1000 px images; shrink them for small tiles."""
    s = min(max(1.0, getattr(params, "scale_cap", 1.0)), max(0.25, max(tw, th) / 1000.0))
    if isinstance(params, LineworkParams):
        return replace(params, scale=s, max_side=0)
    return replace(params, scale=s, max_side=0)


def trace_tile(args):
    """Worker: load one file, fit it into the tile and trace it. Top-level so it can be pickled."""
    path, params, tw, th, fit = args
    rgb, alpha = load_image_alpha(path, (tw, th))
    bgr, opaque = fit_tile(rgb, alpha, tw, th, fit)
    p = tile_params(params, tw, th)
    res = linework_bgr(bgr, p, opaque) if isinstance(p, LineworkParams) else trace_bgr(bgr, p, opaque)
    items_from_result(res)              # polyline items for the morph, computed here (in parallel) instead of in the GUI process
    return res


def trace_jobs(jobs: Sequence[tuple], workers: int = 0,
               progress: Optional[Callable[[int, int], bool]] = None) -> List:
    """jobs = (path, params, tile_w, tile_h, fit) each – params may differ from job to job (per-row settings).
    `progress(done, total)` may return True to cancel (returns what is done). If memory runs short the work continues
    with fewer processes instead of failing."""
    n = len(jobs)
    workers = resolve_workers(workers)
    # biggest tiles first keeps all cores busy to the end; results are returned in the original order
    order = sorted(range(n), key=lambda i: -(jobs[i][2] * jobs[i][3]))
    res, cancelled = pool.map_jobs(trace_tile, jobs, 1 if n == 1 else workers, progress, order)
    return [r for r in res if r is not None] if cancelled else res


def trace_many(paths: Sequence[str], params, tw: int, th: int, fit: str = "cover", workers: int = 0,
               progress: Optional[Callable[[int, int], bool]] = None) -> List:
    """Same settings for every picture."""
    return trace_jobs([(p, params, tw, th, fit) for p in paths], workers, progress)


def grid_jobs(rows: Sequence[RowSpec], base_params) -> List[tuple]:
    """One trace job (path, params, tile_w, tile_h, fit) per square, in picture order."""
    jobs = []
    for r in rows:
        r.sync()
        for i, path in enumerate(r.paths):
            tw, th = r.tile_size(i)
            jobs.append((path, params_for_cell(r, i, base_params), tw, th, r.tile_fit(i)))
    return jobs


def job_key(job: tuple) -> tuple:
    """Identity of a trace job (for re-using results when only other squares changed)."""
    path, params, tw, th, fit = job
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = 0.0
    return (path, mt, repr(params), tw, th, fit)


def trace_rows(rows: Sequence[RowSpec], base_params, workers: int = 0,
               progress: Optional[Callable[[int, int], bool]] = None) -> List:
    """Trace every picture of every row with its own settings. Result order = row by row."""
    return trace_jobs(grid_jobs(rows, base_params), workers, progress)


def _plan_one(args):
    size, items, lines, opts = args
    return build_plan(items, lines, size, opts)


def plan_many(results: Sequence, lineworks: Sequence, opts: MorphOptions, workers: int = 0,
              progress: Optional[Callable[[int, int], bool]] = None) -> List:
    """Morph plans for every tile (None where a tile has no linework). Runs in parallel."""
    jobs = []
    for res, lw in zip(results, lineworks):
        if lw is None:
            jobs.append(None)
            continue
        W, H = res.size
        jobs.append(((W, H), items_from_result(res), fit_linework(lw, W, H, opts), opts))
    n = len(jobs)
    todo = [i for i, j in enumerate(jobs) if j is not None]
    out: List = [None] * n
    if not todo:
        return out
    res, _ = pool.map_jobs(_plan_one, jobs, resolve_workers(workers), progress, todo)
    return res
