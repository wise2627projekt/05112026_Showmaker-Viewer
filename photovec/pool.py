"""The ONE process pool of the program, with the safety nets a 16 GB machine needs.

* a single pool for everything (grid tracing, morph matching, slider frames, ring fitting of big traces) – before, the
  ring fitter and the grid code each started up to 8 processes of their own;
* workers are limited by memory, run with low priority and one thread each (no BLAS / OpenCV thread explosion);
* an idle pool is shut down after a while, so a finished job does not keep gigabytes reserved;
* when a worker dies (Windows kills processes when memory runs out) the work is re-run with half as many processes,
  down to running in the calling process – a build slows down instead of failing.
"""
from __future__ import annotations

import atexit
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Callable, List, Optional, Sequence

from . import memory

IDLE_SECONDS = 120.0

_POOL: Optional[ProcessPoolExecutor] = None
_N = 0
_LOCK = threading.RLock()
_BUSY = 0
_LAST_USE = time.time()
_TIMER: Optional[threading.Timer] = None
_NOTICES: List[str] = []


def init_worker() -> None:
    """Runs once in every worker process: one thread, low priority."""
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)   # below normal
        else:
            os.nice(5)
    except Exception:
        pass


def _single_thread_env() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"          # inherited by the worker processes that are started from now on


def get(workers: int) -> ProcessPoolExecutor:
    """The shared pool with at least `workers` processes (an existing larger pool is reused)."""
    global _POOL, _N, _LAST_USE
    workers = max(1, workers)
    with _LOCK:
        _LAST_USE = time.time()
        if _POOL is not None and (getattr(_POOL, "_broken", False) or _N < workers):
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None
        if _POOL is None:
            _single_thread_env()
            kw = {"max_tasks_per_child": 400} if sys.version_info >= (3, 11) else {}     # fresh processes now and then
            _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=init_worker, **kw)
            _N = workers
        _schedule_idle_check()
        return _POOL


def size() -> int:
    return _N if _POOL is not None else 0


def shutdown() -> None:
    global _POOL, _N
    with _LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL, _N = None, 0


atexit.register(shutdown)


@contextmanager
def busy():
    """Mark the pool as in use (it is not shut down while a job runs)."""
    global _BUSY, _LAST_USE
    with _LOCK:
        _BUSY += 1
    try:
        yield
    finally:
        with _LOCK:
            _BUSY -= 1
            _LAST_USE = time.time()
            _schedule_idle_check()


def _schedule_idle_check() -> None:
    global _TIMER
    if _TIMER is not None:
        _TIMER.cancel()
    _TIMER = threading.Timer(IDLE_SECONDS + 1.0, _idle_check)
    _TIMER.daemon = True
    _TIMER.start()


def _idle_check() -> None:
    with _LOCK:
        if _POOL is not None and _BUSY == 0 and time.time() - _LAST_USE >= IDLE_SECONDS:
            shutdown()


def notice(msg: str) -> None:
    _NOTICES.append(msg)


def take_notices() -> List[str]:
    out = list(_NOTICES)
    _NOTICES.clear()
    return out


def map_jobs(fn: Callable, jobs: Sequence, workers: int, progress: Optional[Callable[[int, int], bool]] = None,
             order: Optional[Sequence[int]] = None, sink: Optional[Callable[[int, object], None]] = None):
    """Run fn(job) for every job on the shared pool. Returns (results, cancelled); results are in job order
    (None where cancelled, or everywhere when a `sink(index, result)` takes the results instead).
    A dying worker (out of memory) halves the number of processes and re-runs what was not finished; with one process left
    the jobs run in the calling process. `progress(done, total)` may return True to cancel."""
    n = len(jobs)
    res: list = [None] * n
    done = [False] * n
    todo = list(order) if order is not None else list(range(n))
    ndone = 0
    w = max(1, workers)

    def put(i, r):
        if sink is not None:
            sink(i, r)
        else:
            res[i] = r
        done[i] = True

    with busy():
        while todo:
            if w <= 1 or len(todo) == 1:
                for i in todo:
                    if progress and progress(ndone, n):
                        return res, True
                    put(i, fn(jobs[i]))
                    ndone += 1
                break
            ex = get(w)
            queue, inflight = list(todo), {}
            try:
                while queue or inflight:
                    while queue and len(inflight) < w * 2:
                        if inflight and memory.is_low():
                            break                                  # let running jobs finish first, that frees memory
                        i = queue.pop(0)
                        inflight[ex.submit(fn, jobs[i])] = i
                    finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                    for f in finished:
                        i = inflight.pop(f)
                        put(i, f.result())
                        ndone += 1
                    if progress and progress(ndone, n):
                        for f in inflight:
                            f.cancel()
                        return res, True
                break
            except (BrokenProcessPool, MemoryError):
                for f in inflight:
                    f.cancel()
                todo = [i for i in todo if not done[i]]
                w = max(1, w // 2)
                shutdown()
                notice(f"Memory ran short and a worker process was stopped by the system – continuing with "
                       f"{w} process{'es' if w > 1 else ''}" + ("" if w > 1 else " (in the program itself, slower)") + ".")
                memory.wait_until_free(1.5, 20.0)
    return res, False
