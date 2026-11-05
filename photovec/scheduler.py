"""Time-budget scheduler: per-stage clocks, fallbacks, watchdog, cancellation."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

from .config import STAGES, BudgetProfile


class Cancelled(Exception):
    """Raised at a checkpoint after the user pressed Cancel."""


class Scheduler:
    def __init__(self, profile: BudgetProfile,
                 progress: Optional[Callable[[str, float], None]] = None):
        self.profile = profile
        self.limit = profile.limit_s
        self.fractions = dict(profile.stage_fractions)
        self.timings: Dict[str, float] = {}
        self.fallbacks: List[str] = []
        self.progress = progress
        self._t0 = time.perf_counter()
        self._cancel = threading.Event()
        self._expired = threading.Event()
        self._timer: Optional[threading.Timer] = None
        self._starts: Dict[str, float] = {}
        acc = 0.0
        for s in STAGES:
            self._starts[s] = acc
            acc += self.fractions.get(s, 0.0)

    # --- lifecycle -------------------------------------------------------
    @property
    def limited(self) -> bool:
        return self.limit is not None

    def start(self) -> None:
        self._t0 = time.perf_counter()
        if self.limited:
            reserve = self.fractions.get("export", 0.1) * self.limit
            self._timer = threading.Timer(max(0.0, self.limit - reserve), self._expired.set)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()

    def cancel(self) -> None:
        self._cancel.set()

    # --- queries ----------------------------------------------------------
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def expired(self) -> bool:
        """Watchdog fired: only the cheapest path to a valid result is allowed."""
        return self._expired.is_set()

    def checkpoint(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def planned_start(self, stage: str) -> float:
        return self._starts[stage] * self.limit if self.limited else float("inf")

    def planned_end(self, stage: str) -> float:
        if not self.limited:
            return float("inf")
        return (self._starts[stage] + self.fractions.get(stage, 0.0)) * self.limit

    def behind(self, stage: str, slack: float = 1.15) -> bool:
        """True if we are later than planned when `stage` begins."""
        return self.limited and (self.expired or self.elapsed() > self.planned_start(stage) * slack)

    def time_left(self, stage: str) -> float:
        return max(0.0, self.planned_end(stage) - self.elapsed()) if self.limited else float("inf")

    # --- bookkeeping ------------------------------------------------------
    def note_fallback(self, msg: str) -> None:
        if msg not in self.fallbacks:
            self.fallbacks.append(msg)

    @contextmanager
    def stage(self, name: str):
        self.checkpoint()
        if self.progress:
            self.progress(name, self.elapsed())
        t = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t
