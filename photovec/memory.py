"""Memory budget: how many worker processes fit into RAM, and a guard that waits instead of crashing.

The grid features run several Python processes (every one imports numpy / OpenCV / SciPy / scikit-image, ~0.4-0.6 GB
each) and render thousands of slider frames. The defaults here are tuned for a 16 GB machine: workers are limited by
free memory (not only by CPU cores), frames of long sweeps go to a temporary file on disk instead of RAM, and big jobs
pause when free memory gets low.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional

GB = 1024 ** 3
WORKER_COST = 0.9 * GB            # one worker process incl. its caches (0.25 GB measured on Linux; frozen Windows builds are larger)
MAIN_RESERVE = 2.0 * GB           # GUI process, Qt, preview frames, tile results
MAX_AUTO_WORKERS = 8              # lowered further on small machines, see auto_workers()

_budget_gb: float = 0.0           # 0 = automatic


def total_ram() -> int:
    """Installed RAM in bytes (0 when unknown)."""
    try:
        import psutil                                    # optional
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if sys.platform.startswith("win"):
            return _win_status()[0]
        if sys.platform == "darwin":
            return int(_run(["sysctl", "-n", "hw.memsize"]).strip())
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 0


def available_ram() -> int:
    """Memory that can be used right now without swapping, in bytes (0 when unknown)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        if sys.platform.startswith("win"):
            return _win_status()[1]
        if sys.platform == "darwin":
            return parse_vm_stat(_run(["vm_stat"]))
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _run(cmd) -> str:
    import subprocess
    return subprocess.check_output(cmd, text=True, timeout=5)


def parse_vm_stat(text: str) -> int:
    """Memory macOS can hand out without swapping, from the output of `vm_stat` (free + inactive + speculative pages), in bytes."""
    import re
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 4096
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(key + r":\s+(\d+)", text)
        if m:
            pages += int(m.group(1))
    return pages * page


def _win_status():
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullTotalPhys), int(m.ullAvailPhys)


def set_budget_gb(gb: float) -> None:
    """Memory the program may use in total (0 = automatic: 60 % of the installed RAM)."""
    global _budget_gb
    _budget_gb = max(0.0, float(gb))


def budget_bytes() -> int:
    if _budget_gb > 0:
        return int(_budget_gb * GB)
    tot = total_ram()
    return int(tot * 0.6) if tot else 8 * GB


def _usable_bytes() -> int:
    """What the workers may use: the budget, but never more than 70 % of what is free right now (other programs count)."""
    b = budget_bytes()
    a = available_ram()
    return min(b, int(a * 0.7)) if a else b


def _fit_by_memory() -> int:
    return max(1, int((_usable_bytes() - MAIN_RESERVE) // WORKER_COST))


def auto_workers(cpus: Optional[int] = None) -> int:
    """Worker processes for the automatic setting: CPU cores - 1, limited by free memory and by the size of the machine
    (16 GB -> at most 4; 32 GB -> 8)."""
    cpus = cpus or os.cpu_count() or 1
    tot = total_ram()
    cap = max(2, min(MAX_AUTO_WORKERS, int(tot // (4 * GB)))) if tot else 4
    return max(1, min(max(1, cpus - 1), _fit_by_memory(), cap))


def limit_workers(n: int) -> int:
    """A user-chosen worker count, reduced when it would not fit into the memory budget / the free memory."""
    return max(1, min(n, _fit_by_memory()))


def process_rss() -> int:
    """Resident memory of this process (bytes, 0 when unknown)."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def is_low(reserve_gb: float = 1.2) -> bool:
    """True when less than `reserve_gb` is free (unknown = never low)."""
    a = available_ram()
    return bool(a) and a < reserve_gb * GB


def wait_until_free(reserve_gb: float = 1.2, max_wait: float = 60.0,
                    cancelled: Optional[Callable[[], bool]] = None) -> bool:
    """Pause while memory is short. True = enough memory again, False = still short after `max_wait` seconds."""
    t0 = time.time()
    while is_low(reserve_gb):
        if cancelled is not None and cancelled():
            return True
        if time.time() - t0 > max_wait:
            return False
        time.sleep(0.5)
    return True


def describe() -> str:
    tot, av = total_ram(), available_ram()
    if not tot:
        return "memory: unknown"
    return (f"RAM {tot / GB:.0f} GB, {av / GB:.1f} GB free · budget {budget_bytes() / GB:.0f} GB · "
            f"{auto_workers()} worker processes")
