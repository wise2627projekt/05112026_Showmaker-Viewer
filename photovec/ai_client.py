"""PhotoVec's side of the AI helper (see ai_worker.py): finds the helper's Python, starts it on demand and talks to it.

PhotoVec itself needs neither PyTorch nor any model. The AI lives in its own Python environment (made by install_ai.bat /
install_ai.sh) that this module finds by itself:
  1. the environment variable PHOTOVEC_AI_PYTHON,
  2. the path in  ~/.photovec/ai.json  ({"python": "...", "models": "..."}),
  3. the default environment  ~/.photovec/ai/venv,
  4. the Python that runs PhotoVec, if it has ultralytics installed (running from source).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional, Sequence

import cv2
import numpy as np

HOME = os.path.join(os.path.expanduser("~"), ".photovec")
CONFIG = os.path.join(HOME, "ai.json")


class AIError(RuntimeError):
    """The helper is missing, not ready or refused a request. The message is meant for the user."""


def _venv_python(root: str) -> str:
    return os.path.join(root, "Scripts", "python.exe") if os.name == "nt" else os.path.join(root, "bin", "python")


def find_python() -> Optional[str]:
    cand: List[str] = []
    if os.environ.get("PHOTOVEC_AI_PYTHON"):
        cand.append(os.environ["PHOTOVEC_AI_PYTHON"])
    try:
        with open(CONFIG, encoding="utf-8") as f:
            py = json.load(f).get("python")
            if py:
                cand.append(py)
    except (OSError, ValueError):
        pass
    cand.append(_venv_python(os.path.join(HOME, "ai", "venv")))
    if not getattr(sys, "frozen", False):
        try:
            import importlib.util
            if importlib.util.find_spec("ultralytics") is not None:
                cand.append(sys.executable)
        except Exception:
            pass
    for c in cand:
        if c and os.path.exists(c):
            return c
    return None


def models_dir() -> str:
    try:
        with open(CONFIG, encoding="utf-8") as f:
            m = json.load(f).get("models")
            if m:
                return m
    except (OSError, ValueError):
        pass
    return os.path.join(HOME, "ai", "models")


def worker_script() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "ai_worker.py"), os.path.join(getattr(sys, "_MEIPASS", here), "photovec", "ai_worker.py")):
        if os.path.exists(p):
            return p
    raise AIError("ai_worker.py was not found next to the program.")


class AIClient:
    """One helper process; requests are answered one after the other (a lock keeps two threads from mixing them up)."""

    def __init__(self, python: Optional[str] = None, models: Optional[str] = None, timeout: float = 600.0,
                 script: Optional[str] = None):
        self.python = python or find_python()
        self.models = models or models_dir()
        self.timeout = timeout
        self.script = script
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._next = 1
        self._tmp = tempfile.mkdtemp(prefix="photovec_ai_")
        self._image_key: Optional[tuple] = None
        self._image_path = ""
        self._status: Optional[dict] = None

    # ------------------------------------------------------------------ process
    def available(self) -> bool:
        return bool(self.python)

    def _start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        if not self.python:
            raise AIError("The AI helper is not installed. Run install_ai.bat (Windows) or install_ai.sh once - see the README.")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen([self.python, "-u", self.script or worker_script(), self.models], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL if os.environ.get("PHOTOVEC_AI_QUIET") else None,
                                         text=True, bufsize=1, creationflags=flags)
        except OSError as exc:
            raise AIError(f"Could not start the AI helper ({self.python}): {exc}")
        hello = self._readline()
        if not hello or not hello.get("ok"):
            self.close()
            raise AIError("The AI helper did not start. Is PyTorch / ultralytics installed in " + self.python + "?")

    def _readline(self) -> Optional[dict]:
        box: List[Optional[str]] = []

        def read():
            try:
                box.append(self.proc.stdout.readline())
            except Exception:
                box.append("")
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            self.close()
            raise AIError("The AI helper did not answer in time (the first request loads the model - give it a minute, then try again).")
        line = box[0] if box else ""
        if not line:
            return None
        try:
            return json.loads(line)
        except ValueError:
            return None

    def _call(self, cmd: str, **kw) -> dict:
        with self._lock:
            self._start()
            rid = self._next
            self._next += 1
            try:
                self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd, **kw}) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                self.close()
                raise AIError("The AI helper stopped unexpectedly. Try again.")
            while True:
                ans = self._readline()
                if ans is None:
                    self.close()
                    raise AIError("The AI helper stopped unexpectedly (out of graphics memory?). Try again, or a smaller picture.")
                if ans.get("id") == rid:
                    break
            if not ans.get("ok"):
                raise AIError(str(ans.get("error", "unknown error")))
            return ans["result"]

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is not None:
            try:
                if p.poll() is None:
                    p.stdin.write(json.dumps({"id": 0, "cmd": "quit"}) + "\n")
                    p.stdin.flush()
                    p.wait(timeout=3)
            except Exception:
                pass
            try:
                p.kill()
            except Exception:
                pass
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = tempfile.mkdtemp(prefix="photovec_ai_")
        self._image_key = None

    # ------------------------------------------------------------------ requests
    def status(self, refresh: bool = False) -> dict:
        if self._status is None or refresh:
            self._status = self._call("status")
        return self._status

    def _put_image(self, bgr: np.ndarray, key: object) -> str:
        k = (id(bgr), bgr.shape, key)
        if self._image_key != k:
            self._image_path = os.path.join(self._tmp, f"img_{abs(hash(k))}.png")
            cv2.imwrite(self._image_path, bgr)
            self._image_key = k
        return self._image_path

    def _read_masks(self, res: dict, shape) -> List[dict]:
        out = []
        for m in res.get("masks", []):
            a = cv2.imread(m["path"], cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            if a.shape[:2] != tuple(shape[:2]):
                a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            out.append({"mask": a, "score": m.get("score", 1.0), "box": m.get("box")})
        return out

    def find(self, bgr: np.ndarray, prompts: Sequence[str] | str, engine: str = "auto", conf: float = 0.25, key: object = None) -> dict:
        """Everything that matches the words (e.g. "graffiti"): {"engine": ..., "masks": [{"mask", "score", "box"}, ...]}."""
        if isinstance(prompts, str):
            prompts = [p.strip() for p in prompts.split(",") if p.strip()]
        if not prompts:
            raise AIError("Type what to look for, e.g. graffiti.")
        path = self._put_image(bgr, key)
        res = self._call("text", image=path, prompts=list(prompts), engine=engine, conf=conf, folder=self._tmp)
        return {"engine": res.get("engine", engine), "masks": self._read_masks(res, bgr.shape)}

    def segment(self, bgr: np.ndarray, points: Sequence[Sequence[float]] = (), labels: Sequence[int] = (),
                box: Optional[Sequence[float]] = None, model: str = "", key: object = None) -> Optional[np.ndarray]:
        """One precise mask from clicks (label 1 = on it, 0 = not it) and/or a box; None if the model found nothing."""
        path = self._put_image(bgr, key)
        res = self._call("points", image=path, points=[list(p) for p in points], labels=list(labels),
                         box=list(box) if box else None, folder=self._tmp, model=model)
        found = self._read_masks(res, bgr.shape)
        return found[0]["mask"] if found else None
