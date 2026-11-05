"""Multi-Trace project files: everything needed to continue a clip later, in ONE file.

A project (`.photovec`) is a zip archive holding `project.json` (all settings of the window and of every photo) and the
photos themselves, so the project still opens after the original pictures were moved or deleted. A picture that is used in
several panels is stored once."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

FORMAT = 1
APP = "photovec-multi-trace"
EXT = ".photovec"
MAX_JSON = 8 * 1024 * 1024
MAX_MASK = 64 * 1024 * 1024     # a mask is a PNG of one grey channel: far smaller than this


class ProjectError(Exception):
    pass


def _safe_name(name: str) -> str:
    n = re.sub(r"[^\w.\- ()]+", "_", os.path.basename(name.replace("\\", "/"))).strip(" .")
    return n or "picture"


def write_project(path: str, state: dict, panel_paths: Sequence[str], masks: Optional[Sequence[Optional[np.ndarray]]] = None,
                  app: str = APP) -> None:
    """Write `state` (the dict of the window) plus the picture of every panel (`panel_paths[i]` = file or "") to `path`.
    The file is written completely first and moved into place afterwards, so a failed save never damages an older project."""
    state = json.loads(json.dumps(state))                    # a private copy that is guaranteed to be plain JSON
    state.update(format=FORMAT, app=app, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = path + ".part"
    stored: Dict[str, str] = {}                              # content hash -> name inside the archive
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            panels = state.get("panels", [])
            for i, src in enumerate(panel_paths):
                panel = panels[i]
                panel["image"] = None
                if not src:
                    continue
                if not os.path.isfile(src):
                    raise ProjectError(f"Photo {i + 1} cannot be read any more: {src}")
                with open(src, "rb") as f:
                    data = f.read()
                h = hashlib.blake2b(data, digest_size=12).hexdigest()
                if h not in stored:
                    stored[h] = f"pictures/{len(stored) + 1:02d}_{_safe_name(src)}"
                    z.writestr(stored[h], data)
                panel["image"] = stored[h]
                panel["name"] = os.path.basename(src)
            for i, m in enumerate(masks or []):                  # layer masks: one grey PNG per photo that has one
                if i < len(panels):
                    panels[i]["mask"] = None
                    if m is not None:
                        ok, buf = cv2.imencode(".png", np.ascontiguousarray(m, dtype=np.uint8))
                        if ok:
                            panels[i]["mask"] = f"masks/{i + 1:02d}.png"
                            z.writestr(panels[i]["mask"], buf.tobytes())
            z.writestr("project.json", json.dumps(state, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_project(path: str, dest_dir: str, app: str = APP) -> Tuple[dict, List[str]]:
    """Open a project. The pictures are unpacked into `dest_dir` (one folder per panel, original file names) and the state
    is returned together with the file of every panel ("" = empty panel)."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProjectError(f"This is not a Multi-Trace project file: {exc}") from exc
    with z:
        try:
            info = z.getinfo("project.json")
        except KeyError:
            raise ProjectError("This is not a Multi-Trace project file (project.json is missing).")
        if info.file_size > MAX_JSON:
            raise ProjectError("The project file is damaged (settings too large).")
        try:
            state = json.loads(z.read(info))
        except ValueError as exc:
            raise ProjectError(f"The project file is damaged: {exc}") from exc
        if not isinstance(state, dict) or state.get("app") != app:
            raise ProjectError("This is not a Multi-Trace project file.")
        if int(state.get("format", 0)) > FORMAT:
            raise ProjectError("This project was saved by a newer version of the program.")
        out: List[str] = []
        done: Dict[str, str] = {}
        for i, panel in enumerate(state.get("panels", [])):
            member = panel.get("image")
            if not member:
                out.append("")
                continue
            if member not in done:
                if member not in z.namelist():
                    raise ProjectError(f"Photo {i + 1} is missing in the project file.")
                folder = os.path.join(dest_dir, f"photo{i + 1:02d}")
                os.makedirs(folder, exist_ok=True)
                target = os.path.join(folder, _safe_name(panel.get("name") or member))     # never a path from the file
                with z.open(member) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                done[member] = target
            out.append(done[member])
    return state, out


def read_masks(path: str, state: dict) -> List[Optional[np.ndarray]]:
    """The layer masks of a project (uint8 0..255, one per panel, None = no mask). Older projects have none."""
    out: List[Optional[np.ndarray]] = []
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProjectError(f"This is not a Multi-Trace project file: {exc}") from exc
    with z:
        names = set(z.namelist())
        for i, panel in enumerate(state.get("panels", [])):
            member = panel.get("mask")
            if not member or member not in names or not member.startswith("masks/"):
                out.append(None)
                continue
            info = z.getinfo(member)
            if info.file_size > MAX_MASK:
                raise ProjectError(f"The mask of photo {i + 1} is damaged (too large).")
            m = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None or max(m.shape) > 16384:
                raise ProjectError(f"The mask of photo {i + 1} is damaged.")
            out.append(m)
    return out
