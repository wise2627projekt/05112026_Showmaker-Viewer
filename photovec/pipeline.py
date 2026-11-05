"""General-mode pipeline: load -> preprocess -> foliage -> extract -> fuse -> vectorise."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from . import extract as ex
from .config import DETAILS, PROFILES, RunConfig
from .geometry import Chain, Path, dedup_groups, join_chains, vectorise_chain
from .imageio_ import load_image
from .scheduler import Cancelled, Scheduler

log = logging.getLogger("photovec")


@dataclass
class Result:
    paths: List[Path]
    size: tuple                    # (width, height) in working pixels
    image_bgr: np.ndarray          # working-resolution photo (for overlays)
    foliage_mask: np.ndarray
    timings: Dict[str, float]
    fallbacks: List[str]
    line_aware: bool
    notes: List[str] = field(default_factory=list)


def _wait(fut, sched: Scheduler, stage: str, label: str, optional: bool):
    """Collect a future; optional ones are abandoned if the stage allowance is blown."""
    if not sched.limited or not optional:
        return fut.result()
    try:
        return fut.result(timeout=max(0.3, sched.time_left(stage) * 1.25))
    except FutTimeout:
        sched.note_fallback(f"{label} skipped (over time)")
        return []


def run_pipeline(path: str, cfg: RunConfig, sched: Optional[Scheduler] = None) -> Result:
    cfg.validate()
    prof, det = PROFILES[cfg.budget], DETAILS[cfg.detail]
    np.random.seed(cfg.seed)
    cv2.setRNGSeed(cfg.seed)
    sched = sched or Scheduler(prof)
    sched.start()
    notes: List[str] = []
    try:
        with sched.stage("load"):
            rgb = load_image(path)
        with sched.stage("preprocess"):
            h0, w0 = rgb.shape[:2]
            long_side = prof.work_long_side
            if sched.behind("preprocess"):
                long_side = int(long_side * 0.75)
                sched.note_fallback("resolution reduced")
            scale = min(1.0, long_side / max(h0, w0))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if scale < 1.0:
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape
            k = max(H, W) / 1280.0                      # px thresholds are defined at 1280
            min_chain, lsd_min = det.min_chain_len * k, det.lsd_min_len * k
            img_area = float(H * W)
        with sched.stage("foliage"):
            fol = np.zeros((H, W), bool) if cfg.foliage == "keep_raw" else ex.foliage_mask(bgr)
        sched.checkpoint()

        with sched.stage("extract"):
            sched.checkpoint()
            skip_opt = sched.behind("extract", 1.0)
            with ThreadPoolExecutor(max_workers=4) as pool:
                f_lsd = pool.submit(ex.detect_lines, gray, lsd_min, fol if fol.any() else None)
                f_edge = pool.submit(ex.edge_chains, gray, prof.edge_scales, det.edge_pct,
                                     det.smooth_std, min_chain)
                f_ton = None if skip_opt else pool.submit(
                    ex.tonal_chains, gray, prof.tonal_levels, 1.6, det.tonal_min_area * img_area)
                f_rid = None if skip_opt else pool.submit(
                    ex.ridge_chains, gray, prof.ridge_sigmas, det.ridge_pct, min_chain,
                    det.smooth_std, prof.ridge_bright_too)
                f_fol = None
                if fol.any() and cfg.foliage in ("stylized", "simplify"):
                    f_fol = pool.submit(ex.tonal_chains, gray, prof.foliage_levels, 3.0,
                                        6 * det.tonal_min_area * img_area, 1.0, fol, "FOLIAGE")
                segs = f_lsd.result()
                edges = f_edge.result()
                tonal = _wait(f_ton, sched, "extract", "tonal contours", True) if f_ton else []
                ridges = _wait(f_rid, sched, "extract", "centre lines", True) if f_rid else []
                fol_inner = f_fol.result() if f_fol else []
            if skip_opt:
                sched.note_fallback("tonal contours and centre lines skipped")

        with sched.stage("fuse"):
            strength = ex.line_structure_strength(segs, max(H, W))
            line_aware = cfg.line_aware if cfg.line_aware is not None else strength >= 2.0
            notes.append(f"line structure strength {strength:.2f} -> line-aware "
                         f"{'ON' if line_aware else 'OFF'}")
            straight = [Chain(np.array([[s[0], s[1]], [s[2], s[3]]]), False, "STRUCTURE", True)
                        for s in segs] if line_aware else []
            silhouette: List[Chain] = []
            if fol.any() and cfg.foliage in ("stylized", "simplify"):
                silhouette = ex.mask_contours(fol, 4e-4 * img_area)
            elif cfg.foliage == "remove" and fol.any():
                silhouette = []

            def fol_frac(c: Chain) -> float:
                return ex.fraction_inside(c.pts, fol) if fol.any() else 0.0

            def treat(chains: List[Chain], keep_thin: bool = False) -> List[Chain]:
                out = []
                for c in chains:
                    f = fol_frac(c)
                    if f <= 0.5 or cfg.foliage == "keep_raw":
                        out.append(c)
                    elif cfg.foliage == "stylized" and keep_thin and c.length() >= 2.0 * min_chain:
                        c.layer = "FOLIAGE"     # twig / branch centre lines
                        out.append(c)
                return out

            ridges_t = treat(ridges, keep_thin=True)
            if cfg.foliage == "simplify":
                ridges_t = [c for c in ridges_t if c.layer != "FOLIAGE"]
            edges_t, tonal_t = treat(edges), treat(tonal)
            if fol.any() and cfg.foliage in ("stylized", "simplify"):
                silhouette += fol_inner
            groups = [(straight, 2), (silhouette, 2), (ridges_t, 3), (edges_t, 2), (tonal_t, 2)]
            fused = dedup_groups(groups, (H, W), min_chain)
            fused = join_chains(fused, gap=4.0 * k)
            fused = [c for c in fused if c.fixed or c.length() >= min_chain]
            # recompute foliage membership for the layer of non-fixed chains
            for c in fused:
                if not c.fixed and c.layer in ("EDGES", "TONAL") and fol_frac(c) > 0.5:
                    c.layer = "FOLIAGE"
                elif c.layer == "EDGES" and not c.fixed:
                    c.layer = "EDGES"

        with sched.stage("vectorise"):
            mode = "bezier" if cfg.budget == "unlimited" else "fast"
            if sched.behind("vectorise", 1.0) and len(fused) > 4000:
                mode = "polyline"
                sched.note_fallback("curve fitting simplified")
            paths: List[Path] = []
            for i, c in enumerate(fused):
                if mode != "polyline" and sched.expired:
                    mode = "polyline"
                    sched.note_fallback("curve fitting simplified (watchdog)")
                if i % 256 == 0:
                    sched.checkpoint()
                p = vectorise_chain(c, tol=prof.curve_tol * cfg.simplify,
                                    line_tol=1.2 * cfg.simplify, mode=mode)
                if p is not None:
                    paths.append(p)
        return Result(paths, (W, H), bgr, fol, dict(sched.timings), list(sched.fallbacks),
                      bool(line_aware), notes)
    except Cancelled:
        raise
    finally:
        sched.stop()
