"""Command line: python -m photovec input output --mode auto --budget 30 ..."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List

from .config import PROFILES, RunConfig
from .export import export_all, path_stats
from .pipeline import run_pipeline

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif")


def _inputs(p: str) -> List[str]:
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        raise SystemExit(f"error: input '{p}' is neither a file nor a folder")
    files = sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(EXTS))
    if not files:
        raise SystemExit(f"error: no images ({', '.join(EXTS)}) in '{p}'")
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="photovec", description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", default="auto", choices=["facade", "general", "auto"])
    ap.add_argument("--budget", default="30", choices=list(PROFILES))
    ap.add_argument("--foliage", default="stylized",
                    choices=["stylized", "simplify", "remove", "keep_raw"])
    ap.add_argument("--detail", default="photocad", choices=["clean", "photocad", "maximum"])
    ap.add_argument("--line-aware", dest="line_aware", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--stroke", type=float, default=None, help="stroke width / drawing width")
    ap.add_argument("--simplify", type=float, default=1.0)
    ap.add_argument("--formats", default="svg,dxf,pdf,png")
    ap.add_argument("--config", default=None, help="YAML RunConfig (CLI flags override)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")

    cfg = RunConfig.from_yaml(a.config) if a.config else RunConfig()
    cfg.budget, cfg.mode, cfg.foliage, cfg.detail = a.budget, a.mode, a.foliage, a.detail
    cfg.line_aware = {"auto": None, "on": True, "off": False}[a.line_aware]
    cfg.simplify = a.simplify
    cfg.formats = tuple(f.strip() for f in a.formats.split(",") if f.strip())
    if a.stroke:
        cfg.stroke_frac = a.stroke
    cfg.validate()
    if cfg.mode == "facade":
        raise SystemExit("error: facade mode (rectification + snapping) is implemented in stage 3. "
                         "Use --mode general or auto.")
    if cfg.mode == "auto":
        print("note: auto mode proposes General in stage 1 (classifier arrives in stage 3).")

    os.makedirs(a.output, exist_ok=True)
    for f in _inputs(a.input):
        t0 = time.perf_counter()
        res = run_pipeline(f, cfg)
        stem = os.path.join(a.output, os.path.splitext(os.path.basename(f))[0])
        t1 = time.perf_counter()
        outs = export_all(stem, res, cfg)
        total = res.timings and sum(res.timings.values()) + (time.perf_counter() - t1)
        st = path_stats(res.paths)
        print(f"{os.path.basename(f)}: {total:.1f}s total  stages="
              + json.dumps({k: round(v, 2) for k, v in res.timings.items()}))
        print(f"  {st['paths']} paths, {st['nodes']} nodes, straight {st['straight_ratio']:.0%}, "
              f"closed {st['closed_ratio']:.0%}, line-aware={'on' if res.line_aware else 'off'}")
        if res.fallbacks:
            print("  Fallbacks used: " + ", ".join(res.fallbacks))
        for n in res.notes:
            logging.info("  " + n)
        print("  wrote " + ", ".join(os.path.basename(v) for v in outs.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
