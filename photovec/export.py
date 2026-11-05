"""Writers: SVG, DXF, PDF, PNG preview/overlay; path statistics."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import cv2
import ezdxf
import numpy as np
from reportlab.pdfgen import canvas as rl_canvas

from .config import RunConfig
from .geometry import Path, bezier_points

LAYER_ORDER = ["STRUCTURE", "EDGES", "CRACKS", "TONAL", "FOLIAGE", "TEXTURE", "DETAIL"]


def ordered(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    """Group by layer; within a layer the longest (structural) paths are drawn first."""
    by: Dict[str, List[Path]] = {}
    for p in paths:
        by.setdefault(p.layer, []).append(p)
    names = sorted(by, key=lambda n: LAYER_ORDER.index(n) if n in LAYER_ORDER else 99)
    return {n: sorted(by[n], key=lambda p: -p.length()) for n in names}


def path_stats(paths: List[Path]) -> Dict[str, float]:
    n = len(paths)
    ink = sum(p.length() for p in paths)
    return {
        "paths": n,
        "nodes": int(sum(p.node_count() for p in paths)),
        "ink_length_px": round(ink, 1),
        "mean_chain_len_px": round(ink / n, 1) if n else 0.0,
        "straight_ratio": round(sum(p.is_straight() for p in paths) / n, 3) if n else 0.0,
        "closed_ratio": round(sum(p.closed for p in paths) / n, 3) if n else 0.0,
    }


def _svg_d(p: Path, f) -> str:
    s0 = p.segs[0][1][0]
    out = [f"M{f(s0[0])} {f(s0[1])}"]
    for kind, c in p.segs:
        if kind == "L":
            out.append(f"L{f(c[1][0])} {f(c[1][1])}")
        else:
            out.append("C" + " ".join(f"{f(q[0])} {f(q[1])}" for q in c[1:]))
    return "".join(out) + ("Z" if p.closed else "")


def write_svg(path: str, paths: List[Path], size: Tuple[int, int], cfg: RunConfig) -> None:
    w, h = size
    m = cfg.margin_frac * max(w, h)
    sw = cfg.stroke_frac * w
    f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{f(-m)} {f(-m)} {f(w + 2*m)} {f(h + 2*m)}" '
             f'width="{f(w + 2*m)}" height="{f(h + 2*m)}">',
             f'<rect x="{f(-m)}" y="{f(-m)}" width="{f(w + 2*m)}" height="{f(h + 2*m)}" fill="#fff"/>',
             f'<g fill="none" stroke="#000" stroke-width="{sw:.3f}" stroke-linecap="round" '
             f'stroke-linejoin="round">']
    for layer, ps in ordered(paths).items():
        lines.append(f'<g id="{layer}">')
        lines.extend(f'<path d="{_svg_d(p, f)}"/>' for p in ps)
        lines.append("</g>")
    lines += ["</g>", "</svg>"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_dxf(path: str, paths: List[Path], size: Tuple[int, int], cfg: RunConfig) -> None:
    s, h = cfg.units_per_px, size[1]
    P = lambda q: (float(q[0]) * s, (h - float(q[1])) * s)
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    for layer, ps in ordered(paths).items():
        if layer not in doc.layers:
            doc.layers.add(layer)
        at = {"layer": layer}
        for p in ps:
            run: List[np.ndarray] = []

            def flush():
                if run:
                    # fit points: start and midpoint of every Bezier, then the final end
                    ordered_fit = []
                    for c in run:
                        ordered_fit += [P(c[0]), P(bezier_points(c, 3)[1])]
                    ordered_fit.append(P(run[-1][3]))
                    msp.add_spline(fit_points=ordered_fit, dxfattribs=at)
                run.clear()

            for kind, c in p.segs:
                if kind == "L":
                    flush()
                    msp.add_line(P(c[0]), P(c[1]), dxfattribs=at)
                else:
                    run.append(c)
            flush()
    doc.saveas(path)


def write_pdf(path: str, paths: List[Path], size: Tuple[int, int], cfg: RunConfig,
              page_w_pt: float = 879.0) -> None:
    w, h = size
    k = page_w_pt / (w * (1 + 2 * cfg.margin_frac * max(w, h) / w))
    m = cfg.margin_frac * max(w, h) * k
    PW, PH = w * k + 2 * m, h * k + 2 * m
    c = rl_canvas.Canvas(path, pagesize=(PW, PH))
    c.setLineWidth(cfg.stroke_frac * w * k)
    c.setLineCap(1)
    c.setLineJoin(1)
    c.setStrokeColorRGB(0, 0, 0)
    X = lambda q: m + float(q[0]) * k
    Y = lambda q: PH - m - float(q[1]) * k
    for ps in ordered(paths).values():
        for p in ps:
            pa = c.beginPath()
            s0 = p.segs[0][1][0]
            pa.moveTo(X(s0), Y(s0))
            for kind, q in p.segs:
                if kind == "L":
                    pa.lineTo(X(q[1]), Y(q[1]))
                else:
                    pa.curveTo(X(q[1]), Y(q[1]), X(q[2]), Y(q[2]), X(q[3]), Y(q[3]))
            if p.closed:
                pa.close()
            c.drawPath(pa, stroke=1, fill=0)
    c.showPage()
    c.save()


def render_png(paths: List[Path], size: Tuple[int, int], cfg: RunConfig,
               photo_bgr: np.ndarray = None, scale: float = 1.0) -> np.ndarray:
    w, h = size
    if photo_bgr is not None:
        canvas = cv2.addWeighted(photo_bgr, 0.35, np.full_like(photo_bgr, 255), 0.65, 0)
        colour = (0, 0, 220)
    else:
        canvas = np.full((h, w, 3), 255, np.uint8)
        colour = (0, 0, 0)
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    th = max(1, int(round(cfg.stroke_frac * w * scale * (1.6 if photo_bgr is not None else 1.0))))
    for ps in ordered(paths).values():
        for p in ps:
            q = np.rint(p.flatten() * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [q], p.closed, colour, th, cv2.LINE_AA)
    return canvas


def export_all(stem: str, res, cfg: RunConfig) -> Dict[str, str]:
    """Write all requested formats next to `stem`; returns {format: path}."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    out: Dict[str, str] = {}
    if "svg" in cfg.formats:
        write_svg(stem + ".svg", res.paths, res.size, cfg); out["svg"] = stem + ".svg"
    if "dxf" in cfg.formats:
        write_dxf(stem + ".dxf", res.paths, res.size, cfg); out["dxf"] = stem + ".dxf"
    if "pdf" in cfg.formats:
        write_pdf(stem + ".pdf", res.paths, res.size, cfg); out["pdf"] = stem + ".pdf"
    if "png" in cfg.formats:
        cv2.imwrite(stem + "_lines.png", render_png(res.paths, res.size, cfg, None, 2.0))
        cv2.imwrite(stem + "_overlay.png", render_png(res.paths, res.size, cfg, res.image_bgr, 1.0))
        out["png"] = stem + "_lines.png"
    return out
