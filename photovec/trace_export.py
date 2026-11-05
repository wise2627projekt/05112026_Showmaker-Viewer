"""Exporters for TraceResult: filled SVG / PDF / DXF (HATCH + SPLINE) / PNG."""
from __future__ import annotations

from typing import Dict, List

import cv2
import ezdxf
import numpy as np
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfgen.canvas import FILL_EVEN_ODD

from .geometry import Path, bezier_points
from .trace import TraceResult, render_trace, render_trace_rgba


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def _d(p: Path, f) -> str:
    s0 = p.segs[0][1][0]
    out = [f"M{f(s0[0])} {f(s0[1])}"]
    for kind, c in p.segs:
        if kind == "L":
            out.append(f"L{f(c[1][0])} {f(c[1][1])}")
        else:
            out.append("C" + " ".join(f"{f(q[0])} {f(q[1])}" for q in c[1:]))
    return "".join(out) + ("Z" if p.closed else "")


def write_svg(path: str, res: TraceResult) -> None:
    W, H = res.size
    f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    if res.opaque is None:      # fully opaque source: white page; transparent source: no background
        L.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')
    grouped = res.params.auto_group
    by: Dict[str, List[str]] = {}
    for sh in res.shapes:
        d = " ".join(_d(r, f) for r in sh.rings)
        by.setdefault(_hex(sh.rgb), []).append(f'<path d="{d}"/>')
    for col, items in by.items():
        if grouped:
            L.append(f'<g id="fill_{col[1:]}" fill="{col}" fill-rule="evenodd">')
        else:
            L.append(f'<g fill="{col}" fill-rule="evenodd">')
        L += items
        L.append("</g>")
    sby: Dict[tuple, List[str]] = {}
    for st in res.strokes:
        sby.setdefault((_hex(st.rgb), round(st.width, 1)), []).append(f'<path d="{_d(st, f)}"/>')
    for (col, w), items in sby.items():
        gid = f' id="stroke_{col[1:]}"' if grouped else ""
        L.append(f'<g{gid} fill="none" stroke="{col}" stroke-width="{w}" stroke-linecap="round" '
                 f'stroke-linejoin="round">')
        L += items
        L.append("</g>")
    L.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_pdf(path: str, res: TraceResult, page_w_pt: float = 879.0) -> None:
    W, H = res.size
    k = page_w_pt / W
    c = rl_canvas.Canvas(path, pagesize=(W * k, H * k))
    X = lambda q: float(q[0]) * k
    Y = lambda q: (H - float(q[1])) * k
    for sh in res.shapes:
        pa = c.beginPath()
        for r in sh.rings:
            s0 = r.segs[0][1][0]
            pa.moveTo(X(s0), Y(s0))
            for kind, q in r.segs:
                if kind == "L":
                    pa.lineTo(X(q[1]), Y(q[1]))
                else:
                    pa.curveTo(X(q[1]), Y(q[1]), X(q[2]), Y(q[2]), X(q[3]), Y(q[3]))
            pa.close()
        c.setFillColorRGB(*[v / 255.0 for v in sh.rgb])
        c.drawPath(pa, stroke=0, fill=1, fillMode=FILL_EVEN_ODD)
    c.setLineCap(1)
    c.setLineJoin(1)
    for st in res.strokes:
        pa = c.beginPath()
        s0 = st.segs[0][1][0]
        pa.moveTo(X(s0), Y(s0))
        for kind, q in st.segs:
            if kind == "L":
                pa.lineTo(X(q[1]), Y(q[1]))
            else:
                pa.curveTo(X(q[1]), Y(q[1]), X(q[2]), Y(q[2]), X(q[3]), Y(q[3]))
        if st.closed:
            pa.close()
        c.setStrokeColorRGB(*[v / 255.0 for v in st.rgb])
        c.setLineWidth(st.width * k)
        c.drawPath(pa, stroke=1, fill=0)
    c.showPage()
    c.save()


def write_dxf(path: str, res: TraceResult, units_per_px: float = 1.0) -> None:
    """Fills -> HATCH (even-odd, holes preserved; curves flattened); strokes -> LINE/SPLINE."""
    H = res.size[1]
    s = units_per_px
    P = lambda q: (float(q[0]) * s, (H - float(q[1])) * s)
    doc = ezdxf.new("R2018", setup=True)
    msp = doc.modelspace()
    grouped = res.params.auto_group

    def layer(name: str) -> str:
        if name not in doc.layers:
            doc.layers.add(name)
        return name

    for sh in res.shapes:
        lay = layer(f"FILL_{_hex(sh.rgb)[1:]}" if grouped else "TRACE_FILL")
        h = msp.add_hatch(dxfattribs={"layer": lay})
        h.rgb = tuple(int(v) for v in sh.rgb)
        for r in sh.rings:
            poly = r.flatten(10)[:-1] if r.closed else r.flatten(10)
            h.paths.add_polyline_path([P(q) for q in poly], is_closed=True)
    for st in res.strokes:
        lay = layer(f"STROKE_{_hex(st.rgb)[1:]}" if grouped else "TRACE_STROKE")
        at = {"layer": lay, "lineweight": int(max(0, min(211, st.width * s * 100)))}
        run: List[np.ndarray] = []

        def flush():
            if run:
                fit = []
                for cb in run:
                    fit += [P(cb[0]), P(bezier_points(cb, 3)[1])]
                fit.append(P(run[-1][3]))
                msp.add_spline(fit_points=fit, dxfattribs=at)
            run.clear()

        for kind, cb in st.segs:
            if kind == "L":
                flush()
                msp.add_line(P(cb[0]), P(cb[1]), dxfattribs=at)
            else:
                run.append(cb)
        flush()
    doc.saveas(path)


def write_png(path: str, res: TraceResult, view: str = "result") -> None:
    img = render_trace_rgba(res) if res.opaque is not None else render_trace(res, view)
    cv2.imencode(".png", img)[1].tofile(path)
