"""The name of a photo, shown like a film subtitle. The default look: yellow letters with a black outline, Helvetica Medium Italic.

Helvetica cannot be shipped (it is not free), so by default the look is made with Inter (SIL Open Font License, in photovec/fonts) - a neutral
grotesque of the same kind - or with a real "Helvetica Neue" when the computer has one. Everything is customisable through a CaptionStyle
(font, bold, italic, size, colour of the letters and of the outline, outline width), which lives in the show file.
Used by the Viewer (on the opened picture) and by the Show Maker (its preview), so both draw exactly the same thing."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QPainter, QPainterPath, QPen

from .showfile import CaptionStyle

BUNDLED = ("Inter-Medium.otf", "Inter-MediumItalic.otf", "Inter-Bold.otf", "Inter-BoldItalic.otf")
BUNDLED_FAMILY = "Inter"
MARGIN_OF_PICTURE = 0.034        # distance of the baseline from the bottom of the picture
MIN_PX, MAX_PX = 11.0, 120.0

_bundled: dict = {}


def _fonts_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _families() -> dict:
    return {f.lower(): f for f in QFontDatabase.families()}


def _bundled_family() -> str:
    """Inter, registered once from the font files that ship with the program ('' if they cannot be read)."""
    if "fam" not in _bundled:
        fam = ""
        for f in BUNDLED:
            path = os.path.join(_fonts_dir(), f)
            if os.path.isfile(path):
                fid = QFontDatabase.addApplicationFont(path)
                got = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                fam = fam or (got[0] if got else "")
        _bundled["fam"] = fam
    return _bundled["fam"]


def register_font_file(path: str) -> Optional[str]:
    """Make a font file known to Qt; returns its family name (None when it cannot be read)."""
    try:
        fid = QFontDatabase.addApplicationFont(path)
    except Exception:
        return None
    fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
    return fams[0] if fams else None


def register_font_bytes(data: bytes) -> Optional[str]:
    """The same for a font that came inside a show file."""
    try:
        fid = QFontDatabase.addApplicationFontFromData(data)
    except Exception:
        return None
    fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
    return fams[0] if fams else None


def resolve_family(style: CaptionStyle) -> tuple:
    """(family, note): the family that draws the names, and a short remark when it is not the one that was asked for ('' otherwise)."""
    have = _families()
    asked = (style.family or "").strip()
    if asked:
        if asked.lower() in have:
            return have[asked.lower()], ""
        return (_bundled_family() or "Arial"), f"“{asked}” is not installed here, so Inter is used instead."
    for fam in ("Helvetica Neue", "HelveticaNeue"):                       # the real thing, when the computer has it
        if fam.lower() in have:
            return have[fam.lower()], ""
    return (_bundled_family() or "Arial"), ""


def caption_font(px: float, style: Optional[CaptionStyle] = None) -> QFont:
    style = style or CaptionStyle()
    fam, _ = resolve_family(style)
    f = QFont(fam)
    f.setPixelSize(max(1, int(round(px))))
    f.setItalic(bool(style.italic))
    if style.bold:
        f.setWeight(QFont.Bold)
    else:                                                                 # the default look is a Medium; a font chosen by name is shown as it is
        f.setWeight(QFont.Medium if not (style.family or "").strip() else QFont.Normal)
    return f


def caption_path(text: str, picture: QRectF, style: Optional[CaptionStyle] = None) -> Optional[tuple]:
    """(the letters as one path - centred, near the bottom of `picture` -, the font size in px), or None when there is nothing to show."""
    style = style or CaptionStyle()
    text = " ".join(str(text or "").split())
    if not text or picture.width() < 20 or picture.height() < 20:
        return None
    px = min(MAX_PX, max(MIN_PX, picture.height() * style.size / 100.0))
    lw = style.outline / 100.0
    room = picture.width() * 0.92
    font = caption_font(px, style)
    while px > MIN_PX and QFontMetricsF(font).horizontalAdvance(text) + px * lw > room:      # a long name gets smaller ...
        px -= 1.0
        font = caption_font(px, style)
    text = QFontMetricsF(font).elidedText(text, Qt.ElideRight, max(10.0, room - px * lw))    # ... and in the end is cut
    path = QPainterPath()
    path.addText(0, 0, font, text)
    br = path.boundingRect()
    path.translate(picture.center().x() - (br.x() + br.width() / 2), picture.bottom() - picture.height() * MARGIN_OF_PICTURE)
    return path, px


def paint_caption(p: QPainter, picture: QRectF, text: str, style: Optional[CaptionStyle] = None, opacity: float = 1.0) -> None:
    """Draw the name on the bottom of `picture` (the rectangle the photo fills): the outline first, the letters on top.
    `opacity` (0..1) fades the finished name as one piece (the outline does not shine through the letters)."""
    style = style or CaptionStyle()
    opacity = float(min(max(opacity, 0.0), 1.0))
    made = caption_path(text, picture, style)
    if made is None or opacity <= 0.003:
        return
    path, px = made
    if opacity < 0.995:
        pad = int(px * style.outline / 100.0) + 3
        box = path.boundingRect().adjusted(-pad, -pad, pad, pad).toAlignedRect()
        dpr = max(1.0, p.device().devicePixelRatioF() if hasattr(p.device(), "devicePixelRatioF") else 1.0)
        layer = QImage(int(box.width() * dpr), int(box.height() * dpr), QImage.Format_ARGB32_Premultiplied)
        layer.setDevicePixelRatio(dpr)
        layer.fill(0)
        lp = QPainter(layer)
        lp.translate(-box.x(), -box.y())
        _draw(lp, path, px, style)
        lp.end()
        p.save()
        p.setOpacity(p.opacity() * opacity)
        p.drawImage(box.topLeft(), layer)
        p.restore()
        return
    _draw(p, path, px, style)


def _draw(p: QPainter, path: QPainterPath, px: float, style: CaptionStyle) -> None:
    p.save()
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
    width = px * style.outline / 100.0
    if width >= 0.5:
        p.setPen(QPen(QColor(style.outline_color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(style.color))
    p.drawPath(path)
    p.restore()
