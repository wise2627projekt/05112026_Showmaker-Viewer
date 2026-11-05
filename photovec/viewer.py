"""PhotoVec Viewer: plays a `.photovecshow` file (made by the Show Maker). Nothing is traced here; everything was prerendered.

Look ("Quiet gallery"): a white window with nothing but the photo grid (black-and-white thumbnails, thin black frames, small gaps).
A click lifts the photo out of its place: the grid is dimmed, the photo grows into a white card in colour, and below the picture two
hairline sliders with a small black square appear, named as chosen in the Show Maker, plus a black arrow that leads back. A photo whose
letters were marked in the Show Maker has a third slider that moves them from the drawing into a line of text in a font.

Keys: Esc / Backspace / the arrow = back   Left, Right = slider 1   Up, Down = slider 2   Page up / down = slider 3   F11 = full screen.
Which show is played is asked when the Viewer starts; to play another one, close the Viewer and start it again (a show file can also be
dropped onto the window, or double-clicked).
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSettings, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from .captions import paint_caption, register_font_bytes
from .layers import _to_u8
from .letters import LetterSet
from .showfile import EXT, PhotoInfo, ShowError, ShowReader, compose, compose_over, facade_look, step_at

# --------------------------------------------------------------------------- look (logical pixels)
GRID_GAP = 6
GRID_MARGIN = 14
FRAME = 2                    # black frame of a thumbnail
DIM = 0.80                   # how dark the grid gets behind the photo (0..1)
ANIM_MS = 320
CARD_PAD = 12                # white border around the photo
CARD_GAP = 84                # extra white band below the photo for the sliders
CARD_GAP3 = 98               # the same with three sliders
CARD_MIN_W = 380
SLIDER_ROWS = (62, 28)       # distance of the two slider lines from the bottom of the card
SLIDER_ROWS3 = (76, 48, 20)  # the same for three sliders (tight rows, the name stands left of its line)
LABEL_COL = (54, 120)        # width of the column with the names (three sliders): at least / at most
SLIDER_LEFT = 52             # left end of the lines (the arrow lives left of it)
SLIDER_RIGHT = 24
GRABBER = 11
INK = QColor(0, 0, 0)
PAPER = QColor(255, 255, 255)


# --------------------------------------------------------------------------- layout (pure functions, no Qt)
def _justify(aspects: List[float], width: float, gap: float, target: float) -> List[Tuple[List[int], float]]:
    """Rows of a justified grid: every row is as wide as `width` and about `target` high; photos keep their shape. The last row is left
    alone (centred, at `target` height) when stretching it to the full width would make it much taller than the others."""
    rows: List[Tuple[List[int], float]] = []
    cur: List[int] = []
    sa = 0.0

    def height(items: List[int], s: float) -> float:
        return (width - gap * (len(items) - 1)) / s

    for i, a in enumerate(aspects):
        cur.append(i)
        sa += a
        if sa * target + gap * (len(cur) - 1) >= width:
            h_with = height(cur, sa)
            if len(cur) > 1 and abs(height(cur[:-1], sa - a) - target) < abs(h_with - target) and sa - a > 0:
                rows.append((cur[:-1], height(cur[:-1], sa - a)))
                cur, sa = [i], a
                continue
            rows.append((cur, h_with))
            cur, sa = [], 0.0
    if cur:
        h = height(cur, sa)
        rows.append((cur, h if h <= 1.3 * target else target))
    return rows


def grid_layout(aspects: List[float], width: float, height: float, gap: float = GRID_GAP, margin: float = GRID_MARGIN,
                min_row: float = 150.0) -> Tuple[List[QRectF], float]:
    """Rectangles (x, y, w, h) of the thumbnails and the height of the whole content. If everything fits into the window the grid is made as
    large as possible and centred; otherwise it scrolls (rows of a pleasant fixed height)."""
    n = len(aspects)
    if n == 0:
        return [], 0.0
    aw, ah = max(10.0, width - 2 * margin), max(10.0, height - 2 * margin)
    best = None
    t = min(ah, aw * 0.6, 520.0)
    while t >= min_row:
        rows = _justify(aspects, aw, gap, t)
        total = sum(h for _, h in rows) + gap * (len(rows) - 1)
        if total <= ah:
            best = (rows, total)
            break
        t *= 0.96
    scroll = best is None
    if scroll:
        t = min(max(min_row, 0.22 * ah), 260.0)
        rows = _justify(aspects, aw, gap, t)
        best = (rows, sum(h for _, h in rows) + gap * (len(rows) - 1))
    rows, total = best
    rects: List[Optional[QRectF]] = [None] * n
    y = margin if scroll else margin + (ah - total) / 2
    for items, h in rows:
        row_w = sum(aspects[i] * h for i in items) + gap * (len(items) - 1)
        x = margin + (aw - row_w) / 2
        for i in items:
            w = aspects[i] * h
            rects[i] = QRectF(x, y, w, h)
            x += w + gap
        y += h + gap
    content = (y - gap + margin) if scroll else height
    return rects, content


def card_gap(rows: int = 2) -> float:
    return CARD_GAP3 if rows >= 3 else CARD_GAP


def detail_layout(W: float, H: float, pw: float, ph: float, rows: int = 2) -> Tuple[QRectF, QRectF]:
    """(card, photo) rectangles of the opened photo: the picture as large as fits, with the white border and the band for the sliders."""
    gap = card_gap(rows)
    m = max(40.0, 0.06 * min(W, H))
    aw = max(40.0, W - 2 * m - 2 * CARD_PAD)
    ah = max(40.0, H - 2 * m - 2 * CARD_PAD - gap)
    s = min(aw / pw, ah / ph)
    fw, fh = pw * s, ph * s
    cw = max(fw + 2 * CARD_PAD, CARD_MIN_W)
    ch = fh + 2 * CARD_PAD + gap
    card = QRectF((W - cw) / 2, (H - ch) / 2, cw, ch)
    photo = QRectF(card.center().x() - fw / 2, card.top() + CARD_PAD, fw, fh)
    return card, photo


def slider_geometry(card: QRectF) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], QRectF]:
    """((x0, x1, y) of slider 1's line, of slider 2's line, the hit rectangle of the back arrow)."""
    x0, x1 = card.left() + SLIDER_LEFT, card.right() - SLIDER_RIGHT
    y1, y2 = card.bottom() - SLIDER_ROWS[0], card.bottom() - SLIDER_ROWS[1]
    arrow = QRectF(card.left() + 4, (y1 + y2) / 2 - 18, 44, 36)
    return (x0, x1, y1), (x0, x1, y2), arrow


def slider_rows(card: QRectF, rows: int = 2, label_w: float = LABEL_COL[0]) -> Tuple[List[Tuple[float, float, float]], QRectF]:
    """([(x0, x1, y) of every slider's line], the hit rectangle of the back arrow). With three sliders the rows are tight and the names stand
    to the left of the lines (`label_w` wide); with two they stand above, as before."""
    if rows < 3:
        a, b, arrow = slider_geometry(card)
        return [a, b], arrow
    x0, x1 = card.left() + SLIDER_LEFT + label_w, card.right() - SLIDER_RIGHT
    ys = [card.bottom() - d for d in SLIDER_ROWS3]
    return [(x0, x1, y) for y in ys], QRectF(card.left() + 4, ys[1] - 18, 44, 36)


def _qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()


def _qgray(g: np.ndarray) -> QImage:
    g = np.ascontiguousarray(g)
    h, w = g.shape[:2]
    return QImage(g.data, w, h, g.strides[0], QImage.Format_Grayscale8).copy()


# --------------------------------------------------------------------------- the photo that is open
class Work:
    """The arrays of ONE opened photo at the size it is shown (prerendered steps are decoded in the background, nearest to the start first)."""
    BYTES = 320 * 1024 * 1024

    def __init__(self, reader: ShowReader, index: int, size: Tuple[int, int]):
        self.reader, self.index, self.size = reader, index, size
        self.info: PhotoInfo = reader.photos[index]
        self.photo: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.steps: "OrderedDict[int, Tuple[np.ndarray, Optional[np.ndarray]]]" = OrderedDict()
        self._lock = threading.Lock()
        self._pending: set = set()
        self.cancelled = False
        self._looks: "OrderedDict[float, Optional[np.ndarray]]" = OrderedDict()      # the background for a few positions of slider 2
        self.letters: Optional[LetterSet] = None                                      # the marked letters (slider 3), if the photo has any
        self._morphed: "OrderedDict[tuple, tuple]" = OrderedDict()                    # (step, slider 3) -> picture with the letters moved
        self._llock = threading.Lock()
        self._photo_f: Optional[np.ndarray] = None                                   # the photo as float32, for the quick compose

    def _fit(self, a: np.ndarray, interp: int) -> np.ndarray:
        w, h = self.size
        return a if a.shape[1] == w and a.shape[0] == h else cv2.resize(a, (w, h), interpolation=interp)

    def load_base(self) -> None:
        photo = self._fit(self.reader.photo(self.index), cv2.INTER_AREA)
        mask = self.reader.mask(self.index) if self.info.masked else None
        if mask is not None:
            mask = self._fit(mask, cv2.INTER_AREA)
        self.photo, self.mask = photo, mask
        if self.info.letters:
            try:
                ls = LetterSet(self.reader.letters(self.index), (self.info.w, self.info.h), self.size)
                if len(ls):
                    ls.prepare()
                    self.letters = ls
            except Exception as exc:                                       # (damaged letters: the photo itself still plays, without slider 3 doing anything)
                print("Viewer: the letters of this photo cannot be used:", exc)

    def load_step(self, k: int) -> bool:
        if self.cancelled:
            return False
        with self._lock:
            if k in self.steps or k in self._pending:
                return False
            self._pending.add(k)
        try:
            vec = self._fit(self.reader.vec(self.index, k), cv2.INTER_AREA)
            ink = self.reader.ink(self.index, k) if self.info.has_ink else None
            if ink is not None:
                ink = self._fit(ink, cv2.INTER_AREA)
        finally:
            with self._lock:
                self._pending.discard(k)
        if self.cancelled:
            return False
        with self._lock:
            self.steps[k] = (vec, ink)
            per = vec.nbytes + (0 if ink is None else ink.nbytes)
            while len(self.steps) > 2 and per * len(self.steps) > self.BYTES:
                self.steps.popitem(last=False)
        return True

    def nearest(self, k: int) -> Optional[int]:
        with self._lock:
            if not self.steps:
                return None
            return min(self.steps, key=lambda j: (abs(j - k), j))

    def picture(self, k: int, pos2: float, pos3: float = 0.0) -> Optional[Tuple[np.ndarray, bool]]:
        """(the picture, True if it is exactly step k) or None while nothing is loaded yet. pos3: how far the letters are on their way into text."""
        j = self.nearest(k)
        if j is None or self.photo is None:
            return None
        with self._lock:
            vec, ink = self.steps[j]
        mask = self.mask
        if pos3 > 0.0 and self.letters is not None:
            key = (j, round(pos3 * 100))
            with self._llock:
                hit = self._morphed.get(key)
                if hit is None:
                    hit = self.letters.apply(vec, ink, self.info.paper, round(pos3 * 100) / 100.0, self.info.lmotion, mask)
                    self._morphed[key] = hit
                    while len(self._morphed) > 4:
                        self._morphed.popitem(last=False)
                else:
                    self._morphed.move_to_end(key)
            vec, ink, mask = hit                                           # (the letters leave the mask: it travels with them)
        if self.info.masked and mask is not None:
            if pos2 not in self._looks:
                look = facade_look(self.info, self.photo, self.mask, pos2)
                if look is None:                                           # (the plain photo: its float copy is kept, converting it for every frame is slow)
                    if self._photo_f is None:
                        self._photo_f = self.photo.astype(np.float32)
                    look = self._photo_f
                self._looks[pos2] = (look, _to_u8(look))
                while len(self._looks) > 3:
                    self._looks.popitem(last=False)
            look, base = self._looks[pos2]
            return compose_over(look, mask, vec, ink, base), j == k
        return compose(self.info, self.photo, mask, vec, ink, pos2), j == k


# --------------------------------------------------------------------------- the window
class Viewer(QWidget):
    _loaded = Signal(int, int)               # (generation, step): a prerendered step has been decoded
    _morphed = Signal(int, object, object)   # (generation, cache key, QImage): a picture with moved letters is ready

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoVec Viewer")
        self.setMinimumSize(520, 380)
        self.resize(1280, 800)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.reader: Optional[ShowReader] = None
        self.gray: List[np.ndarray] = []
        self.colour: List[np.ndarray] = []
        self.aspects: List[float] = []
        self.rects: List[QRectF] = []
        self.content_h = 0.0
        self.scroll = 0.0
        self._thumbs: Dict[tuple, QPixmap] = {}
        self.sel = -1                                           # opened photo (-1 = grid)
        self.p = 0.0                                            # 0 = grid, 1 = fully opened
        self.closing = False
        self.work: Optional[Work] = None
        self.gen = 0
        self.pos: Dict[int, List[float]] = {}                   # remembered slider positions of every photo
        self._drag = -1
        self._cache: "OrderedDict[tuple, QImage]" = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="viewer")
        self._lpool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viewer-letters")      # the letters on their way (slider 3)
        self._want: Optional[tuple] = None                      # the newest picture asked for while the letters move (latest wins)
        self._busy = False
        self._last: Optional[QImage] = None                     # what was drawn last (shown until the new picture is ready)
        self._label_w = LABEL_COL[0]
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(self._anim_done)
        self._loaded.connect(self._on_loaded)
        self._morphed.connect(self._on_morphed)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(160)
        self._resize_timer.timeout.connect(self._resized)
        f = QFont(self.font())
        f.setPointSizeF(9.0)
        self._font = f

    # ---- file
    def load(self, path: str) -> bool:
        try:
            reader = ShowReader(path)
            gray, colour = [], []
            for i in range(len(reader.photos)):
                t = reader.thumb(i)
                colour.append(t)
                gray.append(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY))
        except (ShowError, OSError) as exc:
            QMessageBox.warning(self, "PhotoVec Viewer", str(exc))
            return False
        self._leave_instantly()
        if self.reader is not None:
            self.reader.close()
        try:
            data = reader.caption_font()                          # (the font of the names, when the maker put it into the show)
            fam = register_font_bytes(data) if data else None
            if fam:
                reader.caption_style.family = fam
        except Exception as exc:
            print("Viewer: the font of the names cannot be used:", exc)
        self.reader, self.gray, self.colour = reader, gray, colour
        self.aspects = [max(0.05, g.shape[1] / float(g.shape[0])) for g in gray]
        self.pos = {i: [p.start1, p.start2, p.start3 if p.letters else 0.0] for i, p in enumerate(reader.photos)}
        fm = QFontMetricsF(self._font)
        self._label_w = min(LABEL_COL[1], max(LABEL_COL[0], max(fm.horizontalAdvance(t) for t in reader.labels) + 12))
        self.scroll = 0.0
        self._thumbs.clear()
        self._relayout()
        self.setWindowTitle(reader.title or "PhotoVec Viewer")
        try:
            QSettings("PhotoVec", "Viewer").setValue("last", os.path.abspath(path))
        except Exception:
            pass
        self.update()
        return True

    def open_dialog(self) -> bool:
        """Ask which show to open (the last one is pre-selected). True if a show was opened."""
        start = ""
        try:
            last = str(QSettings("PhotoVec", "Viewer").value("last", ""))
            start = last if os.path.isfile(last) else os.path.dirname(last)
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(self, "Which show do you want to play?", start, f"PhotoVec show (*{EXT})")
        return bool(path) and self.load(path)

    def ask_at_start(self) -> None:
        """The Viewer always asks which show to play when it is started; to play another one, close it and start it again.
        Nothing to look at without a show, so the program ends if the question is cancelled."""
        if not self.open_dialog() and self.reader is None:
            self.close()

    # ---- grid
    def _relayout(self) -> None:
        self.rects, self.content_h = grid_layout(self.aspects, self.width(), self.height())
        self.scroll = min(max(0.0, self.scroll), max(0.0, self.content_h - self.height()))
        self._thumbs.clear()

    def _thumb_pixmap(self, i: int, r: QRectF) -> QPixmap:
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(round(r.width() * dpr))), max(1, int(round(r.height() * dpr)))
        key = (i, w, h)
        pm = self._thumbs.get(key)
        if pm is None:
            g = self.gray[i]
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA if g.shape[1] > w else cv2.INTER_CUBIC)
            pm = QPixmap.fromImage(_qgray(g))
            pm.setDevicePixelRatio(dpr)
            self._thumbs[key] = pm
        return pm

    def _hit(self, pt: QPointF) -> int:
        for i, r in enumerate(self.rects):
            if r.translated(0, -self.scroll).contains(pt):
                return i
        return -1

    # ---- opening / closing
    def _rows(self) -> int:
        """How many sliders the opened photo has."""
        return 3 if self.sel >= 0 and self.reader is not None and self.reader.photos[self.sel].letters else 2

    def _target_geometry(self) -> Tuple[QRectF, QRectF]:
        info = self.reader.photos[self.sel]
        return detail_layout(self.width(), self.height(), info.w, info.h, self._rows())

    def _slider_rows(self, card: QRectF) -> Tuple[List[Tuple[float, float, float]], QRectF]:
        return slider_rows(card, self._rows(), self._label_w)

    def _work_size(self) -> Tuple[int, int]:
        info = self.reader.photos[self.sel]
        _, photo = self._target_geometry()
        dpr = self.devicePixelRatioF()
        w, h = int(np.ceil(photo.width() * dpr)), int(np.ceil(photo.height() * dpr))
        f = min(1.0, info.w / float(max(1, w)), info.h / float(max(1, h)))                    # never bigger than what is stored
        return max(16, int(round(w * f))), max(16, int(round(h * f)))

    def open_photo(self, i: int) -> None:
        if self.reader is None or not (0 <= i < len(self.reader.photos)) or self._anim.state() == QVariantAnimation.Running:
            return
        self.sel, self.closing = i, False
        self._want, self._last = None, None
        self._start_work()
        self._animate(0.0, 1.0)
        self.setCursor(Qt.ArrowCursor)

    def close_photo(self) -> None:
        if self.sel < 0 or self.closing:
            return
        self.closing, self._drag = True, -1
        self._animate(self.p, 0.0)

    def _animate(self, a: float, b: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(float(a))
        self._anim.setEndValue(float(b))
        self._anim.setDuration(int(ANIM_MS * max(0.25, abs(b - a))))
        self._anim.start()

    def _on_anim(self, v) -> None:
        self.p = float(v)
        self.update()

    def _anim_done(self) -> None:
        if self.closing:
            self._leave_instantly()
        self.update()

    def _leave_instantly(self) -> None:
        self._anim.stop()
        self.gen += 1
        if self.work is not None:
            self.work.cancelled = True
        self.work, self.sel, self.p, self.closing, self._drag = None, -1, 0.0, False, -1
        self._want, self._last = None, None
        self._cache.clear()

    def _start_work(self) -> None:
        self.gen += 1
        gen, i = self.gen, self.sel
        if self.work is not None:
            self.work.cancelled = True
        self._cache.clear()
        w = Work(self.reader, i, self._work_size())
        self.work = w
        k0 = step_at(w.info, self.pos[i][0])
        order = sorted(range(w.info.steps), key=lambda k: (abs(k - k0), k))

        def base():
            try:
                w.load_base()
            except Exception as exc:                                                       # (a damaged file must not kill the window)
                print("Viewer: could not load the photo:", exc)
                return
            self._loaded.emit(gen, -1)

        def step(k):
            try:
                if w.load_step(k):
                    self._loaded.emit(gen, k)
            except Exception as exc:
                print("Viewer: could not load a step:", exc)
        self._pool.submit(base)
        for k in order:
            self._pool.submit(step, k)

    def _on_loaded(self, gen: int, k: int) -> None:
        if gen != self.gen or self.sel < 0:
            return
        if k < 0:
            self._cache.clear()
            self.update()
            return
        cur = step_at(self.reader.photos[self.sel], self.pos[self.sel][0])
        if self.work is not None and self.work.nearest(cur) == k:
            self.update()

    def _resized(self) -> None:
        if self.sel >= 0 and self.work is not None and not self.closing:
            w, h = self._work_size()
            if w > self.work.size[0] * 1.15 or h > self.work.size[1] * 1.15 or w < self.work.size[0] * 0.5:
                self._start_work()
        self.update()

    # ---- the picture of the opened photo
    def _image(self) -> Optional[QImage]:
        if self.work is None or self.sel < 0:
            return None
        pos = self.pos[self.sel]
        k = step_at(self.work.info, pos[0])
        p2, p3 = round(pos[1] * 100), (round(pos[2] * 100) if self.work.letters is not None else 0)
        key = (self.gen, self.work.nearest(k), p2, p3)
        im = self._cache.get(key)
        if im is not None:
            self._cache.move_to_end(key)
            self._last = im
            return im
        if p3 > 0:                                                  # moving letters take longer: computed aside, the last picture stays meanwhile
            self._want = (key, k, p2 / 100.0, p3 / 100.0)
            self._kick()
            if self._last is not None:
                return self._last
        res = self.work.picture(k, p2 / 100.0, 0.0)
        if res is None:
            return None
        im = _qimage(res[0])
        if p3 == 0:
            self._cache[key] = im
            self._last = im
        while len(self._cache) > 24:
            self._cache.popitem(last=False)
        return im

    def _kick(self) -> None:
        if self._busy:
            return
        self._busy = True
        gen, work = self.gen, self.work

        def run():
            try:
                while True:
                    want = self._want
                    if want is None or want[0][0] != gen or work.cancelled:
                        return
                    key, k, p2, p3 = want
                    res = work.picture(k, p2, p3)
                    if res is not None:
                        self._morphed.emit(gen, key, _qimage(res[0]))
                    if self._want is want:
                        return
            except Exception as exc:                                  # (never let a failing frame kill the window)
                print("Viewer: could not move the letters:", exc)
            finally:
                self._busy = False
        self._lpool.submit(run)

    def _on_morphed(self, gen: int, key, im) -> None:
        if gen != self.gen or self.sel < 0:
            return
        self._cache[key] = im
        while len(self._cache) > 24:
            self._cache.popitem(last=False)
        if self._want is not None and self._want[0] == key:
            self._last = im
        self.update()

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), PAPER)
        if self.reader is None:
            return
        vis = QRectF(self.rect())
        for i, r in enumerate(self.rects):
            rr = r.translated(0, -self.scroll)
            if not rr.intersects(vis):
                continue
            p.drawPixmap(rr.topLeft(), self._thumb_pixmap(i, r))
            p.setPen(QPen(INK, FRAME))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rr.adjusted(FRAME / 2, FRAME / 2, -FRAME / 2, -FRAME / 2))
        if self.sel >= 0 and self.p > 0.0:
            self._paint_detail(p)

    def _paint_detail(self, p: QPainter) -> None:
        t = self.p
        p.fillRect(self.rect(), QColor(0, 0, 0, int(255 * DIM * min(1.0, t))))
        card_end, photo_end = self._target_geometry()
        start = self.rects[self.sel].translated(0, -self.scroll)
        lerp = lambda a, b: a + (b - a) * t
        photo = QRectF(lerp(start.x(), photo_end.x()), lerp(start.y(), photo_end.y()), lerp(start.width(), photo_end.width()), lerp(start.height(), photo_end.height()))
        pad, gap = CARD_PAD * t, card_gap(self._rows()) * t
        # the card follows the photo (so the picture is never squeezed): it grows a border and, at the end, the band for the sliders
        cw_extra = max(0.0, card_end.width() - photo_end.width() - 2 * CARD_PAD) * t
        card = QRectF(photo.left() - pad - cw_extra / 2, photo.top() - pad, photo.width() + 2 * pad + cw_extra, photo.height() + 2 * pad + gap)
        p.fillRect(card, PAPER)
        img = self._image()
        p.save()
        p.setClipRect(photo)
        if t < 1.0:                                                                            # the grey thumbnail turns to colour while it grows
            p.drawPixmap(photo.toRect(), self._thumb_pixmap(self.sel, self.rects[self.sel]))
            p.setOpacity(min(1.0, t * 1.4))
        p.drawImage(photo, img if img is not None else _qimage(self.colour[self.sel]))
        p.restore()
        p.setOpacity(1.0)
        a = min(1.0, max(0.0, (t - 0.55) / 0.45))
        if a > 0:
            p.setOpacity(a)
            name = self.reader.photos[self.sel].caption
            if name:                                                                           # the photo's name, like a film subtitle on the picture
                p.save()
                p.setClipRect(photo)
                info = self.reader.photos[self.sel]
                pos1 = step_at(info, self.pos[self.sel][0]) / float(max(1, info.steps - 1))
                paint_caption(p, photo, name, self.reader.caption_style, self.reader.caption_style.opacity(pos1))
                p.restore()
            self._paint_controls(p, card_end if t >= 1.0 else card)
            p.setOpacity(1.0)

    def _paint_controls(self, p: QPainter, card: QRectF) -> None:
        rows, arrow = self._slider_rows(card)
        p.setFont(self._font)
        pos = list(self.pos[self.sel])
        info = self.reader.photos[self.sel]
        pos[0] = step_at(info, pos[0]) / float(max(1, info.steps - 1))
        left = len(rows) >= 3
        for (x0, x1, y), v, text in zip(rows, pos, self.reader.labels):
            p.setPen(INK)
            if left:                                                                           # three sliders: the name stands left of its line
                shown = QFontMetricsF(self._font).elidedText(text, Qt.ElideRight, self._label_w - 6)
                p.drawText(QRectF(card.left() + SLIDER_LEFT, y - 10, self._label_w - 4, 20), Qt.AlignLeft | Qt.AlignVCenter, shown)
            else:
                p.drawText(QRectF(x0, y - 27, x1 - x0, 18), Qt.AlignLeft | Qt.AlignBottom, text)
            p.setPen(QPen(INK, 1.25))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            cx = x0 + (x1 - x0) * v
            p.setPen(Qt.NoPen)
            p.setBrush(INK)
            p.drawRect(QRectF(cx - GRABBER / 2, y - GRABBER / 2, GRABBER, GRABBER))
        c = arrow.center()
        p.setPen(QPen(INK, 1.6, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
        p.setBrush(Qt.NoBrush)
        x_l, x_r = c.x() - 12, c.x() + 12
        p.drawLine(QPointF(x_l, c.y()), QPointF(x_r, c.y()))
        p.drawLine(QPointF(x_l, c.y()), QPointF(x_l + 7, c.y() - 7))
        p.drawLine(QPointF(x_l, c.y()), QPointF(x_l + 7, c.y() + 7))

    # ---- mouse / keys
    def _card_end(self) -> QRectF:
        return self._target_geometry()[0]

    def _row_at(self, pt: QPointF) -> int:
        rows, _ = self._slider_rows(self._card_end())
        for row, (x0, x1, y) in enumerate(rows):
            if x0 - 14 <= pt.x() <= x1 + 14 and abs(pt.y() - y) <= (11 if len(rows) >= 3 else 15):
                return row
        return -1

    def _set_from_x(self, row: int, x: float) -> None:
        rows, _ = self._slider_rows(self._card_end())
        x0, x1, _ = rows[row]
        v = min(max((x - x0) / max(1.0, x1 - x0), 0.0), 1.0)
        self._set(row, v)

    def _set(self, row: int, v: float) -> None:
        info = self.reader.photos[self.sel]
        if row == 0:
            v = step_at(info, v) / float(max(1, info.steps - 1))
        else:
            v = round(min(max(v, 0.0), 1.0) * 100) / 100.0
        if v != self.pos[self.sel][row]:
            self.pos[self.sel][row] = v
            if row == 0 and self.work is not None:
                k = step_at(info, v)
                if self.work.nearest(k) != k:                                                # not decoded yet: fetch it now
                    gen = self.gen
                    self._pool.submit(lambda: self.work.load_step(k) and self._loaded.emit(gen, k))
            self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self.reader is None:
            return
        pt = e.position()
        if self.sel < 0:
            i = self._hit(pt)
            if i >= 0:
                self.open_photo(i)
            return
        if self.p < 1.0 or self.closing:
            return
        card = self._card_end()
        if self._slider_rows(card)[1].contains(pt):
            self.close_photo()
            return
        row = self._row_at(pt)
        if row >= 0:
            self._drag = row
            self._set_from_x(row, pt.x())
            return
        if not card.contains(pt):
            self.close_photo()

    def mouseMoveEvent(self, e):
        pt = e.position()
        if self._drag >= 0:
            self._set_from_x(self._drag, pt.x())
            return
        if self.reader is None:
            return
        hand = False
        if self.sel < 0:
            hand = self._hit(pt) >= 0
        elif self.p >= 1.0 and not self.closing:
            card = self._card_end()
            hand = self._slider_rows(card)[1].contains(pt) or self._row_at(pt) >= 0 or not card.contains(pt)
        self.setCursor(Qt.PointingHandCursor if hand else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag = -1

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if not d or self.reader is None:
            return
        if self.sel < 0:
            self.scroll = min(max(0.0, self.scroll - d * 0.6), max(0.0, self.content_h - self.height()))
            self.update()
            return
        if self.p < 1.0 or self.closing:
            return
        row = max(0, self._row_at(e.position()))
        info = self.reader.photos[self.sel]
        if row == 0:
            self._set(0, (step_at(info, self.pos[self.sel][0]) + (1 if d > 0 else -1)) / float(max(1, info.steps - 1)))
        else:
            self._set(row, self.pos[self.sel][row] + (0.05 if d > 0 else -0.05))

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif self.sel >= 0 and (k in (Qt.Key_Escape, Qt.Key_Backspace)):
            self.close_photo()
        elif self.sel >= 0 and self.p >= 1.0 and k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown):
            info = self.reader.photos[self.sel]
            if k in (Qt.Key_Left, Qt.Key_Right):
                self._set(0, (step_at(info, self.pos[self.sel][0]) + (1 if k == Qt.Key_Right else -1)) / float(max(1, info.steps - 1)))
            elif k in (Qt.Key_Up, Qt.Key_Down):
                self._set(1, self.pos[self.sel][1] + (0.05 if k == Qt.Key_Up else -0.05))
            elif self._rows() >= 3:
                self._set(2, self.pos[self.sel][2] + (0.05 if k == Qt.Key_PageUp else -0.05))
        elif self.sel < 0 and k in (Qt.Key_Home, Qt.Key_End, Qt.Key_PageUp, Qt.Key_PageDown):
            page = self.height() * 0.8
            self.scroll = {Qt.Key_Home: 0.0, Qt.Key_End: self.content_h, Qt.Key_PageUp: self.scroll - page, Qt.Key_PageDown: self.scroll + page}[k]
            self.scroll = min(max(0.0, self.scroll), max(0.0, self.content_h - self.height()))
            self.update()
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e):
        self._relayout()
        self._resize_timer.start()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(EXT) for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if f.lower().endswith(EXT):
                self.load(f)
                break

    def closeEvent(self, e):
        self.gen += 1
        if self.work is not None:
            self.work.cancelled = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._lpool.shutdown(wait=False, cancel_futures=True)
        if self.reader is not None:
            self.reader.close()
        super().closeEvent(e)


# --------------------------------------------------------------------------- start
def selftest(out_path: str) -> int:
    """Used by the build: make a tiny show with the show-file writer, open it and draw it (nothing needs the tracer)."""
    import tempfile
    from .showfile import ShowWriter
    lines: List[str] = []
    try:
        d = tempfile.mkdtemp(prefix="photovec_viewer_selftest_")
        path = os.path.join(d, "t" + EXT)
        rng = np.random.default_rng(1)
        wr = ShowWriter(path)
        infos = []
        for i in range(3):
            photo = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
            mask = np.zeros((90, 120), np.uint8)
            mask[20:70, 30:90] = 255
            wr.add_photo(i, photo, mask)
            for k in range(4):
                wr.add_step(i, k, np.full((90, 120, 3), 40 * k, np.uint8), np.full((90, 120), 255, np.uint8))
            infos.append(PhotoInfo(w=120, h=90, steps=4, masked=True, has_ink=True, letters=(i == 1)))
        from .letters import LetterMorph
        sq = lambda x, y, r: np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]], np.float32)
        wr.add_letters(1, [LetterMorph("T", np.array([[30, 40], [60, 40], [90, 40]], np.float32), np.array([[30, 60], [60, 60], [90, 60]], np.float32),
                                       np.full(3, 4, np.float32), np.full(3, 5, np.float32), [(0, 3, False)], [sq(60, 40, 30)], [sq(60, 60, 30)], 12.0)])
        wr.finish(infos, "selftest", ("One", "Two", "Three"))
        app = QApplication.instance() or QApplication([])
        v = Viewer()
        v.resize(800, 600)
        v.show()
        assert v.load(path)
        app.processEvents()
        v.open_photo(1)
        import time
        for _ in range(60):
            app.processEvents()
            QTimer.singleShot(0, lambda: None)
            time.sleep(0.02)
        img = v.grab().toImage()
        lines.append(f"grabbed {img.width()}x{img.height()} opened={v.p:.2f} steps={len(v.work.steps) if v.work else -1}")
        assert v.p == 1.0 and v.work is not None and v.work.photo is not None
        assert v.work.letters is not None and v._rows() == 3
        v._set(2, 0.6)                                                       # the third slider: the letters move (computed in a background thread)
        for _ in range(100):
            app.processEvents()
            time.sleep(0.02)
            if any(k[3] == 60 for k in v._cache):
                break
        assert any(k[3] == 60 for k in v._cache), "the letters did not move"
        lines.append("letters moved")
        lines.append("SELFTEST OK")
        rc = 0
    except BaseException as exc:
        import traceback
        lines.append("SELFTEST FAILED: " + "".join(traceback.format_exception(exc)))
        rc = 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return rc


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:
        i = argv.index("--selftest")
        return selftest(argv[i + 1] if len(argv) > i + 1 else "selftest.txt")
    app = QApplication(argv)
    app.setApplicationName("PhotoVec Viewer")
    w = Viewer()
    w.show()
    target = next((a for a in argv[1:] if a.lower().endswith(EXT) and os.path.isfile(a)), "")
    if target:                                                         # (a show file given on the command line / double-clicked)
        QTimer.singleShot(0, lambda: w.load(target))
    else:
        QTimer.singleShot(150, w.ask_at_start)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
