"""PhotoVec Show Maker: prepare the photos for the Viewer.

For every photo you set how far the Threshold slider goes (lowest and highest value), draw a mask (the part that becomes vector), and
how the background slider works. A live preview shows exactly what the Viewer will show. "Render show" traces every photo at all its
Threshold steps and saves ONE file (`.photovecshow`) that the Viewer opens. The sliders get the names you type here.

Letters: "Mark letters…" opens the letter editor (letter_editor.py): you draw a skeleton on every letter of a tag with a pen and say which
character it is. The third slider then moves the letters from the drawing into a line of text in the font you choose (letters.py).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QDoubleSpinBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
                               QFrame)

from . import multi_project
from .captions import paint_caption, register_font_file, resolve_family
from .letters import Letter, LetterSet, Stroke
from .range_slider import RangeSlider
from .showfile import DEFAULT_LABELS, EXT, MOTIONS, CaptionStyle, compose, step_at
from .showrender import (ShowPhoto, ShowSettings, build_source, check_letters, fit_photo_letters, photo_info, reference_ink, render_show,
                         render_step)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif")
PROJECT_EXT = ".photovec-showmaker"
APP = "photovec-show-maker"
PREVIEW_TILE = 640
MOTION_NAMES = {"melt": "Melt together (all at once)", "wave": "Wave (one after the other, left to right)"}
PRESET_CHOICES = ("Black and White Logo", "Silhouettes", "Sketched Art", "Line Art", "Technical Drawing", "Shades of Gray", "3 Colors", "6 Colors")


def _qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()


def _project_letters(d: dict) -> dict:
    """The letters of a saved panel: they are used when the checkbox was on (projects from before the checkbox had no key: used when present)."""
    letters = [Letter.from_dict(x) for x in d.get("letters", [])]
    on = bool(d.get("letters_on", bool(letters)))
    return {"letters": letters if on else [], "letters_keep": [] if on else letters, "letters_on": on}


class ImageView(QWidget):
    """A picture, as large as fits, on a neutral background."""

    def __init__(self):
        super().__init__()
        self.img: Optional[QImage] = None
        self.note = ""
        self.caption = ""                                       # the photo's name, drawn on the picture exactly as the Viewer will
        self.caption_style = CaptionStyle()
        self.caption_opacity = 1.0                              # (0..1) from the link to slider 1
        self.setMinimumSize(360, 260)

    def set_image(self, img: Optional[QImage], note: str = "") -> None:
        self.img, self.note = img, note
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(44, 44, 48))
        if self.img is not None:
            r = QRectF(self.rect()).adjusted(8, 8, -8, -8)
            s = min(r.width() / self.img.width(), r.height() / self.img.height())
            w, h = self.img.width() * s, self.img.height() * s
            pic = QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)
            p.drawImage(pic, self.img)
            if self.caption:
                p.save()
                p.setClipRect(pic)
                paint_caption(p, pic, self.caption, self.caption_style, self.caption_opacity)
                p.restore()
        if self.note:
            p.setPen(QColor(200, 200, 205))
            p.drawText(self.rect().adjusted(10, 0, -10, -8), Qt.AlignHCenter | Qt.AlignBottom, self.note)


class RenderJob(QThread):
    progress = Signal(str, int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, photos: List[ShowPhoto], ss: ShowSettings, path: str):
        super().__init__()
        self.photos, self.ss, self.path = photos, ss, path
        self._cancel = False
        self.notes: List[str] = []

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from . import pool as pool_mod
            from .batch import resolve_workers, shared_pool
            nw = resolve_workers(0)
            with pool_mod.busy():
                pool = shared_pool(nw) if nw > 1 else None
                ok = render_show(self.photos, self.ss, self.path, lambda t, i, n: (self.progress.emit(t, i, n), self._cancel)[1], nw, pool, self.notes)
            if ok and not self._cancel:
                self.done.emit(self.path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def viewer_command(path: str) -> Optional[List[str]]:
    """How to start the Viewer: its own program next to this one (built version), or the Python module (from the source)."""
    if getattr(sys, "frozen", False):
        exe = os.path.join(os.path.dirname(sys.executable), "PhotoVecViewer" + (".exe" if os.name == "nt" else ""))
        return [exe, path] if os.path.isfile(exe) else None
    return [sys.executable, "-m", "photovec.viewer", path]


class ShowMaker(QWidget):
    _preview_ready = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoVec Show Maker")
        self.resize(1320, 860)
        self.setAcceptDrops(True)
        self.ai = None
        self.job: Optional[RenderJob] = None
        self.last_show = ""
        self.project_file = ""
        self._temp_dirs: List[str] = []
        self._busy = False                                               # widgets are being filled (no edits)
        self._finish_rev = 0
        self._pgen = 0
        self._psrc: Dict[tuple, tuple] = {}                              # preview sources (key -> (source, per-source cache))
        self._pfit: Dict[tuple, object] = {}                             # fitted letters for the preview: key -> LetterSet or an error text
        self._pvec: Dict[tuple, tuple] = {}                              # rendered steps: key -> (vec, ink, paper)
        self._plock = threading.Lock()
        self._pexec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="showmaker")
        self._ptimer = QTimer(self)
        self._ptimer.setSingleShot(True)
        self._ptimer.setInterval(60)
        self._ptimer.timeout.connect(self._preview_now)
        self._preview_ready.connect(self._on_preview_ready)
        self._build()
        self._enable()

    # ---------------------------------------------------------------- UI
    def _build(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.add_btn = QPushButton("Add photos…")
        self.add_btn.clicked.connect(self.add_dialog)
        self.rem_btn = QPushButton("Remove")
        self.rem_btn.clicked.connect(self.remove_current)
        self.up_btn, self.dn_btn = QPushButton("▲"), QPushButton("▼")
        self.up_btn.setToolTip("Move the photo up (earlier in the grid)")
        self.dn_btn.setToolTip("Move the photo down (later in the grid)")
        self.up_btn.clicked.connect(lambda: self.move_current(-1))
        self.dn_btn.clicked.connect(lambda: self.move_current(1))
        for b in (self.add_btn, self.rem_btn, self.up_btn, self.dn_btn):
            top.addWidget(b)
        top.addStretch(1)
        top.addWidget(QLabel("Slider 1 name"))
        self.name1 = QLineEdit(DEFAULT_LABELS[0])
        self.name2 = QLineEdit(DEFAULT_LABELS[1])
        self.name3 = QLineEdit(DEFAULT_LABELS[2])
        for e in (self.name1, self.name2, self.name3):
            e.setMaxLength(40)
            e.setFixedWidth(150)
            e.textChanged.connect(self._names_changed)
        top.addWidget(self.name1)
        top.addWidget(QLabel("Slider 2 name"))
        top.addWidget(self.name2)
        self.name3_label = QLabel("Slider 3 name")
        top.addWidget(self.name3_label)
        top.addWidget(self.name3)
        top.addWidget(QLabel("Show title"))
        self.title = QLineEdit()
        self.title.setPlaceholderText("(window title in the Viewer)")
        self.title.setFixedWidth(200)
        top.addWidget(self.title)
        root.addLayout(top)

        mid = QHBoxLayout()
        self.list = QListWidget()
        self.list.setIconSize(self._icon_size())
        self.list.setFixedWidth(210)
        self.list.setDragDropMode(QListWidget.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.currentRowChanged.connect(self._select)
        self.list.model().rowsMoved.connect(lambda *_: self._ptimer.start())
        mid.addWidget(self.list)

        centre = QVBoxLayout()
        self.view = ImageView()
        centre.addWidget(self.view, 1)
        self.s1_label, self.s2_label, self.s3_label = QLabel(DEFAULT_LABELS[0]), QLabel(DEFAULT_LABELS[1]), QLabel(DEFAULT_LABELS[2])
        self.s1, self.s2, self.s3 = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
        self.s3_row = QWidget()
        for sl in (self.s1, self.s2, self.s3):
            sl.setRange(0, 1000)
            sl.valueChanged.connect(self._sliders_changed)
        for lab, sl in ((self.s1_label, self.s1), (self.s2_label, self.s2), (self.s3_label, self.s3)):
            row = QHBoxLayout()
            lab.setFixedWidth(150)
            row.addWidget(lab)
            row.addWidget(sl, 1)
            if sl is self.s3:
                row.setContentsMargins(0, 0, 0, 0)
                self.s3_row.setLayout(row)
                centre.addWidget(self.s3_row)
            else:
                centre.addLayout(row)
        self.s3_row.setVisible(False)
        hint = QLabel("These are the Viewer's sliders: where you leave them is where the photo starts in the Viewer. The third one appears when the photo has marked letters.")
        hint.setStyleSheet("color: gray")
        hint.setWordWrap(True)
        centre.addWidget(hint)
        mid.addLayout(centre, 1)

        right = QVBoxLayout()
        box = QGroupBox("This photo")
        form = QFormLayout(box)
        self.caption = QLineEdit()
        self.caption.setMaxLength(120)
        self.caption.setPlaceholderText("(no name)")
        self.caption.setToolTip("The name of this photo. The Viewer shows it on the picture like a film subtitle: yellow letters with a black outline.")
        self.caption.textChanged.connect(self._caption_changed)
        form.addRow("Name on the picture", self.caption)
        self.thr = RangeSlider(0, 255, 40, 220)
        self.thr.changed.connect(self._thr_changed)
        self.thr_text = QLabel()
        form.addRow("Threshold range", self.thr)
        form.addRow("", self.thr_text)
        self.steps = QSpinBox()
        self.steps.setRange(8, 240)
        self.steps.setValue(40)
        self.steps.setToolTip("How many pictures slider 1 has. More = smoother, but a longer render and a bigger file.")
        form.addRow("Pictures (steps)", self.steps)
        self.noise = QSpinBox()
        self.noise.setRange(1, 100)
        self.noise.setValue(25)
        form.addRow("Noise", self.noise)
        self.preset = QComboBox()
        self.preset.addItems(PRESET_CHOICES)
        form.addRow("Style", self.preset)
        self.bg_min, self.bg_max = QSpinBox(), QSpinBox()
        for s, v in ((self.bg_min, 0), (self.bg_max, 100)):
            s.setRange(0, 100)
            s.setValue(v)
            s.setSuffix(" %")
        bgrow = QHBoxLayout()
        bgrow.addWidget(self.bg_min)
        bgrow.addWidget(QLabel("→"))
        bgrow.addWidget(self.bg_max)
        form.addRow("Slider 2 goes from", bgrow)
        self.blur = QSpinBox()
        self.blur.setRange(0, 100)
        self.blur.setSuffix(" %")
        self.blur.setToolTip("How blurred the background is when slider 2 is at its left end (it fades and blurs together).")
        form.addRow("Blur at the left end", self.blur)
        self.mask_btn = QPushButton("Draw mask…")
        self.mask_btn.clicked.connect(self.edit_mask)
        self.mask_clear = QPushButton("Remove mask")
        self.mask_clear.clicked.connect(lambda: self.set_mask(None))
        mrow = QHBoxLayout()
        mrow.addWidget(self.mask_btn)
        mrow.addWidget(self.mask_clear)
        form.addRow("Mask", mrow)
        self.mask_note = QLabel()
        self.mask_note.setWordWrap(True)
        self.mask_note.setStyleSheet("color: gray")
        form.addRow("", self.mask_note)
        for w in (self.steps, self.noise, self.bg_min, self.bg_max, self.blur):
            w.valueChanged.connect(self._settings_changed)
        self.preset.currentTextChanged.connect(self._settings_changed)
        right.addWidget(box)

        lbox = QGroupBox("Letters → font (slider 3)")
        ll = QFormLayout(lbox)
        self.letters_on = QCheckBox("Use letters → font on this photo (adds a third slider)")
        self.letters_on.setToolTip("Off: the photo has two sliders and the marked letters (if any) are kept but not used.")
        self.letters_on.toggled.connect(self._letters_toggled)
        ll.addRow(self.letters_on)
        self.letters_btn = QPushButton("Mark letters…")
        self.letters_btn.setToolTip("Draw a skeleton on every letter of the tag and say which character it is")
        self.letters_btn.clicked.connect(self.edit_letters)
        self.letters_clear = QPushButton("Remove letters")
        self.letters_clear.clicked.connect(self.clear_letters)
        lrow = QHBoxLayout()
        lrow.addWidget(self.letters_btn)
        lrow.addWidget(self.letters_clear)
        ll.addRow(lrow)
        self.letters_note = QLabel("No letters marked: this photo has no third slider.")
        self.letters_note.setWordWrap(True)
        self.letters_note.setStyleSheet("color: gray")
        ll.addRow(self.letters_note)
        self.font_combo = QComboBox()
        self.font_combo.setToolTip("The font the letters arrive in at the right end of slider 3. Futura, Consolas, OCR-B and MS Gothic need to be installed; "
                                   "a free look-alike stands in when they are not.")
        self._fill_fonts()
        self.font_file_btn = QPushButton("Load font file…")
        self.font_file_btn.clicked.connect(self.load_font_file)
        frow = QHBoxLayout()
        frow.addWidget(self.font_combo, 1)
        frow.addWidget(self.font_file_btn)
        ll.addRow("Font", frow)
        self.font_note = QLabel("")
        self.font_note.setWordWrap(True)
        self.font_note.setStyleSheet("color: #b36b00")
        ll.addRow(self.font_note)
        self.motion = QComboBox()
        for k in MOTIONS:
            self.motion.addItem(MOTION_NAMES[k], k)
        ll.addRow("Motion", self.motion)
        self.reach = QSpinBox()
        self.reach.setRange(50, 300)
        self.reach.setValue(100)
        self.reach.setSuffix(" %")
        self.reach.setToolTip("How far around a letter's skeleton the ink still belongs to that letter (100 = automatic). Raise it if flourishes stay behind.")
        ll.addRow("Reach", self.reach)
        self.font_combo.currentIndexChanged.connect(self._letters_changed)
        self.motion.currentIndexChanged.connect(self._letters_changed)
        self.reach.valueChanged.connect(self._letters_changed)
        right.addWidget(lbox)

        cbox = QGroupBox("Name style (all photos)")
        cl = QFormLayout(cbox)
        self.cap_font = QComboBox()
        self.cap_font.setToolTip("The font of the names on the pictures. The default is Helvetica Neue Medium Italic when the computer has it, otherwise Inter.")
        self._cap_files: Dict[str, str] = {}                              # family -> font file (fonts loaded with the button)
        self._fill_cap_fonts()
        self.cap_font_btn = QPushButton("Load font file…")
        self.cap_font_btn.clicked.connect(self.load_cap_font)
        crow = QHBoxLayout()
        crow.addWidget(self.cap_font, 1)
        crow.addWidget(self.cap_font_btn)
        cl.addRow("Font", crow)
        self.cap_bold = QCheckBox("Bold")
        self.cap_italic = QCheckBox("Italic")
        self.cap_italic.setChecked(True)
        brow = QHBoxLayout()
        brow.addWidget(self.cap_bold)
        brow.addWidget(self.cap_italic)
        brow.addStretch(1)
        cl.addRow("Style", brow)
        self.cap_size = QDoubleSpinBox()
        self.cap_size.setRange(1.5, 20.0)
        self.cap_size.setSingleStep(0.2)
        self.cap_size.setDecimals(1)
        self.cap_size.setValue(5.2)
        self.cap_size.setSuffix(" % of the picture height")
        cl.addRow("Size", self.cap_size)
        self.cap_color, self.cap_outline_color = QPushButton(), QPushButton()
        self._cap_colors = {"color": "#FFE100", "outline_color": "#000000"}
        self.cap_color.clicked.connect(lambda: self._pick_cap_color("color"))
        self.cap_outline_color.clicked.connect(lambda: self._pick_cap_color("outline_color"))
        for b in (self.cap_color, self.cap_outline_color):
            b.setMinimumWidth(130)
        corow = QHBoxLayout()
        corow.addWidget(QLabel("Letters"))
        corow.addWidget(self.cap_color)
        corow.addWidget(QLabel("Outline"))
        corow.addWidget(self.cap_outline_color)
        corow.addStretch(1)
        cl.addRow("Colours", corow)
        self.cap_outline = QSpinBox()
        self.cap_outline.setRange(0, 60)
        self.cap_outline.setValue(21)
        self.cap_outline.setSuffix(" % of the letter size  (0 = no outline)")
        cl.addRow("Outline width", self.cap_outline)
        self.cap_link = QCheckBox("Link the visibility to slider 1 (Threshold)")
        self.cap_link.setToolTip("The name fades in and out with the first slider. Position = percent along the slider (0 = left end, 100 = right end).")
        cl.addRow(self.cap_link)
        self.cap_a, self.cap_a_op, self.cap_b, self.cap_b_op = (QSpinBox() for _ in range(4))
        for sp, lo, hi, v, suf in ((self.cap_a, 0, 100, 0, " %"), (self.cap_a_op, 0, 100, 1, " % visible"),
                                   (self.cap_b, 0, 100, 80, " %"), (self.cap_b_op, 0, 100, 100, " % visible")):
            sp.setRange(lo, hi)
            sp.setValue(v)
            sp.setSuffix(suf)
        row_a, row_b = QHBoxLayout(), QHBoxLayout()
        row_a.addWidget(self.cap_a)
        row_a.addWidget(QLabel("→"))
        row_a.addWidget(self.cap_a_op, 1)
        row_b.addWidget(self.cap_b)
        row_b.addWidget(QLabel("→"))
        row_b.addWidget(self.cap_b_op, 1)
        cl.addRow("At slider position", row_a)
        cl.addRow("and at", row_b)
        self.cap_link_note = QLabel("In between the visibility follows the slider in a straight line; beyond the two positions it stays at the nearer value.")
        self.cap_link_note.setWordWrap(True)
        self.cap_link_note.setStyleSheet("color: gray")
        cl.addRow(self.cap_link_note)
        self.cap_embed = QCheckBox("Put the font file into the show")
        self.cap_embed.setToolTip("Then the names look the same on every computer. Only tick it when the font's licence allows passing it on.")
        cl.addRow(self.cap_embed)
        self.cap_note = QLabel("")
        self.cap_note.setWordWrap(True)
        self.cap_note.setStyleSheet("color: #b36b00")
        cl.addRow(self.cap_note)
        self.cap_reset = QPushButton("Back to the subtitle look")
        self.cap_reset.clicked.connect(lambda: self._set_cap_style(CaptionStyle()))
        cl.addRow(self.cap_reset)
        for sig in (self.cap_font.currentIndexChanged, self.cap_bold.toggled, self.cap_italic.toggled, self.cap_size.valueChanged,
                    self.cap_outline.valueChanged, self.cap_embed.toggled, self.cap_link.toggled, self.cap_a.valueChanged, self.cap_a_op.valueChanged,
                    self.cap_b.valueChanged, self.cap_b_op.valueChanged):
            sig.connect(self._cap_changed)
        right.addWidget(cbox)
        self._paint_cap_buttons()

        fin = QGroupBox("Mask edge (all photos)")
        fl = QFormLayout(fin)
        self.m_vector = QCheckBox("Vector outline (sharp at any size)")
        self.m_vector.setChecked(True)
        self.m_smooth = QSpinBox()
        self.m_smooth.setRange(0, 100)
        self.m_soft = QSpinBox()
        self.m_soft.setRange(0, 100)
        fl.addRow(self.m_vector)
        fl.addRow("Smooth", self.m_smooth)
        fl.addRow("Soft edge", self.m_soft)
        for w in (self.m_smooth, self.m_soft):
            w.valueChanged.connect(self._finish_changed)
        self.m_vector.toggled.connect(self._finish_changed)
        right.addWidget(fin)
        right.addStretch(1)
        side = QWidget()                                                    # (the settings can be taller than a small screen: they scroll)
        side.setLayout(right)
        right.setContentsMargins(0, 0, 6, 0)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(max(side.sizeHint().width(), side.minimumSizeHint().width(), 500) + scroll.verticalScrollBar().sizeHint().width() + 8)
        mid.addWidget(scroll)
        root.addLayout(mid, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Quality"))
        self.quality = QComboBox()
        for label, v in (("1200 px (fast)", 1200), ("1600 px", 1600), ("2000 px", 2000), ("2400 px (large file)", 2400)):
            self.quality.addItem(label, v)
        self.quality.setCurrentIndex(1)
        bottom.addWidget(self.quality)
        bottom.addSpacing(18)
        self.open_proj_btn = QPushButton("Open project…")
        self.save_proj_btn = QPushButton("Save project…")
        self.open_proj_btn.clicked.connect(self.open_project_dialog)
        self.save_proj_btn.clicked.connect(self.save_project_dialog)
        bottom.addWidget(self.open_proj_btn)
        bottom.addWidget(self.save_proj_btn)
        bottom.addStretch(1)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.bar.setFixedWidth(260)
        bottom.addWidget(self.bar)
        self.status = QLabel("Add the photos of your show.")
        bottom.addWidget(self.status, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.view_btn = QPushButton("Open in Viewer")
        self.view_btn.clicked.connect(self.open_in_viewer)
        self.render_btn = QPushButton("Render show…")
        self.render_btn.setDefault(True)
        self.render_btn.clicked.connect(self.render_dialog)
        for b in (self.cancel_btn, self.view_btn, self.render_btn):
            bottom.addWidget(b)
        root.addLayout(bottom)

    def _icon_size(self):
        from PySide6.QtCore import QSize
        return QSize(88, 66)

    # ---------------------------------------------------------------- photos
    def photos(self) -> List[ShowPhoto]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def current(self) -> Optional[ShowPhoto]:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    def add_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos", str(QSettings("PhotoVec", "ShowMaker").value("dir", "")),
                                                "Images (" + " ".join("*" + e for e in IMG_EXT) + ")")
        if files:
            try:
                QSettings("PhotoVec", "ShowMaker").setValue("dir", os.path.dirname(files[0]))
            except Exception:
                pass
            self.add_photos(files)

    def add_photos(self, files: List[str], settings: Optional[List[ShowPhoto]] = None) -> None:
        from .imageio_ import load_image_alpha
        from .rows import fit_size, image_size
        first_new = self.list.count()
        for n, f in enumerate(files):
            p = settings[n] if settings else ShowPhoto(path=f)
            try:
                iw, ih = image_size(f)
                tw, th = fit_size("fit_long", 160, 160, iw / max(1, ih))
                rgb, _ = load_image_alpha(f, (tw, th))
                icon = QIcon(QPixmap.fromImage(_qimage(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))))
            except Exception as exc:
                self.status.setText(f"Could not read {os.path.basename(f)}: {exc}")
                continue
            it = QListWidgetItem(icon, os.path.basename(f))
            it.setData(Qt.UserRole, p)
            self.list.addItem(it)
        if self.list.count() > first_new:
            self.list.setCurrentRow(first_new)
        self._enable()

    def remove_current(self) -> None:
        r = self.list.currentRow()
        if r >= 0:
            self.list.takeItem(r)
            self._enable()
            self._select(self.list.currentRow())

    def move_current(self, d: int) -> None:
        r = self.list.currentRow()
        n = r + d
        if 0 <= r < self.list.count() and 0 <= n < self.list.count():
            it = self.list.takeItem(r)
            self.list.insertItem(n, it)
            self.list.setCurrentRow(n)

    def _enable(self) -> None:
        has = self.list.count() > 0
        busy = self.job is not None
        for w in (self.rem_btn, self.up_btn, self.dn_btn):
            w.setEnabled(has and not busy)
        self.render_btn.setEnabled(has and not busy)
        self.save_proj_btn.setEnabled(has and not busy)
        self.open_proj_btn.setEnabled(not busy)
        self.add_btn.setEnabled(not busy)
        self.view_btn.setVisible(bool(self.last_show) and os.path.isfile(self.last_show))
        self._letters_ui()

    # ---------------------------------------------------------------- one photo's settings <-> widgets
    def _select(self, row: int) -> None:
        p = self.current()
        self._busy = True
        try:
            for w in (self.thr, self.steps, self.noise, self.preset, self.bg_min, self.bg_max, self.blur, self.s1, self.s2, self.s3, self.mask_btn,
                      self.mask_clear, self.caption):
                w.setEnabled(p is not None)
            if p is not None:
                self.thr.set_values(int(round(p.thr_min)), int(round(p.thr_max)))
                self.steps.setValue(int(p.steps))
                self.noise.setValue(int(round(p.noise)))
                self.preset.setCurrentText(p.preset if p.preset in PRESET_CHOICES else PRESET_CHOICES[0])
                self.bg_min.setValue(int(round(p.bg_min)))
                self.bg_max.setValue(int(round(p.bg_max)))
                self.blur.setValue(int(round(p.blur)))
                self.s1.setValue(int(round(p.start1 * 1000)))
                self.s2.setValue(int(round(p.start2 * 1000)))
                self.s3.setValue(int(round(p.start3 * 1000)))
                self.caption.setText(p.caption)
                self._set_font_combo(p.font)
                self.motion.setCurrentIndex(max(0, self.motion.findData(p.lmotion)))
                self.reach.setValue(int(round(p.reach * 100)))
        finally:
            self._busy = False
        if p is None:
            self.caption.setText("")
        self.view.caption = p.caption if p is not None else ""
        self._refresh_caption_opacity()
        self._letters_ui()
        self._mask_ui()
        self._thr_text()
        self._ptimer.start()

    def _refresh_caption_opacity(self) -> None:
        """The preview shows the name as visible as the Viewer will show it with slider 1 where it stands now."""
        p = self.current()
        op = 1.0
        if p is not None:
            pos = step_at(photo_info(p, 1, 1), p.start1) / float(max(1, p.steps - 1))
            op = self.view.caption_style.opacity(pos)
        self.view.caption_opacity = op
        self.view.update()

    def _caption_changed(self, text: str) -> None:
        p = self.current()
        if p is None or self._busy:
            return
        p.caption = text
        self.view.caption = text
        self.view.update()

    def _mask_ui(self) -> None:
        p = self.current()
        has = p is not None and p.mask is not None
        self.mask_clear.setEnabled(has)
        self.blur.setEnabled(has)
        self.mask_btn.setText("Edit mask…" if has else "Draw mask…")
        self.mask_note.setText("Only the masked part is traced. Slider 2 fades the photo around it." if has else
                               "No mask: the whole photo is traced, and slider 2 fades the drawing into the original photo.")

    def _thr_text(self) -> None:
        a, b = self.thr.values()[:2]
        self.thr_text.setText(f"lowest {a}  →  highest {b}   (slider 1 sweeps through this range)")

    def _thr_changed(self) -> None:
        if not self._busy:
            self._thr_text()
            self._settings_changed()

    def _settings_changed(self, *_) -> None:
        p = self.current()
        if p is None or self._busy:
            return
        a, b = sorted(self.thr.values()[:2])
        if b - a < 1:
            b = min(255, a + 1)
            a = b - 1
        p.thr_min, p.thr_max = float(a), float(b)
        p.steps, p.noise = int(self.steps.value()), float(self.noise.value())
        self._refresh_caption_opacity()
        if p.preset != self.preset.currentText():
            p.preset = self.preset.currentText()
            p.rev += 1
        p.bg_min, p.bg_max, p.blur = float(self.bg_min.value()), float(self.bg_max.value()), float(self.blur.value())
        self._ptimer.start()

    def _sliders_changed(self) -> None:
        p = self.current()
        if p is not None and not self._busy:
            p.start1, p.start2, p.start3 = self.s1.value() / 1000.0, self.s2.value() / 1000.0, self.s3.value() / 1000.0
            self._refresh_caption_opacity()
            self._ptimer.start()

    def _names_changed(self) -> None:
        self.s1_label.setText(self.name1.text() or DEFAULT_LABELS[0])
        self.s2_label.setText(self.name2.text() or DEFAULT_LABELS[1])
        self.s3_label.setText(self.name3.text() or DEFAULT_LABELS[2])

    def _finish_changed(self, *_) -> None:
        self._finish_rev += 1
        self._ptimer.start()

    def settings(self) -> ShowSettings:
        return ShowSettings(title=self.title.text().strip(), labels=(self.name1.text().strip() or DEFAULT_LABELS[0], self.name2.text().strip() or DEFAULT_LABELS[1],
                                                                      self.name3.text().strip() or DEFAULT_LABELS[2]),
                            quality=int(self.quality.currentData()), mask_smooth=float(self.m_smooth.value()), mask_soft=float(self.m_soft.value()),
                            mask_vector=self.m_vector.isChecked(), caption_style=self._cap_style(), caption_font_file=self._cap_embed_file())

    # ---------------------------------------------------------------- the style of the names
    def _fill_cap_fonts(self) -> None:
        self.cap_font.blockSignals(True)
        self.cap_font.clear()
        self.cap_font.addItem("Default (Helvetica Neue / Inter)", "")
        self.cap_font.insertSeparator(1)
        try:
            for fam in QFontDatabase.families():
                if not fam.startswith("."):
                    self.cap_font.addItem(fam, fam)
        except Exception as exc:
            print("Show Maker: could not list the fonts:", exc)
        self.cap_font.blockSignals(False)

    def _cap_style(self) -> CaptionStyle:
        return CaptionStyle(family=str(self.cap_font.currentData() or ""), bold=self.cap_bold.isChecked(), italic=self.cap_italic.isChecked(),
                            color=self._cap_colors["color"], outline_color=self._cap_colors["outline_color"], size=float(self.cap_size.value()),
                            outline=float(self.cap_outline.value()), link=self.cap_link.isChecked(), link_from=float(self.cap_a.value()),
                            link_from_opacity=float(self.cap_a_op.value()), link_to=float(self.cap_b.value()),
                            link_to_opacity=float(self.cap_b_op.value())).clamp()

    def _cap_embed_file(self) -> str:
        fam = str(self.cap_font.currentData() or "")
        return self._cap_files.get(fam, "") if (fam and self.cap_embed.isChecked()) else ""

    def _paint_cap_buttons(self) -> None:
        for btn, key in ((self.cap_color, "color"), (self.cap_outline_color, "outline_color")):
            c = QColor(self._cap_colors[key])
            fg = "#000000" if c.lightness() > 140 else "#ffffff"
            btn.setText(self._cap_colors[key].upper())
            btn.setStyleSheet(f"QPushButton {{ background-color: {self._cap_colors[key]}; color: {fg}; border: 1px solid #888; padding: 3px 8px; }}")

    def _pick_cap_color(self, key: str) -> None:
        c = QColorDialog.getColor(QColor(self._cap_colors[key]), self, "Letters" if key == "color" else "Outline")
        if c.isValid():
            self._cap_colors[key] = c.name().upper()
            self._paint_cap_buttons()
            self._cap_changed()

    def load_cap_font(self) -> None:
        start = str(QSettings("PhotoVec", "ShowMaker").value("fontdir", ""))
        path, _ = QFileDialog.getOpenFileName(self, "Load a font file", start, "Fonts (*.ttf *.otf)")
        if not path:
            return
        fam = register_font_file(path)
        if not fam:
            QMessageBox.warning(self, "Show Maker", "This font file cannot be read.")
            return
        try:
            QSettings("PhotoVec", "ShowMaker").setValue("fontdir", os.path.dirname(path))
        except Exception:
            pass
        self._select_cap_family(fam, path)

    def _select_cap_family(self, fam: str, path: str = "") -> None:
        if path:
            self._cap_files[fam] = path
        i = self.cap_font.findData(fam)
        if i < 0:                                                         # (a family that is not installed here is still kept: it may be on the other computer)
            self.cap_font.addItem(fam, fam)
            i = self.cap_font.count() - 1
        self.cap_font.setCurrentIndex(i)

    def _cap_changed(self, *_) -> None:
        if self._busy:
            return
        style = self._cap_style()
        self.view.caption_style = style
        for w in (self.cap_a, self.cap_a_op, self.cap_b, self.cap_b_op):
            w.setEnabled(style.link)
        self._refresh_caption_opacity()
        fam = style.family
        loaded = fam in self._cap_files
        self.cap_embed.setEnabled(loaded)
        if not fam:
            note = ""
        else:
            _, remark = resolve_family(style)
            note = remark or ("" if (loaded and self.cap_embed.isChecked()) else
                              ("Other computers need this font installed, or tick “Put the font file into the show”." if loaded else
                               "Other computers need this font installed (or they show Inter instead)."))
        self.cap_note.setText(note)

    def _set_cap_style(self, st: CaptionStyle, path: str = "", embed: bool = False) -> None:
        self._busy = True
        try:
            if st.family:
                if path and os.path.isfile(path):
                    fam = register_font_file(path)
                    if fam:
                        self._cap_files[fam] = path
                self._select_cap_family(st.family)
            else:
                self.cap_font.setCurrentIndex(0)
            self.cap_bold.setChecked(st.bold)
            self.cap_italic.setChecked(st.italic)
            self.cap_size.setValue(st.size)
            self.cap_outline.setValue(int(round(st.outline)))
            self.cap_embed.setChecked(embed and st.family in self._cap_files)
            self.cap_link.setChecked(st.link)
            self.cap_a.setValue(int(round(st.link_from)))
            self.cap_a_op.setValue(int(round(st.link_from_opacity)))
            self.cap_b.setValue(int(round(st.link_to)))
            self.cap_b_op.setValue(int(round(st.link_to_opacity)))
            self._cap_colors.update(color=st.color.upper(), outline_color=st.outline_color.upper())
            self._paint_cap_buttons()
        finally:
            self._busy = False
        self._cap_changed()

    # ---------------------------------------------------------------- mask
    def _get_ai(self):
        if self.ai is None:
            from .ai_client import AIClient
            self.ai = AIClient()
        return self.ai

    def edit_mask(self) -> None:
        from .imageio_ import load_image_alpha
        from .mask_editor import MAX_EDIT, MaskEditor
        from .rows import fit_size, image_size
        p = self.current()
        if p is None:
            return
        try:
            iw, ih = image_size(p.path)
            tw, th = fit_size("fit_long", MAX_EDIT, MAX_EDIT, iw / max(1, ih))
            rgb, _ = load_image_alpha(p.path, (tw, th))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self.status.setText(f"Could not read the photo: {exc}")
            return
        dlg = MaskEditor(bgr, p.mask, os.path.basename(p.path), self, ai_factory=self._get_ai)
        if dlg.exec() == QDialog.Accepted:
            m = dlg.result()
            self.set_mask(m if m is not None and m.any() else None)

    def set_mask(self, m: Optional[np.ndarray]) -> None:
        p = self.current()
        if p is None:
            return
        had = p.mask is not None
        p.mask = m
        p.rev += 1
        if (m is not None) != had:                                       # slider 2 means something else with / without a mask: start at the sensible end
            p.start2 = 1.0 if m is not None else 0.0
            self._busy = True
            self.s2.setValue(int(round(p.start2 * 1000)))
            self._busy = False
        self._mask_ui()
        self._ptimer.start()

    # ---------------------------------------------------------------- letters
    def _fill_fonts(self) -> None:
        from .letterfit import FONT_CHOICES, installed_families
        self.font_combo.blockSignals(True)
        self.font_combo.clear()
        for n in FONT_CHOICES:
            self.font_combo.addItem(n, n)
        self.font_combo.insertSeparator(self.font_combo.count())
        try:
            for fam in installed_families():
                if fam not in FONT_CHOICES:
                    self.font_combo.addItem(fam, fam)
        except Exception as exc:
            print("Show Maker: could not list the fonts:", exc)
        self.font_combo.blockSignals(False)

    def _set_font_combo(self, font: str) -> None:
        i = self.font_combo.findData(font)
        if i < 0:                                                         # a font file (or a family that is not installed here)
            self.font_combo.addItem(os.path.basename(font) if os.path.isfile(font) else font, font)
            i = self.font_combo.count() - 1
        self.font_combo.setCurrentIndex(i)

    def load_font_file(self) -> None:
        from .letterfit import register_font_file
        start = str(QSettings("PhotoVec", "ShowMaker").value("fontdir", ""))
        path, _ = QFileDialog.getOpenFileName(self, "Load a font file", start, "Fonts (*.ttf *.otf *.ttc)")
        if not path:
            return
        fam = register_font_file(path)
        if not fam:
            QMessageBox.warning(self, "Show Maker", "This font file cannot be read.")
            return
        try:
            QSettings("PhotoVec", "ShowMaker").setValue("fontdir", os.path.dirname(path))
        except Exception:
            pass
        self.font_combo.addItem(f"{fam}  ({os.path.basename(path)})", path)
        self.font_combo.setCurrentIndex(self.font_combo.count() - 1)

    def _update_font_note(self) -> None:
        from .letterfit import resolve_font
        name = self.font_combo.currentData() or ""
        try:
            _, note = resolve_font(name)
            self.font_note.setText(note)
            self.font_note.setStyleSheet("color: #b36b00")
        except Exception as exc:
            self.font_note.setText(str(exc))
            self.font_note.setStyleSheet("color: #c0392b")

    def _letters_ui(self) -> None:
        p = self.current()
        on = p is not None and p.letters_on
        has = on and bool(p.letters)
        self.letters_on.blockSignals(True)
        self.letters_on.setChecked(on)
        self.letters_on.setEnabled(p is not None)
        self.letters_on.blockSignals(False)
        self.s3_row.setVisible(has)
        self.letters_clear.setEnabled(has)
        self.letters_btn.setEnabled(on and self.job is None)
        for w in (self.font_combo, self.font_file_btn, self.motion, self.reach):
            w.setEnabled(has)
        self.letters_btn.setText("Edit letters…" if has else "Mark letters…")
        if not on:
            kept = len(p.letters_keep) if p is not None else 0
            self.letters_note.setText("Off: this photo has two sliders." + (f" {kept} marked letters are kept and come back when you switch it on." if kept else ""))
        elif has:
            chars = "".join(l.char or "?" for l in p.letters)
            bad = check_letters(p.letters)
            self.letters_note.setText(f"{len(p.letters)} letters marked: {chars}" + (f"\n⚠ {bad}" if bad else ""))
        else:
            self.letters_note.setText("No letters marked yet: this photo has no third slider.")
        if has:
            self._update_font_note()
        else:
            self.font_note.setText("")
        anyon = any(q.letters for q in self.photos())                                     # the slider-3 name is only asked for when somebody uses it
        self.name3.setVisible(anyon)
        self.name3_label.setVisible(anyon)

    def _letters_toggled(self, on: bool) -> None:
        p = self.current()
        if p is None or self._busy:
            return
        if on:
            p.letters_on = True
            p.letters, p.letters_keep = list(p.letters_keep), []
        else:
            p.letters_on = False
            p.letters_keep, p.letters = list(p.letters), []
            p.start3 = 0.0
            self._busy = True
            self.s3.setValue(0)
            self._busy = False
        p.rev += 1
        self._letters_ui()
        self._ptimer.start()

    def _letters_changed(self, *_) -> None:
        p = self.current()
        if p is None or self._busy:
            return
        p.font = self.font_combo.currentData() or p.font
        p.lmotion = self.motion.currentData() or "melt"
        p.reach = self.reach.value() / 100.0
        self._update_font_note()
        self._ptimer.start()

    def clear_letters(self) -> None:
        p = self.current()
        if p is not None and p.letters:
            p.letters = []
            p.letters_keep = []
            p.start3 = 0.0
            self._busy = True
            self.s3.setValue(0)
            self._busy = False
            self._letters_ui()
            self._ptimer.start()

    def edit_letters(self) -> None:
        from .imageio_ import load_image_alpha
        from .letter_editor import LetterEditor
        from .mask_editor import MAX_EDIT
        from .rows import fit_size, image_size
        p = self.current()
        if p is None:
            return
        try:
            iw, ih = image_size(p.path)
            tw, th = fit_size("fit_long", MAX_EDIT, MAX_EDIT, iw / max(1, ih))
            rgb, _ = load_image_alpha(p.path, (tw, th))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self.status.setText(f"Could not read the photo: {exc}")
            return
        snap = ShowPhoto(path=p.path, mask=p.mask, thr_min=p.thr_min, thr_max=p.thr_max, steps=p.steps, noise=p.noise, preset=p.preset,
                         bg_min=p.bg_min, bg_max=p.bg_max, blur=p.blur, start1=p.start1, start2=p.start2)
        ss = self.settings()

        def drawing():
            def job():
                src = build_source(snap, ss, 1400)
                k = step_at(photo_info(snap, 1, 1), snap.start1)
                vec, ink, paper = render_step(src, snap, k, {})
                return compose(photo_info(snap, src.bgr.shape[1], src.bgr.shape[0], paper, ink is not None), src.bgr, src.mask, vec, ink, snap.start2)
            return self._pexec.submit(job).result()                      # (the tracer is used by one thread at a time)
        dlg = LetterEditor(self, bgr, (iw, ih), p.letters, os.path.basename(p.path), p.font, drawing)
        if dlg.exec() == QDialog.Accepted:
            p.letters = dlg.result_letters()
            if not p.letters:
                p.start3 = 0.0
                self.s3.setValue(0)
            self._letters_ui()
            self._ptimer.start()

    def _lsig(self, p: ShowPhoto) -> tuple:
        return (json.dumps([l.to_dict() for l in p.letters], separators=(",", ":")), p.font, round(p.reach, 3))

    # ---------------------------------------------------------------- live preview (the same maths the Viewer uses)
    def _skey(self, p: ShowPhoto) -> tuple:
        return (p.path, p.rev, self._finish_rev, id(p))

    def _vkey(self, p: ShowPhoto, k: int) -> tuple:
        return self._skey(p) + (p.thr_min, p.thr_max, p.steps, p.noise, k)

    def _preview_now(self) -> None:
        p = self.current()
        if p is None:
            self.view.set_image(None, "")
            return
        k = step_at(photo_info(p, 1, 1), p.start1)
        key = self._vkey(p, k)
        fkey = (self._skey(p) + self._lsig(p)) if p.letters else None
        with self._plock:
            hit = self._pvec.get(key)
            fit = self._pfit.get(fkey) if fkey is not None else None
        if hit is not None and (fkey is None or fit is not None):
            self._show(p, hit, fit)
            return
        self.view.note = "Tracing…" if hit is None else "Fitting the letters…"
        self.view.update()
        self._pgen += 1
        gen = self._pgen
        ss = self.settings()
        snap = ShowPhoto(path=p.path, mask=p.mask, thr_min=p.thr_min, thr_max=p.thr_max, steps=p.steps, noise=p.noise, preset=p.preset,
                         letters=list(p.letters), font=p.font, reach=p.reach)

        def job():
            try:
                got = hit
                if got is None:
                    skey = self._skey(p)
                    with self._plock:
                        entry = self._psrc.get(skey)
                    if entry is None:
                        entry = (build_source(snap, ss, PREVIEW_TILE), {})
                        with self._plock:
                            if len(self._psrc) > 2:
                                self._psrc.clear()
                            self._psrc[skey] = entry
                    res = render_step(entry[0], snap, k, entry[1])
                    got = (entry[0].bgr, entry[0].mask, *res)
                    with self._plock:
                        if len(self._pvec) > 80:
                            self._pvec.clear()
                        self._pvec[key] = got
                if fkey is not None and fit is None:
                    from .rows import image_size
                    bgr, _, vec, ink, paper = got
                    size = (bgr.shape[1], bgr.shape[0])
                    try:
                        morphs, note = fit_photo_letters(snap, size, image_size(snap.path), reference_ink(vec, ink, paper))
                        ls = LetterSet(morphs, size, size)
                        ls.prepare()
                        result = (ls, note)
                    except Exception as exc:                          # (an unfinished letter, a character the font lacks, ...)
                        result = (None, str(exc))
                    with self._plock:
                        if len(self._pfit) > 8:
                            self._pfit.clear()
                        self._pfit[fkey] = result
            except Exception as exc:
                print("Show Maker: preview failed", exc)
                return
            self._preview_ready.emit(gen)
        self._pexec.submit(job)

    def _on_preview_ready(self, gen: int) -> None:
        if gen == self._pgen:
            self._preview_now()

    def _show(self, p: ShowPhoto, hit: tuple, fit=None) -> None:
        bgr, mask, vec, ink, paper = hit
        note = ""
        if fit is not None:
            ls, text = fit
            if ls is None:
                note = "Letters: " + text
            elif p.start3 > 0:
                vec, ink, mask = ls.apply(vec, ink, paper, p.start3, p.lmotion, mask)
        info = photo_info(p, bgr.shape[1], bgr.shape[0], paper, ink is not None)
        out = compose(info, bgr, mask, vec, ink, p.start2)
        self.view.set_image(_qimage(out), note)

    # ---------------------------------------------------------------- render
    def render_dialog(self) -> None:
        if not self.photos():
            return
        start = str(QSettings("PhotoVec", "ShowMaker").value("showdir", ""))
        name = (self.title.text().strip() or "show").replace("/", "_").replace("\\", "_")
        path, _ = QFileDialog.getSaveFileName(self, "Save the show", os.path.join(start, name + EXT), f"PhotoVec show (*{EXT})")
        if not path:
            return
        if not path.lower().endswith(EXT):
            path += EXT
        try:
            QSettings("PhotoVec", "ShowMaker").setValue("showdir", os.path.dirname(path))
        except Exception:
            pass
        self.render_to(path)

    def render_to(self, path: str) -> None:
        photos = self.photos()
        for i, p in enumerate(photos):
            if not os.path.isfile(p.path):
                QMessageBox.warning(self, "Show Maker", f"The photo {i + 1} cannot be read any more:\n{p.path}")
                return
        for i, p in enumerate(photos):
            bad = check_letters(p.letters) if p.letters else None
            if bad:
                QMessageBox.warning(self, "Show Maker", f"Photo {i + 1} ({os.path.basename(p.path)}): {bad}")
                return
        self.job = RenderJob(photos, self.settings(), path)
        self.job.progress.connect(self._progress)
        self.job.done.connect(self._rendered)
        self.job.failed.connect(self._failed)
        self.job.finished.connect(self._render_finished)
        self.cancel_btn.setVisible(True)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.status.setText("Starting…")
        self._enable()
        self.job.start()

    def _progress(self, text: str, i: int, n: int) -> None:
        self.bar.setRange(0, max(1, n))
        self.bar.setValue(i)
        self.status.setText(text)

    def _rendered(self, path: str) -> None:
        self.last_show = path
        mb = os.path.getsize(path) / 1e6
        notes = sorted(set(self.job.notes)) if self.job is not None else []
        self.status.setText(f"Saved {path}  ({mb:.1f} MB)" + ("   " + " ".join(notes) if notes else ""))

    def _failed(self, msg: str) -> None:
        self.status.setText("Render failed: " + msg)
        QMessageBox.warning(self, "Render failed", msg)

    def _cancel(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.status.setText("Cancelling…")

    def _render_finished(self) -> None:
        self.job = None
        self.bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._enable()

    def wait_render(self) -> None:
        if self.job is not None:
            self.job.wait()
            QApplication.processEvents()

    def open_in_viewer(self) -> None:
        cmd = viewer_command(self.last_show)
        if cmd is None:
            QMessageBox.information(self, "Show Maker", "The Viewer program (PhotoVecViewer) was not found next to this program.\nOpen the show file with it yourself.")
            return
        try:
            subprocess.Popen(cmd)
        except OSError as exc:
            QMessageBox.warning(self, "Show Maker", f"Could not start the Viewer: {exc}")

    # ---------------------------------------------------------------- projects
    def state(self) -> dict:
        s = self.settings()
        panels = []
        for p in self.photos():
            panels.append({"thr_min": p.thr_min, "thr_max": p.thr_max, "steps": p.steps, "noise": p.noise, "preset": p.preset,
                           "bg_min": p.bg_min, "bg_max": p.bg_max, "blur": p.blur, "start1": p.start1, "start2": p.start2,
                           "start3": p.start3, "font": p.font, "lmotion": p.lmotion, "reach": p.reach,
                           "letters": [l.to_dict() for l in (p.letters if p.letters_on else p.letters_keep)],
                           "letters_on": bool(p.letters_on), "caption": p.caption})
        return {"title": s.title, "labels": list(s.labels), "quality": s.quality, "mask_smooth": s.mask_smooth, "mask_soft": s.mask_soft,
                "mask_vector": s.mask_vector, "current": self.list.currentRow(), "panels": panels,
                "caption_style": asdict(s.caption_style), "caption_font_path": self._cap_files.get(s.caption_style.family, ""),
                "caption_embed": self.cap_embed.isChecked()}

    def save_project_dialog(self) -> None:
        start = self.project_file or os.path.join(str(QSettings("PhotoVec", "ShowMaker").value("showdir", "")), "show" + PROJECT_EXT)
        path, _ = QFileDialog.getSaveFileName(self, "Save the project", start, f"Show Maker project (*{PROJECT_EXT})")
        if path:
            self.save_project(path if path.lower().endswith(PROJECT_EXT) else path + PROJECT_EXT)

    def save_project(self, path: str) -> None:
        photos = self.photos()
        try:
            multi_project.write_project(path, self.state(), [p.path for p in photos], [p.mask for p in photos], app=APP)
        except Exception as exc:
            QMessageBox.warning(self, "Show Maker", f"Could not save the project: {exc}")
            return
        self.project_file = path
        self.status.setText(f"Project saved: {path}")

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a project", self.project_file, f"Show Maker project (*{PROJECT_EXT})")
        if path:
            self.open_project(path)

    def open_project(self, path: str) -> bool:
        try:
            tmp = tempfile.mkdtemp(prefix="photovec_showmaker_")
            self._temp_dirs.append(tmp)
            state, files = multi_project.read_project(path, tmp, app=APP)
            masks = multi_project.read_masks(path, state)
        except Exception as exc:
            QMessageBox.warning(self, "Show Maker", str(exc))
            return False
        self.list.clear()
        panels = state.get("panels", [])
        keep, settings = [], []
        for f, d, m in zip(files, panels, masks):
            if not f:
                continue
            keep.append(f)
            settings.append(ShowPhoto(path=f, mask=m, thr_min=float(d.get("thr_min", 0)), thr_max=float(d.get("thr_max", 255)), steps=int(d.get("steps", 40)),
                                      noise=float(d.get("noise", 25)), preset=str(d.get("preset", PRESET_CHOICES[0])), bg_min=float(d.get("bg_min", 0)),
                                      bg_max=float(d.get("bg_max", 100)), blur=float(d.get("blur", 0)), start1=float(d.get("start1", 0.5)),
                                      start2=float(d.get("start2", 1.0)), start3=float(d.get("start3", 0.0)), font=str(d.get("font", "Futura")),
                                      lmotion=str(d.get("lmotion", "melt")), reach=float(d.get("reach", 1.0)),
                                      caption=str(d.get("caption", "")), **_project_letters(d)))
        for w, v in ((self.m_smooth, state.get("mask_smooth", 0)), (self.m_soft, state.get("mask_soft", 0))):
            w.setValue(int(v))
        self.m_vector.setChecked(bool(state.get("mask_vector", True)))
        self._set_cap_style(CaptionStyle.from_dict(state.get("caption_style")), str(state.get("caption_font_path", "")), bool(state.get("caption_embed", False)))
        self.title.setText(str(state.get("title", "")))
        labs = state.get("labels") or list(DEFAULT_LABELS)
        self.name1.setText(str(labs[0]))
        self.name2.setText(str(labs[1]) if len(labs) > 1 else DEFAULT_LABELS[1])
        self.name3.setText(str(labs[2]) if len(labs) > 2 else DEFAULT_LABELS[2])
        i = self.quality.findData(int(state.get("quality", 1600)))
        self.quality.setCurrentIndex(i if i >= 0 else 1)
        self.add_photos(keep, settings)
        self.list.setCurrentRow(min(max(0, int(state.get("current", 0))), self.list.count() - 1))
        self.project_file = path
        self.status.setText(f"Project opened: {path}")
        return True

    # ---------------------------------------------------------------- drag & drop, close
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls()]
        proj = [f for f in files if f.lower().endswith(PROJECT_EXT)]
        if proj:
            self.open_project(proj[0])
            return
        imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
        if imgs:
            self.add_photos(imgs)

    def closeEvent(self, e):
        if self.job is not None:
            self.job.cancel()
            self.job.wait(5000)
        self._pexec.shutdown(wait=False, cancel_futures=True)
        import shutil
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(e)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:
        i = argv.index("--selftest")
        return selftest(argv[i + 1] if len(argv) > i + 1 else "selftest.txt")
    app = QApplication(argv)
    app.setApplicationName("PhotoVec Show Maker")
    w = ShowMaker()
    w.show()
    from .trace import warm_pool
    QTimer.singleShot(500, lambda: threading.Thread(target=warm_pool, daemon=True).start())
    files = [a for a in argv[1:] if os.path.isfile(a)]
    proj = [f for f in files if f.lower().endswith(PROJECT_EXT)]
    if proj:
        QTimer.singleShot(0, lambda: w.open_project(proj[0]))
    elif files:
        QTimer.singleShot(0, lambda: w.add_photos([f for f in files if f.lower().endswith(IMG_EXT)]))
    return app.exec()


def selftest(out_path: str) -> int:
    """Used by the build: render a tiny show from a made-up photo and read it back."""
    lines: List[str] = []
    try:
        from .showfile import ShowReader
        d = tempfile.mkdtemp(prefix="photovec_maker_selftest_")
        img = np.full((160, 200, 3), 200, np.uint8)
        cv2.circle(img, (100, 80), 50, (20, 20, 200), -1)
        cv2.imwrite(os.path.join(d, "a.png"), img)
        m = np.zeros((160, 200), np.uint8)
        m[30:130, 50:150] = 255
        ss = ShowSettings(title="selftest", quality=200)
        path = os.path.join(d, "t" + EXT)
        ok = render_show([ShowPhoto(os.path.join(d, "a.png"), m, steps=3)], ss, path)
        r = ShowReader(path)
        assert ok and len(r.photos) == 1 and r.photos[0].steps == 3 and r.vec(0, 2).shape[2] == 3
        lines.append("rendered and read back")
        # letters: a "T" marked on the circle, fitted to the (bundled look-alike of the) Consolas font
        QApplication.instance() or QApplication([])
        t = Letter("T", [Stroke([(60, 50), (140, 50)]), Stroke([(100, 50), (100, 120)])])
        notes: List[str] = []
        path2 = os.path.join(d, "l" + EXT)
        ok = render_show([ShowPhoto(os.path.join(d, "a.png"), m, steps=3, letters=[t], font="Consolas")], ss, path2, notes=notes)
        r2 = ShowReader(path2)
        ms = r2.letters(0)
        assert ok and r2.photos[0].letters and len(ms) == 1 and ms[0].char == "T" and len(ms[0].glyph1[0]) >= 4
        lines.append("letters fitted: " + (" ".join(notes) or "Consolas found"))
        lines.append("SELFTEST OK")
        rc = 0
    except BaseException as exc:
        import traceback
        lines.append("SELFTEST FAILED: " + "".join(traceback.format_exception(exc)))
        rc = 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
