# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: the Show Maker and the Viewer, side by side in one folder (they share one copy of the libraries).
# The program's own modules (the "photovec" folder) are copied next to the exe as plain .py files (not packed into the archive).
import os
from PyInstaller.utils.hooks import collect_all

datas, binaries = [], []
hiddenimports = ["PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore", "psutil"]
for pkg in ("skimage", "pillow_heif", "ezdxf", "imageio_ffmpeg", "svgelements"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

SCRIPTS = [("photovec_show_maker.py", "PhotoVecShowMaker"), ("photovec_viewer.py", "PhotoVecViewer")]


def analyse(script):
    light = script == "photovec_viewer.py"         # the Viewer needs no tracer, so it does not pull in the heavy libraries
    return Analysis([script], pathex=["."], binaries=[] if light else binaries, datas=[] if light else datas,
                    hiddenimports=["PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore"] if light else hiddenimports,
                    hookspath=[], runtime_hooks=[], excludes=[], noarchive=False)


exes, bins, dats = [], [], []
for script, name in SCRIPTS:
    a = analyse(script)
    a.pure[:] = [x for x in a.pure if not (x[0] == "photovec" or x[0].startswith("photovec."))]
    pyz = PYZ(a.pure)
    exes.append(EXE(pyz, a.scripts, [], exclude_binaries=True, name=name, console=False))
    bins += a.binaries
    dats += a.datas

for f in sorted(os.listdir(os.path.join("photovec", "fonts"))):                      # fonts of the names / letters and their licences
    dats.append((os.path.join("photovec", "fonts", f), os.path.abspath(os.path.join("photovec", "fonts", f)), "DATA"))
for f in sorted(os.listdir("photovec")):                                              # the program itself, as plain files
    if f.endswith(".py"):
        dats.append((os.path.join("photovec", f), os.path.abspath(os.path.join("photovec", f)), "DATA"))

coll = COLLECT(*exes, bins, dats, strip=False, upx=False, name="PhotoVecShow")
