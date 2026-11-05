"""Double-click / PyInstaller entry point for the PhotoVec Viewer (plays a .photovecshow file; needs no tracer)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from photovec.viewer import main
    sys.exit(main())
