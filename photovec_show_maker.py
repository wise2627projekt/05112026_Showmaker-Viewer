"""Double-click / PyInstaller entry point for the PhotoVec Show Maker (prepares the photo shows the Viewer plays)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()     # required for the Windows .exe + worker pool
    from photovec.show_maker import main
    sys.exit(main())
