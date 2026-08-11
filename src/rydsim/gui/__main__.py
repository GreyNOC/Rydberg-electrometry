"""Windowed entry point: ``python -m rydsim.gui`` and ``RydSim.exe``.

This is the DOUBLE-CLICK target of the portable build. It exists as a
separate executable rather than as a mode of the CLI binary because
detecting "was I double-clicked?" is not reliable: the console-ownership
probe (GetConsoleProcessList) answers differently depending on how the
process was started, so a console binary that tries to guess ends up
printing argparse help into a console that Windows tears down immediately —
which is exactly the "terminal flashes and disappears" failure this
replaces.

A windowed binary cannot make that mistake: Windows gives it no console at
all, and it opens the GUI unconditionally.

Absolute import for the same reason as rydsim/__main__.py: PyInstaller
compiles the entry script as a top-level ``__main__`` with no package
context, so a relative import raises ImportError before anything runs.
"""

from __future__ import annotations

import multiprocessing
import sys
import traceback

from rydsim.gui.app import main

def _install_startup_trace():
    """Opt-in startup diagnostics: set RYDSIM_STARTUP_TRACE=1.

    A windowed build has no console, so a hang before the window maps is
    indistinguishable from "it does nothing". This dumps every thread's
    stack to a file beside the executable after a timeout, which is the only
    practical way to see where a frozen GUI is stuck.
    """
    import os

    if not os.environ.get("RYDSIM_STARTUP_TRACE"):
        return None
    import faulthandler

    try:
        from rydsim.paths import executable_dir

        target = executable_dir() / "rydsim-startup-trace.txt"
    except Exception:
        target = None
    if target is None:
        return None
    fh = open(target, "w", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)
    timeout = float(os.environ.get("RYDSIM_STARTUP_TRACE_TIMEOUT", "20"))
    faulthandler.dump_traceback_later(timeout, repeat=False,
                                      file=fh, exit=True)
    return fh


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _trace_fh = _install_startup_trace()
    try:
        sys.exit(main())
    except Exception:                                    # pragma: no cover
        # A windowed build has no console, so an unhandled traceback would
        # vanish silently and look identical to "it does nothing". Put it
        # somewhere the user can actually read it.
        detail = traceback.format_exc()
        try:
            import tkinter as tk
            from tkinter import scrolledtext

            root = tk.Tk()
            root.title("RydSim — startup error")
            root.geometry("900x520")
            box = scrolledtext.ScrolledText(root, wrap="word")
            box.pack(fill="both", expand=True)
            box.insert("1.0",
                       "RydSim could not start.\n\n"
                       "Please report this with the text below.\n\n" + detail)
            box.configure(state="disabled")
            root.mainloop()
        except Exception:
            # Tk itself is unavailable — leave a file beside the executable.
            try:
                from rydsim.paths import executable_dir

                log = executable_dir() / "rydsim-startup-error.txt"
                log.write_text(detail, encoding="utf-8")
            except Exception:
                pass
        sys.exit(1)
