"""Entry point for ``python -m rydsim`` and for the frozen portable binary.

Kept as a thin shim so the PyInstaller build and the module invocation share
exactly one code path with the ``rydsim`` console script (rydsim.cli:main).
"""

from __future__ import annotations

import multiprocessing
import sys

# ABSOLUTE import, deliberately. PyInstaller compiles the entry script as a
# top-level `__main__` with NO package context, so `from .cli import main`
# raises "attempted relative import with no known parent package" and the
# frozen binary dies before executing a line of physics. The build emits no
# warning, so a build-only CI gate goes green on a binary that cannot run.
# (Caught by the pre-release security audit, which built a replica and
# reproduced it.) Works identically for `python -m rydsim`.
from rydsim.cli import main

if __name__ == "__main__":
    # Required before anything spawns a process in a frozen build: without
    # it, a PyInstaller onefile binary re-executes its own bootloader per
    # child and forks endlessly. Harmless when running from source.
    multiprocessing.freeze_support()
    sys.exit(main())
