# PyInstaller spec for the RydSim portable Windows build.
#
#   cd <repo root> && .venv-build/Scripts/python -m PyInstaller packaging/rydsim.spec --noconfirm
#
# TWO executables are produced from one code base:
#
#   RydSim.exe      windowed (console=False) — the DOUBLE-CLICK target.
#                   Opens the GUI unconditionally.
#   rydsim-cli.exe  console — the CLI, for terminals and scripts.
#
# The CLI is NOT named rydsim.exe: Windows filenames are case-insensitive,
# so "rydsim.exe" and "RydSim.exe" are the same file and the second EXE()
# silently overwrote the first. Both builds reported success and the bundle
# shipped one binary wearing the other's name.
#
# Why two rather than one binary that decides for itself: a console binary
# cannot reliably tell whether it was double-clicked. The obvious probe
# (GetConsoleProcessList == 1) answers differently depending on how the
# process was launched, so the single-binary version fell through to
# argparse help inside a console Windows destroyed immediately — the
# "terminal flashes and disappears" failure. A windowed binary has no
# console to lose and cannot make that mistake. Guessing the launch context
# was the bug; not guessing is the fix.
#
# Other release-bar decisions:
#
# * ONEFILE. "Portable" means an artifact a user can drop anywhere and run.
#   Costs ~1-3 s of first-run unpacking to %TEMP%.
#
# * NO TEST RUNNER, NO ORACLE. pytest/sympy are excluded. Shipping a test
#   harness inside a distributed artifact widens its attack surface for no
#   user benefit, and sympy is specifically the exact-symbolic ORACLE the
#   Wigner tests check the shipped float/exact-rational paths against —
#   bundling the oracle with the thing it validates defeats the point.
#   Both binaries detect the frozen build and say so rather than failing
#   obscurely (cli.cmd_validate, gui App.on_validate).
#
# * NO THIRD-PARTY PAPER TEXT. kaulakys_text.txt is arXiv material we may
#   not redistribute; `datas` is an explicit list with no Tree()/glob, so
#   there is no path by which it enters. The release checklist greps the
#   built binaries for it as a belt-and-braces check.
#
# * Build from the LOCKED venv (packaging/requirements-build.lock). Building
#   from a developer's global interpreter let PyInstaller's optional-backend
#   hooks reach toward torch/CUDA and would produce an untruthful SBOM.

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = []
# scikit-learn resolves some estimators dynamically; collect what the
# DESIGNER layer needs so the frozen build can fit surrogates.
hiddenimports += collect_submodules("sklearn.gaussian_process")
hiddenimports += collect_submodules("sklearn.ensemble")
hiddenimports += ["sklearn.utils._typedefs", "sklearn.utils._heap",
                  "sklearn.utils._sorting", "sklearn.utils._vector_sentinel"]
hiddenimports += ["matplotlib.backends.backend_tkagg"]

excludes = [
    # test-only / oracle (see header)
    "pytest", "_pytest", "py", "sympy", "mpmath",
    # never used by the shipped code paths
    "IPython", "jupyter", "notebook", "pandas", "PIL.ImageQt",
    "PySide6", "PyQt5", "PyQt6", "wx",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_webagg",
]

datas = [
    ("../docs/spec/00-conventions.md", "docs/spec"),
    ("../docs/spec/00-integrity-audit.md", "docs/spec"),
    ("../README.md", "."),
    ("../LICENSE", "."),
]

common = dict(
    pathex=["../src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ---- console CLI ----------------------------------------------------------
a_cli = Analysis(["../src/rydsim/__main__.py"], **common)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
exe_cli = EXE(
    pyz_cli, a_cli.scripts, a_cli.binaries, a_cli.zipfiles, a_cli.datas, [],
    name="rydsim-cli",
    debug=False, bootloader_ignore_signals=False, strip=False,
    upx=False,           # UPX-packed binaries trip AV heuristics
    upx_exclude=[], runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
)

# ---- windowed GUI (the double-click target) -------------------------------
a_gui = Analysis(["../src/rydsim/gui/__main__.py"], **common)
pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)
exe_gui = EXE(
    pyz_gui, a_gui.scripts, a_gui.binaries, a_gui.zipfiles, a_gui.datas, [],
    name="RydSim",
    debug=False, bootloader_ignore_signals=False, strip=False,
    upx=False, upx_exclude=[], runtime_tmpdir=None,
    console=False,       # no console: nothing to flash, nothing to lose
    disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
)
