# PyInstaller spec for the RydSim portable Windows binary.
#
#   cd <repo root> && python -m PyInstaller packaging/rydsim.spec --noconfirm
#
# Design notes (these are release-bar decisions, not incidental):
#
# * ONEFILE. "Portable" means a single artifact a user can drop anywhere and
#   run. It costs ~1-3 s of first-run unpacking to %TEMP%; acceptable for a
#   tool whose own runs take seconds to minutes.
#
# * NO TEST RUNNER, NO ORACLE. pytest/sympy are excluded. Shipping a test
#   harness inside a distributed artifact widens its attack surface for no
#   user benefit, and sympy is specifically the exact-symbolic ORACLE the
#   Wigner tests check the shipped float/exact-rational paths against —
#   bundling the oracle with the thing it validates defeats the point.
#   `rydsim validate` detects the frozen build and says so explicitly rather
#   than failing obscurely (see cli.cmd_validate).
#
# * NO THIRD-PARTY PAPER TEXT. kaulakys_text.txt is arXiv material we do not
#   have the right to redistribute; it is .gitignore'd and must never enter
#   the bundle. Nothing here adds it, and the release checklist greps the
#   built binary for a phrase from it as a belt-and-braces check.
#
# * The docs the binary actually cites at runtime ARE bundled (the CLI
#   points users at the conventions and integrity documents), so a portable
#   copy is not left referring to files it does not have.

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = []
# scikit-learn's estimators are resolved dynamically in places; collect them
# so the DESIGNER layer works in the frozen build.
hiddenimports += collect_submodules("sklearn.gaussian_process")
hiddenimports += collect_submodules("sklearn.ensemble")
hiddenimports += ["sklearn.utils._typedefs", "sklearn.utils._heap",
                  "sklearn.utils._sorting", "sklearn.utils._vector_sentinel"]
# Tk backend for the GUI subcommand.
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

a = Analysis(
    ["../src/rydsim/__main__.py"],
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="rydsim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed binaries trip AV heuristics; not worth it
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,       # CLI-first; `rydsim gui` opens the Tk window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
