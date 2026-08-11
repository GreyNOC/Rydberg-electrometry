"""Filesystem locations, resolved correctly for both source and frozen runs.

Why this module exists
----------------------
Both the CLI and the GUI derived the findings directory from
``__file__``'s ancestors. In a PyInstaller onefile build ``__file__``
points inside the extracted bundle under ``%TEMP%\\_MEIxxxxxx``, so
``parents[2]/"findings"`` resolved to ``%TEMP%/findings`` — meaning every
provenance-tagged finding, the artifact the whole pipeline exists to
produce, was written somewhere Windows Storage Sense deletes. The
pre-release security audit caught this by running a replica build.

Rules:
- Source checkout: findings live in the repo, exactly as before.
- Frozen binary: findings live next to the executable, so a portable copy
  keeps its outputs with it. If that location is not writable (Program
  Files, a read-only share, a network path), fall back to the per-user
  documents area rather than silently losing data — and say where.
"""

from __future__ import annotations

import os
import pathlib
import sys


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) bundle."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> pathlib.Path | None:
    """Directory of extracted bundle data (``sys._MEIPASS``), if frozen.

    NOTE: this directory is DELETED when the process exits, so nothing
    durable may be written here and nothing here survives for the user to
    read afterwards.
    """
    mei = getattr(sys, "_MEIPASS", None)
    return pathlib.Path(mei) if mei else None


def executable_dir() -> pathlib.Path:
    """Directory containing the running executable (frozen) or the repo."""
    if is_frozen():
        return pathlib.Path(sys.executable).resolve().parent
    return repo_root()


def repo_root() -> pathlib.Path:
    """Source-checkout root (…/src/rydsim/paths.py -> three parents up)."""
    return pathlib.Path(__file__).resolve().parents[2]


def _writable(path: pathlib.Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".rydsim-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def findings_dir() -> pathlib.Path:
    """Where provenance-tagged findings are written.

    Honours ``RYDSIM_FINDINGS_DIR`` if set, so a user or a CI job can pin
    the location explicitly.
    """
    override = os.environ.get("RYDSIM_FINDINGS_DIR")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    if not is_frozen():
        return repo_root() / "findings"
    beside = executable_dir() / "findings"
    if _writable(beside):
        return beside
    fallback = pathlib.Path.home() / "Documents" / "RydSim" / "findings"
    return fallback


def docs_path(relative: str) -> pathlib.Path | None:
    """Locate a bundled/repo doc, or None if it is not present.

    Returns a path inside the bundle when frozen — readable while the
    process lives, gone afterwards — and the repo copy otherwise.
    """
    base = bundle_dir() if is_frozen() else repo_root()
    if base is None:
        return None
    candidate = base / relative
    return candidate if candidate.exists() else None
