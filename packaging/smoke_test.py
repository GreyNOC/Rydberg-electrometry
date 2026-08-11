"""Verify BUILT ARTIFACTS, not source. Run by CI and before any release.

Why this exists
---------------
The pre-release security audit of v0.2.0 established the governing fact for
this file: *the PyInstaller build emitted zero errors and zero warnings while
producing a binary that died on startup.* A CI gate that only builds is
therefore worthless — it goes green on an artifact nobody can run. Every
check here drives the actual executable.

Each check corresponds to a defect that really shipped or nearly shipped:

  startup            relative import in the frozen entry point -> ImportError
                     before a line of physics ran
  distinct names     "rydsim.exe" and "RydSim.exe" are the SAME file on
                     Windows; one EXE() silently overwrote the other while
                     both builds reported success
  physics            the frozen numerics must equal the source install, not
                     merely "run"
  findings location  paths derived from __file__ resolved into %TEMP% in a
                     onefile build, so reports landed where Windows purges
  validate refusal   the suite is not bundled; the binary must SAY so rather
                     than fail obscurely
  hygiene            no build-machine username, no non-redistributable paper
                     text, no torch/CUDA baked in

Usage:
    python packaging/smoke_test.py dist/rydsim-cli.exe [dist/RydSim.exe]
Exit code 0 = all checks passed; 1 = at least one failed (details on stdout).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

# Physics the frozen binary must reproduce exactly. These are the values the
# source install produces for this configuration; a frozen build that differs
# has a broken numerical stack (wrong BLAS, missing hook, truncated data).
EXPECTED_RABI_MHZ = "18.2997"
EXPECTED_FIELD = "1.43017"

FORBIDDEN_STRINGS = [
    (b"Kaulakys", "non-redistributable arXiv paper text"),
    (b"cudnn", "torch/CUDA runtime"),
    (b"libtorch", "torch runtime"),
]

failures: list[str] = []
checks_run = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global checks_run
    checks_run += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name}: {detail}")
    return ok


def run(exe: pathlib.Path, *args: str, cwd: pathlib.Path | None = None,
        timeout: int = 300) -> subprocess.CompletedProcess:
    """Invoke the artifact, turning launch failures into reportable results.

    A corrupt or wrong-architecture binary raises OSError from CreateProcess
    rather than returning a code. Letting that escape would crash the gate
    with a traceback instead of naming which check failed — the negative
    control that caught this had a hostile artifact abort the whole run.
    """
    try:
        return subprocess.run([str(exe), *args], capture_output=True,
                              text=True, timeout=timeout,
                              cwd=str(cwd) if cwd else None)
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=[str(exe), *args], returncode=126, stdout="",
            stderr=f"could not execute artifact: {exc}")
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=[str(exe), *args], returncode=124, stdout="",
            stderr=f"timed out after {timeout}s: {exc}")


def check_cli(exe: pathlib.Path) -> None:
    print(f"\n== console binary: {exe.name} ==")

    r = run(exe, "--version")
    check("starts and reports a version",
          r.returncode == 0 and "rydsim" in r.stdout.lower(),
          f"rc={r.returncode} out={r.stdout.strip()[:80]!r} err={r.stderr.strip()[:120]!r}")

    r = run(exe)
    check("no-args prints help and exits 0 (not an argparse error)",
          r.returncode == 0 and "usage:" in r.stdout,
          f"rc={r.returncode}")

    # Real physics through the frozen numerical stack.
    r = run(exe, "at", "--rf-rabi-mhz", "18", "--rf-dipole-ea0", "1000")
    got_rabi = EXPECTED_RABI_MHZ in r.stdout
    got_field = EXPECTED_FIELD in r.stdout
    check("frozen physics reproduces the source install",
          r.returncode == 0 and got_rabi and got_field,
          f"expected {EXPECTED_RABI_MHZ} MHz / {EXPECTED_FIELD} V/m; "
          f"rc={r.returncode} out={r.stdout.strip()[-160:]!r}")

    # Findings must land beside the executable, never in %TEMP%.
    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td)
        portable = work / exe.name
        portable.write_bytes(exe.read_bytes())
        portable.chmod(0o755)
        r = run(portable, "at", "--rf-rabi-mhz", "18", "--rf-dipole-ea0",
                "1000", "--finding", cwd=work)
        produced = list((work / "findings").glob("*.md")) if (work / "findings").is_dir() else []
        check("writes findings beside the executable, not into %TEMP%",
              r.returncode == 0 and len(produced) == 1,
              f"rc={r.returncode} found={[p.name for p in produced]}")

    # The suite is not bundled: refuse clearly instead of failing obscurely.
    r = run(exe, "validate")
    combined = r.stdout + r.stderr
    check("validate refuses clearly in the portable build",
          "portable build" in combined.lower() or "not bundled" in combined.lower(),
          f"rc={r.returncode} out={combined.strip()[:160]!r}")


def check_gui(exe: pathlib.Path) -> None:
    print(f"\n== windowed binary: {exe.name} ==")
    # A windowed build has no console, so the only thing assertable without a
    # desktop session is that it starts and STAYS alive (mainloop running)
    # rather than exiting immediately. On a headless runner a Tk window cannot
    # map, so absence of a window is not evidence of failure here.
    import time

    try:
        proc = subprocess.Popen([str(exe)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except OSError as exc:
        check("windowed binary starts and does not exit immediately",
              False, f"could not execute artifact: {exc}")
        return
    try:
        time.sleep(20)
        alive = proc.poll() is None
        detail = "still running (event loop up)" if alive else \
                 f"exited early rc={proc.returncode}"
        if not alive:
            out, err = proc.communicate(timeout=10)
            detail += f" err={err.decode(errors='replace').strip()[:200]!r}"
        check("windowed binary starts and does not exit immediately",
              alive, detail)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)


def check_hygiene(exes: list[pathlib.Path]) -> None:
    print("\n== artifact hygiene ==")

    # Windows filenames are case-insensitive: two artifacts differing only by
    # case are ONE file, and the second build silently overwrites the first.
    lowered = [e.name.lower() for e in exes]
    check("artifact names are distinct case-insensitively",
          len(set(lowered)) == len(lowered),
          f"names={[e.name for e in exes]}")

    import getpass

    try:
        user = getpass.getuser().encode()
    except Exception:
        user = b""
    for exe in exes:
        blob = exe.read_bytes()
        for needle, what in FORBIDDEN_STRINGS:
            check(f"{exe.name}: no {what}", needle not in blob)
        if user and len(user) >= 4:
            check(f"{exe.name}: no build-machine username baked in",
                  user not in blob)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    exes = [pathlib.Path(a).resolve() for a in argv]
    missing = [e for e in exes if not e.is_file()]
    if missing:
        print(f"ERROR: artifact(s) not found: {[str(m) for m in missing]}")
        return 1

    print(f"Verifying {len(exes)} artifact(s)")
    for e in exes:
        print(f"  {e.name}  {e.stat().st_size/1e6:.1f} MB")

    for exe in exes:
        # The windowed build is identified by name; it has no console output.
        if exe.stem.lower() == "rydsim" and exe.stem != "rydsim-cli":
            check_gui(exe)
        else:
            check_cli(exe)
    check_hygiene(exes)

    print(f"\n{checks_run - len(failures)}/{checks_run} checks passed")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All artifact checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
