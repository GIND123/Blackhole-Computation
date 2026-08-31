"""Runtime and source provenance recorded in production archives."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys

import numpy as np
import scipy


def _command_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def _worktree_state() -> str:
    """Return ``clean``, ``dirty``, or ``unknown`` for the current worktree.

    This cannot go through :func:`_command_output`.  A clean worktree makes
    ``git status --porcelain`` print nothing, and that helper collapses empty
    output to ``None``, so routing the probe through it makes a clean worktree
    indistinguishable from a failed probe.  Treating the pair as dirty then
    marks every archive dirty however clean the checkout is, which is what
    happened to every archive written after the flag became
    ``status is None or bool(status)``.
    """

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return "dirty" if completed.stdout.strip() else "clean"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def reproducibility_metadata() -> dict:
    """Return exact software, operating system, compiler, and Git metadata."""

    commit = _command_output(["git", "rev-parse", "HEAD"])
    worktree = _worktree_state()
    mpi = _command_output(["mpiexec", "--version"])
    return {
        "git_commit": commit,
        # An unknown state is still recorded as dirty, because an archive that
        # cannot prove its provenance must not be promoted to production.  The
        # state beside it says which of the two it was.
        "git_worktree_dirty": worktree != "clean",
        "git_worktree_state": worktree,
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "dedalus": _package_version("dedalus"),
        "mpi4py": _package_version("mpi4py"),
        "mpi": mpi,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
