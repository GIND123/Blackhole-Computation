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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def reproducibility_metadata() -> dict:
    """Return exact software, operating system, compiler, and Git metadata."""

    commit = _command_output(["git", "rev-parse", "HEAD"])
    status = _command_output(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    )
    mpi = _command_output(["mpiexec", "--version"])
    return {
        "git_commit": commit,
        "git_worktree_dirty": status is None or bool(status),
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
