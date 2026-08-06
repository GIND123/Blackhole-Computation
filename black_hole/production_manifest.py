"""Create the immutable reproduction manifest for the caustic production suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .source_evolution import load_sourced_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simulation_command(output_dir: Path, archive: Path) -> str:
    relative = archive.relative_to(output_dir)
    name = archive.stem
    backend = ""
    if relative.parts[:2] == ("cross_code", "finite_difference"):
        name = f"cross_{name}"
    elif relative.parts[:2] == ("cross_code", "dedalus"):
        name = f"cross_{name}"
        backend = " --backend dedalus"
    return (
        f"python -m black_hole.production_suite {name} "
        f"--output-dir {output_dir.as_posix()}{backend}"
    )


def create_manifest(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    archives = sorted(output_dir.glob("**/*.npz"))
    if not archives:
        raise FileNotFoundError(f"No raw archives found under {output_dir}")

    archive_rows = []
    commits: set[str] = set()
    for archive in archives:
        result = load_sourced_result(archive)
        reproducibility = result.metadata.get("reproducibility", {})
        commit = str(reproducibility.get("git_commit", ""))
        dirty = bool(reproducibility.get("git_worktree_dirty", True))
        if not commit or dirty:
            raise ValueError(f"Archive does not record a clean Git commit: {archive}")
        commits.add(commit)
        archive_rows.append(
            {
                "path": archive.relative_to(output_dir).as_posix(),
                "sha256": _sha256(archive),
                "bytes": archive.stat().st_size,
                "git_commit": commit,
                "command": _simulation_command(output_dir, archive),
            }
        )
    if len(commits) != 1:
        raise ValueError(f"Archives were not generated from one frozen commit: {commits}")

    paths = {row["path"]: row["sha256"] for row in archive_rows}
    production = sorted(path for path in paths if path.startswith("raw/"))
    pilots = sorted(path for path in paths if path.startswith("pilots/raw/"))
    cross_code = sorted(path for path in paths if path.startswith("cross_code/"))
    width = [path for path in pilots if "/width_" in f"/{path}"]
    d1_convergence = [
        path
        for path in pilots
        if "sds_L20_radial" in path
        or "sds_L80_radial" in path
        or "sds_L20_temporal" in path
        or "sds_L80_temporal" in path
        or "sds_L20_angular" in path
        or "sds_L80_angular" in path
        or path.endswith("radial_N1536.npz")
        or path.endswith("radial_N2048.npz")
        or path.endswith("temporal_dt0.002.npz")
        or path.endswith("temporal_dt0.001.npz")
        or path.endswith("angular_w0p5_lmax42.npz")
        or path.endswith("angular_w0p5_lmax46.npz")
    ]

    analysis_names = {
        "production_analysis.json",
        "observable_convergence.png",
        "tables/cross_code_observables.csv",
        "tables/cross_code_D1.csv",
    }
    derived_rows = []
    for artifact in sorted(output_dir.glob("**/*")):
        if not artifact.is_file() or artifact.suffix == ".npz":
            continue
        relative = artifact.relative_to(output_dir).as_posix()
        if relative.startswith("logs/") or relative == "manifest.json":
            continue
        if relative in analysis_names:
            inputs = pilots + cross_code
            command = (
                "python -m black_hole.production_analysis "
                f"--output-dir {output_dir.as_posix()} --include-cross-code"
            )
        else:
            command = (
                "python -m black_hole.production_report "
                f"--output-dir {output_dir.as_posix()}"
            )
            if "source_width" in relative:
                inputs = width
            elif "D1_convergence" in relative:
                inputs = d1_convergence
            elif "error_budget" in relative or relative == "production_summary.json":
                inputs = production + width + d1_convergence
            else:
                inputs = production
        derived_rows.append(
            {
                "path": relative,
                "sha256": _sha256(artifact),
                "command": command,
                "inputs": [{"path": path, "sha256": paths[path]} for path in inputs],
            }
        )

    manifest = {
        "schema_version": 1,
        "frozen_git_commit": next(iter(commits)),
        "raw_archives_are_read_only_inputs": True,
        "archives": archive_rows,
        "derived_artifacts": derived_rows,
        "manifest_command": (
            "python -m black_hole.production_manifest "
            f"--output-dir {output_dir.as_posix()}"
        ),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(create_manifest(arguments.output_dir))


if __name__ == "__main__":
    main()
