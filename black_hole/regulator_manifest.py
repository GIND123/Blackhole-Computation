"""Create and verify the newline-stable v3 regulator reproduction manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .sds_result import load_sds_result
from .source_evolution import load_sourced_result


TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".out", ".err"}


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    """Hash UTF-8 text after canonical CRLF/CR to LF normalization."""

    content = Path(path).read_bytes().decode("utf-8")
    canonical = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_record(path: Path) -> dict:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return {
            "hash_mode": "sha256_utf8_canonical_lf_v1",
            "sha256": sha256_lf(path),
            "byte_sha256": sha256_bytes(path),
        }
    return {"hash_mode": "sha256_bytes", "sha256": sha256_bytes(path)}


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _archive_metadata(path: Path) -> dict:
    if "flat" in path.parts:
        return load_sds_result(path).metadata
    return load_sourced_result(path).metadata


def _commands(output_dir: Path, output_label: str, archives: list[Path]) -> Path:
    rows = []
    for archive in archives:
        metadata = _archive_metadata(archive)
        provenance = metadata["simulation_provenance"]
        command = provenance["command"]
        if "raw/flat/" in archive.relative_to(output_dir).as_posix():
            command = "OMP_NUM_THREADS=1 " + command
        rows.append(command)
    rows.extend(
        (
            "python -m black_hole.regulator_analysis --output-dir "
            f"{output_label}",
            "python -m black_hole.regulator_manifest --output-dir "
            f"{output_label}",
            "python -m black_hole.regulator_manifest --output-dir "
            f"{output_label} --verify",
        )
    )
    path = output_dir / "logs" / "recorded_commands.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(rows) + "\n")
    return path


def _input_group(relative: str, groups: dict[str, list[dict]]) -> list[dict]:
    if relative.startswith("tables/flat_") or relative.startswith("flat_") or relative == "nested_extrapolants.png":
        return groups["flat"]
    if relative.startswith("tables/D1_") or relative.startswith("D1_"):
        return groups["source"]
    if "L12_phase" in relative:
        return groups["l12"]
    if "source_width" in relative:
        return groups["source_width"]
    return groups["all"]


def create_manifest(output_dir: Path, repository_root: Path | None = None) -> Path:
    repository_root = Path(repository_root or Path.cwd()).resolve()
    output_dir = Path(output_dir).resolve()
    output_label = output_dir.relative_to(repository_root).as_posix()
    archives = sorted((output_dir / "raw").glob("**/*.npz"))
    if not archives:
        raise FileNotFoundError("No v3 raw archives were found.")
    analysis_commit = _git_output("rev-parse", "HEAD")
    tracked_status = _git_output("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise ValueError("Analysis artifacts must be generated from a clean analysis commit.")

    archive_rows = []
    simulation_commits = set()
    for archive in archives:
        metadata = _archive_metadata(archive)
        provenance = metadata["simulation_provenance"]
        if provenance["git_worktree_dirty"]:
            raise ValueError(f"Archive records a dirty worktree: {archive}")
        simulation_commits.add(provenance["git_commit"])
        archive_rows.append(
            {
                "path": archive.relative_to(repository_root).as_posix(),
                **_hash_record(archive),
                "bytes": archive.stat().st_size,
                "simulation_commit": provenance["git_commit"],
                "case": provenance["case"],
                "refinement_level": provenance["refinement_level"],
                "physical_contract_sha256": provenance[
                    "physical_contract_sha256"
                ],
                "command": provenance["command"],
            }
        )
    if len(simulation_commits) != 1:
        raise ValueError(f"Expected one simulation commit, found {simulation_commits}")

    external_paths = {
        "l12": [
            repository_root
            / "results/caustic_production_v2/tables/production_generic_angles.csv",
            repository_root
            / "results/caustic_production_v2/tables/production_generic_phase.csv",
        ],
        "source_width": [
            repository_root
            / "results/caustic_production_v2/tables/source_width_delay_sensitivity.csv"
        ],
    }
    external_rows = {
        name: [
            {
                "path": path.relative_to(repository_root).as_posix(),
                **_hash_record(path),
                "role": "read_only_historical_input",
            }
            for path in paths
        ]
        for name, paths in external_paths.items()
    }
    groups = {
        "flat": [row for row in archive_rows if "/raw/flat/" in f"/{row['path']}"],
        "source": [row for row in archive_rows if "/raw/source/" in f"/{row['path']}"],
        "l12": external_rows["l12"],
        "source_width": external_rows["source_width"],
    }
    groups["all"] = (
        groups["flat"]
        + groups["source"]
        + groups["l12"]
        + groups["source_width"]
    )

    commands_path = _commands(output_dir, output_label, archives)
    derived_rows = []
    for artifact in sorted(output_dir.glob("**/*")):
        if not artifact.is_file() or "raw" in artifact.parts:
            continue
        relative = artifact.relative_to(output_dir).as_posix()
        if relative == "manifest.json" or relative.startswith("logs/simulation/"):
            continue
        derived_rows.append(
            {
                "path": artifact.relative_to(repository_root).as_posix(),
                **_hash_record(artifact),
                "bytes": artifact.stat().st_size,
                "analysis_commit": analysis_commit,
                "command": (
                    "python -m black_hole.regulator_analysis "
                    f"--output-dir {output_label}"
                    if relative != "logs/recorded_commands.txt"
                    else "generated by regulator_manifest"
                ),
                "inputs": [
                    {key: value for key, value in row.items() if key in {"path", "hash_mode", "sha256"}}
                    for row in _input_group(relative, groups)
                ],
            }
        )

    manifest = {
        "schema_version": 2,
        "purpose": "Artificial cosmology regulator production and analysis record",
        "simulation_commit": next(iter(simulation_commits)),
        "analysis_commit": analysis_commit,
        "simulation_and_analysis_commits_are_distinct": (
            analysis_commit != next(iter(simulation_commits))
        ),
        "text_hash_definition": (
            "SHA256 of UTF-8 bytes after CRLF and CR are canonicalized to LF"
        ),
        "raw_archives_are_read_only_inputs": True,
        "L1280_was_run": False,
        "archives": archive_rows,
        "external_inputs": [*external_rows["l12"], *external_rows["source_width"]],
        "derived_artifacts": derived_rows,
        "manifest_commands": {
            "create": (
                "python -m black_hole.regulator_manifest --output-dir "
                f"{output_label}"
            ),
            "verify": (
                "python -m black_hole.regulator_manifest --output-dir "
                f"{output_label} --verify"
            ),
        },
    }
    path = output_dir / "manifest.json"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def verify_manifest(path: Path) -> dict:
    path = Path(path)
    root = Path.cwd().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    checked = 0
    for section in ("archives", "external_inputs", "derived_artifacts"):
        for row in manifest[section]:
            artifact = root / row["path"]
            if not artifact.exists():
                failures.append(f"missing:{row['path']}")
                continue
            actual = (
                sha256_lf(artifact)
                if row["hash_mode"] == "sha256_utf8_canonical_lf_v1"
                else sha256_bytes(artifact)
            )
            checked += 1
            if actual != row["sha256"]:
                failures.append(f"hash:{row['path']}")
    result = {"verified": not failures, "checked_files": checked, "failures": failures}
    if failures:
        raise ValueError(f"Manifest verification failed: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/regulator_production_v3")
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        print(json.dumps(verify_manifest(arguments.output_dir / "manifest.json"), indent=2))
    else:
        print(create_manifest(arguments.output_dir, arguments.repository_root))


if __name__ == "__main__":
    main()
