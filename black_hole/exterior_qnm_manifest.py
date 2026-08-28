"""Create and verify the unified exterior-QNM evidence manifest.

The headline comparison is assembled from three physically distinct inputs:

* frozen Schwarzschild and uniform-SdS controls in the regulator package;
* unchanged ``L/M=80,160`` members of the original exterior family; and
* the fixed-width ``L/M=320,640`` exterior production archives.

Keeping the files in their original packages avoids duplicating immutable raw
data.  This manifest nevertheless records all of them in one checksum index
and links every derived QNM artifact to that complete input set.  Source-file
hashes are authoritative because the historical exterior archives themselves
record that their Git worktrees were dirty; the manifest preserves that fact
rather than promoting those archives to clean-production provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .far_regulator_production import (
    CONTROL_ROOT,
    LENGTHS,
    OUTPUT_ROOT,
    archive_path,
    case_catalogue,
    contract_sha256,
)
from .far_regulator_production_analysis import (
    LEGACY_CANDIDATE_ROOT,
    _candidate_archive_path,
    analysis_contract_sha256,
    load_candidates,
    load_controls,
    validate_archives,
)
from .regulator_manifest import _hash_record, sha256_bytes, sha256_lf
from .regulator_suite import LEVELS
from .sds_result import load_sds_result


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
INPUT_SET = "complete_exterior_qnm_analysis"
DERIVED_SUFFIXES = {".csv", ".json", ".pdf", ".png"}

# These are the modules directly responsible for the model, evolution,
# archive format, controls, and postprocessing.  Recording their content
# hashes makes the released source state explicit even when HEAD contains
# unrelated manuscript changes.
SOURCE_PATHS = (
    Path("black_hole/exterior_qnm_manifest.py"),
    Path("black_hole/far_regulator_production.py"),
    Path("black_hole/far_regulator_production_analysis.py"),
    Path("black_hole/exterior_sds_model.py"),
    Path("black_hole/sds_solver.py"),
    Path("black_hole/sds_model.py"),
    Path("black_hole/schwarzschild_scalar.py"),
    Path("black_hole/regulator_suite.py"),
    Path("black_hole/regulator_analysis.py"),
    Path("black_hole/sds_result.py"),
    Path("black_hole/reproducibility.py"),
)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root).as_posix()


def _control_path(control_dir: Path, level: str, length: int | None) -> Path:
    label = "schwarzschild" if length is None else f"sds_L{length}"
    return Path(control_dir) / "raw" / "flat" / level / f"{label}.npz"


def _archive_record(
    path: Path,
    root: Path,
    *,
    family: str,
    level: str,
    length: int | None,
    storage: str,
    compatibility: str,
    reproduction_command: str | None = None,
) -> dict:
    result = load_sds_result(path)
    metadata = result.metadata
    provenance = metadata["simulation_provenance"]
    numerical = metadata["numerical"]
    dirty = bool(provenance["git_worktree_dirty"])
    return {
        "path": _relative(path, root),
        **_hash_record(path),
        "bytes": path.stat().st_size,
        "role": "raw_waveform_input",
        "family": family,
        "cosmological_length_over_M": length,
        "refinement_level": level,
        "resolution": int(numerical["resolution"]),
        "timestep_over_M": float(numerical["timestep"]),
        "end_time_over_M": float(numerical["end_time"]),
        "storage": storage,
        "compatibility": compatibility,
        "simulation_case": provenance["case"],
        "simulation_git_base_commit": provenance["git_commit"],
        "git_worktree_dirty": dirty,
        "provenance_grade": "screening" if dirty else "production",
        "physical_contract_sha256": provenance["physical_contract_sha256"],
        "archived_command": provenance["command"],
        "reproduction_command": reproduction_command or provenance["command"],
    }


def _archive_rows(
    output_dir: Path,
    control_dir: Path,
    legacy_dir: Path,
    root: Path,
) -> list[dict]:
    rows: list[dict] = []
    output_label = _relative(output_dir, root)

    for level in LEVELS:
        rows.append(
            _archive_record(
                _control_path(control_dir, level, None),
                root,
                family="schwarzschild_control",
                level=level,
                length=None,
                storage="external_frozen_control",
                compatibility="native_regulator_v3_control",
            )
        )
        for length in LENGTHS:
            rows.append(
                _archive_record(
                    _control_path(control_dir, level, length),
                    root,
                    family="uniform_sds_control",
                    level=level,
                    length=length,
                    storage="external_frozen_control",
                    compatibility="native_regulator_v3_control",
                )
            )

    catalogue = case_catalogue()
    for length in LENGTHS:
        for level in LEVELS:
            path = _candidate_archive_path(output_dir, legacy_dir, length, level)
            current = archive_path(output_dir, length, level)
            storage = (
                "internal_width_floor_archive"
                if path.resolve() == current.resolve()
                else "external_unchanged_legacy_archive"
            )
            compatibility = (
                "native_fixed_width_family"
                if storage == "internal_width_floor_archive"
                else "validated_legacy_geometry_equivalent_at_L80_L160"
            )
            case = f"width_floor_sds_L{length}_{level}"
            if case not in catalogue:
                raise ValueError(f"No reproduction case is defined for {case}.")
            rows.append(
                _archive_record(
                    path,
                    root,
                    family="exterior_supported_sds",
                    level=level,
                    length=length,
                    storage=storage,
                    compatibility=compatibility,
                    reproduction_command=(
                        "python -m black_hole.far_regulator_production "
                        f"{case} --output-dir {output_label}"
                    ),
                )
            )
    return rows


def _source_rows(root: Path) -> list[dict]:
    rows = []
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing exterior-QNM source file: {path}")
        rows.append(
            {
                "path": relative.as_posix(),
                **_hash_record(path),
                "bytes": path.stat().st_size,
                "role": "active_reproduction_source",
            }
        )
    return rows


def _derived_rows(output_dir: Path, root: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DERIVED_SUFFIXES:
            continue
        relative_to_package = path.relative_to(output_dir)
        if "raw" in relative_to_package.parts or path.name == MANIFEST_NAME:
            continue
        rows.append(
            {
                "path": _relative(path, root),
                **_hash_record(path),
                "bytes": path.stat().st_size,
                "role": "derived_qnm_artifact",
                "command": (
                    "python -m black_hole.far_regulator_production_analysis "
                    f"--output-dir {_relative(output_dir, root)} "
                    f"--control-dir {_relative(CONTROL_ROOT, root)} "
                    "--legacy-candidate-dir "
                    f"{_relative(LEGACY_CANDIDATE_ROOT, root)}"
                ),
                "input_set": INPUT_SET,
            }
        )
    return rows


def _present_package_files(output_dir: Path, root: Path) -> set[str]:
    return {
        _relative(path, root)
        for path in output_dir.rglob("*")
        if path.is_file()
    }


def create_manifest(
    output_dir: Path = OUTPUT_ROOT,
    repository_root: Path | None = None,
    *,
    control_dir: Path = CONTROL_ROOT,
    legacy_dir: Path = LEGACY_CANDIDATE_ROOT,
) -> Path:
    """Validate the archived analysis and write its unified checksum index."""

    root = Path(repository_root or Path.cwd()).resolve()
    output_dir = Path(output_dir).resolve()
    control_dir = Path(control_dir).resolve()
    legacy_dir = Path(legacy_dir).resolve()
    output_label = _relative(output_dir, root)
    control_label = _relative(control_dir, root)
    legacy_label = _relative(legacy_dir, root)

    # This is the same full metadata/physics validation performed before the
    # archived tables and figures are generated.
    controls = load_controls(control_dir, LENGTHS)
    candidates = load_candidates(output_dir, LENGTHS, legacy_dir)
    validate_archives(controls, candidates, LENGTHS)

    summary_path = output_dir / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["simulation_contract_sha256"] != contract_sha256():
        raise ValueError("Analysis summary and released simulation contract differ.")
    if summary["analysis_contract_sha256"] != analysis_contract_sha256():
        raise ValueError("Analysis summary and released analysis contract differ.")

    archives = _archive_rows(output_dir, control_dir, legacy_dir, root)
    sources = _source_rows(root)
    derived = _derived_rows(output_dir, root)
    if len(archives) != 27:
        raise ValueError(f"Expected 27 unified raw inputs, found {len(archives)}.")
    if not derived:
        raise FileNotFoundError("No exterior-QNM analysis artifacts were found.")

    archive_inputs = [
        {
            "path": row["path"],
            "hash_mode": row["hash_mode"],
            "sha256": row["sha256"],
        }
        for row in archives
    ]
    relevant_status = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *(path.as_posix() for path in SOURCE_PATHS),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Unified exterior-supported SdS QNM comparison evidence",
        "package": output_label,
        "control_package": control_label,
        "legacy_candidate_package": legacy_label,
        "analysis_git_base_commit": _git_output("rev-parse", "HEAD"),
        "analysis_source_files_are_dirty_relative_to_base": bool(relevant_status),
        "source_hashes_are_authoritative": True,
        "provenance_note": (
            "The exterior archives record dirty simulation worktrees and are "
            "therefore retained as screening-grade evidence. Their immutable "
            "bytes, embedded contracts, commands, numerical parameters, and "
            "compatibility with the released source are verified here; this "
            "manifest does not relabel them as clean-production archives."
        ),
        "simulation_contract_sha256": contract_sha256(),
        "analysis_contract_sha256": analysis_contract_sha256(),
        "raw_archives_are_read_only_inputs": True,
        "input_sets": {INPUT_SET: archive_inputs},
        "manifest_commands": {
            "create": (
                "python -m black_hole.exterior_qnm_manifest --output-dir "
                f"{output_label} --control-dir {control_label} "
                f"--legacy-candidate-dir {legacy_label}"
            ),
            "verify": (
                "python -m black_hole.exterior_qnm_manifest --output-dir "
                f"{output_label} --verify"
            ),
        },
        "archives": archives,
        "source_files": sources,
        "derived_artifacts": derived,
    }
    destination = output_dir / MANIFEST_NAME
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def verify_manifest(path: Path) -> dict:
    """Verify hashes, input links, contracts, and package completeness."""

    path = Path(path).resolve()
    root = Path.cwd().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0

    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if manifest.get("simulation_contract_sha256") != contract_sha256():
        failures.append("simulation_contract")
    if manifest.get("analysis_contract_sha256") != analysis_contract_sha256():
        failures.append("analysis_contract")

    indexed: dict[str, dict] = {}
    for section in ("archives", "source_files", "derived_artifacts"):
        for row in manifest.get(section, []):
            if row["path"] in indexed:
                failures.append(f"duplicate:{row['path']}")
            indexed[row["path"]] = row
            artifact = root / row["path"]
            if not artifact.is_file():
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

    inputs = manifest.get("input_sets", {}).get(INPUT_SET, [])
    if len(inputs) != len(manifest.get("archives", [])):
        failures.append("input_set_size")
    for item in inputs:
        row = indexed.get(item["path"])
        if row is None or any(
            item.get(key) != row.get(key) for key in ("hash_mode", "sha256")
        ):
            failures.append(f"input_link:{item['path']}")
    for row in manifest.get("derived_artifacts", []):
        if row.get("input_set") != INPUT_SET:
            failures.append(f"derived_input_set:{row['path']}")

    output_dir = root / manifest["package"]
    listed_package = {
        row["path"]
        for section in ("archives", "derived_artifacts")
        for row in manifest.get(section, [])
        if (root / row["path"]).is_relative_to(output_dir)
    }
    listed_package.add(path.relative_to(root).as_posix())
    unlisted = sorted(_present_package_files(output_dir, root) - listed_package)
    failures.extend(f"unlisted:{item}" for item in unlisted)

    report = {
        "verified": not failures,
        "checked_files": checked,
        "raw_inputs": len(manifest.get("archives", [])),
        "source_files": len(manifest.get("source_files", [])),
        "derived_artifacts": len(manifest.get("derived_artifacts", [])),
        "unlisted_package_files": len(unlisted),
        "failures": failures,
    }
    if failures:
        raise ValueError(f"Exterior-QNM manifest verification failed: {failures}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--control-dir", type=Path, default=CONTROL_ROOT)
    parser.add_argument(
        "--legacy-candidate-dir", type=Path, default=LEGACY_CANDIDATE_ROOT
    )
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        print(
            json.dumps(
                verify_manifest(arguments.output_dir / MANIFEST_NAME), indent=2
            )
        )
    else:
        print(
            create_manifest(
                arguments.output_dir,
                arguments.repository_root,
                control_dir=arguments.control_dir,
                legacy_dir=arguments.legacy_candidate_dir,
            )
        )


if __name__ == "__main__":
    main()
