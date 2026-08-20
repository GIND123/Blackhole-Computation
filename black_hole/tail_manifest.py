"""Hash verified manifest for the large cosmological length tail package.

The regulator package separates immutable raw simulations from derived
analysis artifacts and records both with their code provenance.  This does the
same for the tail campaign, with one difference that the campaign forces.

The screening archives were produced before the explicit potential split was
committed, so each of them records a dirty worktree.  They decided which
length to carry forward and nothing else; the implicit split set they replaced
is kept beside them under ``results/large_l_tail_implicit_split``.  Refusing
to record them would leave the package undocumented, and recording them
silently beside the final ladder would present them as production data.  Each
archive therefore carries a ``provenance_grade``: ``production`` when its own
run recorded a clean worktree, ``screening`` when it did not.  A strict build
refuses only when a final ladder archive is not production grade, which is the
grade the paper quotes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import large_l_tail as tail
from .regulator_manifest import _hash_record, sha256_bytes, sha256_lf


#: Suffixes of derived artifacts recorded beside the raw archives.
DERIVED_SUFFIXES = {".csv", ".json", ".png", ".pdf"}
#: Transient files a running or interrupted case leaves behind.
TRANSIENT_SUFFIXES = (".checkpoint.npz", ".running", ".incomplete.npz")


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def physical_contract() -> dict:
    """Return the frozen physical and coordinate choices of the campaign."""

    return {
        "mass": 1.0,
        "gauge": "minimal bridge",
        "initial_data": asdict(tail.INITIAL_DATA),
        "initial_field": "u = psi = 0; pi = G/A with G the areal velocity bump",
        "retarded_time": "U = tau - q_L with q_L evaluated analytically",
        "time_translation_fitted": False,
        "reference_normalization_radius_over_M": tail.REFERENCE_RADIUS,
        "finite_radius_observers_over_M": list(tail.FINITE_OBSERVERS),
        "primary_observer": "cosmological horizon, compared with Schwarzschild scri+",
        "price_target_at_infinity": tail.PRICE_TARGET_INFINITY,
        "price_target_at_fixed_radius": tail.PRICE_TARGET_FIXED_RADIUS,
        "price_relative_tolerance": tail.PRICE_TOLERANCE,
        "price_continuous_duration_over_M": tail.PRICE_DURATION,
        "cosmological_relative_tolerance": tail.COSMOLOGICAL_TOLERANCE,
        "cosmological_minimum_scaled_duration": (
            tail.COSMOLOGICAL_MINIMUM_SCALED_DURATION
        ),
        "imex_split": (
            "explicit_potential" if tail.EXPLICIT_POTENTIAL else "implicit_potential"
        ),
    }


def contract_sha256() -> str:
    payload = json.dumps(physical_contract(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_label(metadata: dict) -> str:
    """Return the IMEX split as one word.

    Archives written by the current solver carry a dict describing both sides
    of the split; the field is absent altogether on archives written before it
    existed, which were all integrated with the potential term implicit.
    """

    split = metadata.get("imex_split")
    if isinstance(split, dict):
        return f"{split.get('potential_term', 'implicit')}_potential"
    return split or "implicit_potential"


def _is_transient(path: Path) -> bool:
    name = path.name
    return any(name.endswith(suffix) for suffix in TRANSIENT_SUFFIXES)


def _archive_rows(output_dir: Path, root: Path) -> list[dict]:
    rows: list[dict] = []
    for archive in sorted((output_dir / "raw").glob("*.npz")):
        if _is_transient(archive):
            continue
        with np.load(archive, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        provenance = metadata["reproducibility"]
        numerical = metadata["numerical"]
        name = archive.stem
        rows.append(
            {
                "path": archive.relative_to(root).as_posix(),
                "case": name,
                "role": "final" if name.startswith("final_") else "screen",
                "provenance_grade": (
                    "screening" if provenance["git_worktree_dirty"] else "production"
                ),
                "simulation_commit": provenance["git_commit"],
                "git_worktree_dirty": bool(provenance["git_worktree_dirty"]),
                "background": metadata["background"],
                "cosmological_length_over_M": metadata.get("cosmological_length"),
                "resolution": numerical["resolution"],
                "timestep_over_M": numerical["timestep"],
                "end_time_over_M": numerical["end_time"],
                "imex_split": _split_label(metadata),
                "maximum_constraint_linf": float(
                    np.max(np.abs(np.load(archive, allow_pickle=False)["constraint_linf"]))
                ),
                "bytes": archive.stat().st_size,
                "command": (
                    "python -m black_hole.large_l_tail --output-dir "
                    f"{output_dir.as_posix()} run {name}"
                ),
                **_hash_record(archive),
            }
        )
    return rows


def _derived_rows(output_dir: Path, root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DERIVED_SUFFIXES:
            continue
        if "raw" in path.relative_to(output_dir).parts or _is_transient(path):
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                **_hash_record(path),
            }
        )
    return rows


def create_manifest(
    output_dir: Path,
    repository_root: Path | None = None,
    *,
    strict: bool = True,
) -> Path:
    """Write the manifest and return its path."""

    repository_root = Path(repository_root or Path.cwd()).resolve()
    output_dir = Path(output_dir).resolve()
    label = output_dir.relative_to(repository_root).as_posix()

    archives = _archive_rows(output_dir, repository_root)
    if not archives:
        raise FileNotFoundError("No tail archives were found.")

    analysis_commit = _git_output("rev-parse", "HEAD")
    tracked = _git_output(
        "status", "--porcelain", "--untracked-files=no", "--",
        ".", f":(exclude){label}/**",
    )
    if tracked:
        raise ValueError(
            "Analysis source must be clean; only regenerated artifacts inside "
            "the output directory may differ from the analysis commit."
        )

    final_screening = [
        row["case"]
        for row in archives
        if row["role"] == "final" and row["provenance_grade"] != "production"
    ]
    if strict and final_screening:
        raise ValueError(
            "Final ladder archives must record a clean worktree: "
            f"{sorted(final_screening)}"
        )

    manifest = {
        "schema_version": 2,
        "package": label,
        "output_directory": label,
        "analysis_commit": analysis_commit,
        "simulation_commits": sorted({row["simulation_commit"] for row in archives}),
        "physical_contract": physical_contract(),
        "physical_contract_sha256": contract_sha256(),
        "provenance_grades": {
            grade: sum(1 for row in archives if row["provenance_grade"] == grade)
            for grade in ("production", "screening")
        },
        "screening_grade_note": (
            "Screening archives were written before the explicit potential "
            "split was committed and record a dirty worktree. They select the "
            "cosmological length and are not quoted as production data. The "
            "implicit split set they replaced is kept under "
            "results/large_l_tail_implicit_split."
        ),
        "manifest_commands": {
            "create": (
                "python -m black_hole.tail_manifest --output-dir " + label
            ),
            "verify": (
                "python -m black_hole.tail_manifest --output-dir " + label
                + " --verify"
            ),
        },
        "archives": archives,
        "derived_artifacts": _derived_rows(output_dir, repository_root),
    }

    destination = output_dir / "manifest.json"
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return destination


def _present_artifacts(output_dir: Path, root: Path) -> set[str]:
    """Return every non transient artifact the output directory holds."""

    present: set[str] = set()
    for archive in (output_dir / "raw").glob("*.npz"):
        if not _is_transient(archive):
            present.add(archive.relative_to(root).as_posix())
    for candidate in output_dir.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in DERIVED_SUFFIXES:
            continue
        if "raw" in candidate.relative_to(output_dir).parts or _is_transient(candidate):
            continue
        present.add(candidate.relative_to(root).as_posix())
    return present


def verify_manifest(path: Path) -> dict:
    """Check the recorded hashes and that nothing is present but unrecorded.

    Hashes alone cannot detect an archive that was produced and then never
    listed, which is the failure that matters when a campaign is extended: the
    manifest still verifies while the package silently holds results nobody
    recorded.  Completeness is therefore checked in both directions.
    """

    path = Path(path).resolve()
    root = Path.cwd().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for section in ("archives", "derived_artifacts"):
        for row in manifest[section]:
            artifact = root / row["path"]
            if artifact.resolve() == path.resolve():
                continue
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
    if manifest["physical_contract_sha256"] != contract_sha256():
        failures.append("physical_contract")

    listed = {
        row["path"]
        for section in ("archives", "derived_artifacts")
        for row in manifest[section]
    }
    listed.add(path.relative_to(root).as_posix())
    # Schema 1 manifests predate the recorded output directory.
    output_dir = root / manifest.get("output_directory", manifest["package"])
    unlisted = sorted(_present_artifacts(output_dir, root) - listed)
    failures.extend(f"unlisted:{item}" for item in unlisted)

    result = {
        "verified": not failures,
        "checked_files": checked,
        "listed_files": len(listed),
        "unlisted_files": len(unlisted),
        "failures": failures,
    }
    if failures:
        raise ValueError(f"Manifest verification failed: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=tail.OUTPUT_ROOT)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--allow-screening-final",
        action="store_true",
        help="record final archives that do not carry production provenance",
    )
    arguments = parser.parse_args()
    if arguments.verify:
        print(
            json.dumps(
                verify_manifest(arguments.output_dir / "manifest.json"), indent=2
            )
        )
    else:
        print(
            create_manifest(
                arguments.output_dir,
                arguments.repository_root,
                strict=not arguments.allow_screening_final,
            )
        )


if __name__ == "__main__":
    main()
