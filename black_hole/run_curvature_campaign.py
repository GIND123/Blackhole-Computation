"""Durable local orchestrator for curvature-coupling production cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .curvature_coupling_production import archive_path, case_catalogue


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_one(
    python: Path,
    source_root: Path,
    output_dir: Path,
    logs_dir: Path,
    name: str,
) -> dict:
    catalogue = case_catalogue()
    case = catalogue[name]
    destination = archive_path(output_dir, case)
    if destination.exists():
        return {
            "case": name,
            "status": "skipped_existing",
            "archive": destination.as_posix(),
            "returncode": 0,
            "started_at": _timestamp(),
            "finished_at": _timestamp(),
            "wall_seconds": 0.0,
        }
    reservation = destination.with_suffix(".running")
    checkpoint = destination.with_suffix(".checkpoint.npz")
    command = [
        python.as_posix(),
        "-m",
        "black_hole.curvature_coupling_production",
        name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    if reservation.exists():
        if checkpoint.exists() and case.group == "tail":
            command.append("--resume-interrupted")
        else:
            return {
                "case": name,
                "status": "blocked_by_reservation",
                "archive": destination.as_posix(),
                "returncode": 2,
                "started_at": _timestamp(),
                "finished_at": _timestamp(),
                "wall_seconds": 0.0,
            }
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = _timestamp()
    clock = time.monotonic()
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"[{started}] command: {' '.join(command)}\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=source_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "case": name,
        "status": "completed" if completed.returncode == 0 else "failed",
        "archive": destination.as_posix(),
        "log": log_path.as_posix(),
        "returncode": completed.returncode,
        "started_at": started,
        "finished_at": _timestamp(),
        "wall_seconds": time.monotonic() - clock,
    }


def _run_group(
    *,
    group: str,
    names: list[str] | None,
    workers: int,
    python: Path,
    source_root: Path,
    output_dir: Path,
    logs_dir: Path,
    status: dict,
    status_path: Path,
) -> bool:
    catalogue = case_catalogue()
    if names is None:
        names = [name for name, case in catalogue.items() if case.group == group]
    status["phases"][group] = {
        "started_at": _timestamp(),
        "workers": workers,
        "cases": names,
        "status": "running",
    }
    _write_status(status_path, status)
    passed = True
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                python,
                source_root,
                output_dir,
                logs_dir,
                name,
            ): name
            for name in names
        }
        for future in as_completed(futures):
            result = future.result()
            status["results"][result["case"]] = result
            passed &= result["returncode"] == 0
            _write_status(status_path, status)
    phase = status["phases"][group]
    phase["finished_at"] = _timestamp()
    phase["status"] = "completed" if passed else "failed"
    _write_status(status_path, status)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/curvature_coupling_production_v2"),
    )
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--group", choices=("qnm", "tail", "all"), default="all")
    parser.add_argument("--qnm-workers", type=int, default=4)
    parser.add_argument("--tail-workers", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.qnm_workers < 1 or arguments.tail_workers < 1:
        raise ValueError("Worker counts must be positive.")
    source_root = arguments.source_root.resolve()
    output_dir = arguments.output_dir.resolve()
    logs_dir = (
        arguments.logs_dir.resolve()
        if arguments.logs_dir is not None
        else output_dir / "logs"
    )
    status_path = output_dir / "campaign_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["resumed_at"] = _timestamp()
    else:
        status = {
            "schema_version": 1,
            "started_at": _timestamp(),
            "campaign_id": os.environ.get("SDS_CAMPAIGN_ID"),
            "campaign_source_sha256": os.environ.get(
                "SDS_CAMPAIGN_SOURCE_SHA256"
            ),
            "python": arguments.python.resolve().as_posix(),
            "source_root": source_root.as_posix(),
            "output_dir": output_dir.as_posix(),
            "phases": {},
            "results": {},
        }
    _write_status(status_path, status)
    catalogue = case_catalogue()
    unknown = sorted(set(arguments.cases) - set(catalogue))
    if unknown:
        raise ValueError(f"Unknown curvature-coupling cases: {unknown}")
    if arguments.cases:
        selected_by_group = {
            group: [
                name for name in arguments.cases if catalogue[name].group == group
            ]
            for group in ("qnm", "tail")
        }
        groups = tuple(
            group for group in ("qnm", "tail") if selected_by_group[group]
        )
        if arguments.group != "all" and any(
            group != arguments.group for group in groups
        ):
            raise ValueError("Named cases must belong to the selected group.")
    else:
        selected_by_group = {"qnm": None, "tail": None}
        groups = (
            ("qnm", "tail")
            if arguments.group == "all"
            else (arguments.group,)
        )
    passed = True
    for group in groups:
        group_passed = _run_group(
            group=group,
            names=selected_by_group[group],
            workers=(
                arguments.qnm_workers if group == "qnm" else arguments.tail_workers
            ),
            python=arguments.python.resolve(),
            source_root=source_root,
            output_dir=output_dir,
            logs_dir=logs_dir,
            status=status,
            status_path=status_path,
        )
        passed &= group_passed
        if not group_passed:
            break
    status["finished_at"] = _timestamp()
    status["status"] = "completed" if passed else "failed"
    _write_status(status_path, status)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
