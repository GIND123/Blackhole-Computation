"""Parallel orchestrator for the staged exterior-tail repair tests."""

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

from .exterior_tail_repair import archive_path, case_catalogue


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def primary_conformal_cases() -> list[str]:
    return [
        name
        for name, case in case_catalogue().items()
        if case.curvature_coupling == 1.0 / 6.0
        and case.formulation in {"standard", "conservative", "factored_gamma1"}
        and case.timestep == 0.0025
    ]


def _run_one(
    python: Path,
    source_root: Path,
    output_dir: Path,
    logs_dir: Path,
    name: str,
) -> dict:
    case = case_catalogue()[name]
    destination = archive_path(output_dir, case)
    started = _timestamp()
    if destination.exists():
        return {
            "case": name,
            "status": "skipped_existing",
            "returncode": 0,
            "archive": destination.as_posix(),
            "started_at": started,
            "finished_at": _timestamp(),
            "wall_seconds": 0.0,
        }
    logs_dir.mkdir(parents=True, exist_ok=True)
    log = logs_dir / f"{name}.log"
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
    command = [
        python.as_posix(),
        "-m",
        "black_hole.exterior_tail_repair",
        name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    clock = time.monotonic()
    with log.open("w", encoding="utf-8", newline="\n") as stream:
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
        "returncode": completed.returncode,
        "archive": destination.as_posix(),
        "log": log.as_posix(),
        "started_at": started,
        "finished_at": _timestamp(),
        "wall_seconds": time.monotonic() - clock,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--subset", choices=("primary_conformal", "all"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/exterior_tail_repair_v1")
    )
    parser.add_argument("--logs-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("Worker count must be positive.")
    if arguments.cases:
        selected = list(arguments.cases)
    elif arguments.subset == "primary_conformal":
        selected = primary_conformal_cases()
    elif arguments.subset == "all":
        selected = list(case_catalogue())
    else:
        raise ValueError("Supply cases or select a campaign subset.")
    unknown = sorted(set(selected) - set(case_catalogue()))
    if unknown:
        raise ValueError(f"Unknown repair cases: {unknown}")

    source_root = arguments.source_root.resolve()
    output_dir = arguments.output_dir.resolve()
    logs_dir = (
        arguments.logs_dir.resolve()
        if arguments.logs_dir is not None
        else output_dir / "logs"
    )
    status_path = output_dir / "campaign_status.json"
    status = {
        "schema_version": 1,
        "started_at": _timestamp(),
        "source_root": source_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "workers": arguments.workers,
        "selected_cases": selected,
        "status": "running",
        "results": {},
    }
    _write_status(status_path, status)
    passed = True
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                arguments.python.resolve(),
                source_root,
                output_dir,
                logs_dir,
                name,
            ): name
            for name in selected
        }
        for future in as_completed(futures):
            result = future.result()
            status["results"][result["case"]] = result
            passed &= result["returncode"] == 0
            _write_status(status_path, status)
    status["finished_at"] = _timestamp()
    status["status"] = "completed" if passed else "failed"
    _write_status(status_path, status)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
