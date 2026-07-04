"""Command-line interface for the black-hole perturbation project."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .analysis import create_plots, write_diagnostics
from .model import InitialData, ModelParameters
from .solver import NumericalParameters, load_result, run_simulation
from .study import run_convergence_study


def numerical_from_args(args: argparse.Namespace) -> NumericalParameters:
    return NumericalParameters(
        resolution=args.resolution,
        timestep=args.timestep,
        end_time=args.end_time,
        signal_dt=args.signal_dt,
        snapshot_dt=args.snapshot_dt,
        timestepper=args.timestepper,
    )


def add_numerical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--timestep", type=float, default=0.02)
    parser.add_argument("--end-time", type=float, default=1000.0)
    parser.add_argument("--signal-dt", type=float, default=0.1)
    parser.add_argument("--snapshot-dt", type=float, default=1.0)
    parser.add_argument(
        "--timestepper",
        choices=("RK111", "RK222", "RK443", "SBDF2", "SBDF4"),
        default="RK222",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolve gravitational perturbations of a Schwarzschild black hole."
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="run one Dedalus evolution")
    add_numerical_arguments(simulate)
    simulate.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="analyze a saved evolution")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--ringdown-start", type=float, default=40)
    analyze.add_argument("--ringdown-end", type=float, default=100)
    analyze.add_argument("--tail-start", type=float, default=400)
    analyze.add_argument("--tail-end", type=float, default=900)

    study = subparsers.add_parser(
        "study", help="run the primary evolution and both convergence studies"
    )
    add_numerical_arguments(study)
    study.add_argument("--output-dir", type=Path, default=Path("results/black_hole"))
    study.add_argument("--convergence-end-time", type=float, default=200)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    model = ModelParameters()
    initial = InitialData()

    if args.command == "simulate":
        result = run_simulation(model, initial, numerical_from_args(args))
        result.save(args.output)
        print(f"saved {args.output}")
        return

    if args.command == "analyze":
        result = load_result(args.input)
        diagnostics = write_diagnostics(
            result,
            args.output_dir,
            (args.ringdown_start, args.ringdown_end),
            (args.tail_start, args.tail_end),
        )
        create_plots(result, args.output_dir)
        print(json.dumps(diagnostics, indent=2))
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_simulation(model, initial, numerical_from_args(args))
    raw_path = output_dir / "primary_evolution.npz"
    result.save(raw_path)
    diagnostics = write_diagnostics(result, output_dir)
    create_plots(result, output_dir)
    convergence = run_convergence_study(
        model,
        initial,
        numerical_from_args(args),
        output_dir / "convergence",
        end_time=args.convergence_end_time,
    )
    print(json.dumps({"diagnostics": diagnostics, "convergence": convergence}, indent=2))


if __name__ == "__main__":
    main()
