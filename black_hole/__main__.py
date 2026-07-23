"""Command-line interface for the black-hole perturbation project."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path

from .analysis import create_plots, write_diagnostics
from .flat_limit_study import run_flat_limit_study
from .model import InitialData, ModelParameters
from .sds_analysis import create_sds_plots, write_sds_diagnostics
from .sds_model import (
    ArealBumpInitialData,
    ArealVelocityBumpInitialData,
    BRIDGE_CHOICES,
    ScalarInitialData,
    SdSParameters,
)
from .sds_solver import (
    SdSNumericalParameters,
    load_sds_result,
    run_sds_simulation,
)
from .sds_study import run_sds_bridge_suite, run_sds_convergence_study
from .solver import NumericalParameters, load_result, run_simulation
from .study import run_convergence_study
from .tail_study import run_tail_study
from .tail_validation import (
    create_profile_sensitivity_report,
    create_resolution_report,
    create_timestep_report,
)


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


def sds_model_from_args(args: argparse.Namespace) -> SdSParameters:
    return SdSParameters(
        mass=args.mass,
        cosmological_length=args.cosmological_length,
        ell=args.ell,
    )


def sds_initial_from_args(args: argparse.Namespace) -> ScalarInitialData:
    return ScalarInitialData(
        center_fraction=args.center_fraction,
        width=args.width,
        time_symmetric=not args.pi_zero,
    )


def flat_limit_initial_from_args(args: argparse.Namespace) -> ArealBumpInitialData:
    return ArealBumpInitialData(
        center_radius=args.center_radius,
        support_half_width=args.support_half_width,
        time_symmetric=True,
    )


def tail_initial_from_args(args: argparse.Namespace) -> ArealVelocityBumpInitialData:
    return ArealVelocityBumpInitialData(
        center_radius=args.center_radius,
        support_half_width=args.support_half_width,
        amplitude=args.amplitude,
    )


def sds_numerical_from_args(args: argparse.Namespace) -> SdSNumericalParameters:
    return SdSNumericalParameters(
        resolution=args.resolution,
        timestep=args.timestep,
        end_time=args.end_time,
        signal_dt=args.signal_dt,
        snapshot_dt=args.snapshot_dt,
        timestepper=args.timestepper,
        bridge=args.bridge,
        dealias=args.dealias,
    )


def add_sds_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--cosmological-length", type=float, default=10.0)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--center-fraction", type=float, default=0.45)
    parser.add_argument(
        "--width",
        type=float,
        default=0.06,
        help="Gaussian width in the common compact coordinate rho",
    )
    parser.add_argument(
        "--pi-zero",
        action="store_true",
        help="set pi=0 instead of time-symmetric physical initial data",
    )


def add_sds_numerical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--timestep", type=float, default=0.02)
    parser.add_argument("--end-time", type=float, default=250.0)
    parser.add_argument("--signal-dt", type=float, default=0.05)
    parser.add_argument("--snapshot-dt", type=float, default=0.5)
    parser.add_argument(
        "--timestepper",
        choices=("RK111", "RK222", "RK443", "SBDF2", "SBDF4"),
        default="RK222",
    )
    parser.add_argument("--bridge", choices=BRIDGE_CHOICES, default="minimal")
    parser.add_argument(
        "--dealias",
        type=float,
        default=1.5,
        help="Chebyshev dealias scale for variable-coefficient products",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolve black-hole perturbations and bridge-coordinate wave models."
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

    sds_simulate = subparsers.add_parser(
        "sds-simulate", help="run one Schwarzschild-de Sitter scalar evolution"
    )
    add_sds_model_arguments(sds_simulate)
    add_sds_numerical_arguments(sds_simulate)
    sds_simulate.add_argument("--output", type=Path, required=True)

    sds_analyze = subparsers.add_parser(
        "sds-analyze", help="analyze a saved Schwarzschild-de Sitter scalar run"
    )
    sds_analyze.add_argument("--input", type=Path, required=True)
    sds_analyze.add_argument("--output-dir", type=Path, required=True)

    sds_suite = subparsers.add_parser(
        "sds-suite", help="run bridge-coordinate scalar-wave comparisons"
    )
    add_sds_model_arguments(sds_suite)
    add_sds_numerical_arguments(sds_suite)
    sds_suite.add_argument("--output-dir", type=Path, default=Path("results/sds_scalar"))
    sds_suite.add_argument(
        "--bridges",
        nargs="+",
        choices=BRIDGE_CHOICES,
        default=list(BRIDGE_CHOICES),
    )
    sds_suite.add_argument("--convergence-bridge", choices=BRIDGE_CHOICES, default="minimal")
    sds_suite.add_argument("--convergence-end-time", type=float, default=120.0)

    flat_limit = subparsers.add_parser(
        "sds-flat-limit",
        help="compare finite-L SdS horizon signals with Schwarzschild scri+",
    )
    flat_limit.add_argument("--mass", type=float, default=1.0)
    flat_limit.add_argument("--ell", type=int, default=2)
    flat_limit.add_argument(
        "--lengths", nargs="+", type=float, default=[20.0, 40.0, 80.0, 160.0]
    )
    flat_limit.add_argument("--center-radius", type=float, default=4.0)
    flat_limit.add_argument(
        "--support-half-width",
        type=float,
        default=1.5,
        help="half-width in areal radius of the smooth compact pulse",
    )
    flat_limit.add_argument("--resolution", type=int, default=256)
    flat_limit.add_argument("--timestep", type=float, default=0.01)
    flat_limit.add_argument("--end-time", type=float, default=200.0)
    flat_limit.add_argument("--signal-dt", type=float, default=0.05)
    flat_limit.add_argument("--snapshot-dt", type=float, default=0.5)
    flat_limit.add_argument(
        "--timestepper",
        choices=("RK111", "RK222", "RK443", "SBDF2", "SBDF4"),
        default="RK222",
    )
    flat_limit.set_defaults(bridge="minimal")
    flat_limit.add_argument("--reference-radius", type=float, default=4.0)
    flat_limit.add_argument(
        "--convergence-lengths",
        nargs="+",
        type=float,
        default=[20.0, 160.0],
    )
    flat_limit.add_argument("--convergence-end-time", type=float, default=100.0)
    flat_limit.add_argument("--skip-convergence", action="store_true")
    flat_limit.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sds_scalar/flat_limit"),
    )
    flat_limit.add_argument("--dealias", type=float, default=1.5)

    tails = subparsers.add_parser(
        "sds-tail-study",
        help="validate Schwarzschild Price laws and finite-L SdS tails",
    )
    tails.add_argument("--mass", type=float, default=1.0)
    tails.add_argument("--ells", nargs="+", type=int, default=[0, 1, 2])
    tails.add_argument(
        "--lengths", nargs="+", type=float, default=[20.0, 40.0, 80.0, 160.0]
    )
    tails.add_argument("--center-radius", type=float, default=6.0)
    tails.add_argument(
        "--support-half-width",
        type=float,
        default=3.0,
        help="areal-radius half-width of the physical velocity bump",
    )
    tails.add_argument("--amplitude", type=float, default=1.0)
    add_sds_numerical_arguments(tails)
    tails.set_defaults(
        bridge="minimal",
        resolution=1024,
        timestep=0.005,
        end_time=120.0,
        signal_dt=0.05,
        snapshot_dt=2.0,
    )
    tails.add_argument(
        "--ell2-resolution",
        type=int,
        default=2048,
        help="targeted ell=2 Chebyshev resolution",
    )
    tails.add_argument(
        "--ell2-timestep",
        type=float,
        default=0.0025,
        help="targeted ell=2 timestep",
    )
    tails.add_argument("--reference-radius", type=float, default=4.0)
    tails.add_argument(
        "--cosmological-timescales",
        type=float,
        default=5.0,
        help="evolve each SdS case for this many inverse surface gravities",
    )
    tails.add_argument(
        "--window-width",
        type=float,
        default=20.0,
        help="width in M of the sliding relative-waveform comparison",
    )
    tails.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse matching raw paths instead of rerunning evolutions",
    )
    tails.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sds_scalar/tails"),
    )

    tail_resolution = subparsers.add_parser(
        "sds-tail-resolution-report",
        help="compare saved tail runs at increasing resolutions",
    )
    tail_resolution.add_argument("--ell", type=int, required=True)
    tail_resolution.add_argument("--length", type=float, required=True)
    tail_resolution.add_argument(
        "--reference-inputs", nargs="+", type=Path, required=True
    )
    tail_resolution.add_argument(
        "--sds-inputs", nargs="+", type=Path, required=True
    )
    tail_resolution.add_argument("--reference-radius", type=float, default=4.0)
    tail_resolution.add_argument("--output-dir", type=Path, required=True)

    tail_profile = subparsers.add_parser(
        "sds-tail-profile-report",
        help="compare saved tail runs for different physical pulse widths",
    )
    tail_profile.add_argument("--ell", type=int, required=True)
    tail_profile.add_argument("--length", type=float, required=True)
    tail_profile.add_argument("--half-widths", nargs="+", type=float, required=True)
    tail_profile.add_argument(
        "--reference-inputs", nargs="+", type=Path, required=True
    )
    tail_profile.add_argument("--sds-inputs", nargs="+", type=Path, required=True)
    tail_profile.add_argument("--reference-radius", type=float, default=4.0)
    tail_profile.add_argument("--output-dir", type=Path, required=True)

    tail_timestep = subparsers.add_parser(
        "sds-tail-timestep-report",
        help="compare saved tail runs at fixed resolution and refined timesteps",
    )
    tail_timestep.add_argument("--ell", type=int, required=True)
    tail_timestep.add_argument("--length", type=float, required=True)
    tail_timestep.add_argument(
        "--reference-inputs", nargs="+", type=Path, required=True
    )
    tail_timestep.add_argument("--sds-inputs", nargs="+", type=Path, required=True)
    tail_timestep.add_argument("--reference-radius", type=float, default=4.0)
    tail_timestep.add_argument("--output-dir", type=Path, required=True)
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

    if args.command == "sds-simulate":
        result = run_sds_simulation(
            sds_model_from_args(args),
            sds_initial_from_args(args),
            sds_numerical_from_args(args),
        )
        result.save(args.output)
        print(f"saved {args.output}")
        return

    if args.command == "sds-analyze":
        result = load_sds_result(args.input)
        diagnostics = write_sds_diagnostics(result, args.output_dir)
        create_sds_plots(result, args.output_dir)
        print(json.dumps(diagnostics, indent=2))
        return

    if args.command == "sds-suite":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        model_sds = sds_model_from_args(args)
        initial_sds = sds_initial_from_args(args)
        numerical_sds = sds_numerical_from_args(args)
        suite = run_sds_bridge_suite(
            model_sds,
            initial_sds,
            numerical_sds,
            output_dir,
            bridges=tuple(args.bridges),
        )
        convergence = run_sds_convergence_study(
            model_sds,
            initial_sds,
            numerical_sds,
            output_dir / "convergence",
            bridge=args.convergence_bridge,
            end_time=args.convergence_end_time,
        )
        print(
            json.dumps(
                {
                    "bridges": list(suite.keys()),
                    "convergence": convergence,
                },
                indent=2,
            )
        )
        return

    if args.command == "sds-flat-limit":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        results = run_flat_limit_study(
            mass=args.mass,
            ell=args.ell,
            lengths=tuple(args.lengths),
            initial=flat_limit_initial_from_args(args),
            numerical=sds_numerical_from_args(args),
            output_dir=output_dir,
            reference_radius=args.reference_radius,
            convergence_lengths=tuple(args.convergence_lengths),
            convergence_end_time=args.convergence_end_time,
            run_convergence=not args.skip_convergence,
        )
        print(json.dumps(results, indent=2))
        return

    if args.command == "sds-tail-study":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        numerical = sds_numerical_from_args(args)
        results = run_tail_study(
            mass=args.mass,
            ells=tuple(args.ells),
            lengths=tuple(args.lengths),
            initial=tail_initial_from_args(args),
            numerical=numerical,
            output_dir=output_dir,
            reference_radius=args.reference_radius,
            cosmological_timescales=args.cosmological_timescales,
            window_width=args.window_width,
            reuse_existing=args.reuse_existing,
            ell2_numerical=replace(
                numerical,
                resolution=args.ell2_resolution,
                timestep=args.ell2_timestep,
            ),
        )
        print(json.dumps(results, indent=2))
        return

    if args.command == "sds-tail-resolution-report":
        rows = create_resolution_report(
            ell=args.ell,
            length=args.length,
            reference_paths=tuple(args.reference_inputs),
            sds_paths=tuple(args.sds_inputs),
            output_dir=args.output_dir,
            reference_radius=args.reference_radius,
        )
        print(json.dumps(rows, indent=2))
        return

    if args.command == "sds-tail-profile-report":
        if not (
            len(args.half_widths)
            == len(args.reference_inputs)
            == len(args.sds_inputs)
        ):
            parser.error(
                "--half-widths, --reference-inputs, and --sds-inputs "
                "must contain the same number of entries"
            )
        rows = create_profile_sensitivity_report(
            ell=args.ell,
            length=args.length,
            cases=tuple(
                zip(args.half_widths, args.reference_inputs, args.sds_inputs)
            ),
            output_dir=args.output_dir,
            reference_radius=args.reference_radius,
        )
        print(json.dumps(rows, indent=2))
        return

    if args.command == "sds-tail-timestep-report":
        rows = create_timestep_report(
            ell=args.ell,
            length=args.length,
            reference_paths=tuple(args.reference_inputs),
            sds_paths=tuple(args.sds_inputs),
            output_dir=args.output_dir,
            reference_radius=args.reference_radius,
        )
        print(json.dumps(rows, indent=2))
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
