"""Compare exterior-supported SdS waves with frozen regulator controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from matplotlib.ticker import NullLocator

from .exterior_regulator_suite import (
    CONTROL_ROOT,
    EXTERIOR_LENGTHS,
    LEVELS,
    OUTPUT_ROOT,
    archive_path,
    contract_sha256,
)
from .regulator_analysis import (
    ANALYSIS_WINDOWS,
    CUMULATIVE_WINDOWS,
    WINDOW_LATEX_LABELS,
    _align_flat,
    _effective_order,
    _flat_signal,
    _l2,
    _retarded_times,
    _window_family,
    _window_mask,
    waveform_metrics,
)
from .sds_result import SdSSimulationResult, load_sds_result
from .exterior_sds_model import (
    ExteriorSdSParameters,
    metric_f_prime as exterior_metric_f_prime,
    transition_radii,
)
from .sds_model import (
    SdSParameters,
    metric_f_prime as uniform_metric_f_prime,
    sds_horizons,
)


RESOLUTIONS = {"coarse": 384, "medium": 512, "fine": 768}
HEADLINE_WINDOW = "radiative_signal"
SUBSTANTIAL_REDUCTION_FACTOR = 0.75


def schwarzschild_potential_integral(mass: float, ell: int) -> float:
    r"""Return ``int V_S dr_*`` from ``2M`` to Schwarzschild scri."""

    return (ell * (ell + 1.0) / 2.0 + 0.25) / mass


def leading_transfer_coefficient(
    family: str, length: float, *, mass: float = 1.0, ell: int = 2
) -> dict[str, float]:
    r"""Return the fit-free first-Born finite-horizon transfer coefficient.

    For the reduced scalar mode, ``V dr_*`` equals
    ``[ell(ell+1)/r^2+f'(r)/r] dr``.  If ``I_L`` is its integral between the
    finite-background horizons and ``I_0`` is the Schwarzschild integral to
    scri, the leading outgoing transfer is

    ``W_L = W_0 - (I_L-I_0)/2 * partial_U^{-1} W_0 + ...``.

    The coefficient therefore depends only on the background; no waveform
    lag, amplitude, or transfer parameter is fitted.
    """

    reference = schwarzschild_potential_integral(mass, ell)
    if family == "uniform_sds":
        parameters = SdSParameters(
            mass=mass, cosmological_length=length, ell=ell
        )
        horizons = sds_horizons(parameters)
        inner = horizons.black_hole
        outer = horizons.cosmological
        points = None

        def integrand(radius: float) -> float:
            return (
                ell * (ell + 1.0) / radius**2
                + float(uniform_metric_f_prime(np.array(radius), parameters))
                / radius
            )

    elif family == "exterior_sds":
        parameters = ExteriorSdSParameters(
            mass=mass, cosmological_length=length, ell=ell
        )
        inner = parameters.black_hole_horizon
        outer = parameters.cosmological_horizon
        points = list(transition_radii(parameters))

        def integrand(radius: float) -> float:
            return (
                ell * (ell + 1.0) / radius**2
                + float(exterior_metric_f_prime(np.array(radius), parameters))
                / radius
            )

    else:
        raise ValueError(f"Unknown finite-horizon family {family!r}.")

    finite, quadrature_error = quad(
        integrand,
        inner,
        outer,
        points=points,
        epsabs=2.0e-12 / mass,
        epsrel=2.0e-12,
        limit=500,
    )
    coefficient = -0.5 * (finite - reference)
    return {
        "potential_integral_finite": float(finite),
        "potential_integral_schwarzschild": float(reference),
        "quadrature_absolute_error": float(quadrature_error),
        "leading_transfer_coefficient_M_inverse": float(coefficient),
        "coefficient_times_cosmological_horizon": float(coefficient * outer),
        "cosmological_horizon_over_M": float(outer / mass),
    }


def align_transfer_corrected(
    result: SdSSimulationResult,
    times: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    r"""Apply the causal leading transfer inverse and interpolate to ``times``."""

    source_times = _retarded_times(result)
    signal = _flat_signal(result)
    primitive = cumulative_trapezoid(signal, source_times, initial=0.0)
    corrected = signal - coefficient * primitive
    if times[0] < source_times[0] or times[-1] > source_times[-1]:
        raise ValueError("Requested corrected waveform lies outside its archive.")
    return np.interp(times, source_times, corrected)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return path
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _strict_json(path: Path, value: dict | list) -> Path:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if np.isfinite(item) else None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def load_control_archives(
    control_dir: Path,
    lengths: Iterable[int] = EXTERIOR_LENGTHS,
) -> dict[str, dict[str, dict[float | None, SdSSimulationResult]]]:
    """Load the frozen Schwarzschild and uniform-SdS controls read-only."""

    root = Path(control_dir) / "raw" / "flat"
    loaded: dict[str, dict[str, dict[float | None, SdSSimulationResult]]] = {}
    for level in LEVELS:
        loaded[level] = {
            "schwarzschild": {
                None: load_sds_result(root / level / "schwarzschild.npz")
            },
            "uniform_sds": {
                float(length): load_sds_result(root / level / f"sds_L{length}.npz")
                for length in lengths
            },
        }
    return loaded


def load_exterior_archives(
    output_dir: Path,
    lengths: Iterable[int] = EXTERIOR_LENGTHS,
) -> dict[str, dict[float, SdSSimulationResult]]:
    """Load the new exterior-supported archives from their isolated package."""

    return {
        level: {
            float(length): load_sds_result(
                archive_path(output_dir, level, float(length))
            )
            for length in lengths
        }
        for level in LEVELS
    }


def _validate_numerical_match(
    controls: dict,
    exterior: dict,
    lengths: Iterable[int],
) -> None:
    """Require like-for-like discretizations and physical initial data."""

    for level in LEVELS:
        reference = controls[level]["schwarzschild"][None]
        expected_numerical = reference.metadata["numerical"]
        expected_initial = reference.metadata["initial_data"]
        for length in lengths:
            candidates = (
                controls[level]["uniform_sds"][float(length)],
                exterior[level][float(length)],
            )
            for candidate in candidates:
                if candidate.metadata["numerical"] != expected_numerical:
                    raise ValueError(
                        f"Numerical settings do not match controls at {level}, "
                        f"L/M={length}."
                    )
                if candidate.metadata["initial_data"] != expected_initial:
                    raise ValueError(
                        f"Initial data do not match controls at {level}, "
                        f"L/M={length}."
                    )
            exterior_result = exterior[level][float(length)]
            provenance = exterior_result.metadata.get("simulation_provenance", {})
            if provenance.get("physical_contract_sha256") != contract_sha256():
                raise ValueError(
                    f"Exterior physical contract mismatch at {level}, L/M={length}."
                )
            model = exterior_result.metadata.get("model", {})
            if float(model.get("cosmological_length", np.nan)) != float(length):
                raise ValueError(
                    f"Exterior model length mismatch at {level}, L/M={length}."
                )
            offset = exterior_result.metadata.get("retarded_time_offset", {}).get("q")
            if offset is None or not np.isfinite(float(offset)):
                raise ValueError(
                    f"Missing finite exterior retarded-time offset at {level}, "
                    f"L/M={length}."
                )


def conservative_refinement_scale(
    coarse_medium: float,
    medium_fine: float,
    *,
    resolutions: tuple[int, int, int] = (384, 512, 768),
) -> dict[str, float]:
    """Return the observed order and conservative fine-grid error scale."""

    order = _effective_order(
        coarse_medium,
        medium_fine,
        resolutions[0],
        resolutions[1],
        resolutions[2],
    )
    ratio = resolutions[2] / resolutions[1]
    richardson = (
        medium_fine / (ratio**order - 1.0)
        if np.isfinite(order) and order > 0.0
        else np.nan
    )
    estimated = richardson if np.isfinite(richardson) else medium_fine
    return {
        "observed_coupled_order": float(order),
        "richardson_fine_E2": float(richardson),
        "estimated_fine_numerical_E2": float(estimated),
        "conservative_numerical_E2": float(max(medium_fine, estimated)),
    }


def improvement_with_margins(
    exterior_error: float,
    exterior_scale: float,
    uniform_error: float,
    uniform_scale: float,
    *,
    substantial_factor: float = SUBSTANTIAL_REDUCTION_FACTOR,
) -> dict[str, float | bool]:
    """Test whether exterior support improves agreement beyond grid uncertainty."""

    exterior_upper = exterior_error + exterior_scale
    uniform_lower = uniform_error - uniform_scale
    return {
        "exterior_upper_E2": exterior_upper,
        "uniform_lower_E2": uniform_lower,
        "resolved_improvement_with_numerical_margins": (
            exterior_upper < uniform_lower
        ),
        "resolved_reduction_at_least_25_percent": (
            exterior_upper <= substantial_factor * uniform_lower
        ),
    }


def analyze(
    output_dir: Path,
    control_dir: Path = CONTROL_ROOT,
    lengths: Iterable[int] = EXTERIOR_LENGTHS,
) -> dict:
    """Compute direct errors, paired refinement scales, and success tests."""

    lengths = tuple(int(length) for length in lengths)
    if not lengths:
        raise ValueError("At least one cosmological length is required.")
    if any(length not in EXTERIOR_LENGTHS for length in lengths):
        raise ValueError(f"Lengths must be selected from {EXTERIOR_LENGTHS}.")
    controls = load_control_archives(control_dir, lengths)
    exterior = load_exterior_archives(output_dir, lengths)
    _validate_numerical_match(controls, exterior, lengths)

    fine_reference_result = controls["fine"]["schwarzschild"][None]
    reference_times = _retarded_times(fine_reference_result)
    common = (reference_times >= 0.0) & (reference_times <= 160.0)
    times = reference_times[common]
    reference = _flat_signal(fine_reference_result)[common]

    fine_signals = {
        "uniform_sds": {
            length: _align_flat(
                controls["fine"]["uniform_sds"][float(length)], times
            )
            for length in lengths
        },
        "exterior_sds": {
            length: _align_flat(exterior["fine"][float(length)], times)
            for length in lengths
        },
    }
    transfer_coefficients = {
        family: {
            length: leading_transfer_coefficient(family, float(length))
            for length in lengths
        }
        for family in ("uniform_sds", "exterior_sds")
    }
    fine_corrected_signals = {
        "uniform_sds": {
            length: align_transfer_corrected(
                controls["fine"]["uniform_sds"][float(length)],
                times,
                transfer_coefficients["uniform_sds"][length][
                    "leading_transfer_coefficient_M_inverse"
                ],
            )
            for length in lengths
        },
        "exterior_sds": {
            length: align_transfer_corrected(
                exterior["fine"][float(length)],
                times,
                transfer_coefficients["exterior_sds"][length][
                    "leading_transfer_coefficient_M_inverse"
                ],
            )
            for length in lengths
        },
    }

    numerical_rows: list[dict] = []
    numerical_lookup: dict[tuple[str, int, str], float] = {}
    for family in ("uniform_sds", "exterior_sds"):
        for length in lengths:
            paired_residuals = {}
            for level in LEVELS:
                candidate_result = (
                    controls[level]["uniform_sds"][float(length)]
                    if family == "uniform_sds"
                    else exterior[level][float(length)]
                )
                candidate = _align_flat(candidate_result, times)
                level_reference = _align_flat(
                    controls[level]["schwarzschild"][None], times
                )
                paired_residuals[level] = candidate - level_reference

            for window, start, end in ANALYSIS_WINDOWS:
                mask = _window_mask(times, start, end)
                denominator = _l2(reference[mask], times[mask])
                coarse_medium = _l2(
                    paired_residuals["coarse"][mask]
                    - paired_residuals["medium"][mask],
                    times[mask],
                ) / denominator
                medium_fine = _l2(
                    paired_residuals["medium"][mask]
                    - paired_residuals["fine"][mask],
                    times[mask],
                ) / denominator
                refinement = conservative_refinement_scale(
                    coarse_medium, medium_fine
                )
                numerical_lookup[(family, length, window)] = refinement[
                    "conservative_numerical_E2"
                ]
                numerical_rows.append(
                    {
                        "background_family": family,
                        "cosmological_length_over_M": length,
                        "window": window,
                        "window_family": _window_family(window),
                        "window_start_U_over_M": start,
                        "window_end_U_over_M": end,
                        "coarse_medium_paired_E2": coarse_medium,
                        "medium_fine_paired_E2": medium_fine,
                        **refinement,
                        "paired_residual_definition": (
                            "(W_background-W_Schwarzschild)_q at each refinement"
                        ),
                    }
                )

    corrected_numerical_rows: list[dict] = []
    corrected_numerical_lookup: dict[tuple[str, int, str], float] = {}
    for family in ("uniform_sds", "exterior_sds"):
        for length in lengths:
            coefficient = transfer_coefficients[family][length][
                "leading_transfer_coefficient_M_inverse"
            ]
            paired_residuals = {}
            for level in LEVELS:
                candidate_result = (
                    controls[level]["uniform_sds"][float(length)]
                    if family == "uniform_sds"
                    else exterior[level][float(length)]
                )
                candidate = align_transfer_corrected(
                    candidate_result, times, coefficient
                )
                level_reference = _align_flat(
                    controls[level]["schwarzschild"][None], times
                )
                paired_residuals[level] = candidate - level_reference

            for window, start, end in ANALYSIS_WINDOWS:
                mask = _window_mask(times, start, end)
                denominator = _l2(reference[mask], times[mask])
                coarse_medium = _l2(
                    paired_residuals["coarse"][mask]
                    - paired_residuals["medium"][mask],
                    times[mask],
                ) / denominator
                medium_fine = _l2(
                    paired_residuals["medium"][mask]
                    - paired_residuals["fine"][mask],
                    times[mask],
                ) / denominator
                refinement = conservative_refinement_scale(
                    coarse_medium, medium_fine
                )
                corrected_numerical_lookup[(family, length, window)] = refinement[
                    "conservative_numerical_E2"
                ]
                corrected_numerical_rows.append(
                    {
                        "background_family": family,
                        "cosmological_length_over_M": length,
                        "window": window,
                        "window_family": _window_family(window),
                        "window_start_U_over_M": start,
                        "window_end_U_over_M": end,
                        "coarse_medium_paired_E2": coarse_medium,
                        "medium_fine_paired_E2": medium_fine,
                        **refinement,
                        "paired_residual_definition": (
                            "(W_background_corrected-W_Schwarzschild)_q at "
                            "each refinement"
                        ),
                    }
                )

    comparison_rows: list[dict] = []
    corrected_comparison_rows: list[dict] = []
    for length in lengths:
        for window, start, end in ANALYSIS_WINDOWS:
            uniform_metrics = waveform_metrics(
                times,
                fine_signals["uniform_sds"][length],
                reference,
                start,
                end,
            )
            exterior_metrics = waveform_metrics(
                times,
                fine_signals["exterior_sds"][length],
                reference,
                start,
                end,
            )
            uniform_scale = numerical_lookup[("uniform_sds", length, window)]
            exterior_scale = numerical_lookup[("exterior_sds", length, window)]
            margin_test = improvement_with_margins(
                exterior_metrics["E2"],
                exterior_scale,
                uniform_metrics["E2"],
                uniform_scale,
            )
            comparison_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "uniform_sds_E2": uniform_metrics["E2"],
                    "exterior_sds_E2": exterior_metrics["E2"],
                    "uniform_sds_Einf": uniform_metrics["Einf"],
                    "exterior_sds_Einf": exterior_metrics["Einf"],
                    "uniform_conservative_numerical_E2": uniform_scale,
                    "exterior_conservative_numerical_E2": exterior_scale,
                    "central_improvement_factor": (
                        uniform_metrics["E2"] / exterior_metrics["E2"]
                        if exterior_metrics["E2"] > 0.0
                        else np.inf
                    ),
                    "central_relative_reduction_fraction": (
                        1.0 - exterior_metrics["E2"] / uniform_metrics["E2"]
                    ),
                    **margin_test,
                    "substantial_reduction_definition": (
                        "exterior upper numerical bound <= 0.75 times uniform "
                        "lower numerical bound"
                    ),
                    "time_translation_fitted": False,
                    "amplitude_rescaling_fitted": False,
                }
            )
            uniform_corrected_metrics = waveform_metrics(
                times,
                fine_corrected_signals["uniform_sds"][length],
                reference,
                start,
                end,
            )
            exterior_corrected_metrics = waveform_metrics(
                times,
                fine_corrected_signals["exterior_sds"][length],
                reference,
                start,
                end,
            )
            uniform_corrected_scale = corrected_numerical_lookup[
                ("uniform_sds", length, window)
            ]
            exterior_corrected_scale = corrected_numerical_lookup[
                ("exterior_sds", length, window)
            ]
            corrected_margin_test = improvement_with_margins(
                exterior_corrected_metrics["E2"],
                exterior_corrected_scale,
                uniform_corrected_metrics["E2"],
                uniform_corrected_scale,
            )
            corrected_comparison_rows.append(
                {
                    "cosmological_length_over_M": length,
                    "window": window,
                    "window_family": _window_family(window),
                    "window_start_U_over_M": start,
                    "window_end_U_over_M": end,
                    "uniform_sds_corrected_E2": uniform_corrected_metrics["E2"],
                    "exterior_sds_corrected_E2": exterior_corrected_metrics["E2"],
                    "uniform_sds_corrected_Einf": uniform_corrected_metrics[
                        "Einf"
                    ],
                    "exterior_sds_corrected_Einf": exterior_corrected_metrics[
                        "Einf"
                    ],
                    "uniform_conservative_numerical_E2": uniform_corrected_scale,
                    "exterior_conservative_numerical_E2": exterior_corrected_scale,
                    "central_improvement_factor": (
                        uniform_corrected_metrics["E2"]
                        / exterior_corrected_metrics["E2"]
                        if exterior_corrected_metrics["E2"] > 0.0
                        else np.inf
                    ),
                    "central_relative_reduction_fraction": (
                        1.0
                        - exterior_corrected_metrics["E2"]
                        / uniform_corrected_metrics["E2"]
                    ),
                    **corrected_margin_test,
                    "transfer_correction": "fit-free first-Born causal inverse",
                    "time_translation_fitted": False,
                    "amplitude_rescaling_fitted": False,
                }
            )

    threshold_rows: list[dict] = []
    for window, _, _ in ANALYSIS_WINDOWS:
        selected = [row for row in comparison_rows if row["window"] == window]
        for family in ("uniform_sds", "exterior_sds"):
            for threshold in (0.05, 0.02, 0.01):
                qualified = [
                    row
                    for row in selected
                    if row[f"{family}_E2"]
                    + row[
                        "uniform_conservative_numerical_E2"
                        if family == "uniform_sds"
                        else "exterior_conservative_numerical_E2"
                    ]
                    <= threshold
                ]
                threshold_rows.append(
                    {
                        "background_family": family,
                        "window": window,
                        "window_family": _window_family(window),
                        "threshold_fraction": threshold,
                        "smallest_tested_L_over_M": (
                            min(
                                row["cosmological_length_over_M"]
                                for row in qualified
                            )
                            if qualified
                            else ""
                        ),
                        "status": (
                            "attained_with_numerical_margin"
                            if qualified
                            else f"not_attained_through_L{max(lengths)}"
                        ),
                        "criterion": (
                            "E2 + conservative paired numerical E2 <= threshold"
                        ),
                    }
                )

    return {
        "lengths": lengths,
        "times": times,
        "reference": reference,
        "fine_signals": fine_signals,
        "fine_corrected_signals": fine_corrected_signals,
        "transfer_coefficients": transfer_coefficients,
        "numerical": numerical_rows,
        "corrected_numerical": corrected_numerical_rows,
        "comparisons": comparison_rows,
        "corrected_comparisons": corrected_comparison_rows,
        "thresholds": threshold_rows,
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png = Path(output_dir) / f"{stem}.png"
    pdf = Path(output_dir) / f"{stem}.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def create_plots(output_dir: Path, analysis: dict) -> list[Path]:
    """Write waveform/residual and error-reduction figures."""

    output_dir = Path(output_dir)
    times = analysis["times"]
    reference = analysis["reference"]
    lengths = analysis["lengths"]
    signals = analysis["fine_signals"]
    written: list[Path] = []

    figure, axes = plt.subplots(
        len(lengths), 2, figsize=(11.2, 2.65 * len(lengths)), sharex=True
    )
    axes = np.atleast_2d(axes)
    window = (times >= 0.0) & (times <= 80.0)
    for row, length in enumerate(lengths):
        uniform = signals["uniform_sds"][length]
        exterior = signals["exterior_sds"][length]
        waveform_axis, residual_axis = axes[row]
        waveform_axis.plot(
            times[window], reference[window], color="black", linewidth=1.5,
            label="Schwarzschild",
        )
        waveform_axis.plot(
            times[window], uniform[window], color="#d95f02", linewidth=1.0,
            label="uniform SdS",
        )
        waveform_axis.plot(
            times[window], exterior[window], color="#1b9e77", linewidth=1.0,
            linestyle="--", label="exterior-supported SdS",
        )
        residual_axis.plot(
            times[window], uniform[window] - reference[window],
            color="#d95f02", linewidth=1.0, label="uniform minus Schwarzschild",
        )
        residual_axis.plot(
            times[window], exterior[window] - reference[window],
            color="#1b9e77", linewidth=1.0,
            label="exterior-supported minus Schwarzschild",
        )
        waveform_axis.set_ylabel(rf"$W$ ($L/M={length}$)")
        residual_axis.set_ylabel(r"$\Delta W$")
        for axis in (waveform_axis, residual_axis):
            axis.grid(alpha=0.2)
        if row == 0:
            waveform_axis.legend(fontsize=7, ncol=3)
            residual_axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel(r"$U/M$")
    axes[-1, 1].set_xlabel(r"$U/M$")
    axes[0, 0].set_title("Outer-boundary waveforms")
    axes[0, 1].set_title("Unshifted residuals from Schwarzschild")
    figure.tight_layout()
    written.extend(_save_figure(figure, output_dir, "exterior_waveforms"))
    plt.close(figure)

    comparison = analysis["comparisons"]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for color, (window_name, _, _) in zip(colors, CUMULATIVE_WINDOWS):
        rows = [row for row in comparison if row["window"] == window_name]
        x = np.asarray([row["cosmological_length_over_M"] for row in rows])
        uniform = np.asarray([row["uniform_sds_E2"] for row in rows])
        exterior = np.asarray([row["exterior_sds_E2"] for row in rows])
        uniform_scale = np.asarray(
            [row["uniform_conservative_numerical_E2"] for row in rows]
        )
        exterior_scale = np.asarray(
            [row["exterior_conservative_numerical_E2"] for row in rows]
        )
        label = WINDOW_LATEX_LABELS[window_name]
        axes[0].errorbar(
            x, uniform, yerr=uniform_scale, marker="o", linestyle="--",
            color=color, capsize=2, label=f"uniform, {label}",
        )
        axes[0].errorbar(
            x, exterior, yerr=exterior_scale, marker="s", linestyle="-",
            color=color, capsize=2, label=f"exterior, {label}",
        )
        axes[1].plot(
            x,
            [row["central_improvement_factor"] for row in rows],
            marker="o", color=color, label=label,
        )
    axes[0].set(
        xscale="log", yscale="log", xlabel=r"$L/M$", ylabel=r"$E_2$",
        title="Direct disagreement with Schwarzschild",
    )
    axes[1].axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
    axes[1].set(
        xscale="log", xlabel=r"$L/M$",
        ylabel=r"$E_2^{\rm uniform}/E_2^{\rm exterior}$",
        title="Central improvement factor",
    )
    for axis in axes:
        axis.set_xticks(lengths, [str(length) for length in lengths])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=7)
    figure.tight_layout()
    written.extend(_save_figure(figure, output_dir, "exterior_error_reduction"))
    plt.close(figure)
    return written


def create_analysis(
    output_dir: Path,
    control_dir: Path = CONTROL_ROOT,
    lengths: Iterable[int] = EXTERIOR_LENGTHS,
) -> list[Path]:
    """Run the comparison and write tables, figures, and a strict summary."""

    output_dir = Path(output_dir)
    result = analyze(output_dir, control_dir, lengths)
    tables = output_dir / "tables"
    written = [
        _write_csv(tables / "exterior_numerical_errors.csv", result["numerical"]),
        _write_csv(tables / "exterior_vs_uniform.csv", result["comparisons"]),
        _write_csv(
            tables / "transfer_corrected_numerical_errors.csv",
            result["corrected_numerical"],
        ),
        _write_csv(
            tables / "transfer_corrected_vs_uniform.csv",
            result["corrected_comparisons"],
        ),
        _write_csv(tables / "exterior_direct_thresholds.csv", result["thresholds"]),
    ]

    aligned = tables / "exterior_aligned_waveforms.csv"
    aligned.parent.mkdir(parents=True, exist_ok=True)
    columns = [result["times"], result["reference"]]
    headers = ["U_over_M", "schwarzschild"]
    for length in result["lengths"]:
        columns.extend(
            (
                result["fine_signals"]["uniform_sds"][length],
                result["fine_signals"]["exterior_sds"][length],
                result["fine_corrected_signals"]["uniform_sds"][length],
                result["fine_corrected_signals"]["exterior_sds"][length],
            )
        )
        headers.extend(
            (
                f"uniform_sds_L{length}",
                f"exterior_sds_L{length}",
                f"uniform_sds_L{length}_transfer_corrected",
                f"exterior_sds_L{length}_transfer_corrected",
            )
        )
    with aligned.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(np.column_stack(columns).tolist())
    written.append(aligned)
    written.extend(create_plots(output_dir, result))

    headline = [
        row for row in result["comparisons"] if row["window"] == HEADLINE_WINDOW
    ]
    corrected_headline = [
        row
        for row in result["corrected_comparisons"]
        if row["window"] == HEADLINE_WINDOW
    ]
    summary = {
        "purpose": (
            "Falsifiable comparison of exterior-supported and uniform SdS "
            "regulators against the same frozen Schwarzschild waveform"
        ),
        "control_package": Path(control_dir).as_posix(),
        "control_archives_modified": False,
        "new_lengths_over_M": result["lengths"],
        "headline_window": HEADLINE_WINDOW,
        "headline_comparisons": headline,
        "headline_transfer_corrected_comparisons": corrected_headline,
        "all_comparisons": result["comparisons"],
        "all_transfer_corrected_comparisons": result["corrected_comparisons"],
        "transfer_coefficients": result["transfer_coefficients"],
        "numerical_errors": result["numerical"],
        "transfer_corrected_numerical_errors": result["corrected_numerical"],
        "direct_thresholds": result["thresholds"],
        "time_translation_fitted": False,
        "amplitude_rescaling_fitted": False,
        "transfer_parameter_fitted": False,
        "transfer_approximation": "leading first-Born causal inverse",
        "substantial_reduction_factor": SUBSTANTIAL_REDUCTION_FACTOR,
    }
    written.append(_strict_json(output_dir / "analysis_summary.json", summary))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--control-dir", type=Path, default=CONTROL_ROOT)
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(EXTERIOR_LENGTHS),
        choices=EXTERIOR_LENGTHS,
        help="Analyze a completed subset, for example --lengths 160 for the pilot.",
    )
    arguments = parser.parse_args()
    for path in create_analysis(
        arguments.output_dir,
        arguments.control_dir,
        arguments.lengths,
    ):
        print(path)


if __name__ == "__main__":
    main()
