"""Price-law and Schwarzschild-de Sitter tail/crossover study.

This workflow implements physically matched initially dynamical data,
geometric retarded-time alignment, Schwarzschild Price-law validation,
finite-L exponential/constant tail diagnostics, and sliding-window trust
times.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .sds_model import (
    ArealVelocityBumpInitialData,
    SdSParameters,
    compact_areal_velocity_profile,
    propagation_coefficient,
    retarded_time_offset as sds_retarded_time_offset,
    sds_horizons,
)
from .sds_solver import (
    SdSNumericalParameters,
    SdSSimulationResult,
    load_sds_result,
    run_schwarzschild_scalar_simulation,
    run_sds_simulation,
)
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    propagation_coefficient as schwarzschild_propagation_coefficient,
    retarded_time_offset as schwarzschild_retarded_time_offset,
)
from .tail_analysis import (
    aligned_signals,
    asymptotic_constant,
    json_safe,
    local_decay_rates,
    numerical_amplitude_floor,
    select_stable_power_fit,
    select_stable_exponential_fit,
    sliding_window_difference,
    trust_times,
)

LOGGER = logging.getLogger(__name__)
PRICE_SCRI = lambda ell: ell + 2
PRICE_HORIZON = lambda ell: 2 * ell + 3


def _observer_signal(result: SdSSimulationResult, location: float) -> np.ndarray:
    index = int(np.argmin(np.abs(result.observer_rho - location)))
    return np.asarray(result.signals[:, index], dtype=float)


def _retarded_signal(
    result: SdSSimulationResult, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(result.signal_times, dtype=float) - float(offset),
        _observer_signal(result, 1.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolved_fit(
    function: Callable[[], object],
) -> tuple[dict | None, str | None]:
    try:
        fit = function()
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return None, str(exc)
    return fit.as_dict(), None


def _run_and_save(
    path: Path,
    runner: Callable[[], SdSSimulationResult],
    *,
    reuse_existing: bool,
) -> SdSSimulationResult:
    if reuse_existing and path.exists():
        LOGGER.info("reusing %s", path)
        return load_sds_result(path)
    result = runner()
    result.save(path)
    return result


def _production_end_time(kappa_c: float, cosmological_timescales: float) -> float:
    return max(120.0, cosmological_timescales / kappa_c)


def _plot_initial_velocity(
    mass: float,
    lengths: tuple[float, ...],
    initial: ArealVelocityBumpInitialData,
    output_dir: Path,
) -> None:
    left = initial.center_radius - 1.15 * initial.support_half_width
    right = initial.center_radius + 1.15 * initial.support_half_width
    radius = np.linspace(left, right, 2400)
    velocity = compact_areal_velocity_profile(radius, initial)

    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.3), sharex=True)
    axes[0].plot(radius / mass, velocity, color="black", linewidth=2, label=r"$G(r)$")
    schwarzschild_model = SchwarzschildScalarParameters(mass=mass, ell=0)
    rho_zero = 1.0 - 2.0 * mass / radius
    reconstructed_zero = (
        schwarzschild_propagation_coefficient(rho_zero, schwarzschild_model)
        * velocity
        / schwarzschild_propagation_coefficient(rho_zero, schwarzschild_model)
    )
    axes[1].plot(
        radius / mass,
        reconstructed_zero,
        color="black",
        linewidth=2,
        label="Schwarzschild",
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(lengths)))
    for color, length in zip(colors, lengths):
        model = SdSParameters(mass=mass, cosmological_length=length, ell=0)
        horizons = sds_horizons(model)
        rho = (1.0 - horizons.black_hole / radius) / (
            1.0 - horizons.black_hole / horizons.cosmological
        )
        coefficient = propagation_coefficient(rho, model, "minimal")
        momentum = velocity / coefficient
        axes[1].plot(
            radius / mass,
            coefficient * momentum,
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=rf"$L/M={length / mass:g}$",
        )
    axes[0].set(ylabel=r"physical velocity $G(r)$")
    axes[1].set(
        xlabel=r"areal radius $r/M$",
        ylabel=r"reconstructed $A_L\pi_L$",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=3)
    axes[0].set_title("Identical initially dynamical data in areal radius")
    fig.tight_layout()
    fig.savefig(output_dir / "initial_velocity_profiles.png", dpi=240)
    plt.close(fig)

    _write_rows(
        output_dir / "initial_velocity_profile.csv",
        [
            {"radius": float(r), "G": float(g)}
            for r, g in zip(radius, velocity)
        ],
    )


def _analyze_schwarzschild(
    ell: int,
    result: SdSSimulationResult,
    offset: float,
) -> dict:
    outer_times, outer_signal = _retarded_signal(result, offset)
    horizon_signal = _observer_signal(result, 0.0)
    horizon_times = np.asarray(result.signal_times, dtype=float)
    maximum_outer = min(
        0.9 * float(outer_times[-1]),
        {0: 1200.0, 1: 1000.0, 2: 700.0}.get(ell, 600.0),
    )
    maximum_horizon = min(
        0.85 * float(horizon_times[-1]),
        {0: 600.0, 1: 350.0, 2: 220.0}.get(ell, 200.0),
    )
    outer_fit, outer_error = _resolved_fit(
        lambda: select_stable_power_fit(
            outer_times,
            outer_signal,
            minimum_time=60.0,
            maximum_time=maximum_outer,
        )
    )
    horizon_fit, horizon_error = _resolved_fit(
        lambda: select_stable_power_fit(
            horizon_times,
            horizon_signal,
            minimum_time=45.0,
            maximum_time=maximum_horizon,
        )
    )
    outer_status = (
        "validated"
        if outer_fit is not None
        and outer_fit["r_squared"] >= 0.99
        and abs(outer_fit["rate"] - PRICE_SCRI(ell)) / PRICE_SCRI(ell) <= 0.10
        else "unresolved"
    )
    horizon_status = (
        "validated"
        if horizon_fit is not None
        and horizon_fit["r_squared"] >= 0.99
        and abs(horizon_fit["rate"] - PRICE_HORIZON(ell))
        / PRICE_HORIZON(ell)
        <= 0.15
        else "unresolved"
    )
    return {
        "ell": ell,
        "retarded_time_offset": offset,
        "expected_scri_power": PRICE_SCRI(ell),
        "scri_fit": outer_fit,
        "scri_fit_error": outer_error,
        "scri_status": outer_status,
        "expected_horizon_power": PRICE_HORIZON(ell),
        "horizon_fit": horizon_fit,
        "horizon_fit_error": horizon_error,
        "horizon_status": horizon_status,
        "maximum_constraint_linf": float(np.max(result.constraint_linf)),
    }


def _analyze_sds_case(
    ell: int,
    length: float,
    result: SdSSimulationResult,
    offset: float,
    reference: SdSSimulationResult,
    reference_offset: float,
    window_width: float,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    model = SdSParameters(
        mass=float(result.metadata["model"]["mass"]),
        cosmological_length=length,
        ell=ell,
    )
    horizons = sds_horizons(model)
    kappa = horizons.kappa_cosmological
    times, signal = _retarded_signal(result, offset)
    reference_times, reference_signal = _retarded_signal(reference, reference_offset)
    common_times, aligned_reference, aligned_candidate = aligned_signals(
        reference_times, reference_signal, times, signal
    )
    difference_times, sliding = sliding_window_difference(
        common_times,
        aligned_reference,
        aligned_candidate,
        window_width=window_width,
    )
    crossings = trust_times(
        difference_times,
        sliding,
        sustained_width=max(2.0, window_width / 4.0),
        start_time=max(
            20.0,
            float(common_times[np.argmax(np.abs(aligned_reference))])
            + window_width,
        ),
    )

    row: dict = {
        "ell": ell,
        "L": length,
        "resolution": int(result.metadata["numerical"]["resolution"]),
        "timestep": float(result.metadata["numerical"]["timestep"]),
        "Lambda": model.cosmological_constant,
        "r_black_hole": horizons.black_hole,
        "r_cosmological": horizons.cosmological,
        "kappa_cosmological": kappa,
        "kappa_inverse": 1.0 / kappa,
        "evolution_timescales": float(result.signal_times[-1] * kappa),
        "retarded_time_offset": offset,
        "maximum_constraint_linf": float(np.max(result.constraint_linf)),
        "trust_time_1_percent": crossings[0.01],
        "trust_time_5_percent": crossings[0.05],
        "trust_time_10_percent": crossings[0.10],
    }

    intermediate_end = min(0.8 / kappa, 0.75 * float(times[-1]))
    intermediate_fit, intermediate_error = _resolved_fit(
        lambda: select_stable_power_fit(
            times,
            signal,
            minimum_time=35.0,
            maximum_time=intermediate_end,
        )
    )
    row["intermediate_power_fit"] = intermediate_fit
    row["intermediate_power_fit_error"] = intermediate_error

    if ell == 0:
        constant = asymptotic_constant(times, signal, start_fraction=0.78)
        row["asymptotic_constant"] = constant
        row["exponential_fit"] = None
        row["gamma_over_kappa"] = None
    else:
        exponential_fit, exponential_error = _resolved_fit(
            lambda: select_stable_exponential_fit(times, signal, kappa)
        )
        row["exponential_fit"] = exponential_fit
        row["exponential_fit_error"] = exponential_error
        row["gamma_over_kappa"] = (
            float(exponential_fit["rate"]) / kappa
            if exponential_fit is not None
            else None
        )
    return row, (difference_times, sliding)


def _plot_schwarzschild_validation(
    ell: int,
    result: SdSSimulationResult,
    offset: float,
    diagnostics: dict,
    output_dir: Path,
) -> None:
    outer_times, outer_signal = _retarded_signal(result, offset)
    horizon_times = np.asarray(result.signal_times)
    horizon_signal = _observer_signal(result, 0.0)
    _, outer_power, _ = local_decay_rates(outer_times, outer_signal)
    _, horizon_power, _ = local_decay_rates(horizon_times, horizon_signal)
    outer_floor = max(np.finfo(float).tiny, 1000 * np.finfo(float).eps * np.max(np.abs(outer_signal)))
    horizon_floor = max(np.finfo(float).tiny, 1000 * np.finfo(float).eps * np.max(np.abs(horizon_signal)))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    valid_outer = outer_times > 0
    axes[0, 0].loglog(
        outer_times[valid_outer],
        np.maximum(np.abs(outer_signal[valid_outer]), outer_floor),
        linewidth=1.1,
    )
    axes[0, 1].loglog(
        horizon_times[horizon_times > 0],
        np.maximum(np.abs(horizon_signal[horizon_times > 0]), horizon_floor),
        linewidth=1.1,
    )
    axes[1, 0].semilogx(outer_times[valid_outer], outer_power[valid_outer])
    axes[1, 1].semilogx(horizon_times[horizon_times > 0], horizon_power[horizon_times > 0])
    axes[1, 0].axhline(PRICE_SCRI(ell), color="black", linestyle="--", label=rf"$p={PRICE_SCRI(ell)}$")
    axes[1, 1].axhline(PRICE_HORIZON(ell), color="black", linestyle="--", label=rf"$p={PRICE_HORIZON(ell)}$")
    for location, axis in (("scri_fit", axes[0, 0]), ("horizon_fit", axes[0, 1])):
        fit = diagnostics.get(location)
        if fit:
            axis.axvspan(fit["start"], fit["end"], color="tab:green", alpha=0.16, label="fit interval")
            axis.legend(fontsize=8)
    axes[0, 0].set(title="Future null infinity", ylabel=r"$|u|$")
    axes[0, 1].set(title="Black-hole horizon", ylabel=r"$|u|$")
    axes[1, 0].set(xlabel=r"retarded time $U/M$", ylabel=r"$p_{\rm eff}$")
    axes[1, 1].set(xlabel=r"horizon time $\tau/M$", ylabel=r"$p_{\rm eff}$")
    for axis in axes.ravel():
        axis.grid(alpha=0.25, which="both")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(rf"Schwarzschild Price-law validation, $\ell={ell}$")
    fig.tight_layout()
    fig.savefig(output_dir / "schwarzschild_price_law.png", dpi=240)
    plt.close(fig)


def _plot_sds_sequence(
    ell: int,
    reference: SdSSimulationResult,
    reference_offset: float,
    finite: dict[float, SdSSimulationResult],
    offsets: dict[float, float],
    rows: list[dict],
    differences: dict[float, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(finite)))
    reference_times, reference_signal = _retarded_signal(reference, reference_offset)
    positive_reference = reference_times > 0.0

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=False)
    early_end = min(180.0, float(reference_times[-1]))
    early = (reference_times >= -5.0) & (reference_times <= early_end)
    axes[0].plot(reference_times[early], reference_signal[early], color="black", linewidth=1.8, label="Schwarzschild")
    reference_amplitude = np.abs(reference_signal)
    resolved_reference = (
        positive_reference
        & (reference_amplitude > numerical_amplitude_floor(reference_signal))
    )
    axes[1].loglog(
        reference_times[resolved_reference],
        reference_amplitude[resolved_reference],
        color="black",
        linewidth=1.8,
        label="Schwarzschild",
    )
    for color, length in zip(colors, finite):
        times, signal = _retarded_signal(finite[length], offsets[length])
        early = (times >= -5.0) & (times <= early_end)
        axes[0].plot(times[early], signal[early], color=color, linewidth=1.0, label=rf"$L/M={length:g}$")
        amplitude = np.abs(signal)
        positive = (
            (times > 0.0)
            & (amplitude > numerical_amplitude_floor(signal))
        )
        axes[1].loglog(
            times[positive],
            amplitude[positive],
            color=color,
            linewidth=1.0,
            label=rf"$L/M={length:g}$",
        )
    axes[0].set(xlabel=r"$U/M$", ylabel=r"$u$", title="Geometrically aligned early waveforms")
    axes[1].set(xlabel=r"$U/M$", ylabel=r"$|u|$", title="Power-law and cosmological regimes")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8, ncol=3)
    fig.suptitle(rf"Schwarzschild--de Sitter crossover, $\ell={ell}$")
    fig.tight_layout()
    fig.savefig(output_dir / "aligned_waveforms.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    power_limit = max(8.0, 2.2 * PRICE_SCRI(ell))
    exponential_limit = max(3.0, 2.2 * max(ell, 1))
    for color, length in zip(colors, finite):
        model = SdSParameters(
            mass=float(finite[length].metadata["model"]["mass"]),
            cosmological_length=length,
            ell=ell,
        )
        kappa = sds_horizons(model).kappa_cosmological
        times, signal = _retarded_signal(finite[length], offsets[length])
        row = next(item for item in rows if item["L"] == length)
        offset = (
            float(row["asymptotic_constant"]["value"])
            if ell == 0 and row.get("asymptotic_constant")
            else 0.0
        )
        _, power, exponential = local_decay_rates(times, signal, offset=offset)
        rate_mask = (
            (times > max(20.0, 0.1 / kappa))
            & np.isfinite(power)
            & (power > 0.0)
            & (power < power_limit)
        )
        displayed_power = np.where(rate_mask, power, np.nan)
        axes[0].semilogx(
            times,
            displayed_power,
            color=color,
            linewidth=1.0,
            label=rf"$L/M={length:g}$",
        )
        scaled = kappa * times
        scaled_exponential = exponential / kappa
        exponential_mask = (
            (scaled > 1.0)
            & np.isfinite(scaled_exponential)
            & (scaled_exponential > 0.0)
            & (scaled_exponential < exponential_limit)
        )
        displayed_exponential = np.where(
            exponential_mask, scaled_exponential, np.nan
        )
        axes[1].plot(
            scaled,
            displayed_exponential,
            color=color,
            linewidth=1.0,
            label=rf"$L/M={length:g}$",
        )
        intermediate_fit = row.get("intermediate_power_fit")
        if intermediate_fit and intermediate_fit["r_squared"] >= 0.98:
            axes[0].hlines(
                intermediate_fit["rate"],
                intermediate_fit["start"],
                intermediate_fit["end"],
                color=color,
                linewidth=3.0,
            )
        exponential_fit = row.get("exponential_fit")
        if exponential_fit:
            axes[1].hlines(
                exponential_fit["rate"] / kappa,
                kappa * exponential_fit["start"],
                kappa * exponential_fit["end"],
                color=color,
                linewidth=3.0,
            )
    axes[0].axhline(PRICE_SCRI(ell), color="black", linestyle="--", linewidth=1.0, label="Schwarzschild scri target")
    if ell > 0:
        axes[1].axhline(ell, color="black", linestyle="--", linewidth=1.0, label=rf"$\gamma/\kappa_c={ell}$")
    axes[0].set(xlabel=r"$U/M$", ylabel=r"$p_{\rm eff}$", title="Local effective power index")
    axes[1].set(xlabel=r"$\kappa_c U$", ylabel=r"$\gamma_{\rm eff}/\kappa_c$", title="Local exponential rate")
    axes[1].set_xlim(left=0.0)
    axes[0].set_ylim(0.0, power_limit)
    axes[1].set_ylim(0.0, exponential_limit)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "local_decay_rates.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, length in zip(colors, finite):
        model = SdSParameters(
            mass=float(finite[length].metadata["model"]["mass"]),
            cosmological_length=length,
            ell=ell,
        )
        kappa = sds_horizons(model).kappa_cosmological
        times, signal = _retarded_signal(finite[length], offsets[length])
        scaled = kappa * times
        if ell == 0:
            axis.plot(
                scaled,
                signal,
                color=color,
                linewidth=1.0,
                label=rf"$L/M={length:g}$",
            )
        else:
            amplitude = np.abs(signal)
            resolved = (
                (scaled >= 0.5)
                & (scaled <= 6.1)
                & (amplitude > numerical_amplitude_floor(signal))
            )
            axis.semilogy(
                scaled[resolved],
                amplitude[resolved],
                color=color,
                linewidth=1.0,
                label=rf"$L/M={length:g}$",
            )
            row = next(item for item in rows if item["L"] == length)
            fit = row.get("exponential_fit")
            if fit:
                axis.axvspan(
                    kappa * fit["start"],
                    kappa * fit["end"],
                    color=color,
                    alpha=0.10,
                )
    axis.set_xlim(0.0, 6.1)
    axis.set(
        xlabel=r"scaled retarded time $\kappa_c U$",
        ylabel=r"$u$" if ell == 0 else r"$|u|$",
        title=(
            rf"Nonzero SdS monopole constants, $\ell={ell}$"
            if ell == 0
            else rf"Late SdS exponential regime, $\ell={ell}$"
        ),
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "late_time_semilog.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, length in zip(colors, finite):
        times, difference = differences[length]
        axis.semilogy(times, difference, color=color, linewidth=1.1, label=rf"$L/M={length:g}$")
    for threshold in (0.01, 0.05, 0.10):
        axis.axhline(threshold, color="black", linewidth=0.8, linestyle=":")
    axis.set(
        xlabel=r"geometric retarded time $U/M$",
        ylabel="sliding relative waveform difference",
        title=rf"Schwarzschild trust intervals, $\ell={ell}$",
        ylim=(5e-4, 2.0),
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sliding_waveform_differences.png", dpi=240)
    plt.close(fig)


def _plot_summary(rows: list[dict], output_dir: Path) -> None:
    ells = sorted({int(row["ell"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for ell in ells:
        selected = sorted(
            (row for row in rows if int(row["ell"]) == ell), key=lambda row: row["L"]
        )
        if ell > 0:
            valid_rates = [
                (row["L"], row["gamma_over_kappa"])
                for row in selected
                if row["gamma_over_kappa"] is not None
                and np.isfinite(row["gamma_over_kappa"])
            ]
            if valid_rates:
                rate_lengths, rates = zip(*valid_rates)
                axes[0].semilogx(
                    rate_lengths,
                    rates,
                    "o-",
                    label=rf"$\ell={ell}$",
                )
                axes[0].axhline(ell, color=f"C{ell}", linestyle=":", alpha=0.6)
        for threshold, marker, label in (
            ("trust_time_1_percent", "o", "1%"),
            ("trust_time_5_percent", "s", "5%"),
            ("trust_time_10_percent", "^", "10%"),
        ):
            valid_trust = [
                (row["L"], row[threshold])
                for row in selected
                if np.isfinite(row[threshold])
            ]
            if valid_trust:
                trust_lengths, values = zip(*valid_trust)
                axes[1].loglog(
                    trust_lengths,
                    values,
                    marker=marker,
                    label=rf"$\ell={ell}$, {label}",
                )
    axes[0].set(
        xlabel=r"$L/M$",
        ylabel=r"fitted $\gamma/\kappa_c$",
        title="Late SdS exponential rates",
    )
    axes[1].set(
        xlabel=r"$L/M$",
        ylabel=r"trust time $U/M$",
        title="Useful asymptotically flat interval",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "tail_summary.png", dpi=240)
    plt.close(fig)


def run_tail_study(
    *,
    mass: float,
    ells: tuple[int, ...],
    lengths: tuple[float, ...],
    initial: ArealVelocityBumpInitialData,
    numerical: SdSNumericalParameters,
    output_dir: Path,
    reference_radius: float = 4.0,
    cosmological_timescales: float = 5.0,
    window_width: float = 20.0,
    reuse_existing: bool = False,
    ell2_numerical: SdSNumericalParameters | None = None,
) -> dict:
    """Run the complete production tail sequence and create its diagnostics."""

    if not ells or any(ell < 0 for ell in ells):
        raise ValueError("At least one nonnegative ell is required.")
    if tuple(sorted(set(ells))) != ells:
        raise ValueError("ells must be unique and increasing.")
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise ValueError("Cosmological lengths must be unique and increasing.")
    if numerical.bridge != "minimal":
        raise ValueError("The tail study uses the minimal gauge.")
    if ell2_numerical is not None and ell2_numerical.bridge != "minimal":
        raise ValueError("The ell=2 tail refinement uses the minimal gauge.")
    if cosmological_timescales < 3.0:
        raise ValueError("Tail runs require at least three cosmological timescales.")

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _plot_initial_velocity(mass, lengths, initial, output_dir)

    longest_model = SdSParameters(
        mass=mass, cosmological_length=lengths[-1], ell=ells[0]
    )
    longest_kappa = sds_horizons(longest_model).kappa_cosmological
    reference_end = _production_end_time(longest_kappa, cosmological_timescales)

    schwarzschild_rows: list[dict] = []
    sds_rows: list[dict] = []
    diagnostics: dict = {
        "purpose": "Schwarzschild Price laws and finite-L SdS tail crossover",
        "paper": "Brady et al., Phys. Rev. D 60, 064003 (1999), arXiv:gr-qc/9902010v2",
        "initial_data": initial.as_dict(),
        "geometric_time": "U=tau-q_L with analytic q_L",
        "window_width": window_width,
        "cosmological_timescales": cosmological_timescales,
        "numerical_base": asdict(numerical),
        "ell2_numerical": (
            asdict(ell2_numerical) if ell2_numerical is not None else None
        ),
        "ells": {},
    }

    for ell in ells:
        ell_numerical = (
            ell2_numerical
            if ell == 2 and ell2_numerical is not None
            else numerical
        )
        ell_dir = output_dir / f"ell{ell}"
        ell_raw = raw_dir / f"ell{ell}"
        ell_dir.mkdir(parents=True, exist_ok=True)
        ell_raw.mkdir(parents=True, exist_ok=True)

        schwarzschild_model = SchwarzschildScalarParameters(mass=mass, ell=ell)
        reference_numerical = replace(ell_numerical, end_time=reference_end)
        reference = _run_and_save(
            ell_raw / "schwarzschild.npz",
            lambda model=schwarzschild_model, settings=reference_numerical: run_schwarzschild_scalar_simulation(
                model, initial, settings
            ),
            reuse_existing=reuse_existing,
        )
        reference_offset = schwarzschild_retarded_time_offset(
            schwarzschild_model, reference_radius
        )
        reference.metadata["retarded_time_offset"] = {
            "q": reference_offset,
            "definition": "lim_(r->infinity)(h+r_*)",
            "evaluation": "analytic",
        }
        reference.save(ell_raw / "schwarzschild.npz")
        schwarzschild_diagnostics = _analyze_schwarzschild(
            ell, reference, reference_offset
        )
        schwarzschild_rows.append(
            {
                "ell": ell,
                "resolution": int(reference.metadata["numerical"]["resolution"]),
                "timestep": float(reference.metadata["numerical"]["timestep"]),
                "expected_scri_power": PRICE_SCRI(ell),
                "measured_scri_power": (
                    schwarzschild_diagnostics["scri_fit"]["rate"]
                    if schwarzschild_diagnostics["scri_fit"]
                    else ""
                ),
                "scri_r_squared": (
                    schwarzschild_diagnostics["scri_fit"]["r_squared"]
                    if schwarzschild_diagnostics["scri_fit"]
                    else ""
                ),
                "scri_status": schwarzschild_diagnostics["scri_status"],
                "expected_horizon_power": PRICE_HORIZON(ell),
                "measured_horizon_power": (
                    schwarzschild_diagnostics["horizon_fit"]["rate"]
                    if schwarzschild_diagnostics["horizon_fit"]
                    else ""
                ),
                "horizon_r_squared": (
                    schwarzschild_diagnostics["horizon_fit"]["r_squared"]
                    if schwarzschild_diagnostics["horizon_fit"]
                    else ""
                ),
                "horizon_status": schwarzschild_diagnostics["horizon_status"],
                "maximum_constraint_linf": schwarzschild_diagnostics[
                    "maximum_constraint_linf"
                ],
            }
        )
        _plot_schwarzschild_validation(
            ell,
            reference,
            reference_offset,
            schwarzschild_diagnostics,
            ell_dir,
        )

        finite: dict[float, SdSSimulationResult] = {}
        offsets: dict[float, float] = {}
        ell_rows: list[dict] = []
        differences: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for length in lengths:
            model = SdSParameters(
                mass=mass, cosmological_length=length, ell=ell
            )
            horizons = sds_horizons(model)
            end_time = _production_end_time(
                horizons.kappa_cosmological, cosmological_timescales
            )
            settings = replace(ell_numerical, end_time=end_time)
            result = _run_and_save(
                ell_raw / f"sds_L{length:g}.npz",
                lambda model=model, settings=settings: run_sds_simulation(
                    model, initial, settings
                ),
                reuse_existing=reuse_existing,
            )
            offset = sds_retarded_time_offset(model, reference_radius)
            result.metadata["retarded_time_offset"] = {
                "q": offset,
                "definition": "lim_(r->r_c)(h+r_*)",
                "evaluation": "analytic",
            }
            result.save(ell_raw / f"sds_L{length:g}.npz")
            row, difference = _analyze_sds_case(
                ell,
                length,
                result,
                offset,
                reference,
                reference_offset,
                window_width,
            )
            finite[length] = result
            offsets[length] = offset
            ell_rows.append(row)
            sds_rows.append(row)
            differences[length] = difference

        _plot_sds_sequence(
            ell,
            reference,
            reference_offset,
            finite,
            offsets,
            ell_rows,
            differences,
            ell_dir,
        )
        diagnostics["ells"][str(ell)] = {
            "schwarzschild": schwarzschild_diagnostics,
            "sds": ell_rows,
        }

    _write_rows(output_dir / "schwarzschild_price_law.csv", schwarzschild_rows)
    flattened_sds_rows = []
    for row in sds_rows:
        flattened = {key: value for key, value in row.items() if not isinstance(value, dict)}
        if row.get("exponential_fit"):
            flattened.update(
                {
                    f"exponential_{key}": value
                    for key, value in row["exponential_fit"].items()
                }
            )
        if row.get("asymptotic_constant"):
            flattened.update(
                {
                    f"constant_{key}": value
                    for key, value in row["asymptotic_constant"].items()
                }
            )
        if row.get("intermediate_power_fit"):
            flattened.update(
                {
                    f"intermediate_{key}": value
                    for key, value in row["intermediate_power_fit"].items()
                }
            )
        flattened_sds_rows.append(flattened)
    _write_rows(output_dir / "sds_tail_summary.csv", flattened_sds_rows)
    _write_rows(
        output_dir / "trust_times.csv",
        [
            {
                key: row[key]
                for key in (
                    "ell",
                    "L",
                    "trust_time_1_percent",
                    "trust_time_5_percent",
                    "trust_time_10_percent",
                )
            }
            for row in sds_rows
        ],
    )
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(json_safe(diagnostics), stream, indent=2, allow_nan=False)
    _plot_summary(sds_rows, output_dir)
    return diagnostics
