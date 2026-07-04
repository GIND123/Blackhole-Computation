"""Diagnostics and publication-style plots for black-hole evolutions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .solver import SimulationResult

SCHWARZSCHILD_L2_QNM = {
    "omega_real": 0.37367168,
    "omega_imag_abs": 0.08896232,
}


@dataclass(frozen=True)
class RingdownFit:
    start: float
    end: float
    omega: float
    alpha: float
    relative_rms: float


@dataclass(frozen=True)
class TailFit:
    observer: float
    start: float
    end: float
    exponent: float
    r_squared: float
    points: int


def _observer_index(result: SimulationResult, location: float) -> int:
    return int(np.argmin(np.abs(result.observer_rho - location)))


def fit_ringdown(
    result: SimulationResult, start: float = 40.0, end: float = 100.0
) -> RingdownFit:
    """Fit exp(-alpha*t) times a sinusoid to the infinity signal."""

    index = _observer_index(result, 1.0)
    mask = (result.signal_times >= start) & (result.signal_times <= end)
    t = result.signal_times[mask]
    y = result.signals[mask, index]
    if len(t) < 20:
        raise ValueError("Ringdown window contains too few samples.")
    shifted = t - t[0]
    scale = np.max(np.abs(y))
    if scale == 0:
        raise ValueError("Ringdown signal is identically zero.")

    def model(parameters: np.ndarray) -> np.ndarray:
        alpha, omega, cosine, sine, offset = parameters
        envelope = np.exp(-alpha * shifted)
        return envelope * (
            cosine * np.cos(omega * shifted) + sine * np.sin(omega * shifted)
        ) + offset

    initial = np.array(
        [
            SCHWARZSCHILD_L2_QNM["omega_imag_abs"],
            SCHWARZSCHILD_L2_QNM["omega_real"],
            y[0],
            0.0,
            0.0,
        ]
    )
    fit = least_squares(
        lambda parameters: (model(parameters) - y) / scale,
        initial,
        bounds=(
            [0.0, 0.1, -np.inf, -np.inf, -np.inf],
            [0.5, 1.0, np.inf, np.inf, np.inf],
        ),
        max_nfev=20_000,
    )
    residual = model(fit.x) - y
    relative_rms = float(np.sqrt(np.mean(residual**2)) / scale)
    return RingdownFit(
        start=start,
        end=end,
        omega=float(fit.x[1]),
        alpha=float(fit.x[0]),
        relative_rms=relative_rms,
    )


def fit_tail(
    result: SimulationResult,
    observer: float,
    start: float = 400.0,
    end: float = 900.0,
) -> TailFit:
    """Fit |u| proportional to tau**p over a late-time window."""

    index = _observer_index(result, observer)
    mask = (result.signal_times >= start) & (result.signal_times <= end)
    t = result.signal_times[mask]
    amplitude = np.abs(result.signals[mask, index])
    numerical_floor = 100 * np.finfo(float).eps * np.max(
        np.abs(result.signals[:, index])
    )
    valid = np.isfinite(amplitude) & (amplitude > numerical_floor) & (t > 0)
    t = t[valid]
    amplitude = amplitude[valid]
    if len(t) < 20:
        raise ValueError(
            f"Tail window at rho={observer} is below the numerical floor."
        )
    x = np.log(t)
    y = np.log(amplitude)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = np.sum((y - fitted) ** 2)
    total_sum = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return TailFit(
        observer=float(result.observer_rho[index]),
        start=start,
        end=end,
        exponent=float(slope),
        r_squared=float(r_squared),
        points=len(t),
    )


def local_power_index(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed d(log|u|)/d(log tau) diagnostic."""

    valid = (times > 0) & (np.abs(signal) > 0) & np.isfinite(signal)
    t = times[valid]
    log_amplitude = np.log(np.abs(signal[valid]))
    if len(t) >= 51:
        window = min(501, len(t) // 2 * 2 - 1)
        window = max(51, window)
        log_amplitude = savgol_filter(log_amplitude, window, 3)
    return t, np.gradient(log_amplitude, np.log(t))


def write_diagnostics(
    result: SimulationResult,
    output_dir: Path,
    ringdown_window: tuple[float, float] = (40.0, 100.0),
    tail_window: tuple[float, float] = (400.0, 900.0),
) -> dict:
    """Compute scalar diagnostics and write JSON/CSV summaries."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ringdown = fit_ringdown(result, *ringdown_window)

    tails: list[TailFit] = []
    for observer in (1.0, 0.9):
        try:
            tails.append(fit_tail(result, observer, *tail_window))
        except ValueError:
            continue

    diagnostics = {
        "ringdown": {
            **ringdown.__dict__,
            "reference_omega": SCHWARZSCHILD_L2_QNM["omega_real"],
            "reference_alpha": SCHWARZSCHILD_L2_QNM["omega_imag_abs"],
            "omega_relative_error": abs(
                ringdown.omega - SCHWARZSCHILD_L2_QNM["omega_real"]
            )
            / SCHWARZSCHILD_L2_QNM["omega_real"],
            "alpha_relative_error": abs(
                ringdown.alpha - SCHWARZSCHILD_L2_QNM["omega_imag_abs"]
            )
            / SCHWARZSCHILD_L2_QNM["omega_imag_abs"],
        },
        "tails": [tail.__dict__ for tail in tails],
        "constraint": {
            "maximum_linf": float(np.max(result.constraint_linf)),
            "final_linf": float(result.constraint_linf[-1]),
            "maximum_l2": float(np.max(result.constraint_l2)),
        },
    }
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, indent=2)

    with (output_dir / "waveforms.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["tau", *[f"u_rho_{point:g}" for point in result.observer_rho]]
        )
        writer.writerows(
            np.column_stack((result.signal_times, result.signals)).tolist()
        )
    return diagnostics


def create_plots(result: SimulationResult, output_dir: Path) -> None:
    """Create the core spacetime, waveform, tail, and constraint figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(result.rho)
    rho = result.rho[order]
    snapshots = result.u_snapshots[:, order]

    early_end = min(50.0, result.snapshot_times[-1])
    early = result.snapshot_times <= early_end
    early_snapshots = snapshots[early]
    color_limit = float(np.max(np.abs(early_snapshots)))
    fig, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.pcolormesh(
        rho,
        result.snapshot_times[early],
        early_snapshots,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit),
        rasterized=True,
    )
    fig.colorbar(image, ax=axis, label=r"$u(\tau,\rho)$")
    axis.set(
        xlabel=r"$\rho$",
        ylabel=r"$\tau/M$",
        title=rf"Black-hole perturbation: $0\leq\tau\leq{early_end:g}M$",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "spacetime.png", dpi=220)
    plt.close(fig)

    infinity = _observer_index(result, 1.0)
    t = result.signal_times
    y = result.signals[:, infinity]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.semilogy(t, np.maximum(np.abs(y), np.finfo(float).tiny), linewidth=1)
    axis.axvspan(0, 20, color="coral", alpha=0.18, label="transient")
    axis.axvspan(20, 200, color="lightgreen", alpha=0.18, label="ringdown")
    axis.axvspan(200, t[-1], color="skyblue", alpha=0.18, label="tail")
    axis.set(
        xlabel=r"$\tau/M$",
        ylabel=r"$|u(\tau,\rho=1)|$",
        title="Gravitational-wave signal at future null infinity",
        xlim=(0, t[-1]),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_infinity.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for observer in (0.9, 0.95, 1.0):
        index = _observer_index(result, observer)
        label = rf"$\rho={result.observer_rho[index]:g}$"
        axes[0].loglog(
            t[t > 0],
            np.maximum(np.abs(result.signals[t > 0, index]), np.finfo(float).tiny),
            label=label,
        )
        local_t, index_values = local_power_index(t, result.signals[:, index])
        mask = local_t >= max(200, 0.2 * t[-1])
        axes[1].plot(local_t[mask], index_values[mask], label=label)
    axes[0].set(xlabel=r"$\tau/M$", ylabel=r"$|u|$", title="Late-time decay")
    axes[1].axhline(-4, color="black", linestyle="--", alpha=0.5, label=r"$-4$")
    axes[1].axhline(-7, color="gray", linestyle=":", alpha=0.7, label=r"$-7$")
    axes[1].set(
        xlabel=r"$\tau/M$",
        ylabel=r"$d\log|u|/d\log\tau$",
        title="Local power index",
        ylim=(-10, 0),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tail_decay.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.4))
    axis.semilogy(
        result.snapshot_times,
        np.maximum(result.constraint_linf, np.finfo(float).tiny),
        label=r"$\|\psi-\partial_\rho u\|_\infty$",
    )
    axis.semilogy(
        result.snapshot_times,
        np.maximum(result.constraint_l2, np.finfo(float).tiny),
        label="RMS",
    )
    axis.set(xlabel=r"$\tau/M$", ylabel="constraint error", title="Constraint preservation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "constraint.png", dpi=220)
    plt.close(fig)
