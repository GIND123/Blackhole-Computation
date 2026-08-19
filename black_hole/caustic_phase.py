"""Measurement of the phase rotation an echo acquires at a caustic.

A wavefront that focuses on the axis behind a black hole does not return a
scaled copy of the direct pulse.  Geometrical optics fails at the focus, the
Jacobian of the ray congruence changes sign, and the amplitude picks up a
phase.  For Schwarzschild the resulting fourfold structure of the retarded
Green function was identified by Ori and analyzed by Casals and collaborators,
and each caustic passage acts as a Hilbert transform of the arriving profile
(Zenginoglu and Galley, Phys. Rev. D 86, 064030 (2012)).

The archives here hold one radial response per retained ``ell`` at three
observers, so the field along any direction follows from the addition theorem
without reconstructing a sphere.  That makes the phase directly measurable:
write the analytic signal of the direct pulse as ``z_0 = u_0 + i H[u_0]`` and
fit

    u_gamma(U) = Re[ A exp(i phi) z_0(U - Delta) ]
               = A cos(phi) u_0(U - Delta) - A sin(phi) H[u_0](U - Delta),

for the amplitude ``A``, the rotation ``phi``, and the delay ``Delta``.  An
unrotated copy of the direct pulse gives ``phi = 0``, a sign reversal gives
``phi = 180``, and a full Hilbert transform of either sign gives ``phi``
equal to minus or plus ninety degrees.

The rotation is reported together with the fraction of the variance the two
term model explains, so a poor fit cannot be mistaken for a measured phase.
Nothing here fits a relative time translation between backgrounds: ``Delta``
is a delay within one evolution, between two directions of the same archive.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import special
from scipy.signal import hilbert

from .localized_source import angular_spectral_weights
from .source_evolution import SourcedSimulationResult, load_sourced_result


#: Directions used for the angular scan, in degrees from the source.
SCAN_ANGLES_DEGREES = (
    0.0, 30.0, 60.0, 90.0, 120.0, 140.0, 150.0, 160.0, 170.0, 175.0, 180.0,
)
#: Retarded time before which the outer signal is still at the numerical floor.
ARRIVAL_SEARCH_START_M = 20.0
#: Half widths of the fit window around a measured arrival.
FIT_WINDOW_BEFORE_M = 8.0
FIT_WINDOW_AFTER_M = 14.0
#: Stride applied to the archived cadence before fitting.  The pulses are a
#: few M wide and the archives store every 0.0005M, so this is decimation of
#: an oversampled trace and not a change of the measured quantity.
FIT_STRIDE = 20


def angular_coefficients(result: SourcedSimulationResult) -> np.ndarray:
    """Return the source weights ``g_ell (2 ell + 1) / (4 pi)``."""

    orders = np.asarray(result.response_ell, dtype=int)
    weights = angular_spectral_weights(
        float(result.metadata["source"]["angular_concentration"]), int(orders[-1])
    )
    return weights[orders] * (2.0 * orders + 1.0) / (4.0 * np.pi)


def direction_trace(
    result: SourcedSimulationResult,
    gamma: float,
    observer: int | None = None,
    *,
    ell_max: int | None = None,
) -> np.ndarray:
    """Return ``u(U)`` along one direction by the spherical addition theorem.

    ``gamma`` is the angle from the emitter direction in radians.  The sum is
    exact for the stored modes: no sphere is sampled and no interpolation in
    angle is involved.
    """

    observer = result.outer_index() if observer is None else observer
    orders = np.asarray(result.response_ell, dtype=int)
    keep = (
        np.ones(orders.size, dtype=bool)
        if ell_max is None
        else orders <= int(ell_max)
    )
    legendre = np.asarray(
        [
            special.eval_legendre(int(ell), float(np.cos(gamma)))
            for ell in orders[keep]
        ]
    )
    coefficients = angular_coefficients(result)[keep] * legendre
    responses = np.asarray(result.response_signals[:, observer, :], dtype=float)
    return responses[:, keep] @ coefficients


@dataclass(frozen=True)
class PhaseFit:
    """One quadrature fit of a direction against the direct pulse."""

    phase_degrees: float
    amplitude: float
    delay_over_M: float
    variance_explained: float
    window_start_U_over_M: float
    window_end_U_over_M: float

    def as_dict(self) -> dict:
        return asdict(self)


def phase_fit(
    times: np.ndarray,
    direct: np.ndarray,
    signal: np.ndarray,
    window: tuple[float, float],
    *,
    delays: tuple[float, float, float] = (0.0, 26.0, 0.02),
    stride: int = FIT_STRIDE,
) -> PhaseFit:
    """Fit ``signal`` to a rotated, delayed copy of ``direct``.

    The Hilbert transform is taken on the full archived trace before the fit
    window is applied, so the quadrature partner is not distorted by window
    edges.  The window itself is applied to the data and to the model alike.
    """

    times = np.asarray(times, dtype=float)
    direct = np.asarray(direct, dtype=float)
    quadrature = np.imag(hilbert(direct))
    inside = (times >= window[0]) & (times <= window[1])
    if not np.any(inside):
        raise ValueError("The fit window contains no archived samples.")
    local_times = times[inside][::stride]
    target = np.asarray(signal, dtype=float)[inside][::stride]
    energy = float(target @ target)
    if energy <= 0.0:
        raise ValueError("The requested direction carries no signal.")

    best: tuple[float, float, np.ndarray] | None = None
    for delay in np.arange(*delays):
        design = np.column_stack(
            [
                np.interp(local_times, times + delay, direct, left=0.0, right=0.0),
                np.interp(
                    local_times, times + delay, quadrature, left=0.0, right=0.0
                ),
            ]
        )
        # Two column normal equations.  The design is well conditioned as
        # long as the pulse and its quadrature partner overlap the window,
        # and the closed form keeps the delay scan cheap enough to run over
        # every direction, observer and background.
        gram = design.T @ design
        moment = design.T @ target
        determinant = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
        if abs(determinant) <= 0.0:
            continue
        coefficients = np.array(
            [
                (gram[1, 1] * moment[0] - gram[0, 1] * moment[1]) / determinant,
                (gram[0, 0] * moment[1] - gram[1, 0] * moment[0]) / determinant,
            ]
        )
        residual = target - design @ coefficients
        explained = 1.0 - float(residual @ residual) / energy
        if best is None or explained > best[0]:
            best = (explained, float(delay), coefficients)

    explained, delay, (cosine_part, sine_part) = best
    return PhaseFit(
        phase_degrees=float(np.degrees(np.arctan2(-sine_part, cosine_part))),
        amplitude=float(np.hypot(cosine_part, sine_part)),
        delay_over_M=delay,
        variance_explained=explained,
        window_start_U_over_M=float(window[0]),
        window_end_U_over_M=float(window[1]),
    )


def arrival_time(times: np.ndarray, signal: np.ndarray, start: float) -> float:
    """Return the analytic envelope maximum of a trace after ``start``."""

    times = np.asarray(times, dtype=float)
    envelope = np.abs(hilbert(np.asarray(signal, dtype=float)))
    late = times >= start
    return float(times[late][envelope[late].argmax()])


def _fit_window(
    times: np.ndarray, peak: float, before: float, after: float
) -> tuple[float, float]:
    return (
        max(float(times[0]), peak - before),
        min(float(times[-1]), peak + after),
    )


def scan_archive(
    archive: Path,
    *,
    angles_degrees: tuple[float, ...] = SCAN_ANGLES_DEGREES,
    observers: tuple[int, ...] | None = None,
    ell_max: int | None = None,
) -> list[dict]:
    """Fit every requested direction of one archive at every observer."""

    result = load_sourced_result(Path(archive))
    times = np.asarray(result.retarded_time, dtype=float)
    if observers is None:
        observers = tuple(range(result.response_signals.shape[1]))

    rows: list[dict] = []
    for observer in observers:
        direct = direction_trace(result, 0.0, observer, ell_max=ell_max)
        radius = float(result.observer_areal_radius[observer])
        for degrees in angles_degrees:
            signal = direction_trace(
                result, float(np.radians(degrees)), observer, ell_max=ell_max
            )
            peak = arrival_time(times, signal, ARRIVAL_SEARCH_START_M)
            window = _fit_window(
                times, peak, FIT_WINDOW_BEFORE_M, FIT_WINDOW_AFTER_M
            )
            fit = phase_fit(times, direct, signal, window)
            rows.append(
                {
                    "archive": Path(archive).as_posix(),
                    "observer_areal_radius_over_M": radius,
                    "gamma_degrees": float(degrees),
                    "arrival_U_over_M": peak,
                    "angular_ell_max": int(
                        result.response_ell[-1] if ell_max is None else ell_max
                    ),
                    **fit.as_dict(),
                }
            )
    return rows


def truncation_sensitivity(
    archive: Path, orders: tuple[int, ...] = (30, 36, 40, 44, 46, 48, 50)
) -> list[dict]:
    """Repeat the antipodal fit at several angular truncations.

    The emitter is band limited by its own width, so the caustic is smoothed
    by the source rather than by the truncation.  These rows record that the
    measurement does not depend on where the retained sum is cut.
    """

    result = load_sourced_result(Path(archive))
    times = np.asarray(result.retarded_time, dtype=float)
    observer = result.outer_index()
    rows: list[dict] = []
    for order in orders:
        direct = direction_trace(result, 0.0, observer, ell_max=order)
        antipode = direction_trace(result, np.pi, observer, ell_max=order)
        peak = arrival_time(times, antipode, ARRIVAL_SEARCH_START_M)
        fit = phase_fit(
            times,
            direct,
            antipode,
            _fit_window(times, peak, FIT_WINDOW_BEFORE_M, FIT_WINDOW_AFTER_M),
        )
        rows.append(
            {
                "angular_ell_max": int(order),
                "antipodal_peak_amplitude": float(np.abs(antipode).max()),
                "arrival_U_over_M": peak,
                **fit.as_dict(),
            }
        )
    return rows


def window_sensitivity(archive: Path) -> list[dict]:
    """Repeat the antipodal fit over displaced and rescaled fit windows."""

    result = load_sourced_result(Path(archive))
    times = np.asarray(result.retarded_time, dtype=float)
    observer = result.outer_index()
    direct = direction_trace(result, 0.0, observer)
    antipode = direction_trace(result, np.pi, observer)
    peak = arrival_time(times, antipode, ARRIVAL_SEARCH_START_M)
    variants = {
        "primary": (FIT_WINDOW_BEFORE_M, FIT_WINDOW_AFTER_M),
        "inset_1M": (FIT_WINDOW_BEFORE_M - 1.0, FIT_WINDOW_AFTER_M - 1.0),
        "expanded_1M": (FIT_WINDOW_BEFORE_M + 1.0, FIT_WINDOW_AFTER_M + 1.0),
        "shift_early_1M": (FIT_WINDOW_BEFORE_M + 1.0, FIT_WINDOW_AFTER_M - 1.0),
        "shift_late_1M": (FIT_WINDOW_BEFORE_M - 1.0, FIT_WINDOW_AFTER_M + 1.0),
    }
    rows: list[dict] = []
    for name, (before, after) in variants.items():
        fit = phase_fit(
            times, direct, antipode, _fit_window(times, peak, before, after)
        )
        rows.append({"setting": name, **fit.as_dict()})
    return rows


def null_ray_consistency(
    archive: Path, cosmological_length: float | None = 80.0
) -> list[dict]:
    """Compare the measured arrivals with inward turning null rays.

    The rotation says what shape the echo has; this says whether it arrives
    where geometry puts it.  The ray is traced on the same background from the
    emitter radius at the source time centre out to the outer boundary, so the
    comparison uses no fitted quantity except the delay, which the phase fit
    returns as a free parameter.
    """

    from .null_geodesics import trace_null_ray

    result = load_sourced_result(Path(archive))
    times = np.asarray(result.retarded_time, dtype=float)
    source = result.metadata["source"]
    direct = direction_trace(result, 0.0)
    antipode = direction_trace(result, np.pi)
    measured = {
        0.0: arrival_time(times, direct, 0.0),
        np.pi: arrival_time(times, antipode, ARRIVAL_SEARCH_START_M),
    }

    rows: list[dict] = []
    for gamma, label in ((0.0, "direct"), (np.pi, "antipodal")):
        ray = trace_null_ray(
            source_radius=float(source["center_radius"]),
            observer_radius=None,
            target_angle=float(gamma),
            emission_time=float(source["time_center"]),
            cosmological_length=cosmological_length,
        )
        rows.append(
            {
                "arrival": label,
                "gamma_degrees": float(np.degrees(gamma)),
                "null_ray_U_over_M": float(ray.arrival_u),
                "measured_U_over_M": measured[gamma],
                "difference_over_M": float(measured[gamma] - ray.arrival_u),
                "impact_parameter_over_M": float(ray.impact_parameter),
                "turning_radius_over_M": float(ray.turning_radius),
            }
        )
    delay_ray = rows[1]["null_ray_U_over_M"] - rows[0]["null_ray_U_over_M"]
    delay_measured = rows[1]["measured_U_over_M"] - rows[0]["measured_U_over_M"]
    rows.append(
        {
            "arrival": "delay",
            "gamma_degrees": float("nan"),
            "null_ray_U_over_M": delay_ray,
            "measured_U_over_M": delay_measured,
            "difference_over_M": delay_measured - delay_ray,
            "impact_parameter_over_M": float("nan"),
            "turning_radius_over_M": float("nan"),
        }
    )
    return rows


DEFAULT_ARCHIVE_ROOT = Path("results/regulator_production_v3/raw/source/fine")


def create_report(
    output_dir: Path, archives: dict[str, Path] | None = None
) -> list[Path]:
    """Write the angular scan, its sensitivities, and a summary record."""

    archives = archives or {
        "sds_L80": DEFAULT_ARCHIVE_ROOT / "sds_L80.npz",
        "schwarzschild": DEFAULT_ARCHIVE_ROOT / "schwarzschild.npz",
    }
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    scan: list[dict] = []
    for label, archive in archives.items():
        for row in scan_archive(archive):
            scan.append({"case": label, **row})

    truncation = truncation_sensitivity(archives["sds_L80"])
    windows = window_sensitivity(archives["sds_L80"])

    antipodal = [row for row in scan if row["gamma_degrees"] == 180.0]
    away = [
        row
        for row in scan
        if 0.0 < row["gamma_degrees"] <= 150.0
    ]
    phases = np.asarray([row["phase_degrees"] for row in antipodal])
    summary = {
        "definition": (
            "u_gamma(U) = Re[A exp(i phi) z_0(U - Delta)] with z_0 the "
            "analytic signal of the direct pulse of the same evolution"
        ),
        "reference_direction": "gamma = 0, the emitter direction",
        "antipodal_phase_degrees_mean": float(phases.mean()),
        "antipodal_phase_degrees_spread": float(phases.max() - phases.min()),
        "antipodal_phase_degrees_by_case": {
            "{0}_r{1:.0f}".format(
                row["case"], row["observer_areal_radius_over_M"]
            ): row["phase_degrees"]
            for row in antipodal
        },
        "antipodal_amplitude_by_case": {
            "{0}_r{1:.0f}".format(
                row["case"], row["observer_areal_radius_over_M"]
            ): row["amplitude"]
            for row in antipodal
        },
        "antipodal_variance_explained_min": float(
            min(row["variance_explained"] for row in antipodal)
        ),
        "largest_phase_away_from_the_axis_degrees": float(
            max(abs(row["phase_degrees"]) for row in away)
        ),
        "full_hilbert_transform_degrees": 90.0,
        "unrotated_copy_degrees": 0.0,
        "sign_reversal_degrees": 180.0,
        "truncation_phase_spread_degrees": float(
            max(row["phase_degrees"] for row in truncation)
            - min(row["phase_degrees"] for row in truncation)
        ),
        "window_phase_spread_degrees": float(
            max(row["phase_degrees"] for row in windows)
            - min(row["phase_degrees"] for row in windows)
        ),
        "time_translation_between_backgrounds_fitted": False,
        "interval_limitation": (
            "the archives end about 13M after the antipodal arrival, so the "
            "slowly decaying quadrature tail is truncated and the fitted "
            "rotation is a conservative estimate of its magnitude"
        ),
    }

    rays = null_ray_consistency(archives["sds_L80"])
    delay_row = [row for row in rays if row["arrival"] == "delay"][0]
    summary["null_ray_delay_difference_over_M"] = delay_row["difference_over_M"]
    summary["fitted_delay_against_null_ray_over_M"] = float(
        [row["delay_over_M"] for row in antipodal
         if row["case"] == "sds_L80"
         and row["observer_areal_radius_over_M"] > 50.0][0]
        - delay_row["null_ray_U_over_M"]
    )

    written = [
        _write_csv(tables / "caustic_phase_scan.csv", scan),
        _write_csv(tables / "caustic_phase_truncation.csv", truncation),
        _write_csv(tables / "caustic_phase_windows.csv", windows),
        _write_csv(tables / "caustic_phase_null_rays.csv", rays),
    ]
    destination = output_dir / "caustic_phase.json"
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    written.append(destination)
    return written


def _write_csv(path: Path, rows: list[dict]) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/caustic_visualizations")
    )
    arguments = parser.parse_args()
    for path in create_report(arguments.output_dir):
        print(path)


if __name__ == "__main__":
    main()
