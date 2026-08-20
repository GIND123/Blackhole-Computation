"""Large cosmological length dipole tail study.

This module implements the staged calculation requested after the first
crossover study.  Screening and final evolutions are separate commands so a
long final run is not started until the smallest acceptable cosmological
length has been measured.  Analysis never translates one waveform in time.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .sds_model import (
    ArealVelocityBumpInitialData,
    SdSParameters,
    compact_radius,
    retarded_time_offset as sds_retarded_time_offset,
    sds_horizons,
)
from .sds_result import SdSSimulationResult, load_sds_result
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    retarded_time_offset as schwarzschild_retarded_time_offset,
)
from .tail_analysis import json_safe, numerical_amplitude_floor


OUTPUT_ROOT = Path("results/large_l_tail")
REFERENCE_RADIUS = 4.0
FINITE_OBSERVERS = (8.0, 16.0)
SCREEN_LENGTHS = (640.0, 1280.0, 2560.0, 5120.0)
SCREEN_RESOLUTIONS = (1536, 2048)
FINAL_RESOLUTIONS = (1536, 2048, 3072)
TIMESTEP = 0.0025
HALVED_TIMESTEP = 0.00125
SCREEN_END_U = 500.0
SCREEN_PRICE_ANCHOR_U = 300.0
# Both backgrounds carry the bounded potential term on the explicit side of the
# IMEX split, so the SdS waveform and its Schwarzschild reference are integrated
# by an identical scheme.  See sds_solver._run_scalar_simulation.
EXPLICIT_POTENTIAL = True
PRICE_TARGET_INFINITY = 3.0
PRICE_TARGET_FIXED_RADIUS = 5.0
PRICE_TOLERANCE = 0.05
PRICE_DURATION = 150.0
COSMOLOGICAL_TOLERANCE = 0.1
COSMOLOGICAL_MINIMUM_SCALED_DURATION = 0.4
# A rate is read only where the envelope clears the ladder spread by this
# factor, so a decay law is never fitted to inter-resolution disagreement.
FLOOR_SAFETY_FACTOR = 10.0
INITIAL_DATA = ArealVelocityBumpInitialData(
    center_radius=6.0, support_half_width=3.0, amplitude=1.0
)


@dataclass(frozen=True)
class TailCase:
    """One immutable evolution specification."""

    background: str
    resolution: int
    timestep: float
    end_u: float
    length: float | None = None
    role: str = "screen"

    @property
    def name(self) -> str:
        geometry = (
            (
                "schwarzschild"
                if self.length is None
                else f"schwarzschild_for_L{self.length:g}"
            )
            if self.background == "schwarzschild"
            else f"sds_L{self.length:g}"
        )
        step = str(self.timestep).replace(".", "p")
        return f"{self.role}_{geometry}_N{self.resolution}_dt{step}"


@dataclass(frozen=True)
class LocalFitSettings:
    """Estimator widths and numerical floor used by local fits."""

    envelope_width: float = 10.0
    price_window: float = 40.0
    exponential_scaled_window: float = 0.25
    floor_multiplier: float = 100.0


def price_target(observer: int) -> float:
    """Return the dipole Price exponent for one extraction observer."""

    if observer not in range(3):
        raise ValueError("Observer index must be 0, 1, or 2.")
    return PRICE_TARGET_INFINITY if observer == 2 else PRICE_TARGET_FIXED_RADIUS


def cosmological_rate(length: float) -> float:
    """Return the exact cosmological horizon surface gravity."""

    model = SdSParameters(mass=1.0, cosmological_length=length, ell=1)
    return float(sds_horizons(model).kappa_cosmological)


def final_end_u(length: float) -> float:
    """Return the requested final retarded time, at least four decay times."""

    return 4.0 / cosmological_rate(length)


def screening_cases(length: float) -> tuple[TailCase, ...]:
    """Return the two SdS cases and matched Schwarzschild references."""

    if length not in SCREEN_LENGTHS:
        raise ValueError(f"Screening length must be one of {SCREEN_LENGTHS}.")
    cases: list[TailCase] = []
    for resolution in SCREEN_RESOLUTIONS:
        cases.append(
            TailCase("schwarzschild", resolution, TIMESTEP, SCREEN_END_U)
        )
        cases.append(
            TailCase("sds", resolution, TIMESTEP, SCREEN_END_U, length)
        )
    return tuple(cases)


def final_cases(length: float) -> tuple[TailCase, ...]:
    """Return the three-level final study and the halved-step control."""

    end_u = final_end_u(length)
    cases: list[TailCase] = []
    for resolution in FINAL_RESOLUTIONS:
        cases.extend(
            (
                TailCase(
                    "schwarzschild",
                    resolution,
                    TIMESTEP,
                    end_u,
                    length=length,
                    role="final",
                ),
                TailCase("sds", resolution, TIMESTEP, end_u, length, "final"),
            )
        )
    cases.extend(
        (
            TailCase(
                "schwarzschild",
                2048,
                HALVED_TIMESTEP,
                end_u,
                length=length,
                role="final",
            ),
            TailCase("sds", 2048, HALVED_TIMESTEP, end_u, length, "final"),
        )
    )
    return tuple(cases)


def _offset(case: TailCase) -> float:
    if case.background == "schwarzschild":
        return float(
            schwarzschild_retarded_time_offset(
                SchwarzschildScalarParameters(mass=1.0, ell=1), REFERENCE_RADIUS
            )
        )
    if case.length is None:
        raise ValueError("An SdS case requires a cosmological length.")
    return float(
        sds_retarded_time_offset(
            SdSParameters(mass=1.0, cosmological_length=case.length, ell=1),
            REFERENCE_RADIUS,
        )
    )


def _observer_coordinates(case: TailCase) -> tuple[float, ...]:
    if case.background == "schwarzschild":
        return tuple(1.0 - 2.0 / radius for radius in FINITE_OBSERVERS) + (1.0,)
    if case.length is None:
        raise ValueError("An SdS case requires a cosmological length.")
    model = SdSParameters(mass=1.0, cosmological_length=case.length, ell=1)
    return tuple(
        float(value) for value in compact_radius(np.asarray(FINITE_OBSERVERS), model)
    ) + (1.0,)


def archive_path(output_dir: Path, case: TailCase) -> Path:
    return Path(output_dir) / "raw" / f"{case.name}.npz"


def run_case(
    case: TailCase,
    output_dir: Path,
    *,
    resume_interrupted: bool = False,
) -> Path:
    """Run one case and atomically publish its archive."""

    destination = archive_path(output_dir, case)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_suffix(".running")
    checkpoint = destination.with_suffix(".checkpoint.npz")
    if reservation.exists():
        if not resume_interrupted:
            raise FileExistsError(
                f"Case reservation already exists: {reservation}. Use "
                "--resume-interrupted only after confirming the earlier process "
                "is no longer running."
            )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot resume {case.name} because its checkpoint is missing."
            )
    else:
        with reservation.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(case.name + "\n")

    # Dedalus is imported only by the simulation command.  Reports therefore
    # remain usable in a lightweight NumPy and SciPy environment.
    from .sds_solver import (
        SdSNumericalParameters,
        run_schwarzschild_scalar_simulation,
        run_sds_simulation,
    )

    offset = _offset(case)
    numerical = SdSNumericalParameters(
        resolution=case.resolution,
        timestep=case.timestep,
        end_time=case.end_u + offset,
        signal_dt=0.05,
        snapshot_dt=25.0,
        observers=_observer_coordinates(case),
        timestepper="RK222",
        bridge="minimal",
        dealias=1.5,
    )
    if case.background == "schwarzschild":
        model = SchwarzschildScalarParameters(mass=1.0, ell=1)
        result = run_schwarzschild_scalar_simulation(
            model,
            INITIAL_DATA,
            numerical,
            checkpoint_path=checkpoint,
            checkpoint_dt=500.0,
            explicit_potential=EXPLICIT_POTENTIAL,
        )
        limit = "lim_(r->infinity)(h+r_*)"
    else:
        if case.length is None:
            raise ValueError("An SdS case requires a cosmological length.")
        model = SdSParameters(
            mass=1.0, cosmological_length=case.length, ell=1
        )
        result = run_sds_simulation(
            model,
            INITIAL_DATA,
            numerical,
            checkpoint_path=checkpoint,
            checkpoint_dt=500.0,
            explicit_potential=EXPLICIT_POTENTIAL,
        )
        limit = "lim_(r->r_c)(h+r_*)"
    result.metadata["retarded_time_offset"] = {
        "q": offset,
        "definition": limit,
        "evaluation": "analytic",
    }
    result.metadata["large_l_tail_case"] = {
        **asdict(case),
        "retarded_time": "U=tau-q",
        "time_translation_fitted": False,
        "finite_observers": list(FINITE_OBSERVERS),
        "explicit_potential": EXPLICIT_POTENTIAL,
        "outer_observer": (
            "future null infinity"
            if case.background == "schwarzschild"
            else "future cosmological horizon"
        ),
    }
    temporary = destination.with_suffix(".incomplete.npz")
    try:
        result.save(temporary)
        temporary.rename(destination)
        checkpoint.unlink(missing_ok=True)
        reservation.unlink()
    except BaseException:
        # The reservation deliberately remains after a failed evolution.
        raise
    return destination


def retarded_series(
    result: SdSSimulationResult, observer: int
) -> tuple[np.ndarray, np.ndarray]:
    offset = float(result.metadata["retarded_time_offset"]["q"])
    return result.signal_times - offset, result.signals[:, observer]


def _odd_samples(width: float, step: float, minimum: int = 7) -> int:
    count = max(minimum, int(round(width / step)))
    return count + (count % 2 == 0)


def _centered_sum(values: np.ndarray, count: int) -> np.ndarray:
    """Return complete centered rolling sums in linear time.

    The window sums are accumulated inside blocks of ``count`` samples rather
    than by differencing one global cumulative sum.  A waveform that carries a
    prompt pulse of order unity followed by a tail near the double-precision
    floor spans a dynamic range far beyond ``1/eps``, so a global cumulative
    sum loses the tail completely: the difference of two nearly equal partial
    sums returns zero where the true window sum is finite.  Every sum formed
    here runs over at most ``count`` neighbouring samples, so the rounding
    error stays proportional to the local amplitude instead of to the largest
    amplitude anywhere in the record.
    """

    values = np.asarray(values, dtype=float)
    if count <= 0 or count % 2 == 0 or count > values.size:
        raise ValueError("A rolling window must be odd and fit inside the data.")
    size = values.size
    starts = size - count + 1
    blocks = int(np.ceil((size + count) / count))
    padded = np.zeros(blocks * count, dtype=float)
    padded[:size] = values
    grid = padded.reshape(blocks, count)
    # Prefix sums run from the start of a block, suffix sums from its end, so
    # a window that straddles two blocks is the sum of one suffix and one
    # prefix and never references a partial sum from outside those blocks.
    prefix = np.cumsum(grid, axis=1).reshape(-1)
    suffix = np.cumsum(grid[:, ::-1], axis=1)[:, ::-1].reshape(-1)
    begin = np.arange(starts)
    end = begin + count - 1
    window_sums = suffix[begin] + prefix[end]
    # A window that begins on a block boundary lies inside that single block,
    # where the suffix already spans it.
    aligned = begin % count == 0
    window_sums[aligned] = suffix[begin[aligned]]
    output = np.zeros_like(values)
    half = count // 2
    output[half : size - half] = window_sums
    return output


def rms_envelope(
    times: np.ndarray,
    signal: np.ndarray,
    width: float,
    *,
    floor_multiplier: float = 100.0,
    measured_floor: np.ndarray | None = None,
) -> np.ndarray:
    """Return a centered RMS envelope with unresolved values marked NaN.

    ``measured_floor`` supplies a per-sample amplitude floor obtained from the
    spread of the refinement ladder.  When it is given, a sample survives only
    where the envelope also exceeds ``floor_multiplier`` times that measured
    value, so the reported decay rates stop where the resolutions themselves
    stop agreeing rather than where a round-off estimate says they might.
    """

    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if times.ndim != 1 or signal.shape != times.shape or times.size < 16:
        raise ValueError("Times and signal must be one-dimensional peers.")
    step = float(np.median(np.diff(times)))
    count = _odd_samples(width, step)
    if count >= times.size:
        raise ValueError("Envelope window is longer than the waveform.")
    envelope = np.sqrt(np.maximum(_centered_sum(signal**2, count) / count, 0.0))
    floor = floor_multiplier * numerical_amplitude_floor(signal)
    valid = envelope > floor
    if measured_floor is not None:
        measured_floor = np.asarray(measured_floor, dtype=float)
        if measured_floor.shape != envelope.shape:
            raise ValueError("The measured floor must match the sample grid.")
        resolved = np.isfinite(measured_floor) & (measured_floor > 0.0)
        valid &= ~resolved | (envelope > FLOOR_SAFETY_FACTOR * measured_floor)
    valid[: count // 2] = False
    valid[-(count // 2) :] = False
    envelope[~valid] = np.nan
    return envelope


def ladder_envelope_floor(
    results: dict,
    observer: int,
    settings: "LocalFitSettings",
    background: str = "sds",
) -> dict:
    """Return the amplitude floor measured by the refinement ladder.

    The floor at each retarded time is the larger of the spatial and temporal
    envelope differences carried by the ladder.  The spatial term compares the
    two finest grids and therefore bounds the error of the finer one; the
    coarse term compares the two coarser grids and is reported only so that the
    ratio between them shows the differences are still falling with resolution.
    Nothing here depends on machine epsilon: the floor is what the calculation
    itself says it cannot resolve.
    """

    def envelope(resolution: int, timestep: float) -> tuple[np.ndarray, np.ndarray]:
        result = results[(background, resolution, timestep)]
        times, signal = retarded_series(result, observer)
        return times, rms_envelope(
            times, signal, settings.envelope_width, floor_multiplier=0.0
        )

    times, finest = envelope(FINAL_RESOLUTIONS[2], TIMESTEP)
    medium_times, medium = envelope(FINAL_RESOLUTIONS[1], TIMESTEP)
    coarse_times, coarse = envelope(FINAL_RESOLUTIONS[0], TIMESTEP)
    halved_times, halved = envelope(FINAL_RESOLUTIONS[1], HALVED_TIMESTEP)
    medium_on_grid = _interpolate(medium_times, medium, times)
    coarse_on_grid = _interpolate(coarse_times, coarse, times)
    halved_on_grid = _interpolate(halved_times, halved, times)
    spatial_fine = np.abs(finest - medium_on_grid)
    spatial_coarse = np.abs(medium_on_grid - coarse_on_grid)
    temporal = np.abs(medium_on_grid - halved_on_grid)
    floor = np.maximum(spatial_fine, temporal)
    usable = (
        np.isfinite(spatial_fine)
        & np.isfinite(spatial_coarse)
        & (spatial_fine > 0.0)
        & (spatial_coarse > 0.0)
    )
    ratio = np.full_like(times, np.nan)
    ratio[usable] = spatial_coarse[usable] / spatial_fine[usable]
    return {
        "times": times,
        "amplitude": finest,
        "floor": floor,
        "spatial_fine": spatial_fine,
        "spatial_coarse": spatial_coarse,
        "temporal": temporal,
        "convergence_ratio": ratio,
    }


def trusted_interval_end(
    times: np.ndarray,
    amplitude: np.ndarray,
    floor: np.ndarray,
    *,
    after: float = 100.0,
) -> float | None:
    """Return the end of the first run where the envelope clears the floor."""

    admissible = (
        np.isfinite(amplitude)
        & np.isfinite(floor)
        & (floor > 0.0)
        & (amplitude > FLOOR_SAFETY_FACTOR * floor)
        & (times > after)
    )
    indices = np.flatnonzero(admissible)
    if indices.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(indices) > 1)
    first = indices[: breaks[0] + 1] if breaks.size else indices
    return float(times[first[-1]])


def local_log_fit_rate(
    times: np.ndarray,
    amplitude: np.ndarray,
    width: float,
    *,
    logarithmic_time: bool,
) -> np.ndarray:
    """Return the slope from centered local log-log or semilog fits."""

    times = np.asarray(times, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    step = float(np.median(np.diff(times)))
    count = _odd_samples(width, step, minimum=21)
    if count >= times.size:
        raise ValueError("Local fit window is longer than the waveform.")
    if logarithmic_time:
        x = np.full_like(times, np.nan)
        positive_time = times > 0.0
        x[positive_time] = np.log(times[positive_time])
    else:
        x = times.copy()
    valid = np.isfinite(amplitude) & (amplitude > 0.0) & np.isfinite(x)
    if logarithmic_time:
        valid &= times > 0.0
    y = np.zeros_like(amplitude)
    y[valid] = np.log(amplitude[valid])
    x_work = np.where(valid, x, 0.0)
    n = _centered_sum(valid.astype(float), count)
    sx = _centered_sum(x_work, count)
    sy = _centered_sum(y, count)
    sxx = _centered_sum(x_work * x_work, count)
    sxy = _centered_sum(x_work * y, count)
    denominator = n * sxx - sx * sx
    rate = np.full_like(times, np.nan)
    complete = (n == count) & (denominator > 0.0)
    rate[complete] = -(
        n[complete] * sxy[complete] - sx[complete] * sy[complete]
    ) / denominator[complete]
    return rate


def effective_rates(
    times: np.ndarray,
    signal: np.ndarray,
    settings: LocalFitSettings,
    *,
    kappa: float,
    measured_floor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``A``, ``p_eff`` and ``gamma_eff/kappa``."""

    amplitude = rms_envelope(
        times,
        signal,
        settings.envelope_width,
        floor_multiplier=settings.floor_multiplier,
        measured_floor=measured_floor,
    )
    power = local_log_fit_rate(
        times, amplitude, settings.price_window, logarithmic_time=True
    )
    try:
        gamma = local_log_fit_rate(
            times,
            amplitude,
            settings.exponential_scaled_window / kappa,
            logarithmic_time=False,
        )
    except ValueError as error:
        if str(error) != "Local fit window is longer than the waveform.":
            raise
        gamma = np.full_like(times, np.nan)
    return amplitude, power, gamma / kappa


def _longest_true_interval(
    times: np.ndarray, flag: np.ndarray
) -> tuple[float, float] | None:
    valid = np.flatnonzero(flag)
    if valid.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(valid) > 1) + 1
    groups = np.split(valid, breaks)
    start, end = max(groups, key=lambda group: times[group[-1]] - times[group[0]])[
        [0, -1]
    ]
    return float(times[start]), float(times[end])


def _interpolate(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full_like(target, np.nan)
    output = np.interp(target, times[valid], values[valid])
    output[(target < times[valid][0]) | (target > times[valid][-1])] = np.nan
    return output


def screen_observer(
    coarse_sds: SdSSimulationResult,
    fine_sds: SdSSimulationResult,
    coarse_reference: SdSSimulationResult,
    fine_reference: SdSSimulationResult,
    observer: int,
    settings: LocalFitSettings = LocalFitSettings(),
) -> dict:
    """Apply the continuous Price-law and refinement acceptance criteria."""

    length = float(fine_sds.metadata["model"]["cosmological_length"])
    kappa = cosmological_rate(length)
    fine_t, fine_y = retarded_series(fine_sds, observer)
    coarse_t, coarse_y = retarded_series(coarse_sds, observer)
    ref_t, ref_y = retarded_series(fine_reference, observer)
    coarse_ref_t, coarse_ref_y = retarded_series(coarse_reference, observer)
    fine_a, fine_p, _ = effective_rates(fine_t, fine_y, settings, kappa=kappa)
    coarse_a, _, _ = effective_rates(coarse_t, coarse_y, settings, kappa=kappa)
    ref_a, ref_p, _ = effective_rates(ref_t, ref_y, settings, kappa=kappa)
    coarse_ref_a, _, _ = effective_rates(
        coarse_ref_t, coarse_ref_y, settings, kappa=kappa
    )
    ref_on_fine = _interpolate(ref_t, ref_p, fine_t)
    coarse_a_on_fine = _interpolate(coarse_t, coarse_a, fine_t)
    coarse_ref_on_reference = _interpolate(
        coarse_ref_t, coarse_ref_a, ref_t
    )
    target = price_target(observer)
    tolerance = PRICE_TOLERANCE * target
    price = (
        (np.abs(fine_p - target) <= tolerance)
        & (np.abs(ref_on_fine - target) <= tolerance)
        & (np.abs(fine_p - ref_on_fine) <= tolerance)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sds_refinement = np.abs(fine_a - coarse_a_on_fine) / fine_a
        reference_refinement_native = (
            np.abs(ref_a - coarse_ref_on_reference) / ref_a
        )
    reference_refinement = _interpolate(
        ref_t, reference_refinement_native, fine_t
    )
    refinement = (sds_refinement <= 0.01) & (reference_refinement <= 0.01)
    accepted = price & refinement & (fine_t >= 0.0)
    interval = _longest_true_interval(fine_t, accepted)
    duration = 0.0 if interval is None else interval[1] - interval[0]
    covers_anchor = bool(
        interval is not None
        and interval[0] <= SCREEN_PRICE_ANCHOR_U <= interval[1]
    )
    return {
        "length_over_M": length,
        "observer": "outer" if observer == 2 else f"r{FINITE_OBSERVERS[observer]:g}M",
        "price_interval_start_U_over_M": None if interval is None else interval[0],
        "price_interval_end_U_over_M": None if interval is None else interval[1],
        "continuous_duration_U_over_M": duration,
        "passes_price_duration": bool(duration >= PRICE_DURATION),
        "covers_screen_anchor_U_over_M": covers_anchor,
        "passes_screen": bool(duration >= PRICE_DURATION and covers_anchor),
        "price_target": target,
        "relative_price_tolerance": PRICE_TOLERANCE,
        "required_duration_U_over_M": PRICE_DURATION,
        "maximum_sds_refinement_in_interval": (
            None
            if interval is None
            else float(
                np.nanmax(
                    sds_refinement[(fine_t >= interval[0]) & (fine_t <= interval[1])]
                )
            )
        ),
        "maximum_reference_refinement_in_interval": (
            None
            if interval is None
            else float(
                np.nanmax(
                    reference_refinement[
                        (fine_t >= interval[0]) & (fine_t <= interval[1])
                    ]
                )
            )
        ),
        "time_translation_fitted": False,
    }


def _load_screen_set(output_dir: Path, length: float) -> tuple[SdSSimulationResult, ...]:
    by_key: dict[tuple[str, int], SdSSimulationResult] = {}
    for case in screening_cases(length):
        path = archive_path(output_dir, case)
        if not path.exists():
            raise FileNotFoundError(path)
        by_key[(case.background, case.resolution)] = load_sds_result(path)
    return (
        by_key[("sds", 1536)],
        by_key[("sds", 2048)],
        by_key[("schwarzschild", 1536)],
        by_key[("schwarzschild", 2048)],
    )


def analyze_screen(output_dir: Path, length: float) -> list[dict]:
    """Analyze all three observers and write a screening decision."""

    results = _load_screen_set(output_dir, length)
    rows = [screen_observer(*results, observer) for observer in range(3)]
    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / f"screen_L{length:g}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "length_over_M": length,
        "primary_observer": "outer",
        "accepted": bool(rows[-1]["passes_screen"]),
        "criteria": {
            "outer_price_target": PRICE_TARGET_INFINITY,
            "fixed_radius_price_target": PRICE_TARGET_FIXED_RADIUS,
            "relative_tolerance": PRICE_TOLERANCE,
            "continuous_duration_U_over_M": PRICE_DURATION,
            "required_price_anchor_U_over_M": SCREEN_PRICE_ANCHOR_U,
            "maximum_refinement_fraction": 0.01,
            "relative_time_translation": "not fitted",
        },
        "observers": rows,
    }
    with (tables / f"screen_L{length:g}.json").open("w", encoding="utf-8") as stream:
        json.dump(json_safe(decision), stream, indent=2, allow_nan=False)
    return rows


def select_length(output_dir: Path) -> float | None:
    """Return the smallest completed screening length accepted at the horizon."""

    for length in SCREEN_LENGTHS:
        decision = Path(output_dir) / "tables" / f"screen_L{length:g}.json"
        if not decision.exists():
            return None
        with decision.open(encoding="utf-8") as stream:
            if json.load(stream)["accepted"]:
                return length
    return None


def _anchored_long_interval_end(
    times: np.ndarray,
    flag: np.ndarray,
    minimum_duration: float,
    anchor: float,
) -> float | None:
    """Return the end of the qualifying interval that contains ``anchor``."""

    valid = np.flatnonzero(np.asarray(flag, dtype=bool))
    if valid.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(valid) > 1) + 1
    groups = np.split(valid, breaks)
    for group in groups:
        start = float(times[group[0]])
        end = float(times[group[-1]])
        if end - start >= minimum_duration and start <= anchor <= end:
            return end
    return None


def persistent_cosmological_entry(
    times: np.ndarray,
    normalized_gamma: np.ndarray,
    departure: float,
    *,
    tolerance: float = COSMOLOGICAL_TOLERANCE,
    minimum_scaled_duration: float = COSMOLOGICAL_MINIMUM_SCALED_DURATION,
    kappa: float,
) -> float | None:
    """Return the start of the final resolved run consistent with unity."""

    valid = np.isfinite(normalized_gamma) & (times > departure)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return None
    # A numerical floor ends the resolved record. Entry must persist from its
    # start to that endpoint, not through samples that have already been cut.
    final = indices[-1]
    consistent = valid & (np.abs(normalized_gamma - 1.0) <= tolerance)
    start = final
    while start >= 0 and consistent[start]:
        start -= 1
    start += 1
    if start > final:
        return None
    duration = times[final] - times[start]
    if kappa * duration < minimum_scaled_duration:
        return None
    return float(times[start])


def _weighted_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Return the ordinary least squares intercept and slope of ``y`` on ``x``."""

    if x.size < 3:
        return None
    matrix = np.vstack((np.ones_like(x), x)).T
    solution, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return float(solution[0]), float(solution[1])


def fitted_law_intersection(
    times: np.ndarray,
    amplitude: np.ndarray,
    price_interval: tuple[float, float],
    cosmological_interval: tuple[float, float],
) -> dict | None:
    r"""Intersect the two fitted decay laws.

    A power law ``ln A = a_p - p ln U`` is fitted on the Price interval and an
    exponential ``ln A = a_c - \gamma U`` on the cosmological interval.  Their
    crossing is quoted only as a representative transition time: each law is
    extrapolated well outside the window it was fitted on, so the measured
    crossover interval remains the defensible result.
    """

    times = np.asarray(times, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    usable = np.isfinite(amplitude) & (amplitude > 0.0) & (times > 0.0)

    price = usable & (times >= price_interval[0]) & (times <= price_interval[1])
    cosmological = (
        usable
        & (times >= cosmological_interval[0])
        & (times <= cosmological_interval[1])
    )
    if np.count_nonzero(price) < 3 or np.count_nonzero(cosmological) < 3:
        return None

    power_fit = _weighted_line(
        np.log(times[price]), np.log(amplitude[price])
    )
    exponential_fit = _weighted_line(
        times[cosmological], np.log(amplitude[cosmological])
    )
    if power_fit is None or exponential_fit is None:
        return None
    power_intercept, power_slope = power_fit
    exponential_intercept, exponential_slope = exponential_fit

    def difference(value: float) -> float:
        return (power_intercept + power_slope * np.log(value)) - (
            exponential_intercept + exponential_slope * value
        )

    left = max(price_interval[1], 1e-6)
    right = max(cosmological_interval[1], left * (1.0 + 1e-9))
    if difference(left) * difference(right) > 0.0:
        return None
    for _ in range(200):
        middle = 0.5 * (left + right)
        if difference(left) * difference(middle) <= 0.0:
            right = middle
        else:
            left = middle
    crossing = 0.5 * (left + right)
    return {
        "intersection_U_over_M": float(crossing),
        "fitted_price_index": float(-power_slope),
        "fitted_cosmological_rate": float(-exponential_slope),
        "price_fit_interval_U_over_M": [float(v) for v in price_interval],
        "cosmological_fit_interval_U_over_M": [
            float(v) for v in cosmological_interval
        ],
        "status": "representative_only_laws_extrapolated_outside_their_windows",
    }


def measure_transition(
    sds: SdSSimulationResult,
    reference: SdSSimulationResult,
    observer: int,
    settings: LocalFitSettings,
    *,
    price_duration: float = PRICE_DURATION,
    cosmological_tolerance: float = COSMOLOGICAL_TOLERANCE,
    measured_floor: np.ndarray | None = None,
) -> dict:
    """Measure Price departure and persistent cosmological entry."""

    length = float(sds.metadata["model"]["cosmological_length"])
    kappa = cosmological_rate(length)
    times, signal = retarded_series(sds, observer)
    reference_times, reference_signal = retarded_series(reference, observer)
    amplitude, power, normalized_gamma = effective_rates(
        times, signal, settings, kappa=kappa, measured_floor=measured_floor
    )
    _, reference_power, _ = effective_rates(
        reference_times, reference_signal, settings, kappa=kappa
    )
    reference_on_times = _interpolate(reference_times, reference_power, times)
    target = price_target(observer)
    tolerance = PRICE_TOLERANCE * target
    price = (
        (times >= 0.0)
        & (np.abs(power - target) <= tolerance)
        & (np.abs(reference_on_times - target) <= tolerance)
        & (np.abs(power - reference_on_times) <= tolerance)
    )
    departure = _anchored_long_interval_end(
        times,
        price,
        price_duration,
        SCREEN_PRICE_ANCHOR_U,
    )
    entry = (
        None
        if departure is None
        else persistent_cosmological_entry(
            times,
            normalized_gamma,
            departure,
            tolerance=cosmological_tolerance,
            kappa=kappa,
        )
    )
    return {
        "departure_U_over_M": departure,
        "entry_U_over_M": entry,
        "departure_kappa_U": None if departure is None else kappa * departure,
        "entry_kappa_U": None if entry is None else kappa * entry,
        "crossover_duration_U_over_M": (
            None if departure is None or entry is None else entry - departure
        ),
        "cosmological_rate_relative_tolerance": cosmological_tolerance,
        "minimum_cosmological_scaled_duration": (
            COSMOLOGICAL_MINIMUM_SCALED_DURATION
        ),
        "status": (
            "no_resolved_price_interval"
            if departure is None
            else (
                "no_cosmological_entry_before_ladder_floor"
                if measured_floor is not None
                else "no_cosmological_entry_before_round_off_floor"
            )
            if entry is None
            else "resolved"
        ),
        "floor_source": (
            "refinement_ladder" if measured_floor is not None else "round_off_estimate"
        ),
        "trusted_until_U_over_M": (
            None
            if measured_floor is None
            else trusted_interval_end(times, amplitude, measured_floor)
        ),
        "amplitude": amplitude,
        "power": power,
        "normalized_gamma": normalized_gamma,
        "times": times,
    }


def _load_final_set(
    output_dir: Path, length: float
) -> dict[tuple[str, int, float], SdSSimulationResult]:
    results: dict[tuple[str, int, float], SdSSimulationResult] = {}
    for case in final_cases(length):
        path = archive_path(output_dir, case)
        if not path.exists():
            raise FileNotFoundError(path)
        results[(case.background, case.resolution, case.timestep)] = (
            load_sds_result(path)
        )
    return results


def _waveform_sensitivity(
    reference: SdSSimulationResult,
    candidate: SdSSimulationResult,
    observer: int,
    interval: tuple[float, float],
    settings: LocalFitSettings,
) -> float | None:
    reference_times, reference_signal = retarded_series(reference, observer)
    candidate_times, candidate_signal = retarded_series(candidate, observer)
    reference_amplitude = rms_envelope(
        reference_times,
        reference_signal,
        settings.envelope_width,
        floor_multiplier=settings.floor_multiplier,
    )
    candidate_amplitude = rms_envelope(
        candidate_times,
        candidate_signal,
        settings.envelope_width,
        floor_multiplier=settings.floor_multiplier,
    )
    candidate_on_reference = _interpolate(
        candidate_times, candidate_amplitude, reference_times
    )
    inside = (
        (reference_times >= interval[0])
        & (reference_times <= interval[1])
        & np.isfinite(reference_amplitude)
        & np.isfinite(candidate_on_reference)
    )
    if not inside.any():
        return None
    return float(
        np.max(
            np.abs(reference_amplitude[inside] - candidate_on_reference[inside])
            / reference_amplitude[inside]
        )
    )


def analyze_final(output_dir: Path, length: float) -> dict:
    """Create the final transition table, sensitivities, and three-panel figure."""

    results = _load_final_set(output_dir, length)
    primary_sds = results[("sds", 3072, TIMESTEP)]
    primary_reference = results[("schwarzschild", 3072, TIMESTEP)]
    kappa_c = cosmological_rate(length)
    rows: list[dict] = []
    primary: dict[int, dict] = {}
    ladder: dict[int, dict] = {}
    for observer in range(3):
        ladder[observer] = ladder_envelope_floor(
            results, observer, LocalFitSettings()
        )
        for scaled_window in (0.15, 0.25, 0.4):
            for floor_multiplier in (10.0, 100.0, 1000.0):
                settings = LocalFitSettings(
                    exponential_scaled_window=scaled_window,
                    floor_multiplier=floor_multiplier,
                )
                measurement = measure_transition(
                    primary_sds,
                    primary_reference,
                    observer,
                    settings,
                    measured_floor=ladder[observer]["floor"],
                )
                if scaled_window == 0.25 and floor_multiplier == 100.0:
                    primary[observer] = measurement
                rows.append(
                    {
                        "observer": (
                            "outer"
                            if observer == 2
                            else f"r{FINITE_OBSERVERS[observer]:g}M"
                        ),
                        "resolution": 3072,
                        "timestep_over_M": TIMESTEP,
                        "exponential_scaled_window": scaled_window,
                        "floor_multiplier": floor_multiplier,
                        "floor_fraction_of_waveform_peak": (
                            floor_multiplier
                            * 1000.0
                            * np.finfo(float).eps
                        ),
                        **{
                            key: value
                            for key, value in measurement.items()
                            if not isinstance(value, np.ndarray)
                        },
                    }
                )

    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    transition_path = tables / f"final_L{length:g}_transition_sweep.csv"
    with transition_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numerical_transition_rows: list[dict] = []
    for resolution, timestep in (
        (1536, TIMESTEP),
        (2048, TIMESTEP),
        (3072, TIMESTEP),
        (2048, HALVED_TIMESTEP),
    ):
        measurement = measure_transition(
            results[("sds", resolution, timestep)],
            results[("schwarzschild", resolution, timestep)],
            2,
            LocalFitSettings(),
        )
        numerical_transition_rows.append(
            {
                "observer": "outer",
                "resolution": resolution,
                "timestep_over_M": timestep,
                **{
                    key: value
                    for key, value in measurement.items()
                    if not isinstance(value, np.ndarray)
                },
            }
        )
    numerical_transition_path = (
        tables / f"final_L{length:g}_transition_numerical.csv"
    )
    with numerical_transition_path.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(numerical_transition_rows[0])
        )
        writer.writeheader()
        writer.writerows(numerical_transition_rows)

    primary_outer = primary[2]
    if primary_outer["departure_U_over_M"] is None:
        price_interval = (0.0, 0.0)
    else:
        price_interval = (
            primary_outer["departure_U_over_M"] - PRICE_DURATION,
            primary_outer["departure_U_over_M"],
        )
    if primary_outer["entry_U_over_M"] is None:
        cosmological_interval = (0.0, 0.0)
    else:
        cosmological_interval = (
            primary_outer["entry_U_over_M"],
            float(
                primary_outer["times"][
                    np.flatnonzero(
                        np.isfinite(primary_outer["normalized_gamma"])
                    )[-1]
                ]
            ),
        )
    floor_rows: list[dict] = []
    for observer in range(3):
        record = ladder[observer]
        finite = np.isfinite(record["convergence_ratio"])
        inside = finite & (record["times"] > 200.0)
        if primary[observer]["departure_U_over_M"] is not None:
            inside &= record["times"] < primary[observer]["departure_U_over_M"]
        trusted = trusted_interval_end(
            record["times"], record["amplitude"], record["floor"]
        )
        floor_rows.append(
            {
                "observer": (
                    "outer" if observer == 2 else f"r{FINITE_OBSERVERS[observer]:g}M"
                ),
                "price_target": price_target(observer),
                "trusted_until_U_over_M": trusted,
                "trusted_until_kappa_U": (
                    None if trusted is None else kappa_c * trusted
                ),
                "median_spatial_convergence_ratio": (
                    float(np.median(record["convergence_ratio"][inside]))
                    if np.any(inside)
                    else None
                ),
                "floor_safety_factor": FLOOR_SAFETY_FACTOR,
                "amplitude_at_trusted_end": (
                    None
                    if trusted is None
                    else float(
                        record["amplitude"][
                            int(np.searchsorted(record["times"], trusted))
                        ]
                    )
                ),
            }
        )
    floor_path = tables / f"final_L{length:g}_ladder_floor.csv"
    with floor_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(floor_rows[0]))
        writer.writeheader()
        writer.writerows(floor_rows)

    sensitivity_rows: list[dict] = []
    for background, kind, sensitivity_reference, candidate in (
        (
            "sds",
            "spatial_N2048_to_N3072",
            primary_sds,
            results[("sds", 2048, TIMESTEP)],
        ),
        (
            "sds",
            "spatial_N1536_to_N2048",
            results[("sds", 2048, TIMESTEP)],
            results[("sds", 1536, TIMESTEP)],
        ),
        (
            "sds",
            "timestep_0p00125_to_0p0025_at_N2048",
            results[("sds", 2048, TIMESTEP)],
            results[("sds", 2048, HALVED_TIMESTEP)],
        ),
        (
            "schwarzschild",
            "spatial_N2048_to_N3072",
            primary_reference,
            results[("schwarzschild", 2048, TIMESTEP)],
        ),
        (
            "schwarzschild",
            "spatial_N1536_to_N2048",
            results[("schwarzschild", 2048, TIMESTEP)],
            results[("schwarzschild", 1536, TIMESTEP)],
        ),
        (
            "schwarzschild",
            "timestep_0p00125_to_0p0025_at_N2048",
            results[("schwarzschild", 2048, TIMESTEP)],
            results[("schwarzschild", 2048, HALVED_TIMESTEP)],
        ),
    ):
        sensitivity_rows.append(
            {
                "background": background,
                "sensitivity": kind,
                "observer": "outer",
                "maximum_relative_envelope_difference_price": _waveform_sensitivity(
                    sensitivity_reference,
                    candidate,
                    2,
                    price_interval,
                    LocalFitSettings(),
                ),
                "maximum_relative_envelope_difference_cosmological": _waveform_sensitivity(
                    sensitivity_reference,
                    candidate,
                    2,
                    cosmological_interval,
                    LocalFitSettings(),
                ),
            }
        )
    sensitivity_path = tables / f"final_L{length:g}_numerical_sensitivities.csv"
    with sensitivity_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), sharex=True)
    colors_by_observer = ("#0072B2", "#009E73", "#D55E00")
    labels = (r"$r=8M$", r"$r=16M$", r"$\mathcal{H}_c^+$")
    kappa = cosmological_rate(length)
    trusted_marks: dict[int, float] = {}
    for observer, color, label in zip(range(3), colors_by_observer, labels):
        measurement = primary[observer]
        reference_times, reference_signal = retarded_series(
            primary_reference, observer
        )
        # The Schwarzschild reference carries its own refinement ladder and its
        # own floor.  Without it the reference curve runs on past the point
        # where the resolutions disagree, and the wander it develops there
        # would read as a measured Price index rather than as noise.
        reference_ladder = ladder_envelope_floor(
            results, observer, LocalFitSettings(), background="schwarzschild"
        )
        reference_amplitude, reference_power, _ = effective_rates(
            reference_times,
            reference_signal,
            LocalFitSettings(),
            kappa=kappa,
            measured_floor=reference_ladder["floor"],
        )
        trusted_marks[observer] = max(
            value
            for value in (
                trusted_interval_end(
                    measurement["times"],
                    measurement["amplitude"],
                    ladder[observer]["floor"],
                )
                or 0.0,
                trusted_interval_end(
                    reference_ladder["times"],
                    reference_ladder["amplitude"],
                    reference_ladder["floor"],
                )
                or 0.0,
            )
        )
        axes[0].semilogy(
            measurement["times"], measurement["amplitude"], color=color, label=label
        )
        axes[0].semilogy(
            reference_times,
            reference_amplitude,
            color=color,
            linestyle="--",
            alpha=0.72,
        )
        axes[1].plot(measurement["times"], measurement["power"], color=color)
        axes[1].plot(
            reference_times,
            reference_power,
            color=color,
            linestyle="--",
            alpha=0.72,
        )
        axes[2].plot(
            measurement["times"], measurement["normalized_gamma"], color=color
        )
    axes[1].axhspan(2.85, 3.15, color="#D55E00", alpha=0.12)
    axes[1].axhspan(4.75, 5.25, color="#0072B2", alpha=0.08)
    axes[1].text(
        0.99,
        3.16,
        r"$p=3$ at the outer boundary",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[1].text(
        0.99,
        5.26,
        r"$p=5$ at fixed radius",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[2].axhspan(0.9, 1.1, color="0.85")
    axes[2].set_yscale("log")
    axes[2].text(
        0.99,
        1.15,
        r"$\gamma_{\rm eff}/\kappa_c=1$, the cosmological target",
        transform=axes[2].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    outer_trusted = trusted_marks.get(2)
    if outer_trusted:
        for axis in axes:
            axis.axvline(outer_trusted, color="#7f2704", linestyle="-", linewidth=1.1,
                         alpha=0.85)
        axes[1].text(
            outer_trusted,
            0.97,
            "  ladder floor:\n  nothing is read beyond",
            transform=axes[1].get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=7.5,
            color="#7f2704",
        )
        for axis in axes:
            axis.set_xlim(-0.02 * outer_trusted, 1.06 * outer_trusted)
    departure = primary_outer["departure_U_over_M"]
    entry = primary_outer["entry_U_over_M"]
    intersection = fitted_law_intersection(
        primary_outer["times"],
        primary_outer["amplitude"],
        price_interval,
        cosmological_interval,
    )
    if intersection is not None:
        for axis in axes:
            axis.axvline(
                intersection["intersection_U_over_M"],
                color="#8B4513",
                linestyle="-.",
                linewidth=1.0,
                alpha=0.8,
            )
        axes[2].text(
            intersection["intersection_U_over_M"],
            0.03,
            "fitted-law\ncrossing",
            transform=axes[2].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#8B4513",
        )
    if departure is not None and entry is None:
        for axis in axes:
            axis.axvline(departure, color="0.25", linestyle=":")
        axes[0].text(
            departure,
            0.03,
            r"  $U_{\rm P}$",
            transform=axes[0].get_xaxis_transform(),
            ha="left",
            va="bottom",
        )
    if departure is not None and entry is not None:
        for axis in axes:
            axis.axvspan(departure, entry, color="#F0E442", alpha=0.22)
            axis.axvline(departure, color="0.25", linestyle=":")
            axis.axvline(entry, color="0.25", linestyle="--")
        axes[0].text(
            departure,
            0.03,
            r"$U_{\rm P}$",
            transform=axes[0].get_xaxis_transform(),
            ha="right",
            va="bottom",
        )
        axes[0].text(
            entry,
            0.03,
            r"$U_{\rm dS}$",
            transform=axes[0].get_xaxis_transform(),
            ha="left",
            va="bottom",
        )
    axes[0].set_ylabel(r"RMS envelope $A$")
    axes[1].set_ylabel(r"$p_{\rm eff}$")
    axes[2].set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
    axes[2].set_xlabel(r"geometric retarded time $U/M$")
    observer_handles, observer_labels = axes[0].get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="0.2", linestyle="-", label="SdS"),
        Line2D([0], [0], color="0.2", linestyle="--", label="Schwarzschild"),
    ]
    axes[0].legend(
        observer_handles + style_handles,
        observer_labels + ["SdS", "Schwarzschild"],
        frameon=False,
        ncols=3,
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    png = Path(output_dir) / f"large_L{length:g}_tail_transition.png"
    pdf = Path(output_dir) / f"large_L{length:g}_tail_transition.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    summary = {
        "length_over_M": length,
        "primary_observer": "outer",
        "primary": {
            key: value
            for key, value in primary_outer.items()
            if not isinstance(value, np.ndarray)
        },
        # Secondary to the crossover interval, by construction.
        "fitted_law_intersection": intersection,
        "price_fit_interval_U_over_M": [float(v) for v in price_interval],
        "cosmological_fit_interval_U_over_M": [
            float(v) for v in cosmological_interval
        ],
        "transition_sweep": str(transition_path),
        "transition_numerical": str(numerical_transition_path),
        "numerical_sensitivities": str(sensitivity_path),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
    }
    with (tables / f"final_L{length:g}_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)
    return summary


def _plot_screen(output_dir: Path, length: float) -> Path:
    coarse, fine, coarse_ref, fine_ref = _load_screen_set(output_dir, length)
    del coarse, coarse_ref
    kappa = cosmological_rate(length)
    settings = LocalFitSettings()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), sharex=True)
    for observer, label, color in ((0, r"$r=8M$", "#0072B2"), (1, r"$r=16M$", "#009E73"), (2, r"$\mathcal{H}_c^+$", "#D55E00")):
        times, signal = retarded_series(fine, observer)
        ref_times, ref_signal = retarded_series(fine_ref, observer)
        amplitude, power, gamma = effective_rates(times, signal, settings, kappa=kappa)
        _, ref_power, _ = effective_rates(ref_times, ref_signal, settings, kappa=kappa)
        axes[0].semilogy(times, amplitude, color=color, label=label)
        axes[1].plot(times, power, color=color)
        axes[1].plot(ref_times, ref_power, color=color, alpha=0.45, linestyle=":")
        axes[2].plot(times, gamma, color=color)
    axes[1].axhspan(2.85, 3.15, color="#D55E00", alpha=0.12)
    axes[1].axhspan(4.75, 5.25, color="#0072B2", alpha=0.08)
    axes[1].text(
        0.99,
        3.16,
        r"$p=3$ at the outer boundary",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[1].text(
        0.99,
        5.26,
        r"$p=5$ at fixed radius",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[2].axhspan(0.9, 1.1, color="0.85")
    axes[0].set_ylabel(r"RMS envelope $A$")
    axes[1].set_ylabel(r"$p_{\rm eff}$")
    axes[2].set_ylabel(r"$\gamma_{\rm eff}/\kappa_c$")
    axes[2].set_xlabel(r"geometric retarded time $U/M$")
    axes[0].legend(frameon=False, ncols=3)
    axes[0].set_xlim(0.0, SCREEN_END_U)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    path = Path(output_dir) / f"screen_L{length:g}_rates.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run named screening or final cases")
    run.add_argument("cases", nargs="+")
    screen = subparsers.add_parser("screen", help="run one two-level screening set")
    screen.add_argument("length", type=float, choices=SCREEN_LENGTHS)
    analyze = subparsers.add_parser("analyze-screen")
    analyze.add_argument("length", type=float, choices=SCREEN_LENGTHS)
    final = subparsers.add_parser("final", help="run the final ladder for one length")
    final.add_argument("length", type=float, choices=SCREEN_LENGTHS)
    final_report = subparsers.add_parser("report-final")
    final_report.add_argument("length", type=float, choices=SCREEN_LENGTHS)
    campaign = subparsers.add_parser(
        "campaign",
        help="resume screening in increasing L and run the first accepted final case",
    )
    campaign.add_argument(
        "--resume-interrupted",
        action="store_true",
        help="reuse a reserved case that has a compatible checkpoint",
    )
    subparsers.add_parser("list")
    arguments = parser.parse_args()

    catalogue = {
        case.name: case
        for length in SCREEN_LENGTHS
        for case in screening_cases(length)
    }
    catalogue.update(
        {
            case.name: case
            for length in SCREEN_LENGTHS
            for case in final_cases(length)
        }
    )
    if arguments.command == "list":
        for name in catalogue:
            print(name)
    elif arguments.command == "run":
        for name in arguments.cases:
            if name not in catalogue:
                raise ValueError(f"Unknown case {name!r}.")
            print(run_case(catalogue[name], arguments.output_dir))
    elif arguments.command == "screen":
        for case in screening_cases(arguments.length):
            path = archive_path(arguments.output_dir, case)
            if path.exists():
                print(path)
            else:
                print(run_case(case, arguments.output_dir))
        analyze_screen(arguments.output_dir, arguments.length)
        print(_plot_screen(arguments.output_dir, arguments.length))
    elif arguments.command == "analyze-screen":
        print(analyze_screen(arguments.output_dir, arguments.length))
        print(_plot_screen(arguments.output_dir, arguments.length))
    elif arguments.command == "final":
        for case in final_cases(arguments.length):
            path = archive_path(arguments.output_dir, case)
            if path.exists():
                print(path)
            else:
                print(run_case(case, arguments.output_dir))
        print(analyze_final(arguments.output_dir, arguments.length))
    elif arguments.command == "report-final":
        print(analyze_final(arguments.output_dir, arguments.length))
    elif arguments.command == "campaign":
        selected = None
        for length in SCREEN_LENGTHS:
            for case in screening_cases(length):
                path = archive_path(arguments.output_dir, case)
                print(
                    path
                    if path.exists()
                    else run_case(
                        case,
                        arguments.output_dir,
                        resume_interrupted=arguments.resume_interrupted,
                    )
                )
            rows = analyze_screen(arguments.output_dir, length)
            print(_plot_screen(arguments.output_dir, length))
            if rows[-1]["passes_screen"]:
                selected = length
                break
        if selected is None:
            raise RuntimeError(
                "No screened length resolves the required asymptotic Price interval."
            )
        print(f"selected L/M={selected:g}")
        for case in final_cases(selected):
            path = archive_path(arguments.output_dir, case)
            print(
                path
                if path.exists()
                else run_case(
                    case,
                    arguments.output_dir,
                    resume_interrupted=arguments.resume_interrupted,
                )
            )
        print(analyze_final(arguments.output_dir, selected))


if __name__ == "__main__":
    main()
