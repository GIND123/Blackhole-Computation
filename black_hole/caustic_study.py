r"""Retarded-Green-function study: caustic echoes and the SdS flat limit.

This module defines and drives the sourced three-dimensional suite:

*   a Schwarzschild run that reproduces the direct signal and the caustic
    echo train of Zenginoglu and Galley, Phys. Rev. D 86, 064030 (2012);
*   the same physical emitter on Schwarzschild--de Sitter for
    ``L/M = 20, 40, 80, 160``, compared with the Schwarzschild waveform on
    the common geometric clock ``U = tau - q_L``;
*   radial, temporal, angular, and stencil-order refinement ladders;
*   an independent static-coordinate cross-check of the source term; and
*   a narrower emitter at higher resolution, following the instruction to
    begin with a broad source and then sharpen it.

Analysis helpers that turn modal archives into angular waveforms, caustic
echo measurements, and flat-limit norms live here as well.  Figures and
tables are assembled in :mod:`black_hole.caustic_report`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import correlate, find_peaks, hilbert

from .localized_source import LocalizedSourceParameters
from .source_evolution import (
    SourcedNumericalParameters,
    SourcedSimulationResult,
    load_sourced_result,
    run_sourced_simulation,
)
from .three_d_solver import real_spherical_harmonic

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Suite definition
# --------------------------------------------------------------------------

BROAD_SOURCE = LocalizedSourceParameters(
    amplitude=1.0,
    center_radius=6.0,
    radial_half_width=1.5,
    time_center=30.0,
    time_half_width=4.0,
    angular_concentration=16.0,
)
"""The broad emitter used for the production suite (``sigma = 0.25`` rad)."""

NARROW_SOURCE = LocalizedSourceParameters(
    amplitude=1.0,
    center_radius=6.0,
    radial_half_width=1.0,
    time_center=30.0,
    time_half_width=2.5,
    angular_concentration=49.0,
)
"""A sharper emitter (``sigma = 1/7`` rad) run at higher resolution."""

PRODUCTION = SourcedNumericalParameters(
    radial_resolution=1024,
    angular_ell_max=16,
    timestep=0.004,
    end_time=600.0,
    signal_dt=0.1,
    diagnostic_dt=2.0,
    snapshot_dt=2.0,
    snapshot_end_time=0.0,
    snapshot_radial_points=180,
    observer_radii=(8.0, 12.0, None),
)

NARROW_NUMERICAL = replace(
    PRODUCTION,
    angular_ell_max=24,
    end_time=200.0,
    snapshot_end_time=0.0,
)

COSMOLOGICAL_LENGTHS = (20.0, 40.0, 80.0, 160.0)

END_TIME = {
    "schwarzschild": 600.0,
    20.0: 250.0,
    40.0: 350.0,
    80.0: 450.0,
    160.0: 600.0,
}
"""Final bridge times.  Each SdS run reaches ``kappa_c U > 3.5``."""

SNAPSHOT_CASES = {"schwarzschild", 80.0}
"""Cases that store the equatorial field history for the caustic figures."""

SNAPSHOT_SETTINGS = {
    "snapshot_dt": 2.0,
    "snapshot_end_time": 180.0,
    "snapshot_radial_points": 180,
}

RADIAL_LADDER = (512, 768, 1024, 1536)
TIMESTEP_LADDER = (0.004, 0.002, 0.001)
ANGULAR_LADDER = (8, 12, 16, 20)
CONVERGENCE_END_TIME = 160.0

PHOTON_SPHERE_PERIOD = 6.0 * np.sqrt(3.0) * np.pi
"""Coordinate period ``2 pi r/ sqrt(f)`` of the ``r=3M`` photon orbit."""

VALIDATION_WINDOW = (5.0, 115.0)
"""Retarded-time window, in ``M``, used for every flat-limit norm."""

SOURCE_BACKENDS = ("finite-difference", "dedalus")
"""Available radial backends for the same localized-source equations."""


def case_label(case: str | float) -> str:
    if case == "schwarzschild":
        return "schwarzschild"
    return f"sds_L{float(case):g}"


def case_title(case: str | float) -> str:
    if case == "schwarzschild":
        return "Schwarzschild"
    return rf"SdS $L/M={float(case):g}$"


# --------------------------------------------------------------------------
# Run drivers
# --------------------------------------------------------------------------


def _archive(directory: Path, name: str) -> Path:
    return Path(directory) / "raw" / f"{name}.npz"


def _backend_directory(directory: Path, backend: str) -> Path:
    if backend not in SOURCE_BACKENDS:
        raise ValueError(f"Unknown source backend {backend!r}.")
    if backend == "dedalus":
        return Path(directory) / "dedalus"
    return Path(directory)


def _execute(
    path: Path,
    *,
    case: str | float,
    source: LocalizedSourceParameters,
    numerical: SourcedNumericalParameters,
    force: bool,
    backend: str,
) -> Path:
    if path.exists() and not force:
        LOGGER.info("reusing %s", path)
        return path
    background = "schwarzschild" if case == "schwarzschild" else "sds"
    length = 80.0 if case == "schwarzschild" else float(case)
    LOGGER.info(
        "running %s: N=%d, ell_max=%d, dt=%g, T=%g",
        path.name,
        numerical.radial_resolution,
        numerical.angular_ell_max,
        numerical.timestep,
        numerical.end_time,
    )
    arguments = {
        "background": background,
        "source": source,
        "numerical": numerical,
        "cosmological_length": length,
    }
    if backend == "finite-difference":
        result = run_sourced_simulation(**arguments)
    elif backend == "dedalus":
        # Keep Dedalus optional for report generation and for the lightweight
        # NumPy/SciPy environment used by the finite-difference production run.
        from .dedalus_source_evolution import run_sourced_dedalus_simulation

        result = run_sourced_dedalus_simulation(**arguments)
    else:
        raise ValueError(f"Unknown source backend {backend!r}.")
    result.save(path)
    return path


def run_production_case(
    output_dir: Path,
    case: str | float,
    *,
    force: bool = False,
    backend: str = "finite-difference",
) -> Path:
    """Run one production case of the broad-source suite."""

    numerical = replace(PRODUCTION, end_time=END_TIME[case])
    if case in SNAPSHOT_CASES:
        numerical = replace(numerical, **SNAPSHOT_SETTINGS)
    return _execute(
        _archive(_backend_directory(output_dir, backend), case_label(case)),
        case=case,
        source=BROAD_SOURCE,
        numerical=numerical,
        force=force,
        backend=backend,
    )


def run_narrow_case(
    output_dir: Path,
    case: str | float,
    *,
    force: bool = False,
    backend: str = "finite-difference",
) -> Path:
    """Run one case of the sharpened-emitter follow-up."""

    numerical = replace(
        NARROW_NUMERICAL,
        end_time=min(END_TIME[case], NARROW_NUMERICAL.end_time),
    )
    return _execute(
        _backend_directory(output_dir, backend)
        / "narrow"
        / "raw"
        / f"{case_label(case)}.npz",
        case=case,
        source=NARROW_SOURCE,
        numerical=numerical,
        force=force,
        backend=backend,
    )


LATE_TIME_CHECK_END_TIME = 300.0
"""Final time of the sharpened late-time resolution check at ``L/M = 80``."""


def convergence_cases() -> list[tuple[str, str | float, SourcedNumericalParameters]]:
    """Return the labelled refinement ladders and the case each one uses."""

    base = replace(
        PRODUCTION, end_time=CONVERGENCE_END_TIME, snapshot_end_time=0.0
    )
    cases: list[tuple[str, str | float, SourcedNumericalParameters]] = []
    for resolution in RADIAL_LADDER:
        # A fixed small timestep isolates the spatial error in this ladder.
        cases.append(
            (
                f"radial_N{resolution}",
                "schwarzschild",
                replace(base, radial_resolution=resolution, timestep=0.002),
            )
        )
    for step in TIMESTEP_LADDER:
        cases.append(
            (f"timestep_dt{step:g}", "schwarzschild", replace(base, timestep=step))
        )
    for ell_max in ANGULAR_LADDER:
        cases.append(
            (
                f"angular_lmax{ell_max}",
                "schwarzschild",
                replace(base, angular_ell_max=ell_max),
            )
        )
    cases.append(
        (
            "stencil_order6",
            "schwarzschild",
            replace(base, finite_difference_order=6, timestep=0.002),
        )
    )
    # The late-time multipole rates are the one result whose limiting factor
    # is not obvious from a Schwarzschild ladder, so the SdS case that shows
    # it is repeated at higher radial resolution.
    cases.append(
        (
            "sds_L80_N1536",
            80.0,
            replace(
                base,
                radial_resolution=1536,
                timestep=0.0026,
                end_time=LATE_TIME_CHECK_END_TIME,
            ),
        )
    )
    return cases


def run_convergence_case(
    output_dir: Path,
    name: str,
    *,
    force: bool = False,
    backend: str = "finite-difference",
) -> Path:
    """Run one entry of the refinement ladders."""

    lookup = {entry[0]: entry[1:] for entry in convergence_cases()}
    if name not in lookup:
        raise ValueError(f"Unknown convergence case {name!r}.")
    if backend == "dedalus" and name == "stencil_order6":
        raise ValueError("stencil_order6 is specific to the finite-difference backend.")
    case, numerical = lookup[name]
    return _execute(
        _backend_directory(output_dir, backend)
        / "convergence"
        / "raw"
        / f"{name}.npz",
        case=case,
        source=BROAD_SOURCE,
        numerical=numerical,
        force=force,
        backend=backend,
    )


def all_case_names(
    output_dir: Path, *, backend: str = "finite-difference"
) -> list[str]:
    """Return every runnable case name for the command-line driver."""

    names = ["schwarzschild"] + [case_label(value) for value in COSMOLOGICAL_LENGTHS]
    names += ["narrow:schwarzschild"]
    convergence = [entry[0] for entry in convergence_cases()]
    if backend == "dedalus":
        convergence.remove("stencil_order6")
    names += [f"convergence:{name}" for name in convergence]
    return names


def run_named_case(
    output_dir: Path,
    name: str,
    *,
    force: bool = False,
    backend: str = "finite-difference",
) -> Path:
    """Dispatch one case name from :func:`all_case_names`."""

    if name.startswith("convergence:"):
        return run_convergence_case(
            output_dir,
            name.split(":", 1)[1],
            force=force,
            backend=backend,
        )
    if name.startswith("narrow:"):
        target = name.split(":", 1)[1]
        case = "schwarzschild" if target == "schwarzschild" else 80.0
        return run_narrow_case(output_dir, case, force=force, backend=backend)
    if name == "schwarzschild":
        return run_production_case(
            output_dir, "schwarzschild", force=force, backend=backend
        )
    for length in COSMOLOGICAL_LENGTHS:
        if name == case_label(length):
            return run_production_case(
                output_dir, length, force=force, backend=backend
            )
    raise ValueError(f"Unknown case {name!r}.")


def load_case(output_dir: Path, case: str | float) -> SourcedSimulationResult:
    return load_sourced_result(_archive(output_dir, case_label(case)))


def load_narrow(output_dir: Path, case: str | float) -> SourcedSimulationResult:
    return load_sourced_result(
        Path(output_dir) / "narrow" / "raw" / f"{case_label(case)}.npz"
    )


def load_convergence(output_dir: Path, name: str) -> SourcedSimulationResult:
    return load_sourced_result(
        Path(output_dir) / "convergence" / "raw" / f"{name}.npz"
    )


# --------------------------------------------------------------------------
# Angular reconstruction
# --------------------------------------------------------------------------


def harmonic_matrix(
    result: SourcedSimulationResult,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Return ``Y^R_{lm}`` of every stored mode on the requested directions."""

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta, phi = np.broadcast_arrays(theta, phi)
    return np.stack(
        [
            real_spherical_harmonic(int(ell), int(m), theta, phi)
            for ell, m in zip(result.mode_ell, result.mode_m)
        ]
    )


def equatorial_waveform(
    result: SourcedSimulationResult,
    phi: np.ndarray,
    observer: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``U`` and ``u(U, varphi)`` in the equatorial plane.

    The default observer is the outer boundary: future null infinity on
    Schwarzschild and the cosmological horizon on Schwarzschild--de Sitter.
    """

    index = result.outer_index() if observer is None else int(observer)
    phi = np.asarray(phi, dtype=float)
    basis = harmonic_matrix(result, np.full_like(phi, 0.5 * np.pi), phi)
    # The special-function evaluation above can leave floating-point status
    # flags set, which numpy then reports against the following matmul.  The
    # result is checked for finiteness directly instead.
    with np.errstate(all="ignore"):
        if result.uses_compact_modal_storage:
            angular_weights = result.mode_source_amplitude[:, None] * basis
            ell_weights = np.stack(
                [
                    np.sum(angular_weights[result.mode_ell == ell], axis=0)
                    for ell in result.response_ell
                ]
            )
            field = result.response_signals[:, index, :] @ ell_weights
        else:
            field = result.modal_signals[:, index, :] @ basis
    if not np.isfinite(field).all():
        raise FloatingPointError("The reconstructed waveform is not finite.")
    return result.retarded_time, field


def direction_waveform(
    result: SourcedSimulationResult,
    phi: float,
    observer: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the waveform along one equatorial direction."""

    times, field = equatorial_waveform(result, np.asarray([phi]), observer)
    return times, field[:, 0]


def equatorial_snapshot(
    result: SourcedSimulationResult, phi: np.ndarray
) -> np.ndarray:
    r"""Return ``Phi = u/r`` in the equatorial plane at every snapshot time.

    The returned array has shape ``(snapshots, radii, angles)``.  The outer
    grid point is future null infinity on Schwarzschild, where ``Phi``
    vanishes identically; the reduced field ``u`` is the meaningful variable
    there and is plotted separately.
    """

    phi = np.asarray(phi, dtype=float)
    basis = harmonic_matrix(result, np.full_like(phi, 0.5 * np.pi), phi)
    if result.uses_compact_modal_storage:
        angular_weights = result.mode_source_amplitude[:, None] * basis
        ell_weights = np.stack(
            [
                np.sum(angular_weights[result.mode_ell == ell], axis=0)
                for ell in result.response_ell
            ]
        )
        reduced = np.einsum("tlr,lp->trp", result.response_snapshots, ell_weights)
    else:
        reduced = np.einsum("tmr,mp->trp", result.modal_snapshots, basis)
    radius = result.snapshot_areal_radius
    scale = np.where(np.isfinite(radius) & (radius > 0.0), radius, np.inf)
    return reduced / scale[None, :, None]


def modal_energy_spectrum(
    result: SourcedSimulationResult,
    window: tuple[float, float],
    observer: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``l`` and the outgoing power ``sum_m \int u_{lm}^2 dU``."""

    index = result.outer_index() if observer is None else int(observer)
    times = result.retarded_time
    inside = (times >= window[0]) & (times <= window[1])
    signals = result.expanded_modal_signals()[inside, index, :]
    squared = signals**2
    # Trapezoidal rule written out; ``np.trapz`` has been renamed once and
    # deprecated once across the NumPy versions this project has to run on.
    power = np.sum(
        0.5 * (squared[1:] + squared[:-1]) * np.diff(times[inside])[:, None],
        axis=0,
    )
    ells = np.unique(result.mode_ell)
    totals = np.asarray(
        [float(np.sum(power[result.mode_ell == ell])) for ell in ells]
    )
    return ells, totals


# --------------------------------------------------------------------------
# Caustic echo measurement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EchoPulse:
    """One measured pulse of the caustic sequence."""

    index: int
    phi: float
    time: float
    amplitude: float
    envelope: float
    windings: int

    def as_dict(self) -> dict:
        return {
            "pulse": self.index,
            "phi_over_pi": self.phi / np.pi,
            "U_over_M": self.time,
            "signed_amplitude": self.amplitude,
            "envelope_amplitude": self.envelope,
            "half_orbit_windings": self.windings,
        }


def _refined_extremum(
    times: np.ndarray, values: np.ndarray, index: int
) -> tuple[float, float]:
    """Return the parabolic-vertex refinement of a discrete extremum."""

    if index <= 0 or index >= times.size - 1:
        return float(times[index]), float(values[index])
    left, middle, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * middle + right
    if denominator == 0.0:
        return float(times[index]), float(values[index])
    shift = 0.5 * (left - right) / denominator
    step = float(times[index + 1] - times[index])
    return (
        float(times[index] + shift * step),
        float(middle - 0.25 * (left - right) * shift),
    )


def find_caustic_pulses(
    result: SourcedSimulationResult,
    *,
    start: float = 5.0,
    end: float | None = None,
    relative_prominence: float = 0.01,
    separation: float = 9.0,
) -> list[EchoPulse]:
    r"""Locate the caustic sequence on the source axis and its antipode.

    Rays from an equatorial emitter refocus on the axis through the source
    every half orbit, alternating between the source direction
    ``varphi = 0`` and the antipode ``varphi = pi``.  Peaks are therefore
    collected from both directions and merged in retarded time; the winding
    count assigned to each pulse is its position in that merged sequence.

    Pulses are located on the analytic-signal envelope ``|u + i H[u]|``
    rather than on ``|u|``.  The envelope is insensitive to the carrier
    phase, so an echo that is phase shifted relative to the previous one is
    still timed at its centre instead of at whichever oscillation happens to
    be largest.  The reported amplitude is the signed field at that time.

    On Schwarzschild--de Sitter the field settles to a nonzero constant, so
    the late-time offset is removed before the envelope is formed; otherwise
    the frozen field would register as an unbroken plateau of spurious
    maxima.  On Schwarzschild the subtracted constant is the decayed tail and
    is negligible.  Peaks are then required to have a prominence of at least
    ``relative_prominence`` of the largest envelope, which rejects the
    ringdown shoulders that ride on each pulse.
    """

    times = result.retarded_time
    limit = float(times[-1]) if end is None else float(end)
    pulses: list[EchoPulse] = []
    scale = 0.0
    envelopes: dict[float, np.ndarray] = {}
    traces: dict[float, np.ndarray] = {}
    late = times > times[-1] - 0.05 * (times[-1] - times[0])
    for phi in (0.0, np.pi):
        _, trace = direction_waveform(result, phi)
        wave = trace - float(np.median(trace[late]))
        traces[phi] = wave
        envelopes[phi] = np.abs(hilbert(wave))
        scale = max(scale, float(np.max(envelopes[phi])))
    minimum_gap = max(1, int(round(separation / float(np.median(np.diff(times))))))
    for phi in (0.0, np.pi):
        trace = traces[phi]
        magnitude = envelopes[phi]
        # Prominence must be measured on the whole record: masking the
        # envelope to zero outside the window would manufacture prominence at
        # the window edges.  The window is applied to the result instead.
        found, _ = find_peaks(
            magnitude,
            distance=minimum_gap,
            prominence=relative_prominence * scale,
        )
        inside = (times[found] >= start) & (times[found] <= limit)
        for position in found[inside]:
            peak_time, _ = _refined_extremum(times, magnitude, position)
            pulses.append(
                EchoPulse(
                    index=0,
                    phi=phi,
                    time=peak_time,
                    amplitude=float(np.interp(peak_time, times, trace)),
                    envelope=float(magnitude[position]),
                    windings=0,
                )
            )
    pulses.sort(key=lambda pulse: pulse.time)
    # The caustic sequence alternates between the two directions and cannot
    # produce two crossings within a fraction of a half orbit.  Imposing both
    # conditions on the merged list makes the crossing count insensitive to
    # the prominence threshold: a candidate that violates either condition
    # replaces the previous entry only if it is the stronger of the two.
    minimum_merged_gap = 0.4 * PHOTON_SPHERE_PERIOD / 2.0
    kept: list[EchoPulse] = []
    for pulse in pulses:
        if kept and (
            pulse.phi == kept[-1].phi
            or pulse.time - kept[-1].time < minimum_merged_gap
        ):
            if pulse.envelope > kept[-1].envelope:
                kept[-1] = pulse
            continue
        kept.append(pulse)
    return [
        replace(pulse, index=number, windings=number)
        for number, pulse in enumerate(kept)
    ]


def echo_phase_shifts(
    result: SourcedSimulationResult,
    pulses: list[EchoPulse],
    *,
    half_width: float = 9.0,
) -> list[dict]:
    r"""Measure the phase relation between consecutive caustic pulses.

    Each pulse is windowed, converted to its analytic signal, and correlated
    against the next pulse.  The modulus of the correlation at its maximum
    fixes the relative delay and amplitude, and its argument is the phase
    that the waveform accumulates between the two caustic crossings.  A
    value near ``-pi/2`` per crossing is the geometric-optics Gouy shift; no
    value is assumed here, the phase is read off the data.
    """

    times = result.retarded_time
    step = float(np.median(np.diff(times)))
    span = int(round(half_width / step))
    analytic: dict[float, np.ndarray] = {}
    late = times > times[-1] - 0.05 * (times[-1] - times[0])
    for phi in (0.0, np.pi):
        _, trace = direction_waveform(result, phi)
        analytic[phi] = hilbert(trace - float(np.median(trace[late])))
    rows: list[dict] = []
    for first, second in zip(pulses, pulses[1:]):
        centres = [
            int(np.argmin(np.abs(times - pulse.time))) for pulse in (first, second)
        ]
        if min(centres) - span < 0 or max(centres) + span >= times.size:
            continue
        windows = [
            analytic[pulse.phi][centre - span : centre + span + 1]
            for pulse, centre in zip((first, second), centres)
        ]
        reference, target = windows
        # correlate(target, reference) = sum_n target[n+k] conj(reference[n]),
        # so its argument at the best lag is the phase of the later pulse
        # relative to the earlier one, without any periodic wrap-around.
        correlations = correlate(target, reference, mode="full")
        lags = np.arange(1 - reference.size, reference.size)
        allowed = np.abs(lags) <= span // 2
        best = int(np.flatnonzero(allowed)[np.argmax(np.abs(correlations[allowed]))])
        norm = float(np.linalg.norm(reference) * np.linalg.norm(target))
        rows.append(
            {
                "pulse_pair": f"{first.index}->{second.index}",
                "U_first_over_M": first.time,
                "U_second_over_M": second.time,
                "delay_over_M": second.time - first.time,
                "delay_from_correlation_over_M": (
                    second.time - first.time - float(lags[best]) * step
                ),
                "phase_over_half_pi": float(
                    np.angle(correlations[best]) / (0.5 * np.pi)
                ),
                "coherence": float(np.abs(correlations[best]) / norm)
                if norm > 0.0
                else np.nan,
                "amplitude_ratio": abs(second.amplitude / first.amplitude),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Flat-limit comparison
# --------------------------------------------------------------------------


def _resampled(
    times: np.ndarray, signal: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    return np.interp(grid, times, signal)


def flat_limit_norms(
    reference: SourcedSimulationResult,
    candidate: SourcedSimulationResult,
    phi_values: tuple[float, ...] = (0.0, np.pi),
    window: tuple[float, float] = VALIDATION_WINDOW,
    samples: int = 4001,
) -> dict:
    r"""Compare one SdS waveform with the Schwarzschild waveform on ``U``.

    Both signals are reconstructed in the equatorial plane, resampled onto a
    common retarded-time grid, and compared in relative ``L^2`` and ``L^inf``
    norms.  The window is chosen to contain the direct pulse and the first
    caustic echoes, well before the cosmological decay sets in for any
    length in the sequence.
    """

    grid = np.linspace(window[0], window[1], samples)
    row: dict = {}
    squared_error = 0.0
    squared_signal = 0.0
    worst = 0.0
    scale = 0.0
    for phi in phi_values:
        reference_times, reference_trace = direction_waveform(reference, phi)
        candidate_times, candidate_trace = direction_waveform(candidate, phi)
        exact = _resampled(reference_times, reference_trace, grid)
        approximate = _resampled(candidate_times, candidate_trace, grid)
        difference = approximate - exact
        squared_error += float(np.sum(difference**2))
        squared_signal += float(np.sum(exact**2))
        worst = max(worst, float(np.max(np.abs(difference))))
        scale = max(scale, float(np.max(np.abs(exact))))
        row[f"relative_l2_phi{phi / np.pi:g}pi"] = float(
            np.linalg.norm(difference) / np.linalg.norm(exact)
        )
    row["relative_l2"] = float(np.sqrt(squared_error / squared_signal))
    row["absolute_linf"] = worst
    row["relative_linf"] = worst / scale
    return row


def pulse_timing_shifts(
    reference: list[EchoPulse], candidate: list[EchoPulse]
) -> list[dict]:
    """Match caustic pulses between two runs and return their time shifts."""

    rows: list[dict] = []
    lookup = {other.index: other for other in reference}
    for pulse in candidate:
        closest = lookup.get(pulse.index)
        # Matching by crossing number rather than by proximity keeps the
        # pairing correct once the cosmological shift grows beyond half a
        # crossing interval.  A mismatched direction or an implausible shift
        # means the two sequences do not correspond and the pulse is skipped.
        if closest is None or closest.phi != pulse.phi:
            continue
        if abs(closest.time - pulse.time) > 12.0:
            continue
        rows.append(
            {
                "pulse": pulse.index,
                "phi_over_pi": pulse.phi / np.pi,
                "U_schwarzschild_over_M": closest.time,
                "U_sds_over_M": pulse.time,
                "timing_shift_over_M": pulse.time - closest.time,
                "envelope_ratio": pulse.envelope / closest.envelope,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Command-line driver
# --------------------------------------------------------------------------


def main() -> None:
    """Run one or more named cases of the sourced suite."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        help="case names; omit to list every available case",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/green_function")
    )
    parser.add_argument(
        "--backend", choices=SOURCE_BACKENDS, default="finite-difference"
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not arguments.cases:
        for name in all_case_names(arguments.output_dir, backend=arguments.backend):
            print(name)
        return
    for name in arguments.cases:
        path = run_named_case(
            arguments.output_dir,
            name,
            force=arguments.force,
            backend=arguments.backend,
        )
        LOGGER.info("wrote %s", path)


if __name__ == "__main__":
    main()
