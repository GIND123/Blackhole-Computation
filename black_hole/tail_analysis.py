"""Numerically robust diagnostics for Schwarzschild and SdS scalar tails."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter


def json_safe(value):
    """Return a strict-JSON representation, mapping non-finite values to null."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


@dataclass(frozen=True)
class DecayFit:
    """A linear fit in logarithmic amplitude coordinates."""

    kind: str
    start: float
    end: float
    rate: float
    slope: float
    intercept: float
    r_squared: float
    points: int
    amplitude_floor: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def numerical_amplitude_floor(signal: np.ndarray, multiplier: float = 1000.0) -> float:
    """Return a scale-aware double-precision amplitude floor."""

    signal = np.asarray(signal, dtype=float)
    scale = float(np.nanmax(np.abs(signal))) if signal.size else 0.0
    return max(np.finfo(float).tiny, multiplier * np.finfo(float).eps * scale)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return float(slope), float(intercept), float(r_squared)


def fit_power_law(
    times: np.ndarray,
    signal: np.ndarray,
    start: float,
    end: float,
    *,
    minimum_points: int = 40,
) -> DecayFit:
    r"""Fit ``|u|=C U^{-p}`` and return the positive decay exponent ``p``."""

    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    floor = numerical_amplitude_floor(signal)
    valid = (
        (times >= start)
        & (times <= end)
        & (times > 0.0)
        & np.isfinite(signal)
        & (np.abs(signal) > floor)
    )
    selected_times = times[valid]
    amplitudes = np.abs(signal[valid])
    if selected_times.size < minimum_points:
        raise ValueError("Power-law fit interval has too few resolved samples.")
    slope, intercept, r_squared = _linear_fit(
        np.log(selected_times), np.log(amplitudes)
    )
    return DecayFit(
        kind="power",
        start=float(selected_times[0]),
        end=float(selected_times[-1]),
        rate=float(-slope),
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        points=int(selected_times.size),
        amplitude_floor=floor,
    )


def fit_exponential(
    times: np.ndarray,
    signal: np.ndarray,
    start: float,
    end: float,
    *,
    offset: float = 0.0,
    minimum_points: int = 40,
) -> DecayFit:
    r"""Fit ``|u-offset|=C exp(-gamma U)`` and return ``gamma``."""

    times = np.asarray(times, dtype=float)
    residual_signal = np.asarray(signal, dtype=float) - float(offset)
    floor = numerical_amplitude_floor(residual_signal)
    valid = (
        (times >= start)
        & (times <= end)
        & np.isfinite(residual_signal)
        & (np.abs(residual_signal) > floor)
    )
    selected_times = times[valid]
    amplitudes = np.abs(residual_signal[valid])
    if selected_times.size < minimum_points:
        raise ValueError("Exponential fit interval has too few resolved samples.")
    slope, intercept, r_squared = _linear_fit(selected_times, np.log(amplitudes))
    return DecayFit(
        kind="exponential",
        start=float(selected_times[0]),
        end=float(selected_times[-1]),
        rate=float(-slope),
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        points=int(selected_times.size),
        amplitude_floor=floor,
    )


def select_stable_exponential_fit(
    times: np.ndarray,
    signal: np.ndarray,
    kappa: float,
    *,
    minimum_scaled_time: float = 1.5,
    maximum_scaled_time: float = 7.5,
) -> DecayFit:
    """Select a stable late exponential window in units of ``kappa*time``."""

    if kappa <= 0.0:
        raise ValueError("Surface gravity must be positive.")
    scaled_end = min(maximum_scaled_time, kappa * float(times[-1]))
    if scaled_end - minimum_scaled_time < 1.2:
        raise ValueError("Evolution is too short for a stable exponential fit.")
    starts = np.linspace(minimum_scaled_time, scaled_end - 1.0, 14)
    candidates: list[tuple[float, DecayFit]] = []
    for scaled_start in starts:
        ends = np.linspace(scaled_start + 1.0, scaled_end, 12)
        for candidate_end in ends:
            start = scaled_start / kappa
            end = candidate_end / kappa
            split = 0.5 * (start + end)
            try:
                whole = fit_exponential(times, signal, start, end)
                first = fit_exponential(
                    times, signal, start, split, minimum_points=20
                )
                second = fit_exponential(
                    times, signal, split, end, minimum_points=20
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if whole.rate <= 0.0:
                continue
            stability = abs(first.rate - second.rate) / whole.rate
            score = (
                stability
                + 8.0 * max(0.0, 1.0 - whole.r_squared)
                + 0.02 / (candidate_end - scaled_start)
            )
            candidates.append((score, whole))
    if not candidates:
        raise ValueError("No positive resolved exponential interval was found.")
    score, fit = min(candidates, key=lambda item: item[0])
    if fit.r_squared < 0.98 or score > 0.35:
        raise ValueError(
            "No exponential interval passed the stability and fit-quality criteria."
        )
    return fit


def select_stable_power_fit(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    minimum_time: float = 40.0,
    maximum_time: float | None = None,
) -> DecayFit:
    """Select a power-law window by fit stability, without using a target exponent."""

    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if maximum_time is None:
        maximum_time = float(times[-1])
    positive = times[(times >= minimum_time) & (times <= maximum_time)]
    if positive.size < 160:
        raise ValueError("Not enough samples to select a stable power-law window.")

    lower = max(minimum_time, float(positive[0]))
    upper = float(positive[-1])
    starts = np.geomspace(lower, max(lower * 1.05, upper / 2.2), 16)
    ends = np.geomspace(max(lower * 2.0, upper / 2.5), upper, 18)
    candidates: list[tuple[float, DecayFit]] = []
    for start in starts:
        for end in ends:
            if end / start < 1.65:
                continue
            try:
                whole = fit_power_law(times, signal, float(start), float(end))
                split = np.sqrt(start * end)
                first = fit_power_law(times, signal, float(start), float(split), minimum_points=20)
                second = fit_power_law(times, signal, float(split), float(end), minimum_points=20)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if whole.rate <= 0.0:
                continue
            stability = abs(first.rate - second.rate) / max(whole.rate, 1e-12)
            log_span = np.log(end / start)
            score = stability + 5.0 * max(0.0, 1.0 - whole.r_squared) + 0.02 / log_span
            candidates.append((score, whole))
    if not candidates:
        raise ValueError("No resolved stable power-law interval was found.")
    return min(candidates, key=lambda item: item[0])[1]


def asymptotic_constant(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    start_fraction: float = 0.8,
) -> dict[str, float | int]:
    """Estimate a late constant and its residual drift."""

    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal, dtype=float)
    start = float(times[0] + start_fraction * (times[-1] - times[0]))
    mask = (times >= start) & np.isfinite(signal)
    values = signal[mask]
    if values.size < 20:
        raise ValueError("Too few samples to estimate the asymptotic constant.")
    split = values.size // 2
    first = float(np.mean(values[:split]))
    second = float(np.mean(values[split:]))
    value = float(np.mean(values))
    standard_deviation = float(np.std(values))
    return {
        "start": start,
        "end": float(times[-1]),
        "value": value,
        "standard_deviation": standard_deviation,
        "half_interval_drift": abs(second - first),
        "relative_half_interval_drift": abs(second - first)
        / max(abs(value), numerical_amplitude_floor(signal)),
        "points": int(values.size),
    }


def local_decay_rates(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    window: int = 301,
    offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local positive power and exponential decay rates."""

    times = np.asarray(times, dtype=float)
    residual = np.asarray(signal, dtype=float) - float(offset)
    floor = numerical_amplitude_floor(residual)
    if times.size < 7:
        return times.copy(), np.full_like(times, np.nan), np.full_like(times, np.nan)
    window = min(window, times.size if times.size % 2 else times.size - 1)
    window = max(7, window)
    if window % 2 == 0:
        window -= 1
    log_amplitude = np.log(np.maximum(np.abs(residual), floor))
    smooth = savgol_filter(log_amplitude, window, 3, mode="interp")
    exponential = -np.gradient(smooth, times)
    power = exponential * times
    resolved = np.abs(residual) > 10.0 * floor
    positive_time = times > 0.0
    power[~(resolved & positive_time)] = np.nan
    exponential[~resolved] = np.nan
    return times.copy(), power, exponential


def aligned_signals(
    reference_times: np.ndarray,
    reference_signal: np.ndarray,
    candidate_times: np.ndarray,
    candidate_signal: np.ndarray,
    *,
    sample_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two signals onto their common physical-time interval."""

    start = max(float(reference_times[0]), float(candidate_times[0]))
    end = min(float(reference_times[-1]), float(candidate_times[-1]))
    if end <= start:
        raise ValueError("Signals do not overlap in physical time.")
    if sample_dt is None:
        reference_step = float(np.median(np.diff(reference_times)))
        candidate_step = float(np.median(np.diff(candidate_times)))
        sample_dt = max(reference_step, candidate_step)
    count = int(np.floor((end - start) / sample_dt)) + 1
    if count < 3:
        raise ValueError("Signal overlap contains too few samples.")
    times = start + sample_dt * np.arange(count)
    return (
        times,
        np.interp(times, reference_times, reference_signal),
        np.interp(times, candidate_times, candidate_signal),
    )


def sliding_window_difference(
    times: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    window_width: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the centered sliding relative ``L2`` waveform difference."""

    times = np.asarray(times, dtype=float)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if not (times.shape == reference.shape == candidate.shape):
        raise ValueError("Time and waveform arrays must have identical shapes.")
    if times.size < 3 or window_width <= 0.0:
        raise ValueError("A positive window and at least three samples are required.")
    step = float(np.median(np.diff(times)))
    samples = max(3, int(round(window_width / step)))
    if samples % 2 == 0:
        samples += 1
    if samples >= times.size:
        raise ValueError("Sliding window is longer than the waveform.")
    kernel = np.ones(samples, dtype=float)
    numerator = np.convolve((candidate - reference) ** 2, kernel, mode="same")
    denominator = np.convolve(reference**2, kernel, mode="same")
    floor = numerical_amplitude_floor(reference)
    resolved = denominator > samples * (10.0 * floor) ** 2
    difference = np.full_like(times, np.nan)
    difference[resolved] = np.sqrt(numerator[resolved] / denominator[resolved])
    half = samples // 2
    difference[:half] = np.nan
    difference[-half:] = np.nan
    return times.copy(), difference


def trust_times(
    times: np.ndarray,
    difference: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.01, 0.05, 0.10),
    sustained_width: float = 5.0,
    start_time: float | None = None,
) -> dict[float, float]:
    """Return exits from a previously established trusted interval.

    A threshold is only reported after the difference has first remained
    below it for ``sustained_width`` and then remained above it for the same
    duration.  This prevents the low-amplitude leading edge from being
    mistaken for the loss of an already established agreement interval.
    """

    times = np.asarray(times, dtype=float)
    difference = np.asarray(difference, dtype=float)
    step = float(np.median(np.diff(times)))
    sustained_samples = max(1, int(round(sustained_width / step)))
    answer: dict[float, float] = {}
    for threshold in thresholds:
        finite = np.isfinite(difference)
        if start_time is not None:
            finite &= times >= start_time
        above = finite & (difference >= threshold)
        below = finite & (difference < threshold)
        crossing = float("nan")
        armed = False
        for index in range(0, above.size - sustained_samples + 1):
            stop = index + sustained_samples
            if not armed and np.all(below[index:stop]):
                armed = True
                continue
            if armed and np.all(above[index:stop]):
                crossing = float(times[index])
                break
        answer[threshold] = crossing
    return answer
