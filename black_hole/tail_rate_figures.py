"""Create restricted-range, multi-radius Schwarzschild tail-rate figures.

This is a lightweight post-processing entry point for the saved tail archives.
It intentionally does not import Dedalus, so figures can be regenerated from
the committed ``.npz`` products in a standard NumPy/SciPy/Matplotlib
environment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .tail_analysis import local_decay_rates, numerical_amplitude_floor


DEFAULT_RADII = (20.0, 50.0, 100.0, 200.0)
REVIEWED_RADII = {
    0: DEFAULT_RADII,
    1: DEFAULT_RADII,
    # The ell=2 signal at 20M crosses its spatial-truncation floor in the
    # relevant interval; it must not be presented as a resolved local rate.
    2: (50.0, 100.0, 200.0),
}
DISPLAY_WINDOWS = {
    0: (150.0, 720.0, (-3.05, -1.95)),
    1: (220.0, 320.0, (-5.1, -2.9)),
    2: (300.0, 440.0, (-7.1, -3.9)),
}


def chebyshev_gauss_grid(size: int) -> np.ndarray:
    """Return the ascending Chebyshev--Gauss grid mapped to ``[0, 1]``."""

    if size < 2:
        raise ValueError("At least two Chebyshev points are required.")
    angles = np.pi * (np.arange(size, dtype=float) + 0.5) / size
    return 0.5 * (1.0 - np.cos(angles))


def interpolate_chebyshev_snapshots(
    snapshots: np.ndarray, target_rho: float
) -> np.ndarray:
    """Evaluate rows of Chebyshev--Gauss data at one compact radius.

    The saved snapshots are recorded at the dealiased scale, which can be
    larger than the base grid stored in the archive. Reconstructing the grid
    from the snapshot width avoids silently pairing data with the wrong grid.
    """

    values = np.asarray(snapshots, dtype=float)
    if values.ndim != 2:
        raise ValueError("Snapshots must be a two-dimensional time-by-space array.")
    if not 0.0 <= target_rho <= 1.0:
        raise ValueError("The target compact radius must lie in [0, 1].")

    size = values.shape[1]
    angles = np.pi * (np.arange(size, dtype=float) + 0.5) / size
    grid = 0.5 * (1.0 - np.cos(angles))
    nearest = int(np.argmin(np.abs(grid - target_rho)))
    if abs(grid[nearest] - target_rho) <= 8.0 * np.finfo(float).eps:
        return values[:, nearest].copy()

    # First-kind Chebyshev roots have barycentric weights proportional to
    # (-1)^j sin(theta_j). The affine map to [0,1] changes only a common
    # factor, which cancels in the barycentric quotient.
    weights = (-1.0) ** np.arange(size) * np.sin(angles)
    quotient_weights = weights / (target_rho - grid)
    return values @ quotient_weights / np.sum(quotient_weights)


def _archive_metadata(archive: np.lib.npyio.NpzFile) -> dict:
    raw = archive["metadata"]
    serialized = raw.item() if np.ndim(raw) == 0 else str(raw)
    return json.loads(str(serialized))


def _observer_series(
    archive: np.lib.npyio.NpzFile,
    radius: float,
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(archive["snapshot_times"], dtype=float)
    rho = 1.0 - 2.0 * mass / radius
    signal = interpolate_chebyshev_snapshots(archive["u_snapshots"], rho)
    return times, signal


def _scri_series(
    archive: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    snapshot_times = np.asarray(archive["snapshot_times"], dtype=float)
    signal = np.interp(
        snapshot_times,
        np.asarray(archive["signal_times"], dtype=float),
        np.asarray(archive["signals"], dtype=float)[:, -1],
    )
    return snapshot_times, signal


def _smoothing_samples(times: np.ndarray, physical_width: float = 100.0) -> int:
    step = float(np.median(np.diff(times)))
    samples = max(7, int(round(physical_width / step)))
    if samples % 2 == 0:
        samples += 1
    return samples


def create_schwarzschild_distance_rate_figure(
    archive_path: Path,
    output_path: Path,
    *,
    ell: int,
    radii: tuple[float, ...] | None = None,
) -> list[dict[str, float | str]]:
    """Plot the signed local decay rate for several fixed radii and scri+."""

    if ell not in DISPLAY_WINDOWS:
        raise ValueError(f"No reviewed display window is configured for ell={ell}.")
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if radii is None:
        radii = REVIEWED_RADII[ell]

    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = _archive_metadata(archive)
        mass = float(metadata["model"]["mass"])
        retarded = metadata.get("retarded_time_offset", {})
        time_offset = float(retarded.get("q", 0.0))

        series: list[tuple[str, np.ndarray, np.ndarray]] = []
        for radius in radii:
            times, signal = _observer_series(archive, radius, mass)
            series.append((rf"$r/M={radius / mass:g}$", times, signal))
        times, signal = _scri_series(archive)
        series.append((r"$\mathscr{I}^{+}$", times, signal))

    start, end, y_limits = DISPLAY_WINDOWS[ell]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(series) - 1))
    rows: list[dict[str, float | str]] = []
    fig, axis = plt.subplots(figsize=(9.8, 5.2))

    for index, (label, coordinate_times, signal) in enumerate(series):
        times = coordinate_times - time_offset
        scaled_times = times / mass
        window = _smoothing_samples(times)
        _, power, _ = local_decay_rates(times, signal, window=window)
        signed_rate = -power
        resolved = np.abs(signal) > 10.0 * numerical_amplitude_floor(signal)
        display = (
            (scaled_times >= start)
            & (scaled_times <= end)
            & resolved
            & np.isfinite(signed_rate)
        )
        color = "black" if index == len(series) - 1 else colors[index]
        linewidth = 2.0 if index == len(series) - 1 else 1.45
        axis.plot(
            scaled_times[display],
            signed_rate[display],
            color=color,
            linewidth=linewidth,
            label=label,
        )
        observer = "scri+" if index == len(series) - 1 else label.replace("$", "")
        rows.extend(
            {
                "ell": ell,
                "observer": observer,
                "U_over_M": float(time),
                "n_eff": float(rate),
                "positive_power": float(-rate),
            }
            for time, rate in zip(scaled_times[display], signed_rate[display])
        )

    scri_target = -(ell + 2)
    finite_target = -(2 * ell + 3)
    axis.axhline(
        scri_target,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label=rf"$\mathscr{{I}}^+$ target: $n={scri_target}$",
    )
    axis.axhline(
        finite_target,
        color="0.35",
        linewidth=1.0,
        linestyle=":",
        label=rf"finite-$r$ target: $n={finite_target}$",
    )
    axis.set(
        xlim=(start, end),
        ylim=y_limits,
        xlabel=r"shifted hyperboloidal time $U/M$",
        ylabel=r"$n_{\rm eff}=d\ln|u|/d\ln U$",
        title=rf"Schwarzschild local tail decay rate, $\ell={ell}$",
    )
    axis.grid(alpha=0.25)
    axis.legend(
        fontsize=8.5,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return rows


def create_distance_rate_report(
    input_root: Path,
    output_dir: Path,
    *,
    ells: tuple[int, ...] = (0, 1, 2),
    radii: tuple[float, ...] | None = None,
) -> list[Path]:
    """Create one restricted-range multi-radius figure and CSV per mode."""

    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ell in ells:
        stem = f"schwarzschild_local_decay_rate_ell{ell}"
        figure_path = output_dir / f"{stem}.png"
        rows = create_schwarzschild_distance_rate_figure(
            input_root / f"ell{ell}" / "schwarzschild.npz",
            figure_path,
            ell=ell,
            radii=REVIEWED_RADII[ell] if radii is None else radii,
        )
        csv_path = output_dir / f"{stem}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "ell",
                    "observer",
                    "U_over_M",
                    "n_eff",
                    "positive_power",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        outputs.extend((figure_path, csv_path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot restricted-range Schwarzschild tail rates at fixed radii."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/sds_scalar/tails/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sds_scalar/tails/distance_rates"),
    )
    parser.add_argument("--ells", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=None,
        help="finite areal radii in units consistent with the archive",
    )
    args = parser.parse_args()
    paths = create_distance_rate_report(
        args.input_root,
        args.output_dir,
        ells=tuple(args.ells),
        radii=tuple(args.radii) if args.radii is not None else None,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
