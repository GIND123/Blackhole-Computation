"""Create the manuscript figure comparing bridge flat limits and conditioning."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from black_hole.foliation_diagnostics import evaluate_foliation_table


OUTPUT = HERE / "figs/foliation_flat_limit.pdf"
LENGTHS = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0)

LABELS = {
    "minimum": "minimum height",
    "minimal": "minimal gauge",
    "linear": "linear boost",
    "modified_linear": "modified linear",
    "mavrogiannis": "Mavrogiannis",
    "slow_roll": "slow-roll",
}
COLORS = {
    "minimum": "#0072B2",
    "minimal": "#E69F00",
    "linear": "#009E73",
    "modified_linear": "#D55E00",
    "mavrogiannis": "#CC79A7",
    "slow_roll": "#6F4E37",
}
STYLES = {
    "minimum": "-",
    "minimal": (0, (7, 2)),
    "linear": (0, (3, 1)),
    "modified_linear": (0, (5, 1, 1, 1)),
    "mavrogiannis": "-.",
    "slow_roll": ":",
}
MARKERS = {
    "minimum": "o",
    "minimal": "s",
    "linear": "^",
    "modified_linear": "D",
    "mavrogiannis": "v",
    "slow_roll": "P",
}


def flat_limit_boost(name: str, radius: np.ndarray) -> np.ndarray:
    """Return the fixed-r Schwarzschild limit of each SdS bridge boost."""

    if name == "minimum":
        return 3.0 / radius - 0.5
    if name in {"minimal", "modified_linear"}:
        return -1.0 + 8.0 / radius**2
    if name == "linear":
        return np.ones_like(radius)
    if name == "mavrogiannis":
        return -(1.0 - 3.0 / radius) * np.sqrt(1.0 + 6.0 / radius)
    if name == "slow_roll":
        return 4.0 / radius**2
    raise ValueError(name)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
axes = axes.ravel()

rho_zero = np.linspace(0.0, 0.995, 1000)
radius = 2.0 / (1.0 - rho_zero)
for name in LABELS:
    axes[0].plot(
        rho_zero,
        flat_limit_boost(name, radius),
        color=COLORS[name],
        linestyle=STYLES[name],
        label=LABELS[name],
    )
axes[0].axhline(-1.0, color="0.35", linewidth=0.9, linestyle=(0, (2, 2)))
axes[0].text(
    0.97,
    -0.94,
    r"$\mathcal{I}^+$",
    ha="right",
    va="bottom",
    color="0.3",
    fontsize=8,
)
axes[0].set(
    xlabel=r"$\rho_0=1-2M/r$",
    ylabel=r"fixed-$r$ limit $B_0$",
    xlim=(0.0, 1.0),
    ylim=(-1.08, 1.08),
)

diagnostics = evaluate_foliation_table(LENGTHS, tuple(LABELS))
by_bridge = {
    name: [row for row in diagnostics if row.bridge == name] for name in LABELS
}
for name in LABELS:
    axes[2].loglog(
        LENGTHS,
        [row.maximum_characteristic_speed for row in by_bridge[name]],
        color=COLORS[name],
        linestyle=STYLES[name],
        marker=MARKERS[name],
        markersize=3.6,
        label=LABELS[name],
    )
axes[2].set(
    xlabel=r"$L/M$",
    ylabel=r"$M\max_{\rho}|c_\pm|$",
    xlim=(9.0, 360.0),
)

for name in LABELS:
    axes[3].loglog(
        LENGTHS,
        [row.minimum_propagation_coefficient for row in by_bridge[name]],
        color=COLORS[name],
        linestyle=STYLES[name],
        marker=MARKERS[name],
        markersize=3.6,
        label=LABELS[name],
    )
axes[3].set(
    xlabel=r"$L/M$",
    ylabel=r"$M\min_{\rho} A$",
    xlim=(9.0, 360.0),
)

for name in LABELS:
    axes[1].loglog(
        LENGTHS,
        [row.retarded_time_offset for row in by_bridge[name]],
        color=COLORS[name],
        linestyle=STYLES[name],
        marker=MARKERS[name],
        markersize=3.6,
        label=LABELS[name],
    )
axes[1].axhline(
    4.0 * np.log(2.0),
    color="0.35",
    linewidth=0.9,
    linestyle=(0, (2, 2)),
)
axes[1].text(
    10.2,
    4.0 * np.log(2.0) * 1.12,
    r"$q_0=4M\ln 2$",
    color="0.3",
    fontsize=8,
)
axes[1].set(
    xlabel=r"$L/M$",
    ylabel=r"$q_B/M$",
    xlim=(9.0, 360.0),
)

for axis in axes:
    axis.grid(alpha=0.2, which="both")
    axis.text(
        0.03,
        0.95,
        f"({chr(ord('a') + list(axes).index(axis))})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    ncols=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
)
fig.subplots_adjust(
    left=0.09,
    right=0.99,
    bottom=0.17,
    top=0.985,
    wspace=0.30,
    hspace=0.34,
)
fig.savefig(OUTPUT)
