r"""Sourced three-dimensional scalar evolution on Schwarzschild and SdS.

This module solves the *inhomogeneous* problem

.. math::

   \Box\Phi = S,\qquad
   \Phi(\tau=0,\cdot)=\partial_\tau\Phi(\tau=0,\cdot)=0,

for the localized emitter of :mod:`black_hole.localized_source`, on both
the Schwarzschild minimal-gauge hyperboloidal foliation and the
Schwarzschild--de Sitter minimal-gauge bridge foliation.

Reduction to modes
------------------

With ``Phi = r^{-1} sum_{lm} u_{lm} Y^R_{lm}`` and
``S = sum_{lm} S_{lm} Y^R_{lm}``, the identity

.. math::

   \Box\Phi=\frac{Y^{\rm R}_{\ell m}}{f\,r}
            \left[-\partial_t^2 u+\partial_{r_*}^2u-V_\ell u\right]

turns the wave equation into the sourced one-dimensional problem

.. math::

   \left(-\partial_t^2+\partial_{r_*}^2-V_\ell\right)u_{\ell m}
        = f\,r\,S_{\ell m}.

Transforming to the bridge coordinates ``tau = t + h_L(r)``, ``rho = g(r)``
and reducing to first order with ``psi = d_rho u`` and
``pi = (1-B^2)p^{-1} d_tau u - B psi`` gives the evolution system used
throughout this project with a single extra term,

.. math::

   \partial_\tau u &= A(B\psi+\pi),\\
   \partial_\tau \psi &= \partial_\rho\left[A(B\psi+\pi)\right],\\
   \partial_\tau \pi &= \partial_\rho\left[A(\psi+B\pi)\right]-P_\ell u
                        -\frac{r}{G}\,S_{\ell m}\!\left(\tau-h_L(r),\,r\right),

because ``f r S_{lm}/p = r S_{lm}/G`` with ``p = f G``.  The source factor
``r/G`` is bounded on the compact support of the emitter, so the added term
is as regular as the rest of the system.  Evaluating ``S`` at
``t = tau - h_L(r)`` is what makes one and the same physical emitter act on
every background of the cosmological-length sequence.

The angular modes decouple on a spherically symmetric background, so the
whole excited catalogue is advanced simultaneously as a single vectorized
state of shape ``(3, modes, radial points)`` with one shared eighth-order
radial derivative.  The discretization is deliberately the independent
finite-difference discretization of :mod:`black_hole.three_d_solver`, not
the Chebyshev--Dedalus discretization of the one-dimensional studies.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .localized_source import (
    LocalizedSourceParameters,
    SourceModeCatalogue,
    build_mode_catalogue,
    radial_profile,
    time_profile,
    verify_angular_expansion,
)
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    areal_radius as schwarzschild_areal_radius,
    minimal_boost as schwarzschild_minimal_boost,
    minimal_height as schwarzschild_minimal_height,
    propagation_coefficient as schwarzschild_propagation_coefficient,
    rescaled_scalar_potential as schwarzschild_potential,
    retarded_time_offset as schwarzschild_offset,
)
from .sds_model import (
    SdSParameters,
    areal_radius,
    bridge_boost,
    compact_radius,
    compactification_derivative,
    minimal_height,
    propagation_coefficient,
    rescaled_scalar_potential,
    retarded_time_offset,
    sds_horizons,
)
from .three_d_solver import UniformFiniteDifference
from .reproducibility import reproducibility_metadata

LOGGER = logging.getLogger(__name__)

REFERENCE_RADIUS = 4.0
"""Areal radius at which every height function and clock is normalized."""

__all__ = [
    "SourcedNumericalParameters",
    "SourcedSimulationResult",
    "load_sourced_result",
    "run_sourced_simulation",
]


@dataclass(frozen=True)
class SourcedNumericalParameters:
    """Numerical controls for one sourced three-dimensional evolution."""

    radial_resolution: int = 512
    angular_ell_max: int = 16
    timestep: float = 0.008
    end_time: float = 400.0
    signal_dt: float = 0.05
    diagnostic_dt: float = 2.0
    snapshot_dt: float = 2.0
    snapshot_end_time: float = 220.0
    snapshot_radial_points: int = 200
    finite_difference_order: int = 8
    observer_radii: tuple[float | None, ...] = (8.0, 12.0, None)
    compact_modal_storage: bool = False

    def __post_init__(self) -> None:
        if self.radial_resolution < self.finite_difference_order + 3:
            raise ValueError("The radial resolution is too small for the stencil.")
        if self.angular_ell_max < 0:
            raise ValueError("angular_ell_max must be nonnegative.")
        if self.timestep <= 0.0 or self.end_time <= 0.0:
            raise ValueError("The timestep and end time must be positive.")
        if self.snapshot_radial_points < 8:
            raise ValueError("At least eight snapshot radii are required.")
        for cadence in (self.signal_dt, self.diagnostic_dt, self.snapshot_dt):
            if cadence < self.timestep:
                raise ValueError("Every output cadence must be at least one timestep.")


@dataclass
class SourcedSimulationResult:
    """Waveforms, field snapshots, and diagnostics of one sourced evolution."""

    rho: np.ndarray
    areal_radius: np.ndarray
    mode_ell: np.ndarray
    mode_m: np.ndarray
    mode_source_amplitude: np.ndarray
    response_ell: np.ndarray
    signal_times: np.ndarray
    observer_rho: np.ndarray
    observer_areal_radius: np.ndarray
    modal_signals: np.ndarray
    response_signals: np.ndarray
    diagnostic_times: np.ndarray
    constraint_linf: np.ndarray
    constraint_l2: np.ndarray
    field_linf: np.ndarray
    source_activity: np.ndarray
    snapshot_times: np.ndarray
    snapshot_rho: np.ndarray
    snapshot_areal_radius: np.ndarray
    modal_snapshots: np.ndarray
    response_snapshots: np.ndarray
    metadata: dict

    @property
    def retarded_time(self) -> np.ndarray:
        """Signal times shifted to the common geometric clock ``U=tau-q``."""

        return self.signal_times - float(self.metadata["retarded_time_offset"]["q"])

    def outer_index(self) -> int:
        """Index of the outer boundary observer (``scri`` or ``H_c^+``)."""

        matches = np.flatnonzero(self.observer_rho >= 1.0 - 1e-12)
        if matches.size == 0:
            raise ValueError("This run has no outer-boundary observer.")
        return int(matches[0])

    @property
    def uses_compact_modal_storage(self) -> bool:
        return self.modal_signals.shape[-1] == 0 and self.response_signals.size > 0

    def expanded_modal_signals(self) -> np.ndarray:
        """Return modal signals, expanding compact radial responses if needed."""

        if not self.uses_compact_modal_storage:
            return self.modal_signals
        lookup = {int(ell): index for index, ell in enumerate(self.response_ell)}
        indices = np.asarray([lookup[int(ell)] for ell in self.mode_ell])
        return (
            self.response_signals[..., indices]
            * self.mode_source_amplitude[None, None, :]
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["reproducibility"] = reproducibility_metadata()
        np.savez_compressed(
            path,
            rho=self.rho,
            areal_radius=self.areal_radius,
            mode_ell=self.mode_ell,
            mode_m=self.mode_m,
            mode_source_amplitude=self.mode_source_amplitude,
            response_ell=self.response_ell,
            signal_times=self.signal_times,
            observer_rho=self.observer_rho,
            observer_areal_radius=self.observer_areal_radius,
            modal_signals=self.modal_signals,
            response_signals=(
                self.response_signals.astype(np.float32)
                if self.uses_compact_modal_storage
                else self.response_signals
            ),
            diagnostic_times=self.diagnostic_times,
            constraint_linf=self.constraint_linf,
            constraint_l2=self.constraint_l2,
            field_linf=self.field_linf,
            source_activity=self.source_activity,
            snapshot_times=self.snapshot_times,
            snapshot_rho=self.snapshot_rho,
            snapshot_areal_radius=self.snapshot_areal_radius,
            modal_snapshots=self.modal_snapshots,
            response_snapshots=(
                self.response_snapshots.astype(np.float32)
                if self.uses_compact_modal_storage
                else self.response_snapshots
            ),
            metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
        )


def load_sourced_result(path: Path) -> SourcedSimulationResult:
    """Load a sourced-evolution archive without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        response_ell = data["response_ell"] if "response_ell" in data else np.empty(0, dtype=int)
        response_signals = data["response_signals"] if "response_signals" in data else np.empty((0, 0, 0))
        response_snapshots = data["response_snapshots"] if "response_snapshots" in data else np.empty((0, 0, 0))
        return SourcedSimulationResult(
            rho=data["rho"],
            areal_radius=data["areal_radius"],
            mode_ell=data["mode_ell"],
            mode_m=data["mode_m"],
            mode_source_amplitude=data["mode_source_amplitude"],
            response_ell=response_ell,
            signal_times=data["signal_times"],
            observer_rho=data["observer_rho"],
            observer_areal_radius=data["observer_areal_radius"],
            modal_signals=data["modal_signals"],
            response_signals=response_signals,
            diagnostic_times=data["diagnostic_times"],
            constraint_linf=data["constraint_linf"],
            constraint_l2=data["constraint_l2"],
            field_linf=data["field_linf"],
            source_activity=data["source_activity"],
            snapshot_times=data["snapshot_times"],
            snapshot_rho=data["snapshot_rho"],
            snapshot_areal_radius=data["snapshot_areal_radius"],
            modal_snapshots=data["modal_snapshots"],
            response_snapshots=response_snapshots,
            metadata=json.loads(data["metadata"].item()),
        )


def _cadence_stride(cadence: float, timestep: float) -> int:
    return max(1, int(round(cadence / timestep)))


class _Background:
    """Bundle the minimal-gauge coefficients shared by both backgrounds."""

    def __init__(
        self,
        key: str,
        rho: np.ndarray,
        catalogue: SourceModeCatalogue,
        mass: float,
        cosmological_length: float,
    ) -> None:
        self.key = key
        distinct = sorted(set(int(value) for value in catalogue.ell))
        if key == "sds":
            model = SdSParameters(
                mass=mass, cosmological_length=cosmological_length, ell=0
            )
            horizons = sds_horizons(model)
            self.model_dict = model.as_dict()
            self.horizons = horizons.as_dict()
            self.label = f"Schwarzschild-de Sitter, L/M={cosmological_length / mass:g}"
            self.areal_radius = areal_radius(rho, model)
            self.boost = bridge_boost(rho, model, "minimal")
            self.coefficient_a = propagation_coefficient(rho, model, "minimal")
            self.potential = np.stack(
                [
                    rescaled_scalar_potential(
                        rho,
                        SdSParameters(
                            mass=mass,
                            cosmological_length=cosmological_length,
                            ell=ell,
                        ),
                    )
                    for ell in distinct
                ]
            )
            self.offset = retarded_time_offset(model, REFERENCE_RADIUS)
            self.outer_radius = horizons.cosmological
            self.surface_gravity = horizons.kappa_cosmological
            self.parameters = model
        else:
            model = SchwarzschildScalarParameters(mass=mass, ell=0)
            self.model_dict = model.as_dict()
            self.horizons = {
                "black_hole": model.black_hole_horizon,
                "future_null_infinity": "rho=1 (r=infinity)",
            }
            self.label = "Schwarzschild"
            self.areal_radius = schwarzschild_areal_radius(rho, model)
            self.boost = schwarzschild_minimal_boost(rho)
            self.coefficient_a = schwarzschild_propagation_coefficient(rho, model)
            self.potential = np.stack(
                [
                    schwarzschild_potential(
                        rho, SchwarzschildScalarParameters(mass=mass, ell=ell)
                    )
                    for ell in distinct
                ]
            )
            self.offset = schwarzschild_offset(model, REFERENCE_RADIUS)
            self.outer_radius = np.inf
            self.surface_gravity = 0.0
            self.parameters = model
        index = {ell: position for position, ell in enumerate(distinct)}
        self.mode_potential = self.potential[
            [index[int(ell)] for ell in catalogue.ell]
        ]

    def height(self, radius: np.ndarray) -> np.ndarray:
        if self.key == "sds":
            return minimal_height(radius, self.parameters, REFERENCE_RADIUS)
        return schwarzschild_minimal_height(radius, self.parameters, REFERENCE_RADIUS)

    def compactification_derivative(self, radius: np.ndarray) -> np.ndarray:
        if self.key == "sds":
            return compactification_derivative(radius, self.parameters)
        return 2.0 * self.parameters.mass / radius**2

    def observer_coordinate(self, value: float | None) -> float:
        if value is None:
            return 1.0
        if self.key == "sds":
            horizons = sds_horizons(self.parameters)
            if not horizons.black_hole < value < horizons.cosmological:
                raise ValueError(
                    f"Observer r={value} must lie between the SdS horizons "
                    f"({horizons.black_hole:.4f}, {horizons.cosmological:.4f})."
                )
            return float(compact_radius(np.asarray([value]), self.parameters)[0])
        if value <= self.parameters.black_hole_horizon:
            raise ValueError("Every finite Schwarzschild observer must be outside 2M.")
        return float(1.0 - 2.0 * self.parameters.mass / value)


def run_sourced_simulation(
    *,
    background: str,
    source: LocalizedSourceParameters,
    numerical: SourcedNumericalParameters,
    cosmological_length: float = 80.0,
    mass: float = 1.0,
) -> SourcedSimulationResult:
    """Evolve ``Box Phi = S`` from rest with the localized emitter."""

    if background not in {"sds", "schwarzschild"}:
        raise ValueError("background must be 'sds' or 'schwarzschild'.")
    started = time.perf_counter()

    radial = UniformFiniteDifference(
        numerical.radial_resolution, numerical.finite_difference_order
    )
    rho = np.linspace(0.0, 1.0, numerical.radial_resolution)
    catalogue = build_mode_catalogue(source, numerical.angular_ell_max)
    geometry = _Background(background, rho, catalogue, mass, cosmological_length)
    radius = geometry.areal_radius

    left, right = source.radial_support
    if geometry.key == "sds":
        horizons = sds_horizons(geometry.parameters)
        if not horizons.black_hole < left < right < horizons.cosmological:
            raise ValueError("The emitter support must lie strictly between horizons.")
    elif left <= 2.0 * mass:
        raise ValueError("The emitter support must lie strictly outside r=2M.")

    # The source kernel -(r/G) A R(r) is evaluated only where R(r) is nonzero,
    # which keeps r/G away from its formal divergence at future null infinity.
    profile = radial_profile(radius, source)
    support = np.flatnonzero(profile > 0.0)
    if support.size < 12:
        raise ValueError(
            "The emitter is resolved by fewer than twelve radial points; "
            "increase the resolution or the support half-width."
        )
    window = slice(int(support[0]), int(support[-1]) + 1)
    support_radius = radius[window]
    source_kernel = (
        -source.amplitude
        * profile[window]
        * support_radius
        / geometry.compactification_derivative(support_radius)
    )
    support_height = geometry.height(support_radius)
    earliest_tau = float(
        np.min(source.killing_time_support[0] + support_height)
    )
    if earliest_tau <= 0.0:
        raise ValueError(
            "The emitter would already be active at tau=0; increase "
            f"time_center (earliest activation tau={earliest_tau:.3f})."
        )
    latest_tau = float(np.max(source.killing_time_support[1] + support_height))
    distinct_ells = np.unique(catalogue.ell)
    ell_to_response = {int(ell): index for index, ell in enumerate(distinct_ells)}
    mode_response_index = np.asarray(
        [ell_to_response[int(ell)] for ell in catalogue.ell], dtype=int
    )

    speeds = np.maximum(
        np.abs(-geometry.coefficient_a * (1.0 + geometry.boost)),
        np.abs(geometry.coefficient_a * (1.0 - geometry.boost)),
    )
    characteristic_speed_bound = float(np.max(speeds))
    coordinate_cfl = (
        characteristic_speed_bound * numerical.timestep / radial.spacing
    )
    if coordinate_cfl > 1.4:
        raise ValueError(
            f"Coordinate CFL {coordinate_cfl:.3f} exceeds the validated RK4 "
            "limit 1.4; reduce the timestep or the resolution."
        )

    observer_rho = np.asarray(
        [geometry.observer_coordinate(value) for value in numerical.observer_radii]
    )
    observer_radius = np.asarray(
        [
            geometry.outer_radius if value is None else float(value)
            for value in numerical.observer_radii
        ]
    )

    coefficient_a = geometry.coefficient_a
    coefficient_ab = geometry.coefficient_a * geometry.boost
    potential = geometry.potential
    state = np.zeros((3, distinct_ells.size, numerical.radial_resolution))

    def right_hand_side(values: np.ndarray, current_time: float) -> np.ndarray:
        velocity = coefficient_ab * values[1] + coefficient_a * values[2]
        flux = coefficient_a * values[1] + coefficient_ab * values[2]
        momentum = radial.differentiate(flux) - potential * values[0]
        if earliest_tau <= current_time <= latest_tau:
            temporal = time_profile(current_time - support_height, source)
            if np.any(temporal):
                momentum[:, window] += source_kernel * temporal
        return np.stack([velocity, radial.differentiate(velocity), momentum])

    def source_strength(current_time: float) -> float:
        if not earliest_tau <= current_time <= latest_tau:
            return 0.0
        return float(
            np.max(np.abs(time_profile(current_time - support_height, source)))
        )

    signal_stride = _cadence_stride(numerical.signal_dt, numerical.timestep)
    diagnostic_stride = _cadence_stride(numerical.diagnostic_dt, numerical.timestep)
    snapshot_stride = _cadence_stride(numerical.snapshot_dt, numerical.timestep)
    snapshot_indices = np.unique(
        np.linspace(
            0,
            numerical.radial_resolution - 1,
            numerical.snapshot_radial_points,
        ).astype(int)
    )
    total_steps = int(np.ceil(numerical.end_time / numerical.timestep))
    progress_stride = max(1, total_steps // 12)

    signal_times: list[float] = []
    modal_signals: list[np.ndarray] = []
    response_signals: list[np.ndarray] = []
    diagnostic_times: list[float] = []
    constraint_linf: list[float] = []
    constraint_l2: list[float] = []
    field_linf: list[float] = []
    activity: list[float] = []
    snapshot_times: list[float] = []
    modal_snapshots: list[np.ndarray] = []
    response_snapshots: list[np.ndarray] = []

    def record_signal(current_time: float) -> None:
        signal_times.append(current_time)
        responses = np.stack(
            [radial.interpolate(state[0], point) for point in observer_rho]
        )
        response_signals.append(responses)
        if not numerical.compact_modal_storage:
            modal_signals.append(
                responses[:, mode_response_index] * catalogue.amplitude[None, :]
            )

    def record_diagnostics(current_time: float) -> None:
        response_constraint = state[1] - radial.differentiate(state[0])
        constraint = (
            response_constraint[mode_response_index]
            * catalogue.amplitude[:, None]
        )
        modal_field = (
            state[0][mode_response_index] * catalogue.amplitude[:, None]
        )
        diagnostic_times.append(current_time)
        constraint_linf.append(float(np.max(np.abs(constraint))))
        constraint_l2.append(float(np.sqrt(np.mean(constraint**2))))
        field_linf.append(float(np.max(np.abs(modal_field))))
        activity.append(source_strength(current_time))

    def record_snapshot(current_time: float) -> None:
        snapshot_times.append(current_time)
        response = state[0][:, snapshot_indices]
        response_snapshots.append(response)
        if not numerical.compact_modal_storage:
            modal_snapshots.append(
                response[mode_response_index] * catalogue.amplitude[:, None]
            )

    current_time = 0.0
    record_signal(current_time)
    record_diagnostics(current_time)
    record_snapshot(current_time)
    for step_number in range(1, total_steps + 1):
        step = (
            numerical.timestep
            if step_number < total_steps
            else numerical.end_time - current_time
        )
        first = right_hand_side(state, current_time)
        second = right_hand_side(state + 0.5 * step * first, current_time + 0.5 * step)
        third = right_hand_side(state + 0.5 * step * second, current_time + 0.5 * step)
        fourth = right_hand_side(state + step * third, current_time + step)
        state += step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
        current_time += step
        is_final = step_number == total_steps
        if step_number % signal_stride == 0 or is_final:
            record_signal(current_time)
        if step_number % diagnostic_stride == 0 or is_final:
            record_diagnostics(current_time)
        if (
            current_time <= numerical.snapshot_end_time
            and step_number % snapshot_stride == 0
        ) or is_final:
            record_snapshot(current_time)
        if step_number % progress_stride == 0:
            LOGGER.info(
                "sourced %s: tau=%9.2f / %.2f, max|u|=%.3e",
                geometry.label,
                current_time,
                numerical.end_time,
                float(np.max(np.abs(state[0]))),
            )
        if not np.isfinite(state).all():
            raise FloatingPointError(
                f"The sourced evolution lost finiteness at tau={current_time:.4f}."
            )

    elapsed = time.perf_counter() - started
    metadata = {
        "background": geometry.label,
        "background_key": geometry.key,
        "model": geometry.model_dict,
        "horizons": geometry.horizons,
        "cosmological_length": float(cosmological_length),
        "surface_gravity_cosmological": float(geometry.surface_gravity),
        "source": source.as_dict(),
        "source_modes": catalogue.as_dict(),
        "evolved_radial_responses": int(distinct_ells.size),
        "angular_expansion_check": verify_angular_expansion(
            source, numerical.angular_ell_max
        ),
        "source_support": {
            "radial_points": int(support.size),
            "rho_window": [float(rho[window][0]), float(rho[window][-1])],
            "areal_radius_window": [
                float(support_radius[0]),
                float(support_radius[-1]),
            ],
            "height_window": [
                float(np.min(support_height)),
                float(np.max(support_height)),
            ],
            "bridge_time_window": [earliest_tau, latest_tau],
        },
        "numerical": asdict(numerical),
        "radial_discretization": {
            "coordinate": "uniform rho in [0,1]",
            "derivative": (
                f"{numerical.finite_difference_order}th-order centered finite "
                "difference with matched one-sided endpoint stencils"
            ),
            "time_stepper": "explicit classical RK4 with staged source evaluation",
            "coordinate_cfl": coordinate_cfl,
            "maximum_characteristic_speed": characteristic_speed_bound,
        },
        "equations": {
            "field": "Phi=r^{-1} sum_lm u_lm Y^R_lm, Box Phi = S",
            "u": "dt(u)=A*(B*psi+pi)",
            "psi": "dt(psi)=d_rho[A*(B*psi+pi)]",
            "pi": "dt(pi)=d_rho[A*(psi+B*pi)]-P_ell*u-(r/G)*S_lm(tau-h_L(r),r)",
        },
        "retarded_time_offset": {
            "q": float(geometry.offset),
            "reference_radius": REFERENCE_RADIUS,
            "evaluation": "analytic",
        },
        "iterations": total_steps,
        "final_time": current_time,
        "wall_seconds": elapsed,
    }
    LOGGER.info(
        "finished sourced %s: modes=%d, wall=%.1fs, max constraint=%.3e",
        geometry.label,
        catalogue.count,
        elapsed,
        max(constraint_linf),
    )
    return SourcedSimulationResult(
        rho=rho,
        areal_radius=radius,
        mode_ell=catalogue.ell,
        mode_m=catalogue.m,
        mode_source_amplitude=catalogue.amplitude,
        response_ell=distinct_ells,
        signal_times=np.asarray(signal_times),
        observer_rho=observer_rho,
        observer_areal_radius=observer_radius,
        modal_signals=(
            np.asarray(modal_signals)
            if modal_signals
            else np.empty((len(signal_times), observer_rho.size, 0))
        ),
        response_signals=np.asarray(response_signals),
        diagnostic_times=np.asarray(diagnostic_times),
        constraint_linf=np.asarray(constraint_linf),
        constraint_l2=np.asarray(constraint_l2),
        field_linf=np.asarray(field_linf),
        source_activity=np.asarray(activity),
        snapshot_times=np.asarray(snapshot_times),
        snapshot_rho=rho[snapshot_indices],
        snapshot_areal_radius=radius[snapshot_indices],
        modal_snapshots=(
            np.asarray(modal_snapshots)
            if modal_snapshots
            else np.empty((len(snapshot_times), 0, snapshot_indices.size))
        ),
        response_snapshots=np.asarray(response_snapshots),
        metadata=metadata,
    )
