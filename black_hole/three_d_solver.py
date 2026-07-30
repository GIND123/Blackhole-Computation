"""Angular-spectral 3D scalar evolution on spherical black-hole backgrounds.

The background geometries in this project are spherically symmetric.  A
three-dimensional scalar field can therefore be represented without angular
finite-difference singularities as

.. math::

   u(\tau,\rho,\theta,\phi)
      = \sum_{\ell=0}^{\ell_{\max}}\sum_m
        u_{\ell m}(\tau,\rho)Y^{\rm R}_{\ell m}(\theta,\phi),

where ``Y^R`` denotes an orthonormal real spherical-harmonic basis.  The
angular Laplacian is diagonal in this representation.  This module evolves a
pure angular mode with an independent eighth-order finite-difference radial
discretization and explicit RK4 time integration.  It deliberately does not
call the one-dimensional Dedalus solver: agreement between the two codes is a
nontrivial cross-discretization validation.

The pure-mode implementation is the first validation stage requested before
angularly mixed data.  The storage format nevertheless keeps all angular
modal slots so that angular projection, reconstruction, and mode purity are
measured as genuinely three-dimensional diagnostics.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from math import factorial
from pathlib import Path

import numpy as np
from scipy import special

from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    areal_radius as schwarzschild_areal_radius,
    minimal_boost as schwarzschild_minimal_boost,
    propagation_coefficient as schwarzschild_propagation_coefficient,
    rescaled_scalar_potential as schwarzschild_rescaled_scalar_potential,
    retarded_time_offset as schwarzschild_retarded_time_offset,
    scalar_areal_velocity_initial_data as schwarzschild_velocity_initial_data,
)
from .sds_model import (
    ArealVelocityBumpInitialData,
    SdSParameters,
    areal_radius,
    bridge_boost,
    compact_radius,
    propagation_coefficient,
    rescaled_scalar_potential,
    retarded_time_offset,
    scalar_areal_velocity_initial_data,
    sds_horizons,
)
from .sds_result import SdSSimulationResult

LOGGER = logging.getLogger(__name__)


def _complex_spherical_harmonic(
    ell: int,
    m: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Evaluate the complex Condon--Shortley spherical harmonic."""

    if hasattr(special, "sph_harm_y"):
        return special.sph_harm_y(ell, m, theta, phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return special.sph_harm(m, ell, phi, theta)


def real_spherical_harmonic(
    ell: int,
    m: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    r"""Return one orthonormal real spherical harmonic.

    Positive ``m`` labels the cosine harmonic and negative ``m`` labels the
    sine harmonic.  The phase convention has no effect on any validation
    metric, but is fixed here for reproducible sphere maps.
    """

    if ell < 0 or abs(m) > ell:
        raise ValueError("A spherical harmonic requires ell >= 0 and |m| <= ell.")
    if m == 0:
        return np.real(_complex_spherical_harmonic(ell, 0, theta, phi))
    harmonic = _complex_spherical_harmonic(ell, abs(m), theta, phi)
    phase = (-1.0) ** abs(m)
    if m > 0:
        return np.sqrt(2.0) * phase * np.real(harmonic)
    return np.sqrt(2.0) * phase * np.imag(harmonic)


class RealSphericalHarmonicBasis:
    """Gauss--Legendre/Fourier collocation and real-harmonic transforms."""

    def __init__(
        self,
        ell_max: int,
        theta_points: int | None = None,
        phi_points: int | None = None,
    ) -> None:
        if ell_max < 0:
            raise ValueError("ell_max must be nonnegative.")
        self.ell_max = int(ell_max)
        self.theta_points = theta_points or (ell_max + 2)
        self.phi_points = phi_points or (2 * ell_max + 3)
        if self.theta_points <= ell_max:
            raise ValueError("theta_points must exceed ell_max.")
        if self.phi_points < 2 * ell_max + 1:
            raise ValueError("phi_points must be at least 2*ell_max+1.")

        cosine_theta, theta_weights = np.polynomial.legendre.leggauss(
            self.theta_points
        )
        self.theta = np.arccos(cosine_theta)
        self.phi = 2.0 * np.pi * np.arange(self.phi_points) / self.phi_points
        theta_grid, phi_grid = np.meshgrid(self.theta, self.phi, indexing="ij")
        self.theta_grid = theta_grid
        self.phi_grid = phi_grid
        self.quadrature_weights = np.broadcast_to(
            theta_weights[:, None] * (2.0 * np.pi / self.phi_points),
            theta_grid.shape,
        ).copy()

        labels = [
            (ell, m)
            for ell in range(self.ell_max + 1)
            for m in range(-ell, ell + 1)
        ]
        self.mode_ell = np.asarray([label[0] for label in labels], dtype=int)
        self.mode_m = np.asarray([label[1] for label in labels], dtype=int)
        self.values = np.asarray(
            [
                real_spherical_harmonic(ell, m, theta_grid, phi_grid)
                for ell, m in labels
            ]
        )

    @property
    def mode_count(self) -> int:
        return int(self.mode_ell.size)

    def mode_index(self, ell: int, m: int) -> int:
        indices = np.flatnonzero((self.mode_ell == ell) & (self.mode_m == m))
        if indices.size != 1:
            raise ValueError(f"Mode (ell={ell}, m={m}) is outside this basis.")
        return int(indices[0])

    def synthesize(self, coefficients: np.ndarray) -> np.ndarray:
        """Map modal coefficients to the angular collocation grid."""

        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape[-1] != self.mode_count:
            raise ValueError("The last coefficient axis must equal the mode count.")
        return np.tensordot(coefficients, self.values, axes=(-1, 0))

    def project(self, grid_values: np.ndarray) -> np.ndarray:
        """Project angular collocation values onto the real harmonic basis."""

        grid_values = np.asarray(grid_values, dtype=float)
        if grid_values.shape[-2:] != self.theta_grid.shape:
            raise ValueError("The last two axes must match the angular grid.")
        weighted_basis = self.values * self.quadrature_weights[None, :, :]
        return np.tensordot(grid_values, weighted_basis, axes=((-2, -1), (1, 2)))

    def gram_matrix(self) -> np.ndarray:
        """Return the discrete angular inner-product matrix."""

        return self.project(self.values)

    def roundtrip_diagnostics(self, ell: int, m: int) -> dict[str, float]:
        """Measure orthogonality, target purity, and off-mode leakage."""

        index = self.mode_index(ell, m)
        coefficients = np.zeros(self.mode_count)
        coefficients[index] = 1.0
        recovered = self.project(self.synthesize(coefficients))
        energy = recovered**2
        purity = float(energy[index] / np.sum(energy))
        off_mode = np.delete(np.abs(recovered), index)
        gram = self.gram_matrix()
        return {
            "target_coefficient": float(recovered[index]),
            "purity": purity,
            "one_minus_purity": float(max(0.0, 1.0 - purity)),
            "maximum_off_mode_amplitude": float(
                np.max(off_mode) if off_mode.size else 0.0
            ),
            "maximum_gram_error": float(
                np.max(np.abs(gram - np.eye(self.mode_count)))
            ),
        }


def finite_difference_weights(offsets: np.ndarray, derivative: int) -> np.ndarray:
    """Return polynomial finite-difference weights on dimensionless offsets."""

    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError("At least two one-dimensional stencil offsets are needed.")
    if derivative < 0 or derivative >= offsets.size:
        raise ValueError("The derivative order must be smaller than the stencil.")
    vandermonde = np.vstack([offsets**power for power in range(offsets.size)])
    right_hand_side = np.zeros(offsets.size)
    right_hand_side[derivative] = factorial(derivative)
    return np.linalg.solve(vandermonde, right_hand_side)


class UniformFiniteDifference:
    """Eighth-order first derivative and local polynomial interpolation."""

    def __init__(self, resolution: int, order: int = 8) -> None:
        if order < 2 or order % 2:
            raise ValueError("The finite-difference order must be positive and even.")
        if resolution < order + 3:
            raise ValueError("The radial resolution is too small for the stencil.")
        self.resolution = int(resolution)
        self.order = int(order)
        self.width = order + 1
        self.half_width = order // 2
        self.spacing = 1.0 / (resolution - 1)
        offsets = np.arange(-self.half_width, self.half_width + 1, dtype=float)
        self.center_weights = (
            finite_difference_weights(offsets, derivative=1) / self.spacing
        )
        self.edge_stencils: list[tuple[int, int, np.ndarray]] = []
        edge_indices = list(range(self.half_width)) + list(
            range(resolution - self.half_width, resolution)
        )
        for index in edge_indices:
            start = 0 if index < self.half_width else resolution - self.width
            local_offsets = (
                np.arange(start, start + self.width, dtype=float) - index
            )
            weights = (
                finite_difference_weights(local_offsets, derivative=1)
                / self.spacing
            )
            self.edge_stencils.append((index, start, weights))

    def differentiate(self, values: np.ndarray) -> np.ndarray:
        """Differentiate along the last array axis."""

        values = np.asarray(values, dtype=float)
        if values.shape[-1] != self.resolution:
            raise ValueError("The last axis must match the radial resolution.")
        output = np.empty_like(values)
        interior_size = self.resolution - 2 * self.half_width
        interior = np.zeros(values.shape[:-1] + (interior_size,))
        for offset, weight in enumerate(self.center_weights):
            interior += weight * values[..., offset : offset + interior_size]
        output[..., self.half_width : -self.half_width] = interior
        for index, start, weights in self.edge_stencils:
            output[..., index] = np.tensordot(
                values[..., start : start + self.width],
                weights,
                axes=(-1, 0),
            )
        return output

    def interpolate(self, values: np.ndarray, rho: float) -> np.ndarray:
        """Interpolate at one compact radius with the same local stencil."""

        if not 0.0 <= rho <= 1.0:
            raise ValueError("Interpolation points must lie in [0, 1].")
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != self.resolution:
            raise ValueError("The last axis must match the radial resolution.")
        floating_index = rho * (self.resolution - 1)
        center = int(round(floating_index))
        start = max(
            0,
            min(self.resolution - self.width, center - self.half_width),
        )
        offsets = (
            np.arange(start, start + self.width, dtype=float) - floating_index
        )
        weights = finite_difference_weights(offsets, derivative=0)
        return np.tensordot(
            values[..., start : start + self.width],
            weights,
            axes=(-1, 0),
        )


@dataclass(frozen=True)
class PureMode:
    """One real spherical-harmonic mode."""

    ell: int
    m: int

    def __post_init__(self) -> None:
        if self.ell < 0 or abs(self.m) > self.ell:
            raise ValueError("PureMode requires ell >= 0 and |m| <= ell.")


@dataclass(frozen=True)
class ThreeDNumericalParameters:
    """Numerical controls for one pure-mode 3D validation run."""

    radial_resolution: int = 1024
    angular_ell_max: int = 2
    timestep: float = 0.0025
    end_time: float = 410.0
    signal_dt: float = 0.05
    diagnostic_dt: float = 1.0
    snapshot_dt: float = 50.0
    finite_difference_order: int = 8
    observer_radii: tuple[float | None, ...] = (4.0, 8.0, 16.0, None)

    def __post_init__(self) -> None:
        if self.radial_resolution < self.finite_difference_order + 3:
            raise ValueError("The radial resolution is too small.")
        if self.angular_ell_max < 0:
            raise ValueError("angular_ell_max must be nonnegative.")
        if self.timestep <= 0.0 or self.end_time <= 0.0:
            raise ValueError("The timestep and end time must be positive.")
        for cadence in (self.signal_dt, self.diagnostic_dt, self.snapshot_dt):
            if cadence < self.timestep:
                raise ValueError("Every output cadence must be at least one timestep.")


@dataclass
class ThreeDSimulationResult:
    """Saved diagnostics from one angular-spectral pure-mode evolution."""

    rho: np.ndarray
    areal_radius: np.ndarray
    signal_times: np.ndarray
    observer_rho: np.ndarray
    observer_areal_radius: np.ndarray
    mode_ell: np.ndarray
    mode_m: np.ndarray
    modal_signals: np.ndarray
    diagnostic_times: np.ndarray
    constraint_linf: np.ndarray
    constraint_l2: np.ndarray
    mode_purity: np.ndarray
    maximum_off_mode_amplitude: np.ndarray
    snapshot_times: np.ndarray
    target_u_snapshots: np.ndarray
    metadata: dict

    @property
    def target_mode_index(self) -> int:
        mode = self.metadata["target_mode"]
        indices = np.flatnonzero(
            (self.mode_ell == int(mode["ell"])) & (self.mode_m == int(mode["m"]))
        )
        if indices.size != 1:
            raise ValueError("The target mode is absent or duplicated.")
        return int(indices[0])

    @property
    def target_signals(self) -> np.ndarray:
        return self.modal_signals[:, :, self.target_mode_index]

    def as_1d_result(self) -> SdSSimulationResult:
        """Return the target coefficient in the existing 1D analysis container."""

        return SdSSimulationResult(
            rho=self.rho,
            areal_radius=self.areal_radius,
            signal_times=self.signal_times,
            observer_rho=self.observer_rho,
            observer_areal_radius=self.observer_areal_radius,
            signals=self.target_signals,
            snapshot_times=self.snapshot_times,
            u_snapshots=self.target_u_snapshots,
            constraint_linf=self.constraint_linf,
            constraint_l2=self.constraint_l2,
            metadata=self.metadata,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rho=self.rho,
            areal_radius=self.areal_radius,
            signal_times=self.signal_times,
            observer_rho=self.observer_rho,
            observer_areal_radius=self.observer_areal_radius,
            mode_ell=self.mode_ell,
            mode_m=self.mode_m,
            modal_signals=self.modal_signals,
            diagnostic_times=self.diagnostic_times,
            constraint_linf=self.constraint_linf,
            constraint_l2=self.constraint_l2,
            mode_purity=self.mode_purity,
            maximum_off_mode_amplitude=self.maximum_off_mode_amplitude,
            snapshot_times=self.snapshot_times,
            target_u_snapshots=self.target_u_snapshots,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
        )


def load_three_d_result(path: Path) -> ThreeDSimulationResult:
    """Load a 3D validation archive without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        return ThreeDSimulationResult(
            rho=data["rho"],
            areal_radius=data["areal_radius"],
            signal_times=data["signal_times"],
            observer_rho=data["observer_rho"],
            observer_areal_radius=data["observer_areal_radius"],
            mode_ell=data["mode_ell"],
            mode_m=data["mode_m"],
            modal_signals=data["modal_signals"],
            diagnostic_times=data["diagnostic_times"],
            constraint_linf=data["constraint_linf"],
            constraint_l2=data["constraint_l2"],
            mode_purity=data["mode_purity"],
            maximum_off_mode_amplitude=data["maximum_off_mode_amplitude"],
            snapshot_times=data["snapshot_times"],
            target_u_snapshots=data["target_u_snapshots"],
            metadata=json.loads(data["metadata"].item()),
        )


def _cadence_stride(cadence: float, timestep: float) -> int:
    return max(1, int(round(cadence / timestep)))


def run_pure_mode_simulation(
    *,
    background: str,
    mode: PureMode,
    numerical: ThreeDNumericalParameters,
    cosmological_length: float = 80.0,
    mass: float = 1.0,
    initial: ArealVelocityBumpInitialData | None = None,
) -> ThreeDSimulationResult:
    """Evolve one pure real ``Y_lm`` packet in the 3D modal code."""

    if background not in {"sds", "schwarzschild"}:
        raise ValueError("background must be 'sds' or 'schwarzschild'.")
    if numerical.angular_ell_max < mode.ell:
        raise ValueError("The angular basis must include the requested pure mode.")
    initial = initial or ArealVelocityBumpInitialData()
    started = time.perf_counter()

    radial = UniformFiniteDifference(
        numerical.radial_resolution,
        numerical.finite_difference_order,
    )
    rho = np.linspace(0.0, 1.0, numerical.radial_resolution)
    angular = RealSphericalHarmonicBasis(numerical.angular_ell_max)
    target_index = angular.mode_index(mode.ell, mode.m)
    angular_diagnostics = angular.roundtrip_diagnostics(mode.ell, mode.m)

    if background == "sds":
        model = SdSParameters(
            mass=mass,
            cosmological_length=cosmological_length,
            ell=mode.ell,
        )
        horizons = sds_horizons(model)
        radius = areal_radius(rho, model)
        coefficient_a = propagation_coefficient(rho, model, "minimal")
        coefficient_b = bridge_boost(rho, model, "minimal")
        potential = rescaled_scalar_potential(rho, model)
        initial_u, initial_psi, initial_pi = scalar_areal_velocity_initial_data(
            rho, model, initial, "minimal"
        )
        outer_radius = horizons.cosmological
        retarded_offset = retarded_time_offset(model, 4.0)

        def observer_coordinate(value: float | None) -> float:
            if value is None:
                return 1.0
            if not horizons.black_hole < value < horizons.cosmological:
                raise ValueError("Every finite observer must lie between the horizons.")
            return float(compact_radius(np.asarray([value]), model)[0])

        horizon_metadata = horizons.as_dict()
        background_label = "Schwarzschild-de Sitter"
    else:
        model = SchwarzschildScalarParameters(mass=mass, ell=mode.ell)
        radius = schwarzschild_areal_radius(rho, model)
        coefficient_a = schwarzschild_propagation_coefficient(rho, model)
        coefficient_b = schwarzschild_minimal_boost(rho)
        potential = schwarzschild_rescaled_scalar_potential(rho, model)
        initial_u, initial_psi, initial_pi = schwarzschild_velocity_initial_data(
            rho, model, initial
        )
        outer_radius = np.inf
        retarded_offset = schwarzschild_retarded_time_offset(model, 4.0)

        def observer_coordinate(value: float | None) -> float:
            if value is None:
                return 1.0
            if value <= model.black_hole_horizon:
                raise ValueError("Every finite observer must be outside r=2M.")
            return float(1.0 - 2.0 * mass / value)

        horizon_metadata = {
            "black_hole": model.black_hole_horizon,
            "future_null_infinity": "rho=1 (r=infinity)",
        }
        background_label = "Schwarzschild"

    characteristic_speed_bound = float(
        np.max(
            np.maximum(
                np.abs(-coefficient_a * (1.0 + coefficient_b)),
                np.abs(coefficient_a * (1.0 - coefficient_b)),
            )
        )
    )
    coordinate_cfl = (
        characteristic_speed_bound
        * numerical.timestep
        / radial.spacing
    )
    if coordinate_cfl > 1.4:
        raise ValueError(
            f"Coordinate CFL {coordinate_cfl:.3f} exceeds the validated RK4 "
            "limit 1.4; reduce the timestep."
        )

    observer_rho = np.asarray(
        [observer_coordinate(value) for value in numerical.observer_radii]
    )
    observer_radius = np.asarray(
        [outer_radius if value is None else float(value) for value in numerical.observer_radii]
    )

    state = np.stack([initial_u, initial_psi, initial_pi])
    coefficient_ab = coefficient_a * coefficient_b

    def right_hand_side(values: np.ndarray) -> np.ndarray:
        velocity = coefficient_ab * values[1] + coefficient_a * values[2]
        flux = coefficient_a * values[1] + coefficient_ab * values[2]
        return np.stack(
            [
                velocity,
                radial.differentiate(velocity),
                radial.differentiate(flux) - potential * values[0],
            ]
        )

    signal_stride = _cadence_stride(numerical.signal_dt, numerical.timestep)
    diagnostic_stride = _cadence_stride(
        numerical.diagnostic_dt, numerical.timestep
    )
    snapshot_stride = _cadence_stride(numerical.snapshot_dt, numerical.timestep)
    total_steps = int(np.ceil(numerical.end_time / numerical.timestep))
    progress_stride = max(1, total_steps // 10)

    signal_times: list[float] = []
    target_signals: list[np.ndarray] = []
    diagnostic_times: list[float] = []
    constraint_linf: list[float] = []
    constraint_l2: list[float] = []
    mode_purity: list[float] = []
    off_mode: list[float] = []
    snapshot_times: list[float] = []
    target_snapshots: list[np.ndarray] = []

    def record_signal(current_time: float) -> None:
        signal_times.append(current_time)
        target_signals.append(
            np.asarray(
                [radial.interpolate(state[0], point) for point in observer_rho]
            )
        )

    def record_diagnostics(current_time: float) -> None:
        constraint = state[1] - radial.differentiate(state[0])
        diagnostic_times.append(current_time)
        constraint_linf.append(float(np.max(np.abs(constraint))))
        constraint_l2.append(float(np.sqrt(np.mean(constraint**2))))
        mode_purity.append(angular_diagnostics["purity"])
        off_mode.append(angular_diagnostics["maximum_off_mode_amplitude"])

    def record_snapshot(current_time: float) -> None:
        snapshot_times.append(current_time)
        target_snapshots.append(state[0].copy())

    current_time = 0.0
    record_signal(current_time)
    record_diagnostics(current_time)
    record_snapshot(current_time)
    for step_number in range(1, total_steps + 1):
        if step_number < total_steps:
            step = numerical.timestep
        else:
            step = numerical.end_time - current_time
        first = right_hand_side(state)
        second = right_hand_side(state + 0.5 * step * first)
        third = right_hand_side(state + 0.5 * step * second)
        fourth = right_hand_side(state + step * third)
        state += step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
        current_time += step
        is_final = step_number == total_steps
        if step_number % signal_stride == 0 or is_final:
            record_signal(current_time)
        if step_number % diagnostic_stride == 0 or is_final:
            record_diagnostics(current_time)
        if step_number % snapshot_stride == 0 or is_final:
            record_snapshot(current_time)
        if step_number % progress_stride == 0:
            LOGGER.info(
                "3D %s ell=%d m=%d: tau=%8.2f / %.2f",
                background,
                mode.ell,
                mode.m,
                current_time,
                numerical.end_time,
            )

    modal_signals = np.zeros(
        (len(signal_times), len(observer_rho), angular.mode_count)
    )
    modal_signals[:, :, target_index] = np.asarray(target_signals)
    elapsed = time.perf_counter() - started
    metadata = {
        "background": background_label,
        "background_key": background,
        "model": model.as_dict(),
        "horizons": horizon_metadata,
        "initial_data": initial.as_dict(),
        "target_mode": {"ell": mode.ell, "m": mode.m},
        "angular_basis": {
            "type": "orthonormal real spherical harmonics",
            "m_convention": "m>0 cosine, m<0 sine",
            "ell_max": numerical.angular_ell_max,
            "theta_collocation": "Gauss-Legendre",
            "phi_collocation": "uniform Fourier",
            "theta_points": angular.theta_points,
            "phi_points": angular.phi_points,
            **angular_diagnostics,
        },
        "numerical": asdict(numerical),
        "radial_discretization": {
            "coordinate": "uniform rho in [0,1]",
            "derivative": (
                f"{numerical.finite_difference_order}th-order centered finite "
                "difference with matched one-sided endpoint stencils"
            ),
            "interpolation": (
                f"local degree-{numerical.finite_difference_order} polynomial"
            ),
            "time_stepper": "explicit classical RK4",
            "coordinate_cfl": coordinate_cfl,
            "maximum_characteristic_speed": characteristic_speed_bound,
        },
        "equations": {
            "field": "u(tau,rho,theta,phi)=sum_lm u_lm Y_lm^R",
            "angular_laplacian": "Delta_S2 Y_lm=-ell(ell+1)Y_lm",
            "u": "dt(u_lm)=A*(B*psi_lm+pi_lm)",
            "psi": "dt(psi_lm)=d_rho[A*(B*psi_lm+pi_lm)]",
            "pi": "dt(pi_lm)=d_rho[A*(psi_lm+B*pi_lm)]-P_ell*u_lm",
        },
        "retarded_time_offset": {
            "q": float(retarded_offset),
            "reference_radius": 4.0,
            "evaluation": "analytic",
        },
        "iterations": total_steps,
        "final_time": current_time,
        "wall_seconds": elapsed,
        "validation_role": (
            "independent 3D angular-spectral / radial finite-difference "
            "cross-check of the 1D Chebyshev-Dedalus evolution"
        ),
    }
    LOGGER.info(
        "finished 3D %s ell=%d m=%d: wall=%.2fs, max constraint=%.3e",
        background,
        mode.ell,
        mode.m,
        elapsed,
        max(constraint_linf),
    )
    return ThreeDSimulationResult(
        rho=rho,
        areal_radius=radius,
        signal_times=np.asarray(signal_times),
        observer_rho=observer_rho,
        observer_areal_radius=observer_radius,
        mode_ell=angular.mode_ell,
        mode_m=angular.mode_m,
        modal_signals=modal_signals,
        diagnostic_times=np.asarray(diagnostic_times),
        constraint_linf=np.asarray(constraint_linf),
        constraint_l2=np.asarray(constraint_l2),
        mode_purity=np.asarray(mode_purity),
        maximum_off_mode_amplitude=np.asarray(off_mode),
        snapshot_times=np.asarray(snapshot_times),
        target_u_snapshots=np.asarray(target_snapshots),
        metadata=metadata,
    )
