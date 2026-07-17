"""Dedalus solver for scalar waves on Schwarzschild-de Sitter bridges."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import dedalus.public as d3
import numpy as np

from .sds_model import (
    ArealBumpInitialData,
    BRIDGE_CHOICES,
    ScalarInitialData,
    SdSParameters,
    areal_radius,
    bridge_boost,
    propagation_coefficient,
    rescaled_scalar_potential,
    scalar_gaussian_initial_data,
    scalar_areal_bump_initial_data,
    sds_horizons,
)
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    areal_radius as schwarzschild_areal_radius,
    minimal_boost as schwarzschild_minimal_boost,
    propagation_coefficient as schwarzschild_propagation_coefficient,
    rescaled_scalar_potential as schwarzschild_rescaled_scalar_potential,
    scalar_gaussian_initial_data as schwarzschild_initial_data,
    scalar_areal_bump_initial_data as schwarzschild_areal_bump_initial_data,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdSNumericalParameters:
    """Numerical settings for one SdS scalar evolution."""

    resolution: int = 256
    timestep: float = 0.02
    end_time: float = 250.0
    signal_dt: float = 0.05
    snapshot_dt: float = 0.5
    observers: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    timestepper: str = "RK222"
    bridge: str = "minimal"

    def __post_init__(self) -> None:
        if self.resolution < 32:
            raise ValueError("Resolution must be at least 32.")
        if self.timestep <= 0 or self.end_time <= 0:
            raise ValueError("Timestep and end time must be positive.")
        if self.signal_dt < self.timestep:
            raise ValueError("signal_dt cannot be smaller than timestep.")
        if self.snapshot_dt < self.timestep:
            raise ValueError("snapshot_dt cannot be smaller than timestep.")
        if any(point < 0 or point > 1 for point in self.observers):
            raise ValueError("Observer locations must lie in [0, 1].")
        if self.bridge not in BRIDGE_CHOICES:
            raise ValueError(f"Unknown bridge {self.bridge!r}.")


@dataclass
class SdSSimulationResult:
    """In-memory output from one SdS scalar evolution."""

    rho: np.ndarray
    areal_radius: np.ndarray
    signal_times: np.ndarray
    observer_rho: np.ndarray
    observer_areal_radius: np.ndarray
    signals: np.ndarray
    snapshot_times: np.ndarray
    u_snapshots: np.ndarray
    constraint_linf: np.ndarray
    constraint_l2: np.ndarray
    metadata: dict

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
            signals=self.signals,
            snapshot_times=self.snapshot_times,
            u_snapshots=self.u_snapshots,
            constraint_linf=self.constraint_linf,
            constraint_l2=self.constraint_l2,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
        )


def load_sds_result(path: Path) -> SdSSimulationResult:
    """Load a saved SdS scalar result without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        return SdSSimulationResult(
            rho=data["rho"],
            areal_radius=data["areal_radius"],
            signal_times=data["signal_times"],
            observer_rho=data["observer_rho"],
            observer_areal_radius=data["observer_areal_radius"],
            signals=data["signals"],
            snapshot_times=data["snapshot_times"],
            u_snapshots=data["u_snapshots"],
            constraint_linf=data["constraint_linf"],
            constraint_l2=data["constraint_l2"],
            metadata=json.loads(data["metadata"].item()),
        )


def _timestepper(name: str):
    steppers = {
        "RK111": d3.RK111,
        "RK222": d3.RK222,
        "RK443": d3.RK443,
        "SBDF2": d3.SBDF2,
        "SBDF4": d3.SBDF4,
    }
    try:
        return steppers[name]
    except KeyError as exc:
        raise ValueError(f"Unknown timestepper {name!r}; choose from {steppers}") from exc


def _run_scalar_simulation(
    model: SdSParameters | SchwarzschildScalarParameters,
    initial: ScalarInitialData | ArealBumpInitialData,
    numerical: SdSNumericalParameters,
    *,
    background: str,
    radius_function: Callable[[np.ndarray], np.ndarray],
    boost_function: Callable[[np.ndarray], np.ndarray],
    propagation_function: Callable[[np.ndarray], np.ndarray],
    potential_function: Callable[[np.ndarray], np.ndarray],
    initial_function: Callable[
        [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    horizon_metadata: dict,
) -> SdSSimulationResult:
    """Evolve the common first-order scalar system on one background."""

    started = time.perf_counter()
    dtype = np.float64

    rho_coord = d3.Coordinate("rho")
    dist = d3.Distributor(rho_coord, dtype=dtype)
    basis = d3.ChebyshevT(
        rho_coord,
        size=numerical.resolution,
        bounds=(0.0, 1.0),
        dealias=1,
    )
    rho = np.asarray(dist.local_grid(basis)).ravel()
    radius = radius_function(rho)

    u = dist.Field(name="u", bases=basis)
    psi = dist.Field(name="psi", bases=basis)
    pi = dist.Field(name="pi", bases=basis)
    coefficient_a = dist.Field(name="coefficient_a", bases=basis)
    coefficient_b = dist.Field(name="coefficient_b", bases=basis)
    potential = dist.Field(name="potential", bases=basis)

    drho = lambda field: d3.Differentiate(field, rho_coord)

    coefficient_a["g"] = propagation_function(rho)
    coefficient_b["g"] = boost_function(rho)
    potential["g"] = potential_function(rho)
    initial_u, initial_psi, initial_pi = initial_function(rho)
    u["g"] = initial_u
    psi["g"] = initial_psi
    pi["g"] = initial_pi
    if isinstance(initial, ArealBumpInitialData):
        # The analytic chain-rule values are used by the model validation.
        # In the finite Chebyshev representation, initialize the auxiliary
        # variable with the derivative of the represented u itself.  This is
        # the spectrally consistent realization of psi=(du/dr)(dr/d rho) and
        # prevents an avoidable O(truncation) first-order constraint at t=0.
        psi["g"] = np.asarray(drho(u).evaluate()["g"]).ravel()
        if initial.time_symmetric:
            pi["g"] = -np.asarray(coefficient_b["g"]).ravel() * psi["g"]

    problem = d3.IVP([u, psi, pi], namespace=locals())
    problem.add_equation(
        "dt(u) - coefficient_a * (coefficient_b * psi + pi) = 0"
    )
    problem.add_equation(
        "dt(psi) - drho(coefficient_a * (coefficient_b * psi + pi)) = 0"
    )
    problem.add_equation(
        "dt(pi) - drho(coefficient_a * (psi + coefficient_b * pi))"
        " + potential * u = 0"
    )

    solver = problem.build_solver(_timestepper(numerical.timestepper))
    solver.stop_sim_time = numerical.end_time

    observer_operators = [u(rho=point) for point in numerical.observers]
    observer_rho = np.asarray(numerical.observers, dtype=float)
    observer_radius = radius_function(observer_rho)
    constraint_operator = psi - drho(u)

    signal_stride = max(1, round(numerical.signal_dt / numerical.timestep))
    snapshot_stride = max(1, round(numerical.snapshot_dt / numerical.timestep))
    progress_stride = max(1, round(numerical.end_time / numerical.timestep / 10))

    signal_times: list[float] = []
    signals: list[list[float]] = []
    snapshot_times: list[float] = []
    u_snapshots: list[np.ndarray] = []
    constraint_linf: list[float] = []
    constraint_l2: list[float] = []

    def record_signal() -> None:
        values = [
            float(operator.evaluate()["g"].ravel()[0])
            for operator in observer_operators
        ]
        signal_times.append(float(solver.sim_time))
        signals.append(values)

    def record_snapshot() -> None:
        constraint = np.asarray(constraint_operator.evaluate()["g"]).ravel()
        snapshot_times.append(float(solver.sim_time))
        u_snapshots.append(np.asarray(u["g"]).ravel().copy())
        constraint_linf.append(float(np.max(np.abs(constraint))))
        constraint_l2.append(float(np.sqrt(np.mean(constraint**2))))

    record_signal()
    record_snapshot()

    number_of_steps = int(np.ceil(numerical.end_time / numerical.timestep))
    for step_number in range(1, number_of_steps + 1):
        if step_number < number_of_steps:
            step = numerical.timestep
        else:
            step = numerical.end_time - solver.sim_time
        solver.step(step)
        is_final = step_number == number_of_steps
        if solver.iteration % signal_stride == 0 or is_final:
            record_signal()
        if solver.iteration % snapshot_stride == 0 or is_final:
            record_snapshot()
        if solver.iteration % progress_stride == 0:
            LOGGER.info(
                "%s (%s): tau=%8.2f / %.2f, iteration=%d",
                numerical.bridge,
                background,
                solver.sim_time,
                numerical.end_time,
                solver.iteration,
            )

    elapsed = time.perf_counter() - started
    metadata = {
        "background": background,
        "model": model.as_dict(),
        "horizons": horizon_metadata,
        "initial_data": initial.as_dict(),
        "numerical": asdict(numerical),
        "iterations": solver.iteration,
        "final_time": solver.sim_time,
        "wall_seconds": elapsed,
        "equations": {
            "u": "dt(u) = A*(B*psi + pi)",
            "psi": "dt(psi) = d_rho[A*(B*psi + pi)]",
            "pi": "dt(pi) = d_rho[A*(psi + B*pi)] - P*u",
            "A": "(f*d rho/dr)/(1-B^2)",
            "B": "f*dh/dr",
            "P": "V_scalar/(f*d rho/dr)",
        },
        "initialization": {
            "psi": (
                "Chebyshev derivative D_rho of the represented common u(r); "
                "continuum identity psi=(du/dr)(dr/d rho)"
                if isinstance(initial, ArealBumpInitialData)
                else "analytic derivative of the rho-Gaussian"
            ),
            "pi": "-B*psi" if initial.time_symmetric else "constant amplitude",
        },
    }
    LOGGER.info(
        "finished %s: tau=%.6f, iterations=%d, wall=%.2fs, max constraint=%.3e",
        numerical.bridge,
        solver.sim_time,
        solver.iteration,
        elapsed,
        max(constraint_linf),
    )

    return SdSSimulationResult(
        rho=rho,
        areal_radius=radius,
        signal_times=np.asarray(signal_times),
        observer_rho=observer_rho,
        observer_areal_radius=observer_radius,
        signals=np.asarray(signals),
        snapshot_times=np.asarray(snapshot_times),
        u_snapshots=np.asarray(u_snapshots),
        constraint_linf=np.asarray(constraint_linf),
        constraint_l2=np.asarray(constraint_l2),
        metadata=metadata,
    )


def run_sds_simulation(
    model: SdSParameters,
    initial: ScalarInitialData | ArealBumpInitialData,
    numerical: SdSNumericalParameters,
) -> SdSSimulationResult:
    """Evolve the reduced scalar wave equation on an SdS bridge."""

    horizons = sds_horizons(model)
    bridge = numerical.bridge
    if isinstance(initial, ArealBumpInitialData):
        initial_function = lambda rho: scalar_areal_bump_initial_data(
            rho, model, initial, bridge
        )
    else:
        initial_function = lambda rho: scalar_gaussian_initial_data(
            rho, model, initial, bridge
        )
    return _run_scalar_simulation(
        model,
        initial,
        numerical,
        background="Schwarzschild-de Sitter",
        radius_function=lambda rho: areal_radius(rho, model),
        boost_function=lambda rho: bridge_boost(rho, model, bridge),
        propagation_function=lambda rho: propagation_coefficient(
            rho, model, bridge
        ),
        potential_function=lambda rho: rescaled_scalar_potential(rho, model),
        initial_function=initial_function,
        horizon_metadata=horizons.as_dict(),
    )


def run_schwarzschild_scalar_simulation(
    model: SchwarzschildScalarParameters,
    initial: ScalarInitialData | ArealBumpInitialData,
    numerical: SdSNumericalParameters,
) -> SdSSimulationResult:
    """Evolve the Lambda=0 scalar reference in Schwarzschild minimal gauge."""

    if numerical.bridge != "minimal":
        raise ValueError("The Schwarzschild reference uses only the minimal gauge.")
    if isinstance(initial, ArealBumpInitialData):
        initial_function = lambda rho: schwarzschild_areal_bump_initial_data(
            rho, model, initial
        )
    else:
        initial_function = lambda rho: schwarzschild_initial_data(
            rho, model, initial
        )
    return _run_scalar_simulation(
        model,
        initial,
        numerical,
        background="Schwarzschild",
        radius_function=lambda rho: schwarzschild_areal_radius(rho, model),
        boost_function=schwarzschild_minimal_boost,
        propagation_function=lambda rho: schwarzschild_propagation_coefficient(
            rho, model
        ),
        potential_function=lambda rho: schwarzschild_rescaled_scalar_potential(
            rho, model
        ),
        initial_function=initial_function,
        horizon_metadata={
            "black_hole": model.black_hole_horizon,
            "future_null_infinity": "rho=1 (r=infinity)",
        },
    )
