"""Dedalus solver for compactified Schwarzschild perturbations."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import dedalus.public as d3
import numpy as np

from .model import (
    InitialData,
    ModelParameters,
    boost,
    gaussian_initial_data,
    propagation_coefficient,
    rescaled_potential,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NumericalParameters:
    """Numerical settings for one evolution."""

    resolution: int = 384
    timestep: float = 0.02
    end_time: float = 1000.0
    signal_dt: float = 0.1
    snapshot_dt: float = 1.0
    observers: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 1.0)
    timestepper: str = "RK222"

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


@dataclass
class SimulationResult:
    """In-memory output from one evolution."""

    rho: np.ndarray
    signal_times: np.ndarray
    observer_rho: np.ndarray
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
            signal_times=self.signal_times,
            observer_rho=self.observer_rho,
            signals=self.signals,
            snapshot_times=self.snapshot_times,
            u_snapshots=self.u_snapshots,
            constraint_linf=self.constraint_linf,
            constraint_l2=self.constraint_l2,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
        )


def load_result(path: Path) -> SimulationResult:
    """Load a saved result without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        return SimulationResult(
            rho=data["rho"],
            signal_times=data["signal_times"],
            observer_rho=data["observer_rho"],
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


def run_simulation(
    model: ModelParameters,
    initial: InitialData,
    numerical: NumericalParameters,
) -> SimulationResult:
    """Evolve the first-order hyperboloidal Regge--Wheeler system."""

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

    u = dist.Field(name="u", bases=basis)
    psi = dist.Field(name="psi", bases=basis)
    pi = dist.Field(name="pi", bases=basis)
    coefficient_a = dist.Field(name="coefficient_a", bases=basis)
    coefficient_h = dist.Field(name="coefficient_h", bases=basis)
    potential = dist.Field(name="potential", bases=basis)

    coefficient_a["g"] = propagation_coefficient(rho, model)
    coefficient_h["g"] = boost(rho)
    potential["g"] = rescaled_potential(rho, model)
    u["g"], psi["g"], pi["g"] = gaussian_initial_data(rho, initial)

    drho = lambda field: d3.Differentiate(field, rho_coord)

    problem = d3.IVP([u, psi, pi], namespace=locals())
    problem.add_equation(
        "dt(u) - coefficient_a * (coefficient_h * psi + pi) = 0"
    )
    problem.add_equation(
        "dt(psi) - drho(coefficient_a * (coefficient_h * psi + pi)) = 0"
    )
    # The potential has a minus sign in the evolution equation:
    # pi_t = d_rho[A(psi + H pi)] - potential*u.
    problem.add_equation(
        "dt(pi) - drho(coefficient_a * (psi + coefficient_h * pi))"
        " + potential * u = 0"
    )

    solver = problem.build_solver(_timestepper(numerical.timestepper))
    solver.stop_sim_time = numerical.end_time

    observer_operators = [u(rho=point) for point in numerical.observers]
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
                "progress: tau=%8.2f / %.2f, iteration=%d",
                solver.sim_time,
                numerical.end_time,
                solver.iteration,
            )

    elapsed = time.perf_counter() - started
    metadata = {
        "model": model.as_dict(),
        "initial_data": initial.as_dict(),
        "numerical": asdict(numerical),
        "iterations": solver.iteration,
        "final_time": solver.sim_time,
        "wall_seconds": elapsed,
        "equations": {
            "u": "dt(u) = A*(H*psi + pi)",
            "psi": "dt(psi) = d_rho[A*(H*psi + pi)]",
            "pi": "dt(pi) = d_rho[A*(psi + H*pi)] - P*u",
            "A": "1/[8*M*(2-rho)]",
            "H": "-1 + 2*(1-rho)^2",
            "P": "[ell*(ell+1)-3*(1-rho)]/(2*M)",
        },
    }
    LOGGER.info(
        "finished: tau=%.6f, iterations=%d, wall=%.2fs, max constraint=%.3e",
        solver.sim_time,
        solver.iteration,
        elapsed,
        max(constraint_linf),
    )

    return SimulationResult(
        rho=rho,
        signal_times=np.asarray(signal_times),
        observer_rho=np.asarray(numerical.observers),
        signals=np.asarray(signals),
        snapshot_times=np.asarray(snapshot_times),
        u_snapshots=np.asarray(u_snapshots),
        constraint_linf=np.asarray(constraint_linf),
        constraint_l2=np.asarray(constraint_l2),
        metadata=metadata,
    )
