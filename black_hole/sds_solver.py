"""Dedalus solver for scalar waves on Schwarzschild-de Sitter bridges."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import dedalus.public as d3
import numpy as np

from .exterior_sds_model import (
    ExteriorSdSParameters,
    areal_radius as exterior_areal_radius,
    bridge_boost as exterior_bridge_boost,
    bridge_one_plus_boost as exterior_bridge_one_plus_boost,
    propagation_coefficient as exterior_propagation_coefficient,
    rescaled_scalar_potential as exterior_rescaled_scalar_potential,
    scalar_areal_bump_initial_data as exterior_scalar_areal_bump_initial_data,
    scalar_areal_velocity_initial_data as exterior_scalar_areal_velocity_initial_data,
    scalar_gaussian_initial_data as exterior_scalar_gaussian_initial_data,
    transition_compact_radii as exterior_transition_compact_radii,
    transition_radii as exterior_transition_radii,
)
from .sds_model import (
    ArealBumpInitialData,
    ArealVelocityBumpInitialData,
    BRIDGE_CHOICES,
    ScalarInitialData,
    SdSParameters,
    areal_radius,
    bridge_boost,
    propagation_coefficient,
    rescaled_scalar_potential,
    scalar_gaussian_initial_data,
    scalar_areal_bump_initial_data,
    scalar_areal_velocity_initial_data,
    sds_horizons,
)
from .sds_result import SdSSimulationResult
from .schwarzschild_scalar import (
    SchwarzschildScalarParameters,
    areal_radius as schwarzschild_areal_radius,
    minimal_boost as schwarzschild_minimal_boost,
    propagation_coefficient as schwarzschild_propagation_coefficient,
    rescaled_scalar_potential as schwarzschild_rescaled_scalar_potential,
    scalar_gaussian_initial_data as schwarzschild_initial_data,
    scalar_areal_bump_initial_data as schwarzschild_areal_bump_initial_data,
    scalar_areal_velocity_initial_data as schwarzschild_areal_velocity_initial_data,
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
    dealias: float = 1.5

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
        if not np.isfinite(self.dealias) or self.dealias < 1.0:
            raise ValueError("The spectral dealias factor must be at least one.")


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
    model: SdSParameters | SchwarzschildScalarParameters | ExteriorSdSParameters,
    initial: ScalarInitialData | ArealBumpInitialData | ArealVelocityBumpInitialData,
    numerical: SdSNumericalParameters,
    *,
    background: str,
    radius_function: Callable[[np.ndarray], np.ndarray],
    boost_function: Callable[[np.ndarray], np.ndarray],
    propagation_function: Callable[[np.ndarray], np.ndarray],
    potential_function: Callable[[np.ndarray], np.ndarray],
    one_plus_boost_function: Callable[[np.ndarray], np.ndarray] | None = None,
    initial_function: Callable[
        [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    horizon_metadata: dict,
    checkpoint_path: Path | None = None,
    checkpoint_dt: float | None = None,
    explicit_potential: bool = False,
) -> SdSSimulationResult:
    """Evolve the common first-order scalar system on one background.

    ``explicit_potential`` selects which side of the IMEX split carries the
    zeroth-order term ``P*u``.  Both choices discretize the same continuum
    system.  The default keeps that term implicit, reproducing every archived
    production run exactly.  Moving it to the explicit right-hand side is
    numerically preferable on Schwarzschild-de Sitter bridges: ``P`` is
    bounded and of order unity on every background used here, so it is not
    stiff, yet its Chebyshev spectrum is broad at large ``L`` (bandwidth 301 at
    ``L/M=5120`` against 2 on Schwarzschild).  Kept implicit it turns the banded
    subproblem matrices nearly dense, which costs an order of magnitude in both
    matrix construction and per-step solve without buying any stability.
    """

    started = time.perf_counter()
    dtype = np.float64

    rho_coord = d3.Coordinate("rho")
    dist = d3.Distributor(rho_coord, dtype=dtype)
    basis = d3.ChebyshevT(
        rho_coord,
        size=numerical.resolution,
        bounds=(0.0, 1.0),
        dealias=numerical.dealias,
    )
    rho = np.asarray(dist.local_grid(basis)).ravel()
    radius = radius_function(rho)

    u = dist.Field(name="u", bases=basis)
    psi = dist.Field(name="psi", bases=basis)
    pi = dist.Field(name="pi", bases=basis)
    coefficient_a = dist.Field(name="coefficient_a", bases=basis)
    coefficient_b = dist.Field(name="coefficient_b", bases=basis)
    coefficient_cplus = dist.Field(name="coefficient_cplus", bases=basis)
    potential = dist.Field(name="potential", bases=basis)

    drho = lambda field: d3.Differentiate(field, rho_coord)

    coefficient_a["g"] = propagation_function(rho)
    coefficient_b["g"] = boost_function(rho)
    if one_plus_boost_function is not None:
        coefficient_cplus["g"] = one_plus_boost_function(rho)
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
        represented_derivative = drho(u).evaluate()
        represented_derivative.change_scales(1)
        psi["g"] = np.asarray(represented_derivative["g"]).ravel()
        if initial.time_symmetric:
            pi["g"] = -np.asarray(coefficient_b["g"]).ravel() * psi["g"]

    problem = d3.IVP([u, psi, pi], namespace=locals())
    if one_plus_boost_function is None:
        outgoing_flux = "coefficient_b * psi + pi"
        ingoing_flux = "psi + coefficient_b * pi"
    else:
        # B=-1+Cplus.  These exactly equivalent fluxes avoid subtracting the
        # horizon-scale remainder Cplus from one inside a spectral product.
        outgoing_flux = "(pi - psi) + coefficient_cplus * psi"
        ingoing_flux = "(psi - pi) + coefficient_cplus * pi"
    problem.add_equation(
        f"dt(u) - coefficient_a * ({outgoing_flux}) = 0"
    )
    problem.add_equation(
        f"dt(psi) - drho(coefficient_a * ({outgoing_flux})) = 0"
    )
    if explicit_potential:
        problem.add_equation(
            f"dt(pi) - drho(coefficient_a * ({ingoing_flux}))"
            " = -potential * u"
        )
    else:
        problem.add_equation(
            f"dt(pi) - drho(coefficient_a * ({ingoing_flux}))"
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

    configuration = json.dumps(
        {
            "background": background,
            "model": model.as_dict(),
            "initial_data": initial.as_dict(),
            "numerical": asdict(numerical),
            # Recorded only when set, so checkpoints written by the default
            # implicit split stay byte-identical and remain resumable.
            **({"imex_split": "explicit_potential"} if explicit_potential else {}),
        },
        sort_keys=True,
    )
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    if (checkpoint is None) != (checkpoint_dt is None):
        raise ValueError("checkpoint_path and checkpoint_dt must be supplied together.")
    if checkpoint_dt is not None and checkpoint_dt <= 0.0:
        raise ValueError("checkpoint_dt must be positive.")

    elapsed_before_restart = 0.0
    resumed_from_checkpoint = False

    def save_checkpoint(elapsed_wall_seconds: float) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                configuration=np.array(configuration),
                sim_time=np.array(solver.sim_time),
                iteration=np.array(solver.iteration),
                elapsed_wall_seconds=np.array(elapsed_wall_seconds),
                state_scales=np.asarray(u.scales),
                u=np.asarray(u["g"]).ravel(),
                psi=np.asarray(psi["g"]).ravel(),
                pi=np.asarray(pi["g"]).ravel(),
                signal_times=np.asarray(signal_times),
                signals=np.asarray(signals),
                snapshot_times=np.asarray(snapshot_times),
                u_snapshots=np.asarray(u_snapshots),
                constraint_linf=np.asarray(constraint_linf),
                constraint_l2=np.asarray(constraint_l2),
            )
        os.replace(temporary, checkpoint)

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

    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            saved_configuration = saved["configuration"].item()
            if saved_configuration != configuration:
                raise ValueError(
                    f"Checkpoint configuration does not match this run: {checkpoint}"
                )
            state_scales = tuple(saved["state_scales"].tolist())
            for field in (u, psi, pi):
                field.change_scales(state_scales)
            u["g"] = saved["u"]
            psi["g"] = saved["psi"]
            pi["g"] = saved["pi"]
            solver.sim_time = solver.initial_sim_time = float(saved["sim_time"])
            solver.iteration = solver.initial_iteration = int(saved["iteration"])
            elapsed_before_restart = float(saved["elapsed_wall_seconds"])
            signal_times.extend(saved["signal_times"].tolist())
            signals.extend(saved["signals"].tolist())
            snapshot_times.extend(saved["snapshot_times"].tolist())
            u_snapshots.extend(saved["u_snapshots"].tolist())
            constraint_linf.extend(saved["constraint_linf"].tolist())
            constraint_l2.extend(saved["constraint_l2"].tolist())
        resumed_from_checkpoint = True
        LOGGER.info(
            "resumed %s from tau=%.6f, iteration=%d",
            checkpoint,
            solver.sim_time,
            solver.iteration,
        )
    else:
        record_signal()
        record_snapshot()
        save_checkpoint(time.perf_counter() - started)

    if checkpoint_dt is None:
        next_checkpoint_time = np.inf
    else:
        next_checkpoint_time = (
            np.floor(solver.sim_time / checkpoint_dt) + 1.0
        ) * checkpoint_dt

    tolerance = 32.0 * np.finfo(float).eps * max(1.0, numerical.end_time)
    while solver.sim_time < numerical.end_time - tolerance:
        step = min(numerical.timestep, numerical.end_time - solver.sim_time)
        solver.step(step)
        is_final = solver.sim_time >= numerical.end_time - tolerance
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

        if solver.sim_time >= next_checkpoint_time - tolerance:
            elapsed_now = elapsed_before_restart + time.perf_counter() - started
            save_checkpoint(elapsed_now)
            while next_checkpoint_time <= solver.sim_time + tolerance:
                next_checkpoint_time += checkpoint_dt

    elapsed = elapsed_before_restart + time.perf_counter() - started
    metadata = {
        "background": background,
        "model": model.as_dict(),
        "horizons": horizon_metadata,
        "initial_data": initial.as_dict(),
        "numerical": asdict(numerical),
        "iterations": solver.iteration,
        "final_time": solver.sim_time,
        "wall_seconds": elapsed,
        "checkpoint_restart": {
            "enabled": checkpoint is not None,
            "interval_sim_time": checkpoint_dt,
            "resumed": resumed_from_checkpoint,
        },
        "equations": {
            "u": f"dt(u) = A*({outgoing_flux})",
            "psi": f"dt(psi) = d_rho[A*({outgoing_flux})]",
            "pi": f"dt(pi) = d_rho[A*({ingoing_flux})] - P*u",
            "A": "(f*d rho/dr)/(1-B^2)",
            "B": "f*dh/dr",
            "P": "V_scalar/(f*d rho/dr)",
        },
        "imex_split": {
            "potential_term": "explicit" if explicit_potential else "implicit",
            "transport_terms": "implicit",
            "note": (
                "The stiffness of the Chebyshev discretization resides in the "
                "transport terms, which are implicit in both variants. P is "
                "bounded and of order unity, and therefore non-stiff."
            ),
        },
        "initialization": {
            "psi": (
                "identically zero"
                if isinstance(initial, ArealVelocityBumpInitialData)
                else (
                    "Chebyshev derivative D_rho of the represented common u(r); "
                    "continuum identity psi=(du/dr)(dr/d rho)"
                    if isinstance(initial, ArealBumpInitialData)
                    else "analytic derivative of the rho-Gaussian"
                )
            ),
            "pi": (
                "G(r)/A(r), giving partial_tau u=G(r)"
                if isinstance(initial, ArealVelocityBumpInitialData)
                else "-B*psi"
                if initial.time_symmetric
                else "constant amplitude"
            ),
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
    initial: ScalarInitialData | ArealBumpInitialData | ArealVelocityBumpInitialData,
    numerical: SdSNumericalParameters,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_dt: float | None = None,
    explicit_potential: bool = False,
) -> SdSSimulationResult:
    """Evolve the reduced scalar wave equation on an SdS bridge."""

    horizons = sds_horizons(model)
    bridge = numerical.bridge
    if isinstance(initial, ArealVelocityBumpInitialData):
        initial_function = lambda rho: scalar_areal_velocity_initial_data(
            rho, model, initial, bridge
        )
    elif isinstance(initial, ArealBumpInitialData):
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
        checkpoint_path=checkpoint_path,
        checkpoint_dt=checkpoint_dt,
        explicit_potential=explicit_potential,
    )


def run_schwarzschild_scalar_simulation(
    model: SchwarzschildScalarParameters,
    initial: ScalarInitialData | ArealBumpInitialData | ArealVelocityBumpInitialData,
    numerical: SdSNumericalParameters,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_dt: float | None = None,
    explicit_potential: bool = False,
) -> SdSSimulationResult:
    """Evolve the Lambda=0 scalar reference in Schwarzschild minimal gauge."""

    if numerical.bridge != "minimal":
        raise ValueError("The Schwarzschild reference uses only the minimal gauge.")
    if isinstance(initial, ArealVelocityBumpInitialData):
        initial_function = lambda rho: schwarzschild_areal_velocity_initial_data(
            rho, model, initial
        )
    elif isinstance(initial, ArealBumpInitialData):
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
        checkpoint_path=checkpoint_path,
        checkpoint_dt=checkpoint_dt,
        explicit_potential=explicit_potential,
    )


def run_exterior_sds_simulation(
    model: ExteriorSdSParameters,
    initial: ScalarInitialData | ArealBumpInitialData | ArealVelocityBumpInitialData,
    numerical: SdSNumericalParameters,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_dt: float | None = None,
    explicit_potential: bool = False,
) -> SdSSimulationResult:
    """Evolve a scalar mode on the exterior-supported SdS background."""

    if numerical.bridge != "minimal":
        raise ValueError("The exterior-supported construction uses the minimal gauge.")
    if isinstance(initial, ArealVelocityBumpInitialData):
        initial_function = lambda rho: exterior_scalar_areal_velocity_initial_data(
            rho, model, initial
        )
    elif isinstance(initial, ArealBumpInitialData):
        initial_function = lambda rho: exterior_scalar_areal_bump_initial_data(
            rho, model, initial
        )
    else:
        initial_function = lambda rho: exterior_scalar_gaussian_initial_data(
            rho, model, initial
        )
    transition_r0, transition_r1 = exterior_transition_radii(model)
    transition_rho0, transition_rho1 = exterior_transition_compact_radii(model)
    return _run_scalar_simulation(
        model,
        initial,
        numerical,
        background="exterior-supported Schwarzschild-de Sitter",
        radius_function=lambda rho: exterior_areal_radius(rho, model),
        boost_function=lambda rho: exterior_bridge_boost(rho, model),
        propagation_function=lambda rho: exterior_propagation_coefficient(rho, model),
        potential_function=lambda rho: exterior_rescaled_scalar_potential(rho, model),
        one_plus_boost_function=lambda rho: exterior_bridge_one_plus_boost(
            rho, model
        ),
        initial_function=initial_function,
        horizon_metadata={
            "black_hole": model.black_hole_horizon,
            "cosmological": model.cosmological_horizon,
            "transition_inner_radius": transition_r0,
            "transition_outer_radius": transition_r1,
            "transition_inner_rho": transition_rho0,
            "transition_outer_rho": transition_rho1,
        },
        checkpoint_path=checkpoint_path,
        checkpoint_dt=checkpoint_dt,
        explicit_potential=explicit_potential,
    )
