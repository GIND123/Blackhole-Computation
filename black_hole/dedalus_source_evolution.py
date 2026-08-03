r"""Dedalus backend for the localized-source Green-function evolution.

The finite-difference implementation in :mod:`black_hole.source_evolution`
remains the production calculation and an independent discretization.  This
module solves the same modal equations with Dedalus 3 and Chebyshev radial
fields, providing a direct spectral cross-check and a path to larger angular
catalogues.

On a spherically symmetric background all real spherical harmonics with the
same ``ell`` obey the same radial equation.  Their zero-data solutions differ
only by the constant source projection
``g_ell Y^R_ell,m(theta_s, phi_s)``.  Dedalus therefore evolves one response
for each distinct ``ell`` and reconstructs every excited ``m`` coefficient
exactly when diagnostics are recorded.  This is equivalent to evolving every
``(ell, m)`` field separately while avoiding a large redundant state vector.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

import numpy as np

try:
    import dedalus.public as d3
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise ModuleNotFoundError(
        "The Dedalus Green-function backend requires Dedalus 3. Create the "
        "pinned environment with `conda env create -f environment.yml` and "
        "run this command from that environment."
    ) from exc

from .localized_source import (
    LocalizedSourceParameters,
    build_mode_catalogue,
    radial_profile,
    time_profile,
    verify_angular_expansion,
)
from .sds_model import sds_horizons
from .source_evolution import (
    REFERENCE_RADIUS,
    SourcedNumericalParameters,
    SourcedSimulationResult,
    _Background,
    _cadence_stride,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["run_sourced_dedalus_simulation"]


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
        choices = ", ".join(steppers)
        raise ValueError(
            f"Unknown Dedalus timestepper {name!r}; choose from {choices}."
        ) from exc


def run_sourced_dedalus_simulation(
    *,
    background: str,
    source: LocalizedSourceParameters,
    numerical: SourcedNumericalParameters,
    cosmological_length: float = 80.0,
    mass: float = 1.0,
    timestepper: str = "RK443",
    dealias: float = 1.5,
) -> SourcedSimulationResult:
    """Evolve ``Box Phi = S`` with Dedalus Chebyshev radial fields.

    The returned archive has exactly the same layout as
    :func:`black_hole.source_evolution.run_sourced_simulation`, so all angular
    reconstruction and caustic-analysis routines work unchanged.
    """

    if background not in {"sds", "schwarzschild"}:
        raise ValueError("background must be 'sds' or 'schwarzschild'.")
    if not np.isfinite(dealias) or dealias < 1.0:
        raise ValueError("The Dedalus dealias factor must be at least one.")
    started = time.perf_counter()
    dtype = np.float64

    rho_coord = d3.Coordinate("rho")
    dist = d3.Distributor(rho_coord, dtype=dtype)
    basis = d3.ChebyshevT(
        rho_coord,
        size=numerical.radial_resolution,
        bounds=(0.0, 1.0),
        dealias=dealias,
    )
    rho = np.asarray(dist.local_grid(basis)).ravel()
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

    # Dedalus evaluates nonlinear variable-coefficient products on the
    # dealiased grid.  GeneralFunction must therefore return source data on
    # that same layout, rather than on the coefficient-resolution grid used
    # for saved fields and diagnostics.
    source_rho = np.asarray(dist.local_grid(basis, scale=dealias)).ravel()
    source_geometry = _Background(
        background, source_rho, catalogue, mass, cosmological_length
    )
    source_radius_grid = source_geometry.areal_radius
    profile = radial_profile(source_radius_grid, source)
    support = np.flatnonzero(profile > 0.0)
    if support.size < 8:
        raise ValueError(
            "The emitter is resolved by fewer than eight Chebyshev points; "
            "increase the resolution or the support half-width."
        )
    window = slice(int(support[0]), int(support[-1]) + 1)
    support_radius = source_radius_grid[window]
    support_height = source_geometry.height(support_radius)
    source_kernel = (
        -source.amplitude
        * profile[window]
        * support_radius
        / source_geometry.compactification_derivative(support_radius)
    )
    earliest_tau = float(
        np.min(source.killing_time_support[0] + support_height)
    )
    if earliest_tau <= 0.0:
        raise ValueError(
            "The emitter would already be active at tau=0; increase "
            f"time_center (earliest activation tau={earliest_tau:.3f})."
        )
    latest_tau = float(np.max(source.killing_time_support[1] + support_height))

    observer_rho = np.asarray(
        [geometry.observer_coordinate(value) for value in numerical.observer_radii]
    )
    observer_radius = np.asarray(
        [
            geometry.outer_radius if value is None else float(value)
            for value in numerical.observer_radii
        ]
    )

    distinct_ells = np.unique(catalogue.ell)
    ell_to_response = {int(ell): index for index, ell in enumerate(distinct_ells)}
    mode_response_index = np.asarray(
        [ell_to_response[int(ell)] for ell in catalogue.ell], dtype=int
    )

    coefficient_a = dist.Field(name="coefficient_a", bases=basis)
    coefficient_b = dist.Field(name="coefficient_b", bases=basis)
    coefficient_a["g"] = geometry.coefficient_a
    coefficient_b["g"] = geometry.boost
    potential_fields = []
    for index, ell in enumerate(distinct_ells):
        potential = dist.Field(name=f"potential_l{int(ell)}", bases=basis)
        potential["g"] = geometry.potential[index]
        potential_fields.append(potential)

    u_fields = [
        dist.Field(name=f"u_l{int(ell)}", bases=basis) for ell in distinct_ells
    ]
    psi_fields = [
        dist.Field(name=f"psi_l{int(ell)}", bases=basis) for ell in distinct_ells
    ]
    pi_fields = [
        dist.Field(name=f"pi_l{int(ell)}", bases=basis) for ell in distinct_ells
    ]
    variables = [*u_fields, *psi_fields, *pi_fields]
    tau = dist.Field(name="tau")
    drho = lambda field: d3.Differentiate(field, rho_coord)

    problem = d3.IVP(variables, time=tau, namespace=locals())

    def staged_source(time_field) -> np.ndarray:
        """Evaluate ``-(r/G) S(tau-h(r),r)`` at an RK stage."""

        current_time = float(np.asarray(time_field["g"]).ravel()[0])
        values = np.zeros_like(source_rho)
        if earliest_tau <= current_time <= latest_tau:
            values[window] = source_kernel * time_profile(
                current_time - support_height, source
            )
        return values

    source_operator = d3.GeneralFunction(
        dist,
        u_fields[0].domain,
        tensorsig=(),
        dtype=dtype,
        layout="g",
        func=staged_source,
        args=[tau],
    )
    for u, psi, pi, potential in zip(
        u_fields, psi_fields, pi_fields, potential_fields
    ):
        velocity = coefficient_a * (coefficient_b * psi + pi)
        flux = coefficient_a * (psi + coefficient_b * pi)
        problem.add_equation((d3.dt(u) - velocity, 0))
        problem.add_equation((d3.dt(psi) - drho(velocity), 0))
        problem.add_equation(
            (d3.dt(pi) - drho(flux) + potential * u, source_operator)
        )

    solver = problem.build_solver(_timestepper(timestepper))
    solver.stop_sim_time = numerical.end_time

    observer_operators = [
        [u(rho=point) for u in u_fields] for point in observer_rho
    ]
    constraint_operators = [
        psi - drho(u) for u, psi in zip(u_fields, psi_fields)
    ]
    snapshot_indices = np.unique(
        np.linspace(
            0,
            numerical.radial_resolution - 1,
            numerical.snapshot_radial_points,
        ).astype(int)
    )

    signal_stride = _cadence_stride(numerical.signal_dt, numerical.timestep)
    diagnostic_stride = _cadence_stride(
        numerical.diagnostic_dt, numerical.timestep
    )
    snapshot_stride = _cadence_stride(numerical.snapshot_dt, numerical.timestep)
    total_steps = int(np.ceil(numerical.end_time / numerical.timestep))
    progress_stride = max(1, total_steps // 12)

    signal_times: list[float] = []
    modal_signals: list[np.ndarray] = []
    diagnostic_times: list[float] = []
    constraint_linf: list[float] = []
    constraint_l2: list[float] = []
    field_linf: list[float] = []
    activity: list[float] = []
    snapshot_times: list[float] = []
    modal_snapshots: list[np.ndarray] = []

    def response_grid() -> np.ndarray:
        return np.stack([np.asarray(field["g"]).ravel() for field in u_fields])

    def source_strength(current_time: float) -> float:
        if not earliest_tau <= current_time <= latest_tau:
            return 0.0
        return float(
            np.max(np.abs(time_profile(current_time - support_height, source)))
        )

    def record_signal() -> None:
        responses = np.asarray(
            [
                [float(operator.evaluate()["g"].ravel()[0]) for operator in row]
                for row in observer_operators
            ]
        )
        modal = (
            responses[:, mode_response_index]
            * catalogue.amplitude[np.newaxis, :]
        )
        signal_times.append(float(solver.sim_time))
        modal_signals.append(modal)

    def record_diagnostics() -> None:
        response_constraints = np.stack(
            [
                np.asarray(operator.evaluate()["g"]).ravel()
                for operator in constraint_operators
            ]
        )
        modal_constraints = (
            response_constraints[mode_response_index]
            * catalogue.amplitude[:, np.newaxis]
        )
        modal_u = (
            response_grid()[mode_response_index]
            * catalogue.amplitude[:, np.newaxis]
        )
        diagnostic_times.append(float(solver.sim_time))
        constraint_linf.append(float(np.max(np.abs(modal_constraints))))
        constraint_l2.append(float(np.sqrt(np.mean(modal_constraints**2))))
        field_linf.append(float(np.max(np.abs(modal_u))))
        activity.append(source_strength(float(solver.sim_time)))

    def record_snapshot() -> None:
        response = response_grid()[:, snapshot_indices]
        modal = (
            response[mode_response_index]
            * catalogue.amplitude[:, np.newaxis]
        )
        snapshot_times.append(float(solver.sim_time))
        modal_snapshots.append(modal)

    record_signal()
    record_diagnostics()
    record_snapshot()
    for step_number in range(1, total_steps + 1):
        step = (
            numerical.timestep
            if step_number < total_steps
            else numerical.end_time - solver.sim_time
        )
        solver.step(step)
        is_final = step_number == total_steps
        if solver.iteration % signal_stride == 0 or is_final:
            record_signal()
        if solver.iteration % diagnostic_stride == 0 or is_final:
            record_diagnostics()
        if (
            solver.sim_time <= numerical.snapshot_end_time
            and solver.iteration % snapshot_stride == 0
        ) or is_final:
            record_snapshot()
        if solver.iteration % progress_stride == 0:
            LOGGER.info(
                "Dedalus sourced %s: tau=%9.2f / %.2f, max|u|=%.3e",
                geometry.label,
                solver.sim_time,
                numerical.end_time,
                max(float(np.max(np.abs(field["g"]))) for field in u_fields),
            )
        if not all(np.isfinite(field["g"]).all() for field in variables):
            raise FloatingPointError(
                "The Dedalus sourced evolution lost finiteness at "
                f"tau={solver.sim_time:.4f}."
            )

    elapsed = time.perf_counter() - started
    coefficient_speeds = np.maximum(
        np.abs(-geometry.coefficient_a * (1.0 + geometry.boost)),
        np.abs(geometry.coefficient_a * (1.0 - geometry.boost)),
    )
    metadata = {
        "background": geometry.label,
        "background_key": geometry.key,
        "model": geometry.model_dict,
        "horizons": geometry.horizons,
        "cosmological_length": float(cosmological_length),
        "surface_gravity_cosmological": float(geometry.surface_gravity),
        "source": source.as_dict(),
        "source_modes": catalogue.as_dict(),
        "angular_expansion_check": verify_angular_expansion(
            source, numerical.angular_ell_max
        ),
        "source_support": {
            "radial_points": int(support.size),
            "rho_window": [
                float(source_rho[window][0]),
                float(source_rho[window][-1]),
            ],
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
            "backend": "Dedalus 3",
            "coordinate": "ChebyshevT rho in [0,1]",
            "resolution": int(numerical.radial_resolution),
            "dealias": float(dealias),
            "time_stepper": timestepper,
            "source_evaluation": "GeneralFunction evaluated at every RK stage",
            "maximum_characteristic_speed": float(np.max(coefficient_speeds)),
        },
        "angular_reduction": {
            "evolved_responses": int(distinct_ells.size),
            "reconstructed_modes": int(catalogue.count),
            "identity": (
                "u_lm = g_l Y_lm(theta_s,phi_s) u_l for zero data on a "
                "spherically symmetric background"
            ),
        },
        "equations": {
            "field": "Phi=r^{-1} sum_lm u_lm Y^R_lm, Box Phi = S",
            "u": "dt(u)=A*(B*psi+pi)",
            "psi": "dt(psi)=d_rho[A*(B*psi+pi)]",
            "pi": (
                "dt(pi)=d_rho[A*(psi+B*pi)]-P_ell*u"
                "-(r/G)*S_lm(tau-h_L(r),r)"
            ),
        },
        "retarded_time_offset": {
            "q": float(geometry.offset),
            "reference_radius": REFERENCE_RADIUS,
            "evaluation": "analytic",
        },
        "iterations": int(solver.iteration),
        "final_time": float(solver.sim_time),
        "wall_seconds": elapsed,
    }
    LOGGER.info(
        "finished Dedalus sourced %s: ell responses=%d, modes=%d, "
        "wall=%.1fs, max constraint=%.3e",
        geometry.label,
        distinct_ells.size,
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
        signal_times=np.asarray(signal_times),
        observer_rho=observer_rho,
        observer_areal_radius=observer_radius,
        modal_signals=np.asarray(modal_signals),
        diagnostic_times=np.asarray(diagnostic_times),
        constraint_linf=np.asarray(constraint_linf),
        constraint_l2=np.asarray(constraint_l2),
        field_linf=np.asarray(field_linf),
        source_activity=np.asarray(activity),
        snapshot_times=np.asarray(snapshot_times),
        snapshot_rho=rho[snapshot_indices],
        snapshot_areal_radius=radius[snapshot_indices],
        modal_snapshots=np.asarray(modal_snapshots),
        metadata=metadata,
    )
