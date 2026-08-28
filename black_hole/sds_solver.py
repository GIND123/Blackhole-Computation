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
from scipy.fft import dct

from .exterior_sds_model import (
    ExteriorSdSParameters,
    areal_radius as exterior_areal_radius,
    bridge_boost as exterior_bridge_boost,
    bridge_one_minus_boost as exterior_bridge_one_minus_boost,
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
NCC_CUTOFF = 1.0e-6
ENTRY_CUTOFF = 1.0e-12


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


def _chebyshev_tail_ratio(values: np.ndarray, tail_count: int = 16) -> float:
    """Return the relative tail of Chebyshev coefficients on Gauss nodes."""

    values = np.asarray(values, dtype=float).ravel()
    if values.size < 4:
        raise ValueError("At least four Gauss values are required.")
    # Dedalus stores this basis from rho=0 to rho=1, opposite to the usual
    # descending cosine-node order expected by a type-II DCT.
    coefficients = dct(values[::-1], type=2) / values.size
    coefficients[0] *= 0.5
    scale = float(np.max(np.abs(coefficients)))
    if scale == 0.0:
        return 0.0
    count = min(int(tail_count), values.size)
    return float(np.max(np.abs(coefficients[-count:])) / scale)


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
    one_minus_boost_function: Callable[[np.ndarray], np.ndarray] | None = None,
    initial_function: Callable[
        [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    horizon_metadata: dict,
    checkpoint_path: Path | None = None,
    checkpoint_dt: float | None = None,
    explicit_potential: bool = False,
    endpoint_factored_characteristic_variables: bool = False,
    conservative_characteristic_variables: bool = False,
    characteristic_constraint_damping: float = 0.0,
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

    ``endpoint_factored_characteristic_variables`` evolves ``u`` together
    with ``H`` and ``J``, defined by

    ``A(1+B)(pi+psi)=(1-rho)H`` and
    ``A(1-B)(pi-psi)=rho J``.

    The factors impose the two regularity conditions at the null boundaries
    analytically.  Cancellation-free functions for both ``1+B`` and ``1-B``
    are required in this mode.  ``characteristic_constraint_damping`` damps
    the redundant compatibility constraint through the incoming
    characteristic sector without changing a continuum scalar solution.  Its
    default of zero leaves the reduction undamped.

    ``conservative_characteristic_variables`` instead evolves
    ``h=pi+psi`` and ``j=pi-psi``.  Their fluxes
    ``Fplus=A(1+B)h`` and ``Fminus=A(1-B)j`` are reused literally in the
    ``u``, ``h``, and ``j`` equations.  Consequently the semidiscrete
    compatibility constraint ``(h-j)/2-d_rho(u)`` is preserved without
    relying on a discrete product rule or on separately truncated
    coefficient identities.  This is the preferred exterior-tail
    formulation; the endpoint-factored system is retained as an independent
    damped check.
    """

    started = time.perf_counter()
    dtype = np.float64

    if (
        not np.isfinite(characteristic_constraint_damping)
        or characteristic_constraint_damping < 0.0
    ):
        raise ValueError(
            "Characteristic constraint damping must be finite and nonnegative."
        )
    if (
        characteristic_constraint_damping != 0.0
        and not endpoint_factored_characteristic_variables
    ):
        raise ValueError(
            "Characteristic constraint damping requires endpoint-factored "
            "characteristic variables."
        )
    if (
        endpoint_factored_characteristic_variables
        and conservative_characteristic_variables
    ):
        raise ValueError(
            "Choose either endpoint-factored or conservative characteristic "
            "variables, not both."
        )

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
    coefficient_a = dist.Field(name="coefficient_a", bases=basis)
    coefficient_b = dist.Field(name="coefficient_b", bases=basis)
    coefficient_cplus = dist.Field(name="coefficient_cplus", bases=basis)
    coefficient_cminus = dist.Field(name="coefficient_cminus", bases=basis)
    coefficient_rho = dist.Field(name="coefficient_rho", bases=basis)
    coefficient_one_minus_rho = dist.Field(
        name="coefficient_one_minus_rho", bases=basis
    )
    coefficient_vplus = dist.Field(name="coefficient_vplus", bases=basis)
    coefficient_vminus = dist.Field(name="coefficient_vminus", bases=basis)
    coefficient_alpha_plus = dist.Field(name="coefficient_alpha_plus", bases=basis)
    coefficient_alpha_minus = dist.Field(
        name="coefficient_alpha_minus", bases=basis
    )
    coefficient_inverse_two_alpha_plus = dist.Field(
        name="coefficient_inverse_two_alpha_plus", bases=basis
    )
    coefficient_inverse_two_alpha_minus = dist.Field(
        name="coefficient_inverse_two_alpha_minus", bases=basis
    )
    coefficient_alpha_ratio = dist.Field(
        name="coefficient_alpha_ratio", bases=basis
    )
    potential_alpha_plus = dist.Field(name="potential_alpha_plus", bases=basis)
    potential_alpha_minus = dist.Field(name="potential_alpha_minus", bases=basis)
    potential = dist.Field(name="potential", bases=basis)

    drho = lambda field: d3.Differentiate(field, rho_coord)

    coefficient_a["g"] = propagation_function(rho)
    coefficient_b["g"] = boost_function(rho)
    if one_plus_boost_function is not None:
        coefficient_cplus["g"] = one_plus_boost_function(rho)
    if one_minus_boost_function is not None:
        coefficient_cminus["g"] = one_minus_boost_function(rho)
    potential["g"] = potential_function(rho)
    initial_u, initial_psi, initial_pi = initial_function(rho)
    u["g"] = initial_u
    if isinstance(initial, ArealBumpInitialData):
        # The analytic chain-rule values are used by the model validation.
        # In the finite Chebyshev representation, initialize the auxiliary
        # variable with the derivative of the represented u itself.  This is
        # the spectrally consistent realization of psi=(du/dr)(dr/d rho) and
        # prevents an avoidable O(truncation) first-order constraint at t=0.
        represented_derivative = drho(u).evaluate()
        represented_derivative.change_scales(1)
        initial_psi = np.asarray(represented_derivative["g"]).ravel()
        if initial.time_symmetric:
            initial_pi = -np.asarray(coefficient_b["g"]).ravel() * initial_psi

    characteristic_variables = bool(
        endpoint_factored_characteristic_variables
        or conservative_characteristic_variables
    )
    if characteristic_variables:
        if one_plus_boost_function is None or one_minus_boost_function is None:
            raise ValueError(
                "Characteristic variables require "
                "cancellation-free 1+B and 1-B."
            )
        a_values = np.asarray(coefficient_a["g"]).ravel()
        cplus_values = np.asarray(coefficient_cplus["g"]).ravel()
        cminus_values = np.asarray(coefficient_cminus["g"]).ravel()
        vplus_values = a_values * cplus_values
        vminus_values = a_values * cminus_values
        alpha_plus_values = vplus_values / (1.0 - rho)
        alpha_minus_values = vminus_values / rho
        coefficient_rho["g"] = rho
        coefficient_one_minus_rho["g"] = 1.0 - rho
        coefficient_alpha_plus["g"] = alpha_plus_values
        coefficient_alpha_minus["g"] = alpha_minus_values
        if not (
            np.all(np.isfinite(alpha_plus_values))
            and np.all(np.isfinite(alpha_minus_values))
            and np.all(alpha_plus_values > 0.0)
            and np.all(alpha_minus_values > 0.0)
        ):
            raise ValueError("Characteristic coefficients are not regular.")

    if endpoint_factored_characteristic_variables:
        potential_values = np.asarray(potential["g"]).ravel()
        coefficient_vplus["g"] = vplus_values
        coefficient_vminus["g"] = vminus_values
        coefficient_inverse_two_alpha_plus["g"] = 0.5 / alpha_plus_values
        coefficient_inverse_two_alpha_minus["g"] = 0.5 / alpha_minus_values
        coefficient_alpha_ratio["g"] = alpha_plus_values / alpha_minus_values
        potential_alpha_plus["g"] = alpha_plus_values * potential_values
        potential_alpha_minus["g"] = alpha_minus_values * potential_values

        alpha_plus_outer = float(
            coefficient_alpha_plus(rho=1.0).evaluate()["g"].ravel()[0]
        )
        alpha_minus_inner = float(
            coefficient_alpha_minus(rho=0.0).evaluate()["g"].ravel()[0]
        )
        expected_alpha_plus_outer = float(
            (model.cosmological_horizon - 3.0 * model.mass)
            / model.cosmological_horizon**2
        )
        expected_alpha_minus_inner = float(1.0 / (4.0 * model.mass))
        factored_coefficient_audit = {
            "finite_and_positive": True,
            "alpha_plus_minimum": float(np.min(alpha_plus_values)),
            "alpha_plus_maximum": float(np.max(alpha_plus_values)),
            "alpha_minus_minimum": float(np.min(alpha_minus_values)),
            "alpha_minus_maximum": float(np.max(alpha_minus_values)),
            "alpha_plus_chebyshev_tail_ratio": _chebyshev_tail_ratio(
                alpha_plus_values
            ),
            "alpha_minus_chebyshev_tail_ratio": _chebyshev_tail_ratio(
                alpha_minus_values
            ),
            "alpha_plus_over_minus_maximum": float(
                np.max(alpha_plus_values / alpha_minus_values)
            ),
            "alpha_plus_over_minus_chebyshev_tail_ratio": _chebyshev_tail_ratio(
                alpha_plus_values / alpha_minus_values
            ),
            "alpha_plus_outer_represented": alpha_plus_outer,
            "alpha_plus_outer_exact_kappa_c": expected_alpha_plus_outer,
            "alpha_plus_outer_absolute_error": abs(
                alpha_plus_outer - expected_alpha_plus_outer
            ),
            "alpha_minus_inner_represented": alpha_minus_inner,
            "alpha_minus_inner_exact": expected_alpha_minus_inner,
            "alpha_minus_inner_absolute_error": abs(
                alpha_minus_inner - expected_alpha_minus_inner
            ),
        }

        H = dist.Field(name="H", bases=basis)
        J = dist.Field(name="J", bases=basis)
        H["g"] = alpha_plus_values * (initial_pi + initial_psi)
        J["g"] = alpha_minus_values * (initial_pi - initial_psi)
        constraint_operator = (
            coefficient_inverse_two_alpha_plus * H
            - coefficient_inverse_two_alpha_minus * J
            - drho(u)
        )
        state_fields = (u, H, J)
        checkpoint_state_keys = ("u", "H", "J")
        problem = d3.IVP(list(state_fields), namespace=locals())
        problem.add_equation(
            "dt(u) - 0.5 * (coefficient_one_minus_rho * H"
            " + coefficient_rho * J) = 0"
        )
        if explicit_potential:
            problem.add_equation(
                "dt(H) - coefficient_vplus * drho(H)"
                " + coefficient_alpha_plus * H"
                " + characteristic_constraint_damping * H"
                " - characteristic_constraint_damping"
                " * coefficient_alpha_ratio * J"
                " - 2 * characteristic_constraint_damping"
                " * coefficient_alpha_plus * drho(u)"
                " = -potential_alpha_plus * u"
            )
            problem.add_equation(
                "dt(J) + coefficient_vminus * drho(J)"
                " + coefficient_alpha_minus * J = -potential_alpha_minus * u"
            )
        else:
            problem.add_equation(
                "dt(H) - coefficient_vplus * drho(H)"
                " + coefficient_alpha_plus * H"
                " + characteristic_constraint_damping * H"
                " - characteristic_constraint_damping"
                " * coefficient_alpha_ratio * J"
                " - 2 * characteristic_constraint_damping"
                " * coefficient_alpha_plus * drho(u)"
                " + potential_alpha_plus * u = 0"
            )
            problem.add_equation(
                "dt(J) + coefficient_vminus * drho(J)"
                " + coefficient_alpha_minus * J + potential_alpha_minus * u = 0"
            )
    elif conservative_characteristic_variables:
        h = dist.Field(name="h", bases=basis)
        j = dist.Field(name="j", bases=basis)
        h["g"] = initial_pi + initial_psi
        j["g"] = initial_pi - initial_psi
        # Keep the linear endpoint factors as explicit nested operators.  If
        # vplus=(1-rho)*alpha_plus and vminus=rho*alpha_minus are first
        # projected into separate NCC fields, coefficient truncation leaves
        # small nonzero endpoint speeds.  These expressions retain the exact
        # null factors while still reusing literally identical flux operators
        # in all three equations.
        flux_plus = coefficient_one_minus_rho * (coefficient_alpha_plus * h)
        flux_minus = coefficient_rho * (coefficient_alpha_minus * j)
        constraint_operator = 0.5 * (h - j) - drho(u)
        state_fields = (u, h, j)
        checkpoint_state_keys = ("u", "h", "j")
        problem = d3.IVP(list(state_fields), namespace=locals())
        problem.add_equation(
            "dt(u) - 0.5 * (flux_plus + flux_minus) = 0"
        )
        if explicit_potential:
            problem.add_equation(
                "dt(h) - drho(flux_plus) = -potential * u"
            )
            problem.add_equation(
                "dt(j) + drho(flux_minus) = -potential * u"
            )
        else:
            problem.add_equation(
                "dt(h) - drho(flux_plus) + potential * u = 0"
            )
            problem.add_equation(
                "dt(j) + drho(flux_minus) + potential * u = 0"
            )
    else:
        psi = dist.Field(name="psi", bases=basis)
        pi = dist.Field(name="pi", bases=basis)
        psi["g"] = initial_psi
        pi["g"] = initial_pi
        state_fields = (u, psi, pi)
        checkpoint_state_keys = ("u", "psi", "pi")
        problem = d3.IVP(list(state_fields), namespace=locals())
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
        constraint_operator = psi - drho(u)

    solver = problem.build_solver(
        _timestepper(numerical.timestepper),
        ncc_cutoff=NCC_CUTOFF,
        entry_cutoff=ENTRY_CUTOFF,
    )
    solver.stop_sim_time = numerical.end_time

    observer_operators = [u(rho=point) for point in numerical.observers]
    observer_rho = np.asarray(numerical.observers, dtype=float)
    observer_radius = radius_function(observer_rho)
    constraint_probe_rho: list[float] = []
    constraint_probe_labels: list[str] = []
    constraint_probe_operators = None
    if characteristic_variables:
        for index, point in enumerate(numerical.observers):
            constraint_probe_rho.append(float(point))
            constraint_probe_labels.append(f"waveform_observer_{index}")
        if "transition_inner_rho" in horizon_metadata:
            transition_inner = float(horizon_metadata["transition_inner_rho"])
            transition_outer = float(horizon_metadata["transition_outer_rho"])
            for label, point in (
                ("transition_inner", transition_inner),
                ("transition_midpoint", 0.5 * (transition_inner + transition_outer)),
                ("transition_outer", transition_outer),
            ):
                constraint_probe_rho.append(point)
                constraint_probe_labels.append(label)
        constraint_probe_operators = [
            constraint_operator(rho=point) for point in constraint_probe_rho
        ]

    signal_stride = max(1, round(numerical.signal_dt / numerical.timestep))
    snapshot_stride = max(1, round(numerical.snapshot_dt / numerical.timestep))
    progress_stride = max(1, round(numerical.end_time / numerical.timestep / 10))

    signal_times: list[float] = []
    signals: list[list[float]] = []
    snapshot_times: list[float] = []
    u_snapshots: list[np.ndarray] = []
    constraint_linf: list[float] = []
    constraint_l2: list[float] = []
    constraint_probe_values: list[list[float]] = []

    configuration = json.dumps(
        {
            "background": background,
            "model": model.as_dict(),
            "initial_data": initial.as_dict(),
            "numerical": asdict(numerical),
            # Recorded only when set, so checkpoints written by the default
            # implicit split stay byte-identical and remain resumable.
            **({"imex_split": "explicit_potential"} if explicit_potential else {}),
            **(
                {
                    "state_variables": (
                        "endpoint_factored_characteristic_H_J"
                        if endpoint_factored_characteristic_variables
                        else "conservative_characteristic_h_j"
                    )
                }
                if characteristic_variables
                else {}
            ),
            **(
                {
                    "characteristic_flux_discretization": (
                        "conservative_nested_endpoint_flux_v1"
                    )
                }
                if conservative_characteristic_variables
                else {}
            ),
            **(
                {
                    "characteristic_constraint_damping": (
                        characteristic_constraint_damping
                    )
                }
                if characteristic_constraint_damping != 0.0
                else {}
            ),
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
        state_arrays = {
            key: np.asarray(field["g"]).ravel()
            for key, field in zip(checkpoint_state_keys, state_fields)
        }
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                configuration=np.array(configuration),
                sim_time=np.array(solver.sim_time),
                iteration=np.array(solver.iteration),
                elapsed_wall_seconds=np.array(elapsed_wall_seconds),
                # ``state_scales`` is retained for checkpoints made by older
                # versions.  The per-field scales handle the case where a
                # Dedalus interpolation or product has temporarily placed
                # different state fields at different dealias scales.
                state_scales=np.asarray(u.scales),
                state_field_scales=np.asarray(
                    [field.scales for field in state_fields], dtype=float
                ),
                **state_arrays,
                signal_times=np.asarray(signal_times),
                signals=np.asarray(signals),
                snapshot_times=np.asarray(snapshot_times),
                u_snapshots=np.asarray(u_snapshots),
                constraint_linf=np.asarray(constraint_linf),
                constraint_l2=np.asarray(constraint_l2),
                **(
                    {
                        "constraint_probe_values": np.asarray(
                            constraint_probe_values
                        )
                    }
                    if characteristic_variables
                    else {}
                ),
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
        constraint_field = constraint_operator.evaluate()
        constraint_field.change_scales(1)
        constraint = np.asarray(constraint_field["g"]).ravel().copy()
        u.change_scales(1)
        snapshot_times.append(float(solver.sim_time))
        u_snapshots.append(np.asarray(u["g"]).ravel().copy())
        constraint_linf.append(float(np.max(np.abs(constraint))))
        constraint_l2.append(float(np.sqrt(np.mean(constraint**2))))
        if constraint_probe_operators is not None:
            constraint_probe_values.append(
                [
                    float(operator.evaluate()["g"].ravel()[0])
                    for operator in constraint_probe_operators
                ]
            )

    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            saved_configuration = saved["configuration"].item()
            if saved_configuration != configuration:
                raise ValueError(
                    f"Checkpoint configuration does not match this run: {checkpoint}"
                )
            if "state_field_scales" in saved:
                field_scales = [
                    tuple(np.atleast_1d(scales).tolist())
                    for scales in saved["state_field_scales"]
                ]
            else:
                # Legacy checkpoints recorded only the scale of ``u``.  Infer
                # every one-dimensional field's actual scale from its stored
                # grid length so even a mixed-scale checkpoint remains usable.
                field_scales = [
                    (float(saved[key].size) / numerical.resolution,)
                    for key in checkpoint_state_keys
                ]
            for field, key, scales in zip(
                state_fields, checkpoint_state_keys, field_scales
            ):
                field.change_scales(scales)
                field["g"] = saved[key]
            solver.sim_time = solver.initial_sim_time = float(saved["sim_time"])
            solver.iteration = solver.initial_iteration = int(saved["iteration"])
            elapsed_before_restart = float(saved["elapsed_wall_seconds"])
            signal_times.extend(saved["signal_times"].tolist())
            signals.extend(saved["signals"].tolist())
            snapshot_times.extend(saved["snapshot_times"].tolist())
            saved_snapshots = np.asarray(saved["u_snapshots"])
            if saved_snapshots.ndim != 2 or saved_snapshots.shape[1] != rho.size:
                raise ValueError(
                    "Checkpoint snapshots do not use the base-grid width; "
                    "restart this diagnostic from clean initial data."
                )
            u_snapshots.extend(saved_snapshots.tolist())
            constraint_linf.extend(saved["constraint_linf"].tolist())
            constraint_l2.extend(saved["constraint_l2"].tolist())
            if characteristic_variables:
                constraint_probe_values.extend(
                    saved["constraint_probe_values"].tolist()
                )
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
    if endpoint_factored_characteristic_variables:
        equation_metadata = {
            "variables": (
                "F=(1-rho)*H=A*(1+B)*(pi+psi), "
                "G=rho*J=A*(1-B)*(pi-psi)"
            ),
            "u": "dt(u) = ((1-rho)*H+rho*J)/2",
            "H": (
                "dt(H) = vplus*d_rho(H)-alpha_plus*(H+P*u)"
                "-2*gamma*alpha_plus*C"
            ),
            "J": "dt(J) = -vminus*d_rho(J)-alpha_minus*(J+P*u)",
            "vplus": "A*(1+B)=(1-rho)*alpha_plus",
            "vminus": "A*(1-B)=rho*alpha_minus",
            "alpha_plus": "A*(1+B)/(1-rho)",
            "alpha_minus": "A*(1-B)/rho",
            "A": "(f*d rho/dr)/(1-B^2)",
            "B": "f*dh/dr",
            "P": "V_scalar/(f*d rho/dr)",
            "gamma": characteristic_constraint_damping,
        }
        constraint_metadata = {
            "definition": (
                "C=H/(2*alpha_plus)-J/(2*alpha_minus)-d_rho(u)"
            ),
            "continuum_identity": "C=psi-d_rho(u)",
            "endpoint_safe": True,
            "saved_norms": "unweighted C",
            "normalization": 1.0,
            "damping_rate": characteristic_constraint_damping,
            "damping_rate_units": "inverse coordinate time",
            "dimensionless_rate": "gamma*M",
            "continuum_propagation": "d_tau(C)=-gamma*C",
            "damped_sector": "pi+psi (H); pi-psi (J) unchanged",
            "probe_rho": constraint_probe_rho,
            "probe_labels": constraint_probe_labels,
            "probe_values": constraint_probe_values,
            "probe_sample_times": "snapshot_times",
            "note": (
                "The endpoint factors are carried by the evolved fluxes; "
                "alpha_plus and alpha_minus remain finite and nonzero at "
                "the two null boundaries."
            ),
        }
        characteristic_initialization = {
            "evolved_variables": "u, H, J",
            "H": "alpha_plus*(pi+psi)",
            "J": "alpha_minus*(pi-psi)",
        }
    elif conservative_characteristic_variables:
        equation_metadata = {
            "variables": "h=pi+psi, j=pi-psi",
            "flux_plus": (
                "Fplus=(1-rho)*(alpha_plus*h)=A*(1+B)*h"
            ),
            "flux_minus": "Fminus=rho*(alpha_minus*j)=A*(1-B)*j",
            "flux_discretization": "conservative_nested_endpoint_flux_v1",
            "u": "dt(u)=(Fplus+Fminus)/2",
            "h": "dt(h)=d_rho(Fplus)-P*u",
            "j": "dt(j)=-d_rho(Fminus)-P*u",
            "A": "(f*d rho/dr)/(1-B^2)",
            "B": "f*dh/dr",
            "P": "V_scalar/(f*d rho/dr)",
        }
        constraint_metadata = {
            "definition": "C=(h-j)/2-d_rho(u)",
            "continuum_identity": "C=psi-d_rho(u)",
            "endpoint_safe": True,
            "saved_norms": "unweighted C",
            "normalization": 1.0,
            "damping_rate": 0.0,
            "continuum_propagation": "d_tau(C)=0",
            "semidiscrete_propagation": (
                "d_tau(C)=0 because identical Fplus and Fminus operators "
                "are reused in all three equations"
            ),
            "probe_rho": constraint_probe_rho,
            "probe_labels": constraint_probe_labels,
            "probe_values": constraint_probe_values,
            "probe_sample_times": "snapshot_times",
        }
        characteristic_initialization = {
            "evolved_variables": "u, h, j",
            "h": "pi+psi",
            "j": "pi-psi",
        }
    else:
        equation_metadata = {
            "u": f"dt(u) = A*({outgoing_flux})",
            "psi": f"dt(psi) = d_rho[A*({outgoing_flux})]",
            "pi": f"dt(pi) = d_rho[A*({ingoing_flux})] - P*u",
            "A": "(f*d rho/dr)/(1-B^2)",
            "B": "f*dh/dr",
            "P": "V_scalar/(f*d rho/dr)",
        }
        constraint_metadata = None
        characteristic_initialization = {}

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
        "equations": equation_metadata,
        **(
            {"constraint": constraint_metadata}
            if constraint_metadata is not None
            else {}
        ),
        **(
            {"factored_coefficient_audit": factored_coefficient_audit}
            if endpoint_factored_characteristic_variables
            else {}
        ),
        "imex_split": {
            "potential_term": "explicit" if explicit_potential else "implicit",
            "transport_terms": "implicit",
            "note": (
                "The stiffness of the Chebyshev discretization resides in the "
                "transport terms, which are implicit in both variants. P is "
                "bounded and of order unity, and therefore non-stiff."
            ),
        },
        "dedalus_matrix_assembly": {
            "ncc_cutoff": NCC_CUTOFF,
            "entry_cutoff": ENTRY_CUTOFF,
        },
        "initialization": {
            **characteristic_initialization,
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
    endpoint_factored_characteristic_variables: bool = False,
    conservative_characteristic_variables: bool = False,
    characteristic_constraint_damping: float = 0.0,
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
        one_minus_boost_function=lambda rho: exterior_bridge_one_minus_boost(
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
        endpoint_factored_characteristic_variables=(
            endpoint_factored_characteristic_variables
        ),
        conservative_characteristic_variables=(
            conservative_characteristic_variables
        ),
        characteristic_constraint_damping=characteristic_constraint_damping,
    )
