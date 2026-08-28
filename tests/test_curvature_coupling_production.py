"""Contract tests for the matched curvature-coupling campaign."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from black_hole.curvature_coupling_production import (
    CONFORMAL_COUPLING,
    TAIL_SETTINGS,
    _contract,
    _result_audit,
    archive_path,
    case_catalogue,
    exterior_spectral_preflight,
    qnm_cases,
    tail_cases,
)


class CurvatureCouplingProductionTests(unittest.TestCase):
    def test_case_matrix_is_unique_and_complete(self) -> None:
        catalogue = case_catalogue()
        self.assertEqual(len(qnm_cases()), 25)
        self.assertEqual(len(tail_cases()), 20)
        self.assertEqual(len(catalogue), 45)
        self.assertEqual(
            sum(case.curvature_coupling == CONFORMAL_COUPLING for case in qnm_cases()),
            25,
        )
        for background in ("schwarzschild", "uniform", "exterior"):
            selected = [case for case in tail_cases() if case.background == background]
            expected = 4 if background == "schwarzschild" else 8
            self.assertEqual(len(selected), expected)
        self.assertEqual(len(TAIL_SETTINGS), 4)

    def test_contracts_and_paths_record_coupling(self) -> None:
        root = Path("new-output")
        for case in qnm_cases() + tail_cases():
            with self.subTest(case=case.name):
                contract = _contract(case)
                self.assertEqual(contract["curvature_coupling"], case.curvature_coupling)
                if case.group == "tail" and case.background == "exterior":
                    self.assertTrue(contract["conservative_characteristic_variables"])
                    self.assertFalse(
                        contract["endpoint_factored_characteristic_variables"]
                    )
                    self.assertEqual(
                        contract["characteristic_flux_discretization"],
                        "conservative_nested_endpoint_flux_v1",
                    )
                path = archive_path(root, case)
                self.assertTrue(path.as_posix().startswith("new-output/raw/"))
                self.assertIn(case.coupling_label, path.parts)

    def test_every_exterior_preflight_passes(self) -> None:
        for case in qnm_cases() + tail_cases():
            if case.background != "exterior":
                continue
            with self.subTest(case=case.name):
                audit = exterior_spectral_preflight(case)
                self.assertTrue(audit["passed"])
                self.assertGreaterEqual(audit["transition_nodes"], 12)
                self.assertGreaterEqual(audit["outer_cap_nodes"], 12)

    def test_tail_audit_rejects_finite_instability(self) -> None:
        case = next(
            case
            for case in tail_cases()
            if case.background == "exterior"
            and case.curvature_coupling == CONFORMAL_COUPLING
        )
        times = np.array([0.0, 60.0, 960.0, 980.0, 1000.0])
        stable = SimpleNamespace(
            rho=np.linspace(0.0, 1.0, 8),
            signal_times=times,
            signals=np.array([[1.0], [1.0e-3], [2.0e-5], [1.0e-5], [5.0e-6]]),
            snapshot_times=times,
            u_snapshots=np.zeros((times.size, 8)),
            constraint_linf=np.full(times.size, 1.0e-8),
            constraint_l2=np.full(times.size, 1.0e-9),
        )
        self.assertTrue(_result_audit(stable, case, 0.0)["passed"])
        unstable = SimpleNamespace(**vars(stable))
        unstable.signals = stable.signals.copy()
        unstable.signals[-1, 0] = 1.0e6
        audit = _result_audit(unstable, case, 0.0)
        self.assertTrue(audit["finite"])
        self.assertFalse(audit["passed"])


if __name__ == "__main__":
    unittest.main()
