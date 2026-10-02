"""Independent identities for the experimental multidomain SBP assembly."""

import unittest
import subprocess
import tempfile
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.linalg import expm

from black_hole.sbp_hermite import assemble, dense_matrices, lobatto, write_input, standard_boundaries
from black_hole.sbp_geometry import Geometry


class PolynomialGeometry:
    def coefficients(self,rho):
        return mp.mpf(1),1-2*rho,rho*(1-rho),mp.mpf(2)

    def initial(self,rho):
        return 1+rho,2-rho


class SBPAssemblyTests(unittest.TestCase):
    def test_lobatto_sbp_and_polynomials(self):
        with mp.workdps(50):
            for degree in (4,8,16):
                x,h,d = lobatto(degree)
                for i in range(degree+1):
                    for j in range(degree+1):
                        b = (-1 if i == 0 else 1 if i == degree else 0) if i == j else 0
                        self.assertLess(abs(h[i]*d[i,j]+h[j]*d[j,i]-b),mp.mpf("1e-44"))
                for power in range(degree+1):
                    computed = d*mp.matrix([v**power for v in x])
                    exact = [power*v**(power-1) if power else mp.mpf(0) for v in x]
                    self.assertLess(max(abs(a-b) for a,b in zip(computed,exact)),mp.mpf("1e-43"))

    def test_assembled_energy_and_polynomial_pde(self):
        with mp.workdps(50):
            operator = assemble(PolynomialGeometry(),[mp.mpf(0),mp.mpf(".37"),mp.mpf(1)],8)
            self.assertLess(operator["energy_defect"],mp.mpf("1e-44"))
            self.assertLess(operator["stiffness_defect"],mp.mpf("1e-42"))
            mass,k,c = dense_matrices(operator)
            x = np.array(list(map(float,operator["grid"])))
            u,v = 1+x,2-x
            # 2B v' + B'v + (p u')' - P u = -7 + 2rho.
            acceleration = (c@v-k@u)/mass
            np.testing.assert_allclose(acceleration,-7+2*x,atol=3e-10,rtol=0)
            energy_derivative = v@(c@v-k@u)+u@(k@v)
            self.assertAlmostEqual(energy_derivative,-v[0]**2-v[-1]**2,places=11)

    def test_split_layer_changes_only_computational_boundaries(self):
        with mp.workdps(50):
            geometry = Geometry("exterior", "640", xi="1/6", dps=50)
            standard = standard_boundaries(geometry)
            split = standard_boundaries(geometry, "split-layer")
            self.assertEqual(len(split),len(standard)+3)
            self.assertTrue(all(x in split for x in standard))
            added = [x for x in split if x not in standard]
            self.assertTrue(all(geometry.rho0 < x < geometry.rho1 for x in added))
            wave = standard_boundaries(geometry, "wave-resolved")
            self.assertEqual(len(wave),len(split)+4)
            self.assertTrue(all(x in wave for x in split))
            for radius in (80,120,160,200):
                self.assertIn(geometry.compact_radius(radius),wave)

    def test_displacement_support_sets_computational_boundaries(self):
        geometry = Geometry("exterior", "640", ell=2, initial_data="displacement")
        boundaries = standard_boundaries(geometry, "wave-resolved")
        self.assertTrue(all(x in boundaries for x in geometry.support_rho))
        self.assertEqual(geometry.support_rho,
                         (geometry.compact_radius("2.5"), geometry.compact_radius("5.5")))

    def test_hermite_manufactured_polynomial_and_energy(self):
        backend = Path(__file__).resolve().parents[1]/"black_hole/high_precision/_build/hermite_banded_dd"
        if not backend.exists():
            self.skipTest("Build the double-double backend first")
        with mp.workdps(50), tempfile.TemporaryDirectory(prefix="sbp-polynomial-") as temporary:
            root = Path(temporary)
            operator = assemble(PolynomialGeometry(),[mp.mpf(0),mp.mpf(".37"),mp.mpf(1)],8)
            write_input(operator,root/"operator.txt")
            analytic_generator = np.array([[0,0,1,0],[0,0,0,1],[-2,1,-2,2],[0,-4,0,-6]],float)
            a,b,c,d = expm(analytic_generator)@np.array([1,1,2,-1])
            mass,k,_ = dense_matrices(operator)
            x = np.array(list(map(float,operator["grid"])))
            initial_energy = ((2-x)@(mass*(2-x))+(1+x)@(k@(1+x)))/2
            for integrator, lower, upper in (("hermite",14,18),("radau3",7,9)):
                errors = []
                for dt in ("0.1","0.05","0.025"):
                    destination = root/f"{integrator}-{dt}"
                    subprocess.run([str(backend),str(root/"operator.txt"),str(destination),
                                    dt,"1","1","30","--integrator",integrator],
                                   check=True,capture_output=True,text=True,timeout=35)
                    final = np.genfromtxt(destination/"final_state.csv",delimiter=",",names=True)
                    errors.append(max(np.max(abs(final["u"]-(a+b*x))),
                                      np.max(abs(final["v"]-(c+d*x)))))
                    final_energy = (final["v"]@(mass*final["v"])+final["u"]@(k@final["u"]))/2
                    self.assertLess(final_energy,initial_energy)
                self.assertTrue(lower < errors[0]/errors[1] < upper,(integrator,errors))
                self.assertTrue(lower < errors[1]/errors[2] < upper,(integrator,errors))


if __name__ == "__main__":
    unittest.main()
