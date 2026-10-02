"""Small independent algebra, convergence and precision checks for the backend.

Run after ``sh build.sh dd`` with a Python environment containing mpmath:
``python -m unittest black_hole.high_precision.test_backend -v``.
No production data are touched; all generated fixtures live in TemporaryDirectory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import mpmath as mp

BACKEND = Path(__file__).parent / "_build" / "hermite_banded_dd"


@unittest.skipUnless(BACKEND.exists(), "Build the DD backend first")
class BackendTests(unittest.TestCase):
    def setUp(self):
        self.context = mp.workdps(60)
        self.context.__enter__()
        self.scratch = tempfile.TemporaryDirectory(prefix="sds-hermite-tests-")
        self.root = Path(self.scratch.name)
        self.serial = 0

    def tearDown(self):
        self.scratch.cleanup()
        self.context.__exit__(None, None, None)

    def run_problem(self, j, g, u, v, dt, end, bw=None, cadence=1, integrator="hermite"):
        self.serial += 1
        n = len(u)
        bw = n-1 if bw is None else bw
        source = self.root / f"input-{self.serial}.txt"
        out = self.root / f"output-{self.serial}"
        with source.open("w") as stream:
            stream.write(f"{n} {bw}\n")
            for i in range(n):
                stream.write(f"{i} {mp.nstr(u[i],55)} {mp.nstr(v[i],55)}\n")
            for i in range(n):
                for k in range(max(0,i-bw),min(n,i+bw+1)):
                    stream.write(f"{mp.nstr(j[i,k],55)} {mp.nstr(g[i,k],55)}\n")
        command = [str(BACKEND), str(source), str(out), str(dt), str(end), str(cadence), "30"]
        if integrator != "hermite":
            command.extend(["--integrator",integrator])
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        with (out / "final_state.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        values = mp.matrix([mp.mpf(row["u"]) for row in rows] + [mp.mpf(row["v"]) for row in rows])
        metadata = json.loads((out / "backend.json").read_text())
        return values, metadata, out, command

    def test_oscillator_fourth_order(self):
        j, g = mp.matrix([[1]]), mp.matrix([[0]])
        errors = []
        for dt in ("0.2", "0.1", "0.05"):
            final, metadata, _, _ = self.run_problem(j,g,[mp.mpf(1)],[mp.mpf(0)],dt,"2")
            errors.append(mp.norm(final-mp.matrix([mp.cos(2),-mp.sin(2)])))
            self.assertLess(mp.mpf(metadata["discarded_imaginary_max"]), mp.mpf("1e-29"))
        self.assertGreater(errors[0]/errors[1],15.8)
        self.assertGreater(errors[1]/errors[2],15.9)
        self.assertLess(errors[0]/errors[1],16.1)

    def test_exact_decimal_input_and_partial_step(self):
        j = g = mp.matrix([[0]])
        tiny = mp.mpf("1e-70")
        u = mp.mpf("1.0000000000000000000000001")
        final, metadata, out, command = self.run_problem(j,g,[u],[tiny],"0.2","0.45")
        self.assertLess(abs(final[0]-u),mp.mpf("1e-30"))
        self.assertLess(abs(final[1]/tiny-1),mp.mpf("1e-30"))
        self.assertEqual(metadata["full_steps"],2)
        self.assertLess(abs(mp.mpf(metadata["partial_step"])-mp.mpf(".05")),mp.mpf("1e-31"))
        with (out / "history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        self.assertLess(abs(mp.mpf(history[-1]["tau"])-mp.mpf(".45")),mp.mpf("1e-31"))
        failed = subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(failed.returncode,0)
        self.assertIn("already exists",failed.stderr)

    def compare_to_dense_pade(self, j, g, dt, bw, integrator="hermite"):
        n = j.rows
        u = mp.matrix([mp.mpf(i+1)/(n+1) for i in range(n)])
        v = mp.matrix([mp.mpf((-1)**i)/(i+2) for i in range(n)])
        a = mp.zeros(2*n)
        for i in range(n):
            a[i,n+i] = 1
            for k in range(n):
                a[n+i,k] = -j[i,k]
                a[n+i,n+k] = g[i,k]
        dt = mp.mpf(dt)
        identity = mp.eye(2*n)
        if integrator == "radau3":
            numerator = identity + dt*a/3
            denominator = identity - 2*dt*a/3 + dt**2*a*a/6
        else:
            numerator = identity + dt*a/2 + dt**2*a*a/12
            denominator = identity - dt*a/2 + dt**2*a*a/12
        initial = mp.matrix(list(u)+list(v))
        expected = mp.lu_solve(denominator,numerator*initial)
        actual, _, _, _ = self.run_problem(j,g,u,v,mp.nstr(dt,55),mp.nstr(dt,55),bw=bw,integrator=integrator)
        self.assertLess(mp.norm(actual-expected)/max(1,mp.norm(expected)),mp.mpf("3e-28"))

    def test_nonsymmetric_banded_dense_reference(self):
        n, bw = 11, 3
        j, g = mp.zeros(n), mp.zeros(n)
        for i in range(n):
            for k in range(max(0,i-bw),min(n,i+bw+1)):
                j[i,k] = mp.mpf(((i+2)*(k+3))%17-8)/7
                g[i,k] = mp.mpf(((i+5)*(k+1))%13-6)/11
            j[i,i] += 4
        self.compare_to_dense_pade(j,g,"0.125",bw)

    def test_partial_pivoting(self):
        # The complex Schur matrix has zero diagonal in exact arithmetic:
        # 1-alpha*(6/dt)+alpha**2*(12/dt**2)=0. Off-diagonal
        # entries make it nonsingular and force banded row pivots.
        n, bw = 8, 2
        j, g = mp.zeros(n), mp.zeros(n)
        dt = mp.mpf("0.125")
        for i in range(n):
            j[i,i], g[i,i] = 12/dt**2,6/dt
            for k in range(max(0,i-bw),min(n,i+bw+1)):
                if i != k:
                    j[i,k] = ((i+3)*(k+7))%19+1
        self.compare_to_dense_pade(j,g,dt,bw)

    def test_radau_third_order(self):
        j, g = mp.matrix([[1]]), mp.matrix([[0]])
        errors = []
        for dt in ("0.2", "0.1", "0.05"):
            final, metadata, _, _ = self.run_problem(j,g,[mp.mpf(1)],[mp.mpf(0)],dt,"2",integrator="radau3")
            errors.append(mp.norm(final-mp.matrix([mp.cos(2),-mp.sin(2)])))
            self.assertEqual(metadata["integrator"],"radau3")
            self.assertLess(mp.mpf(metadata["discarded_imaginary_max"]),mp.mpf("1e-29"))
        self.assertGreater(errors[0]/errors[1],7.9)
        self.assertGreater(errors[1]/errors[2],7.95)
        self.assertLess(errors[0]/errors[1],8.1)

    def test_radau_stiff_damping(self):
        j, g = mp.matrix([[0]]), mp.matrix([[-1000000]])
        hermite, _, _, _ = self.run_problem(j,g,[mp.mpf(0)],[mp.mpf(1)],"1","1")
        radau, _, _, _ = self.run_problem(j,g,[mp.mpf(0)],[mp.mpf(1)],"1","1",integrator="radau3")
        self.assertGreater(abs(hermite[1]),mp.mpf("0.99"))
        self.assertLess(abs(radau[1]),mp.mpf("3e-6"))
        z = mp.mpf(-1000000)
        expected = (1+z/3)/(1-2*z/3+z*z/6)
        self.assertLess(abs(radau[1]-expected),mp.mpf("1e-29"))

    def test_radau_banded_dense_reference(self):
        n, bw = 7, 2
        j, g = mp.zeros(n), mp.zeros(n)
        for i in range(n):
            for k in range(max(0,i-bw),min(n,i+bw+1)):
                j[i,k] = mp.mpf(((i+2)*(k+3))%17-8)/7
                g[i,k] = mp.mpf(((i+5)*(k+1))%13-6)/11
            j[i,i] += 4
        self.compare_to_dense_pade(j,g,"0.125",bw,integrator="radau3")

    def test_radau_partial_step(self):
        j, g = mp.matrix([[0]]), mp.matrix([[0]])
        final, metadata, _, _ = self.run_problem(j,g,[mp.mpf(1)],[mp.mpf(2)],"0.2","0.45",integrator="radau3")
        self.assertLess(abs(final[0]-mp.mpf("1.9")),mp.mpf("1e-29"))
        self.assertLess(abs(final[1]-2),mp.mpf("1e-29"))
        self.assertEqual(metadata["full_steps"],2)

    def test_decimal_underflow_and_small_normal(self):
        j = g = mp.matrix([[0]])
        final, metadata, _, _ = self.run_problem(j,g,[mp.mpf("1.5203399123123e-428")],[mp.mpf("1.23456789012345678901234567890123456789012345e-300")],"0.1","0")
        self.assertEqual(final[0],0)
        self.assertEqual(metadata["decimal_underflow_count"],1)
        self.assertLess(abs(final[1]/mp.mpf("1.23456789012345678901234567890123456789012345e-300")-1),mp.mpf("1e-22"))


if __name__ == "__main__":
    unittest.main()
