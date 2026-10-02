"""Run the predeclared quadrupole constraint campaign, two jobs at most."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("results/ringdown_constraint_revision_v1")


def execute(length, resolution, dt):
    label = f"L{length}_N{resolution}_dt{str(dt).replace('.', 'p')}"
    destination = ROOT / "raw" / "shared_flux" / f"L{length}" / f"N{resolution}_dt{str(dt).replace('.', 'p')}.npz"
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {"case": label, "status": "existing_refused"}
    command = [sys.executable, "-m", "black_hole.ringdown_constraint_diagnostic",
               "--length", str(length), "--resolution", str(resolution),
               "--dt", str(dt), "--end-u", "80", "--signal-dt", "0.01",
               "--constraint-dt", "0.1", "--snapshot-dt", "5",
               "--output", str(destination)]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    start = time.perf_counter()
    print(json.dumps({"case": label, "event": "starting", "command": command}), flush=True)
    with (logs / f"{label}.log").open("x") as stream:
        completed = subprocess.run(command, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    report = {"case": label, "returncode": completed.returncode,
              "wall_seconds": time.perf_counter() - start}
    print(json.dumps(report), flush=True)
    return report


def main():
    plan = json.loads((ROOT / "campaign_plan.json").read_text())
    # Start the most expensive cases first; this affects cost, never case selection.
    coarse, medium, fine = plan["resolutions"]
    settings = [(fine, plan["fixed_timestep_over_M"]),
                (medium, plan["halved_timestep_check"]["timestep_over_M"]),
                (medium, plan["fixed_timestep_over_M"]),
                (coarse, plan["fixed_timestep_over_M"])]
    cases = [(length, n, dt) for n, dt in settings for length in plan["lengths_over_M"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, *case) for case in cases]
        outcomes = [future.result() for future in as_completed(futures)]
    if any(row.get("returncode", 1) != 0 for row in outcomes):
        raise SystemExit("One or more campaign cases failed; inspect individual logs")


if __name__ == "__main__":
    main()
