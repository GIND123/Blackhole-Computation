"""Regenerate the manuscript's production figures from the frozen archives."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from black_hole.regulator_analysis import (
    create_plots,
    flat_analysis,
    l12_phase_cleanup,
    source_analysis,
)
from black_hole.curvature_coupling_tail_analysis import (
    create_submission_figure as create_tail_figure,
)


def main() -> None:
    production = REPOSITORY_ROOT / "results" / "regulator_production_v3"
    destination = Path(__file__).resolve().parent / "figs"

    flat = flat_analysis(production)
    source = source_analysis(production, REPOSITORY_ROOT)
    phase_rows = l12_phase_cleanup(REPOSITORY_ROOT)

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sds-paper-figures-") as temporary:
        generated = create_plots(Path(temporary), flat, source, phase_rows)
        for path in generated:
            if path.suffix.lower() == ".pdf":
                target = destination / path.name
                shutil.copy2(path, target)
                print(target)

    tail_root = REPOSITORY_ROOT / "results" / "curvature_coupling_production_v2"
    tail_figure = destination / "tail_outer_boundary_comparison.pdf"
    create_tail_figure(tail_root, tail_figure)
    print(tail_figure)


if __name__ == "__main__":
    main()
