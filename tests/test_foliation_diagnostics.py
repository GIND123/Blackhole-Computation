"""Regression tests for formula-generated foliation diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
import unittest

from black_hole.foliation_diagnostics import evaluate_foliation_table


DATA = Path("paper/figs/data")
BRIDGES = (
    "minimum",
    "minimal",
    "linear",
    "modified_linear",
    "mavrogiannis",
    "slow_roll",
)


def _read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class FoliationDiagnosticTests(unittest.TestCase):
    def test_formulas_reproduce_archived_figure_values(self) -> None:
        speed = _read("foliation_conditioning.csv")
        minimum = _read("foliation_min_A.csv")
        offset = _read("foliation_retarded_offsets.csv")
        lengths = tuple(float(row["L_over_M"]) for row in speed)
        calculated = evaluate_foliation_table(lengths, BRIDGES)
        lookup = {(row.length_over_mass, row.bridge): row for row in calculated}
        for speed_row, minimum_row, offset_row in zip(speed, minimum, offset):
            length = float(speed_row["L_over_M"])
            for bridge in BRIDGES:
                row = lookup[(length, bridge)]
                self.assertAlmostEqual(
                    row.maximum_characteristic_speed,
                    float(speed_row[bridge]),
                    places=9,
                )
                self.assertAlmostEqual(
                    row.minimum_propagation_coefficient,
                    float(minimum_row[bridge]),
                    places=9,
                )
                self.assertAlmostEqual(
                    row.retarded_time_offset,
                    float(offset_row[bridge]),
                    places=8,
                )


if __name__ == "__main__":
    unittest.main()
