"""The manuscript inventory must stay synchronized with its evidence index."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from black_hole.evidence_index import OUTPUT, RECORD, build, render


ROOT = Path(__file__).resolve().parents[1]


class EvidenceIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build(ROOT)

    def test_every_manuscript_artifact_is_accounted_for(self) -> None:
        self.assertEqual(self.index["problems"], [])

    def test_matched_tail_evidence_is_the_retained_table(self) -> None:
        entries = {entry["name"]: entry for entry in self.index["entries"]}
        self.assertNotIn("tail_outer_boundary_comparison.pdf", entries)
        self.assertEqual(
            entries["tab:matched-tail-intervals"]["artifact"],
            "results/sbp_matched_tail_revision_v1/analysis/matched_tail_intervals.csv",
        )

    def test_generated_records_match_the_current_manuscript(self) -> None:
        self.assertEqual(
            json.loads((ROOT / RECORD).read_text(encoding="utf-8")), self.index
        )
        self.assertEqual(
            (ROOT / OUTPUT).read_text(encoding="utf-8"), render(self.index)
        )

    def test_condensed_caustic_audit_has_no_obsolete_floats(self) -> None:
        names = {entry["name"] for entry in self.index["entries"]}
        self.assertNotIn("D1_scaling.pdf", names)
        self.assertNotIn("tab:timing", names)


if __name__ == "__main__":
    unittest.main()
