"""Checks for the large L tail manifest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from black_hole.tail_manifest import (
    _split_label,
    contract_sha256,
    physical_contract,
    verify_manifest,
)


MANIFEST = Path("results/large_l_tail/manifest.json")


class ContractTests(unittest.TestCase):
    def test_contract_digest_is_stable_and_covers_the_criteria(self) -> None:
        contract = physical_contract()
        for key in (
            "gauge",
            "initial_data",
            "retarded_time",
            "price_relative_tolerance",
            "price_continuous_duration_over_M",
            "cosmological_relative_tolerance",
            "imex_split",
        ):
            self.assertIn(key, contract)
        self.assertFalse(contract["time_translation_fitted"])
        self.assertEqual(contract_sha256(), contract_sha256())

    def test_split_label_reduces_every_recorded_form(self) -> None:
        self.assertEqual(
            _split_label({"imex_split": {"potential_term": "explicit"}}),
            "explicit_potential",
        )
        self.assertEqual(
            _split_label({"imex_split": "explicit_potential"}), "explicit_potential"
        )
        # Archives written before the field existed used the implicit split.
        self.assertEqual(_split_label({}), "implicit_potential")


@unittest.skipUnless(MANIFEST.exists(), "tail manifest has not been built")
class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_every_recorded_hash_matches(self) -> None:
        report = verify_manifest(MANIFEST)
        self.assertTrue(report["verified"])
        self.assertGreater(report["checked_files"], 0)

    def test_transient_files_are_not_recorded(self) -> None:
        paths = [row["path"] for row in self.manifest["archives"]]
        paths += [row["path"] for row in self.manifest["derived_artifacts"]]
        for path in paths:
            self.assertFalse(path.endswith(".running"), path)
            self.assertFalse(path.endswith(".checkpoint.npz"), path)

    def test_every_archive_carries_a_provenance_grade(self) -> None:
        for row in self.manifest["archives"]:
            self.assertIn(row["provenance_grade"], ("production", "screening"))
            self.assertEqual(
                row["provenance_grade"] == "screening", row["git_worktree_dirty"]
            )

    def test_final_archives_are_production_grade(self) -> None:
        for row in self.manifest["archives"]:
            if row["role"] == "final":
                self.assertEqual(row["provenance_grade"], "production", row["case"])

    def test_recorded_contract_matches_the_module(self) -> None:
        self.assertEqual(
            self.manifest["physical_contract_sha256"], contract_sha256()
        )


if __name__ == "__main__":
    unittest.main()
