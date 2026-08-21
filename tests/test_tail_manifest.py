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

    def test_no_final_archive_carries_production_provenance(self) -> None:
        """Every final archive was written from a dirty worktree.

        Provenance is recorded when an archive is saved, not when its run is
        launched, so a campaign started from a clean commit still records a
        dirty worktree if anything in the repository is edited while it runs.
        The whole final ladder is in that state.  Clearing it needs the
        worktree frozen for the entire duration of the runs, not merely at the
        moment they start.

        The assertion is that the dirty set is exactly the whole final set: a
        rerun that fixes some cases must shrink it, and this test must then be
        updated to name what is left.
        """

        final = {
            row["case"] for row in self.manifest["archives"] if row["role"] == "final"
        }
        dirty = {
            row["case"]
            for row in self.manifest["archives"]
            if row["role"] == "final" and row["provenance_grade"] != "production"
        }
        self.assertEqual(dirty, final)
        self.assertEqual(len(final), 32)

    def test_verification_detects_an_unlisted_artifact(self) -> None:
        """Hashes alone cannot catch a result that was never recorded."""

        probe = MANIFEST.parent / "tables" / "_unlisted_probe.csv"
        probe.write_text("probe" + chr(10), encoding="utf-8")
        try:
            with self.assertRaises(ValueError) as caught:
                verify_manifest(MANIFEST)
            self.assertIn("unlisted:", str(caught.exception))
        finally:
            probe.unlink()
        self.assertTrue(verify_manifest(MANIFEST)["verified"])

    def test_verification_reports_how_much_it_listed(self) -> None:
        report = verify_manifest(MANIFEST)
        self.assertEqual(report["unlisted_files"], 0)
        self.assertGreaterEqual(report["listed_files"], report["checked_files"])

    def test_recorded_contract_matches_the_module(self) -> None:
        self.assertEqual(
            self.manifest["physical_contract_sha256"], contract_sha256()
        )


if __name__ == "__main__":
    unittest.main()
