"""Checks for the unified exterior-QNM evidence manifest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from black_hole.exterior_qnm_manifest import INPUT_SET, verify_manifest


MANIFEST = Path("results/exterior_regulator_width_floor_qnm_v5/manifest.json")


@unittest.skipUnless(MANIFEST.exists(), "exterior-QNM manifest has not been built")
class ExteriorQNMManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_verifies(self) -> None:
        report = verify_manifest(MANIFEST)
        self.assertTrue(report["verified"])
        self.assertEqual(report["raw_inputs"], 27)
        self.assertEqual(report["unlisted_package_files"], 0)

    def test_complete_archive_inventory(self) -> None:
        rows = self.manifest["archives"]
        self.assertEqual(len(rows), 27)
        self.assertEqual(
            sum(row["family"] == "schwarzschild_control" for row in rows), 3
        )
        self.assertEqual(
            sum(row["family"] == "uniform_sds_control" for row in rows), 12
        )
        self.assertEqual(
            sum(row["family"] == "exterior_supported_sds" for row in rows), 12
        )

    def test_every_derived_artifact_uses_the_complete_input_set(self) -> None:
        inputs = self.manifest["input_sets"][INPUT_SET]
        self.assertEqual(len(inputs), 27)
        for row in self.manifest["derived_artifacts"]:
            self.assertEqual(row["input_set"], INPUT_SET)

    def test_dirty_exterior_provenance_is_not_promoted(self) -> None:
        exterior = [
            row
            for row in self.manifest["archives"]
            if row["family"] == "exterior_supported_sds"
        ]
        self.assertTrue(all(row["git_worktree_dirty"] for row in exterior))
        self.assertTrue(all(row["provenance_grade"] == "screening" for row in exterior))
        controls = [
            row
            for row in self.manifest["archives"]
            if row["family"] != "exterior_supported_sds"
        ]
        self.assertTrue(all(row["provenance_grade"] == "production" for row in controls))

    def test_legacy_and_fixed_width_members_are_explicit(self) -> None:
        exterior = [
            row
            for row in self.manifest["archives"]
            if row["family"] == "exterior_supported_sds"
        ]
        legacy = [
            row for row in exterior if row["storage"] == "external_unchanged_legacy_archive"
        ]
        current = [
            row for row in exterior if row["storage"] == "internal_width_floor_archive"
        ]
        self.assertEqual({row["cosmological_length_over_M"] for row in legacy}, {80, 160})
        self.assertEqual({row["cosmological_length_over_M"] for row in current}, {320, 640})
        self.assertEqual(len(legacy), 6)
        self.assertEqual(len(current), 6)


if __name__ == "__main__":
    unittest.main()
