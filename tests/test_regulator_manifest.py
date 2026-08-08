"""Tests for newline-stable regulator hashes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from black_hole.regulator_manifest import sha256_bytes, sha256_lf


class RegulatorManifestTests(unittest.TestCase):
    def test_text_hash_is_newline_stable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"one\ntwo\n")
            crlf.write_bytes(b"one\r\ntwo\r\n")
            self.assertEqual(sha256_lf(lf), sha256_lf(crlf))
            self.assertNotEqual(sha256_bytes(lf), sha256_bytes(crlf))


if __name__ == "__main__":
    unittest.main()
