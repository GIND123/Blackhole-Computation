"""A clean worktree must be recorded as clean.

The provenance flag decides whether an archive can be quoted as production
evidence, so the one thing it must never do is report every checkout as dirty.
It did: ``git status --porcelain`` prints nothing for a clean worktree, that
empty output was collapsed to ``None`` by the shared command helper, and the
flag read ``status is None or bool(status)``.  Clean and unreadable were then
the same value.  These tests drive the probe against real repositories in all
three states.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from black_hole import reproducibility


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class WorktreeStateTests(unittest.TestCase):
    """The three states have to be distinguishable from one another."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self._origin = Path.cwd()
        self.addCleanup(lambda: __import__("os").chdir(self._origin))

    def _repository(self) -> Path:
        root = self.root / "repository"
        root.mkdir()
        _git("init", "--quiet", cwd=root)
        _git("config", "user.email", "test@example.invalid", cwd=root)
        _git("config", "user.name", "test", cwd=root)
        (root / "tracked.txt").write_text("one\n", encoding="utf-8")
        _git("add", "tracked.txt", cwd=root)
        _git("commit", "--quiet", "-m", "first", cwd=root)
        return root

    def test_a_clean_worktree_is_clean(self) -> None:
        import os

        root = self._repository()
        os.chdir(root)
        self.assertEqual(reproducibility._worktree_state(), "clean")
        self.assertFalse(reproducibility.reproducibility_metadata()["git_worktree_dirty"])

    def test_a_modified_worktree_is_dirty(self) -> None:
        import os

        root = self._repository()
        (root / "tracked.txt").write_text("two\n", encoding="utf-8")
        os.chdir(root)
        self.assertEqual(reproducibility._worktree_state(), "dirty")
        self.assertTrue(reproducibility.reproducibility_metadata()["git_worktree_dirty"])

    def test_an_untracked_file_alone_does_not_make_it_dirty(self) -> None:
        # Archives are written into the tree while it runs, so untracked files
        # must not demote the run that is writing them.
        import os

        root = self._repository()
        (root / "output.npz").write_bytes(b"\x00")
        os.chdir(root)
        self.assertEqual(reproducibility._worktree_state(), "clean")

    def test_a_failed_probe_is_unknown_and_counts_as_dirty(self) -> None:
        import os

        outside = self.root / "not-a-repository"
        outside.mkdir()
        os.chdir(outside)
        self.assertEqual(reproducibility._worktree_state(), "unknown")
        metadata = reproducibility.reproducibility_metadata()
        self.assertTrue(metadata["git_worktree_dirty"])
        self.assertEqual(metadata["git_worktree_state"], "unknown")


if __name__ == "__main__":
    unittest.main()
