"""Runtime path-contract tests shared by setup, train, watch, and evaluation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from curriculum_objectives import resolve_runtime_start_state_path, stage_start_state_path


class RuntimeStateResolutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.canonical = self.root / "runtime_data" / "states" / "training_start.state"
        self.canonical.parent.mkdir(parents=True)
        self.canonical.write_bytes(b"canonical")
        self.milestones = self.canonical.parent / "milestones"
        self.milestones.mkdir()
        self.starter_milestone = self.milestones / "starter.state"
        self.starter_milestone.write_bytes(b"completed starter")
        self.leave_lab_milestone = self.milestones / "leave_birch_lab.state"
        self.leave_lab_milestone.write_bytes(b"completed lab")

    def test_setup_always_targets_canonical_even_when_env_points_at_milestone(self):
        result = resolve_runtime_start_state_path(
            self.root,
            "leave_birch_lab",
            mode="setup",
            training_start_state=self.starter_milestone,
        )
        self.assertEqual(result, self.canonical.resolve())

    def test_setup_preserves_legacy_custom_canonical_path_outside_milestones(self):
        custom = self.root / "custom_route101.state"
        custom.write_bytes(b"custom canonical")
        result = resolve_runtime_start_state_path(
            self.root,
            "starter",
            mode="setup",
            training_start_state=custom,
        )
        self.assertEqual(result, custom.resolve())

    def test_setup_rejects_canonical_path_inside_milestones(self):
        with self.assertRaisesRegex(ValueError, "cannot point inside"):
            resolve_runtime_start_state_path(
                self.root,
                "starter",
                mode="setup",
                canonical_start_state=self.starter_milestone,
            )

    def test_starter_rejects_inherited_completed_milestone(self):
        with self.assertRaisesRegex(ValueError, "must use its mapped start state"):
            resolve_runtime_start_state_path(
                self.root,
                "starter",
                mode="train",
                canonical_start_state=self.canonical,
                training_start_state=self.starter_milestone,
            )

    def test_later_stage_accepts_only_immediately_preceding_milestone(self):
        expected = stage_start_state_path(self.root, "leave_birch_lab", self.canonical)
        self.assertEqual(expected, self.starter_milestone.resolve())
        resolved = resolve_runtime_start_state_path(
            self.root,
            "leave_birch_lab",
            mode="watch",
            canonical_start_state=self.canonical,
            training_start_state=self.starter_milestone,
        )
        self.assertEqual(resolved, expected)

    def test_new_route101_stage_starts_from_the_littleroot_milestone(self):
        expected = stage_start_state_path(self.root, "reach_route101", self.canonical)
        self.assertEqual(expected, self.leave_lab_milestone.resolve())
        resolved = resolve_runtime_start_state_path(
            self.root, "reach_route101", mode="train",
            canonical_start_state=self.canonical,
            training_start_state=self.leave_lab_milestone,
        )
        self.assertEqual(resolved, expected)

    def test_later_stage_rejects_wrong_milestone_path(self):
        with self.assertRaisesRegex(ValueError, "must use its mapped start state"):
            resolve_runtime_start_state_path(
                self.root,
                "leave_birch_lab",
                mode="evaluate",
                canonical_start_state=self.canonical,
                training_start_state=self.leave_lab_milestone,
            )

    def test_legacy_starter_training_path_is_promoted_to_canonical(self):
        custom = self.root / "older_canonical.state"
        custom.write_bytes(b"legacy")
        resolved = resolve_runtime_start_state_path(
            self.root,
            "starter",
            mode="train",
            training_start_state=custom,
        )
        self.assertEqual(resolved, custom.resolve())

    def test_legacy_milestone_path_is_not_promoted_to_canonical(self):
        with self.assertRaisesRegex(ValueError, "must use its mapped start state"):
            resolve_runtime_start_state_path(
                self.root,
                "starter",
                mode="train",
                training_start_state=self.starter_milestone,
            )


if __name__ == "__main__":
    unittest.main()
