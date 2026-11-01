"""Regression tests for the v5 Emerald objective map and prerequisite contract."""
from __future__ import annotations

import unittest

from curriculum_objectives import (
    BIRCH_LAB_MAP_ID,
    CLOCK_SET_FLAG_ID,
    CURRICULUM_PLAN_ID,
    LITTLEROOT_TOWN_MAP_ID,
    OLDALE_TOWN_MAP_ID,
    POKEDEX_OBTAINED_FLAG_ID,
    ROUTE_101_MAP_ID,
    ROUTE_103_MAP_ID,
    RIVAL_DEFEATED_ROUTE103_FLAG_ID,
    STAGES,
    STARTER_OBTAINED_FLAG_ID,
    objective_satisfied,
    objective_start_error,
    stage_episode_rollouts,
    stage_start_state_path,
)


class CurriculumObjectiveTests(unittest.TestCase):
    @staticmethod
    def state(
        *,
        map_id: int = BIRCH_LAB_MAP_ID,
        party_count: int = 1,
        party_levels: int = 5,
        story_flags: set[int] | None = None,
        badge_count: int = 0,
        **extra,
    ) -> dict:
        return {
            "map_id": map_id,
            "party_count": party_count,
            "party_levels": party_levels,
            "story_flags": set() if story_flags is None else set(story_flags),
            "badge_count": badge_count,
            **extra,
        }

    def test_stage_order_is_incremental_and_shared(self):
        self.assertEqual(CURRICULUM_PLAN_ID, "emerald_intro_v5")
        self.assertEqual(
            STAGES,
            (
                "starter",
                "leave_birch_lab",
                "reach_route101",
                "reach_oldale",
                "reach_route103",
                "defeat_rival",
                "return_birch_lab",
                "pokedex",
                "first_badge",
            ),
        )
        for earlier, later in zip(STAGES, STAGES[1:]):
            self.assertLess(STAGES.index(earlier), STAGES.index(later))
        self.assertGreater(stage_episode_rollouts("reach_route101"), 0)

    def test_map_objectives_are_separate(self):
        starter_party = {STARTER_OBTAINED_FLAG_ID}

        littleroot_state = self.state(
            map_id=LITTLEROOT_TOWN_MAP_ID,
            story_flags=starter_party,
        )
        route101_state = self.state(
            map_id=ROUTE_101_MAP_ID,
            story_flags=starter_party,
        )
        oldale_state = self.state(
            map_id=OLDALE_TOWN_MAP_ID,
            story_flags=starter_party,
        )
        route103_state = self.state(
            map_id=ROUTE_103_MAP_ID,
            story_flags=starter_party,
        )

        # The lab-exit milestone is the first outside map (Littleroot), not Route 101.
        self.assertTrue(objective_satisfied("leave_birch_lab", littleroot_state))
        self.assertFalse(objective_satisfied("leave_birch_lab", route101_state))
        self.assertTrue(objective_satisfied("reach_route101", route101_state))
        self.assertFalse(objective_satisfied("reach_route101", littleroot_state))
        self.assertTrue(objective_satisfied("reach_oldale", oldale_state))
        self.assertTrue(objective_satisfied("reach_route103", route103_state))

    def test_rival_flag_and_return_to_lab_are_distinct_from_pokedex(self):
        base_flags = {STARTER_OBTAINED_FLAG_ID}
        rival_state = self.state(
            map_id=ROUTE_103_MAP_ID,
            story_flags=base_flags | {RIVAL_DEFEATED_ROUTE103_FLAG_ID},
        )
        self.assertTrue(objective_satisfied("defeat_rival", rival_state))
        self.assertFalse(objective_satisfied("pokedex", rival_state))

        returned_state = self.state(
            map_id=BIRCH_LAB_MAP_ID,
            story_flags=base_flags | {RIVAL_DEFEATED_ROUTE103_FLAG_ID},
        )
        self.assertTrue(objective_satisfied("return_birch_lab", returned_state))
        self.assertFalse(objective_satisfied("pokedex", returned_state))

        pokedex_state = self.state(
            map_id=BIRCH_LAB_MAP_ID,
            story_flags=base_flags | {
                RIVAL_DEFEATED_ROUTE103_FLAG_ID,
                POKEDEX_OBTAINED_FLAG_ID,
            },
        )
        self.assertTrue(objective_satisfied("pokedex", pokedex_state))

    def test_malformed_or_nonfinite_state_never_counts_as_success(self):
        invalid_states = (
            None,
            {},
            self.state(map_id=float("nan")),
            self.state(party_count=7),
            self.state(party_levels=float("inf")),
            dict(self.state(), story_flags=None),
            self.state(story_flags={object()}),
            self.state(badge_count=9),
        )
        for invalid_state in invalid_states:
            with self.subTest(state=invalid_state):
                self.assertFalse(objective_satisfied("leave_birch_lab", invalid_state))
                self.assertFalse(objective_satisfied("reach_route101", invalid_state))

    def test_start_validation_checks_story_prerequisites(self):
        valid_starter_start = self.state(
            map_id=ROUTE_101_MAP_ID,
            party_count=0,
            party_levels=0,
            story_flags={CLOCK_SET_FLAG_ID},
        )
        self.assertIsNone(objective_start_error("starter", valid_starter_start))

        invalid_starter_start = self.state(
            map_id=ROUTE_101_MAP_ID,
            party_count=0,
            party_levels=0,
            story_flags=set(),
        )
        self.assertIn("clock-set flag", objective_start_error("starter", invalid_starter_start))

        no_starter_party = self.state(
            map_id=BIRCH_LAB_MAP_ID,
            party_count=0,
            party_levels=0,
        )
        self.assertIn("valid starter", objective_start_error("leave_birch_lab", no_starter_party))

        no_rival_flag = self.state(
            map_id=BIRCH_LAB_MAP_ID,
            story_flags={STARTER_OBTAINED_FLAG_ID},
        )
        self.assertIn("rival-defeated milestone", objective_start_error("pokedex", no_rival_flag))

    def test_start_validation_rejects_mismatched_route_milestones(self):
        party_state = self.state(map_id=BIRCH_LAB_MAP_ID)
        self.assertIsNone(objective_start_error("leave_birch_lab", party_state))
        self.assertIsNone(
            objective_start_error(
                "leave_birch_lab", self.state(map_id=LITTLEROOT_TOWN_MAP_ID)
            )
        )
        self.assertIsNotNone(
            objective_start_error("leave_birch_lab", self.state(map_id=ROUTE_101_MAP_ID))
        )

        self.assertIsNone(
            objective_start_error("reach_route101", self.state(map_id=LITTLEROOT_TOWN_MAP_ID))
        )
        self.assertIsNone(
            objective_start_error("reach_route101", self.state(map_id=ROUTE_101_MAP_ID))
        )
        self.assertIsNotNone(
            objective_start_error("reach_route101", self.state(map_id=OLDALE_TOWN_MAP_ID))
        )

    def test_flag_dependent_objectives_reject_unverified_flag_telemetry(self):
        rival_state = self.state(
            map_id=ROUTE_103_MAP_ID,
            story_flags={RIVAL_DEFEATED_ROUTE103_FLAG_ID},
            flag_telemetry_valid=False,
        )
        self.assertFalse(objective_satisfied("defeat_rival", rival_state))
        self.assertIsNotNone(objective_start_error("defeat_rival", rival_state))

        # Map-only navigation does not require story-flag telemetry.
        self.assertTrue(
            objective_satisfied(
                "leave_birch_lab",
                self.state(map_id=LITTLEROOT_TOWN_MAP_ID, flag_telemetry_valid=False),
            )
        )

    def test_unstable_pointer_never_satisfies_map_objective(self):
        state = self.state(
            map_id=LITTLEROOT_TOWN_MAP_ID,
            saveblock_pointer_stable=False,
        )
        self.assertFalse(objective_satisfied("leave_birch_lab", state))
        self.assertIsNotNone(objective_start_error("leave_birch_lab", state))

    def test_first_badge_requires_a_verified_badge_flag_snapshot(self):
        self.assertFalse(objective_satisfied("first_badge", self.state(badge_count=0)))
        self.assertTrue(objective_satisfied("first_badge", self.state(badge_count=1)))
        invalid = self.state(badge_count=1, flag_telemetry_valid=False)
        self.assertFalse(objective_satisfied("first_badge", invalid))

    def test_stage_start_paths_use_the_immediately_preceding_milestone(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            canonical = root / "runtime_data" / "states" / "training_start.state"
            self.assertEqual(
                stage_start_state_path(root, "starter", canonical), canonical.resolve()
            )
            self.assertEqual(
                stage_start_state_path(root, "leave_birch_lab", canonical),
                (root / "runtime_data/states/milestones/starter.state").resolve(),
            )
            self.assertEqual(
                stage_start_state_path(root, "reach_route101", canonical),
                (root / "runtime_data/states/milestones/leave_birch_lab.state").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
