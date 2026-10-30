"""Offline tests for the staged Pokémon Emerald introductory curriculum."""
from __future__ import annotations

import unittest

from curriculum_objectives import (
    BIRCH_LAB_MAP_ID,
    CLOCK_SET_FLAG_ID,
    CURRICULUM_PLAN_ID,
    OBJECTIVE_DEFINITIONS,
    POKEDEX_OBTAINED_FLAG_ID,
    RIVAL_DEFEATED_ROUTE103_FLAG_ID,
    ROUTE_101_MAP_ID,
    ROUTE_103_MAP_ID,
    STARTER_OBTAINED_FLAG_ID,
    STAGES,
    OLDALE_TOWN_MAP_ID,
    objective_satisfied,
    objective_start_error,
)


class CurriculumObjectiveTests(unittest.TestCase):
    def state(self, *, map_id=ROUTE_101_MAP_ID, party_count=1, party_levels=5, flags=(), badges=0):
        return {
            "map_id": map_id,
            "party_count": party_count,
            "party_levels": party_levels,
            "story_flags": set(flags),
            "badge_count": badges,
            "badge_flags": 0,
        }

    def test_stage_order_is_incremental_and_shared(self):
        self.assertEqual(CURRICULUM_PLAN_ID, "emerald_intro_v4")
        self.assertEqual(
            STAGES,
            (
                "starter", "leave_birch_lab", "reach_oldale", "reach_route103",
                "defeat_rival", "return_birch_lab", "pokedex", "first_badge",
            ),
        )
        self.assertEqual(set(STAGES), set(OBJECTIVE_DEFINITIONS))

    def test_map_objectives_are_separate(self):
        starter = self.state(flags={STARTER_OBTAINED_FLAG_ID})
        self.assertFalse(objective_satisfied("starter", starter))
        self.assertTrue(objective_satisfied("starter", self.state(map_id=BIRCH_LAB_MAP_ID, flags={STARTER_OBTAINED_FLAG_ID})))
        self.assertTrue(objective_satisfied("leave_birch_lab", self.state(map_id=ROUTE_101_MAP_ID)))
        self.assertTrue(objective_satisfied("reach_oldale", self.state(map_id=OLDALE_TOWN_MAP_ID)))
        self.assertTrue(objective_satisfied("reach_route103", self.state(map_id=ROUTE_103_MAP_ID)))

    def test_rival_flag_and_return_to_lab_are_distinct_from_pokedex(self):
        base_flags = {STARTER_OBTAINED_FLAG_ID}
        rival_state = self.state(map_id=ROUTE_103_MAP_ID, flags=base_flags | {RIVAL_DEFEATED_ROUTE103_FLAG_ID})
        self.assertTrue(objective_satisfied("defeat_rival", rival_state))
        self.assertFalse(objective_satisfied("return_birch_lab", rival_state))
        lab_after_battle = self.state(map_id=BIRCH_LAB_MAP_ID, flags=base_flags | {RIVAL_DEFEATED_ROUTE103_FLAG_ID})
        self.assertTrue(objective_satisfied("return_birch_lab", lab_after_battle))
        self.assertFalse(objective_satisfied("pokedex", lab_after_battle))
        dex_state = self.state(
            map_id=BIRCH_LAB_MAP_ID,
            flags=base_flags | {RIVAL_DEFEATED_ROUTE103_FLAG_ID, POKEDEX_OBTAINED_FLAG_ID},
        )
        self.assertTrue(objective_satisfied("pokedex", dex_state))

    def test_start_validation_checks_story_prerequisites(self):
        before_starter = self.state(party_count=0, party_levels=0, flags={CLOCK_SET_FLAG_ID})
        self.assertIsNone(objective_start_error("starter", before_starter))
        self.assertIn("live party", objective_start_error("leave_birch_lab", before_starter))
        with_rival_flag = self.state(map_id=ROUTE_103_MAP_ID, flags={STARTER_OBTAINED_FLAG_ID, RIVAL_DEFEATED_ROUTE103_FLAG_ID})
        self.assertIsNone(objective_start_error("return_birch_lab", with_rival_flag))
        no_rival = self.state(flags={STARTER_OBTAINED_FLAG_ID})
        self.assertIn("0x82", objective_start_error("return_birch_lab", no_rival))
        dex_without_rival = self.state(flags={STARTER_OBTAINED_FLAG_ID, POKEDEX_OBTAINED_FLAG_ID})
        self.assertIn("0x82", objective_start_error("pokedex", dex_without_rival))

    def test_start_validation_rejects_mismatched_route_milestones(self):
        oldale_state = self.state(map_id=OLDALE_TOWN_MAP_ID, flags={STARTER_OBTAINED_FLAG_ID})
        self.assertIsNone(objective_start_error("reach_route103", oldale_state))
        self.assertIn("start in Oldale", objective_start_error(
            "reach_route103", self.state(map_id=ROUTE_101_MAP_ID, flags={STARTER_OBTAINED_FLAG_ID})
        ))
        lab_state = self.state(map_id=BIRCH_LAB_MAP_ID, flags={STARTER_OBTAINED_FLAG_ID})
        self.assertIsNone(objective_start_error("leave_birch_lab", lab_state))

    def test_malformed_or_nonfinite_state_never_counts_as_success(self):
        self.assertFalse(objective_satisfied("starter", None))
        bad = self.state()
        bad["party_levels"] = float("nan")
        self.assertFalse(objective_satisfied("starter", bad))
        bad = self.state()
        bad["story_flags"] = "not-a-flag-set"
        self.assertFalse(objective_satisfied("pokedex", bad))


if __name__ == "__main__":
    unittest.main()
