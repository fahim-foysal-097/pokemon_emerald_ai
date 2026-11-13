"""Regression tests for the v6 Emerald objective map and prerequisite contract."""
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
    evaluation_gate_unreachable,
    is_navigation_objective,
    VAR_ROUTE110_STATE_ID,
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
        self.assertEqual(CURRICULUM_PLAN_ID, "emerald_intro_v6")
        self.assertEqual(len(STAGES), 58)
        self.assertEqual(len(set(STAGES)), 58)
        self.assertEqual(set(STAGES), set(__import__("curriculum_objectives").OBJECTIVE_DEFINITIONS))
        self.assertEqual(STAGES[0], "starter")
        self.assertEqual(STAGES[1], "leave_birch_lab")
        self.assertEqual(STAGES[2], "reach_route101")
        self.assertIn("first_badge", STAGES)
        self.assertIn("second_badge", STAGES)
        self.assertIn("eighth_badge", STAGES)
        self.assertEqual(STAGES[-1], "defeat_champion")
        for earlier, later in zip(STAGES, STAGES[1:]):
            self.assertLess(STAGES.index(earlier), STAGES.index(later))
        self.assertGreater(stage_episode_rollouts("reach_route101"), 0)
        self.assertGreater(stage_episode_rollouts("first_badge"), 0)
        self.assertGreater(stage_episode_rollouts("defeat_champion"), 0)

    def test_destination_stages_use_navigation_reward_profile(self):
        for stage in (
            "leave_birch_lab", "reach_route101", "reach_oldale", "reach_route103",
            "return_birch_lab", "sail_to_dewford", "surf_route118", "reach_ever_grande",
        ):
            with self.subTest(stage=stage):
                self.assertTrue(is_navigation_objective(stage))
        for stage in (
            "starter", "defeat_rival", "pokedex", "first_badge", "oceanic_museum",
            "pass_victory_road", "defeat_champion",
        ):
            with self.subTest(stage=stage):
                self.assertFalse(is_navigation_objective(stage))
        self.assertFalse(is_navigation_objective("unknown_stage"))

    def test_evaluation_futility_uses_integer_success_requirements(self):
        # Five episodes at a 90% threshold require five successes. One failure
        # makes the gate unreachable, so the remaining four tests can be skipped.
        self.assertTrue(evaluation_gate_unreachable(0, 1, 5, 0.9))
        self.assertFalse(evaluation_gate_unreachable(1, 1, 5, 0.9))

        # Ten episodes at 90% require nine successes: one failure is still
        # recoverable, while two failures are not.
        self.assertFalse(evaluation_gate_unreachable(0, 1, 10, 0.9))
        self.assertTrue(evaluation_gate_unreachable(0, 2, 10, 0.9))
        self.assertFalse(evaluation_gate_unreachable(1, 2, 10, 0.9))

        # At an 80% gate over five, one failure still allows exactly four wins.
        self.assertFalse(evaluation_gate_unreachable(0, 1, 5, 0.8))
        self.assertTrue(evaluation_gate_unreachable(0, 2, 5, 0.8))
        self.assertFalse(evaluation_gate_unreachable(5, 5, 5, 0.9))

    def test_evaluation_futility_rejects_invalid_counts_and_thresholds(self):
        for args in ((0, 0, 0, 0.9), (0, 2, 1, 0.9), (2, 1, 5, 0.9), (0, 1, 5, 0.0), (0, 1, 5, 1.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluation_gate_unreachable(*args)

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


    def test_story_event_flags_match_vanilla_emerald_and_reject_old_false_positives(self):
        from curriculum_objectives import (
            CURRICULUM_STORY_FLAG_IDS,
            FLAG_WALLY_TUTORIAL_COMPLETED, FLAG_RESCUED_DEVON_RESEARCHER,
            FLAG_RESCUED_DEVON_RESEARCHER_AQUA_GRUNT,
            FLAG_RUSTURF_TUNNEL_PEEKO_RESCUED, FLAG_STEVEN_LETTER_DELIVERED,
            FLAG_POKENAV_RECEIVED, FLAG_OCEANIC_MUSEUM_GRUNTS_HIDDEN,
            FLAG_OCEANIC_MUSEUM_GRUNT1_HIDDEN, FLAG_OCEANIC_MUSEUM_GRUNT2_HIDDEN,
            FLAG_OCEANIC_MUSEUM_ARCHIE_HIDDEN, FLAG_OCEANIC_MUSEUM_STERN_HIDDEN,
            FLAG_RUSTURF_TUNNEL_OPENED, FLAG_METEOR_FALLS_TEAM_SCENE,
            FLAG_RECEIVED_CASTFORM, FLAG_RIVAL_DEFEATED_ROUTE119,
            FLAG_RECEIVED_DEVON_SCOPE, FLAG_MT_PYRE_MAXIE_HIDDEN,
            FLAG_RECEIVED_RED_OR_BLUE_ORB, FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN,
            FLAG_AWAKENED_RAYQUAZA, FLAG_HIDE_SKY_PILLAR_TOP_RAYQUAZA_STILL, FLAG_RAYQUAZA_CALMED_WEATHER,
            FLAG_DEFEATED_WALLY_VICTORY_ROAD, FLAG_SYS_GAME_CLEAR,
            POKENAV_OBTAINED_FLAG_ID,
        )
        party = dict(map_id=BIRCH_LAB_MAP_ID)
        cases = (
            ("wally_tutorial", 0x84, {FLAG_WALLY_TUTORIAL_COMPLETED}),
            ("defeat_aqua_woods", 0x73, {FLAG_RESCUED_DEVON_RESEARCHER, FLAG_RESCUED_DEVON_RESEARCHER_AQUA_GRUNT}),
            ("rusturf_tunnel_rescue", 0x85, {FLAG_RUSTURF_TUNNEL_PEEKO_RESCUED}),
            ("deliver_letter_steven", 0x80, {FLAG_STEVEN_LETTER_DELIVERED}),
            ("smash_rusturf_rock", 0x71, {FLAG_RUSTURF_TUNNEL_OPENED}),
            ("meteor_falls", 0x6A, {FLAG_METEOR_FALLS_TEAM_SCENE}),
            ("weather_institute", 0x72, {FLAG_RECEIVED_CASTFORM}),
            ("defeat_rival_route119", 0x87, {FLAG_RIVAL_DEFEATED_ROUTE119}),
            ("devon_scope", 0x7E, {FLAG_RECEIVED_DEVON_SCOPE}),
            ("seafloor_cavern", 0x76, {FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN}),
            ("sky_pillar_rayquaza", 0x66, {FLAG_AWAKENED_RAYQUAZA}),
            ("calm_weather", 0x8E, {FLAG_RAYQUAZA_CALMED_WEATHER}),
            ("pass_victory_road", 0x88, {FLAG_DEFEATED_WALLY_VICTORY_ROAD}),
        )
        for objective, wrong_flag, correct_flags in cases:
            with self.subTest(objective=objective):
                self.assertFalse(objective_satisfied(
                    objective, self.state(**party, story_flags={wrong_flag})
                ))
                self.assertTrue(objective_satisfied(
                    objective, self.state(**party, story_flags=correct_flags)
                ))

        # 0xD3 is FLAG_DEFEATED_RIVAL_RUSTBORO, not the Route 110 rival event.
        self.assertFalse(objective_satisfied(
            "defeat_rival_route110",
            self.state(story_flags={0xD3}, route110_state=0, route110_var_telemetry_valid=True),
        ))
        self.assertFalse(objective_satisfied(
            "defeat_rival_route110",
            self.state(route110_state=1, route110_var_telemetry_valid=False),
        ))
        self.assertFalse(objective_satisfied(
            "defeat_rival_route110", self.state(route110_state=0, route110_var_telemetry_valid=True)
        ))
        self.assertTrue(objective_satisfied(
            "defeat_rival_route110", self.state(route110_state=1, route110_var_telemetry_valid=True)
        ))
        self.assertEqual(VAR_ROUTE110_STATE_ID, 0x4069)
        self.assertFalse(objective_satisfied(
            "sky_pillar_rayquaza", self.state(story_flags={FLAG_HIDE_SKY_PILLAR_TOP_RAYQUAZA_STILL})
        ))
        self.assertEqual(FLAG_AWAKENED_RAYQUAZA, 0x305)
        self.assertFalse(objective_satisfied("receive_pokenav", self.state(story_flags={0x80})))
        self.assertTrue(objective_satisfied("receive_pokenav", self.state(story_flags={FLAG_POKENAV_RECEIVED})))
        self.assertTrue(objective_satisfied("receive_pokenav", self.state(story_flags={POKENAV_OBTAINED_FLAG_ID})))
        self.assertFalse(objective_satisfied("oceanic_museum", self.state(story_flags={0x61})))
        museum_flags = {
            FLAG_OCEANIC_MUSEUM_GRUNTS_HIDDEN, FLAG_OCEANIC_MUSEUM_GRUNT1_HIDDEN,
            FLAG_OCEANIC_MUSEUM_GRUNT2_HIDDEN, FLAG_OCEANIC_MUSEUM_ARCHIE_HIDDEN,
            FLAG_OCEANIC_MUSEUM_STERN_HIDDEN,
        }
        self.assertFalse(objective_satisfied("oceanic_museum", self.state(story_flags=museum_flags - {FLAG_OCEANIC_MUSEUM_STERN_HIDDEN})))
        self.assertTrue(objective_satisfied("oceanic_museum", self.state(story_flags=museum_flags)))
        self.assertFalse(objective_satisfied("mt_pyre", self.state(story_flags={0x67})))
        self.assertFalse(objective_satisfied("magma_hideout", self.state(story_flags={0x67})))
        self.assertTrue(objective_satisfied("magma_hideout", self.state(story_flags={0x6F})))
        self.assertTrue(objective_satisfied("mt_pyre", self.state(story_flags={FLAG_MT_PYRE_MAXIE_HIDDEN, FLAG_RECEIVED_RED_OR_BLUE_ORB})))
        self.assertTrue(objective_satisfied("defeat_champion", self.state(story_flags={FLAG_SYS_GAME_CLEAR}, party_count=0, party_levels=0)))
        self.assertIn(FLAG_SYS_GAME_CLEAR, CURRICULUM_STORY_FLAG_IDS)
        self.assertIn(POKENAV_OBTAINED_FLAG_ID, CURRICULUM_STORY_FLAG_IDS)

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

    def test_route110_rival_stage_requires_verified_save_var_telemetry(self):
        state = self.state(map_id=25, route110_state=0, route110_var_telemetry_valid=False)
        self.assertIn("Route 110 progression variable", objective_start_error("defeat_rival_route110", state))
        valid_start = self.state(map_id=25, route110_state=0, route110_var_telemetry_valid=True)
        self.assertIsNone(objective_start_error("defeat_rival_route110", valid_start))

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
