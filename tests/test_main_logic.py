"""Unit tests for Pokémon Emerald environment helpers (no ROM execution needed).

Run inside the project's configured .venv:
    python -m unittest discover -s tests -v

The tests use small fake RAM maps to validate decoding and pure helper behavior.
They do not replace a live-mGBA smoke test against the user's own ROM.
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

import main


class MainLogicTests(unittest.TestCase):
    def test_boolean_environment_flag_parser(self):
        old_value = os.environ.get("TEST_BOOLEAN_FLAG")
        try:
            for value in ("1", "true", "YES", "On"):
                os.environ["TEST_BOOLEAN_FLAG"] = value
                self.assertTrue(main._env_flag("TEST_BOOLEAN_FLAG"))
            for value in ("0", "false", "NO", "off", ""):
                os.environ["TEST_BOOLEAN_FLAG"] = value
                self.assertFalse(main._env_flag("TEST_BOOLEAN_FLAG", True))
            os.environ.pop("TEST_BOOLEAN_FLAG", None)
            self.assertFalse(main._env_flag("TEST_BOOLEAN_FLAG", False))
            self.assertTrue(main._env_flag("TEST_BOOLEAN_FLAG", True))
        finally:
            if old_value is None:
                os.environ.pop("TEST_BOOLEAN_FLAG", None)
            else:
                os.environ["TEST_BOOLEAN_FLAG"] = old_value

    def test_start_and_select_action_indices_are_known(self):
        self.assertEqual(main.ACTION_TO_KEY[7], "START")
        self.assertEqual(main.ACTION_TO_KEY[8], "SELECT")
        self.assertEqual(main.ACTION_TO_KEY[0], "WAIT")
        self.assertTrue(hasattr(main.PPO, "load"))

    def setUp(self):
        self.old_mode = main.MODE
        self.old_host_control = main.HOST_CONTROL_ENABLED

    def tearDown(self):
        main.MODE = self.old_mode
        main.HOST_CONTROL_ENABLED = self.old_host_control

    @staticmethod
    def byte_map_write(memory: dict[int, int], address: int, value: int, length: int = 1) -> None:
        for byte_index in range(length):
            memory[address + byte_index] = (int(value) >> (8 * byte_index)) & 0xFF

    def test_arrow_left_codes_never_quit_gui(self):
        main.MODE = "setup"
        main.HOST_CONTROL_ENABLED = True
        env = object.__new__(main.EmeraldEnv)
        env.stop_requested = False
        env.paused = False
        env._save_training_start_requested = False

        self.assertEqual(env._decode_gui_key(65361), 3)  # X11/Wayland Qt key code
        self.assertEqual(env._decode_gui_key(81), 3)      # low-byte fallback shared with uppercase Q
        self.assertFalse(env.stop_requested)

        self.assertIsNone(env._decode_gui_key(ord("q")))
        self.assertTrue(env.stop_requested)

    def test_escape_stops_and_k_requests_start_capture(self):
        main.MODE = "setup"
        main.HOST_CONTROL_ENABLED = True
        env = object.__new__(main.EmeraldEnv)
        env.stop_requested = False
        env.paused = False
        env._save_training_start_requested = False

        self.assertIsNone(env._decode_gui_key(ord("k")))
        self.assertTrue(env._save_training_start_requested)
        self.assertIsNone(env._decode_gui_key(27))
        self.assertTrue(env.stop_requested)

    def test_watch_has_no_automatic_time_limit_but_train_does(self):
        limit = main.MAX_EPISODE_STEPS
        self.assertTrue(main._episode_time_limit_reached("train", limit))
        self.assertFalse(main._episode_time_limit_reached("train", limit - 1))
        self.assertTrue(main._episode_time_limit_reached("evaluate", limit))
        self.assertFalse(main._episode_time_limit_reached("watch", limit * 100))
        self.assertFalse(main._episode_time_limit_reached("setup", limit * 100))

    def test_rtc_chatter_is_hidden_at_verbose_1_but_visible_at_verbose_2(self):
        old_value = os.environ.get("POKEMON_AI_VERBOSE")
        try:
            rtc_lines = (
                b"GBA Pak Hardware: Got RTC command 4",
                b"GBA Pak Hardware: RTC output byte 40",
            )
            os.environ["POKEMON_AI_VERBOSE"] = "1"
            for line in rtc_lines:
                self.assertTrue(main._is_routine_mgba_chatter(line), line)

            os.environ["POKEMON_AI_VERBOSE"] = "2"
            for line in rtc_lines:
                self.assertFalse(main._is_routine_mgba_chatter(line), line)
        finally:
            if old_value is None:
                os.environ.pop("POKEMON_AI_VERBOSE", None)
            else:
                os.environ["POKEMON_AI_VERBOSE"] = old_value

    def test_known_mgba_memory_and_savedata_chatter_is_hidden_at_verbose_1_but_visible_at_verbose_2(self):
        old_value = os.environ.get("POKEMON_AI_VERBOSE")
        old_filter_value = os.environ.get("POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS")
        try:
            lines = (
                b"GBA Memory: Bad BIOS Load8: 0x00001800",
                b"GBA Memory: Bad BIOS Load32: 0x00001E28",
                b"GBA Memory: Bad memory Load8: 0xe55ec003",
                b"GBA Memory: Bad memory Store8: 0x00000529",
                b"GBA Savedata: Performing flash bank switch to bank 1",
                b"GBA Savedata: Performing flash sector erase at 0xb000",
                b"GBA Savedata: Performing unknown sector-size erase at 0xb000",
            )
            os.environ["POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS"] = "1"
            os.environ["POKEMON_AI_VERBOSE"] = "1"
            for line in lines:
                self.assertTrue(main._is_routine_mgba_chatter(line), line)
            # Do not hide unrelated memory diagnostics or application-level output.
            self.assertFalse(
                main._is_routine_mgba_chatter(b"GBA Memory: Bad memory Load32: 0x02000000")
            )
            self.assertFalse(
                main._is_routine_mgba_chatter(b"[RAM Warning] Reward memory read failed")
            )

            os.environ["POKEMON_AI_VERBOSE"] = "2"
            for line in lines:
                self.assertFalse(main._is_routine_mgba_chatter(line), line)
        finally:
            if old_value is None:
                os.environ.pop("POKEMON_AI_VERBOSE", None)
            else:
                os.environ["POKEMON_AI_VERBOSE"] = old_value
            if old_filter_value is None:
                os.environ.pop("POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS", None)
            else:
                os.environ["POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS"] = old_filter_value

    def test_invalid_dma_source_chatter_is_hidden_at_verbose_1_but_visible_at_verbose_2(self):
        old_value = os.environ.get("POKEMON_AI_VERBOSE")
        old_filter_value = os.environ.get("POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS")
        try:
            dma_lines = (
                b"GBA DMA: Invalid DMA source address: 0x000093AA",
                b"GBA DMA: Invalid DMA source address: 0x000066D0",
                b"GBA DMA: Invalid DMA source address: 0x00006D00",
                b"GBA DMA: Invalid DMA source address: 0x00006DBC",
            )
            os.environ["POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS"] = "1"
            os.environ["POKEMON_AI_VERBOSE"] = "1"
            for line in dma_lines:
                self.assertTrue(main._is_routine_mgba_chatter(line), line)

            os.environ["POKEMON_AI_VERBOSE"] = "2"
            for line in dma_lines:
                self.assertFalse(main._is_routine_mgba_chatter(line), line)
        finally:
            if old_value is None:
                os.environ.pop("POKEMON_AI_VERBOSE", None)
            else:
                os.environ["POKEMON_AI_VERBOSE"] = old_value
            if old_filter_value is None:
                os.environ.pop("POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS", None)
            else:
                os.environ["POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS"] = old_filter_value

    def test_model_env_validation_uses_attached_wrapped_environment_spaces(self):
        observation_space = object()
        action_space = object()
        wrapped_env = SimpleNamespace(
            observation_space=observation_space,
            action_space=action_space,
        )
        model = SimpleNamespace(
            observation_space=observation_space,
            action_space=action_space,
            get_env=lambda: wrapped_env,
        )
        self.assertIs(main._get_compatible_model_env(model, "Unit test"), wrapped_env)

        incompatible_model = SimpleNamespace(
            observation_space=object(),
            action_space=action_space,
            get_env=lambda: wrapped_env,
        )
        with self.assertRaisesRegex(RuntimeError, "attached wrapped environment"):
            main._get_compatible_model_env(incompatible_model, "Unit test")

        unattached_model = SimpleNamespace(
            observation_space=observation_space,
            action_space=action_space,
            get_env=lambda: None,
        )
        with self.assertRaisesRegex(RuntimeError, "no attached environment"):
            main._get_compatible_model_env(unattached_model, "Unit test")

    def test_action_distribution_diagnostic_is_sorted_and_includes_percentages(self):
        distribution = main._format_action_distribution({8: 15, 3: 5, 0: 0})
        self.assertEqual(distribution, "SELECT=15 (75.0%), LEFT=5 (25.0%)")
        self.assertEqual(main._format_action_distribution({}), "none")

    def test_evaluate_profile_is_single_headless_and_does_not_record_video(self):
        preset = main._PROFILE_PRESETS["evaluate"]["env"]
        self.assertEqual(preset["POKEMON_AI_MODE"], "evaluate")
        self.assertEqual(preset["POKEMON_AI_GUI_MODE"], "0")
        self.assertEqual(preset["POKEMON_AI_RECORD_VIDEO"], "0")
        self.assertEqual(preset["POKEMON_AI_AUTO_START_TENSORBOARD"], "0")

    def test_wait_action_releases_keys_with_raw_and_fallback_bindings(self):
        env = object.__new__(main.EmeraldEnv)
        raw_calls = []
        env.core = SimpleNamespace(set_keys=lambda **kwargs: raw_calls.append(kwargs))
        env._set_action_keys(0)  # WAIT
        self.assertEqual(raw_calls, [{"raw": 0}])

        fallback_calls = []
        def fallback(*args, **kwargs):
            if "raw" in kwargs:
                raise TypeError("raw keyword not supported")
            fallback_calls.append(args)
        env.core = SimpleNamespace(set_keys=fallback)
        env._set_action_keys(0)
        self.assertEqual(fallback_calls, [()])

    def test_saveblock1_state_decoder_reads_party_location_flags(self):
        env = object.__new__(main.EmeraldEnv)
        memory: dict[int, int] = {}
        env._read8 = lambda address: memory.get(int(address), 0)

        sb1 = 0x02010000
        self.byte_map_write(memory, main.SAVE_BLOCK1_PTR_ADDRESS, sb1, 4)
        self.byte_map_write(memory, sb1 + main.SAVE_BLOCK1_POS_X_OFFSET, 12, 2)
        self.byte_map_write(memory, sb1 + main.SAVE_BLOCK1_POS_Y_OFFSET, 34, 2)
        memory[sb1 + main.SAVE_BLOCK1_LOCATION_OFFSET] = 1
        memory[sb1 + main.SAVE_BLOCK1_LOCATION_OFFSET + 1] = 2
        memory[sb1 + main.SAVE_BLOCK1_PARTY_COUNT_OFFSET] = 1
        mon = sb1 + main.SAVE_BLOCK1_PARTY_OFFSET
        memory[mon + main.PARTY_LEVEL_OFFSET] = 5

        flags_base = sb1 + main.SAVE_BLOCK1_FLAGS_OFFSET
        for flag_id in (main.CLOCK_SET_FLAG_ID, main.STARTER_OBTAINED_FLAG_ID, 0x500, 0x867):
            address = flags_base + (flag_id >> 3)
            memory[address] = memory.get(address, 0) | (1 << (flag_id & 7))

        state = env._read_game_state()
        self.assertEqual(state["party_count"], 1)
        self.assertEqual(state["party_levels"], 5)
        self.assertEqual((state["player_x"], state["player_y"]), (12, 34))
        self.assertEqual((state["map_group"], state["map_number"], state["map_id"]), (1, 2, 258))
        self.assertEqual(state["badge_count"], 1)
        self.assertIn(main.CLOCK_SET_FLAG_ID, state["story_flags"])
        self.assertIn(main.STARTER_OBTAINED_FLAG_ID, state["story_flags"])
        self.assertIn(0x500, state["trainer_flags"])
        self.assertNotIn("route101_state", state)

    def test_invalid_saveblock_pointer_is_rejected(self):
        env = object.__new__(main.EmeraldEnv)
        env._read8 = lambda address: 0
        with self.assertRaisesRegex(ValueError, "SaveBlock1 pointer"):
            env._read_game_state()

    def test_training_start_requires_route101_clock_and_empty_party(self):
        base = {
            "player_x": 11,
            "player_y": 15,
            "map_id": main.ROUTE_101_MAP_ID,
            "party_count": 0,
            "party_levels": 0,
            "badge_count": 0,
            "badge_flags": 0,
            "story_flags": {main.CLOCK_SET_FLAG_ID},
        }
        self.assertIsNone(main._training_start_validation_error(base))
        # Optional unrelated RAM fields do not prescribe a route or policy direction.
        self.assertIsNone(main._training_start_validation_error(dict(base, any_script_var=42)))

        bad_clock = dict(base, story_flags=set())
        self.assertIn("clock-set flag", main._training_start_validation_error(bad_clock))
        already_rescued = dict(base, story_flags={main.CLOCK_SET_FLAG_ID, main.BIRCH_RESCUED_FLAG_ID})
        self.assertIn("before rescuing Birch", main._training_start_validation_error(already_rescued))

        wrong_map = dict(base, map_id=258)
        self.assertIn("Route 101", main._training_start_validation_error(wrong_map))

        with_party = dict(base, party_count=1, party_levels=5)
        self.assertIn("before receiving a starter", main._training_start_validation_error(with_party))

        with_stray_levels = dict(base, party_levels=5)
        self.assertIn("empty party", main._training_start_validation_error(with_stray_levels))

    def test_existing_reward_weights_are_preserved_and_time_cost_is_small(self):
        self.assertGreater(main.STORY_FLAG_REWARDS[main.BIRCH_RESCUED_FLAG_ID], 5.0)
        self.assertGreater(main.STORY_FLAG_REWARDS[main.STARTER_OBTAINED_FLAG_ID], 5.0)
        self.assertAlmostEqual(main.NEW_TILE_REWARD / main.STEP_PENALTY, 400.0, delta=1e-5)
        self.assertGreater(main.TASK_SUCCESS_REWARD, main.STORY_FLAG_REWARDS[main.BIRCH_RESCUED_FLAG_ID])
        for component in ("tile", "map", "screen", "story", "party", "level", "badge", "trainer", "loop", "task_success"):
            self.assertIn(component, main.REWARD_COMPONENT_NAMES)
        self.assertNotIn("birch_approach", main.REWARD_COMPONENT_NAMES)

    def test_novelty_reward_decays_across_repeated_visits(self):
        base = main.NEW_TILE_REWARD
        first = main._diminishing_novelty_reward(base, 0)
        second = main._diminishing_novelty_reward(base, 1)
        tenth = main._diminishing_novelty_reward(base, 10)
        self.assertAlmostEqual(first, base)
        self.assertLess(second, first)
        self.assertLess(tenth, second)
        self.assertGreaterEqual(main._diminishing_novelty_reward(base, -5), 0.0)

    def test_screen_signature_is_stable_and_detects_large_layout_changes(self):
        unchanged_a = np.zeros((main.OBS_HEIGHT, main.OBS_WIDTH, 3), dtype=np.uint8)
        unchanged_b = unchanged_a.copy()
        changed = unchanged_a.copy()
        changed[:, :main.OBS_WIDTH // 2, :] = 255
        self.assertEqual(main._screen_signature(unchanged_a), main._screen_signature(unchanged_b))
        self.assertNotEqual(main._screen_signature(unchanged_a), main._screen_signature(changed))
        self.assertIsNone(main._screen_signature(None))

    def test_previous_action_and_stagnation_are_in_state_features_without_route_hints(self):
        env = object.__new__(main.EmeraldEnv)
        env.visited_tiles = set()
        env.visited_maps = set()
        env.episode_step_counter = main.LOOP_STAGNATION_STEPS
        env.last_novelty_step = 0
        env.last_action_id = 7  # START remains a legal action and is represented in observation.
        state = {
            "map_group": 0, "map_number": 16, "map_id": 16,
            "player_x": 11, "player_y": 15, "party_count": 0,
            "party_levels": 0, "badge_flags": 0, "badge_count": 0,
            "story_flags": {main.CLOCK_SET_FLAG_ID}, "trainer_flags": set(),
        }
        features = env._encode_state_features(state)
        self.assertEqual(features.shape, (len(main.STATE_FEATURE_NAMES),))
        start_index = main.STATE_FEATURE_NAMES.index("previous_action_start")
        self.assertEqual(float(features[start_index]), 1.0)
        stagnation_index = main.STATE_FEATURE_NAMES.index("steps_since_progress")
        self.assertEqual(float(features[stagnation_index]), 1.0)
        self.assertNotIn("birch_target_distance", main.STATE_FEATURE_NAMES)
        self.assertNotIn("rescue_trigger_reached", main.STATE_FEATURE_NAMES)
        self.assertFalse(hasattr(main, "_route101_approach_reward"))
        self.assertIn(7, main.ACTION_TO_KEY)
        self.assertIn(8, main.ACTION_TO_KEY)

    def test_visited_mask_compresses_coordinates_above_63(self):
        env = object.__new__(main.EmeraldEnv)
        env.visited_tiles = {(256, 80, 100), (257, 20, 30)}
        mask = env._visited_mask(256)
        self.assertEqual(mask.shape, (main.OBS_VISITED_GRID_SIZE, main.OBS_VISITED_GRID_SIZE))
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(mask[50, 40], 1.0)
        self.assertEqual(float(mask.sum()), 1.0)

    def test_state_feature_vector_matches_declared_shape_and_range(self):
        self.assertEqual((main.OBS_WIDTH, main.OBS_HEIGHT), (120, 80))
        env = object.__new__(main.EmeraldEnv)
        env.visited_tiles = {(256, 10, 12)}
        env.visited_maps = {256}
        env.episode_step_counter = 100
        state = {
            "map_group": 1,
            "map_number": 2,
            "map_id": 258,
            "player_x": 10,
            "player_y": 12,
            "party_count": 1,
            "party_levels": 5,
            "badge_flags": 1,
            "badge_count": 1,
            "story_flags": {main.CLOCK_SET_FLAG_ID, main.STARTER_OBTAINED_FLAG_ID},
            "trainer_flags": {0x500},
        }
        features = env._encode_state_features(state)
        self.assertEqual(features.shape, (len(main.STATE_FEATURE_NAMES),))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(((features >= 0.0) & (features <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
