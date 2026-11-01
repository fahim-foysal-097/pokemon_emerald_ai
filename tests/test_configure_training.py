import importlib.util
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "configure_training.py"
spec = importlib.util.spec_from_file_location("configure_training", SCRIPT)
ct = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = ct
spec.loader.exec_module(ct)


class ConfigureTrainingTests(unittest.TestCase):
    def setUp(self):
        self.stages = [
            ct.StageInfo("starter", "starter", "Starter + opening battle"),
            ct.StageInfo("leave_birch_lab", "exit Birch's Lab", "Enter Littleroot Town"),
            ct.StageInfo("reach_route101", "reach Route 101", "Exit Littleroot Town"),
            ct.StageInfo("reach_oldale", "reach Oldale", "Travel"),
            ct.StageInfo("reach_route103", "reach Route 103", "Travel"),
            ct.StageInfo("defeat_rival", "defeat rival", "Battle"),
            ct.StageInfo("return_birch_lab", "return to lab", "Return"),
            ct.StageInfo("pokedex", "obtain Pokédex", "Talk to Birch"),
            ct.StageInfo("first_badge", "first badge", "Long objective"),
        ]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project_dir = Path(temporary.name)
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / "curriculum_objectives.py",
            self.project_dir / "curriculum_objectives.py",
        )

    def build(self, **kwargs):
        base = dict(
            mode="stage", profile="fast_train", stage="starter", stages=self.stages,
            project_dir=self.project_dir, hours=3, speed=80, envs=4,
            n_steps=2048, batch_size=256,
        )
        base.update(kwargs)
        return ct.build_plan(**base)

    def test_stage_episode_caps_are_per_env_rollout_aligned_and_stage_specific(self):
        starter = self.build(stage="starter")
        lab = self.build(stage="leave_birch_lab")
        pokedex = self.build(stage="pokedex")
        badge = self.build(stage="first_badge")
        self.assertEqual(starter.max_episode_steps, 6 * 2048)
        route101 = self.build(stage="reach_route101")
        self.assertEqual(lab.max_episode_steps, 15 * 2048)
        self.assertEqual(route101.max_episode_steps, 20 * 2048)
        self.assertEqual(pokedex.max_episode_steps, 8 * 2048)
        self.assertEqual(badge.max_episode_steps, 64 * 2048)
        self.assertLess(starter.max_episode_steps, lab.max_episode_steps)
        self.assertLess(lab.max_episode_steps, badge.max_episode_steps)

    def test_80_total_speed_calculates_hourly_budget_correctly(self):
        plan = self.build(hours=1, speed=80)
        self.assertEqual(plan.rollout_size, 8192)
        self.assertEqual(plan.total_transitions, (288000 // 8192) * 8192)
        self.assertAlmostEqual(plan.estimated_hours, plan.total_transitions / 80 / 3600)
        self.assertAlmostEqual(plan.estimated_episode_minutes, (6 * 2048) / 20 / 60)

    def test_stage_plan_sets_objective_state_resume_episode_and_rollout_intervals(self):
        plan = self.build(stage="reach_oldale")
        values = ct.values_for_plan(plan)
        self.assertEqual(values["POKEMON_AI_OBJECTIVE"], "reach_oldale")
        self.assertTrue(values["POKEMON_AI_TRAINING_START_STATE"].endswith("milestones/reach_route101.state"))
        self.assertEqual(values["POKEMON_AI_MAX_EPISODE_STEPS"], str(20 * 2048))
        self.assertEqual(values["POKEMON_AI_RESUME"], "1")
        self.assertEqual(plan.total_transitions % plan.rollout_size, 0)
        self.assertEqual(int(values["POKEMON_AI_CHECKPOINT_EVERY"]) % plan.rollout_size, 0)
        self.assertEqual(int(values["POKEMON_AI_PROGRESS_EVERY"]) % plan.rollout_size, 0)

    def test_starter_stage_always_uses_canonical_pre_starter_path(self):
        plan = self.build(stage="starter")
        self.assertTrue(plan.start_state.endswith("runtime_data/states/training_start.state"))
        self.assertEqual(
            ct.values_for_plan(plan)["POKEMON_AI_CANONICAL_START_STATE"],
            "runtime_data/states/training_start.state",
        )

    def test_entire_curriculum_uses_the_immediately_preceding_milestone(self):
        expected = {
            "starter": "runtime_data/states/training_start.state",
            "leave_birch_lab": "runtime_data/states/milestones/starter.state",
            "reach_route101": "runtime_data/states/milestones/leave_birch_lab.state",
            "reach_oldale": "runtime_data/states/milestones/reach_route101.state",
            "reach_route103": "runtime_data/states/milestones/reach_oldale.state",
            "defeat_rival": "runtime_data/states/milestones/reach_route103.state",
            "return_birch_lab": "runtime_data/states/milestones/defeat_rival.state",
            "pokedex": "runtime_data/states/milestones/return_birch_lab.state",
            "first_badge": "runtime_data/states/milestones/pokedex.state",
        }
        for stage, suffix in expected.items():
            with self.subTest(stage=stage):
                resolved = ct.stage_start_state(self.project_dir, self.stages, stage)
                self.assertTrue(str(resolved).endswith(suffix), (stage, resolved))

    def test_canonical_path_cannot_be_a_milestone_for_any_selected_stage(self):
        for stage in ("starter", "leave_birch_lab", "first_badge"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "cannot point inside"):
                self.build(
                    stage=stage,
                    canonical_start_state="runtime_data/states/milestones/starter.state",
                )

    def test_later_stage_start_state_is_separate_from_canonical_start_state(self):
        plan = self.build(
            stage="leave_birch_lab",
            canonical_start_state="runtime_data/states/training_start.state",
        )
        values = ct.values_for_plan(plan)
        self.assertTrue(values["POKEMON_AI_TRAINING_START_STATE"].endswith("milestones/starter.state"))
        self.assertEqual(
            values["POKEMON_AI_CANONICAL_START_STATE"],
            "runtime_data/states/training_start.state",
        )

    def test_checkpoint_interval_is_absolute_and_rounded_to_complete_rollout(self):
        preserved = self.build(checkpoint_interval=122880)
        self.assertEqual(preserved.checkpoint_every, 122880)
        rounded = self.build(checkpoint_interval=10000)
        self.assertEqual(rounded.checkpoint_every, 16384)
        self.assertTrue(any("rounded up" in warning for warning in rounded.warnings))

    def test_curriculum_chunk_covers_longest_remaining_episode_and_rollout(self):
        plan = self.build(mode="curriculum", stage="reach_oldale")
        self.assertEqual(plan.max_episode_steps, 64 * 2048)  # first_badge is longest remaining
        self.assertEqual(plan.chunk_transitions, plan.max_episode_steps * 4)
        self.assertEqual(plan.chunk_transitions % plan.rollout_size, 0)
        self.assertEqual(
            plan.curriculum_transitions,
            plan.remaining_stages * plan.chunks_per_stage * plan.chunk_transitions,
        )

    def test_stage_only_uses_one_duration_sized_chunk_and_separate_checkpoint_cadence(self):
        plan = self.build(mode="stage-only", stage="starter", hours=3, speed=80)
        values = ct.values_for_plan(plan)
        expected_budget = (int(3 * 3600 * 80) // plan.rollout_size) * plan.rollout_size
        self.assertEqual(plan.max_episode_steps, 6 * 2048)
        self.assertEqual(plan.chunk_transitions, expected_budget)
        self.assertEqual(plan.chunks_per_stage, 1)
        self.assertEqual(plan.total_transitions, expected_budget)
        self.assertEqual(int(values["POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS"]), expected_budget)
        self.assertEqual(values["POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE"], "1")
        self.assertEqual(int(values["POKEMON_AI_CHECKPOINT_EVERY"]), 122880)
        self.assertEqual(values["POKEMON_AI_LATEST_BACKUP_ENABLED"], "1")

    def test_stage_only_short_decimal_hours_change_budget_and_do_not_round_up_to_checkpoint(self):
        for hours, expected_rollouts in ((0.2, 7), (0.3, 10)):
            with self.subTest(hours=hours):
                plan = self.build(mode="stage-only", stage="starter", hours=hours, speed=80)
                values = ct.values_for_plan(plan)
                expected = expected_rollouts * 8192
                self.assertEqual(plan.total_transitions, expected)
                self.assertEqual(plan.chunk_transitions, expected)
                self.assertEqual(plan.curriculum_transitions, expected)
                self.assertEqual(plan.chunks_per_stage, 1)
                self.assertAlmostEqual(plan.estimated_hours, expected / 80 / 3600)
                # The existing project interval remains independent of run duration.
                self.assertEqual(int(values["POKEMON_AI_CHECKPOINT_EVERY"]), 122880)

    def test_stage_only_shorter_than_episode_cap_warns_without_extending_duration(self):
        plan = self.build(mode="stage-only", stage="leave_birch_lab", hours=0.2, speed=80)
        self.assertEqual(plan.total_transitions, 7 * 8192)
        self.assertEqual(plan.max_episode_steps, 15 * 2048)
        self.assertTrue(any("episodes may be unfinished" in warning for warning in plan.warnings))

    def test_stage_only_uses_duration_derived_chunks_and_disables_evaluation_promotion(self):
        plan = self.build(mode="stage-only", stage="leave_birch_lab", hours=3, speed=80)
        values = ct.values_for_plan(plan)
        self.assertEqual(plan.max_episode_steps, 15 * 2048)
        self.assertEqual(plan.chunk_transitions, plan.total_transitions)
        self.assertEqual(plan.chunks_per_stage, 1)
        self.assertEqual(plan.total_transitions, (int(3 * 3600 * 80) // 8192) * 8192)
        self.assertEqual(values["POKEMON_AI_CURRICULUM_STAGE_ONLY"], "1")
        self.assertEqual(values["POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS"], str(plan.total_transitions))
        self.assertEqual(values["POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE"], "1")
        self.assertEqual(values["POKEMON_AI_OBJECTIVE"], "leave_birch_lab")

    def test_plan_configures_safe_interrupt_wait_windows(self):
        plan = self.build(mode="stage-only")
        values = ct.values_for_plan(plan)
        self.assertEqual(values["POKEMON_AI_SHUTDOWN_TIMEOUT"], "900")
        self.assertEqual(values["POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT"], "60")

    def test_normal_modes_explicitly_disable_stage_only_flag(self):
        for mode in ("stage", "curriculum"):
            with self.subTest(mode=mode):
                plan = self.build(mode=mode)
                self.assertEqual(ct.values_for_plan(plan)["POKEMON_AI_CURRICULUM_STAGE_ONLY"], "0")

    def test_gui_stage_uses_one_effective_env_and_does_not_overwrite_num_envs(self):
        plan = self.build(profile="gui_train", envs=4, speed=80)
        self.assertEqual(plan.rollout_size, 2048)
        self.assertEqual(plan.envs, 1)
        self.assertNotIn("POKEMON_AI_NUM_ENVS", ct.values_for_plan(plan))

    def test_episode_override_rounds_up_to_environment_rollout(self):
        plan = self.build(stage="starter", episode_steps_override=5000)
        self.assertEqual(plan.max_episode_steps, 3 * 2048)

    def test_missing_previous_milestone_path_is_reported_in_plan_not_guessed(self):
        plan = self.build(stage="pokedex")
        self.assertTrue(plan.start_state.endswith("milestones/return_birch_lab.state"))

    def test_dotenv_updates_managed_keys_and_preserves_unrelated_values(self):
        source = ["# keep", "POKEMON_AI_OBJECTIVE=starter", "UNRELATED=keep", "POKEMON_AI_OBJECTIVE=pokedex"]
        result = ct.update_env_text(source, {"POKEMON_AI_OBJECTIVE": "pokedex", "POKEMON_AI_MAX_EPISODE_STEPS": "16384"})
        self.assertEqual(result.count("POKEMON_AI_OBJECTIVE="), 1)
        self.assertIn("UNRELATED=keep", result)
        self.assertIn("# keep", result)
        self.assertIn("POKEMON_AI_MAX_EPISODE_STEPS=16384", result)

    def test_legacy_stage_specific_milestone_is_not_adopted_as_canonical_path(self):
        values = {"POKEMON_AI_TRAINING_START_STATE": "runtime_data/states/milestones/starter.state"}
        canonical = ct.configured_canonical_start_state(self.project_dir, values)
        self.assertEqual(canonical, "runtime_data/states/training_start.state")

    def test_explicit_canonical_path_wins_over_stage_specific_path(self):
        values = {
            "POKEMON_AI_CANONICAL_START_STATE": "runtime_data/states/training_start.state",
            "POKEMON_AI_TRAINING_START_STATE": "runtime_data/states/milestones/starter.state",
        }
        canonical = ct.configured_canonical_start_state(self.project_dir, values)
        self.assertEqual(canonical, "runtime_data/states/training_start.state")

    def test_invalid_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            self.build(stage="unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
