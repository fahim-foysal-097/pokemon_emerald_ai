"""Standard-library source guards for regressions that do not need mGBA/SB3."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = PROJECT_ROOT / "main.py"
CURRICULUM_SOURCE = PROJECT_ROOT / "run_curriculum.py"


def _method_source(tree: ast.AST, class_name: str, method_name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return ast.get_source_segment(MAIN_SOURCE.read_text(encoding="utf-8"), child) or ""
    raise AssertionError(f"Could not find {class_name}.{method_name}")


class MainSourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MAIN_SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(MAIN_SOURCE))

    def test_party_ram_debug_signature_is_not_cleared_on_episode_reset_or_match(self):
        reset = _method_source(self.tree, "EmeraldEnv", "reset")
        step = _method_source(self.tree, "EmeraldEnv", "step")
        self.assertNotIn("self._last_party_debug_signature = None", reset)
        self.assertNotIn("self._last_party_debug_signature = None", step)

    def test_parent_profiler_is_installed_only_for_opt_in_training(self):
        self.assertIn("from runtime_profiler import RuntimeProfiler", self.source)
        self.assertIn("RuntimeProfiler(enabled=TIMING_DIAGNOSTICS)", self.source)
        self.assertIn("runtime_profiler.instrument_training(model, env, globals())", self.source)
        self.assertIn("runtime_profiler.restore()", self.source)

    def test_interrupt_never_aborts_rollout_before_ppo_update(self):
        callback_step = _method_source(self.tree, "StopAndCheckpointCallback", "_on_step")
        self.assertNotIn("return False", callback_step)
        self.assertNotIn("atomic_save_model", callback_step)
        self.assertIn("self._checkpoint_due = True", callback_step)

        safe_train = _method_source(self.tree, "SafeRecurrentPPO", "train")
        self.assertLess(safe_train.index("super().train()"), safe_train.index("_save_after_optimizer_update"))
        collect = _method_source(self.tree, "SafeRecurrentPPO", "collect_rollouts")
        self.assertIn("SAFE_STOP_AFTER_UPDATE", collect)

    def test_runtime_start_state_uses_shared_stage_resolver(self):
        self.assertIn("resolve_runtime_start_state_path(", self.source)
        self.assertIn('mode=MODE,', self.source)
        self.assertIn('canonical_start_state=os.environ.get("POKEMON_AI_CANONICAL_START_STATE"', self.source)

    def test_setup_profile_cannot_overwrite_a_stage_milestone(self):
        self.assertIn('if MODE == "setup" and _configured_start_state:', self.source)
        self.assertIn("setup saves only to the canonical", self.source)
        setup_handler = _method_source(self.tree, "EmeraldEnv", "step")
        self.assertIn("_save_state_file(TRAINING_START_STATE)", setup_handler)
        self.assertIn('objective="starter"', setup_handler)

    def test_checkpoint_default_matches_preserved_project_interval(self):
        self.assertIn('os.environ.get("POKEMON_AI_CHECKPOINT_EVERY", "122880")', self.source)

    def test_periodic_checkpoints_align_to_full_vector_rollouts(self):
        self.assertIn("VECTOR_ROLLOUT_TRANSITIONS = NUM_CPU * PPO_N_STEPS", self.source)
        self.assertIn("CHECKPOINT_SAVE_FREQUENCY = (", self.source)
        self.assertIn("_save_after_optimizer_update", self.source)

    def test_final_save_uses_only_a_completed_optimizer_update(self):
        safe_train = _method_source(self.tree, "SafeRecurrentPPO", "train")
        self.assertIn("LAST_COMPLETED_UPDATE_TIMESTEP = int(self.num_timesteps)", safe_train)
        self.assertIn("PPO_UPDATE_FAILED = True", safe_train)
        self.assertIn("except BaseException", safe_train)
        self.assertIn("if PPO_UPDATE_FAILED:", self.source)
        self.assertIn("model.num_timesteps = final_timestep", self.source)

    def test_atomic_model_save_cleans_temporary_archive_on_base_exception(self):
        save_source = _method_source(self.tree, "StopAndCheckpointCallback", "_save_after_optimizer_update")
        self.assertIn("atomic_save_model", save_source)
        self.assertIn("_save_vector_env_state", save_source)
        atomic_start = self.source.index("def atomic_save_model(")
        atomic_end = self.source.index("def archive_model_checkpoint(", atomic_start)
        self.assertIn("except BaseException", self.source[atomic_start:atomic_end])

    def test_badge_flag_ids_are_defined_before_flag_snapshot_offsets(self):
        badge_definition = self.source.index("BADGE_FLAG_IDS = tuple(range(0x867, 0x86F))")
        byte_offsets = self.source.index("FLAG_BYTE_OFFSETS = tuple(")
        self.assertLess(badge_definition, byte_offsets)

    def test_flag_rewards_require_a_verified_episode_baseline(self):
        step = _method_source(self.tree, "EmeraldEnv", "step")
        self.assertIn("if state.get(\"flag_telemetry_valid\", False):", step)
        self.assertIn("if not self._badge_baseline_verified:", step)
        self.assertIn("if not self._story_flag_baseline_verified:", step)
        self.assertIn("if not self._trainer_flag_baseline_verified:", step)
        self.assertIn("self.seen_badge_flags |= observed_badge_flags", step)

    def test_unstable_saveblock_pointer_cannot_award_map_or_tile_novelty(self):
        step = _method_source(self.tree, "EmeraldEnv", "step")
        self.assertIn('map_telemetry_valid = bool(state.get("saveblock_pointer_stable", False))', step)
        reader = _method_source(self.tree, "EmeraldEnv", "_read_game_state")
        self.assertIn('if not saveblock_pointer_stable:', reader)
        self.assertIn('map_id = -1', reader)
        self.assertIn('player_x = None', reader)
        self.assertIn('player_y = None', reader)
        self.assertIn('state["map_id"] = -1', step)

    def test_game_state_uses_double_read_and_preserves_persistent_flag_history(self):
        reader = _method_source(self.tree, "EmeraldEnv", "_read_game_state")
        self.assertIn("flag_bytes_first", reader)
        self.assertIn("flag_bytes_second", reader)
        self.assertIn("pointer_after", reader)
        self.assertIn("previous_story_flags | set(story_flags_raw)", reader)
        self.assertIn("previous_trainer_flags | set(trainer_flags_first)", reader)

    def test_archive_comparison_requires_objective_success_improvement(self):
        curriculum_source = CURRICULUM_SOURCE.read_text(encoding="utf-8")
        self.assertIn("if score[0] <= baseline_success_rate or score[0] <= 0.0:", curriculum_source)
        self.assertIn("if full_score[0] <= baseline_success_rate:", curriculum_source)

    def test_worker_reset_timing_is_reported_separately(self):
        reset = _method_source(self.tree, "EmeraldEnv", "reset")
        step = _method_source(self.tree, "EmeraldEnv", "step")
        self.assertIn("_pending_reset_elapsed_ms", reset)
        self.assertIn("timing_reset_ms", step)


if __name__ == "__main__":
    unittest.main()
