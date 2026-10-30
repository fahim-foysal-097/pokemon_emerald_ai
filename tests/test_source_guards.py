"""Standard-library source guards for regressions that do not need mGBA/SB3."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = PROJECT_ROOT / "main.py"


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

    def test_worker_reset_timing_is_reported_separately(self):
        reset = _method_source(self.tree, "EmeraldEnv", "reset")
        step = _method_source(self.tree, "EmeraldEnv", "step")
        self.assertIn("_pending_reset_elapsed_ms", reset)
        self.assertIn("timing_reset_ms", step)


if __name__ == "__main__":
    unittest.main()
