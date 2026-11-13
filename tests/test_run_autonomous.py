"""CLI adapter and graceful-stop tests for the autonomous curriculum launcher."""
from __future__ import annotations

import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_autonomous


class AutonomousLauncherTests(unittest.TestCase):
    def _project(self, folder: str) -> Path:
        root = Path(folder)
        (root / "run_curriculum.py").write_text("# fixture\n", encoding="utf-8")
        return root

    def test_single_episode_is_allowed_for_a_90_percent_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self._project(folder)
            process = Mock()
            process.wait.return_value = 0
            with patch.object(run_autonomous.subprocess, "Popen", return_value=process) as popen:
                result = run_autonomous.main([
                    "--project-dir", str(root), "--eval-episodes", "1", "--gate", "0.9", "--dry-run"
                ])
            self.assertEqual(result, 0)
            command = popen.call_args.args[0]
            env = popen.call_args.kwargs["env"]
            self.assertIn("--dry-run", command)
            self.assertEqual(env["POKEMON_AI_CURRICULUM_EVAL_EPISODES"], "1")
            self.assertEqual(env["POKEMON_AI_CURRICULUM_SUCCESS_RATE"], "0.9")
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_default_does_not_override_dotenv_evaluation_or_chunk_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self._project(folder)
            process = Mock()
            process.wait.return_value = 0
            # main.py's module-level setup loads the project's .env into os.environ
            # when the full unittest suite imports it before this test module. The
            # autonomous launcher must be tested with a clean inherited environment
            # so this assertion checks that it does not inject its own defaults.
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(run_autonomous.subprocess, "Popen", return_value=process) as popen:
                    result = run_autonomous.main(["--project-dir", str(root), "--dry-run"])
                env = popen.call_args.kwargs["env"]
            self.assertEqual(result, 0)
            self.assertNotIn("POKEMON_AI_CURRICULUM_EVAL_EPISODES", env)
            self.assertNotIn("POKEMON_AI_CURRICULUM_SUCCESS_RATE", env)
            self.assertNotIn("POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE", env)

    def test_explicit_finite_chunk_limit_is_forwarded_to_curriculum(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self._project(folder)
            process = Mock()
            process.wait.return_value = 0
            with patch.object(run_autonomous.subprocess, "Popen", return_value=process) as popen:
                result = run_autonomous.main([
                    "--project-dir", str(root), "--max-chunks-per-stage", "1", "--dry-run"
                ])
            self.assertEqual(result, 0)
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE"], "1")

    def test_explicit_chunk_limit_is_forwarded_to_curriculum(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self._project(folder)
            process = Mock()
            process.wait.return_value = 0
            with patch.object(run_autonomous.subprocess, "Popen", return_value=process) as popen:
                result = run_autonomous.main([
                    "--project-dir", str(root), "--max-chunks-per-stage", "0", "--dry-run"
                ])
            self.assertEqual(result, 0)
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE"], "0")

    def test_ctrl_c_is_forwarded_and_controller_is_waited_for_without_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self._project(folder)
            process = Mock()
            process.pid = 43210
            process.wait.side_effect = [KeyboardInterrupt(), 130]
            with patch.object(run_autonomous.subprocess, "Popen", return_value=process), \
                 patch.object(run_autonomous.signal, "signal", wraps=signal.signal):
                result = run_autonomous.main(["--project-dir", str(root), "--dry-run"])
            self.assertEqual(result, 130)
            process.send_signal.assert_called_once_with(signal.SIGINT)
            self.assertEqual(process.wait.call_count, 2)
            self.assertGreater(process.wait.call_args_list[1].kwargs["timeout"], 900)

    def test_shutdown_timeout_parser_uses_safe_fallback_for_invalid_values(self):
        self.assertEqual(run_autonomous._positive_timeout({}, "X", 10.0), 10.0)
        self.assertEqual(run_autonomous._positive_timeout({"X": "0"}, "X", 10.0), 10.0)
        self.assertEqual(run_autonomous._positive_timeout({"X": "bad"}, "X", 10.0), 10.0)
        self.assertEqual(run_autonomous._positive_timeout({"X": "12.5"}, "X", 10.0), 12.5)


if __name__ == "__main__":
    unittest.main()
