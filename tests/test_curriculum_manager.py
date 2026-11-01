"""Tests for the stdlib-only curriculum runner configuration/atomic metadata helpers."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import zipfile
from pathlib import Path

import run_curriculum


def isolated_project_env(**overrides: str) -> dict[str, str]:
    """Keep subprocess config tests independent of the developer's exported POKEMON_AI_* vars.

    Each fixture supplies its own .env. Inheriting the real project's POKEMON_AI_*
    exports can silently replace fixture values because shell variables intentionally
    take precedence over dotenv settings.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("POKEMON_AI_")}
    env.update(overrides)
    return env


class CurriculumManagerTests(unittest.TestCase):

    def test_curriculum_uses_shared_incremental_stage_plan(self):
        from curriculum_objectives import CURRICULUM_PLAN_ID, OBJECTIVE_LABELS, STAGES
        self.assertEqual(CURRICULUM_PLAN_ID, "emerald_intro_v5")
        self.assertEqual(run_curriculum.STAGES, STAGES)
        self.assertEqual(run_curriculum.OBJECTIVE_LABELS, OBJECTIVE_LABELS)
        self.assertLess(STAGES.index("starter"), STAGES.index("leave_birch_lab"))
        self.assertLess(STAGES.index("leave_birch_lab"), STAGES.index("reach_route101"))
        self.assertLess(STAGES.index("reach_route101"), STAGES.index("reach_oldale"))
        self.assertLess(STAGES.index("reach_route103"), STAGES.index("defeat_rival"))
        self.assertLess(STAGES.index("return_birch_lab"), STAGES.index("pokedex"))

    def test_critical_shell_overrides_of_dotenv_are_reported(self):
        from contextlib import redirect_stdout
        from io import StringIO

        values = {
            "POKEMON_AI_OBJECTIVE": "starter",
            "POKEMON_AI_CURRICULUM_STAGE_ONLY": "1",
            "POKEMON_AI_CANONICAL_START_STATE": "runtime_data/states/training_start.state",
        }
        env = {
            "POKEMON_AI_OBJECTIVE": "leave_birch_lab",
            "POKEMON_AI_CURRICULUM_STAGE_ONLY": "0",
            "POKEMON_AI_CANONICAL_START_STATE": "/tmp/alternate-start.state",
        }
        output = StringIO()
        with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(output):
            run_curriculum.report_shell_config_overrides(values)
            self.assertFalse(run_curriculum.stage_only_requested(False, values))
        rendered = output.getvalue()
        self.assertIn("POKEMON_AI_OBJECTIVE='leave_birch_lab'", rendered)
        self.assertIn("POKEMON_AI_CURRICULUM_STAGE_ONLY='0'", rendered)
        self.assertIn("POKEMON_AI_CANONICAL_START_STATE", rendered)

    def test_stage_only_flag_uses_cli_or_env_and_defaults_off(self):
        # main.py may load this project's .env into os.environ during unittest
        # discovery. Isolate this test so the project's intentional
        # POKEMON_AI_CURRICULUM_STAGE_ONLY=1 setting cannot change the default-off
        # assertion. Runtime precedence remains shell environment > dotenv values.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POKEMON_AI_CURRICULUM_STAGE_ONLY", None)
            self.assertFalse(run_curriculum.stage_only_requested(False, {}))
            self.assertTrue(run_curriculum.stage_only_requested(True, {}))
            self.assertTrue(run_curriculum.stage_only_requested(
                False, {"POKEMON_AI_CURRICULUM_STAGE_ONLY": "1"}
            ))
            self.assertFalse(run_curriculum.stage_only_requested(
                False, {"POKEMON_AI_CURRICULUM_STAGE_ONLY": "0"}
            ))

    def test_stage_only_dry_run_skips_evaluation_and_promotion_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "main.py").write_text("# fixture\n", encoding="utf-8")
            (root / "PokemonEmerald.gba").write_bytes(b"rom")
            (root / "emerald_ppo_hybrid_v1.zip").write_bytes(b"checkpoint-placeholder")
            (root / "runtime_data/states/milestones").mkdir(parents=True)
            prior_state = root / "runtime_data/states/milestones/starter.state"
            prior_state.write_bytes(b"savestate-placeholder")
            rom_hash = "f3ae088181bf583e55daf962a92bb46f4f1d07b7"
            Path(str(prior_state) + ".json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "objective": "starter",
                    "rom_sha1_expected": rom_hash,
                    "savestate_path": str(prior_state.resolve()),
                    "state": {
                        "party_count": 1,
                        "party_levels": 5,
                        "map_id": 260,
                        "story_flags": [0x51, 0x52],
                        "badge_count": 0,
                    },
                }), encoding="utf-8"
            )
            (root / ".env").write_text(
                "POKEMON_AI_OBJECTIVE=leave_birch_lab\n"
                "POKEMON_AI_CURRICULUM_STAGE_ONLY=1\n"
                "POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS=122880\n"
                "POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE=7\n"
                f"POKEMON_AI_EXPECTED_ROM_SHA1={rom_hash}\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(run_curriculum.__file__).resolve()), "--project-dir", str(root), "--dry-run"],
                text=True, capture_output=True, timeout=10, env=isolated_project_env(),
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr + "\n" + result.stdout)
        self.assertIn("STAGE-ONLY", result.stdout)
        self.assertIn("Evaluation        : DISABLED", result.stdout)
        self.assertIn("Promotion         : DISABLED", result.stdout)
        self.assertNotIn("Evaluate current checkpoint", result.stdout)
        self.assertNotIn("[Curriculum] $", result.stdout)

    def test_controller_ignores_inherited_stage_state_when_resolving_starter_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "main.py").write_text("# fixture\n", encoding="utf-8")
            (root / "PokemonEmerald.gba").write_bytes(b"rom")
            (root / "runtime_data/states/milestones").mkdir(parents=True)
            (root / "runtime_data/states/milestones/starter.state").write_bytes(b"completed milestone")
            (root / "runtime_data/states/training_start.state").write_bytes(b"pre-starter")
            (root / ".env").write_text(
                "POKEMON_AI_OBJECTIVE=starter\n"
                "POKEMON_AI_CURRICULUM_STAGE_ONLY=1\n"
                "POKEMON_AI_TRAINING_START_STATE=runtime_data/states/milestones/starter.state\n"
                "POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS=16384\n"
                "POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE=1\n"
                "POKEMON_AI_EXPECTED_ROM_SHA1=f3ae088181bf583e55daf962a92bb46f4f1d07b7\n",
                encoding="utf-8",
            )
            env = isolated_project_env(
                POKEMON_AI_TRAINING_START_STATE=str(
                    root / "runtime_data/states/milestones/starter.state"
                )
            )
            result = subprocess.run(
                [sys.executable, str(Path(run_curriculum.__file__).resolve()), "--project-dir", str(root), "--dry-run"],
                text=True, capture_output=True, timeout=10, env=env,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr + "\n" + result.stdout)
        self.assertIn("Ignoring inherited POKEMON_AI_TRAINING_START_STATE", result.stdout)
        self.assertIn(str(root / "runtime_data/states/training_start.state"), result.stdout)

    def test_stage_start_validator_rejects_missing_or_detached_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            milestone_dir = root / "runtime_data/states/milestones"
            milestone_dir.mkdir(parents=True)
            prior_state = milestone_dir / "starter.state"
            prior_state.write_bytes(b"state")
            manifest_path = Path(str(prior_state) + ".json")
            valid = {
                "objective": "starter",
                "rom_sha1_expected": "f3ae088181bf583e55daf962a92bb46f4f1d07b7",
                "savestate_path": str(prior_state.resolve()),
                "state": {
                    "party_count": 1, "party_levels": 5, "map_id": 260,
                    "story_flags": [0x51, 0x52], "badge_count": 0,
                },
            }
            manifest_path.write_text(json.dumps(valid), encoding="utf-8")
            run_curriculum.validate_stage_start_state(
                root, "leave_birch_lab", prior_state,
                "f3ae088181bf583e55daf962a92bb46f4f1d07b7",
            )
            manifest_path.write_text(json.dumps({k: v for k, v in valid.items() if k != "savestate_path"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing its savestate_path"):
                run_curriculum.validate_stage_start_state(
                    root, "leave_birch_lab", prior_state,
                    "f3ae088181bf583e55daf962a92bb46f4f1d07b7",
                )

    def test_dotenv_parser_uses_first_occurrence_and_strips_quotes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text(
                "# comment\nPOKEMON_AI_MODEL_NAME='model_a'\nPOKEMON_AI_MODEL_NAME=model_b\nexport POKEMON_AI_EVAL_EPISODES=5\n",
                encoding="utf-8",
            )
            values = run_curriculum.parse_dotenv(path)
        self.assertEqual(values["POKEMON_AI_MODEL_NAME"], "model_a")
        self.assertEqual(values["POKEMON_AI_EVAL_EPISODES"], "5")

    def test_shutdown_timeout_parser_rejects_invalid_values(self):
        self.assertEqual(run_curriculum._shutdown_timeout({}, "POKEMON_AI_SHUTDOWN_TIMEOUT", 900.0), 900.0)
        self.assertEqual(run_curriculum._shutdown_timeout({"POKEMON_AI_SHUTDOWN_TIMEOUT": "120"}, "POKEMON_AI_SHUTDOWN_TIMEOUT", 900.0), 120.0)
        for bad in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                run_curriculum._shutdown_timeout({"POKEMON_AI_SHUTDOWN_TIMEOUT": bad}, "POKEMON_AI_SHUTDOWN_TIMEOUT", 900.0)

    def test_settings_validate_bounds(self):
        self.assertEqual(run_curriculum.int_setting({"N": "3"}, "N", 4), 3)
        self.assertEqual(run_curriculum.float_setting({"RATE": "0.8"}, "RATE", 0.5), 0.8)
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            run_curriculum.int_setting({"N": "0"}, "N", 4)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            run_curriculum.float_setting({"RATE": "1.5"}, "RATE", 0.5)

    def test_atomic_json_replaces_complete_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime" / "state.json"
            run_curriculum.atomic_json(path, {"status": "running", "n": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "running", "n": 2})
            self.assertEqual(list(path.parent.glob("*.tmp.*")), [])

    def test_milestone_requires_matching_checkpoint_identity_and_freshness(self):
        import time
        checkpoint = Path("/tmp/model.zip").resolve()
        started = time.time()
        report = {
            "objective": "starter",
            "checkpoint": str(checkpoint),
            "checkpoint_sha1": "a" * 40,
        }
        manifest = {
            "objective": "starter",
            "rom_sha1_expected": "b" * 40,
            "created_unix": started + 1,
            "evaluation_checkpoint_path": str(checkpoint),
            "evaluation_checkpoint_sha1": "a" * 40,
        }
        self.assertTrue(run_curriculum.milestone_matches_evaluation(manifest, report, "b" * 40, started))
        wrong_hash = dict(manifest, evaluation_checkpoint_sha1="c" * 40)
        self.assertFalse(run_curriculum.milestone_matches_evaluation(wrong_hash, report, "b" * 40, started))
        wrong_path = dict(manifest, evaluation_checkpoint_path="/tmp/other.zip")
        self.assertFalse(run_curriculum.milestone_matches_evaluation(wrong_path, report, "b" * 40, started))
        stale = dict(manifest, created_unix=started - 1)
        self.assertFalse(run_curriculum.milestone_matches_evaluation(stale, report, "b" * 40, started))
        wrong_rom = dict(manifest, rom_sha1_expected="d" * 40)
        self.assertFalse(run_curriculum.milestone_matches_evaluation(wrong_rom, report, "b" * 40, started))

    def test_archive_screening_environment_propagates_exact_single_episode_request(self):
        env = run_curriculum.build_evaluation_environment(
            {"POKEMON_AI_EVAL_EPISODES": "5", "POKEMON_AI_MAX_EPISODE_STEPS": "131072"},
            stage="leave_birch_lab",
            start_state=Path("/tmp/starter.state"),
            requested_episodes=1,
            deterministic=True,
            eval_seed="1234",
            max_episode_steps=30_720,
            no_progress_steps=4096,
            checkpoint_path=Path("/tmp/candidate.zip"),
        )
        self.assertEqual(env["POKEMON_AI_EVAL_EPISODES"], "1")
        self.assertEqual(env["POKEMON_AI_EVAL_MAX_EPISODE_STEPS"], "30720")
        self.assertEqual(env["POKEMON_AI_EVAL_NO_PROGRESS_STEPS"], "4096")
        self.assertEqual(env["POKEMON_AI_EVAL_CHECKPOINT"], "/tmp/candidate.zip")
        self.assertEqual(env["POKEMON_AI_MODE"], "evaluate")

    def test_stage_evaluation_step_limit_uses_selected_objective_not_later_stage(self):
        self.assertEqual(run_curriculum.stage_evaluation_step_limit("leave_birch_lab", 2048), 30_720)
        self.assertEqual(run_curriculum.stage_evaluation_step_limit("reach_route101", 2048), 40_960)
        self.assertEqual(run_curriculum.stage_evaluation_step_limit("first_badge", 2048), 131_072)
        self.assertEqual(run_curriculum.stage_evaluation_step_limit("leave_birch_lab", 2048, 8192), 8192)
        with self.assertRaises(ValueError):
            run_curriculum.stage_evaluation_step_limit("leave_birch_lab", 0)

    def test_checkpoint_score_uses_only_non_flag_progress_as_a_secondary_tiebreaker(self):
        stagnant = {
            "success_rate": 0.0, "mean_return": 100.0, "mean_steps": 1000,
            "episodes": [{"unique_tiles": 8, "unique_maps": 1, "story_flags_episode": 400,
                          "trainer_flags_episode": 453, "party_count": 6, "party_levels": 600,
                          "badge_count": 8, "party_experience_gained": 0}],
        }
        inflated_flags_only = {
            "success_rate": 0.0, "mean_return": 100.0, "mean_steps": 1000,
            "episodes": [{"unique_tiles": 8, "unique_maps": 1, "story_flags_episode": 600,
                          "trainer_flags_episode": 800, "party_count": 6, "party_levels": 600,
                          "badge_count": 8, "party_experience_gained": 0}],
        }
        progressed = {
            "success_rate": 0.0, "mean_return": -10.0, "mean_steps": 5000,
            "episodes": [{"unique_tiles": 20, "unique_maps": 2, "story_flags_episode": 0,
                          "trainer_flags_episode": 0, "party_count": 1, "party_levels": 5,
                          "badge_count": 0, "party_experience_gained": 1000}],
        }
        self.assertEqual(run_curriculum.report_score(inflated_flags_only), run_curriculum.report_score(stagnant))
        self.assertGreater(run_curriculum.report_score(progressed), run_curriculum.report_score(stagnant))

    def test_v4_early_stage_migrates_without_restarting_completed_starter(self):
        migrated = run_curriculum.resolve_saved_curriculum_stage({
            "curriculum_plan": "emerald_intro_v4",
            "status": "stage_not_mastered",
            "stage": "leave_birch_lab",
        })
        self.assertEqual(migrated, ("leave_birch_lab", "migrated_v4_early_stage"))
        self.assertEqual(
            run_curriculum.resolve_saved_curriculum_stage({
                "curriculum_plan": "emerald_intro_v4",
                "status": "stage_not_mastered",
                "stage": "reach_oldale",
            }),
            (None, "incompatible_plan"),
        )

    def test_checkpoint_score_prioritizes_success_then_return_then_steps(self):
        base = {"success_rate": 0.0, "mean_return": 20.0, "mean_steps": 1000}
        better_success = {"success_rate": 0.2, "mean_return": -100.0, "mean_steps": 5000}
        better_return = {"success_rate": 0.0, "mean_return": 21.0, "mean_steps": 900}
        faster = {"success_rate": 0.0, "mean_return": 20.0, "mean_steps": 900}
        self.assertGreater(run_curriculum.report_score(better_success), run_curriculum.report_score(base))
        self.assertGreater(run_curriculum.report_score(better_return), run_curriculum.report_score(base))
        self.assertGreater(run_curriculum.report_score(faster), run_curriculum.report_score(base))

    @staticmethod
    def write_fake_checkpoint(path: Path, contents: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data", contents)

    def test_archive_scan_ignores_current_duplicate_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "emerald.zip"
            archive_dir = root / "runtime_data" / "checkpoints"
            self.write_fake_checkpoint(current, "current")
            self.write_fake_checkpoint(archive_dir / "emerald_step0001.zip", "current")
            self.write_fake_checkpoint(archive_dir / "emerald_step0002.zip", "older-better-candidate")
            (archive_dir / "emerald_step0003.zip").write_bytes(b"not a zip")
            candidates = run_curriculum.archived_checkpoint_candidates(root, "emerald", current, 5)
            self.assertEqual([path.name for path in candidates], ["emerald_step0002.zip"])

    def test_checkpoint_activation_preserves_current_model_and_validates_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "emerald.zip"
            candidate = root / "runtime_data" / "checkpoints" / "emerald_step0002.zip"
            backup_dir = root / "runtime_data" / "backups"
            self.write_fake_checkpoint(current, "current-model")
            self.write_fake_checkpoint(candidate, "candidate-model")
            backup = run_curriculum.activate_checkpoint_candidate(candidate, current, backup_dir)
            self.assertIsNotNone(backup)
            self.assertTrue(backup.is_file())
            with zipfile.ZipFile(backup) as archive:
                self.assertEqual(archive.read("data"), b"current-model")
            with zipfile.ZipFile(current) as archive:
                self.assertEqual(archive.read("data"), b"candidate-model")
            self.assertIsNone(run_curriculum.activate_checkpoint_candidate(candidate, current, backup_dir))

    @unittest.skipUnless(os.name == "posix", "signal-forwarding test requires POSIX")
    def test_stream_command_forwards_interrupt_and_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child_pid_file = root / "child.pid"
            child_code = (
                "import os,signal,sys,time; "
                "signal.signal(signal.SIGINT, lambda _sig,_frame: sys.exit(0)); "
                f"open({str(child_pid_file)!r},'w').write(str(os.getpid())); "
                "time.sleep(60)"
            )
            module_dir = Path(run_curriculum.__file__).resolve().parent
            driver_code = (
                "import os,signal,sys\n"
                "from pathlib import Path\n"
                "import run_curriculum\n"
                "signal.signal(signal.SIGINT, run_curriculum._handle_parent_stop_signal)\n"
                "try:\n"
                f"    run_curriculum.stream_command([sys.executable, '-c', {child_code!r}], os.environ.copy(), "
                f"Path({str(root)!r}), Path({str(root / 'child.log')!r}), False)\n"
                "except KeyboardInterrupt:\n"
                "    sys.exit(130)\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(module_dir) + os.pathsep + env.get("PYTHONPATH", "")
            with (root / "driver.out").open("w") as output:
                driver = subprocess.Popen(
                    [sys.executable, "-c", driver_code], cwd=root, env=env,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                )
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid_file.exists() and driver.poll() is None:
                    time.sleep(0.02)
                self.assertTrue(child_pid_file.exists(), "stream_command child should start")
                child_pid = int(child_pid_file.read_text())
                os.kill(driver.pid, signal.SIGINT)
                driver.wait(timeout=8)
                self.assertEqual(driver.returncode, 130)
                deadline = time.monotonic() + 2
                child_alive = True
                while time.monotonic() < deadline:
                    try:
                        stat = Path(f"/proc/{child_pid}/stat").read_text().split()
                        child_alive = len(stat) > 2 and stat[2] != "Z"
                    except (FileNotFoundError, ProcessLookupError):
                        child_alive = False
                    if not child_alive:
                        break
                    time.sleep(0.05)
                self.assertFalse(child_alive, "Ctrl+C should stop the managed child process")
            finally:
                if driver.poll() is None:
                    try:
                        os.killpg(driver.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    driver.wait(timeout=3)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_stream_command_persists_only_important_child_events(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log_path = root / "concise.log"
            child_code = (
                "print('[Training] run=10,000/250,000 | speed=100/s', flush=True); "
                "print('[Checkpoint] PPO model atomically saved at timestep 25,000', flush=True); "
                "print('[Evaluation Diagnostic] step=10,000 action=B', flush=True); "
                "print('[Curriculum Evaluation] stage=starter successes=5/5 rate=100.0%', flush=True)"
            )
            run_curriculum.stream_command(
                [sys.executable, "-c", child_code], os.environ.copy(), root, log_path, False
            )
            stored = log_path.read_text(encoding="utf-8")
            self.assertIn("[Checkpoint]", stored)
            self.assertIn("[Curriculum Evaluation]", stored)
            self.assertNotIn("[Training] run=", stored)
            self.assertNotIn("[Evaluation Diagnostic]", stored)

    @unittest.skipUnless(os.name == "posix", "process-group shutdown test requires POSIX")
    def test_curriculum_tensorboard_cleanup_stops_owned_server(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        run_curriculum._CURRICULUM_TENSORBOARD_PROCESS = process
        run_curriculum._CURRICULUM_TENSORBOARD_PORT = 61234
        try:
            run_curriculum.stop_curriculum_tensorboard()
            self.assertIsNotNone(process.poll(), "owned TensorBoard server should terminate")
            self.assertIsNone(run_curriculum._CURRICULUM_TENSORBOARD_PROCESS)
            self.assertIsNone(run_curriculum._CURRICULUM_TENSORBOARD_PORT)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)

    @unittest.skipUnless(os.name == "posix", "process-group shutdown test requires POSIX")
    def test_stop_managed_process_escalates_and_reaps_ignored_signals(self):
        with tempfile.TemporaryDirectory() as folder:
            ready_path = Path(folder) / "ready"
            code = (
                "import signal,time; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({str(ready_path)!r},'w').write('ready'); "
                "time.sleep(60)"
            )
            process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not ready_path.exists() and process.poll() is None:
                    time.sleep(0.02)
                self.assertTrue(ready_path.exists(), "child must install its signal handlers before the test signals it")
                run_curriculum.stop_managed_process(
                    process, "test child", graceful_signal=signal.SIGINT,
                    graceful_timeout=0.1, terminate_timeout=0.1,
                )
                self.assertIsNotNone(process.poll(), "managed process should no longer be running")
                with self.assertRaises(ProcessLookupError):
                    os.killpg(process.pid, 0)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)

    @unittest.skipUnless(os.name == "posix", "process-group shutdown test requires POSIX")
    def test_stop_managed_process_terminates_descendants_too(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_path = Path(folder) / "descendant.pid"
            ready_path = Path(folder) / "descendant.ready"
            child_code = (
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({str(ready_path)!r},'w').write('ready'); time.sleep(60)"
            )
            parent_code = (
                "import signal,subprocess,sys,time; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                f"open({str(pid_path)!r},'w').write(str(p.pid)); "
                "time.sleep(60)"
            )
            process = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
            descendant_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not ready_path.exists() and process.poll() is None:
                    time.sleep(0.02)
                self.assertTrue(ready_path.exists(), "descendant signal handler must be installed before cleanup starts")
                self.assertTrue(pid_path.exists(), "descendant pid should be written")
                descendant_pid = int(pid_path.read_text())
                run_curriculum.stop_managed_process(
                    process, "test process tree", graceful_signal=signal.SIGINT,
                    graceful_timeout=0.1, terminate_timeout=0.1,
                )
                self.assertIsNotNone(process.poll(), "process-group owner should be stopped")
                # A killed descendant can briefly remain as a zombie until init reaps it.
                deadline = time.monotonic() + 2
                alive = True
                while time.monotonic() < deadline:
                    try:
                        stat = Path(f"/proc/{descendant_pid}/stat").read_text().split()
                        alive = len(stat) > 2 and stat[2] != "Z"
                    except (FileNotFoundError, ProcessLookupError):
                        alive = False
                    if not alive:
                        break
                    time.sleep(0.05)
                self.assertFalse(alive, "descendant must be stopped, not left running")
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
                if descendant_pid is not None:
                    try:
                        os.kill(descendant_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
