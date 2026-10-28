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
import zipfile
from pathlib import Path

import run_curriculum


class CurriculumManagerTests(unittest.TestCase):
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
