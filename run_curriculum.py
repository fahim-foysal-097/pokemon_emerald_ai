#!/usr/bin/env python3
"""Safely train/evaluate/promote Pokémon Emerald curriculum stages.

Uses only the Python standard library. Run from the project's virtualenv:
    ./.venv/bin/python run_curriculum.py --dry-run
    ./.venv/bin/python run_curriculum.py

The manager starts by evaluating an existing checkpoint before spending training time.
It promotes only after the configured fixed-seed evaluation threshold is reached and
both the objective savestate and its metadata manifest exist.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import threading
import zipfile
from pathlib import Path
from typing import Any

STAGES = ("starter", "pokedex", "first_badge")
OBJECTIVE_LABELS = {
    "starter": "first Pokémon",
    "pokedex": "Pokédex",
    "first_badge": "first Gym Badge",
}

# Only terminate processes created by this curriculum invocation. Existing servers
# owned by the user are never claimed or killed.
_CURRICULUM_TENSORBOARD_PROCESS: subprocess.Popen | None = None
_CURRICULUM_TENSORBOARD_PORT: int | None = None


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse the small KEY=VALUE dotenv format used by main.py (first key wins)."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    key_re = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = key_re.match(line)
        if not match:
            continue
        key = match.group(1)
        if key in result:
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key] = value
    return result


def bool_setting(values: dict[str, str], key: str, default: bool) -> bool:
    value = os.environ.get(key, values.get(key, "1" if default else "0"))
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def int_setting(values: dict[str, str], key: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(key, values.get(key, str(default)))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer; got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}; got {value}")
    return value


def float_setting(values: dict[str, str], key: str, default: float) -> float:
    raw = os.environ.get(key, values.get(key, str(default)))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric; got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1; got {value}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def milestone_matches_evaluation(
    manifest: dict[str, Any],
    report: dict[str, Any],
    expected_rom_hash: str,
    evaluation_started_unix: float,
) -> bool:
    """Reject milestones produced by another checkpoint or by an earlier evaluation."""
    created = manifest.get("created_unix")
    report_checkpoint = report.get("checkpoint")
    report_checkpoint_sha1 = report.get("checkpoint_sha1")
    if not isinstance(report_checkpoint, str) or not report_checkpoint:
        return False
    if not isinstance(report_checkpoint_sha1, str) or not re.fullmatch(r"[0-9a-f]{40}", report_checkpoint_sha1):
        return False
    try:
        same_path = Path(str(manifest.get("evaluation_checkpoint_path", ""))).resolve() == Path(report_checkpoint).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(
        manifest.get("objective") == report.get("objective")
        and manifest.get("rom_sha1_expected") == expected_rom_hash
        and manifest.get("evaluation_checkpoint_sha1") == report_checkpoint_sha1
        and same_path
        and isinstance(created, (int, float))
        and created >= evaluation_started_unix
    )


def report_score(report: dict[str, Any]) -> tuple[float, float, float]:
    """Rank checkpoints by objective success, then mean return, then fewer steps."""
    def finite(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if number != number or number in (float("inf"), float("-inf")):
            return fallback
        return number

    success_rate = max(0.0, min(1.0, finite(report.get("success_rate"), 0.0)))
    mean_return = finite(report.get("mean_return"), float("-inf"))
    mean_steps = finite(report.get("mean_steps"), float("inf"))
    return success_rate, mean_return, -mean_steps


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archived_checkpoint_candidates(root: Path, model_name: str, current: Path, limit: int) -> list[Path]:
    """Return recent, valid, content-distinct archive checkpoints newest first."""
    if limit <= 0:
        return []
    archive_dir = root / "runtime_data" / "checkpoints"
    current_hash = file_sha1(current) if current.is_file() else None
    found: list[tuple[int, Path, str]] = []
    seen_hashes: set[str] = set()
    for path in archive_dir.glob(f"{model_name}_step*.zip"):
        try:
            if not path.is_file() or not zipfile.is_zipfile(path):
                continue
            digest = file_sha1(path)
            if digest == current_hash or digest in seen_hashes:
                continue
            with zipfile.ZipFile(path, "r") as archive:
                if archive.testzip() is not None:
                    continue
            seen_hashes.add(digest)
            found.append((path.stat().st_mtime_ns, path, digest))
        except (OSError, zipfile.BadZipFile, RuntimeError):
            continue
    found.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return [path for _, path, _ in found[:limit]]


def _fsync_dir(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def activate_checkpoint_candidate(source: Path, destination: Path, backup_dir: Path) -> Path | None:
    """Safely promote a validated archive to the root model path and retain the prior model."""
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise RuntimeError(f"Refusing to activate missing or invalid checkpoint: {source}")
    with zipfile.ZipFile(source, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Refusing to activate corrupted checkpoint: {source}")
    source_hash = file_sha1(source)
    if destination.is_file() and file_sha1(destination) == source_hash:
        return None

    backup_path: Path | None = None
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if destination.is_file():
        backup_path = backup_dir / f"{destination.stem}.before_candidate_{stamp}_{time.time_ns()}_{os.getpid()}.zip"
        backup_tmp = backup_path.with_suffix(backup_path.suffix + ".tmp")
        shutil.copy2(destination, backup_tmp)
        if not zipfile.is_zipfile(backup_tmp):
            backup_tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not validate checkpoint backup: {backup_tmp}")
        os.replace(backup_tmp, backup_path)

    temporary = destination.with_suffix(destination.suffix + f".candidate.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        if not zipfile.is_zipfile(temporary) or file_sha1(temporary) != source_hash:
            raise RuntimeError("Candidate checkpoint copy failed ZIP/hash validation")
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Candidate checkpoint copy failed ZIP integrity validation")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_dir(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return backup_path


def _signal_process_group(pgid: int, sig: int) -> bool:
    """Signal a dedicated child process group; returns False if it no longer exists."""
    if os.name == "posix":
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            print(f"[Shutdown Warning] Cannot signal process group {pgid}: {exc}", flush=True)
            return False
    return False


def _process_group_exists(pgid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_managed_process(
    process: subprocess.Popen | None,
    label: str,
    *,
    graceful_signal: int = signal.SIGINT,
    graceful_timeout: float = 20.0,
    terminate_timeout: float = 5.0,
) -> None:
    """Stop an isolated process and descendants without leaving worker processes behind.

    Managed commands are launched with start_new_session=True, making their PID the
    process-group ID. First signal the Python owner alone so main.py can save and close
    its VecEnv. Escalate against the isolated process group only if needed, then clean
    up any descendants that outlive the owner.
    """
    if process is None:
        return
    pgid = process.pid
    if process.poll() is None:
        try:
            process.send_signal(graceful_signal)
        except ProcessLookupError:
            pass
        except OSError as exc:
            print(f"[Shutdown Warning] Could not signal {label} PID {process.pid}: {exc}", flush=True)
        try:
            process.wait(timeout=max(0.0, graceful_timeout))
        except subprocess.TimeoutExpired:
            print(f"[Shutdown] {label} did not exit gracefully; sending SIGTERM to its process group.", flush=True)
            _signal_process_group(pgid, signal.SIGTERM)
            try:
                process.wait(timeout=max(0.0, terminate_timeout))
            except subprocess.TimeoutExpired:
                print(f"[Shutdown] {label} is still alive; sending SIGKILL to its process group.", flush=True)
                _signal_process_group(pgid, signal.SIGKILL)
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    print(f"[Shutdown Error] Could not reap {label} PID {process.pid} after SIGKILL.", flush=True)

    # The owner may have exited while workers remain. This group is isolated to the job.
    if _process_group_exists(pgid):
        _signal_process_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + min(max(terminate_timeout, 0.25), 2.0)
        while time.monotonic() < deadline and _process_group_exists(pgid):
            time.sleep(0.05)
        if _process_group_exists(pgid):
            _signal_process_group(pgid, signal.SIGKILL)

    if process.poll() is not None:
        try:
            process.wait(timeout=0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass


def stream_command(command: list[str], env: dict[str, str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[Curriculum] $ {printable}", flush=True)
    print(f"[Curriculum] log: {log_path}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Isolate the child tree from the terminal process group. Output is drained on a
    # dedicated thread so cleanup can wait for main.py while its shutdown logs continue
    # to drain (otherwise a full stdout pipe could deadlock graceful shutdown).
    child = subprocess.Popen(
        command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
    )
    assert child.stdout is not None
    with log_path.open("a", encoding="utf-8", buffering=1) as logfile:
        output_errors: list[BaseException] = []

        def forward_output() -> None:
            terminal_available = True
            try:
                assert child.stdout is not None
                for line in child.stdout:
                    # Always drain stdout and persist logs even if the terminal is closed.
                    logfile.write(line)
                    if terminal_available:
                        try:
                            print(line, end="", flush=True)
                        except BrokenPipeError:
                            terminal_available = False
            except BaseException as exc:
                output_errors.append(exc)

        output_thread = threading.Thread(
            target=forward_output, name=f"curriculum-output-{child.pid}", daemon=True
        )
        output_thread.start()
        try:
            code = child.wait()
        except KeyboardInterrupt:
            print("\n[Curriculum] Stop requested. Asking main.py to save, then stopping any remaining workers...", flush=True)
            stop_managed_process(child, "active main.py", graceful_signal=signal.SIGINT)
            raise KeyboardInterrupt
        except BaseException:
            # Reap the process tree even on unexpected parent-side failures (e.g. disk/log errors).
            stop_managed_process(child, "active main.py", graceful_signal=signal.SIGINT)
            raise
        finally:
            # If main.py exited but a worker still holds stdout open, terminate that worker
            # before joining the output pump. The group belongs exclusively to this run.
            if _process_group_exists(child.pid):
                stop_managed_process(
                    child, "remaining main.py workers", graceful_signal=signal.SIGTERM,
                    graceful_timeout=0.0, terminate_timeout=0.5,
                )
            output_thread.join(timeout=5.0)
            if output_thread.is_alive():
                print(f"[Shutdown Warning] Output reader for child {child.pid} did not finish promptly.", flush=True)
                try:
                    child.stdout.close()
                except OSError:
                    pass
                output_thread.join(timeout=1.0)
        if output_errors:
            print(f"[Curriculum Warning] Child output forwarding raised {type(output_errors[0]).__name__}: {output_errors[0]}", flush=True)

    if code != 0:
        raise RuntimeError(f"Child process exited with code {code}: {printable}")


def resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def port_is_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def start_curriculum_tensorboard(root: Path, python: Path, cfg: dict[str, str], log_dir: Path) -> None:
    """Start one TensorBoard server owned by this curriculum invocation."""
    global _CURRICULUM_TENSORBOARD_PROCESS, _CURRICULUM_TENSORBOARD_PORT
    enabled = bool_setting(cfg, "POKEMON_AI_AUTO_START_TENSORBOARD", True)
    if not enabled:
        print("[TensorBoard] Auto-start disabled by POKEMON_AI_AUTO_START_TENSORBOARD=0.", flush=True)
        return

    configured_logdir = os.environ.get(
        "POKEMON_AI_TENSORBOARD_LOG_DIR", cfg.get("POKEMON_AI_TENSORBOARD_LOG_DIR", "tensorboard_logs_v2")
    )
    tensorboard_dir = resolve_path(root, configured_logdir)
    requested_port = int_setting(cfg, "POKEMON_AI_TENSORBOARD_PORT", 6006)
    if requested_port > 65535:
        raise ValueError("POKEMON_AI_TENSORBOARD_PORT must be <= 65535")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    actual_port = requested_port
    if port_is_open("127.0.0.1", actual_port):
        for candidate in range(requested_port + 1, min(requested_port + 20, 65536)):
            if not port_is_open("127.0.0.1", candidate):
                actual_port = candidate
                print(
                    f"[TensorBoard] Port {requested_port} is occupied; using port {actual_port} "
                    "for this run's log directory.", flush=True,
                )
                break
        else:
            print(
                f"[TensorBoard Warning] Ports {requested_port}-{min(requested_port + 19, 65535)} "
                "are occupied; no new server started. Metrics will still be written.", flush=True,
            )
            return

    server_log = log_dir / "tensorboard_server.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with server_log.open("a", encoding="utf-8", buffering=1) as log_handle:
            process = subprocess.Popen(
                [
                    str(python), "-m", "tensorboard.main",
                    f"--logdir={tensorboard_dir}", f"--port={actual_port}", "--host=127.0.0.1",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Register ownership immediately, before any startup check can be interrupted.
            _CURRICULUM_TENSORBOARD_PROCESS = process
            _CURRICULUM_TENSORBOARD_PORT = actual_port
        time.sleep(0.6)
        if process.poll() is not None:
            print(f"[TensorBoard Warning] Server exited; inspect {server_log}. Metrics will still be written.", flush=True)
            stop_managed_process(
                process, "failed TensorBoard startup", graceful_signal=signal.SIGTERM,
                graceful_timeout=0.0, terminate_timeout=0.5,
            )
            _CURRICULUM_TENSORBOARD_PROCESS = None
            _CURRICULUM_TENSORBOARD_PORT = None
        else:
            print(f"[TensorBoard] http://127.0.0.1:{actual_port} | logdir={tensorboard_dir}", flush=True)
            print(f"[TensorBoard] This curriculum owns the server; it will be stopped when the curriculum exits. Log={server_log}", flush=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[TensorBoard Warning] Could not start TensorBoard: {exc}. Metrics will still be written.", flush=True)


def stop_curriculum_tensorboard() -> None:
    """Stop only the TensorBoard server started by this curriculum invocation."""
    global _CURRICULUM_TENSORBOARD_PROCESS, _CURRICULUM_TENSORBOARD_PORT
    process = _CURRICULUM_TENSORBOARD_PROCESS
    if process is None:
        return
    port = _CURRICULUM_TENSORBOARD_PORT
    print(f"[TensorBoard] Stopping curriculum-owned server{f' on port {port}' if port else ''}...", flush=True)
    try:
        stop_managed_process(
            process,
            "curriculum TensorBoard",
            graceful_signal=signal.SIGTERM,
            graceful_timeout=3.0,
            terminate_timeout=2.0,
        )
        print("[TensorBoard] Curriculum-owned server stopped.", flush=True)
    finally:
        _CURRICULUM_TENSORBOARD_PROCESS = None
        _CURRICULUM_TENSORBOARD_PORT = None


# Last-resort cleanup also covers exceptions/signals before the stage loop starts.
atexit.register(stop_curriculum_tensorboard)


def copy_checkpoint_before_training(root: Path, model_name: str) -> Path | None:
    checkpoint = root / f"{model_name}.zip"
    if not checkpoint.is_file():
        return None
    destination_dir = root / "runtime_data" / "checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = destination_dir / f"{model_name}.pre_curriculum_{stamp}_{os.getpid()}.zip"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(checkpoint, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(f"[Curriculum] Preserved current model before any training: {destination}", flush=True)
    return destination


def newest_report(log_dir: Path, before: set[Path]) -> Path | None:
    new_paths = set(log_dir.glob("evaluation_*.json")) - before
    candidates = [p for p in new_paths if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


_PARENT_STOP_SIGNAL_SEEN = False


def _handle_parent_stop_signal(signum, frame) -> None:
    global _PARENT_STOP_SIGNAL_SEEN
    # Ignore repeated stop signals while cleanup is in progress; the child supervisor
    # performs timed SIGINT -> SIGTERM -> SIGKILL escalation if a child refuses to exit.
    if _PARENT_STOP_SIGNAL_SEEN:
        return
    _PARENT_STOP_SIGNAL_SEEN = True
    try:
        signal_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        signal_name = str(signum)
    print(f"\n[Curriculum] {signal_name} received; stopping active work and cleaning up child processes...", flush=True)
    raise KeyboardInterrupt


def main() -> int:
    # Route Ctrl+C, SIGTERM, and terminal hangup through the same shutdown path.
    # The child process has its own session, so it receives the graceful signal only
    # when stream_command forwards it, not concurrently from the terminal.
    signal.signal(signal.SIGINT, _handle_parent_stop_signal)
    signal.signal(signal.SIGTERM, _handle_parent_stop_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_parent_stop_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start-stage", choices=STAGES, default=None,
                        help="override the saved curriculum stage; otherwise resume a prior interrupted run")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without launching emulators")
    args = parser.parse_args()
    root = args.project_dir.expanduser().resolve()
    if not (root / "main.py").is_file() or not (root / "PokemonEmerald.gba").is_file():
        raise SystemExit(f"Project or ROM not found under {root}; run from the correct project directory.")
    cfg = parse_dotenv(root / ".env")
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        python = Path(sys.executable)

    model_name = os.environ.get("POKEMON_AI_MODEL_NAME", cfg.get("POKEMON_AI_MODEL_NAME", "emerald_ppo_hybrid_v1"))
    if not model_name or Path(model_name).name != model_name:
        raise SystemExit(f"Unsafe POKEMON_AI_MODEL_NAME: {model_name!r}")
    expected_rom_hash = os.environ.get(
        "POKEMON_AI_EXPECTED_ROM_SHA1", cfg.get("POKEMON_AI_EXPECTED_ROM_SHA1", "f3ae088181bf583e55daf962a92bb46f4f1d07b7")
    ).strip().lower()
    if len(expected_rom_hash) != 40 or any(c not in "0123456789abcdef" for c in expected_rom_hash):
        raise SystemExit("POKEMON_AI_EXPECTED_ROM_SHA1 must be a 40-character hexadecimal SHA-1.")

    initial_state_config = os.environ.get(
        "POKEMON_AI_TRAINING_START_STATE", cfg.get("POKEMON_AI_TRAINING_START_STATE", "runtime_data/states/training_start.state")
    )
    initial_state = resolve_path(root, initial_state_config)
    milestone_dir = root / "runtime_data" / "states" / "milestones"
    log_dir = root / "logs"
    run_state_path = root / "runtime_data" / "curriculum_state.json"
    run_log_path = log_dir / f"curriculum_manager_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"

    chunk_steps = int_setting(cfg, "POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS", 250_000)
    max_chunks = int_setting(cfg, "POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE", 8)
    eval_episodes = int_setting(cfg, "POKEMON_AI_CURRICULUM_EVAL_EPISODES", 5)
    threshold = float_setting(cfg, "POKEMON_AI_CURRICULUM_SUCCESS_RATE", 0.8)
    deterministic = bool_setting(cfg, "POKEMON_AI_CURRICULUM_EVAL_DETERMINISTIC", True)
    compare_checkpoints = int_setting(cfg, "POKEMON_AI_CURRICULUM_COMPARE_CHECKPOINTS", 2, minimum=0)
    if threshold <= 0:
        raise SystemExit("POKEMON_AI_CURRICULUM_SUCCESS_RATE must be greater than zero.")

    current_stage = args.start_stage
    if current_stage is None and run_state_path.is_file():
        try:
            saved_state = json.loads(run_state_path.read_text(encoding="utf-8"))
            if saved_state.get("status") in {"running", "interrupted", "stage_not_mastered"} and saved_state.get("stage") in STAGES:
                current_stage = saved_state["stage"]
                print(f"[Curriculum] Resuming from saved stage {current_stage!r} (it will be re-evaluated first).", flush=True)
        except (OSError, json.JSONDecodeError, AttributeError):
            print("[Curriculum Warning] Prior curriculum state is unreadable; starting from the first stage.", flush=True)
    if current_stage is None:
        current_stage = "starter"
    start_index = STAGES.index(current_stage)

    checkpoint = root / f"{model_name}.zip"
    if not checkpoint.is_file() and current_stage != "starter":
        raise SystemExit(
            f"No model checkpoint {checkpoint}; curriculum must start with 'starter' when training a new model."
        )
    if current_stage != "starter":
        required_prior = milestone_dir / f"{STAGES[STAGES.index(current_stage)-1]}.state"
        required_manifest = Path(str(required_prior) + ".json")
        if not required_prior.is_file() or not required_manifest.is_file():
            raise SystemExit(
                f"Cannot start at {current_stage}: prerequisite milestone is missing or unverified: {required_prior}"
            )

    print("=" * 72, flush=True)
    print(" Pokémon Emerald curriculum controller", flush=True)
    print(f" ROM expected SHA-1 : {expected_rom_hash}", flush=True)
    print(f" Model checkpoint   : {checkpoint}", flush=True)
    print(f" Evaluation gate    : {'deterministic' if deterministic else 'stochastic'}; "
          f"{eval_episodes} episodes; success rate >= {threshold:.0%}", flush=True)
    print(f" Train chunk        : {chunk_steps:,} transitions; maximum {max_chunks} chunks per stage", flush=True)
    print(f" Checkpoint recovery: compare up to {compare_checkpoints} recent distinct archived models when the current model misses the gate", flush=True)
    print(" Milestones         : starter -> Pokédex -> first Gym Badge", flush=True)
    print("=" * 72, flush=True)

    if args.dry_run:
        print("\n[Dry-run] No emulator will be started and no files/checkpoints will be changed.", flush=True)
        for stage in STAGES[start_index:]:
            stage_start = initial_state if stage == "starter" else milestone_dir / f"{STAGES[STAGES.index(stage)-1]}.state"
            print(f"\n[Dry-run] Stage {stage}: start state={stage_start}", flush=True)
            if checkpoint.is_file():
                print(f"  1. Evaluate current checkpoint ({eval_episodes} episodes): {checkpoint}", flush=True)
                if compare_checkpoints:
                    print(f"  2. If below gate, screen up to {compare_checkpoints} recent distinct archives and fully re-evaluate the best candidate.", flush=True)
            else:
                print("  1. No checkpoint yet; start a fresh model for the starter stage.", flush=True)
            print(f"  3. If not mastered, train up to {max_chunks} chunk(s) of {chunk_steps:,} transitions each; evaluate after each chunk.", flush=True)
            print(f"  4. Promote only after success_rate >= {threshold:.0%} and a fresh {stage}.state + matching manifest are created.", flush=True)
        print("\n[Dry-run] Plan preview complete (exit 0).", flush=True)
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    # Unlike main.py's one-process training mode, the curriculum launches many child
    # training/evaluation processes. Start exactly one server here and disable child auto-start.
    try:
        start_curriculum_tensorboard(root, python, cfg, log_dir)
        copy_checkpoint_before_training(root, model_name)
    except KeyboardInterrupt:
        print("\n[Curriculum] Interrupted during startup; cleaning up owned processes.", flush=True)
        stop_curriculum_tensorboard()
        return 130

    # Profile variables are set in the environment without modifying .env or main.py.
    base_env = os.environ.copy()
    base_env["POKEMON_AI_EXPECTED_ROM_SHA1"] = expected_rom_hash
    base_env["POKEMON_AI_REQUIRE_ROM_SHA1"] = "1"
    base_env["POKEMON_AI_MILESTONE_SAVE"] = "1"
    base_env["POKEMON_AI_RESUME"] = "1"
    base_env["POKEMON_AI_AUTO_START_TENSORBOARD"] = "0"
    base_env["POKEMON_AI_TENSORBOARD_LOG_DIR"] = str(resolve_path(
        root, os.environ.get("POKEMON_AI_TENSORBOARD_LOG_DIR", cfg.get("POKEMON_AI_TENSORBOARD_LOG_DIR", "tensorboard_logs_v2"))
    ))
    base_env["POKEMON_AI_EVAL_EPISODES"] = str(eval_episodes)
    base_env["POKEMON_AI_EVAL_DETERMINISTIC"] = "1" if deterministic else "0"
    base_env["POKEMON_AI_REQUIRE_START_STATE"] = "1"

    def eval_stage(
        stage: str,
        start_state: Path,
        checkpoint_path: Path | None = None,
        episode_count: int | None = None,
    ) -> dict[str, Any] | None:
        requested_episodes = int(episode_count if episode_count is not None else eval_episodes)
        report_before = set(log_dir.glob("evaluation_*.json")) if log_dir.is_dir() else set()
        evaluation_started_unix = time.time()
        env = base_env.copy()
        env["POKEMON_AI_PROFILE"] = "evaluate"
        env["POKEMON_AI_MODE"] = "evaluate"
        env["POKEMON_AI_GUI_MODE"] = "0"
        env["POKEMON_AI_RECORD_VIDEO"] = "0"
        env["POKEMON_AI_AUTO_START_TENSORBOARD"] = "0"
        env["POKEMON_AI_OBJECTIVE"] = stage
        env["POKEMON_AI_TRAINING_START_STATE"] = str(start_state)
        env.pop("POKEMON_AI_EVAL_CHECKPOINT", None)
        if checkpoint_path is not None:
            env["POKEMON_AI_EVAL_CHECKPOINT"] = str(checkpoint_path)
        env["POKEMON_AI_EVAL_SEED"] = os.environ.get("POKEMON_AI_EVAL_SEED", cfg.get("POKEMON_AI_EVAL_SEED", "1234"))
        command = [str(python), "main.py"]
        stream_command(command, env, root, run_log_path, args.dry_run)
        if args.dry_run:
            return None
        report_path = newest_report(log_dir, report_before)
        if report_path is None:
            raise RuntimeError("Evaluation exited without a new machine-readable report; refusing to promote.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("objective") != stage or report.get("mode") != ("deterministic" if deterministic else "stochastic"):
            raise RuntimeError(f"Evaluation report does not match the requested stage/mode: {report_path}")
        if int(report.get("episodes_completed", 0)) != requested_episodes:
            raise RuntimeError(
                f"Evaluation report contains {report.get('episodes_completed')} completed episodes; "
                f"expected {requested_episodes}. Refusing to promote."
            )
        wanted_checkpoint = (checkpoint_path or checkpoint).resolve()
        reported_checkpoint = report.get("checkpoint")
        if not isinstance(reported_checkpoint, str) or Path(reported_checkpoint).resolve() != wanted_checkpoint:
            raise RuntimeError(
                f"Evaluation report checkpoint does not match the requested candidate {wanted_checkpoint}: "
                f"{reported_checkpoint!r}. Refusing to use stale or mismatched results."
            )
        reported_sha1 = report.get("checkpoint_sha1")
        if not isinstance(reported_sha1, str) or not re.fullmatch(r"[0-9a-f]{40}", reported_sha1):
            raise RuntimeError(f"Evaluation report has no valid checkpoint SHA-1: {report_path}")
        if not wanted_checkpoint.is_file() or file_sha1(wanted_checkpoint) != reported_sha1:
            raise RuntimeError(
                f"Evaluation report checkpoint SHA-1 does not match the actual file: {wanted_checkpoint}. "
                "Refusing to use stale or mismatched results."
            )
        report["_controller_evaluation_started_unix"] = evaluation_started_unix
        print(
            f"[Curriculum Evaluation] stage={stage} successes={report.get('successes')}/"
            f"{requested_episodes} rate={float(report.get('success_rate', 0.0)):.1%} "
            f"mean_steps={float(report.get('mean_steps', 0.0)):.0f} "
            f"report={report_path}", flush=True,
        )
        return report

    def verified_fresh_milestone(stage: str, report: dict[str, Any] | None) -> bool:
        if not report:
            return False
        milestone = milestone_dir / f"{stage}.state"
        manifest_path = Path(str(milestone) + ".json")
        if not milestone.is_file() or not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if manifest.get("objective") != stage or manifest.get("rom_sha1_expected") != expected_rom_hash:
            return False
        started_unix = report.get("_controller_evaluation_started_unix")
        if not isinstance(started_unix, (int, float)):
            return False
        return milestone_matches_evaluation(manifest, report, expected_rom_hash, float(started_unix))

    def compare_archived_checkpoints(
        stage: str,
        start_state: Path,
        baseline_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Screen recent archived models if the current model misses the stage gate.

        Screening uses one deterministic trajectory by default, then subjects an
        apparently better archive to the full fixed evaluation before promotion.
        Candidates are content-distinct and ZIP-integrity checked. Training resumes
        from the best full-evaluation result, but only a freshly verified stage state
        can unlock the next objective.
        """
        baseline_score = report_score(baseline_report)
        best_screen_score = baseline_score
        best_screen_path: Path | None = None
        candidates = archived_checkpoint_candidates(root, model_name, checkpoint, compare_checkpoints)
        if not candidates:
            return baseline_report
        print(
            f"[Curriculum] Current model missed the gate; screening {len(candidates)} recent archived checkpoint(s).",
            flush=True,
        )
        for candidate in candidates:
            try:
                screen_report = eval_stage(stage, start_state, checkpoint_path=candidate, episode_count=1)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[Curriculum Warning] Could not evaluate archive {candidate}: {exc}", flush=True)
                continue
            if screen_report is None:
                continue
            score = report_score(screen_report)
            print(f"[Checkpoint Compare] {candidate.name} score={score} baseline={baseline_score}", flush=True)
            if score > best_screen_score:
                best_screen_score = score
                best_screen_path = candidate

        if best_screen_path is None:
            print("[Checkpoint Compare] No recent archive outperformed the current checkpoint on the screening run.", flush=True)
            return baseline_report

        print(
            f"[Checkpoint Compare] Full evaluation of promising archive: {best_screen_path}",
            flush=True,
        )
        try:
            full_report = eval_stage(
                stage, start_state, checkpoint_path=best_screen_path, episode_count=eval_episodes
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[Curriculum Warning] Full candidate evaluation failed: {exc}", flush=True)
            return baseline_report
        if full_report is None or report_score(full_report) <= baseline_score:
            print("[Checkpoint Compare] Candidate did not outperform the baseline on the full test; keep current model.", flush=True)
            return baseline_report

        backup = activate_checkpoint_candidate(
            best_screen_path, checkpoint, root / "runtime_data" / "checkpoints"
        )
        print(
            f"[Checkpoint Compare] Activated improved checkpoint {best_screen_path.name}; "
            f"previous model backup={backup if backup else 'not needed'}; full score={report_score(full_report)}",
            flush=True,
        )
        full_report["selected_checkpoint_activated"] = True
        full_report["selected_checkpoint_source"] = str(best_screen_path.resolve())
        full_report["baseline_score_before_candidate"] = list(baseline_score)
        return full_report

    try:
        for stage_index in range(start_index, len(STAGES)):
            stage = STAGES[stage_index]
            if stage == "starter":
                stage_start_state = initial_state
            else:
                prior = STAGES[stage_index - 1]
                stage_start_state = milestone_dir / f"{prior}.state"
                manifest_path = Path(str(stage_start_state) + ".json")
                if not stage_start_state.is_file() or not manifest_path.is_file():
                    raise RuntimeError(
                        f"Stage {stage} requires the prior {prior} milestone and metadata: {stage_start_state}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("objective") != prior or manifest.get("rom_sha1_expected") != expected_rom_hash:
                    raise RuntimeError(f"Milestone metadata does not match required objective/ROM: {manifest_path}")

            print(f"\n[Curriculum] STAGE {stage.upper()}: {OBJECTIVE_LABELS[stage]} | start={stage_start_state}", flush=True)
            if not args.dry_run and not stage_start_state.is_file():
                raise FileNotFoundError(
                    f"Required start state is missing: {stage_start_state}. Save the canonical start or finish the preceding milestone first."
                )

            mastered = False
            chunks_trained = 0
            # Always evaluate the current checkpoint first; avoid needless retraining if already competent.
            report = None
            if checkpoint.is_file():
                report = eval_stage(stage, stage_start_state)
                if report is not None and float(report.get("success_rate", 0.0)) >= threshold:
                    if verified_fresh_milestone(stage, report):
                        mastered = True
                    else:
                        print(
                            f"[Curriculum] Success threshold met but milestone state is absent: "
                            f"{milestone_dir / (stage + '.state')}. "
                            "Will continue training/evaluation rather than advance.", flush=True
                        )
                if report is not None and not mastered:
                    report = compare_archived_checkpoints(stage, stage_start_state, report)
                    if float(report.get("success_rate", 0.0)) >= threshold and verified_fresh_milestone(stage, report):
                        mastered = True

            while not mastered and chunks_trained < max_chunks:
                chunks_trained += 1
                print(
                    f"\n[Curriculum] Training stage={stage} chunk={chunks_trained}/{max_chunks} "
                    f"additional_transitions={chunk_steps:,}.", flush=True,
                )
                train_env = base_env.copy()
                train_env["POKEMON_AI_PROFILE"] = "fast_train"
                train_env["POKEMON_AI_MODE"] = "train"
                train_env["POKEMON_AI_GUI_MODE"] = "0"
                train_env["POKEMON_AI_RECORD_VIDEO"] = "0"
                train_env["POKEMON_AI_AUTO_START_TENSORBOARD"] = "0"
                train_env["POKEMON_AI_OBJECTIVE"] = stage
                train_env["POKEMON_AI_TRAINING_START_STATE"] = str(stage_start_state)
                train_env["POKEMON_AI_TOTAL_TIMESTEPS"] = str(chunk_steps)
                stream_command([str(python), "main.py"], train_env, root, run_log_path, args.dry_run)
                if args.dry_run:
                    continue
                report = eval_stage(stage, stage_start_state)
                if report is None:
                    continue
                if float(report.get("success_rate", 0.0)) >= threshold:
                    mastered = verified_fresh_milestone(stage, report)
                    if not mastered:
                        print(
                            f"[Curriculum] Gate met, but milestone state/manifest failed validation. "
                            "No promotion.", flush=True
                        )
                if not mastered:
                    report = compare_archived_checkpoints(stage, stage_start_state, report)
                    if float(report.get("success_rate", 0.0)) >= threshold:
                        mastered = verified_fresh_milestone(stage, report)

                if not args.dry_run:
                    atomic_json(run_state_path, {
                        "schema_version": 1,
                        "status": "running" if mastered else "stage_not_mastered",
                        "stage": stage,
                        "chunks_trained_this_invocation": chunks_trained,
                        "chunk_timesteps": chunk_steps,
                        "success_threshold": threshold,
                        "last_evaluation_report": str(newest_report(log_dir, set()) or ""),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })

            if not mastered:
                if not args.dry_run:
                    atomic_json(run_state_path, {
                        "schema_version": 1, "status": "stage_not_mastered", "stage": stage,
                        "chunks_trained_this_invocation": chunks_trained, "chunk_timesteps": chunk_steps,
                        "success_threshold": threshold, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })
                print(
                    f"[Curriculum] NOT PROMOTED: {stage} did not meet the evaluation gate after "
                    f"{chunks_trained} training chunks. Your checkpoint is preserved; rerun to continue.",
                    flush=True,
                )
                return 2

            print(f"[Curriculum] MASTERED: {stage} ({OBJECTIVE_LABELS[stage]}); milestone verified.", flush=True)
            if not args.dry_run:
                atomic_json(run_state_path, {
                    "schema_version": 1, "status": "mastered", "stage": stage,
                    "chunks_trained_this_invocation": chunks_trained, "chunk_timesteps": chunk_steps,
                    "success_threshold": threshold, "milestone_state": str(milestone_dir / f"{stage}.state"),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })

        if not args.dry_run:
            atomic_json(run_state_path, {
                "schema_version": 1, "status": "complete", "stage": STAGES[-1],
                "milestones": [str(milestone_dir / f"{stage}.state") for stage in STAGES],
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        print("\n[Curriculum] COMPLETE: all configured stages passed the evaluation gate.", flush=True)
        return 0
    except KeyboardInterrupt:
        if not args.dry_run:
            atomic_json(run_state_path, {
                "schema_version": 1, "status": "interrupted", "stage": stage if 'stage' in locals() else current_stage,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        print("[Curriculum] Interrupted. Model/state checkpoints are retained; rerun to re-evaluate and resume.", flush=True)
        return 130
    except Exception as exc:
        if not args.dry_run:
            try:
                atomic_json(run_state_path, {
                    "schema_version": 1, "status": "error", "stage": stage if 'stage' in locals() else current_stage,
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
            except Exception:
                pass
        raise
    finally:
        # Covers success, stage limit, errors, and KeyboardInterrupt/SIGTERM.
        stop_curriculum_tensorboard()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[Curriculum Error] {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
