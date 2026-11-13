#!/usr/bin/env python3
"""Safely train/evaluate/promote Pokémon Emerald curriculum stages.

Uses only the Python standard library. Run from the project's virtualenv:
    ./.venv/bin/python run_curriculum.py --dry-run
    ./.venv/bin/python run_curriculum.py

Normal mode evaluates an existing checkpoint and promotes only after a verified
fixed-seed objective evaluation. Stage-only mode (--stage-only or
POKEMON_AI_CURRICULUM_STAGE_ONLY=1) trains one selected stage for the configured
number of chunks without evaluating, comparing checkpoints, or promoting.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
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

from important_logging import filter_persistent_log_text, is_important_log_line, should_flush_important_log, is_exception_end
from curriculum_objectives import (
    CURRICULUM_PLAN_ID, OBJECTIVE_LABELS, OBJECTIVES_REQUIRING_FLAG_TELEMETRY, STAGES,
    evaluation_gate_unreachable, objective_satisfied, objective_start_error, stage_start_state_path,
    stage_episode_rollouts,
)


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


def stage_only_requested(cli_enabled: bool, values: dict[str, str]) -> bool:
    """Stage-only mode is enabled by CLI or .env/shell; it never promotes or evaluates."""
    return bool(cli_enabled or bool_setting(values, "POKEMON_AI_CURRICULUM_STAGE_ONLY", False))


def report_shell_config_overrides(values: dict[str, str]) -> None:
    """Make important shell-over-.env precedence visible instead of silently surprising users."""
    keys = (
        "POKEMON_AI_OBJECTIVE",
        "POKEMON_AI_CURRICULUM_STAGE_ONLY",
        "POKEMON_AI_CANONICAL_START_STATE",
        "POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE",
        "POKEMON_AI_CURRICULUM_EVAL_DETERMINISTIC",
    )
    for key in keys:
        shell_value = os.environ.get(key)
        dotenv_value = values.get(key)
        if shell_value is None or dotenv_value is None:
            continue
        shell_normalized = shell_value.strip()
        dotenv_normalized = dotenv_value.strip()
        if key in {"POKEMON_AI_OBJECTIVE", "POKEMON_AI_CURRICULUM_STAGE_ONLY"}:
            shell_normalized = shell_normalized.lower()
            dotenv_normalized = dotenv_normalized.lower()
        if shell_normalized != dotenv_normalized:
            print(
                f"[Config Warning] Shell environment overrides .env: {key}={shell_value!r}; "
                f".env contains {dotenv_value!r}. The shell value is active.",
                flush=True,
            )


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


def milestone_manifest_satisfies_objective(stage: str, manifest: dict[str, Any]) -> bool:
    """Require the persisted decoded state itself to prove its named objective."""
    if not isinstance(manifest, dict) or manifest.get("objective") != stage:
        return False
    state_metrics = manifest.get("state")
    if not isinstance(state_metrics, dict):
        return False
    return objective_satisfied(stage, state_metrics)


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


def _episode_progress_score(episode: dict[str, Any]) -> float:
    """Score observable game progress for checkpoint screening, not policy activity."""
    def nonnegative_int(key: str) -> float:
        value = episode.get(key, 0)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(number):
            return 0.0
        return max(0.0, number)

    # This is deliberately a secondary tiebreaker only. Do not score raw/latched
    # story flags, trainer flags, party totals or badge counts: those were the exact
    # telemetry channels that produced inflated scores in the latest evaluation.
    # Objective success rate is always the first and mandatory promotion signal.
    return (
        nonnegative_int("unique_tiles")
        + 16.0 * nonnegative_int("unique_maps")
        + nonnegative_int("party_experience_gained") / 100.0
    )


def report_score(report: dict[str, Any]) -> tuple[float, float, float, float]:
    """Rank checkpoints by success, actual game progress, return, then efficiency."""
    def finite(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if not math.isfinite(number):
            return fallback
        return number

    # Partial, gate-unreachable reports must not look artificially strong merely
    # because their observed denominator is small (e.g. 1 success in 2 attempts).
    success_rate = max(0.0, min(1.0, finite(
        report.get("success_rate_over_requested", report.get("success_rate")), 0.0
    )))
    episodes = report.get("episodes", [])
    if isinstance(episodes, list) and episodes:
        progress = sum(
            _episode_progress_score(item) for item in episodes if isinstance(item, dict)
        ) / max(1, sum(1 for item in episodes if isinstance(item, dict)))
    else:
        # Backward compatibility for older reports that predate progress telemetry.
        progress = finite(report.get("mean_progress_score"), 0.0)
    mean_return = finite(report.get("mean_return"), float("-inf"))
    mean_steps = finite(report.get("mean_steps"), float("inf"))
    return success_rate, progress, mean_return, -mean_steps


RESUMABLE_CURRICULUM_STATUSES = frozenset({
    "running", "in_progress", "interrupted", "stalled", "ready", "stage_not_mastered",
    "stage_only_training", "stage_only_budget_complete",
})

def resolve_saved_curriculum_stage(saved_state: dict[str, Any]) -> tuple[str | None, str]:
    """Resolve a resumable stage with strict version-aware migration.

    v5 and v6 retain the same 58-stage order and predecessor chain. A v5 stage is
    eligible for migration, but its previous milestone is still revalidated using
    the corrected v6 event predicates before the stage may launch. The earlier v4
    migration is limited to the two stages whose semantics and predecessor state
    were unchanged when Route 101 was split into its own stage.
    """
    if not isinstance(saved_state, dict):
        return None, "unreadable"
    status = str(saved_state.get("status", "")).strip().lower()
    stage = saved_state.get("stage")
    plan = saved_state.get("curriculum_plan")
    if status == "complete":
        if plan != CURRICULUM_PLAN_ID or stage != STAGES[-1]:
            return None, "incompatible_plan"
        return None, "complete"
    if status == "mastered":
        if stage not in STAGES:
            return None, "unknown_stage"
        if plan != CURRICULUM_PLAN_ID:
            return None, "incompatible_plan"
        next_index = STAGES.index(stage) + 1
        if next_index >= len(STAGES):
            return None, "complete"
        return STAGES[next_index], "after_mastered"
    if status not in RESUMABLE_CURRICULUM_STATUSES:
        return None, "not_resumable"
    if stage not in STAGES:
        return None, "unknown_stage"
    if plan == CURRICULUM_PLAN_ID:
        return stage, "current"
    if plan == "emerald_intro_v5":
        # v6 preserves the exact 58-stage order and start-state chain. Every
        # predecessor is still revalidated against the corrected v6 predicates
        # before it is used, so only genuinely valid old milestones survive.
        return stage, "migrated_v5_revalidated"
    if plan == "emerald_intro_v4" and stage in {"starter", "leave_birch_lab"}:
        return stage, "migrated_v4_early_stage"
    return None, "incompatible_plan"


def stage_training_episode_step_limit(stage: str, n_steps: int, override: int | None = None) -> int:
    """Return the per-environment training episode cap for one curriculum stage."""
    if int(n_steps) < 1:
        raise ValueError("POKEMON_AI_PPO_N_STEPS must be positive")
    if override is None:
        return stage_episode_rollouts(stage) * int(n_steps)
    if int(override) < 1:
        raise ValueError("Training episode-step override must be positive")
    # Keep the episode horizon aligned to per-environment PPO rollout lengths.
    return math.ceil(int(override) / int(n_steps)) * int(n_steps)


def align_training_chunk_steps(base_steps: int, n_steps: int, num_envs: int) -> int:
    """Round a requested transition budget up to a complete vector PPO rollout."""
    if min(int(base_steps), int(n_steps), int(num_envs)) < 1:
        raise ValueError("Training chunk size, PPO n_steps, and environment count must be positive")
    vector_rollout = int(n_steps) * int(num_envs)
    return math.ceil(int(base_steps) / vector_rollout) * vector_rollout


def stage_training_chunk_steps(
    stage: str,
    base_steps: int,
    n_steps: int,
    num_envs: int = 4,
    episode_step_override: int | None = None,
) -> int:
    """Size a curriculum chunk for at least one capped episode in every worker."""
    if int(base_steps) < 1:
        raise ValueError("Training chunk size must be positive")
    episode_limit = stage_training_episode_step_limit(stage, n_steps, episode_step_override)
    minimum_episode_budget = episode_limit * int(num_envs)
    return align_training_chunk_steps(max(int(base_steps), minimum_episode_budget), n_steps, num_envs)


def build_training_environment(
    base_env: dict[str, str],
    *,
    stage: str,
    start_state: Path,
    total_timesteps: int,
    max_episode_steps: int,
    stage_only: bool = False,
    reset_entropy_for_stage: bool = False,
) -> dict[str, str]:
    """Build a stage-isolated training child environment with the correct episode cap."""
    if int(total_timesteps) < 1 or int(max_episode_steps) < 1:
        raise ValueError("Training transition and episode-step limits must be positive")
    env = base_env.copy()
    env.update({
        "POKEMON_AI_PROFILE": "fast_train",
        "POKEMON_AI_MODE": "train",
        "POKEMON_AI_GUI_MODE": "0",
        "POKEMON_AI_RECORD_VIDEO": "0",
        "POKEMON_AI_AUTO_START_TENSORBOARD": "0",
        "POKEMON_AI_OBJECTIVE": stage,
        "POKEMON_AI_TRAINING_START_STATE": str(start_state),
        "POKEMON_AI_TOTAL_TIMESTEPS": str(int(total_timesteps)),
        # The .env value is a standalone/single-stage setting; the curriculum must
        # override it for every objective as it advances through the stage list.
        "POKEMON_AI_MAX_EPISODE_STEPS": str(int(max_episode_steps)),
        "POKEMON_AI_CURRICULUM_STAGE_ONLY": "1" if stage_only else "0",
        # Reset exploration at the first chunk of a new stage, then resume it from
        # checkpoint state on any subsequent chunk for that same stage.
        "POKEMON_AI_ENTROPY_RESET_FOR_STAGE": "1" if reset_entropy_for_stage else "0",
    })
    return env


def stage_evaluation_step_limit(stage: str, n_steps: int, override: int | None = None) -> int:
    """Resolve an evaluation horizon from the selected objective, not later stages."""
    if override is not None:
        if int(override) < 1:
            raise ValueError("Evaluation step override must be positive")
        return int(override)
    if int(n_steps) < 1:
        raise ValueError("POKEMON_AI_PPO_N_STEPS must be positive")
    return stage_episode_rollouts(stage) * int(n_steps)


def build_evaluation_environment(
    base_env: dict[str, str], *, stage: str, start_state: Path, requested_episodes: int,
    deterministic: bool, eval_seed: str, max_episode_steps: int, no_progress_steps: int = 4096,
    checkpoint_path: Path | None = None, stop_if_gate_unreachable: bool = False,
    success_threshold: float = 0.9,
) -> dict[str, str]:
    """Build an isolated, exact evaluation subprocess configuration."""
    if requested_episodes < 1 or max_episode_steps < 1 or no_progress_steps < 128:
        raise ValueError("Evaluation episode count/step limit must be positive and no-progress guard >= 128")
    if not math.isfinite(float(success_threshold)) or not 0.0 < float(success_threshold) <= 1.0:
        raise ValueError("Evaluation success threshold must be in (0, 1]")
    env = base_env.copy()
    env.update({
        "POKEMON_AI_PROFILE": "evaluate",
        "POKEMON_AI_MODE": "evaluate",
        "POKEMON_AI_GUI_MODE": "0",
        "POKEMON_AI_RECORD_VIDEO": "0",
        "POKEMON_AI_AUTO_START_TENSORBOARD": "0",
        "POKEMON_AI_OBJECTIVE": stage,
        "POKEMON_AI_TRAINING_START_STATE": str(start_state),
        # Critical: episode_count must reach the child, not just report validation.
        "POKEMON_AI_EVAL_EPISODES": str(int(requested_episodes)),
        "POKEMON_AI_EVAL_DETERMINISTIC": "1" if deterministic else "0",
        "POKEMON_AI_EVAL_MAX_EPISODE_STEPS": str(int(max_episode_steps)),
        "POKEMON_AI_EVAL_NO_PROGRESS_STEPS": str(int(no_progress_steps)),
        "POKEMON_AI_EVAL_SEED": str(eval_seed),
        "POKEMON_AI_EVAL_STOP_IF_GATE_UNREACHABLE": "1" if stop_if_gate_unreachable else "0",
        "POKEMON_AI_EVAL_SUCCESS_THRESHOLD": str(float(success_threshold)),
    })
    env.pop("POKEMON_AI_EVAL_CHECKPOINT", None)
    if checkpoint_path is not None:
        env["POKEMON_AI_EVAL_CHECKPOINT"] = str(checkpoint_path)
    return env


def validate_evaluation_report_progress(
    report: dict[str, Any], requested_episodes: int, threshold: float,
) -> tuple[int, int, float]:
    """Validate full or mathematically-futile early-stopped evaluation reports."""
    if requested_episodes < 1 or not math.isfinite(float(threshold)) or not 0.0 < threshold <= 1.0:
        raise ValueError("Requested episodes and evaluation threshold must be valid")
    try:
        reported_requested = int(report.get("episodes_requested", 0))
        completed = int(report.get("episodes_completed", 0))
        successes = int(report.get("successes", -1))
        success_rate = float(report.get("success_rate", float("nan")))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Evaluation report has invalid episode statistics") from exc
    if reported_requested != requested_episodes:
        raise RuntimeError(
            f"Evaluation report requested {reported_requested} episodes; controller requested {requested_episodes}."
        )
    if completed < 1 or completed > requested_episodes:
        raise RuntimeError(
            f"Evaluation report completed {completed} episodes; expected between 1 and {requested_episodes}."
        )
    if not 0 <= successes <= completed or not math.isfinite(success_rate):
        raise RuntimeError("Evaluation report has out-of-range success statistics")
    expected_rate = successes / completed
    if not math.isclose(success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("Evaluation report success_rate disagrees with successes/episodes_completed")
    episodes = report.get("episodes")
    if (
        not isinstance(episodes, list)
        or len(episodes) != completed
        or any(not isinstance(item, dict) or not isinstance(item.get("success"), bool) for item in episodes)
    ):
        raise RuntimeError("Evaluation report episode details do not match episodes_completed")
    episode_successes = sum(1 for item in episodes if item["success"] is True)
    if episode_successes != successes:
        raise RuntimeError("Evaluation report successes disagrees with per-episode success details")

    stopped_early = bool(report.get("evaluation_stopped_early", False))
    stop_reason = report.get("evaluation_stop_reason")
    remaining = requested_episodes - completed
    reported_threshold = report.get("evaluation_gate_threshold")
    if reported_threshold is not None:
        try:
            parsed_threshold = float(reported_threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Evaluation report has an invalid success-gate threshold") from exc
        if not math.isfinite(parsed_threshold) or not math.isclose(
            parsed_threshold, threshold, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("Evaluation report success-gate threshold does not match the controller")

    if remaining:
        if not stopped_early or stop_reason != "gate_unreachable":
            raise RuntimeError("Evaluation report is incomplete without a validated gate-unreachable stop")
        if not evaluation_gate_unreachable(successes, completed, requested_episodes, threshold):
            raise RuntimeError(
                "Evaluation stopped early even though the configured success threshold remained reachable"
            )
        if reported_threshold is None:
            raise RuntimeError("Early-stopped evaluation report is missing its success-gate threshold")
    elif stopped_early:
        raise RuntimeError("Evaluation report claims an early stop despite completing every requested episode")
    elif stop_reason not in (None, ""):
        raise RuntimeError("Completed evaluation report has an unexpected early-stop reason")

    reported_requested_rate = report.get("success_rate_over_requested")
    if reported_requested_rate is not None:
        try:
            requested_rate = float(reported_requested_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Evaluation report has an invalid success_rate_over_requested") from exc
        expected_requested_rate = successes / requested_episodes
        if not math.isfinite(requested_rate) or not math.isclose(
            requested_rate, expected_requested_rate, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("Evaluation report success_rate_over_requested is inconsistent")

    reported_upper_bound = report.get("success_rate_upper_bound")
    expected_upper_bound = (successes + remaining) / requested_episodes
    if reported_upper_bound is not None:
        try:
            upper_bound = float(reported_upper_bound)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Evaluation report has an invalid success_rate_upper_bound") from exc
        if not math.isfinite(upper_bound) or not math.isclose(
            upper_bound, expected_upper_bound, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("Evaluation report success_rate_upper_bound is inconsistent")
    return completed, successes, success_rate


def evaluation_meets_gate(report: dict[str, Any], threshold: float) -> bool:
    """Only a complete evaluation can satisfy a curriculum promotion gate."""
    try:
        requested = int(report.get("episodes_requested", 0))
        completed = int(report.get("episodes_completed", 0))
        rate = float(report.get("success_rate", float("nan")))
    except (TypeError, ValueError, OverflowError):
        return False
    return requested > 0 and completed == requested and math.isfinite(rate) and rate >= threshold


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
    graceful_timeout: float = 900.0,
    terminate_timeout: float = 60.0,
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


def _shutdown_timeout(env: dict[str, str], key: str, default: float) -> float:
    """Read a positive finite shutdown timeout from the child environment."""
    raw = env.get(key, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a positive finite number, got {raw!r}")
    return value


def stream_command(command: list[str], env: dict[str, str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[Curriculum] $ {printable}", flush=True)
    print(f"[Curriculum] concise event log: {log_path} (routine progress output omitted)", flush=True)
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
    with log_path.open("a", encoding="utf-8", buffering=65536) as logfile:
        output_errors: list[BaseException] = []

        def forward_output() -> None:
            terminal_available = True
            capturing_traceback = False
            try:
                assert child.stdout is not None
                for line in child.stdout:
                    # Always drain the pipe so shutdown cannot deadlock, but write only
                    # high-value events to disk. Progress lines/tables remain terminal-only.
                    if "Traceback (most recent call last)" in line:
                        capturing_traceback = True
                    selected = line if capturing_traceback else filter_persistent_log_text(line)
                    if selected:
                        logfile.write(selected)
                        if should_flush_important_log(selected):
                            logfile.flush()
                    if capturing_traceback and is_exception_end(line):
                        capturing_traceback = False
                    if terminal_available:
                        try:
                            # The parent stdout is line-buffered when attached to a TTY;
                            # explicitly flush only event/error lines to avoid extra syscalls.
                            sys.stdout.write(line)
                            if (not getattr(sys.stdout, "isatty", lambda: False)()) or is_important_log_line(line):
                                sys.stdout.flush()
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
            graceful_timeout = _shutdown_timeout(env, "POKEMON_AI_SHUTDOWN_TIMEOUT", 900.0)
            terminate_timeout = _shutdown_timeout(env, "POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT", 60.0)
            print(
                "\n[Curriculum] Stop requested. main.py will reach its safe shutdown boundary "
                "and close its emulator workers; during training it finishes the current full "
                "PPO rollout/update and checkpoints before exiting "
                f"(graceful window: {graceful_timeout:.0f}s).",
                flush=True,
            )
            stop_managed_process(
                child, "active main.py", graceful_signal=signal.SIGINT,
                graceful_timeout=graceful_timeout, terminate_timeout=terminate_timeout,
            )
            raise KeyboardInterrupt
        except BaseException:
            # Reap the process tree even on unexpected parent-side failures (e.g. disk/log errors).
            graceful_timeout = _shutdown_timeout(env, "POKEMON_AI_SHUTDOWN_TIMEOUT", 900.0)
            terminate_timeout = _shutdown_timeout(env, "POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT", 60.0)
            stop_managed_process(
                child, "active main.py", graceful_signal=signal.SIGINT,
                graceful_timeout=graceful_timeout, terminate_timeout=terminate_timeout,
            )
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
            try:
                child.stdout.close()
            except OSError:
                pass
        if output_errors:
            print(f"[Curriculum Warning] Child output forwarding raised {type(output_errors[0]).__name__}: {output_errors[0]}", flush=True)

    if code != 0:
        raise RuntimeError(f"Child process exited with code {code}: {printable}")


def resolve_path(root: Path, configured: str | Path) -> Path:
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def resolve_canonical_initial_state(
    root: Path,
    cfg: dict[str, str],
    cli_override: str | Path | None = None,
) -> Path:
    """Resolve the permanent pre-starter Route 101 state, not the selected-stage state.

    POKEMON_AI_TRAINING_START_STATE is deliberately not read from the launching
    shell: it is a transient, stage-specific variable and may be left over from
    a previous stage/watch command. For backward compatibility, a legacy .env
    value is reused only when it is outside the completed-milestones directory.
    """
    configured = None
    if cli_override is not None:
        configured = str(cli_override).strip()
    if not configured:
        configured = os.environ.get("POKEMON_AI_CANONICAL_START_STATE", "").strip()
    if not configured:
        configured = cfg.get("POKEMON_AI_CANONICAL_START_STATE", "").strip()

    if not configured:
        legacy = cfg.get("POKEMON_AI_TRAINING_START_STATE", "").strip()
        if legacy:
            legacy_path = resolve_path(root, legacy)
            milestone_root = root / "runtime_data" / "states" / "milestones"
            if not _is_within(legacy_path, milestone_root):
                configured = legacy

    if not configured:
        configured = "runtime_data/states/training_start.state"

    resolved = resolve_path(root, configured)
    milestone_root = root / "runtime_data" / "states" / "milestones"
    if _is_within(resolved, milestone_root):
        raise ValueError(
            "The canonical starter-stage path resolves inside the completed-milestones directory: "
            f"{resolved}. Set POKEMON_AI_CANONICAL_START_STATE to the pre-starter Route 101 savestate."
        )
    return resolved


def validate_stage_start_state(
    root: Path,
    stage: str,
    start_state: Path,
    expected_rom_hash: str,
) -> None:
    """Validate the immediate prerequisite milestone before a child process starts."""
    if stage not in STAGES:
        raise ValueError(f"Unknown curriculum stage {stage!r}; choose one of: {', '.join(STAGES)}")
    if stage == STAGES[0]:
        milestone_root = root / "runtime_data" / "states" / "milestones"
        if _is_within(start_state, milestone_root):
            raise RuntimeError(
                f"Stage 'starter' cannot load a completed milestone as its initial savestate: {start_state}. "
                "Use the canonical pre-starter Route 101 state."
            )
        if not start_state.is_file():
            raise FileNotFoundError(f"Canonical starter start state is missing: {start_state}")
        return

    expected_state = stage_start_state_path(root, stage)
    prior_stage = STAGES[STAGES.index(stage) - 1]
    if start_state.resolve() != expected_state:
        raise RuntimeError(
            f"Stage {stage} must start from the immediately preceding {prior_stage} milestone "
            f"({expected_state}), not {start_state}."
        )
    manifest_path = Path(str(expected_state) + ".json")
    if not expected_state.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Stage {stage} requires the preceding {prior_stage} savestate and metadata: {expected_state}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read prerequisite milestone metadata {manifest_path}: {exc}") from exc

    manifest_plan = manifest.get("curriculum_plan")
    # Older manifests may be reused only when their stage semantics are known to
    # be compatible. Every accepted legacy state is then re-evaluated using the
    # current corrected objective predicate below; the version label alone never
    # grants promotion or stage completion.
    compatible_legacy_manifest = (
        manifest_plan == "emerald_intro_v5"
        or (manifest_plan == "emerald_intro_v4" and prior_stage == "starter")
    )
    if (
        manifest_plan is not None
        and manifest_plan != CURRICULUM_PLAN_ID
        and not compatible_legacy_manifest
    ):
        raise RuntimeError(
            f"Prerequisite metadata {manifest_path} belongs to curriculum plan {manifest_plan!r}, "
            f"not a compatible {CURRICULUM_PLAN_ID!r} predecessor. Recreate the preceding milestone."
        )
    if manifest.get("objective") != prior_stage:
        raise RuntimeError(
            f"Prerequisite metadata {manifest_path} belongs to {manifest.get('objective')!r}, "
            f"not the required prior objective {prior_stage!r}."
        )
    if str(manifest.get("rom_sha1_expected", "")).strip().lower() != expected_rom_hash:
        raise RuntimeError(
            f"Prerequisite milestone {manifest_path} was created for a different ROM identity; "
            "refusing to start this stage."
        )
    recorded_state_path = manifest.get("savestate_path")
    if not isinstance(recorded_state_path, str) or not recorded_state_path.strip():
        raise RuntimeError(
            f"Prerequisite metadata {manifest_path} is missing its savestate_path; "
            "refusing to trust a detached or incomplete milestone manifest."
        )
    try:
        recorded_path = Path(recorded_state_path).expanduser()
        if not recorded_path.is_absolute():
            recorded_path = root / recorded_path
        if recorded_path.resolve() != expected_state:
            raise RuntimeError(
                f"Prerequisite metadata {manifest_path} points to a different savestate: "
                f"{recorded_state_path!r}."
            )
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Invalid savestate path in {manifest_path}: {exc}") from exc

    state_metrics = manifest.get("state")
    if not isinstance(state_metrics, dict):
        raise RuntimeError(f"Prerequisite metadata {manifest_path} has no decoded state object.")
    if state_metrics.get("saveblock_pointer_stable", True) is False:
        raise RuntimeError(
            f"Prerequisite metadata {manifest_path} records an unstable SaveBlock1 pointer; "
            "recreate the milestone from a verified game state."
        )
    if (
        stage in OBJECTIVES_REQUIRING_FLAG_TELEMETRY
        and state_metrics.get("flag_telemetry_valid", True) is False
    ):
        raise RuntimeError(
            f"Prerequisite metadata {manifest_path} records unstable event/badge flags required by {stage}; "
            "recreate the milestone from a verified game state."
        )
    if not objective_satisfied(prior_stage, state_metrics):
        raise RuntimeError(
            f"The {prior_stage} prerequisite manifest does not satisfy its own curriculum objective; "
            "the saved milestone may be stale, incomplete, or based on incorrect telemetry. "
            "Recreate/verify the preceding milestone before continuing."
        )
    prerequisite_error = objective_start_error(stage, state_metrics)
    if prerequisite_error:
        raise RuntimeError(
            f"The {prior_stage} milestone metadata is not a valid start for {stage}: "
            f"{prerequisite_error}. Recreate/verify the preceding milestone before continuing."
        )


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

    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        # The TensorBoard server can emit verbose request/startup logs. Avoid an
        # ever-growing server log file; its health is checked here and status/errors
        # are reported by the curriculum manager.
        process = subprocess.Popen(
            [
                str(python), "-m", "tensorboard.main",
                f"--logdir={tensorboard_dir}", f"--port={actual_port}", "--host=127.0.0.1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Register ownership immediately, before any startup check can be interrupted.
        _CURRICULUM_TENSORBOARD_PROCESS = process
        _CURRICULUM_TENSORBOARD_PORT = actual_port
        time.sleep(0.6)
        if process.poll() is not None:
            print("[TensorBoard Warning] Server exited during startup; check TensorBoard installation, log directory, and port. Metrics will still be written.", flush=True)
            stop_managed_process(
                process, "failed TensorBoard startup", graceful_signal=signal.SIGTERM,
                graceful_timeout=0.0, terminate_timeout=0.5,
            )
            _CURRICULUM_TENSORBOARD_PROCESS = None
            _CURRICULUM_TENSORBOARD_PORT = None
        else:
            print(f"[TensorBoard] http://127.0.0.1:{actual_port} | logdir={tensorboard_dir}", flush=True)
            print("[TensorBoard] Server output is suppressed; this curriculum stops its owned server on exit.", flush=True)
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


def archive_mastered_checkpoint(root: Path, model_name: str, stage: str) -> Path:
    """Atomically save a ZIP-validated, stage-named snapshot of the mastered model."""
    source = root / f"{model_name}.zip"
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise RuntimeError(f"Cannot archive mastered stage {stage}: model checkpoint is missing or invalid: {source}")
    with zipfile.ZipFile(source, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Cannot archive mastered stage {stage}: corrupt model member {bad_member!r}")
    destination_dir = root / "runtime_data" / "checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{model_name}_{stage}_mastered.zip"
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        if file_sha1(source) != file_sha1(temporary) or not zipfile.is_zipfile(temporary):
            raise RuntimeError(f"Mastered checkpoint copy failed ZIP/hash validation: {temporary}")
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"Mastered checkpoint copy is corrupt: {temporary}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_dir(destination_dir)
    finally:
        temporary.unlink(missing_ok=True)
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
                        help="override the saved/current objective stage")
    parser.add_argument("--initial-state", type=Path, default=None,
                        help="explicit canonical pre-starter Route 101 savestate (used only for the starter stage)")
    parser.add_argument("--stage-only", action="store_true",
                        help="train only the selected stage for the configured chunk budget; skip all evaluation, archive comparison, and promotion")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without launching emulators")
    args = parser.parse_args()
    root = args.project_dir.expanduser().resolve()
    if not (root / "main.py").is_file() or not (root / "PokemonEmerald.gba").is_file():
        raise SystemExit(f"Project or ROM not found under {root}; run from the correct project directory.")
    cfg = parse_dotenv(root / ".env")
    report_shell_config_overrides(cfg)
    stage_only = stage_only_requested(args.stage_only, cfg)
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

    initial_state = resolve_canonical_initial_state(root, cfg, args.initial_state)
    inherited_stage_state = os.environ.get("POKEMON_AI_TRAINING_START_STATE", "").strip()
    if inherited_stage_state:
        inherited_path = resolve_path(root, inherited_stage_state)
        if inherited_path != initial_state:
            print(
                "[Config Warning] Ignoring inherited POKEMON_AI_TRAINING_START_STATE for curriculum "
                f"stage mapping ({inherited_path}); canonical starter state is {initial_state}. "
                "Use --initial-state or POKEMON_AI_CANONICAL_START_STATE for an intentional "
                "starter-state override.",
                flush=True,
            )
    milestone_dir = root / "runtime_data" / "states" / "milestones"
    log_dir = root / "logs"
    run_state_path = root / "runtime_data" / "curriculum_state.json"
    run_log_path = log_dir / f"curriculum_manager_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"

    def append_curriculum_event(message: str) -> None:
        """Persist stage/evaluation outcomes, not routine progress output."""
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with run_log_path.open("a", encoding="utf-8", buffering=65536) as stream:
            stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message.rstrip()}\n")
            stream.flush()  # stage transitions/evaluation results are intentionally rare

    chunk_steps = int_setting(cfg, "POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS", 250_000)
    max_chunks = int_setting(cfg, "POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE", 8, minimum=0)
    eval_episodes = int_setting(cfg, "POKEMON_AI_CURRICULUM_EVAL_EPISODES", 5)
    ppo_n_steps = int_setting(cfg, "POKEMON_AI_PPO_N_STEPS", 2048)
    num_envs = int_setting(cfg, "POKEMON_AI_NUM_ENVS", 4)
    train_episode_override_raw = os.environ.get(
        "POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS",
        cfg.get("POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS", ""),
    ).strip()
    train_episode_override = int_setting(
        {"POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS": train_episode_override_raw},
        "POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS", 1,
    ) if train_episode_override_raw else None
    eval_step_override_raw = os.environ.get(
        "POKEMON_AI_CURRICULUM_EVAL_MAX_EPISODE_STEPS",
        cfg.get("POKEMON_AI_CURRICULUM_EVAL_MAX_EPISODE_STEPS", ""),
    ).strip()
    eval_step_override = int_setting(
        {"POKEMON_AI_CURRICULUM_EVAL_MAX_EPISODE_STEPS": eval_step_override_raw},
        "POKEMON_AI_CURRICULUM_EVAL_MAX_EPISODE_STEPS", 1,
    ) if eval_step_override_raw else None
    eval_no_progress_steps = int_setting(cfg, "POKEMON_AI_EVAL_NO_PROGRESS_STEPS", 4096, minimum=128)
    threshold = float_setting(cfg, "POKEMON_AI_CURRICULUM_SUCCESS_RATE", 0.9)
    deterministic = bool_setting(cfg, "POKEMON_AI_CURRICULUM_EVAL_DETERMINISTIC", True)
    compare_checkpoints = int_setting(cfg, "POKEMON_AI_CURRICULUM_COMPARE_CHECKPOINTS", 2, minimum=0)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise SystemExit("POKEMON_AI_CURRICULUM_SUCCESS_RATE must be greater than 0 and at most 1.")
    if eval_episodes < 1:
        raise SystemExit("POKEMON_AI_CURRICULUM_EVAL_EPISODES must be at least 1.")

    current_stage = args.start_stage
    resume_resolution: str | None = None
    configured_objective = os.environ.get("POKEMON_AI_OBJECTIVE", cfg.get("POKEMON_AI_OBJECTIVE", "starter")).strip().lower()
    if stage_only and current_stage is None:
        # In stage-only mode the explicitly configured objective must win over any
        # prior curriculum-state file; the user is asking for that one stage only.
        if configured_objective not in STAGES:
            raise SystemExit(
                f"POKEMON_AI_OBJECTIVE={configured_objective!r} is not a known stage. "
                f"Choose one of: {', '.join(STAGES)}"
            )
        current_stage = configured_objective
    if current_stage is None and run_state_path.is_file():
        try:
            saved_state = json.loads(run_state_path.read_text(encoding="utf-8"))
            resolved_stage, resolution = resolve_saved_curriculum_stage(saved_state)
            resume_resolution = resolution
            if resolved_stage is not None:
                current_stage = resolved_stage
                if resolution == "migrated_v4_early_stage":
                    print(
                        f"[Curriculum] Migrated saved v4 stage {current_stage!r} to v5. "
                        "The completed starter milestone is preserved; `leave_birch_lab` now "
                        "ends at Littleroot Town, and Route 101 is a separate next stage.",
                        flush=True,
                    )
                elif resolution == "migrated_v5_revalidated":
                    print(
                        f"[Curriculum] Migrating saved v5 stage {current_stage!r} to v6. "
                        "Its preceding milestone will be checked against corrected event flags before use.",
                        flush=True,
                    )
                elif resolution == "after_mastered":
                    print(f"[Curriculum] Previous stage was mastered; resuming at its successor {current_stage!r}.", flush=True)
                elif stage_only:
                    print(f"[Curriculum] Stage-only mode selected saved stage {current_stage!r}; evaluations/promotions are disabled.", flush=True)
                else:
                    print(f"[Curriculum] Resuming from saved stage {current_stage!r} (it will be re-evaluated first).", flush=True)
            elif resolution == "incompatible_plan":
                print(
                    "[Curriculum] Previous stage plan is incompatible with this objective sequence; "
                    "restarting stage selection at 'starter'. The existing model checkpoint is retained.",
                    flush=True,
                )
        except (OSError, json.JSONDecodeError, AttributeError):
            print("[Curriculum Warning] Prior curriculum state is unreadable; starting from the first stage.", flush=True)
    if current_stage is None and resume_resolution == "complete" and args.start_stage is None:
        print("[Curriculum] All 58 stages are already marked complete. Use --start-stage to intentionally replay a stage.", flush=True)
        return 0
    if current_stage is None:
        if configured_objective not in STAGES:
            raise SystemExit(
                f"POKEMON_AI_OBJECTIVE={configured_objective!r} is not a known stage. "
                f"Choose one of: {', '.join(STAGES)}"
            )
        current_stage = configured_objective
        print(f"[Curriculum] No resumable state found; using configured POKEMON_AI_OBJECTIVE={current_stage!r}.", flush=True)
    start_index = STAGES.index(current_stage)

    checkpoint = root / f"{model_name}.zip"
    if not checkpoint.is_file() and current_stage != "starter":
        raise SystemExit(
            f"No model checkpoint {checkpoint}; curriculum must start with 'starter' when training a new model."
        )
    current_stage_start_state = stage_start_state_path(root, current_stage, initial_state)
    if not args.dry_run:
        try:
            validate_stage_start_state(root, current_stage, current_stage_start_state, expected_rom_hash)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Cannot start at {current_stage}: {exc}") from exc
    elif current_stage != "starter":
        # Dry-run still refuses absent or mismatched prerequisite metadata; it does
        # not need to open the emulator or inspect the raw savestate bytes.
        try:
            validate_stage_start_state(root, current_stage, current_stage_start_state, expected_rom_hash)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Cannot plan {current_stage}: {exc}") from exc

    print("=" * 72, flush=True)
    print(" Pokémon Emerald curriculum controller", flush=True)
    print(f" ROM expected SHA-1 : {expected_rom_hash}", flush=True)
    print(f" Model checkpoint   : {checkpoint}", flush=True)
    print(f" Canonical starter  : {initial_state}", flush=True)
    print(f" Selected stage     : {current_stage}", flush=True)
    if stage_only:
        print(" Mode                : STAGE-ONLY (evaluation, archive comparison, and promotion disabled)", flush=True)
    else:
        print(f" Evaluation gate    : {'deterministic' if deterministic else 'stochastic'}; "
              f"{eval_episodes} episodes; success rate >= {threshold:.0%}", flush=True)
    print(f" Train chunk        : base {chunk_steps:,} transitions; maximum {'unlimited' if max_chunks == 0 else max_chunks} chunks per stage", flush=True)
    print(" Training episode   : stage-specific action cap (or explicit POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS override)", flush=True)
    print(f" Evaluation guard   : stage-specific action cap; stop after {eval_no_progress_steps:,} actions without meaningful game progress", flush=True)
    if not stage_only:
        print(f" Checkpoint recovery: compare up to {compare_checkpoints} recent distinct archived models when the current model misses the gate", flush=True)
    print(" Milestones         : " + " -> ".join(STAGES), flush=True)
    print("=" * 72, flush=True)

    if args.dry_run:
        print("\n[Dry-run] No emulator will be started and no files/checkpoints will be changed.", flush=True)
        if stage_only:
            stage = current_stage
            stage_start = stage_start_state_path(root, stage, initial_state)
            stage_chunk_steps = align_training_chunk_steps(chunk_steps, ppo_n_steps, num_envs)
            stage_train_limit = stage_training_episode_step_limit(stage, ppo_n_steps, train_episode_override)
            print(f"\n[Dry-run] STAGE-ONLY: {stage} — {OBJECTIVE_LABELS[stage]}", flush=True)
            print(f"  Start state       : {stage_start}", flush=True)
            total = "unlimited" if max_chunks == 0 else f"{max_chunks * stage_chunk_steps:,}"
            print(f"  Training episode  : max {stage_train_limit:,} decisions per environment", flush=True)
            print(f"  Training chunks   : {'unlimited' if max_chunks == 0 else max_chunks} × {stage_chunk_steps:,} = {total} transitions", flush=True)
            print("  Evaluation        : DISABLED", flush=True)
            print("  Checkpoint compare: DISABLED", flush=True)
            print("  Promotion         : DISABLED; controller exits after this stage", flush=True)
            print("\n[Dry-run] Stage-only plan preview complete (exit 0).", flush=True)
            return 0
        for stage in STAGES[start_index:]:
            stage_chunk_steps = stage_training_chunk_steps(
                stage, chunk_steps, ppo_n_steps, num_envs, train_episode_override
            )
            stage_train_limit = stage_training_episode_step_limit(stage, ppo_n_steps, train_episode_override)
            stage_start = stage_start_state_path(root, stage, initial_state)
            print(f"\n[Dry-run] Stage {stage}: start state={stage_start}", flush=True)
            if checkpoint.is_file():
                stage_eval_limit = stage_evaluation_step_limit(stage, ppo_n_steps, eval_step_override)
                print(f"  1. Evaluate current checkpoint ({eval_episodes} episodes; max {stage_eval_limit:,} actions each): {checkpoint}", flush=True)
                if compare_checkpoints:
                    print(f"  2. If below gate, screen up to {compare_checkpoints} recent distinct archives (1 episode each) and fully re-evaluate the best candidate.", flush=True)
            elif stage == "starter":
                print("  1. No checkpoint yet; start a fresh model from the canonical pre-starter state.", flush=True)
            else:
                print("  1. No checkpoint yet; the real run must begin at starter before it can promote to this stage.", flush=True)
            print(f"  3. If not mastered, train with a {stage_train_limit:,}-decision per-env episode cap; use {'unlimited' if max_chunks == 0 else f'up to {max_chunks}'} chunk(s) of {stage_chunk_steps:,} transitions each; evaluate after each chunk.", flush=True)
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
    base_env["POKEMON_AI_CANONICAL_START_STATE"] = str(initial_state)
    base_env["POKEMON_AI_CURRICULUM_STAGE_ONLY"] = "1" if stage_only else "0"
    base_env["POKEMON_AI_SHUTDOWN_TIMEOUT"] = os.environ.get(
        "POKEMON_AI_SHUTDOWN_TIMEOUT", cfg.get("POKEMON_AI_SHUTDOWN_TIMEOUT", "900")
    )
    base_env["POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT"] = os.environ.get(
        "POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT", cfg.get("POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT", "60")
    )

    def eval_stage(
        stage: str,
        start_state: Path,
        checkpoint_path: Path | None = None,
        episode_count: int | None = None,
    ) -> dict[str, Any] | None:
        requested_episodes = int(episode_count if episode_count is not None else eval_episodes)
        report_before = set(log_dir.glob("evaluation_*.json")) if log_dir.is_dir() else set()
        evaluation_started_unix = time.time()
        stage_eval_limit = stage_evaluation_step_limit(stage, ppo_n_steps, eval_step_override)
        env = build_evaluation_environment(
            base_env,
            stage=stage,
            start_state=start_state,
            requested_episodes=requested_episodes,
            deterministic=deterministic,
            eval_seed=os.environ.get("POKEMON_AI_EVAL_SEED", cfg.get("POKEMON_AI_EVAL_SEED", "1234")),
            max_episode_steps=stage_eval_limit,
            no_progress_steps=eval_no_progress_steps,
            checkpoint_path=checkpoint_path,
            stop_if_gate_unreachable=True,
            success_threshold=threshold,
        )
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
        try:
            completed_episodes, successes, success_rate = validate_evaluation_report_progress(
                report, requested_episodes, threshold
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Invalid evaluation episode results in {report_path}: {exc}") from exc
        if str(report.get("rom_sha1_expected", "")).strip().lower() != expected_rom_hash:
            raise RuntimeError(f"Evaluation report ROM identity does not match the configured ROM: {report_path}")
        reported_start_state = report.get("start_state")
        if not isinstance(reported_start_state, str) or Path(reported_start_state).resolve() != start_state.resolve():
            raise RuntimeError(f"Evaluation report start state does not match the requested milestone: {report_path}")
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
        evaluation_summary = (
            f"[Curriculum Evaluation] stage={stage} successes={successes}/{completed_episodes} "
            f"(requested={requested_episodes}) observed_rate={success_rate:.1%} "
            f"max_possible_rate={float(report.get('success_rate_upper_bound', success_rate)):.1%} "
            f"mean_steps={float(report.get('mean_steps', 0.0)):.0f} report={report_path}"
        )
        print(evaluation_summary, flush=True)
        append_curriculum_event(evaluation_summary)
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
        if (
            manifest.get("objective") != stage
            or manifest.get("curriculum_plan") != CURRICULUM_PLAN_ID
            or manifest.get("rom_sha1_expected") != expected_rom_hash
            or not milestone_manifest_satisfies_objective(stage, manifest)
        ):
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
        baseline_success_rate = baseline_score[0]
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
            print(
                f"[Checkpoint Compare] {candidate.name} score={score} baseline={baseline_score} "
                f"(1-episode screen)", flush=True
            )
            # A candidate that does not complete the selected objective in its
            # screening run cannot advance to full evaluation based on wandering,
            # stale flags, or a larger exploration counter.
            if score[0] <= baseline_success_rate or score[0] <= 0.0:
                continue
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
        if full_report is None:
            print("[Checkpoint Compare] Candidate full evaluation is missing; keep current model.", flush=True)
            return baseline_report
        full_score = report_score(full_report)
        if full_score[0] <= baseline_success_rate:
            print(
                "[Checkpoint Compare] Candidate did not improve objective success rate on the full test; "
                "secondary counters cannot override that result. Keep current model.",
                flush=True,
            )
            return baseline_report

        backup = activate_checkpoint_candidate(
            best_screen_path, checkpoint, root / "runtime_data" / "checkpoints"
        )
        print(
            f"[Checkpoint Compare] Activated improved checkpoint {best_screen_path.name}; "
            f"previous model backup={backup if backup else 'not needed'}; full score={full_score}",
            flush=True,
        )
        full_report["selected_checkpoint_activated"] = True
        full_report["selected_checkpoint_source"] = str(best_screen_path.resolve())
        full_report["baseline_score_before_candidate"] = list(baseline_score)
        return full_report

    try:
        if stage_only:
            stage = current_stage
            stage_start_state = stage_start_state_path(root, stage, initial_state)
            validate_stage_start_state(root, stage, stage_start_state, expected_rom_hash)

            stage_chunk_steps = align_training_chunk_steps(chunk_steps, ppo_n_steps, num_envs)
            stage_train_limit = stage_training_episode_step_limit(stage, ppo_n_steps, train_episode_override)
            stage_message = (
                f"[Stage-only] TRAINING {stage.upper()}: {OBJECTIVE_LABELS[stage]} | "
                f"start={stage_start_state} | episode_cap={stage_train_limit:,} decisions/env | "
                f"evaluation/promotions disabled"
            )
            print(f"\n{stage_message}", flush=True)
            append_curriculum_event(stage_message)
            chunk_index = 0
            while max_chunks == 0 or chunk_index < max_chunks:
                chunk_index += 1
                chunk_message = (
                    f"[Stage-only] stage={stage} chunk={chunk_index}/{'unlimited' if max_chunks == 0 else max_chunks} "
                    f"transitions={stage_chunk_steps:,}; no evaluation or promotion"
                )
                print(f"\n{chunk_message}", flush=True)
                append_curriculum_event(chunk_message)
                stage_train_limit = stage_training_episode_step_limit(
                    stage, ppo_n_steps, train_episode_override
                )
                train_env = build_training_environment(
                    base_env,
                    stage=stage,
                    start_state=stage_start_state,
                    total_timesteps=stage_chunk_steps,
                    max_episode_steps=stage_train_limit,
                    stage_only=False,
                    reset_entropy_for_stage=(chunk_index == 1),
                )
                # Do not create an unverified stage milestone during un-gated training.
                # A normal curriculum evaluation can later verify and publish it.
                train_env["POKEMON_AI_MILESTONE_SAVE"] = "0"
                stream_command([str(python), "main.py"], train_env, root, run_log_path, args.dry_run)
                atomic_json(run_state_path, {
                    "schema_version": 2,
                    "curriculum_plan": CURRICULUM_PLAN_ID,
                    "status": "stage_only_training",
                    "stage": stage,
                    "chunk_index": chunk_index,
                    "chunks_total": max_chunks,
                    "chunk_timesteps": stage_chunk_steps,
                    "evaluation_disabled": True,
                    "promotion_disabled": True,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })

            atomic_json(run_state_path, {
                "schema_version": 2,
                "curriculum_plan": CURRICULUM_PLAN_ID,
                "status": "stage_only_budget_complete",
                "stage": stage,
                "chunks_trained_this_invocation": chunk_index,
                "chunk_timesteps": stage_chunk_steps,
                "evaluation_disabled": True,
                "promotion_disabled": True,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            final_message = (
                f"[Stage-only] BUDGET COMPLETE: trained {stage} for "
                f"{('unlimited' if max_chunks == 0 else f'{chunk_index * stage_chunk_steps:,}')} transitions. "
                "No evaluation was run and no promotion occurred."
            )
            print(f"\n{final_message}", flush=True)
            append_curriculum_event(final_message)
            return 0

        for stage_index in range(start_index, len(STAGES)):
            stage = STAGES[stage_index]
            stage_chunk_steps = stage_training_chunk_steps(
                stage, chunk_steps, ppo_n_steps, num_envs, train_episode_override
            )
            stage_train_limit = stage_training_episode_step_limit(stage, ppo_n_steps, train_episode_override)
            stage_start_state = stage_start_state_path(root, stage, initial_state)
            validate_stage_start_state(root, stage, stage_start_state, expected_rom_hash)

            stage_message = f"[Curriculum] STAGE {stage.upper()}: {OBJECTIVE_LABELS[stage]} | start={stage_start_state}"
            print(f"\n{stage_message}", flush=True)
            if not args.dry_run:
                append_curriculum_event(stage_message)
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
                if report is not None and evaluation_meets_gate(report, threshold):
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
                    if evaluation_meets_gate(report, threshold) and verified_fresh_milestone(stage, report):
                        mastered = True

            while not mastered and (max_chunks == 0 or chunks_trained < max_chunks):
                chunks_trained += 1
                chunk_message = (
                    f"[Curriculum] Training stage={stage} chunk={chunks_trained}/{'unlimited' if max_chunks == 0 else max_chunks} "
                    f"additional_transitions={stage_chunk_steps:,}."
                )
                print(f"\n{chunk_message}", flush=True)
                append_curriculum_event(chunk_message)
                train_env = build_training_environment(
                    base_env,
                    stage=stage,
                    start_state=stage_start_state,
                    total_timesteps=stage_chunk_steps,
                    max_episode_steps=stage_train_limit,
                    stage_only=False,
                    reset_entropy_for_stage=(chunks_trained == 1),
                )
                stream_command([str(python), "main.py"], train_env, root, run_log_path, args.dry_run)
                if args.dry_run:
                    continue
                report = eval_stage(stage, stage_start_state)
                if report is None:
                    continue
                if evaluation_meets_gate(report, threshold):
                    mastered = verified_fresh_milestone(stage, report)
                    if not mastered:
                        print(
                            f"[Curriculum] Gate met, but milestone state/manifest failed validation. "
                            "No promotion.", flush=True
                        )
                if not mastered:
                    report = compare_archived_checkpoints(stage, stage_start_state, report)
                    if evaluation_meets_gate(report, threshold):
                        mastered = verified_fresh_milestone(stage, report)

                if not args.dry_run:
                    atomic_json(run_state_path, {
                        "schema_version": 2, "curriculum_plan": CURRICULUM_PLAN_ID,
                        "status": "running" if mastered else "stage_not_mastered",
                        "stage": stage,
                        "chunks_trained_this_invocation": chunks_trained,
                        "chunk_timesteps": stage_chunk_steps,
                        "success_threshold": threshold,
                        "last_evaluation_report": str(newest_report(log_dir, set()) or ""),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })

            if not mastered:
                if not args.dry_run:
                    atomic_json(run_state_path, {
                        "schema_version": 2, "curriculum_plan": CURRICULUM_PLAN_ID, "status": "stage_not_mastered", "stage": stage,
                        "chunks_trained_this_invocation": chunks_trained, "chunk_timesteps": stage_chunk_steps,
                        "success_threshold": threshold, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })
                not_mastered_message = (
                    f"[Curriculum] NOT PROMOTED: {stage} did not meet the evaluation gate after "
                    f"{chunks_trained} training chunks. Checkpoint preserved; rerun to continue."
                )
                print(not_mastered_message, flush=True)
                append_curriculum_event(not_mastered_message)
                return 2

            mastered_archive = archive_mastered_checkpoint(root, model_name, stage)
            mastered_message = (
                f"[Curriculum] MASTERED: {stage} ({OBJECTIVE_LABELS[stage]}); "
                f"milestone verified; snapshot={mastered_archive.name}."
            )
            print(mastered_message, flush=True)
            if not args.dry_run:
                append_curriculum_event(mastered_message)
            if not args.dry_run:
                atomic_json(run_state_path, {
                    "schema_version": 2, "curriculum_plan": CURRICULUM_PLAN_ID, "status": "mastered", "stage": stage,
                    "chunks_trained_this_invocation": chunks_trained, "chunk_timesteps": stage_chunk_steps,
                    "success_threshold": threshold, "milestone_state": str(milestone_dir / f"{stage}.state"),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })

        if not args.dry_run:
            atomic_json(run_state_path, {
                "schema_version": 2, "curriculum_plan": CURRICULUM_PLAN_ID, "status": "complete", "stage": STAGES[-1],
                "milestones": [str(milestone_dir / f"{stage}.state") for stage in STAGES],
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        complete_message = "[Curriculum] COMPLETE: all configured stages passed the evaluation gate."
        print(f"\n{complete_message}", flush=True)
        if not args.dry_run:
            append_curriculum_event(complete_message)
        return 0
    except KeyboardInterrupt:
        if not args.dry_run:
            atomic_json(run_state_path, {
                "schema_version": 2, "curriculum_plan": CURRICULUM_PLAN_ID, "status": "interrupted", "stage": stage if 'stage' in locals() else current_stage,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        interrupted_message = (
            "[Stage-only] Interrupted. Model/checkpoints are retained; no evaluation or promotion was performed."
            if stage_only else
            "[Curriculum] Interrupted. Model/state checkpoints are retained; rerun to re-evaluate and resume."
        )
        print(interrupted_message, flush=True)
        if not args.dry_run:
            append_curriculum_event(interrupted_message)
        return 130
    except Exception as exc:
        if not args.dry_run:
            try:
                append_curriculum_event(f"[Curriculum Error] {type(exc).__name__}: {exc}")
            except Exception:
                pass
            try:
                atomic_json(run_state_path, {
                    "schema_version": 2, "curriculum_plan": CURRICULUM_PLAN_ID, "status": "error", "stage": stage if 'stage' in locals() else current_stage,
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
