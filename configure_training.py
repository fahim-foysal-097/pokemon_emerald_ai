#!/usr/bin/env python3
"""Configure Pokémon Emerald RL .env for one selected objective or a curriculum.

This file is a standalone configuration helper, not an updater or installer.
It edits only <project>/.env, keeps unrelated settings, and relies on the
shared curriculum_objectives.py for the project's actual ordered stage names.

The stage-specific episode limits below are conservative starting heuristics,
not measured guarantees. An episode step is one policy decision, not one GBA
frame. At the current measured recurrent-policy speed of roughly 35 aggregate transitions/s, each
environment averages about 8.75 policy decisions/s when four workers are balanced.

Examples:
    # One hour of training for leaving Birch's Lab; preview only
    .venv/bin/python configure_training.py --project-dir /path/to/project \
        --hours 1 --speed 35 --mode stage --stage leave_birch_lab --dry-run

    # Configure and write a 3-hour single-stage Pokédex run
    .venv/bin/python configure_training.py --project-dir /path/to/project \
        --hours 3 --speed 35 --mode stage --stage pokedex --yes

    # Train exactly one selected curriculum stage with no evaluation or promotion
    .venv/bin/python configure_training.py --project-dir /path/to/project \
        --hours 3 --speed 35 --mode stage-only --stage leave_birch_lab --dry-run

    # Start the automated curriculum at Route 103; note it shares one episode
    # timeout across the remaining stages in the current controller version
    .venv/bin/python configure_training.py --project-dir /path/to/project \
        --hours 3 --speed 35 --mode curriculum --stage reach_route103 --dry-run
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from curriculum_objectives import (
    DEFAULT_STAGE_EPISODE_ROLLOUTS,
    STAGE_EPISODE_ROLLOUTS,
    stage_episode_rollouts,
)

DEFAULT_SPEED = 100.0
DEFAULT_NUM_ENVS = 4
DEFAULT_N_STEPS = 2048
DEFAULT_BATCH_SIZE = 256
DEFAULT_EVAL_RESERVE = 0.20
DEFAULT_MAX_CHUNKS_PER_STAGE = 8
DEFAULT_CHECKPOINT_INTERVAL_TRANSITIONS = 122_880  # preserved project default; 15 × (4 envs × 2048 steps)
PROGRESS_ROLLOUTS = 1
TERMINAL_PROGRESS_ROLLOUTS = 2

KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*transitions/s\s+total", re.I)


@dataclass(frozen=True)
class StageInfo:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class Plan:
    mode: str
    profile: str
    stage: str
    stage_label: str
    start_state: str
    canonical_start_state: str
    checkpoint_every: int
    requested_hours: float
    speed: float
    envs: int
    n_steps: int
    batch_size: int
    rollout_size: int
    total_transitions: int
    estimated_hours: float
    max_episode_steps: int
    estimated_episode_minutes: float
    remaining_stages: int
    chunk_transitions: int | None
    chunks_per_stage: int | None
    curriculum_transitions: int | None
    curriculum_estimated_hours: float | None
    reserved_hours: float | None
    warnings: tuple[str, ...]


def parse_dotenv(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse first occurrence of each key, matching this project's .env behavior."""
    if not path.is_file():
        return {}, []
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = KEY_RE.match(line)
        if not match:
            continue
        key = match.group(1)
        if key in values:
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values, lines


def positive_float(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and greater than zero")
    return value


def positive_int(raw: str, label: str, *, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def current_value(values: dict[str, str], key: str, default: str) -> str:
    # Shell variables override .env when main.py is launched; preserve that
    # established precedence when determining the current defaults.
    return os.environ.get(key, values.get(key, default)).strip()


def detect_recent_speed(project_dir: Path) -> float | None:
    """Return the last aggregate speed found near the end of a recent run log."""
    log_dir = project_dir / "logs"
    if not log_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in log_dir.rglob("*.log") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:30]
    for path in candidates:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - 4 * 1024 * 1024))
                text = stream.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        matches = SPEED_RE.findall(text)
        if matches:
            try:
                speed = float(matches[-1])
            except ValueError:
                continue
            if math.isfinite(speed) and speed > 0:
                return speed
    return None


def load_curriculum_module(project_dir: Path):
    """Load the shared objective module without importing the emulator runtime."""
    module_path = project_dir / "curriculum_objectives.py"
    if not module_path.is_file():
        raise ValueError(
            f"{module_path} was not found. This configurator expects the incremental-curriculum codebase."
        )
    spec = importlib.util.spec_from_file_location("_configure_training_curriculum_objectives", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load curriculum stage definitions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stages(project_dir: Path) -> list[StageInfo]:
    """Read the project's shared curriculum list/labels without importing main.py."""
    module = load_curriculum_module(project_dir)
    module_path = project_dir / "curriculum_objectives.py"
    raw_stages = getattr(module, "STAGES", None)
    if not isinstance(raw_stages, (tuple, list)) or not raw_stages:
        raise ValueError(f"{module_path} does not define a non-empty STAGES list")
    definitions = getattr(module, "OBJECTIVE_DEFINITIONS", {})
    result: list[StageInfo] = []
    for raw_key in raw_stages:
        key = str(raw_key)
        definition = definitions.get(key, {}) if isinstance(definitions, dict) else {}
        result.append(StageInfo(
            key=key,
            label=str(definition.get("label", key.replace("_", " ").title())),
            description=str(definition.get("description", "")),
        ))
    if len({stage.key for stage in result}) != len(result):
        raise ValueError("curriculum_objectives.py contains duplicate stage names")
    return result


def valid_batch_size(requested: int, rollout_size: int) -> int:
    """Choose largest batch <= requested which divides the total rollout exactly."""
    for value in range(min(requested, rollout_size), 0, -1):
        if rollout_size % value == 0:
            return value
    return 1


def stage_start_state(
    project_dir: Path,
    stages: list[StageInfo],
    stage: str,
    canonical_start_state: str | Path = "runtime_data/states/training_start.state",
) -> Path:
    """Use the same stage-to-savestate mapping as run_curriculum.py."""
    keys = [item.key for item in stages]
    if stage not in keys:
        raise ValueError(f"Unknown stage {stage!r}; choose one of: {', '.join(keys)}")
    module = load_curriculum_module(project_dir)
    shared_keys = list(getattr(module, "STAGES", ()))
    if keys != shared_keys:
        raise ValueError(
            "Configurator stage definitions do not match curriculum_objectives.py; "
            "refusing to calculate a possibly incorrect savestate path."
        )
    return module.stage_start_state_path(project_dir, stage, canonical_start_state)


def align_down(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def build_plan(
    *,
    mode: str,
    profile: str,
    stage: str,
    stages: list[StageInfo],
    project_dir: Path,
    hours: float,
    speed: float,
    envs: int,
    n_steps: int,
    batch_size: int,
    eval_reserve: float = DEFAULT_EVAL_RESERVE,
    max_chunks_cap: int | None = None,
    episode_steps_override: int | None = None,
    canonical_start_state: str | Path = "runtime_data/states/training_start.state",
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL_TRANSITIONS,
) -> Plan:
    hours = positive_float(str(hours), "hours")
    speed = positive_float(str(speed), "aggregate transitions/s")
    envs = positive_int(str(envs), "number of environments")
    n_steps = positive_int(str(n_steps), "PPO steps per environment")
    batch_size = positive_int(str(batch_size), "PPO batch size")
    if mode == "direct":  # retain compatibility with the previous CLI spelling
        mode = "stage"
    if mode not in {"stage", "stage-only", "curriculum"}:
        raise ValueError("mode must be 'stage', 'stage-only', or 'curriculum'")
    if max_chunks_cap is not None:
        max_chunks_cap = positive_int(str(max_chunks_cap), "maximum chunks per stage")
    if profile not in {"fast_train", "gui_train"}:
        raise ValueError("profile must be 'fast_train' or 'gui_train'")
    if not 0 <= eval_reserve < 0.75:
        raise ValueError("evaluation reserve must be >= 0 and < 0.75")
    stage_keys = [item.key for item in stages]
    if stage not in stage_keys:
        raise ValueError(f"Unknown stage {stage!r}; choose one of: {', '.join(stage_keys)}")
    index = stage_keys.index(stage)
    canonical_start_state = str(canonical_start_state).strip() or "runtime_data/states/training_start.state"
    selected_start_state = stage_start_state(project_dir, stages, stage, canonical_start_state)
    remaining_stages = stages[index:]
    effective_envs = envs if (mode in {"curriculum", "stage-only"} or profile == "fast_train") else 1
    rollout_size = n_steps * effective_envs
    effective_batch = valid_batch_size(batch_size, rollout_size)
    requested_checkpoint_interval = positive_int(str(checkpoint_interval), "checkpoint interval")
    checkpoint_every = math.ceil(requested_checkpoint_interval / rollout_size) * rollout_size
    requested_total = int(hours * 3600 * speed)
    total_rollouts = max(1, requested_total // rollout_size)
    total_transitions = total_rollouts * rollout_size
    warnings: list[str] = []
    if checkpoint_every != requested_checkpoint_interval:
        warnings.append(
            f"Checkpoint interval {requested_checkpoint_interval:,} was rounded up to {checkpoint_every:,} "
            f"to align with complete {rollout_size:,}-transition PPO rollouts."
        )
    if effective_batch != batch_size:
        warnings.append(
            f"PPO batch size {batch_size} does not divide the {rollout_size:,}-transition rollout; "
            f"using {effective_batch} for exact divisibility."
        )
    if total_transitions < requested_total:
        warnings.append(
            f"Training budget rounded down from {requested_total:,} to {total_transitions:,} "
            "to use complete PPO rollouts."
        )
    elif total_transitions > requested_total:
        warnings.append(
            f"Requested duration only covers {requested_total:,} transitions, less than one full "
            f"vector PPO rollout ({rollout_size:,}); running one complete rollout requires "
            f"{total_transitions:,} transitions (~{total_transitions / speed / 60:.1f} minutes)."
        )

    if mode == "stage":
        episode_rollouts = stage_episode_rollouts(stage)
        if episode_steps_override is not None:
            episode_steps = positive_int(str(episode_steps_override), "episode step override")
            # Episode caps are rounded up to whole per-environment PPO rollout lengths.
            episode_steps = math.ceil(episode_steps / n_steps) * n_steps
        else:
            episode_steps = episode_rollouts * n_steps
        chunk_transitions = None
        chunks_per_stage = None
        curriculum_transitions = None
        curriculum_hours = None
        reserved_hours = None
        if total_transitions < episode_steps * effective_envs:
            warnings.append(
                "The requested wall-time budget is shorter than one complete max-length episode per environment. "
                "The run can still train, but it may end before any worker reaches its episode timeout."
            )
    elif mode == "stage-only":
        # Stage-only means one selected objective, no evaluation/promotion, and a
        # wall-time-derived transition budget. Run it as ONE subprocess chunk of
        # exactly that budget (rounded to full PPO vector rollouts). Do not enlarge
        # the requested runtime to match either the episode timeout or checkpoint
        # cadence: those are independent settings. A short run may end mid-episode;
        # the final shutdown save still preserves the trained policy.
        episode_rollouts = stage_episode_rollouts(stage)
        if episode_steps_override is not None:
            episode_steps = positive_int(str(episode_steps_override), "episode step override")
            episode_steps = math.ceil(episode_steps / n_steps) * n_steps
        else:
            episode_steps = episode_rollouts * n_steps

        chunk_transitions = total_transitions
        chunks_per_stage = 1
        curriculum_transitions = total_transitions
        curriculum_hours = curriculum_transitions / speed / 3600
        reserved_hours = max(0.0, hours - curriculum_hours)
        remaining_stages = [next(item for item in stages if item.key == stage)]

        steps_available_per_env = total_transitions // effective_envs
        if steps_available_per_env < episode_steps:
            warnings.append(
                f"The requested run provides about {steps_available_per_env:,} transitions per environment, "
                f"less than this stage's {episode_steps:,}-decision episode cap. Training will still run for "
                "the requested budget, but episodes may be unfinished when the run ends. Increase --hours "
                "if you want time for a full max-length episode."
            )
        if max_chunks_cap is not None and max_chunks_cap != 1:
            warnings.append(
                "--max-chunks-per-stage does not split stage-only duration into repeated subprocesses; "
                "stage-only uses one chunk for the whole requested budget to avoid restarting workers "
                "and reloading the same stage start state between chunks."
            )
    else:
        # The plan summary and budget estimate use the largest remaining stage
        # horizon. run_curriculum.py applies each stage's own horizon to its child
        # training process, so short early objectives are not given the final-stage cap.
        remaining_rollout_caps = [stage_episode_rollouts(item.key) for item in remaining_stages]
        if episode_steps_override is not None:
            global_episode_steps = positive_int(str(episode_steps_override), "episode step override")
            global_episode_steps = math.ceil(global_episode_steps / n_steps) * n_steps
        else:
            global_episode_steps = max(remaining_rollout_caps) * n_steps
        minimum_chunk_rollouts = math.ceil(global_episode_steps / n_steps)
        chunk_transitions = minimum_chunk_rollouts * rollout_size
        episode_steps = global_episode_steps
        training_target_rollouts = max(1, int(total_rollouts * (1.0 - eval_reserve)))
        stage_count = len(remaining_stages)
        max_chunks_affordable = training_target_rollouts // (minimum_chunk_rollouts * stage_count)
        chunks_per_stage = max(1, min(max_chunks_cap if max_chunks_cap is not None else DEFAULT_MAX_CHUNKS_PER_STAGE, max_chunks_affordable))
        curriculum_transitions = stage_count * chunks_per_stage * chunk_transitions
        curriculum_hours = curriculum_transitions / speed / 3600
        reserved_hours = max(0.0, hours - curriculum_hours)
        if max_chunks_affordable < 1:
            warnings.append(
                f"The requested duration cannot fund even one full episode-cap-sized chunk for each of the "
                f"{stage_count} remaining stages. The minimum estimated training time at this plan is "
                f"{curriculum_hours:.2f} hours, before evaluations and shutdown overhead."
            )
        elif curriculum_transitions < int(total_transitions * (1.0 - eval_reserve)):
            warnings.append(
                "The curriculum allocation leaves some of the training budget unused to preserve equal, "
                "rollout-aligned chunks across remaining stages."
            )

    episode_wall_seconds = episode_steps / (speed / effective_envs)
    return Plan(
        mode=mode,
        profile="fast_train" if mode in {"curriculum", "stage-only"} else profile,
        stage=stage,
        stage_label=next(item.label for item in stages if item.key == stage),
        start_state=str(selected_start_state),
        canonical_start_state=canonical_start_state,
        checkpoint_every=checkpoint_every,
        requested_hours=hours,
        speed=speed,
        envs=effective_envs,
        n_steps=n_steps,
        batch_size=effective_batch,
        rollout_size=rollout_size,
        total_transitions=total_transitions,
        estimated_hours=total_transitions / speed / 3600,
        max_episode_steps=episode_steps,
        estimated_episode_minutes=episode_wall_seconds / 60,
        remaining_stages=len(remaining_stages),
        chunk_transitions=chunk_transitions,
        chunks_per_stage=chunks_per_stage,
        curriculum_transitions=curriculum_transitions,
        curriculum_estimated_hours=curriculum_hours,
        reserved_hours=reserved_hours,
        warnings=tuple(warnings),
    )


def values_for_plan(plan: Plan) -> dict[str, str]:
    rollout = plan.rollout_size
    # Keep the existing absolute checkpoint interval independent of run duration.
    # main.py aligns it to full vector rollouts and writes only after the optimizer update.
    checkpoint_every = plan.checkpoint_every
    values = {
        "POKEMON_AI_PROFILE": plan.profile,
        "POKEMON_AI_OBJECTIVE": plan.stage,
        "POKEMON_AI_CANONICAL_START_STATE": plan.canonical_start_state,
        "POKEMON_AI_TRAINING_START_STATE": plan.start_state,
        "POKEMON_AI_RESUME": "1",
        "POKEMON_AI_MAX_EPISODE_STEPS": str(plan.max_episode_steps),
        "POKEMON_AI_PPO_N_STEPS": str(plan.n_steps),
        "POKEMON_AI_PPO_BATCH_SIZE": str(plan.batch_size),
        "POKEMON_AI_TOTAL_TIMESTEPS": str(plan.total_transitions),
        "POKEMON_AI_CHECKPOINT_EVERY": str(checkpoint_every),
        "POKEMON_AI_LATEST_BACKUP_ENABLED": "1",
        "POKEMON_AI_SHUTDOWN_TIMEOUT": "900",
        "POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT": "60",
        "POKEMON_AI_PROGRESS_EVERY": str(PROGRESS_ROLLOUTS * rollout),
        "POKEMON_AI_TERMINAL_PROGRESS_EVERY": str(TERMINAL_PROGRESS_ROLLOUTS * rollout),
    }
    values["POKEMON_AI_CURRICULUM_STAGE_ONLY"] = "1" if plan.mode == "stage-only" else "0"
    # Always manage this key so a prior stage-only override cannot leak into a
    # later full-curriculum configuration and flatten every stage's own horizon.
    values["POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS"] = (
        str(plan.max_episode_steps) if plan.mode == "stage-only" else ""
    )
    if plan.mode in {"curriculum", "stage-only"} or plan.profile == "fast_train":
        values["POKEMON_AI_NUM_ENVS"] = str(plan.envs)
    if plan.mode in {"curriculum", "stage-only"}:
        assert plan.chunk_transitions is not None and plan.chunks_per_stage is not None
        values["POKEMON_AI_CURRICULUM_CHUNK_TIMESTEPS"] = str(plan.chunk_transitions)
        values["POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE"] = str(plan.chunks_per_stage)
    return values


def update_env_text(original_lines: list[str], replacements: dict[str, str]) -> str:
    """Replace managed keys, remove duplicate copies of those keys, preserve unrelated lines."""
    output: list[str] = []
    seen: set[str] = set()
    for raw in original_lines:
        match = KEY_RE.match(raw)
        if not match:
            output.append(raw)
            continue
        key = match.group(1)
        if key not in replacements:
            output.append(raw)
            continue
        if key in seen:
            continue
        output.append(f"{key}={replacements[key]}")
        seen.add(key)
    missing = [key for key in replacements if key not in seen]
    if missing:
        if output and output[-1].strip():
            output.append("")
        output.append("# Stage-aware values calculated by configure_training.py")
        output.extend(f"{key}={replacements[key]}" for key in missing)
    return "\n".join(output).rstrip() + "\n"


def unified_diff(path: Path, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=str(path), tofile=str(path) + " (proposed)",
    ))


def ask_value(prompt: str, default: str, *, choices: Iterable[str] | None = None) -> str:
    choices_list = list(choices) if choices else None
    choices_text = f" [{'/'.join(choices_list)}]" if choices_list else ""
    while True:
        raw = input(f"{prompt}{choices_text} [{default}]: ").strip()
        value = raw or default
        if choices_list and value not in choices_list:
            print("Please choose one of: " + ", ".join(choices_list))
            continue
        return value


def ask_stage(stages: list[StageInfo], default: str) -> str:
    print("\nCurriculum stages:")
    for index, item in enumerate(stages, start=1):
        cap = STAGE_EPISODE_ROLLOUTS.get(item.key, DEFAULT_STAGE_EPISODE_ROLLOUTS)
        print(f"  {index:>2}. {item.key:<20} ~{cap} per-env rollout(s) | {item.label}")
    choices = {str(index): item.key for index, item in enumerate(stages, start=1)}
    choices.update({item.key: item.key for item in stages})
    while True:
        raw = input(f"Stage number or name [{default}]: ").strip()
        if not raw:
            return default
        if raw in choices:
            return choices[raw]
        print("Please enter a listed stage number or exact stage key.")


def resolve_project_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    if (script_dir / ".env").exists():
        return script_dir
    return Path.cwd().resolve()


def print_plan(plan: Plan, replacements: dict[str, str], stage_infos: list[StageInfo]) -> None:
    per_env_speed = plan.speed / plan.envs
    print("\n================ Stage-aware training plan ================")
    print(f"Run mode/profile          : {plan.mode} / {plan.profile}")
    print(f"Selected stage             : {plan.stage} — {plan.stage_label}")
    print(f"Start state                : {plan.start_state}")
    if not Path(plan.start_state).is_file():
        print("WARNING: this start-state file does not exist yet. Complete/verify the preceding stage first.")
    print(f"Requested wall time        : {plan.requested_hours:.2f} hours")
    print(f"Aggregate measured speed   : {plan.speed:.2f} transitions/s total")
    print(f"Effective environments     : {plan.envs}")
    print(f"Estimated speed/environment: {per_env_speed:.2f} policy decisions/s (assuming balanced workers)")
    print(f"PPO rollout                : {plan.n_steps:,} × {plan.envs} = {plan.rollout_size:,} transitions")
    checkpoint_every = int(replacements.get("POKEMON_AI_CHECKPOINT_EVERY", str(plan.checkpoint_every)))
    print(f"Checkpoint cadence         : every {checkpoint_every:,} transitions (~{checkpoint_every / plan.speed / 60:.1f} min at entered speed; saved after PPO update)")
    print("Checkpoint/archive saves   : old model/archive use hard links where supported; final duplicate saves are skipped")
    print(f"Total transition budget    : {plan.total_transitions:,} (~{plan.estimated_hours:.2f} h at entered speed)")
    print(f"Episode max                : {plan.max_episode_steps:,} decisions/environment")
    print(f"Approx. max episode time   : {plan.estimated_episode_minutes:.1f} minutes/environment at entered speed")
    print("Episode limits are policy-decision limits, not GBA frames; the presets are heuristics, not guarantees.")
    if plan.mode in {"curriculum", "stage-only"}:
        if plan.mode == "stage-only":
            print("Controller behavior        : stage-only; no evaluation, no promotion")
            print("Selected-stage training    : one stage only; exit after the configured chunks")
        else:
            print(f"Stages from selected point  : {plan.remaining_stages}")
        print(f"Curriculum chunk            : {plan.chunk_transitions:,} transitions")
        print(f"Maximum chunks per stage    : {plan.chunks_per_stage}")
        print(f"Planned maximum training    : {plan.curriculum_transitions:,} transitions (~{plan.curriculum_estimated_hours:.2f} h)")
        if plan.mode == "curriculum":
            print("Evaluation time is additional; the controller may stop early when an evaluation gate fails.")
        else:
            print("Stage-only mode trains for the planned chunks and will not evaluate or advance to another stage.")
    print("\nValues to write:")
    for key, value in replacements.items():
        print(f"  {key}={value}")
    if plan.warnings:
        print("\nWarnings / math notes:")
        for warning in plan.warnings:
            print(f"  - {warning}")
    print("===========================================================\n")


def configured_canonical_start_state(project_dir: Path, values: dict[str, str]) -> str:
    """Resolve a stable starter state without reusing a later stage's selected state.

    Older .env files only have POKEMON_AI_TRAINING_START_STATE. Reuse it as the
    canonical state only when it is not inside the milestones directory.
    """
    configured = current_value(values, "POKEMON_AI_CANONICAL_START_STATE", "").strip()
    if configured:
        return configured

    legacy = values.get("POKEMON_AI_TRAINING_START_STATE", "").strip()
    if legacy:
        legacy_path = Path(legacy).expanduser()
        legacy_path = legacy_path.resolve() if legacy_path.is_absolute() else (project_dir / legacy_path).resolve()
        milestone_root = (project_dir / "runtime_data" / "states" / "milestones").resolve()
        try:
            legacy_path.relative_to(milestone_root)
        except ValueError:
            return legacy

    return "runtime_data/states/training_start.state"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", help="project directory containing .env, main.py, curriculum_objectives.py")
    parser.add_argument("--hours", type=float, help="desired training duration in hours")
    parser.add_argument("--speed", type=float, default=None,
                        help=f"measured aggregate transitions/s (default {DEFAULT_SPEED:g}; not per-environment speed)")
    parser.add_argument("--mode", choices=("stage", "direct", "stage-only", "curriculum"),
                        help="stage=direct training; stage-only=controller training with no evaluation/promotion; curriculum=automatic staged evaluation/promotion")
    parser.add_argument("--stage", help="stage key from curriculum_objectives.py; default is current .env objective or starter")
    parser.add_argument("--profile", choices=("fast_train", "gui_train"), help="single-stage execution profile")
    parser.add_argument("--num-envs", type=int, help="headless environment count (default .env value, else 4)")
    parser.add_argument("--n-steps", type=int, help="PPO steps/environment per rollout (default .env value, else 2048)")
    parser.add_argument("--batch-size", type=int, help="PPO minibatch size (default .env value, else 256)")
    parser.add_argument("--episode-steps", type=int, help="override the stage's recommended episode cap (rounded up to PPO_N_STEPS)")
    parser.add_argument("--evaluation-reserve", type=float, default=DEFAULT_EVAL_RESERVE,
                        help="fraction of curriculum wall-time budget reserved for evaluations (default 0.20)")
    parser.add_argument("--max-chunks-per-stage", type=int, default=None,
                        help="optional per-stage chunk cap for curriculum mode; stage-only always uses one duration-sized chunk")
    parser.add_argument("--auto-detect-speed", action="store_true",
                        help="use the most recent aggregate speed found in project logs, when one is found")
    parser.add_argument("--dry-run", action="store_true", help="show proposed .env changes without writing")
    parser.add_argument("--yes", action="store_true", help="write changes without confirmation")
    args = parser.parse_args(argv)

    try:
        project_dir = resolve_project_dir(args.project_dir)
        env_path = project_dir / ".env"
        if not (project_dir / "main.py").is_file():
            raise ValueError(f"main.py not found in project directory: {project_dir}")
        stage_infos = load_stages(project_dir)
        values, original_lines = parse_dotenv(env_path)
        stage_keys = [stage.key for stage in stage_infos]
        default_stage = current_value(values, "POKEMON_AI_OBJECTIVE", "starter")
        if default_stage not in stage_keys:
            default_stage = "starter" if "starter" in stage_keys else stage_keys[0]

        interactive = sys.stdin.isatty() and not any(
            (args.hours is not None, args.mode is not None, args.stage is not None, args.dry_run, args.yes)
        )
        if args.hours is not None:
            hours = positive_float(str(args.hours), "hours")
        elif interactive:
            hours = positive_float(ask_value("Desired training duration in hours", "3"), "hours")
        else:
            hours = 3.0

        if args.mode:
            mode = "stage" if args.mode == "direct" else args.mode
        elif interactive:
            mode_raw = ask_value("Run mode", "stage", choices=("stage", "stage-only", "curriculum"))
            mode = mode_raw
        else:
            mode = "stage"

        if args.stage:
            stage = args.stage.strip()
        elif interactive:
            stage = ask_stage(stage_infos, default_stage)
        else:
            stage = default_stage
        if stage not in stage_keys:
            raise ValueError(f"Unknown stage {stage!r}; choose one of: {', '.join(stage_keys)}")

        if args.auto_detect_speed:
            detected_speed = detect_recent_speed(project_dir)
            speed = detected_speed if detected_speed is not None else DEFAULT_SPEED
            if detected_speed is None:
                print(f"[Speed] No aggregate speed found in logs; using {DEFAULT_SPEED:g} transitions/s.")
            else:
                print(f"[Speed] Using {speed:.2f} aggregate transitions/s detected from recent logs.")
        elif args.speed is not None:
            speed = positive_float(str(args.speed), "speed")
        elif interactive:
            detected_speed = detect_recent_speed(project_dir)
            if detected_speed is not None:
                print(f"[Speed] Most recent parseable log speed is {detected_speed:.2f}; your supplied current default is {DEFAULT_SPEED:g}.")
            speed = positive_float(ask_value("Recent aggregate transitions/s total", f"{DEFAULT_SPEED:g}"), "speed")
        else:
            speed = DEFAULT_SPEED

        old_envs = positive_int(current_value(values, "POKEMON_AI_NUM_ENVS", str(DEFAULT_NUM_ENVS)), "POKEMON_AI_NUM_ENVS")
        old_n_steps = positive_int(current_value(values, "POKEMON_AI_PPO_N_STEPS", str(DEFAULT_N_STEPS)), "POKEMON_AI_PPO_N_STEPS")
        old_batch = positive_int(current_value(values, "POKEMON_AI_PPO_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)), "POKEMON_AI_PPO_BATCH_SIZE")
        envs = positive_int(str(args.num_envs if args.num_envs is not None else old_envs), "number of environments")
        n_steps = positive_int(str(args.n_steps if args.n_steps is not None else old_n_steps), "PPO_N_STEPS")
        batch = positive_int(str(args.batch_size if args.batch_size is not None else old_batch), "PPO_BATCH_SIZE")

        if args.profile:
            profile = args.profile
        else:
            prior = current_value(values, "POKEMON_AI_PROFILE", "fast_train")
            profile = prior if prior in {"fast_train", "gui_train"} else "fast_train"
            if interactive and mode == "stage":
                profile = ask_value("Stage training profile", profile, choices=("fast_train", "gui_train"))
        if mode in {"curriculum", "stage-only"}:
            profile = "fast_train"
        max_chunks_cap = args.max_chunks_per_stage
        if mode == "curriculum" and max_chunks_cap is None:
            max_chunks_cap = DEFAULT_MAX_CHUNKS_PER_STAGE

        canonical_start_state = configured_canonical_start_state(project_dir, values)
        # Use the persistent .env checkpoint interval, not a stale exported value
        # from a shell that may have launched a previous run with an override.
        checkpoint_interval = positive_int(
            values.get("POKEMON_AI_CHECKPOINT_EVERY", str(DEFAULT_CHECKPOINT_INTERVAL_TRANSITIONS)),
            "POKEMON_AI_CHECKPOINT_EVERY",
        )
        plan = build_plan(
            mode=mode, profile=profile, stage=stage, stages=stage_infos, project_dir=project_dir,
            hours=hours, speed=speed, envs=envs, n_steps=n_steps, batch_size=batch,
            eval_reserve=args.evaluation_reserve, max_chunks_cap=max_chunks_cap,
            episode_steps_override=args.episode_steps, canonical_start_state=canonical_start_state,
            checkpoint_interval=checkpoint_interval,
        )
        replacements = values_for_plan(plan)
        old_text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
        new_text = update_env_text(original_lines, replacements)
        print_plan(plan, replacements, stage_infos)
        diff = unified_diff(env_path, old_text, new_text)
        if diff:
            print("Proposed .env diff:")
            print(diff, end="" if diff.endswith("\n") else "\n")
        else:
            print("The .env values are already configured as proposed.")
        if args.dry_run:
            print("[Dry run] .env was not changed.")
            return 0
        if not args.yes:
            try:
                answer = input(f"Write these settings to {env_path}? [y/N]: ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in {"y", "yes"}:
                print("Cancelled; .env was not changed.")
                return 0
        env_path.write_text(new_text, encoding="utf-8")
        print(f"[Configured] Updated {env_path}; no project source files or other files were changed.")
        exported = sorted(key for key in replacements if key in os.environ)
        if exported:
            print("[Warning] These exported variables override .env in the launch shell:")
            for key in exported:
                print(f"  {key}")
            print("Unset these shell variables before launching if you want the new .env values to apply.")
        if plan.mode == "curriculum":
            print(f"Run with: .venv/bin/python run_curriculum.py --start-stage {plan.stage}")
        elif plan.mode == "stage-only":
            print(f"Run with: .venv/bin/python run_curriculum.py --start-stage {plan.stage} --stage-only")
        else:
            print(f"Run with: ./run.sh --profile {plan.profile}")
        print("Note: POKEMON_AI_RESUME=1 is set to continue the existing model checkpoint.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[Configuration Error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
