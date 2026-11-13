#!/usr/bin/env python3
"""Single-command autonomous Pokémon Emerald full-game curriculum runner.

The safety-critical stage loop, evaluation-report validation, milestone freshness
checks, archived-checkpoint comparison/rollback, and TensorBoard lifecycle live in
run_curriculum.py. This launcher owns the outermost process boundary so Ctrl+C is
forwarded to the controller instead of producing an uncaught KeyboardInterrupt.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

from curriculum_objectives import CURRICULUM_PLAN_ID, STAGES

ROOT_DIR = Path(__file__).resolve().parent


def resolve_python(project_dir: Path = ROOT_DIR) -> Path:
    venv_python = project_dir / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return venv_python
    return Path(sys.executable)


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    project_dir = args.project_dir.expanduser().resolve()
    python = resolve_python(project_dir)
    command = [str(python), "run_curriculum.py", "--project-dir", str(project_dir)]
    if args.start_stage:
        command.extend(("--start-stage", args.start_stage))
    if args.dry_run:
        command.append("--dry-run")

    child_env = os.environ.copy()
    if args.gate is not None:
        child_env["POKEMON_AI_CURRICULUM_SUCCESS_RATE"] = str(args.gate)
    if args.eval_episodes is not None:
        child_env["POKEMON_AI_CURRICULUM_EVAL_EPISODES"] = str(args.eval_episodes)
    if args.max_chunks_per_stage is not None:
        child_env["POKEMON_AI_CURRICULUM_MAX_CHUNKS_PER_STAGE"] = str(args.max_chunks_per_stage)
    child_env["POKEMON_AI_CURRICULUM_STAGE_ONLY"] = "0"
    return command, child_env


def _positive_timeout(env: dict[str, str], key: str, default: float) -> float:
    """Read a positive finite shutdown timeout, falling back safely if malformed."""
    try:
        value = float(env.get(key, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _signal_controller_group(pid: int, sig: int) -> None:
    """Escalate only within the isolated run_autonomous controller session."""
    if os.name == "posix":
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            print(f"[Autonomous Controller] Could not signal controller process group: {exc}", flush=True)


def _stop_controller(process: subprocess.Popen, child_env: dict[str, str]) -> int:
    """Forward Ctrl+C and wait for run_curriculum.py to clean up its active child.

    run_curriculum.py has its own supervisor: it sends SIGINT to main.py, waits for
    its safe checkpoint/VecEnv cleanup, then escalates against that isolated child
    session if necessary. This outer wait allows that entire nested cleanup window.
    """
    shutdown_timeout = _positive_timeout(child_env, "POKEMON_AI_SHUTDOWN_TIMEOUT", 900.0)
    terminate_timeout = _positive_timeout(child_env, "POKEMON_AI_SHUTDOWN_TERMINATE_TIMEOUT", 60.0)
    graceful_timeout = shutdown_timeout + terminate_timeout + 30.0

    print(
        "\n[Autonomous Controller] Ctrl+C received. Forwarding SIGINT to the curriculum "
        f"controller and waiting for its cleanup (up to {graceful_timeout:.0f}s)...",
        flush=True,
    )

    # We started run_curriculum.py in a separate session, so it did not receive the
    # terminal's Ctrl+C directly. Forward exactly one graceful signal to its owner.
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass
    except OSError as exc:
        print(f"[Autonomous Controller] Could not forward SIGINT: {exc}", flush=True)

    # Ignore repeated terminal Ctrl+C while the first graceful shutdown is in flight.
    # Restore the caller's handler after the child has been reaped.
    previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            return int(process.wait(timeout=graceful_timeout))
        except subprocess.TimeoutExpired:
            print(
                "[Autonomous Controller] Curriculum controller exceeded its graceful "
                "shutdown window; sending SIGTERM to its isolated process group.",
                flush=True,
            )
            if os.name == "posix":
                _signal_controller_group(process.pid, signal.SIGTERM)
            else:
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                return int(process.wait(timeout=max(terminate_timeout, 5.0)))
            except subprocess.TimeoutExpired:
                print(
                    "[Autonomous Controller] Controller still has not exited; sending "
                    "SIGKILL to its isolated process group as a last resort.",
                    flush=True,
                )
                if os.name == "posix":
                    _signal_controller_group(process.pid, signal.SIGKILL)
                else:
                    try:
                        process.kill()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    print(
                        "[Autonomous Controller] ERROR: could not reap the curriculum controller.",
                        file=sys.stderr,
                        flush=True,
                    )
                return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autonomously train Pokémon Emerald through all 58 verified curriculum stages."
    )
    parser.add_argument("--project-dir", type=Path, default=ROOT_DIR,
                        help="project root containing main.py, ROM, .env, and runtime data")
    parser.add_argument("--start-stage", choices=STAGES, default=None,
                        help="explicitly override the saved/current curriculum stage")
    parser.add_argument("--gate", type=float, default=None,
                        help="override the success rate required to promote (otherwise use .env; default fallback: 0.90)")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="override episodes per evaluation (otherwise use .env; controller fallback: 5)")
    parser.add_argument("--max-chunks-per-stage", type=int, default=None,
                        help="maximum training chunks per stage (0=unlimited; default: use .env/shell setting)")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview the safe curriculum plan without launching emulators")
    args = parser.parse_args(argv)

    if args.gate is not None and not 0.0 < args.gate <= 1.0:
        parser.error("--gate must be greater than 0 and at most 1")
    if args.eval_episodes is not None and args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.max_chunks_per_stage is not None and args.max_chunks_per_stage < 0:
        parser.error("--max-chunks-per-stage cannot be negative")

    project_dir = args.project_dir.expanduser().resolve()
    if not (project_dir / "run_curriculum.py").is_file():
        parser.error(f"run_curriculum.py was not found under {project_dir}")
    command, child_env = build_command(args)
    print("=" * 76, flush=True)
    print(" Pokémon Emerald — Autonomous Full-Game Controller", flush=True)
    print(f" Curriculum plan   : {CURRICULUM_PLAN_ID}", flush=True)
    print(f" Stages            : {len(STAGES)}", flush=True)
    gate_label = f">= {args.gate:.0%}" if args.gate is not None else "from .env (fallback 90%)"
    episode_label = str(args.eval_episodes) if args.eval_episodes is not None else "from .env (fallback 5)"
    print(f" Evaluation gate   : {gate_label}; episodes={episode_label}", flush=True)
    chunk_cap_label = (
        "from .env / inherited shell" if args.max_chunks_per_stage is None
        else ("unlimited" if args.max_chunks_per_stage == 0 else str(args.max_chunks_per_stage))
    )
    print(f" Chunk cap         : {chunk_cap_label}", flush=True)
    print(" Recovery          : validated archived-checkpoint screening and rollback", flush=True)
    print(" .env mutation     : disabled", flush=True)
    print(" Ctrl+C             : forwarded through the curriculum shutdown supervisor", flush=True)
    print("=" * 76, flush=True)

    try:
        # Isolate the controller from the terminal process group. The launcher alone
        # receives Ctrl+C and forwards it down the supervised shutdown chain.
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            env=child_env,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        print(f"[Autonomous Controller] Could not start curriculum runner: {exc}", file=sys.stderr, flush=True)
        return 2

    try:
        return int(process.wait())
    except KeyboardInterrupt:
        code = _stop_controller(process, child_env)
        if code == 130:
            print("[Autonomous Controller] Stop completed; curriculum state and verified checkpoints were retained.", flush=True)
        else:
            print(
                f"[Autonomous Controller] Controller exited after stop request (exit code {code}); "
                "inspect the shutdown log for any save warnings.",
                flush=True,
            )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
