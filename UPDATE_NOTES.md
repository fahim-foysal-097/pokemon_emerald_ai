# Pokémon Emerald RL update notes

## Included

- `main.py`: exact BPEE rev 0 ROM SHA-1 verification by default, live-party RAM reads, objective-aware start-state validation, structured success telemetry, objective milestone savestates and JSON manifests, terminal-only episode metrics, machine-readable evaluation reports, atomic latest-model saving, and rolling checkpoint archives.
- `run_curriculum.py`: standard-library curriculum controller that evaluates the current model, screens recent archived checkpoints after failures, fully evaluates and safely activates a better candidate, trains in bounded chunks, and advances starter → Pokédex → first Gym Badge without manual code edits. Promotion checks the fresh milestone state against the exact evaluated checkpoint path and SHA-1, as well as the configured ROM hash.
- `tests/test_main_logic.py`: regression tests for live vs SaveBlock party data, objective boundaries, malformed values, terminal-only metric aggregation, ROM identity checks, atomic JSON and checkpoint archive retention.
- `tests/test_curriculum_manager.py`: pure standard-library tests for controller configuration, checkpoint ranking/validation, safe candidate activation, atomic state manifests, and rejecting a milestone produced by a different checkpoint.
- `install_update.sh`: transactional updater with atomic file replacement, existing-`.env` merge, backups, syntax checks, regression tests, and rollback on failure.
- `.env`: defaults for a fresh installation; the installer merges only missing keys into an existing project `.env`.
- `README.md`: updated installation, evaluation and optional curriculum instructions.

## Important safety properties

The updater does not replace the ROM, cartridge save, learned model ZIP, existing savestates, `run.sh`, installed virtual environment or requirements file. It preserves existing reward/training/threading settings, turns resume on if the selected checkpoint exists, and backs up each changed source/configuration file. If syntax checks or tests fail, changed files are restored from the timestamped backup. The curriculum manager separately preserves the current model before it launches a training subprocess.

## Validation performed while preparing this bundle

- Python bytecode compilation passed for `main.py`, `run_curriculum.py`, and both test modules.
- `bash -n install_update.sh` passed.
- The curriculum-manager tests cover settings parsing, metric-based checkpoint ranking, corrupted/duplicate archive filtering, atomic candidate activation and manifest writes.
- Helper-level `main.py` tests were run with minimal import stubs because `gymnasium`, Stable-Baselines3, SB3-Contrib, and mGBA are not installed in this bundle-preparation container. This validates pure parsing/metric/filesystem helpers, not live emulator integration. The installer runs the full suite with the target project's `.venv` and automatically rolls back if it fails.
- A live mGBA/ROM smoke test was not possible in this container; perform it after installation before starting long curriculum training.

## Soft mini-episodes and limits

For `pokedex` and `first_badge`, `POKEMON_AI_MINI_EPISODE_STEPS=5000` periodically resets the short-term exploration mask/reward novelty/loop history without resetting the emulator, Gymnasium episode, episode return, story/trainer reward guards, lifetime novelty counters, or recurrent LSTM state. The current tile/map/screen is reseeded to avoid an immediate false novelty reward. Starter training is unchanged; set the interval to zero to disable. These boundaries are reported in the info dictionary and TensorBoard. Since the emulator is stateful and the native mGBA build is not available in the bundle-preparation container, a short live run remains mandatory before trusting the new longer-stage behavior.
