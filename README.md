# Pokémon Emerald RL Agent

A local reinforcement-learning agent for **vanilla Pokémon Emerald (BPEE rev 0)**. It runs the mGBA Python core through Gymnasium and trains a recurrent PPO policy with Stable-Baselines3 and SB3-Contrib.

## Design

- **Policy:** recurrent PPO (`RecurrentPPO`) with LSTM memory.
- **Observation:** 120×80 RGB frame plus structured RAM-derived game state (map and player position, live party/levels/EXP, story flags and badges), visited-tile memory, previous action and stagnation features.
- **Actions:** nine discrete actions — wait, four directions, A, B, Start and Select. Action timing is controlled by `POKEMON_AI_ACTION_REPEAT` and `POKEMON_AI_ACTION_RELEASE_FRAMES`.
- **Reward:** task completion and story/trainer events, party/level/badge progress, small map/tile/screen novelty rewards, EXP-gain shaping, and a loop/stagnation penalty.
- **Parallelism:** `fast_train` normally uses four headless environments; `gui_train` uses one visible environment. PPO collects `POKEMON_AI_PPO_N_STEPS` per environment before each update.
- **Important distinction:** the policy uses both visual observations and privileged RAM-derived features. It is not a vision-only agent or a proven full-game player.

## Requirements and game data

Use the project's existing `.venv` and `run.sh`; the launcher configures the local mGBA build and native-library paths. Dependencies are listed in `requirements.txt`.

Place your legally obtained, unmodified ROM at `PokemonEmerald.gba` beside `main.py`. The default identity check expects SHA-1 `f3ae088181bf583e55daf962a92bb46f4f1d07b7`:

```dotenv
POKEMON_AI_EXPECTED_ROM_SHA1=f3ae088181bf583e55daf962a92bb46f4f1d07b7
POKEMON_AI_REQUIRE_ROM_SHA1=1
```

The ROM, cartridge save, trained weights, savestates and Python environment are local assets; they are not included in source packages.

## Start state

The canonical pre-starter state is `runtime_data/states/training_start.state`: Route 101, bedroom clock set, before rescuing Birch, with an empty party. Create or recapture it with:

```bash
./run.sh --profile setup
```

Play to the required position and press **K** to capture the state, then quit setup. Do not use an in-progress battle or a state with the starter already obtained. Later curriculum stages require the immediately preceding verified milestone at `runtime_data/states/milestones/<stage>.state` and its matching `.json` manifest.

## Run commands

```bash
# Headless training (four environments by default)
./run.sh --profile fast_train

# Training with a visible game (one environment)
./run.sh --profile gui_train

# Evaluate the existing model without updating weights
./run.sh --profile evaluate

# Watch the policy in a visible window / headless at maximum speed
# Watch AI in GUI (press F or click FF to toggle Fast-Forward)
./run.sh --profile gui_watch

# Watch AI on any specific stage (e.g. reach_route101, first_badge)
./run.sh --profile gui_watch --stage reach_route101

# Continuous tour: AI plays from the canonical Route 101 start through all cleared milestones
./run.sh --profile continuous_watch

# Run the fully autonomous full-game controller (auto-configures, trains, evaluates, promotes)
./run.sh --auto

# Prepare the canonical start state
./run.sh --profile setup
```

The model is saved as `<POKEMON_AI_MODEL_NAME>.zip` in the project root (default `emerald_ppo_hybrid_v1.zip`). Resume with `POKEMON_AI_RESUME=1`; with resume enabled, `POKEMON_AI_TOTAL_TIMESTEPS` is the budget for the next training call, not a lifetime cap from timestep zero. Use a new `POKEMON_AI_MODEL_NAME` for a separate experiment.

## Fully Autonomous Full-Game Curriculum

The curriculum spans 58 sequential milestones across the entire Pokémon Emerald story, from the canonical post-clock/pre-starter Route 101 state through all 8 Gym Badges and Champion Wallace. Event predicates use the vanilla BPEE event-flag definitions and are checked against decoded RAM telemetry.

Launch the autonomous controller with a single command:

```bash
# Preview the full roadmap without running
./run.sh --auto --dry-run

# Run the complete autonomous pipeline
./run.sh --auto
```

The autonomous controller automatically:
1. Resumes from the current stage in `runtime_data/curriculum_state.json`.
2. Computes the required rollout horizon and chunk budget dynamically.
3. Passes stage-specific configuration to child processes without rewriting `.env`.
4. Executes stage-scaled, complete vector-rollout training chunks, applying the selected stage's own per-environment episode cap to every training child process.
5. Evaluates using `POKEMON_AI_CURRICULUM_EVAL_EPISODES` from `.env` (the supplied configuration and controller fallback both use 5) against the configured success gate. It stops between episodes as soon as the gate becomes mathematically unreachable; incomplete evaluations can never promote a stage. With 5 episodes and a 90% gate, all 5 must pass (a single failure makes 90% impossible); with 10 episodes and a 90% gate, 9 must pass (the second failure makes 90% impossible). Promotion still requires a complete passing evaluation and a fresh milestone savestate/manifest matching the evaluated checkpoint and ROM.
6. When the current model misses the gate, screens recent distinct, ZIP-validated checkpoints, fully evaluates a promising candidate, and activates it only if its objective success rate improves.
7. Archives `<model>_<stage>_mastered.zip` after each verified promotion and continues until Champion Wallace is defeated. Autonomous mode defaults to unlimited chunks per stage (matching the original runner); pass `--max-chunks-per-stage N` to cap it, with `0` meaning unlimited. The direct `run_curriculum.py` entry point still reads the `.env` chunk limit.

**Ctrl+C shutdown:** `run_autonomous.py` launches the curriculum controller in an isolated process session and forwards Ctrl+C to it. The curriculum controller then signals its active `main.py` process and waits for cleanup. During training, `main.py` finishes the current PPO rollout and optimizer update, atomically saves the latest verified model, archives it when possible, saves emulator states, and closes workers. During evaluation, no model weights are changed; the active evaluation episode is abandoned, the emulator state is saved during environment cleanup, and the curriculum is marked interrupted so the incomplete evaluation cannot promote a stage. The launcher waits for the controller's full shutdown window and only escalates if graceful cleanup times out. A normal Ctrl+C should return to the shell without an uncaught `KeyboardInterrupt` traceback. Do not use `kill -9` as a normal stop method.

See [CURRICULUM_STAGES.md](CURRICULUM_STAGES.md) for the complete 58-stage specification.

### Visual Inspection & Fast Forward

- **Fast Forward**: In any GUI window (`gui_watch`, `gui_train`, `continuous_watch`), press `F` or `Spacebar` (or click `[FF: ON/OFF]` in the top-right overlay) to run unthrottled at maximum hardware speed. Press again to return to 60 FPS.
- **Inspect Any Stage**: Run `./run.sh --profile gui_watch --stage <stage_name>` to boot into any verified milestone.
- **Continuous Tour**: Run `./run.sh --profile continuous_watch` to watch the AI from the canonical `runtime_data/states/training_start.state` (post-clock, pre-starter Route 101) without per-stage resets. A stale stage-specific start-state setting is intentionally ignored in this mode.

## Configure a time budget

`configure_training.py` estimates transitions from **aggregate** transitions/s and rounds work to full PPO rollouts. Decimal hours are supported. Preview before writing `.env`:

```bash
# Example: 30 minutes on Starter at 80 aggregate transitions/s
.venv/bin/python configure_training.py \
  --hours 0.5 --speed 80 --mode stage-only --stage starter --dry-run
```

Repeat without `--dry-run` to review and confirm the `.env` write, or run it without arguments for interactive prompts. Modes:

- `stage` — direct single-objective training with a stage-specific episode cap.
- `stage-only` — controller-managed chunks, with evaluation and promotion disabled.
- `curriculum` — automatic evaluation and promotion across remaining stages. Each training process receives its own stage episode cap; evaluation can stop early only after enough completed failures make the configured gate impossible.

Episode presets are starting heuristics, not guarantees of task completion. `POKEMON_AI_MAX_EPISODE_STEPS` counts policy decisions per environment, not GBA frames, and remains the direct/standalone training setting. The curriculum controller overrides it with each stage's preset. To intentionally override all curriculum training episode caps, set `POKEMON_AI_CURRICULUM_TRAIN_MAX_EPISODE_STEPS`; stage-only plans written by the configurator set this explicitly for the selected stage. Ensure the configured total budget and chunks are compatible with the rollout size. Pass `--eval-episodes N`, `--gate RATE`, or `--max-chunks-per-stage N` to `run_autonomous.py` only when you want to override `.env`; omitted options respect `.env`. The per-episode evaluation hard limit remains stage-specific (the same objective horizon used by its stage preset), and `POKEMON_AI_EVAL_NO_PROGRESS_STEPS` defaults to 4,096 decisions to end a run that is no longer making meaningful game progress.

For `N` environments and `POKEMON_AI_PPO_N_STEPS` steps per environment, one vector rollout is:

`rollout transitions = N × POKEMON_AI_PPO_N_STEPS`

For the common four-environment configuration with 2,048 steps each, a rollout is 8,192 transitions. The measured aggregate speed varies with CPU load, emulator throughput, PPO optimization and saving overhead.

## Saves and runtime data

- Current model: `<POKEMON_AI_MODEL_NAME>.zip` at the project root.
- Rolling model snapshots: `runtime_data/checkpoints/` (bounded by `POKEMON_AI_CHECKPOINT_KEEP`).
- Canonical and milestone states: `runtime_data/states/`.
- Curriculum progress: `runtime_data/curriculum_state.json`.
- Logs: `logs/`; TensorBoard data: the directory set by `POKEMON_AI_TENSORBOARD_LOG_DIR`.

Model writes are atomic. Periodic checkpoints follow `POKEMON_AI_CHECKPOINT_EVERY`; the latest backup and archive snapshots use hard links where supported, with copy fallback. Episode resets normally restore an in-memory mGBA state when supported; file-backed state handling is a compatibility fallback. Milestone savestates and manifests are written by the relevant verified curriculum/evaluation path. Keep checkpoints and milestone states; ordinary logs can be archived or removed separately.

`POKEMON_AI_MINI_EPISODE_STEPS` starts a soft exploration-memory window for later objectives. It resets short-term visit/loop memory only—it does **not** reset the game, end a Gymnasium episode, or reset the LSTM state.

## Console verbosity

`POKEMON_AI_VERBOSE` controls low-level diagnostic output without disabling training progress, checkpoint notifications, warnings, or errors:

- `0` — quiet normal training. Suppresses per-worker SaveBlock1 flag dumps, party RAM mismatch tables, and mini-episode boundary messages.
- `1` — normal verbosity. Reports mini-episode boundaries from worker 0 only; suppresses low-level RAM diagnostic dumps.
- `2` — debug verbosity. Enables SaveBlock1 flag and party RAM diagnostics and reports mini-episode boundaries from every worker. SaveBlock1 flag dumps are triggered by changes in the sampled flag bytes/validity, not by pointer-address relocation alone.

Changing this setting affects logging only. Mini-episode exploration-memory resets, RAM sampling, reward calculation, PPO rollouts, and checkpoint behavior continue unchanged.

## Rewards and monitoring

The active objective receives `POKEMON_AI_TASK_SUCCESS_REWARD` on success. Additional signals include exploration novelty, party/level/badge progress, story flags, trainer wins, small party-EXP increases, and loop/stagnation penalties. EXP reward is a shaping proxy, not a guaranteed one-time reward for every wild-battle victory; the battle's actual event flags and live RAM telemetry remain important.

TensorBoard starts automatically when enabled, normally at [http://127.0.0.1:6006](http://127.0.0.1:6006). Useful metrics include transition speed, completed-episode reward/length, task success rate, reward components, and timeouts. Historical data uses the log directory configured in `.env`; use a new directory for a clean comparison.

## Tests

Run tests in the project's virtual environment:

```bash
.venv/bin/python -m py_compile main.py run_curriculum.py configure_training.py curriculum_objectives.py
.venv/bin/python -m unittest discover -s tests -v
```

Unit tests use mocked RAM and temporary checkpoints; they do not replace a live smoke test with the local ROM and mGBA binding. The `test_main_logic.py` RAM/environment suite requires Gymnasium, Stable-Baselines3, and SB3-Contrib; lightweight environments without those packages skip that module explicitly. See [TEST_REPORT.md](TEST_REPORT.md) for the verification scope and limitations. Review source changes with Git and validate stage transitions before relying on long runs.

Saved v5 milestones can resume only after their decoded state passes the corrected v6 objective predicate. v4 migration is restricted to the unchanged starter predecessor; incompatible or false-positive legacy milestones are rejected rather than silently promoted.


## Train / interrupt

The supplied `.env` keeps the bounded one-chunk stage-only budget for direct stage-only runs; `./run.sh --auto` overrides that with unlimited chunks per stage by default. The autonomous parent owns the optional TensorBoard process; its child training/evaluation workers do not start duplicate servers. To run it:

```bash
.venv/bin/python run_curriculum.py --start-stage starter --stage-only
```

The stage-only controller will run exactly one chunk by default, not silently run eight 250,000-step chunks. For a different duration, first let the configurator calculate and write a rollout-aligned budget; for example:

```bash
.venv/bin/python configure_training.py --hours 1 --mode stage-only --stage starter --yes
```

Then launch the stage-only command above. The configurator updates the duration, episode cap, checkpoint and progress intervals consistently. Press Ctrl+C once to request a safe stop. Look for output similar to:

```text
[Interrupt] SIGINT received. Finishing the current full PPO rollout and optimizer update...
[Checkpoint] Interrupt model atomically saved after a completed PPO update...
[Interrupt] Safe checkpoint completed...
```

The controller should then report that the child exited normally. If it reports SIGTERM/SIGKILL escalation, inspect the child log and do not assume the last update was saved.

To inspect a saved model without training:

```bash
.venv/bin/python - <<'PY'
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load('emerald_ppo_hybrid_v1.zip', device='cpu')
print('timesteps:', model.num_timesteps)
print('PPO updates:', model._n_updates)
PY
```

## Training parameters (what they mean)

The current policy is about **5.88 million trainable parameters**. A parameter is a learned floating-point weight or bias inside the neural network. PPO updates those values using the policy and value losses; the number is not a target or an amount of training data. The checkpoint also stores Adam's moment estimates for the trainable tensors; those optimizer values are state, not additional trainable model parameters.

The input is a Dict observation:

- Screen image: 3 × 80 × 120 pixels, processed into 256 CNN features.
- Visited-tile memory: 64 × 64 = 4,096 values.
- Structured RAM/game state: 61 values.
- Combined feature vector: 4,413 values.

The actor and critic each use a one-layer LSTM with 128 hidden units, followed by `[128, 128]` MLP layers. The policy selects among nine actions: WAIT, UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.

The 4,413 feature count is **not** the model's parameter count. Increasing LSTM width, CNN channels, or MLP widths increases trainable parameters and compute, but does not automatically make the agent smarter. On the Ryzen 5 5600G, PPO optimization is already the major runtime cost. Do not increase parameters until evaluation results show a specific representational limitation. Preserve the existing checkpoint and use a separate model name for any future architecture experiment.
