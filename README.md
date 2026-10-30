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

Use the project's existing `.venv` and `run.sh`; the launcher configures the local mGBA build and native-library paths. Dependencies are listed in `requirements-rl.txt`.

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
./run.sh --profile gui_watch
./run.sh --profile fast_watch

# Prepare the canonical start state
./run.sh --profile setup
```

The model is saved as `<POKEMON_AI_MODEL_NAME>.zip` in the project root (default `emerald_ppo_hybrid_v1.zip`). Resume with `POKEMON_AI_RESUME=1`; with resume enabled, `POKEMON_AI_TOTAL_TIMESTEPS` is the budget for the next training call, not a lifetime cap from timestep zero. Use a new `POKEMON_AI_MODEL_NAME` for a separate experiment.

## Incremental curriculum

The stage order is defined once in `curriculum_objectives.py` and shared by the environment and controller:

1. `starter` — rescue Birch, win the opening wild battle and reach the lab.
2. `leave_birch_lab` — complete the lab introduction and leave onto Route 101.
3. `reach_oldale` — reach Oldale Town.
4. `reach_route103` — reach Route 103.
5. `defeat_rival` — defeat May on Route 103.
6. `return_birch_lab` — return to Birch's Lab.
7. `pokedex` — obtain the Pokédex.
8. `first_badge` — obtain the first Gym Badge.

Run the controller to evaluate, train and promote through verified milestones:

```bash
.venv/bin/python run_curriculum.py --dry-run
.venv/bin/python run_curriculum.py
```

The controller evaluates the current checkpoint before training, trains in chunks when needed, and promotes only after the configured evaluation gate passes and a fresh, matching milestone savestate and manifest exist. Deterministic fixed-start evaluation checks reproducibility; it does not prove robustness across arbitrary starting states.

### Train one stage without evaluation or promotion

Stage-only mode trains one selected objective for the configured number of chunks, then exits. It skips evaluation, archive comparison and automatic stage promotion. Use it to spend longer on a difficult stage without the controller moving on:

```bash
.venv/bin/python run_curriculum.py --start-stage leave_birch_lab --stage-only
```

Alternatively set `POKEMON_AI_CURRICULUM_STAGE_ONLY=1` and `POKEMON_AI_OBJECTIVE=<stage>` in `.env`. Stages after `starter` still require the previous verified milestone. Set stage-only back to `0` before normal evaluate/promote curriculum runs.

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
- `curriculum` — automatic evaluation and promotion across remaining stages; evaluation time is additional and may stop early at a failed gate.

Episode presets are starting heuristics, not guarantees of task completion. `POKEMON_AI_MAX_EPISODE_STEPS` counts policy decisions per environment, not GBA frames. Ensure the configured total budget and chunks are compatible with the rollout size.

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

## Rewards and monitoring

The active objective receives `POKEMON_AI_TASK_SUCCESS_REWARD` on success. Additional signals include exploration novelty, party/level/badge progress, story flags, trainer wins, small party-EXP increases, and loop/stagnation penalties. EXP reward is a shaping proxy, not a guaranteed one-time reward for every wild-battle victory; the battle's actual event flags and live RAM telemetry remain important.

TensorBoard starts automatically when enabled, normally at [http://127.0.0.1:6006](http://127.0.0.1:6006). Useful metrics include transition speed, completed-episode reward/length, task success rate, reward components, and timeouts. Historical data uses the log directory configured in `.env`; use a new directory for a clean comparison.

## Tests

Run tests in the project's virtual environment:

```bash
.venv/bin/python -m py_compile main.py run_curriculum.py configure_training.py curriculum_objectives.py
.venv/bin/python -m unittest discover -s tests -v
```

Unit tests use mocked RAM and temporary checkpoints; they do not replace a live smoke test with the local ROM and mGBA binding. Review source changes with Git and validate stage transitions before relying on long runs.


## Train / interrupt

The supplied `.env` keeps the previous one-chunk starter-stage budget: 245,760 transitions. To run it:

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
