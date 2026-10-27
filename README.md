# Pokémon Emerald Reinforcement-Learning Agent

A local Pokémon Emerald agent driven by the mGBA Python core and Stable-Baselines3. It learns from the current screenshot, structured RAM state, a visited-tile map, and an LSTM's hidden state across consecutive decisions. The goal for this training configuration is to rescue Professor Birch and obtain the first Pokémon from a Route 101 start.

The project expects a legally obtained Pokémon Emerald ROM with game code `BPEE`, revision 0, named `PokemonEmerald.gba` beside `main.py`. The ROM and emulator savestates are not included in this source package.

## Image observations work in headless mode

Yes. mGBA renders frames into its framebuffer whether or not a GUI window is shown. `GUI_MODE` controls display and interactive controls, not screenshot creation. The observation code still reads and resizes each frame in `fast_train`, `fast_watch`, and `evaluate`. The neural policy receives:

- **Screen:** a 120 × 80 RGB image of the game.
- **Structured state:** party size and levels, position/map identifiers, badge/story flags, progress counters, and previous action.
- **Visited tiles:** a compact map of areas explored in the current episode.
- **Recurrent memory:** an LSTM state carried across actions, enabling it to learn sequences such as dialogue advancement and battle menus.

Image observations are arrays processed by the policy's CNN. This is not a remote vision model and does not need the GUI to be open.

## Install/update dependencies

The current architecture uses RecurrentPPO from SB3-Contrib. Install it in the project's environment so SB3-Contrib and Stable-Baselines3 are resolved to compatible versions:

```bash
.venv/bin/pip install -r requirements-rl.txt
```

Keep using `./run.sh`; it selects the project's mGBA build and sets `LD_LIBRARY_PATH` as expected by this installation.

## Preserve or recreate the canonical start state

The environment still requires `runtime_data/states/training_start.state`: Route 101 after the bedroom clock has been set, before Birch's rescue event has been completed, with an empty party. The package does not contain any game or state files.

If you intentionally delete that state, recreate it once using:

```bash
./run.sh --profile setup
```

Use the game controls to reach normal controllable gameplay on Route 101 after setting the clock, but before starting the rescue; press **K** to save and then quit with `q` or `Esc`. The start-state validator checks the map, clock flag, Birch-rescue flag, player coordinates, and party state. It does **not** prescribe a route, coordinates to walk toward, or a scripted route phase to the policy.

## Training

This version changes the observation/policy architecture from feed-forward PPO to recurrent PPO and removes the manually targeted Route 101 coordinate reward. Old `.zip` checkpoints from the earlier policy are not compatible; keep a backup if desired, and remove or rename `emerald_ppo_hybrid_v1.zip` before the first fresh run because `.env` uses `POKEMON_AI_RESUME=0`.

```bash
./run.sh --profile fast_train
```

Default settings are four parallel headless environments, up to 30,000 actions per episode, and 2,000,000 aggregate transitions. Start with this configured budget, inspect the success rate and reward components, and expand training only if the evidence indicates learning is still improving.

After the first run, a compatible model can be continued by setting `POKEMON_AI_RESUME=1` in `.env`. Do not change the policy architecture or observation schema while resuming a checkpoint.

## Reward design

The task-specific directional reward toward a known Route 101 coordinate is intentionally removed. The remaining rewards are retained:

| Event | Default reward |
|---|---:|
| Every action | −0.00005 |
| New map, first discovery | +0.75 |
| New tile, first discovery | +0.02 |
| New coarse screen state | +0.002, capped at +0.05 per episode |
| Rescued Birch story flag | +12.0 |
| Received first Pokémon story flag | +15.0 |
| Added party member | +5.0 |
| Level gained | +1.0 per combined party level |
| New badge | +50.0 |
| Up to four newly defeated trainer flags per decision | +0.75 each |
| Starter objective completed | +30.0 |
| Detected spatially and visually stagnant loop | −0.05 at configured intervals |

Tile, map, and screen novelty rewards decay as a state is encountered in later episodes, using `base_reward / sqrt(1 + prior_visits)`. This reduces farming the same familiar states after the worker resets. Screen novelty is deliberately tiny and capped so animated pixels and repeated menus cannot dominate task rewards. Story, party, level, badge, trainer, and task-success rewards are event/delta-based and remain in the environment.

The per-action cost is small: 1,000 steps cost 0.05 reward, whereas one first-time tile discovery is worth 0.02. Walking is therefore not inherently penalized more than exploration; the small cost makes endlessly repeating actions less attractive. The loop detector requires both nearly static coordinates and nearly static coarse screens, so a menu/dialogue/battle changing while the player stays put is less likely to be punished as a movement loop.

Reward shaping is a learning signal, not a guarantee of intelligence. Story milestone rewards supply sparse progress feedback; discovery bonuses encourage exploration; recurrent PPO can use information from preceding observations. Training success must still be measured empirically.

## Evaluate and watch

```bash
./run.sh --profile evaluate
./run.sh --profile gui_watch
```

Evaluation is deterministic by default. To diagnose action diversity during watch only:

```bash
POKEMON_AI_STOCHASTIC_WATCH=1 ./run.sh --profile gui_watch
```

Watch mode keeps START and SELECT available and stops after the starter objective; it does not block menu actions. The agent has not been trained on the whole game merely because it can obtain the first Pokémon.

## Checks

Run these inside the project virtual environment:

```bash
.venv/bin/python -m py_compile main.py tests/test_main_logic.py
.venv/bin/python -m unittest discover -s tests -v
```

Unit tests exercise reward helpers, state decoding, action mappings, screen signatures, and start-state validation without launching the ROM. A live mGBA smoke test is still needed to validate the exact installed binding and ROM state.
