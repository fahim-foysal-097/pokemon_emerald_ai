# Pokémon Emerald RL agent — recurrent hybrid pipeline

A local Pokémon Emerald reinforcement-learning project using the mGBA Python core, Gymnasium, Stable-Baselines3 and SB3-Contrib `RecurrentPPO`. The policy receives a downscaled RGB frame, structured game state, a visited-tile mask and recurrent LSTM memory.

The source update does **not** include a ROM, a cartridge save, a savestate or learned model weights. Keep your existing `PokemonEmerald.gba`, cartridge save, `runtime_data/`, `.venv`, `run.sh`, and model checkpoint.

## Safe update (recommended)

Extract the update bundle anywhere, then run the installer with your existing project path:

```bash
./install_update.sh /mnt/mew/dev/Python/pokemon_ai_sandbox
```

The installer backs up files it changes, updates `main.py`, tests and this README, and merges only the new settings into your existing `.env`. It does not replace your ROM, model, saves, Python environment, launcher, reward settings, timestep budget or threading settings. If syntax checks or the test suite fail, it restores the previous files from its backup and exits with an error. Run it from the extracted bundle, not from inside the project directory.

## What changed

- Reads the **live** Emerald party globals (`gPlayerPartyCount`/`gPlayerParty`) for acquisition and level detection, rather than relying only on the SaveBlock party copy that is synchronized when the game saves. The SaveBlock count is retained in diagnostics so mismatches are visible.
- Prints a one-time, structured `[Objective] SUCCESS` event with party count, levels, map, coordinates and both live/save-block party counts. Sustained state/reward-processing failures stop the run rather than silently producing a long run of invalid observations.
- Fixes TensorBoard episode-component metrics: component totals are now sampled only on a completed episode, not averaged from a cumulative per-step value. User-stopped incomplete episodes are excluded from success and completed-episode averages.
- Keeps the latest checkpoint atomic and creates a bounded rolling archive at `runtime_data/checkpoints/`. Old snapshots are pruned to the configured retention count; the latest model is not removed by archive pruning.
- Adds regression coverage for the live-party read, curriculum objective conditions, terminal-only metrics, malformed numbers and checkpoint retention.
- Supports switching curriculum targets through `.env`; no source-code edits are required.

## Requirements / environment

Keep using your existing launcher because it selects your mGBA build and native-library paths. If dependencies need to be repaired inside the project's virtual environment:

```bash
.venv/bin/pip install -r requirements-rl.txt
```

The ROM must be a legally obtained, unmodified Pokémon Emerald **BPEE revision 0** image named `PokemonEmerald.gba` beside `main.py`. Startup verifies both the header and, by default, the exact SHA-1 `f3ae088181bf583e55daf962a92bb46f4f1d07b7`. Your confirmed matching hash satisfies this check. To use an intentional alternative build, first verify its own RAM layout, then configure `POKEMON_AI_EXPECTED_ROM_SHA1` to that exact file's SHA-1; do not disable identity checking to silence a mismatch.

## Start state

The configured canonical state is `runtime_data/states/training_start.state`: Route 101 after setting the bedroom clock, before the early-game rescue sequence, with an empty party. It is reused to make training/evaluation comparisons reproducible. If it is missing or invalid, create it once:

```bash
./run.sh --profile setup
```

Use the game controls to set the clock and reach Route 101 before receiving a starter. Press `K` to capture the state, then quit setup. Setup logs the live party, party levels, badge count and story flags. Don't replace this file with an in-progress battle state.

## Run modes

```bash
./run.sh --profile fast_train   # four parallel headless workers by default
./run.sh --profile evaluate     # fixed-start evaluation; no weight updates
./run.sh --profile gui_watch    # visible policy playback
./run.sh --profile fast_watch   # one headless playback worker
./run.sh --profile setup        # manual setup and RAM diagnostics
```

Training checkpoints are saved to `<POKEMON_AI_MODEL_NAME>.zip` at the project root and every configured checkpoint interval to `runtime_data/checkpoints/`. The installer turns resume on automatically when the matching root checkpoint already exists, so the default training command does not accidentally refuse to continue or overwrite learned weights. To deliberately start a fresh experiment, choose a **new** `POKEMON_AI_MODEL_NAME` in `.env`; do not delete the only copy of a model just to switch objectives.

`POKEMON_AI_TOTAL_TIMESTEPS` is the number of transitions for the next `learn()` call. When resuming, the configured value is additional training rather than a total cap from timestep zero. The code keeps PPO's recurrent observation/action architecture stable across objectives so a compatible recurrent checkpoint can be continued.

## Curriculum objectives

Set `POKEMON_AI_OBJECTIVE` in `.env` (or export it for a one-run override):

| Value | Episode success condition | Intended use |
|---|---|---|
| `starter` | At least one live party Pokémon with a valid level | Current beginner objective |
| `pokedex` | A party member exists and the Pokédex-acquired story flag is set | Continue from the same early-game start toward Pokédex progression |
| `first_badge` | At least one badge is detected | A much longer next-stage goal; only use after starter acquisition is reliable |

Example command-line override, without editing the code or permanently changing `.env`:

```bash
POKEMON_AI_OBJECTIVE=pokedex ./run.sh --profile fast_train
```

The start-state validator is objective-aware: `starter` requires the empty-party Route 101 start; `pokedex` requires a verified starter milestone; `first_badge` requires a party and Pokédex milestone. The automatic controller uses these saved stages instead of making you edit Python source or replay the introduction. Later objectives are significantly harder; supporting an objective is not evidence that the policy has learned it.

## Automatic curriculum (optional)

After installing the update, preview the manager without launching an emulator:

```bash
.venv/bin/python run_curriculum.py --dry-run
```

When you are ready for the managed run:

```bash
.venv/bin/python run_curriculum.py
```

The controller first evaluates the existing checkpoint, so it does not immediately spend another training budget if the current policy already passes. If the stage fails, it screens up to `POKEMON_AI_CURRICULUM_COMPARE_CHECKPOINTS` recent, valid, content-distinct archived checkpoints when available, fully evaluates a promising candidate, and only then activates it if its full-test score improves. Any prior root checkpoint is backed up before a candidate is activated. The controller then trains in bounded chunks and re-evaluates. When a stage passes the configured evaluation gate, it requires a **freshly saved milestone savestate plus matching JSON metadata** before promoting to the next stage. The order is `starter` → `pokedex` → `first_badge`. It stores resumable progress at `runtime_data/curriculum_state.json`, logs to `logs/curriculum_manager_*.log`, and preserves the pre-curriculum model under `runtime_data/checkpoints/`. Interrupt with Ctrl+C; the active trainer's own shutdown handler should save its checkpoint before exiting. Re-run the manager to re-evaluate the saved stage and continue.

The default gate is a deterministic fixed-start acceptance check across five episodes (80% threshold). Because those episodes begin from the same exact savestate, a deterministic policy can repeat the same trajectory; this is a **reproducibility gate**, not a claim of five independent test scenarios. Use the separate stochastic evaluation command to assess action-sampling variability. The manager refuses stage promotion if evaluation is incomplete, reports are missing or stale, the selected checkpoint or its SHA-1 doesn't match the report, the milestone's recorded evaluated-checkpoint identity differs, the ROM hash differs, or a fresh milestone state and manifest aren't created during the qualifying evaluation.

### Soft exploration mini-episodes

For the longer `pokedex` and `first_badge` objectives, `POKEMON_AI_MINI_EPISODE_STEPS=5000` starts a soft exploration window every 5,000 policy decisions during training, evaluation, and watch. At a boundary, it clears only the short-term visited tile/map/screen memory and loop-detection history, then seeds the current map/tile/screen so a stationary frame does not immediately earn duplicate discovery reward. The emulator and its story progress are untouched; Gymnasium does not receive `terminated`/`truncated`; episode return/component totals, one-time story/trainer reward guards, lifetime novelty decay, and recurrent LSTM state continue uninterrupted. Whole-Gym-episode unique-tile/map/screen metrics remain cumulative. This is not a real Gymnasium mini-episode or a model-state reset. Set the interval to `0` to disable it. The starter objective is not affected. It is intentionally simple and must still receive a live mGBA smoke test before relying on longer-stage performance.

## Evaluation and watch

Deterministic evaluation is the default and is useful for checking the action-argmax policy. Stochastic evaluation samples the learned distribution and is useful when deterministic selection loops. These answer different questions; report both rather than treating a single successful episode as proof of reliability.

```bash
# Five deterministic episodes (default)
./run.sh --profile evaluate

# Five stochastic episodes, seeded for repeatability
POKEMON_AI_EVAL_DETERMINISTIC=0 POKEMON_AI_EVAL_EPISODES=5 ./run.sh --profile evaluate

# Visible stochastic playback
POKEMON_AI_STOCHASTIC_WATCH=1 ./run.sh --profile gui_watch
```

`POKEMON_AI_EVAL_SEED` defaults to `1234`; it seeds the evaluation's sampling RNG without changing model weights. Evaluation reports success, return, steps, party count, party levels, live/save-block party counts and action counts. A result is still a sample from the policy, not a guarantee of future behavior.

Watch mode reports the objective transition and then continues for `POKEMON_AI_WATCH_SUCCESS_GRACE_STEPS` additional decisions (default 300) so the screen can advance into the starter battle; set it to `0` for immediate stopping. The live-party reader should report a starter as soon as it exists in the active party; a `[Party RAM Diagnostic]` line is printed if the live count differs from the saved party copy. Don't interpret a periodic diagnostic from before the transition as the final episode state.

## Rewards and monitoring

The environment keeps small exploration rewards and event-based milestone rewards. Current defaults include a small per-action cost, decaying new-tile/map/screen bonuses, one-time story-flag rewards, party/level/badge/trainer rewards, a loop penalty, and a terminal objective bonus. These are learning signals, not proof of route mastery.

TensorBoard starts at `http://127.0.0.1:6006/` by default. The warning `TensorFlow installation not found - running with reduced feature set` does not by itself mean the server failed; TensorFlow is not required just to view standard training scalars. Useful metrics include:

- `progress/task_success_rate_last_100_episodes`
- `progress/completed_episode_reward_mean_last100`
- `progress/completed_episode_length_mean_last100`
- `progress/completed_episode_<component>_sum_mean_last100`
- `progress/completed_episode_<component>_sum_this_step`
- `progress/transitions_per_second`
- `progress/mini_episode_count` and `progress/mini_episode_boundary` (later-stage soft exploration windows)
- `progress/completed_episodes_this_run`, `progress/task_successes_this_run`, and `progress/timeouts_this_run` (explicitly scoped to this process so resumed runs do not masquerade as lifetime totals)

The `completed_episode_*` metrics use completed-episode totals only. TensorBoard may continue displaying old historical tags such as `progress/episode_party_sum` from earlier runs; those old points are not retroactively fixed. Start a fresh TensorBoard run/log directory for a clean comparison if the old and new metric names become confusing.

## Tests and limitations

```bash
.venv/bin/python -m py_compile main.py tests/test_main_logic.py
.venv/bin/python -m unittest discover -s tests -v
```

The unit tests use fake RAM maps and temporary ZIP checkpoints; they don't launch the real game. A live smoke test against the user's ROM/mGBA build is still necessary after installing any emulator-facing changes. This environment was reviewed from source and statically compiled, but it cannot prove the native mGBA memory binding against your local ROM or model.

## Design choices informed by other Pokémon RL projects

Other active Pokémon RL projects use modular curriculum targets, event/flag rewards, recurrent policies, automated checkpoints, evaluation harnesses and explicit loop/failure handling. This update adopts the lower-risk subset that fits the existing code: live-state reward conditions, named milestones, reproducible evaluation seeds, terminal-only metrics, bounded checkpoint history and regression tests. It deliberately does **not** add an untested online best-model evaluator or change PPO's architecture/hyperparameters automatically; those changes could be expensive or invalidate the current checkpoint.


## Reference implementation notes

The live-party distinction follows the canonical `pret/pokeemerald` save code: `SavePlayerParty()` copies `gPlayerPartyCount`/`gPlayerParty` into the SaveBlock at save time, while `LoadPlayerParty()` copies the SaveBlock back into those live globals ([source](https://github.com/pret/pokeemerald/blob/master/src/load_save.c)). Stable-Baselines3's `Monitor` reports definitive episode reward/length at terminal steps ([documentation](https://stable-baselines3.readthedocs.io/en/master/common/monitor.html)); that is the boundary used by the corrected episode metrics.

The curriculum and anti-loop direction is consistent with community Pokémon RL projects such as [pokemon-red-ai](https://github.com/amcheste/pokemon-red-ai) (hybrid observations, recurrent PPO, event rewards, evaluation and statistical analysis) and [PokeRL](https://github.com/reddheeraj/PokemonRL) (staged tasks and loop-aware behavior). This project remains Pokémon Emerald-specific; these projects are design references, not drop-in dependencies.
