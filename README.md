# Pokémon Emerald PPO Agent

A Gymnasium environment for Pokémon Emerald driven by the mGBA Python core and Stable-Baselines3 PPO. The application can train a shared PPO policy across separate emulator instances or load that policy in watch mode.

## Setup and launch

This launcher is a convenience script; it is not required. The script included with this bundle uses the existing `.venv` and your mGBA native build. It does not reinstall dependencies or rebuild mGBA.

1. Put your legally obtained compatible Pokémon Emerald ROM beside `main.py`, named `PokemonEmerald.gba`. This project expects game code `BPEE`, revision 0.
2. Make the launcher executable and run it:

   ```bash
   chmod +x run.sh
   ./run.sh
   ```

The launcher uses `.venv/bin/python`, adds the mGBA build directory to `LD_LIBRARY_PATH`, and forwards command-line arguments to `main.py`. It first looks for `/home/fahim/mgba/build-py311`, then a project-local `.runtime/mgba-build-py311`. To select another build, set `MGBA_BUILD_DIR=/path/to/build-py311`.

You can also launch manually, but you must activate the virtual environment and set `LD_LIBRARY_PATH` yourself:

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="/home/fahim/mgba/build-py311${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -u main.py --profile fast_train
```

## One-setting profile switching

You only need **one active `.env` file**. Change `POKEMON_AI_PROFILE` in it, or pass `--profile NAME` to `./run.sh` (or `python main.py`). You no longer need to copy among `.env.fast`, `.env.gui-watch`, or `.env.watch-headless`; those older profile files are not used automatically. After updating, you may delete those old templates with `rm -f .env.fast .env.gui-watch .env.watch-headless` if you want to keep the project folder tidy.

| Profile | What it does | Emulator instances |
|---|---|---:|
| `fast_train` | **Default.** Headless PPO training, with TensorBoard enabled. | 4 by default (configurable) |
| `gui_train` | Train while showing the game. | 1 |
| `gui_watch` | Load the trained policy and watch it in a GUI; recording is enabled by default. | 1 |
| `fast_watch` | Load the trained policy and run headlessly at maximum practical speed; no video or TensorBoard server. | 1 |

### Easiest way: command-line profile

Use `./run.sh` so it sets up `LD_LIBRARY_PATH` automatically and forwards the profile argument:

```bash
# Default fast training: four headless environments
./run.sh --profile fast_train

# Training with a visible game window
./run.sh --profile gui_train

# Watch the trained policy in a GUI
./run.sh --profile gui_watch

# Watch headlessly at maximum speed with one emulator
./run.sh --profile fast_watch
```

`--profile NAME` forces that profile's mode/gui/watch/video/TensorBoard defaults for this launch. Alternatively, edit the single line in `.env` and run `./run.sh` without an argument. The `.env` profile applies at startup; explicit settings exported in your shell can override it when you select the profile in `.env`.

### Optional: edit `.env` instead

For the normal default, leave this line as:

```dotenv
POKEMON_AI_PROFILE=fast_train
```

Change only the value to `gui_train`, `gui_watch`, or `fast_watch`, save the file, and run `./run.sh`. To return to the default, set it back to `fast_train`.

`POKEMON_AI_NUM_ENVS=4` controls the worker count for headless training; try `2`, `4`, or `6` to benchmark your CPU. GUI training and both watch profiles deliberately use one environment, regardless of that setting. Headless watch uses `POKEMON_AI_WATCH_FPS=0` (unlimited frame cap). Fast-forward makes simulation run faster; it does not make the policy reason better per action.

## Runtime files and saves

Yes: emulator state files and temporary/intermediate files are kept **outside the project root**, under `runtime_data/`:

- `runtime_data/states/` — shared/worker `.state` savestates, episode-start state files, and atomic state-write temporary files.
- `runtime_data/tmp/` — temporary model archives, temporary video/WAV/mux files, and diagnostic screenshots.

The `.sav` cartridge save and the `.zip` PPO model checkpoint are separate from emulator state snapshots. Existing root-level legacy state/temp files are migrated at startup when safe; if a destination conflict exists, the old file is preserved under a `legacy_imports/` subfolder rather than overwritten.

## Other settings

| Variable | Default | Purpose |
|---|---:|---|
| `POKEMON_AI_VERBOSE` | `1` | Stable-Baselines3 logging: `0` quiet, `1` normal, `2` debug. Use `1` for useful progress. |
| `POKEMON_AI_MGBA_LOG_LEVEL` | `7` | Native mGBA severity mask (fatal/errors/warnings). Independent of SB3 verbosity. |
| `POKEMON_AI_FILTER_MGBA_CHATTER` | `1` | Filter routine BIOS SWI/DMA C stdout traces. Set `0` for full native output. |
| `POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS` | `1` | Also filter only the repeated DMA source-zero and listed unused video-register messages seen with this setup. Set `0` to inspect them. Other warnings remain visible. |
| `POKEMON_AI_ACTION_REPEAT` | `2` | Number of GBA frames per selected action. Keep `2` when continuing a checkpoint trained with this action timing. |
| `POKEMON_AI_TOTAL_TIMESTEPS` | `1000000` | Additional PPO training transitions requested in a training call. This is aggregate transitions across vectorized environments. |
| `POKEMON_AI_CHECKPOINT_EVERY` | `25000` | Transitions between periodic model and emulator-state checkpoints. |
| `POKEMON_AI_PROGRESS_EVERY` | `1000` | Transitions between console progress summaries. This is not the save interval. |
| `POKEMON_AI_AUTO_START_TENSORBOARD` | Profile-controlled | Training profiles start TensorBoard; watch profiles don't. An explicitly exported shell value can override when using the `.env` profile. |
| `POKEMON_AI_TENSORBOARD_PORT` | `6006` | Local TensorBoard port. |
| `OMP_NUM_THREADS` | `2` | Optional numerical-library thread-pool limit. The best values depend on the workload; benchmark. |
| `MKL_NUM_THREADS` | `2` | Optional numerical-library thread-pool limit. |
| `OPENBLAS_NUM_THREADS` | `1` | Optional numerical-library thread-pool limit. |
| `NUMEXPR_NUM_THREADS` | `1` | Optional numerical-library thread-pool limit. |

For your 6-core Ryzen 5 5600G, begin with four workers, then compare `POKEMON_AI_NUM_ENVS=2`, `4`, and `6` using the reported aggregate transitions/second. More workers can increase memory use or IPC overhead, so more is not automatically faster. To preserve a custom count while choosing a profile, set `POKEMON_AI_NUM_ENVS` in `.env`; the profile does not overwrite this tunable worker-count setting.

## Reward and progress output

Console reward values are **custom reinforcement-learning rewards**, not Pokémon money or an in-game score:

- `step_reward`: reward from the last transition.
- `mean_reward_1k` / `recent_mean_reward`: recent average reward.
- `live_episode_reward`: rewards accumulated so far in the current episode.
- `last_episode_reward`: most recently completed episode return.

Reward is only one signal. Map changes, party levels, and badge changes are more interpretable game-progress indicators, and increasing reward alone does not prove that the agent is completing the game.

A PPO transition is one policy decision/environment step, not one GBA frame. In parallel training, SB3's timestep counter aggregates transitions from all environments.

## Saves and where files are stored

The code keeps runtime-only state and temporary artifacts out of the project root, under `runtime_data/states/` and `runtime_data/tmp/` respectively:

| Path | Purpose |
|---|---|
| `emerald_ppo_checkpoint.zip` | PPO model and optimizer checkpoint; kept at root for compatibility with existing runs. |
| `emerald_ppo_checkpoint.zip.bak` | Previous model archive backup. |
| `PokemonEmerald.gba` | Your ROM; not included with this project. |
| `PokemonEmerald.sav` | Persistent **cartridge save**, next to the ROM. In single-instance watch mode, the code creates/attaches it if missing when the mGBA binding supports explicit save-file loading. In training, no shared `.sav` is written because workers must not overwrite one another's save. |
| `runtime_data/states/emerald_save.state` | Emulator savestate. This resumes the full emulated machine state; it is different from an in-game cartridge save. |
| `runtime_data/states/emerald_save.worker_N.state` | Separate savestate for each parallel training worker. |
| `runtime_data/states/` | Episode-start fallback state files and temporary state-write files. |
| `runtime_data/tmp/` | Temporary model archives, video intermediates, WAV audio, and mux files. |

### Checkpoint filename and recovery

The final PPO model is `emerald_ppo_checkpoint.zip` in the project root. During a save, the application writes an explicit ZIP file under `runtime_data/tmp/` and only publishes the final file after checking that it is a valid SB3-style ZIP. A normal successful save leaves no temporary model archive behind.

If an older version left a root-level `emerald_ppo_checkpoint.tmp` and the final `.zip` is missing, do not delete the `.tmp` file until it has been checked. Stop the old process gracefully with `Ctrl+C`, then run the recovery check below. It only copies a valid ZIP with an SB3 `data` entry, and never overwrites an existing final checkpoint:

```bash
python - <<'PY'
from pathlib import Path
import shutil, zipfile
source = Path("emerald_ppo_checkpoint.tmp")
final = Path("emerald_ppo_checkpoint.zip")
if not source.is_file():
    raise SystemExit(f"Not found: {source}")
if not zipfile.is_zipfile(source):
    raise SystemExit("The .tmp file is not a ZIP archive; keep it for investigation.")
with zipfile.ZipFile(source) as zf:
    names = set(zf.namelist())
    bad = zf.testzip()
    if bad is not None or "data" not in names:
        raise SystemExit(f"Archive validation failed (bad member={bad!r}); kept original.")
destination = final if not final.exists() else Path("emerald_ppo_checkpoint.recovered.zip")
if destination.exists():
    raise SystemExit(f"Refusing to overwrite existing file: {destination}")
shutil.copy2(source, destination)
print(f"Recovered validated model archive to: {destination}")
PY
```

After recovery, install the corrected `main.py` before starting another training run. The corrected saver passes a path ending in `.zip` to Stable-Baselines3 and verifies the same path, preventing suffix/path mismatches.
| `videos/` | Final video recordings. |
| `tensorboard_logs/` | TensorBoard event data. |
| `logs/run_*.log` | Per-run console logs. |
| `logs/tensorboard_server.log` | TensorBoard server log. |
| `logs/session_time_log.txt` | Session duration log. |
| `.runtime/` | mGBA source/build artifacts created by the setup. |
| `.venv/` | Python environment. |

On first launch after this update, old root-level `emerald_save.state`, worker-state files, and `session_time_log.txt` are migrated into the new folders if their destinations don't already exist. Migration copies and verifies before publishing the new file and only removes the old copy after success. If a new destination already exists, the old file is left untouched.

**`.sav` versus `.state`:** a `.sav` stores Pokémon's in-game save data when the game itself saves. A `.state` stores a snapshot of the live emulator, including the exact point in gameplay. Training uses isolated temporary save data and relies on the PPO checkpoint plus per-worker emulator states; it should not write one shared cartridge `.sav` from multiple workers. For normal persistent in-game saves, run watch mode with a single emulator.

## Native mGBA warnings

`GBA Video: Invalid video register: 0x04E` (and `0x056`–`0x05E`) means the emulated game wrote to a video-I/O address mGBA doesn't recognize as a valid register. mGBA's developers note that game-error logs can include questionable things games do even when the game continues to run. The `GBA DMA: Invalid DMA source address: 0x00000000` message means a DMA transfer requested source address zero, which mGBA reports as invalid. If Pokémon Emerald renders and plays normally, these messages alone aren't proof of a fatal problem; investigate if they accompany a black screen, freeze, incorrect gameplay, or crashes. The default optional filter hides only these specific repeated messages to keep the terminal readable. Set `POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS=0` to show them again. [mGBA issue discussion about invalid video-register game-error messages](https://github.com/mgba-emu/mgba/issues/3578).

## TensorBoard

TensorBoard is a graph dashboard for measurable training metrics, not a transcript of the model's private thoughts. With the default local server, open <http://127.0.0.1:6006> while training and select **Scalars**. It can show episode reward/length, PPO losses, throughput, and custom `progress/*` metrics. A server can be online before the first training event; graphs become useful when training logs are written.

The application-managed server shuts down with the program; logs persist under `tensorboard_logs/`. To inspect prior graphs afterwards, start it manually:

```bash
source .venv/bin/activate
tensorboard --logdir tensorboard_logs --host 127.0.0.1 --port 6006
```

## Controls and shutdown

| Input | Action |
|---|---|
| `Q` or `Esc` in GUI | Request graceful stop and save/cleanup |
| `P` in GUI | Pause/resume |
| `M` in GUI | Mute/unmute live audio |
| `W/A/S/D` or arrow keys | Movement in watch mode if host controls are enabled in `main.py` |
| `Z` / `Enter` | A button |
| `X` / `Backspace` | B button |
| `1` / `2` | Start / Select |
| `Ctrl+C` | Graceful stop in GUI or headless mode |

Use a graceful stop rather than `kill -9`. The program attempts to save the PPO model and emulator state on periodic checkpoints and normal shutdown, but no application can guarantee zero lost work after a power failure, OS crash, or forced kill.

## Troubleshooting

- **mGBA import/shared-library error:** run `./setup.sh` then `./verify_setup.py`; don't mix a Python binding and `libmgba.so` from different builds.
- **First-frame hang:** use `./run.sh` so the native library path and pre-PyTorch mGBA warm-up are retained.
- **OpenCV GUI fails on Wayland:** the program selects the bundled XCB plugin if OpenCV's Qt wheel has no Wayland plugin. XWayland and its XCB libraries must still be installed. Try `QT_QPA_PLATFORM=xcb ./run.sh` if necessary.
- **CPU use looks low:** GUI mode deliberately paces one emulator near 60 FPS. Use headless training with four workers or headless watch with `POKEMON_AI_WATCH_FPS=0` for maximum simulation speed.
- **RAM grows:** the program uses bounded audio/reward buffers and closes each environment on shutdown, but a long profiling run is needed to rule out native-library leaks. Logs, event files, states and recordings are disk storage, not RAM.
- **Headless watch has no video:** video is off by default for maximum speed. Set `POKEMON_AI_RECORD_VIDEO=1` when you need a recording; it will add overhead.

Do not redistribute `PokemonEmerald.gba` or original cartridge-save data. Keep generated logs, TensorBoard events, videos and checkpoints out of source control unless you deliberately intend to share them.
