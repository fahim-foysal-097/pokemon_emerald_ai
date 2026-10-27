import atexit
import os
import signal
import sys
import subprocess
import time
import glob
import filecmp
import importlib
import importlib.util
import ctypes
import faulthandler
import queue
import shutil
import socket
import threading
import wave
import zipfile
from collections import deque


# Track environment provenance so the selected profile can replace stale mode settings
# from older .env files while still respecting explicitly exported shell variables.
_SHELL_ENV_KEYS_AT_START = set(os.environ)
_PROJECT_ENV_KEYS_LOADED = set()


def _load_project_env_file():
    """Load simple KEY=VALUE settings from project .env before reading os.environ.

    Existing shell/environment variables win, so one-off commands can override .env.
    This intentionally supports a small, dependency-free .env format (blank lines,
    comments, optional `export`, and single/double-quoted values).
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return

    loaded_keys = []
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or not (key[0].isalpha() or key[0] == "_"):
                    continue
                if not all(char.isalnum() or char == "_" for char in key):
                    continue
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                # Like python-dotenv's default behavior: shell exports take precedence.
                if key not in os.environ:
                    os.environ[key] = value
                    loaded_keys.append(key)
                    _PROJECT_ENV_KEYS_LOADED.add(key)
        if loaded_keys:
            print(
                f"[Config] Loaded {len(loaded_keys)} setting(s) from {env_path}; "
                "existing shell variables take precedence.",
                flush=True,
            )
    except OSError as exc:
        print(f"[Config Warning] Could not read project .env file: {exc}", flush=True)


_load_project_env_file()

# One profile now controls the common mode combination. You can choose it by editing
# POKEMON_AI_PROFILE in .env or by passing --profile NAME to main.py/run.sh.
_PROFILE_PRESETS = {
    "fast_train": {
        "label": "Fast training (headless, four environments)",
        "env": {
            "POKEMON_AI_MODE": "train",
            "POKEMON_AI_GUI_MODE": "0",
            "POKEMON_AI_WATCH_FPS": "0",
            "POKEMON_AI_RECORD_VIDEO": "0",
            "POKEMON_AI_AUTO_START_TENSORBOARD": "1",
        },
    },
    "gui_train": {
        "label": "Training with a visible game (one environment)",
        "env": {
            "POKEMON_AI_MODE": "train",
            "POKEMON_AI_GUI_MODE": "1",
            "POKEMON_AI_WATCH_FPS": "0",
            "POKEMON_AI_RECORD_VIDEO": "0",
            "POKEMON_AI_AUTO_START_TENSORBOARD": "1",
        },
    },
    "setup": {
        "label": "Human-controlled gameplay setup and training-start capture",
        "env": {
            "POKEMON_AI_MODE": "setup",
            "POKEMON_AI_GUI_MODE": "1",
            "POKEMON_AI_WATCH_FPS": "0",
            "POKEMON_AI_RECORD_VIDEO": "0",
            "POKEMON_AI_AUTO_START_TENSORBOARD": "0",
        },
    },
    "gui_watch": {
        "label": "Watch the trained policy in a GUI (one environment)",
        "env": {
            "POKEMON_AI_MODE": "watch",
            "POKEMON_AI_GUI_MODE": "1",
            "POKEMON_AI_WATCH_FPS": "0",
            "POKEMON_AI_RECORD_VIDEO": "1",
            "POKEMON_AI_AUTO_START_TENSORBOARD": "0",
        },
    },
    "fast_watch": {
        "label": "Headless watch at maximum speed (one environment)",
        "env": {
            "POKEMON_AI_MODE": "watch",
            "POKEMON_AI_GUI_MODE": "0",
            "POKEMON_AI_WATCH_FPS": "0",
            "POKEMON_AI_RECORD_VIDEO": "0",
            "POKEMON_AI_AUTO_START_TENSORBOARD": "0",
        },
    },
}
_PROFILE_ALIASES = {
    "default": "fast_train",
    "fast": "fast_train",
    "fast_training": "fast_train",
    "training": "fast_train",
    "training_gui": "gui_train",
    "gui_training": "gui_train",
    "visible_training": "gui_train",
    "human_setup": "setup",
    "setup_gui": "setup",
    "training_setup": "setup",
    "watch_gui": "gui_watch",
    "gui_watching": "gui_watch",
    "watch_headless": "fast_watch",
    "headless_watch": "fast_watch",
    "fast_forward": "fast_watch",
}


def _apply_runtime_profile():
    """Apply a named profile and consume optional --profile CLI arguments."""
    cli_profile = None
    retained_args = [sys.argv[0]]
    args = list(sys.argv[1:])
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--profile":
            if i + 1 >= len(args):
                raise SystemExit("Missing profile name after --profile. Choices: " + ", ".join(_PROFILE_PRESETS))
            if cli_profile is not None:
                raise SystemExit("Specify --profile only once.")
            cli_profile = args[i + 1]
            i += 2
            continue
        if arg.startswith("--profile="):
            if cli_profile is not None:
                raise SystemExit("Specify --profile only once.")
            cli_profile = arg.split("=", 1)[1]
            i += 1
            continue
        retained_args.append(arg)
        i += 1
    sys.argv[:] = retained_args

    requested = cli_profile if cli_profile is not None else os.environ.get("POKEMON_AI_PROFILE", "fast_train")
    normalized = requested.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in _PROFILE_PRESETS:
        choices = ", ".join(f"{name} ({item['label']})" for name, item in _PROFILE_PRESETS.items())
        raise SystemExit(f"Unknown profile {requested!r}. Available profiles: {choices}")

    # A direct CLI --profile selection is explicit and forces the preset. When the
    # profile is selected in .env, variables exported by the caller's shell retain
    # precedence, while older mode values loaded from .env are replaced by the profile.
    for key, value in _PROFILE_PRESETS[normalized]["env"].items():
        if cli_profile is not None or key not in _SHELL_ENV_KEYS_AT_START:
            os.environ[key] = value
    os.environ["POKEMON_AI_PROFILE"] = normalized

    if __name__ == "__main__":
        source = "--profile argument" if cli_profile is not None else "POKEMON_AI_PROFILE/.env default"
        print(f"[Profile] {normalized}: {_PROFILE_PRESETS[normalized]['label']} (selected via {source}).", flush=True)
        if cli_profile is None:
            shell_overrides = sorted(
                key for key in _PROFILE_PRESETS[normalized]["env"]
                if key in _SHELL_ENV_KEYS_AT_START
            )
            if shell_overrides:
                print(
                    "[Profile] Explicit shell overrides retained for: " + ", ".join(shell_overrides),
                    flush=True,
                )


_apply_runtime_profile()

# Some mGBA Python/native builds bypass mLogSetDefaultLogger and fall back to
# direct C printf() calls. In that case the mGBA severity mask cannot catch the
# repetitive BIOS-SWI/DMA trace lines. As a robust fallback, filter only those
# known routine trace lines at the process stdout file-descriptor level; all
# Python output and other native messages continue to the terminal unchanged.
_NATIVE_STDOUT_FILTER_STATE = None


def _is_routine_mgba_chatter(line):
    """Identify noisy mGBA traces and optionally known, non-fatal game-error spam.

    The optional game-warning filter is deliberately narrow: it hides only DMA reads
    from address zero and the specific unused video-register writes seen with this
    ROM/build. All other invalid/error/warning messages remain visible.
    """
    stripped = line.lstrip()
    lowered = stripped.lower()

    # mGBA's standard logger may prefix the same category with tags such as
    # [GAME ERROR]. Normalize one leading bracketed tag for the narrowly scoped
    # warning filter; keep the original line for all other decisions and output.
    payload = stripped
    if payload.startswith(b"["):
        tag_end = payload.find(b"] ")
        if 0 < tag_end <= 32:
            payload = payload[tag_end + 2:].lstrip()

    filter_known_warnings = os.environ.get(
        "POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if filter_known_warnings:
        if payload.startswith(b"GBA DMA: Invalid DMA source address: 0x00000000"):
            return True
        known_unused_video_registers = {
            b"0X04E", b"0X056", b"0X058", b"0X05A", b"0X05C", b"0X05E"
        }
        prefix = b"GBA Video: Invalid video register: "
        if payload.startswith(prefix):
            register = payload[len(prefix):].split(None, 1)[0].rstrip(b"\r\n")
            if register.upper() in known_unused_video_registers:
                return True

    if any(word in lowered for word in (b"error", b"invalid", b"failed", b"fatal", b"warning")):
        return False
    return (
        payload.startswith(b"GBA BIOS: SWI:")
        or payload.startswith(b"GBA DMA: Starting DMA")
        or payload.startswith(b"GBA Serial I/O: GPIO write: Unhandled ")
    )


def _install_native_stdout_filter():
    """Filter noisy mGBA C stdout traces without swallowing useful program output.

    This is deliberately narrower than redirect_stdout(): native printf() bypasses
    Python's sys.stdout. A small pipe reader filters complete output lines at fd 1,
    forwarding everything except known routine BIOS/DMA chatter to a duplicated
    copy of the original terminal descriptor. This is enabled by default and can
    be disabled with POKEMON_AI_FILTER_MGBA_CHATTER=0 for native debugging.
    """
    global _NATIVE_STDOUT_FILTER_STATE
    enabled = os.environ.get("POKEMON_AI_FILTER_MGBA_CHATTER", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"} or not sys.platform.startswith("linux"):
        return False
    if _NATIVE_STDOUT_FILTER_STATE is not None:
        return True

    try:
        # Flush Python and C stdio before redirecting the underlying descriptor.
        try:
            sys.stdout.flush()
        except Exception:
            pass
        libc = ctypes.CDLL(None)
        try:
            libc.fflush.argtypes = [ctypes.c_void_p]
            libc.fflush.restype = ctypes.c_int
            libc.fflush(None)
            # Keep native printf output line-buffered after stdout becomes a pipe.
            stdout_ptr = ctypes.c_void_p.in_dll(libc, "stdout").value
            if stdout_ptr:
                libc.setvbuf.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
                libc.setvbuf.restype = ctypes.c_int
                libc.setvbuf(ctypes.c_void_p(stdout_ptr), None, 1, 0)  # _IOLBF on POSIX
        except Exception:
            pass

        original_fd = os.dup(1)
        read_fd, write_fd = os.pipe()
        try:
            os.dup2(write_fd, 1)
        finally:
            os.close(write_fd)

        def write_all(fd, data):
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    break
                view = view[written:]

        def relay_output():
            pending = b""
            try:
                while True:
                    chunk = os.read(read_fd, 8192)
                    if not chunk:
                        break
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        line_with_newline = line + b"\n"
                        if not _is_routine_mgba_chatter(line):
                            write_all(original_fd, line_with_newline)
                if pending and not _is_routine_mgba_chatter(pending):
                    write_all(original_fd, pending)
            except (OSError, ValueError):
                pass

        relay_thread = threading.Thread(
            target=relay_output,
            name="mgba-stdout-filter",
            daemon=True,
        )
        relay_thread.start()
        _NATIVE_STDOUT_FILTER_STATE = {
            "original_fd": original_fd,
            "read_fd": read_fd,
            "thread": relay_thread,
        }

        def restore_stdout_filter():
            global _NATIVE_STDOUT_FILTER_STATE
            state = _NATIVE_STDOUT_FILTER_STATE
            if not state:
                return
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                os.dup2(state["original_fd"], 1)
            except OSError:
                pass
            # The relay is a daemon: don't risk hanging shutdown if spawned workers
            # still hold inherited pipe descriptors. Give it a brief chance to drain.
            state["thread"].join(timeout=0.35)
            try:
                os.close(state["original_fd"])
            except OSError:
                pass
            _NATIVE_STDOUT_FILTER_STATE = None

        atexit.register(restore_stdout_filter)
        print(
            "[mGBA Console] Filtering routine BIOS/DMA traces; known zero-source DMA and selected unused-video-register "
            "messages are filtered by POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS=1. "
            "Set POKEMON_AI_FILTER_MGBA_CHATTER=0 to show all native stdout.",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[mGBA Console Warning] Could not install native stdout filter: {exc}", flush=True)
        return False


_install_native_stdout_filter()


def _configure_opencv_qt_backend():
    """Prefer the bundled XCB plugin when a pip OpenCV wheel lacks Wayland support.

    OpenCV's pip wheels commonly ship Qt's XCB platform plugin but not the Wayland
    plugin. On Wayland desktops this can make namedWindow() fail before any GUI
    window exists. XCB works through XWayland, when it is installed and running.
    This must run before importing cv2 because cv2 initializes its Qt plugin paths
    during import.
    """
    if not sys.platform.startswith("linux"):
        return

    try:
        spec = importlib.util.find_spec("cv2")
        if spec is None or not spec.origin:
            return

        cv2_dir = os.path.dirname(os.path.abspath(spec.origin))
        bundled_plugin_root = os.path.join(cv2_dir, "qt", "plugins")
        bundled_platforms = os.path.join(bundled_plugin_root, "platforms")
        xcb_plugin = os.path.join(bundled_platforms, "libqxcb.so")
        bundled_wayland_plugins = (
            "libqwayland-egl.so",
            "libqwayland-generic.so",
            "libqwayland.so",
        )
        has_bundled_xcb = os.path.isfile(xcb_plugin)
        has_bundled_wayland = any(
            os.path.isfile(os.path.join(bundled_platforms, name))
            for name in bundled_wayland_plugins
        )

        requested = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
        on_wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or (
            os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        )
        wants_wayland = requested in {"wayland", "wayland-egl", "wayland-xcomposite-egl"} or (
            not requested and on_wayland
        )

        # Respect a custom plugin path if it actually contains a Wayland plugin.
        custom_root = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "").strip()
        custom_wayland_available = False
        if custom_root:
            custom_candidates = [custom_root, os.path.join(custom_root, "platforms")]
            custom_wayland_available = any(
                os.path.isfile(os.path.join(root, name))
                for root in custom_candidates
                for name in bundled_wayland_plugins
            )

        if (
            wants_wayland
            and has_bundled_xcb
            and not has_bundled_wayland
            and not custom_wayland_available
        ):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", bundled_plugin_root)
            print(
                "[GUI] OpenCV's bundled Qt has no Wayland plugin; selecting XCB/XWayland. "
                "If window creation still fails, install/enable xorg-xwayland and XCB dependencies.",
                flush=True,
            )
    except Exception as exc:
        # This is a compatibility convenience; let cv2 produce the definitive error if needed.
        print(f"[GUI Warning] Could not auto-select an OpenCV Qt backend: {exc}", flush=True)


_configure_opencv_qt_backend()

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- Native mGBA log filtering ---
# mGBA native severity filtering is independent of Stable-Baselines3's verbosity.
# The default mask 7 keeps fatal/errors/warnings (1|2|4). A descriptor-level filter
# installed above also catches routine C printf traces if this binding bypasses the
# standard mGBA logger. Override the mask with POKEMON_AI_MGBA_LOG_LEVEL=0..127.
_MGBA_LOGGER_STORAGE = None
_MGBA_LOG_LEVEL_MASK = int(os.environ.get("POKEMON_AI_MGBA_LOG_LEVEL", "7"))
if not 0 <= _MGBA_LOG_LEVEL_MASK <= 127:
    raise ValueError("POKEMON_AI_MGBA_LOG_LEVEL must be an integer from 0 to 127")


def _configure_mgba_native_logging():
    """Install an mGBA standard logger so native messages obey a useful severity mask."""
    global _MGBA_LOGGER_STORAGE
    try:
        pylib = importlib.import_module("mgba._pylib")
        ffi, lib = pylib.ffi, pylib.lib
        required_symbols = ("mStandardLoggerInit", "mLogSetDefaultLogger")
        if not all(hasattr(lib, symbol) for symbol in required_symbols):
            raise RuntimeError("this mGBA binding does not expose the standard logger API")

        logger = ffi.new("struct mStandardLogger *")
        lib.mStandardLoggerInit(logger)
        # Keep filtered messages eligible for the standard logger, but only show
        # fatal/error/warning severities by default (mask 7); no log file is opened.
        logger.logToStdout = True
        logger.logFile = ffi.NULL
        logger.d.filter.defaultLevels = _MGBA_LOG_LEVEL_MASK
        lib.mLogSetDefaultLogger(ffi.addressof(logger.d))
        # Retain both the binding and allocated C logger for the whole process life.
        _MGBA_LOGGER_STORAGE = (pylib, logger)
        print(
            f"[mGBA Logging] Native severity mask={_MGBA_LOG_LEVEL_MASK}; "
            "routine BIOS/DMA trace filtering is also enabled where configured.",
            flush=True,
        )
        return True
    except Exception as exc:
        print(
            f"[mGBA Logging Warning] Could not configure native log filtering: {exc}. "
            "The GBA BIOS/DMA messages may remain visible; Stable-Baselines3 verbosity does not control them.",
            flush=True,
        )
        return False


def _format_duration(seconds):
    """Format a non-negative duration as HH:MM:SS, with hours allowed above 99."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# --- mGBA native-runtime warm-up ---
# Local testing found that a fresh mGBA core can stall on its first run_frame()
# if PyTorch/Stable-Baselines3 has already been imported. A first core that is
# loaded, reset, and advanced BEFORE those imports allowed later fresh cores to
# run normally. Keep this bootstrap core and its video buffer alive for the
# lifetime of this process; do not use it for training observations.
# Include __mp_main__ because SubprocVecEnv's spawn workers re-import this file.
_MGBA_BOOTSTRAP_CORE = None
_MGBA_BOOTSTRAP_IMAGE = None
if __name__ in {"__main__", "__mp_main__"}:
    _configure_mgba_native_logging()
    _bootstrap_rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PokemonEmerald.gba")
    if not os.path.isfile(_bootstrap_rom_path):
        raise FileNotFoundError(
            f"Cannot perform mGBA startup warm-up: ROM not found at {_bootstrap_rom_path}"
        )
    try:
        print("[mGBA Bootstrap] Loading mGBA before PyTorch/Stable-Baselines3...", flush=True)
        import mgba.core as _mgba_bootstrap_core_module
        import mgba.image as _mgba_bootstrap_image_module

        _MGBA_BOOTSTRAP_CORE = _mgba_bootstrap_core_module.load_path(_bootstrap_rom_path)
        if _MGBA_BOOTSTRAP_CORE is None:
            raise RuntimeError("mgba.core.load_path() returned None")
        _MGBA_BOOTSTRAP_IMAGE = _mgba_bootstrap_image_module.Image(240, 160)
        _MGBA_BOOTSTRAP_CORE.set_video_buffer(_MGBA_BOOTSTRAP_IMAGE)
        _MGBA_BOOTSTRAP_CORE.reset()
        print("[mGBA Bootstrap] Advancing one initialization frame before PyTorch import...", flush=True)
        _MGBA_BOOTSTRAP_CORE.run_frame()
        print("[mGBA Bootstrap] Warm-up frame completed; bootstrap core retained.", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "mGBA pre-PyTorch warm-up failed. The application is stopping early rather than "
            "entering the known first-frame hang. Check the mGBA Python binding, ROM, and "
            "LD_LIBRARY_PATH."
        ) from exc

# These imports load PyTorch. Keep them AFTER the mGBA bootstrap above.
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor



# ==========================================
# 🛠️ CENTRAL CONFIGURATION & CONSTANTS
# ==========================================
ACTIVE_PROFILE = os.environ.get("POKEMON_AI_PROFILE", "fast_train").strip().lower()
MODE = os.environ.get("POKEMON_AI_MODE", "train").strip().lower()  # selected by profile
if MODE not in {"train", "watch", "setup"}:
    raise ValueError("POKEMON_AI_MODE must be 'train', 'watch', or 'setup'")

# Paths are resolved relative to this script, so the project can be launched from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Runtime-only artifacts stay out of the project root. Keep ROM, its cartridge .sav,
# and the established PPO model filename at the root for compatibility.
RUNTIME_DATA_DIR = os.path.join(BASE_DIR, "runtime_data")
STATE_DIR = os.path.join(RUNTIME_DATA_DIR, "states")
TEMP_DIR = os.path.join(RUNTIME_DATA_DIR, "tmp")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ROM_PATH = os.path.join(BASE_DIR, "PokemonEmerald.gba")
CARTRIDGE_SAVE_PATH = os.path.join(BASE_DIR, "PokemonEmerald.sav")
MODEL_CHECKPOINT = os.path.join(BASE_DIR, "emerald_ppo_checkpoint")
LEGACY_CHECKPOINT_STATE = os.path.join(BASE_DIR, "emerald_save.state")
CHECKPOINT_STATE = os.path.join(STATE_DIR, "emerald_save.state")
_DEFAULT_TRAINING_START_STATE = os.path.join(STATE_DIR, "training_start.state")
_configured_start_state = os.environ.get("POKEMON_AI_TRAINING_START_STATE", _DEFAULT_TRAINING_START_STATE).strip()
TRAINING_START_STATE = (
    _configured_start_state
    if os.path.isabs(_configured_start_state)
    else os.path.join(BASE_DIR, _configured_start_state)
)
REQUIRE_TRAINING_START_STATE = os.environ.get(
    "POKEMON_AI_REQUIRE_START_STATE", "1"
).strip().lower() not in {"0", "false", "no", "off"}
# Curriculum starting point. clock_set bypasses only the frustrating RTC menu while
# leaving the route to Professor Birch and obtaining a starter as learnable tasks.
# Set to "starter" later for a more advanced navigation/battle curriculum.
TRAINING_START_STAGE = os.environ.get("POKEMON_AI_START_STAGE", "clock_set").strip().lower()
if TRAINING_START_STAGE not in {"clock_set", "starter"}:
    raise ValueError("POKEMON_AI_START_STAGE must be 'clock_set' or 'starter'")
TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "tensorboard_logs")

# GUI / headless and mode are selected together by the active profile above.
# GUI profiles use a single emulator; fast_train uses four; both watch profiles use one.
GUI_MODE = os.environ.get("POKEMON_AI_GUI_MODE", "0").strip().lower() not in {"0", "false", "no", "off"}
GUI_WINDOW_TITLE = "Pokemon Emerald PPO"
GUI_SCALE = 4                     # 240x160 native frame * 4 = 960x640 window
GUI_FPS = 60                      # GUI only: pace rendering near the real GBA refresh rate.
# Headless watch speed cap in emulated frames/sec. 0 = unlimited/fast-forward.
WATCH_FPS_CAP = float(os.environ.get("POKEMON_AI_WATCH_FPS", "0"))
if WATCH_FPS_CAP < 0:
    raise ValueError("POKEMON_AI_WATCH_FPS must be >= 0 (0 means unlimited speed)")

# Human setup mode always gives the keyboard control of the game.
# In watch mode, keyboard overrides remain opt-in.
# W/A/S/D or arrows = directions; Z/Enter = A; X/Backspace = B; 1 = Start; 2 = Select.
# In setup mode press K to save the current point as the canonical training-start savestate.
HOST_CONTROL_ENABLED = MODE == "setup" or os.environ.get(
    "POKEMON_AI_HOST_CONTROL_ENABLED", "0"
).strip().lower() not in {"0", "false", "no", "off"}

# Action repetition lets each policy decision hold a button for multiple emulator frames.
# The GUI refreshes each emulated frame; the policy chooses an action every ACTION_REPEAT frames.
ACTION_REPEAT = int(os.environ.get("POKEMON_AI_ACTION_REPEAT", "2"))
if ACTION_REPEAT < 1:
    raise ValueError("POKEMON_AI_ACTION_REPEAT must be a positive integer")  # Preserve the default two-frame action cadence.
AUDIO_OUTPUT_ENABLED = True        # Audio plays in single-environment runs; parallel headless training stays silent.
AUDIO_RATE_MULTIPLIER = 2.0        # Empirical fix: this binding's captured stream sounded correct at 2x.
AUDIO_QUEUE_CHUNKS = 8             # Bounded queue prevents audio lag from growing without limit.
# Allow the GBA BIOS/game to produce a real frame on cold start before the first PPO observation.
INITIAL_BOOT_FRAMES = 600
RESET_WATCHDOG_SECONDS = 20
FRAME_DIAGNOSTICS = True
MAX_EPISODE_STEPS = int(os.environ.get("POKEMON_AI_MAX_EPISODE_STEPS", "20000"))
if MAX_EPISODE_STEPS < 1:
    raise ValueError("POKEMON_AI_MAX_EPISODE_STEPS must be a positive integer")
# Keep the per-action cost small enough that a 20k-step episode does not drown out
# sparse progress rewards. Novelty bonuses are one-time per worker to discourage loops.
STEP_PENALTY = 0.0001
LEVEL_REWARD_PER_LEVEL = 2.5
BADGE_REWARD = 50.0
NEW_MAP_REWARD = 1.0
NEW_TILE_REWARD = 0.025
PARTY_MEMBER_REWARD = 1.5
SAVE_BLOCK1_PTR_ADDRESS = 0x03005D8C
SAVE_BLOCK1_FLAGS_OFFSET = 0x1270
SAVE_BLOCK1_POS_X_OFFSET = 0x00
SAVE_BLOCK1_POS_Y_OFFSET = 0x02
# FLAG_SET_WALL_CLOCK from the standard Pokémon Emerald BPEE decompilation.
# This persistent SaveBlock1 flag is used to validate the earlier curriculum start.
CLOCK_SET_FLAG_ID = 0x51

# Persistent story flags from the standard Pokémon Emerald (BPEE) decompilation.
# These are progression milestones, not arbitrary per-frame signals.
STORY_FLAG_REWARDS = {
    CLOCK_SET_FLAG_ID: 0.5,   # Set the bedroom wall clock
    0x52: 1.5,   # Rescued Professor Birch
    0x860: 5.0,  # FLAG_SYS_POKEMON_GET: received the first Pokémon
    0x861: 2.0,  # FLAG_SYS_POKEDEX_GET: received the Pokédex
    0x74: 1.5,   # Adventure started / Pokédex obtained
    0x61: 1.0,   # Team Aqua Harbor event
    0x7D: 1.0,   # Rival event on Route 104
    0x82: 1.0,   # Rival event on Route 103
    0x8B: 1.5,   # Magma event on Mt. Chimney
    0x70: 2.0,   # Team Aqua submarine story event
    0x75: 2.0,   # Magma event at the Space Center
    0x6F: 2.0,   # Groudon awakened
}


def _training_start_validation_error(state):
    """Return a human-readable reason if a savestate is not valid for the selected curriculum stage."""
    if state.get("player_x") is None or state.get("player_y") is None:
        return "player coordinates are invalid; save while the character is in normal, controllable gameplay"

    if TRAINING_START_STAGE == "clock_set":
        if CLOCK_SET_FLAG_ID not in state.get("story_flags", set()):
            return (
                "the bedroom clock-set flag is not present; finish setting the clock, close the dialogue/menu, "
                "and press K while the character is controllable"
            )
    elif TRAINING_START_STAGE == "starter":
        if state.get("party_count", 0) < 1 or state.get("party_levels", 0) < 1:
            return "no valid Pokémon party/level was detected; obtain your starter and press K in free movement"

    return None


# Auto-Start TensorBoard Server (Training mode only)
AUTO_START_TENSORBOARD = os.environ.get("POKEMON_AI_AUTO_START_TENSORBOARD", "1").strip().lower() not in {"0", "false", "no", "off"}
TENSORBOARD_PORT = int(os.environ.get("POKEMON_AI_TENSORBOARD_PORT", "6006"))
if not 1 <= TENSORBOARD_PORT <= 65535:
    raise ValueError("POKEMON_AI_TENSORBOARD_PORT must be between 1 and 65535")

# Video recording is on by default for GUI watch mode, but off for headless watch so
# unlimited fast-forward isn't slowed by video/audio encoding. Override explicitly with
# POKEMON_AI_RECORD_VIDEO=1 when a recording is wanted.
_record_video_default = "1" if MODE == "watch" and GUI_MODE else "0"
RECORD_VIDEO = os.environ.get("POKEMON_AI_RECORD_VIDEO", _record_video_default).strip().lower() not in {"0", "false", "no", "off"}
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

# Unique Timestamp generation to prevent video overwriting across runs and days
TIMESTAMP_STR = time.strftime("%Y%m%d_%H%M%S")
TRAIN_VIDEO_PATH = os.path.join(VIDEO_DIR, f"train_journey_{TIMESTAMP_STR}.mp4")
WATCH_VIDEO_PATH = os.path.join(VIDEO_DIR, f"watch_playback_{TIMESTAMP_STR}.mp4")
VIDEO_FPS = 60.0 / max(1, ACTION_REPEAT)  # One saved image per decision; keeps recorded playback near real time
VIDEO_WIDTH = 240 * 4
VIDEO_HEIGHT = 160 * 4

# Desktop Notifications Configuration (Watch Mode Only)
ENABLE_NOTIFICATIONS = True if MODE == "watch" else False

# Environment & Performance Settings
# Keep these dimensions compatible with an existing checkpoint. The original game image is letterboxed
# inside this shape so the GBA's 3:2 aspect ratio is not distorted.
OBS_WIDTH = 80
OBS_HEIGHT = 72
TOTAL_TIMESTEPS = int(os.environ.get("POKEMON_AI_TOTAL_TIMESTEPS", "1000000"))
if TOTAL_TIMESTEPS < 1:
    raise ValueError("POKEMON_AI_TOTAL_TIMESTEPS must be a positive integer")
VERBOSE_LEVEL = int(os.environ.get("POKEMON_AI_VERBOSE", "1"))
if VERBOSE_LEVEL not in (0, 1, 2):
    raise ValueError("POKEMON_AI_VERBOSE must be 0, 1, or 2")
STOCHASTIC_WATCH = False         # False = pure deterministic best behavior in watch mode

# Environment count for headless training. Override with POKEMON_AI_NUM_ENVS.
# More workers can increase CPU use, but also require more RAM because every worker owns
# an emulator and its Python/PyTorch process. GUI and watch modes intentionally stay single-env.
HEADLESS_NUM_ENVS = int(os.environ.get("POKEMON_AI_NUM_ENVS", "4"))
if HEADLESS_NUM_ENVS < 1:
    raise ValueError("POKEMON_AI_NUM_ENVS must be a positive integer")
NUM_CPU = 1 if GUI_MODE or MODE == "watch" else HEADLESS_NUM_ENVS
VIDEO_OUTPUT_PATH = TRAIN_VIDEO_PATH if MODE == "train" else WATCH_VIDEO_PATH
# OpenCV records a temporary MPEG-4 Part 2 file. When FFmpeg is available, it is
# transcoded to broadly compatible H.264/AAC at shutdown and atomically published
# at VIDEO_OUTPUT_PATH. The temporary WAV is recorded at the empirically verified
# effective mGBA playback rate, then resampled by FFmpeg for the AAC track.
VIDEO_INCLUDE_AUDIO = True
VIDEO_H264_CRF = 22              # Lower = better quality/larger file; 20-24 is a useful range.
VIDEO_H264_PRESET = "veryfast"   # Faster encoding with reasonable compression.
_VIDEO_BASENAME = os.path.splitext(os.path.basename(VIDEO_OUTPUT_PATH))[0]
VIDEO_INTERMEDIATE_PATH = os.path.join(TEMP_DIR, _VIDEO_BASENAME + ".opencv.tmp.mp4")
VIDEO_MUX_TEMP_PATH = os.path.join(TEMP_DIR, _VIDEO_BASENAME + ".mux.tmp.mp4")
VIDEO_AUDIO_TEMP_PATH = os.path.join(TEMP_DIR, _VIDEO_BASENAME + ".audio.tmp.wav")
# PPO transition count between periodic checkpoints. Override with POKEMON_AI_CHECKPOINT_EVERY.
CHECKPOINT_SAVE_FREQUENCY = int(os.environ.get("POKEMON_AI_CHECKPOINT_EVERY", "25000"))
PROGRESS_LOG_FREQUENCY = int(os.environ.get("POKEMON_AI_PROGRESS_EVERY", "1000"))
if CHECKPOINT_SAVE_FREQUENCY < 1:
    raise ValueError("POKEMON_AI_CHECKPOINT_EVERY must be a positive integer")
if PROGRESS_LOG_FREQUENCY < 1:
    raise ValueError("POKEMON_AI_PROGRESS_EVERY must be a positive integer")

# Stopwatch Session Time Logging Configuration
SESSION_LOG_PATH = os.path.join(LOG_DIR, "session_time_log.txt")
# ==========================================



# GBA RAM layout for standard Pokémon Emerald (BPEE revision 0).
# SaveBlock1 is dynamically relocated during play, so read party/position/flags
# relative to the live pointer at 0x03005D8C rather than relying on stale absolute
# SaveBlock1 addresses. Offsets are from pret/pokeemerald/include/global.h.
SAVE_BLOCK1_PARTY_COUNT_OFFSET = 0x234
SAVE_BLOCK1_PARTY_OFFSET = 0x238
PARTY_POKEMON_SIZE = 100  # sizeof(struct Pokemon) in the runtime SaveBlock1 party
PARTY_LEVEL_OFFSET = 0x54
# FLAG_BADGE01_GET through FLAG_BADGE08_GET are 0x867..0x86E in standard Emerald.
BADGE_FLAG_IDS = tuple(range(0x867, 0x86F))
MAP_BANK_ADDRESS = 0x020322E4
MAP_NUMBER_ADDRESS = 0x020322E5

# This is the GBA key index order used by mGBA's documented GBA_KEY constants.
# setKeys() expects a bitmask, so each key index is converted to 1 << index.
GBA_KEY_INDEX = {
    "A": 0,
    "B": 1,
    "SELECT": 2,
    "START": 3,
    "RIGHT": 4,
    "LEFT": 5,
    "UP": 6,
    "DOWN": 7,
}
ACTION_TO_KEY = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "A",
    5: "B",
    6: "START",
    7: "SELECT",
}
ACTION_LABELS = dict(ACTION_TO_KEY)
ACTION_KEY_MASK = {action: 1 << GBA_KEY_INDEX[name] for action, name in ACTION_TO_KEY.items()}

# mGBA Python bindings expose raw core snapshots, not the frontend saveStateFile API.
RAW_STATE_MAGIC = b"PYMGBA_RAW_STATE_V1\x00"


tb_process = None
tb_owned = False
tb_log_handle = None
STOP_REQUESTED = False


class SessionTimer:
    """Tracks session runtime and appends duration metrics to a persistent log file."""
    def __init__(self, log_path=SESSION_LOG_PATH, mode=MODE):
        self.log_path = log_path
        self.mode = mode
        self.start_time = time.time()

    def save_session(self):
        elapsed = time.time() - self.start_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Mode: {self.mode.upper()} | Active Duration: {elapsed:.2f} seconds ({elapsed / 60:.2f} mins)\n"
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            # Automatically generates a new log file if deleted
            mode_flag = "a" if os.path.exists(self.log_path) else "w"
            with open(self.log_path, mode_flag, encoding="utf-8") as f:
                f.write(log_entry)
            print(f"[Session Timer] Session duration ({elapsed:.2f}s) successfully logged to {self.log_path}")
        except Exception as e:
            print(f"[Session Timer Error] Could not save session log: {e}")


# Initialize session stopwatch timer instance
session_timer = SessionTimer()


def send_notification(title, message):
    """Triggers native CachyOS desktop notifications using libnotify (notify-send)."""
    if ENABLE_NOTIFICATIONS:
        try:
            subprocess.run(
                ["notify-send", "-u", "normal", "-i", "applications-games", title, message],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except Exception as e:
            print(f"[Notification Error] Could not send desktop alert: {e}")


def _method(obj, *names):
    """Return the first callable method available among snake_case/camelCase aliases."""
    for name in names:
        candidate = getattr(obj, name, None)
        if callable(candidate):
            return candidate
    return None


def _close_bootstrap_core():
    """Release the warm-up core/image after the real environments have closed."""
    global _MGBA_BOOTSTRAP_CORE, _MGBA_BOOTSTRAP_IMAGE
    core = _MGBA_BOOTSTRAP_CORE
    if core is not None:
        close_core = _method(core, "close", "deinit")
        if close_core is not None:
            try:
                close_core()
            except Exception as exc:
                print(f"[mGBA Bootstrap Warning] Could not explicitly close warm-up core: {exc}", flush=True)
    _MGBA_BOOTSTRAP_CORE = None
    _MGBA_BOOTSTRAP_IMAGE = None


atexit.register(_close_bootstrap_core)


def _fsync_parent_dir(path):
    """Best-effort fsync of a file's parent directory after an atomic rename."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems do not support directory fsync; the atomic rename still helps.
        pass


def _check_rom_header(rom_path):
    """Check the GBA header code before using revision-specific memory addresses."""
    with open(rom_path, "rb") as rom_file:
        header = rom_file.read(0xBD)
    if len(header) < 0xBD:
        raise ValueError(f"ROM file is too small to contain a valid GBA header: {rom_path}")
    game_code = header[0xAC:0xB0].decode("ascii", errors="replace")
    revision = header[0xBC]
    if game_code != "BPEE":
        raise ValueError(
            f"Expected Pokémon Emerald game code BPEE, but found {game_code!r}. "
            "The configured RAM addresses are intended for Pokémon Emerald BPEE revision 0."
        )
    if revision != 0:
        raise ValueError(
            f"Expected Pokémon Emerald BPEE revision 0, but the ROM header reports revision {revision}. "
            "Do not use these memory constants until the revision-specific addresses are verified."
        )
    print(f"[ROM] Header verified: game code {game_code}, revision {revision} (Pokémon Emerald).")


class MGBAAudioOutput:
    """Drain mGBA's interleaved stereo buffer and optionally play it asynchronously.

    The mGBA audio-buffer API counts stereo frames, not individual int16 values.
    Samples are drained on the emulator thread and copied into a bounded queue;
    the PortAudio callback only consumes queued NumPy arrays and never calls into
    the emulator core. This keeps the callback predictable and avoids touching the
    native emulator object from PortAudio's audio thread.
    """

    def __init__(self, core, enable_output=False):
        self.core = core
        self.ffi = None
        self.buffer = None
        self.native = None
        self.sample_rate = 0
        self.channels = 2
        self.read_audio = None
        self.available_audio = None
        self.stream = None
        self.output_active = False
        self.muted = False
        self.error = None
        self.queue = queue.Queue(maxsize=max(2, int(AUDIO_QUEUE_CHUNKS)))
        self._current_chunk = None
        self._current_offset = 0
        self._generation = 0
        self._current_generation = 0
        self.overruns = 0
        self.underruns = 0
        self.samples_drained = 0
        self.playback_sample_rate = 0
        self.record_wave = None
        self.recording_path = None
        self.recording_sample_rate = 0
        self.recording_error = None

        try:
            pylib = importlib.import_module("mgba._pylib")
            self.ffi = pylib.ffi
            self.native = core._core
            self.buffer = self.native.getAudioBuffer(self.native)
            if self.buffer == self.ffi.NULL:
                raise RuntimeError("mGBA returned a null audio buffer")
            self.sample_rate = int(self.native.audioSampleRate(self.native))
            self.channels = int(self.buffer.channels)
            self.read_audio = pylib.lib.mAudioBufferRead
            self.available_audio = pylib.lib.mAudioBufferAvailable
            if self.sample_rate <= 0 or self.channels <= 0:
                raise RuntimeError(
                    f"Invalid mGBA audio format: rate={self.sample_rate}, channels={self.channels}"
                )
        except Exception as exc:
            self.error = f"Could not initialize mGBA audio-buffer access: {exc}"
            return

        if not enable_output:
            return

        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = f"Python package 'sounddevice' is unavailable: {exc}"
            return

        preferred_rate = int(round(self.sample_rate * max(0.25, float(AUDIO_RATE_MULTIPLIER))))
        attempts = []
        for rate in (preferred_rate, self.sample_rate):
            if rate > 0 and rate not in attempts:
                attempts.append(rate)

        last_error = None
        for rate in attempts:
            try:
                self.stream = sd.OutputStream(
                    samplerate=rate,
                    channels=self.channels,
                    dtype="int16",
                    blocksize=0,
                    latency="low",
                    callback=self._audio_callback,
                )
                self.stream.start()
                self.playback_sample_rate = rate
                self.output_active = True
                print(
                    f"[Audio] Output active: mGBA reports {self.sample_rate} Hz; "
                    f"playback stream {rate} Hz, {self.channels} channels.",
                    flush=True,
                )
                if rate != preferred_rate:
                    print(
                        "[Audio Warning] Preferred sample rate was unavailable; "
                        "using the fallback rate. Playback may sound slower.",
                        flush=True,
                    )
                return
            except Exception as exc:
                last_error = exc
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None

        self.error = f"Could not open a PortAudio output stream: {last_error}"
        print(f"[Audio Warning] {self.error}", flush=True)

    def _audio_callback(self, outdata, frames, time_info, status):
        # Never print or access the mGBA core from this real-time callback.
        outdata.fill(0)
        copied = 0
        while copied < frames:
            generation = self._generation
            if self._current_generation != generation:
                self._current_chunk = None
                self._current_offset = 0
                self._current_generation = generation
            if self._current_chunk is None or self._current_offset >= len(self._current_chunk):
                try:
                    self._current_chunk = self.queue.get_nowait()
                    self._current_offset = 0
                    self._current_generation = generation
                except queue.Empty:
                    self.underruns += 1
                    break

            remaining = len(self._current_chunk) - self._current_offset
            take = min(frames - copied, remaining)
            if not self.muted:
                outdata[copied:copied + take] = self._current_chunk[
                    self._current_offset:self._current_offset + take
                ]
            copied += take
            self._current_offset += take

    def start_recording(self, path, sample_rate):
        """Record game PCM to a temporary WAV, independent of speaker mute/output."""
        if self.buffer is None or self.ffi is None:
            return False
        try:
            if self.record_wave is not None:
                self.record_wave.close()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            writer = wave.open(path, "wb")
            writer.setnchannels(self.channels)
            writer.setsampwidth(2)
            writer.setframerate(int(sample_rate))
            self.record_wave = writer
            self.recording_path = path
            self.recording_sample_rate = int(sample_rate)
            self.recording_error = None
            print(
                f"[Recorder] Audio capture started: {path} at "
                f"{self.recording_sample_rate} Hz (effective playback rate).",
                flush=True,
            )
            return True
        except Exception as exc:
            self.recording_error = str(exc)
            print(f"[Recorder Warning] Could not start audio capture: {exc}", flush=True)
            self.record_wave = None
            return False

    @property
    def recording_active(self):
        return self.record_wave is not None

    def drain(self, accept=True, record=False):
        """Drain native samples; optionally record PCM and/or queue audible playback."""
        if self.buffer is None or self.ffi is None:
            return
        try:
            available = int(self.available_audio(self.buffer))
            while available > 0:
                pcm = self.ffi.new("int16_t[]", available * self.channels)
                count = int(self.read_audio(self.buffer, pcm, available))
                if count <= 0:
                    break
                byte_count = count * self.channels * self.ffi.sizeof("int16_t")
                samples = np.frombuffer(
                    self.ffi.buffer(pcm, byte_count), dtype=np.int16
                ).reshape(count, self.channels).copy()
                self.samples_drained += count

                # Recorded audio is independent of speaker mute and GUI presence:
                # muting should affect listening, not the saved video soundtrack.
                if record and self.record_wave is not None:
                    try:
                        self.record_wave.writeframesraw(
                            samples.astype("<i2", copy=False).tobytes()
                        )
                    except Exception as exc:
                        self.recording_error = str(exc)
                        try:
                            self.record_wave.close()
                        except Exception:
                            pass
                        self.record_wave = None
                        print(f"[Recorder Warning] Audio capture stopped: {exc}", flush=True)

                if accept and self.output_active and not self.muted:
                    try:
                        self.queue.put_nowait(samples)
                    except queue.Full:
                        # Drop the oldest queued chunk rather than accumulate audible latency.
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.overruns += 1
                        try:
                            self.queue.put_nowait(samples)
                        except queue.Full:
                            self.overruns += 1
                available = int(self.available_audio(self.buffer))
        except Exception as exc:
            # Audio must never crash or stall the emulator/training environment.
            if self.error is None:
                self.error = f"Error while draining mGBA audio: {exc}"
                print(f"[Audio Warning] {self.error}", flush=True)

    def clear_pending(self):
        # Signal the callback to discard its current chunk at the next boundary.
        # The callback owns _current_chunk; the emulator thread does not mutate it.
        self._generation += 1
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def toggle_mute(self):
        if not self.output_active:
            return False
        self.muted = not self.muted
        if self.muted:
            self.clear_pending()
        print(f"[Audio] {'Muted' if self.muted else 'Unmuted'}.", flush=True)
        return True

    @property
    def gui_label(self):
        if not self.output_active:
            return "AUDIO: UNAVAILABLE"
        return "AUDIO: MUTED" if self.muted else "AUDIO: ON"

    def close(self):
        self.output_active = False
        if self.record_wave is not None:
            try:
                self.record_wave.close()
                print(f"[Recorder] Audio capture finalized: {self.recording_path}", flush=True)
            except Exception as exc:
                self.recording_error = str(exc)
                print(f"[Recorder Warning] Could not finalize audio capture: {exc}", flush=True)
            self.record_wave = None
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.clear_pending()


class _TeeStream:
    """Mirror console output into a run log without changing the visible terminal output."""

    def __init__(self, console_stream, log_stream, lock):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self.lock = lock

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    def write(self, value):
        if not value:
            return 0
        self.console_stream.write(value)
        with self.lock:
            self.log_stream.write(value)
        return len(value)

    def flush(self):
        try:
            self.console_stream.flush()
        finally:
            with self.lock:
                self.log_stream.flush()

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())

    def fileno(self):
        return self.console_stream.fileno()


class EmeraldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": GUI_FPS}

    def __init__(self, rom_path=ROM_PATH, checkpoint_file=CHECKPOINT_STATE, env_id=0):
        super().__init__()

        self.action_space = spaces.Discrete(8)  # 0:Up, 1:Down, 2:Left, 3:Right, 4:A, 5:B, 6:Start, 7:Select
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8
        )

        self.rom_path = rom_path
        self.checkpoint_file = checkpoint_file
        self.env_id = env_id
        self.step_counter = 0
        self.episode_step_counter = 0
        self.last_notified_event = None
        self.stop_requested = False
        self.paused = False
        self.manual_action_next = None
        self._save_training_start_requested = False
        self.window_created = False
        self._mouse_callback_available = False
        self._first_reset = True
        self._episode_start_buffer = None
        self._episode_start_buffer_kind = None
        self._framebuffer = None
        self._framebuffer_ffi = None
        self._episode_start_file = f"{self.checkpoint_file}.episode_start_{self.env_id}.state"
        self._screenshot_fallback_path = os.path.join(
            TEMP_DIR, f"emerald_ai_screen_{os.getpid()}_{self.env_id}.png"
        )
        self._last_raw_frame_rgb = None
        self._memory_read_warning_printed = False
        self.visited_maps = set()
        self.visited_tiles = set()
        self.seen_story_flags = set()

        # Tracking variables for reward shaping deltas
        self.prev_party_levels = 0
        self.prev_badges = 0
        self.prev_caught_count = 0
        self.prev_map_id = None

        # --- mGBA Emulator Core Initialization ---
        self.core = None
        self.mgba_module = None
        try:
            self._init_emulator()
        except Exception:
            # If native initialization fails after creating the core, release it even
            # though this partially constructed environment is not yet owned by VecEnv.
            failed_core = self.core
            if failed_core is not None:
                close_core = _method(failed_core, "close", "deinit")
                if close_core is not None:
                    try:
                        close_core()
                    except Exception:
                        pass
            self.core = None
            self._framebuffer = None
            self._framebuffer_ffi = None
            raise

        # mGBA's native ring buffer is drained to prevent accumulation, but audible
        # output is strictly GUI-only. Headless mode never imports/opens sounddevice.
        # If headless watch mode records a video, its soundtrack can still be captured
        # and muxed into the file without enabling live speaker playback.
        self._audio_accepting = False
        self._record_audio_active = False
        try:
            self.audio_output = MGBAAudioOutput(
                self.core,
                enable_output=(AUDIO_OUTPUT_ENABLED and GUI_MODE and NUM_CPU == 1),
            )
            if self.audio_output.error:
                print(f"[Audio Warning] {self.audio_output.error}", flush=True)
        except Exception as exc:
            self.audio_output = None
            print(f"[Audio Warning] Audio subsystem disabled after initialization error: {exc}", flush=True)

        # Video Recorder Setup (Ensures ./videos directory exists first)
        self.video_writer = None
        # In parallel training, record only worker zero to prevent multiple processes writing the same MP4.
        if RECORD_VIDEO and (NUM_CPU == 1 or self.env_id == 0):
            os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)
            os.makedirs(TEMP_DIR, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                VIDEO_INTERMEDIATE_PATH, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
            )
            if not self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                print("[Recorder Warning] OpenCV could not open the temporary MP4 writer; video recording is disabled.")
            else:
                print(
                    f"[Recorder] Recording [{MODE.upper()}] session to a temporary video; "
                    f"final H.264/AAC output: {VIDEO_OUTPUT_PATH}",
                    flush=True,
                )
                if VIDEO_INCLUDE_AUDIO and self.audio_output is not None:
                    # The mGBA build reported 32768 Hz, but user-tested playback was
                    # correct at 2x. Write WAV at that effective rate for muxing.
                    capture_rate = int(round(
                        self.audio_output.sample_rate * max(0.25, float(AUDIO_RATE_MULTIPLIER))
                    ))
                    self.audio_output.start_recording(VIDEO_AUDIO_TEMP_PATH, capture_rate)

        if GUI_MODE:
            try:
                self._create_gui_window()
            except Exception:
                # Construction failed before DummyVecEnv can own this instance; release resources.
                self.close()
                raise

        print(f"[Init] Pokémon Emerald environment initialized in [{MODE.upper()}] mode.")

    def _init_emulator(self):
        try:
            if not os.path.isfile(self.rom_path):
                raise FileNotFoundError(f"ROM file not found at '{self.rom_path}'!")
            _check_rom_header(self.rom_path)

            self.mgba_module = importlib.import_module("mgba.core")
            loader = getattr(self.mgba_module, "load_path", None)
            if not callable(loader):
                raise RuntimeError(
                    "The installed 'mgba.core' module does not expose load_path(). "
                    "Install/use a Python mGBA binding that supports loading a ROM into a core."
                )

            self.core = loader(self.rom_path)
            if self.core is None:
                raise RuntimeError(f"Failed to load ROM from {self.rom_path}")

            # Load the user's cartridge save file when the installed binding supports it.
            # temporary=False allows mGBA to persist changes; temporary=True avoids shared .sav writes during training.
            if os.path.isfile(CARTRIDGE_SAVE_PATH):
                load_save = _method(self.core, "load_save_file", "loadSaveFile")
                if load_save is not None:
                    # Parallel/training environments must not all write to the same cartridge .sav file.
                    # In watch mode, allow mGBA to persist ordinary in-game saves to the user's .sav file.
                    temporary_save = MODE == "train" or NUM_CPU > 1
                    try:
                        load_result = load_save(CARTRIDGE_SAVE_PATH, temporary_save)
                        if load_result is False:
                            print(f"[Save Warning] mGBA reported it could not load {CARTRIDGE_SAVE_PATH}.")
                        else:
                            save_mode = "temporary/read-only during training" if temporary_save else "persistent"
                            print(f"[Save] Loaded cartridge save ({save_mode}): {CARTRIDGE_SAVE_PATH}")
                    except TypeError:
                        try:
                            load_result = load_save(CARTRIDGE_SAVE_PATH)
                            if load_result is False:
                                print(f"[Save Warning] mGBA reported it could not load {CARTRIDGE_SAVE_PATH}.")
                            else:
                                print(f"[Save Warning] Loaded cartridge save, but this binding does not let the script select temporary/persistent mode.")
                        except Exception as e:
                            print(f"[Save Warning] Could not load cartridge save through this binding: {e}")
                    except Exception as e:
                        print(f"[Save Warning] Could not load cartridge save through this binding: {e}")
                else:
                    autoload_save = _method(self.core, "autoload_save", "autoloadSave")
                    if autoload_save is not None and not (MODE == "train" and NUM_CPU > 1):
                        try:
                            print("[Save Warning] This binding lacks loadSaveFile(); trying mGBA's automatic save loading.")
                            autoload_save()
                        except Exception as e:
                            print(f"[Save Warning] Automatic cartridge-save loading failed: {e}")
                    elif autoload_save is not None:
                        print("[Save Warning] Skipping automatic save loading in parallel training to reduce shared .sav-file write risk.")
                    else:
                        print("[Save Warning] This mGBA binding exposes no cartridge save-loading method.")
            else:
                # A cartridge .sav is meaningful for a single watch instance, where in-game
                # saves should persist. mCoreLoadSaveFile uses O_CREAT|O_RDWR, so passing the
                # expected path creates/attaches that exact .sav if it does not yet exist.
                # Training intentionally avoids creating a shared file because parallel workers
                # must not race to write the same cartridge save.
                load_save = _method(self.core, "load_save_file", "loadSaveFile")
                if MODE == "watch" and NUM_CPU == 1 and load_save is not None:
                    try:
                        load_result = load_save(CARTRIDGE_SAVE_PATH, False)
                        if load_result is False:
                            print(
                                f"[Save Warning] mGBA could not initialize persistent cartridge save: {CARTRIDGE_SAVE_PATH}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[Save] Persistent cartridge save attached: {CARTRIDGE_SAVE_PATH} "
                                f"({'created' if os.path.isfile(CARTRIDGE_SAVE_PATH) else 'managed by mGBA; file appears when savedata is flushed'})",
                                flush=True,
                            )
                    except Exception as exc:
                        print(f"[Save Warning] Could not initialize persistent cartridge save: {exc}", flush=True)
                else:
                    # If explicit save-file loading isn't supported, use mGBA's configured
                    # per-ROM path as a fallback only for single-instance watch mode.
                    autoload_save = _method(self.core, "autoload_save", "autoloadSave")
                    if MODE == "watch" and NUM_CPU == 1 and autoload_save is not None:
                        try:
                            result = autoload_save()
                            if result is False:
                                print("[Save Warning] mGBA autoloadSave() could not initialize the default cartridge save.", flush=True)
                            else:
                                print(
                                    f"[Save] mGBA autoloaded its per-ROM cartridge save (expected beside ROM: {CARTRIDGE_SAVE_PATH}).",
                                    flush=True,
                                )
                        except Exception as exc:
                            print(f"[Save Warning] Could not autoload default cartridge save: {exc}", flush=True)
                    else:
                        if MODE == "train":
                            save_note = "training uses isolated emulator save data and does not write the shared cartridge .sav file"
                        else:
                            save_note = "this mGBA binding exposes no persistent save-file loading method"
                        print(
                            f"[Save Info] No cartridge save found at {CARTRIDGE_SAVE_PATH}; {save_note}.",
                            flush=True,
                        )

            reset_core = _method(self.core, "reset")
            if reset_core is None:
                raise RuntimeError("The installed mGBA core exposes no reset() method.")

            # mGBA needs the external image target attached before reset/run_frame.
            self._setup_framebuffer()
            reset_core()
            print("[Emulator] mGBA core loaded, framebuffer attached, and reset successfully.", flush=True)
        except ImportError as e:
            raise RuntimeError(
                "Python module 'mgba.core' was not found. Install/configure the mGBA Python bindings. "
                "The script will not silently train on a simulated black screen."
            ) from e
        except Exception as e:
            # Deliberately do not silently fall back to a fake environment: that can produce a useless PPO model.
            raise RuntimeError(f"[Emulator Error] Could not initialize Pokémon Emerald: {e}") from e

    def _setup_framebuffer(self):
        """Attach the mGBA Image target before the core is reset or advanced."""
        image_module = importlib.import_module("mgba.image")
        image_type = getattr(image_module, "Image", None)
        if image_type is None:
            raise RuntimeError("The installed mGBA binding does not expose mgba.image.Image.")

        dimensions = _method(self.core, "desired_video_dimensions", "desiredVideoDimensions")
        if dimensions is not None:
            try:
                width, height = dimensions()
            except Exception:
                width, height = 240, 160
        else:
            width, height = 240, 160
        width, height = int(width or 240), int(height or 160)
        # Match the standalone probe's Image(width, height) constructor exactly.
        # Keep mGBA's default native color-buffer configuration rather than forcing alpha=False.
        self._framebuffer = image_type(width, height)
        self._framebuffer_ffi = importlib.import_module("mgba._pylib").ffi
        set_video_buffer = _method(self.core, "set_video_buffer", "setVideoBuffer")
        if set_video_buffer is None:
            raise RuntimeError("The mGBA binding exposes neither set_video_buffer() nor setVideoBuffer().")
        set_video_buffer(self._framebuffer)
        print(f"[Video] Framebuffer attached: {width}x{height}", flush=True)

    def _advance_frame_for_observation(self, frame_count=1):
        """Advance the mGBA core without altering its controller state first.

        A newly loaded/reset core starts with no keys pressed. Avoid calling the binding's
        set_keys wrapper immediately before the very first run_frame: the standalone mGBA
        probe does not need that call, and keeping this path minimal helps isolate native-core
        startup issues. During normal gameplay, step() explicitly sets the action keys.
        """
        if self.core is None or self._framebuffer is None:
            return
        run_frame = _method(self.core, "run_frame", "runFrame")
        if run_frame is None:
            raise RuntimeError("The installed mGBA core does not expose run_frame() or runFrame().")
        total = max(1, int(frame_count))
        for frame_index in range(total):
            # These logs distinguish a single wedged native call from a slow frame loop.
            if frame_index == 0:
                details = []
                for label, names in (
                    ("current_frame", ("current_frame", "currentFrame")),
                    ("frame_cycles", ("frame_cycles", "frameCycles")),
                    ("frequency", ("frequency",)),
                ):
                    getter = _method(self.core, *names)
                    if getter is not None:
                        try:
                            details.append(f"{label}={getter()}")
                        except Exception as exc:
                            details.append(f"{label}=ERROR({exc})")
                suffix = "; " + ", ".join(details) if details else ""
                print(f"[Boot] Calling mGBA run_frame() for the first frame{suffix}...", flush=True)
            run_frame()
            self._pump_audio_buffer(accept=False)
            if frame_index == 0:
                print("[Boot] First mGBA frame completed.", flush=True)
            if total >= 100 and (frame_index + 1) % 100 == 0:
                print(f"[Boot] Advanced {frame_index + 1}/{total} GBA frames", flush=True)

    def _pump_audio_buffer(self, accept=None, record=False):
        if self.audio_output is None:
            return
        if accept is None:
            accept = self._audio_accepting
        self.audio_output.drain(accept=bool(accept), record=bool(record))

    def _audio_gui_label(self):
        if not AUDIO_OUTPUT_ENABLED:
            return "AUDIO: DISABLED"
        if self.audio_output is None:
            return "AUDIO: UNAVAILABLE"
        if not self.audio_output.output_active:
            return "AUDIO: UNAVAILABLE" if self.audio_output.error else "AUDIO: OFF"
        return self.audio_output.gui_label

    def _toggle_audio_mute(self):
        if not AUDIO_OUTPUT_ENABLED:
            print("[Audio] Set AUDIO_OUTPUT_ENABLED = True to enable playback.", flush=True)
            return
        if self.audio_output is None or not self.audio_output.output_active:
            print("[Audio] No active output stream; check sounddevice/PortAudio and the run log.", flush=True)
            return
        self.audio_output.toggle_mute()

    def _on_gui_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        rect = getattr(self, "_audio_button_rect", None)
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._toggle_audio_mute()

    def _create_gui_window(self):
        try:
            cv2.namedWindow(GUI_WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(GUI_WINDOW_TITLE, 240 * GUI_SCALE, 160 * GUI_SCALE)

            # Important: on some OpenCV Qt/HighGUI builds, namedWindow() alone does
            # not register a usable native window handler until the first imshow().
            # Registering the mouse callback before that first paint can therefore
            # raise "NULL window handler" even though a simple namedWindow/imshow
            # smoke test works. Paint a temporary startup frame first, pump Qt, and
            # only then attach the callback. The actual emulator frame replaces it
            # on the first render.
            startup_frame = np.zeros(
                (160 * GUI_SCALE, 240 * GUI_SCALE, 3), dtype=np.uint8
            )
            cv2.putText(
                startup_frame,
                "Starting Pokemon Emerald...",
                (18, max(34, (160 * GUI_SCALE) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(GUI_WINDOW_TITLE, startup_frame)
            cv2.waitKey(1)
            visible = cv2.getWindowProperty(GUI_WINDOW_TITLE, cv2.WND_PROP_VISIBLE)
            if visible < 0:
                raise RuntimeError(
                    f"OpenCV did not register window {GUI_WINDOW_TITLE!r} after its first paint."
                )
            # Mouse callbacks are optional: some OpenCV Qt builds can paint and
            # process keyboard input but throw NULL-window-handler from
            # setMouseCallback(). Do not tear down a working GUI because this
            # optional convenience failed; keyboard M remains available.
            try:
                cv2.setMouseCallback(GUI_WINDOW_TITLE, self._on_gui_mouse)
                self._mouse_callback_available = True
            except cv2.error as callback_error:
                self._mouse_callback_available = False
                print(
                    f"[GUI Warning] Mouse callback unavailable ({callback_error}). "
                    "The GUI will continue; press M to mute/unmute audio.",
                    flush=True,
                )

            self.window_created = True
            print("[GUI] Window created. Press Q/Esc to quit, P to pause/resume, M to mute/unmute audio.", flush=True)
            if MODE == "watch":
                if self._mouse_callback_available:
                    print("[GUI] Click the Audio button or press M to mute/unmute.", flush=True)
                else:
                    print("[GUI] Audio button is display-only in this OpenCV build; press M to mute/unmute.", flush=True)
            if HOST_CONTROL_ENABLED and MODE in {"watch", "setup"}:
                print("[GUI] Host controls active: WASD/arrows move, Z/Enter=A, X/Backspace=B, 1=Start, 2=Select.", flush=True)
            if MODE == "setup":
                print(f"[Setup] Play manually, then press K to save the training start state: {TRAINING_START_STATE}", flush=True)
        except Exception as e:
            self.window_created = False
            try:
                cv2.destroyWindow(GUI_WINDOW_TITLE)
                cv2.waitKey(1)
            except Exception:
                pass
            raise RuntimeError(
                "Could not create a usable OpenCV GUI window. On Wayland, this build may need XWayland: "
                "try 'QT_QPA_PLATFORM=xcb python -u main.py' and ensure xorg-xwayland/XCB libraries "
                "are installed. The Qt font-directory warnings are usually non-fatal. "
                "Use GUI_MODE = False only if you intentionally want headless operation."
            ) from e

    def _core_call(self, method_names, *args):
        method = _method(self.core, *method_names)
        if method is None:
            raise AttributeError(f"mGBA core does not expose any of these methods: {method_names}")
        return method(*args)

    def _load_state_file(self, path):
        """Load a frontend savestate when available, otherwise our tagged raw-core snapshot."""
        load_state = _method(self.core, "load_state_file", "loadStateFile")
        if load_state is not None:
            try:
                # Match saveStateFile's full-state flags, including cartridge save-data extdata.
                result = load_state(path, 31)
            except TypeError:
                result = load_state(path)
                print(
                    "[State Warning] This binding does not accept explicit load flags; "
                    "save-data restoration depends on its default flags.",
                    flush=True,
                )
            success = result is not False
            if not success:
                print(f"[State Warning] mGBA rejected save state: {path}")
            return success

        load_raw = _method(self.core, "load_raw_state", "loadRawState")
        save_raw = _method(self.core, "save_raw_state", "saveRawState")
        if load_raw is None or save_raw is None:
            print("[State Warning] This mGBA binding exposes neither file savestates nor raw-state loading.")
            return False
        if not os.path.isfile(path):
            return False

        try:
            with open(path, "rb") as f:
                encoded = f.read()
            if not encoded.startswith(RAW_STATE_MAGIC):
                print(
                    f"[State Warning] {path} is not a tagged raw-core snapshot; "
                    "refusing to load it as raw state."
                )
                return False
            payload = encoded[len(RAW_STATE_MAGIC):]
            template = save_raw()
            if template is None or len(payload) != len(template):
                expected = len(template) if template is not None else "unavailable"
                print(f"[State Warning] Raw-state size mismatch for {path} (got {len(payload)}, expected {expected}).")
                return False

            ffi = self._framebuffer_ffi or importlib.import_module("mgba._pylib").ffi
            native_state = ffi.new("unsigned char[]", len(payload))
            ffi.buffer(native_state, len(payload))[:] = payload
            result = load_raw(native_state)
            success = result is not False
            if not success:
                print(f"[State Warning] mGBA rejected raw core state: {path}")
            return success
        except Exception as e:
            print(f"[State Warning] Could not load raw core state from {path}: {e}")
            return False

    def _save_state_file(self, path):
        """Save an mGBA state atomically, preserving the previous valid file on errors."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary_path = path + ".tmp"
        save_state = _method(self.core, "save_state_file", "saveStateFile")
        if save_state is not None:
            try:
                try:
                    result = save_state(temporary_path)
                except TypeError:
                    result = save_state(temporary_path, 31)
                if result is False:
                    raise RuntimeError("mGBA reported failure while writing the temporary state")
                if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
                    raise RuntimeError("mGBA did not produce a non-empty temporary state file")
                with open(temporary_path, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(temporary_path, path)
                _fsync_parent_dir(path)
                return True
            except Exception as exc:
                print(f"[State Warning] Could not atomically save mGBA state to {path}: {exc}")
                try:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
                except OSError:
                    pass
                return False

        save_raw = _method(self.core, "save_raw_state", "saveRawState")
        if save_raw is None:
            print("[State Warning] This mGBA binding exposes neither file savestates nor raw-state saving.")
            return False

        try:
            state = save_raw()
            if state is None:
                print("[State Warning] mGBA could not produce a raw core-state snapshot.")
                return False
            ffi = self._framebuffer_ffi or importlib.import_module("mgba._pylib").ffi
            payload = bytes(ffi.buffer(state, len(state)))
            with open(temporary_path, "wb") as f:
                f.write(RAW_STATE_MAGIC)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, path)
            _fsync_parent_dir(path)
            print(f"[Save] Raw mGBA core state saved to {path} ({len(payload)} bytes).")
            return True
        except Exception as exc:
            print(f"[State Warning] Could not save raw core state to {path}: {exc}")
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
            return False

    def _capture_episode_start(self):
        """Store an in-memory native or raw snapshot; use the tagged file fallback if necessary."""
        save_buffer = _method(self.core, "save_state_buffer", "saveStateBuffer")
        if save_buffer is not None:
            try:
                buffer = save_buffer()
                if buffer is not None and len(buffer):
                    self._episode_start_buffer = buffer
                    self._episode_start_buffer_kind = "native"
                    return
            except Exception as e:
                print(f"[State Info] Native in-memory snapshot unavailable; trying raw state ({e}).")

        save_raw = _method(self.core, "save_raw_state", "saveRawState")
        if save_raw is not None:
            try:
                state = save_raw()
                if state is not None:
                    # Retain the CFFI array itself so its backing memory remains alive for load_raw_state().
                    self._episode_start_buffer = state
                    self._episode_start_buffer_kind = "raw"
                    return
            except Exception as e:
                print(f"[State Info] Raw in-memory snapshot unavailable; using a file snapshot instead ({e}).")

        if not self._save_state_file(self._episode_start_file):
            print("[State Warning] Could not create an episode-start state; later episodes may reset the emulator core.")

    def _restore_episode_start(self):
        if self._episode_start_buffer is not None:
            if self._episode_start_buffer_kind == "raw":
                load_raw = _method(self.core, "load_raw_state", "loadRawState")
                if load_raw is not None:
                    try:
                        result = load_raw(self._episode_start_buffer)
                        if result is not False:
                            return True
                    except Exception as e:
                        print(f"[State Warning] Could not restore in-memory raw state: {e}")
            else:
                load_buffer = _method(self.core, "load_state_buffer", "loadStateBuffer")
                if load_buffer is not None:
                    try:
                        try:
                            result = load_buffer(self._episode_start_buffer, 31)
                        except TypeError:
                            result = load_buffer(self._episode_start_buffer)
                        if result is not False:
                            return True
                    except Exception as e:
                        print(f"[State Warning] Could not restore the in-memory episode state: {e}")

        if os.path.isfile(self._episode_start_file):
            return self._load_state_file(self._episode_start_file)

        reset_core = _method(self.core, "reset")
        if reset_core is not None:
            reset_core()
        return False

    def _read_framebuffer(self):
        """Return an RGB uint8 frame from mGBA's native video buffer or a wrapper fallback."""
        if self._framebuffer is not None and self._framebuffer_ffi is not None:
            try:
                image = self._framebuffer
                ffi = self._framebuffer_ffi
                pixel_size = int(ffi.sizeof("mColor"))
                byte_count = int(image.stride * image.height * pixel_size)
                raw = ffi.buffer(image.buffer, byte_count)

                if pixel_size == 4:
                    # mgba.image.Image documents this storage as RGBX when alpha=False.
                    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(image.height, image.stride, 4)
                    return np.ascontiguousarray(pixels[:, :image.width, :3]).copy()

                if pixel_size == 2:
                    # mGBA's 16-bit mColor layout is 5-bit R, 5-bit G, 5-bit B.
                    packed = np.frombuffer(raw, dtype=np.uint16).reshape(image.height, image.stride)[:, :image.width]
                    rgb = np.empty((image.height, image.width, 3), dtype=np.uint8)
                    rgb[:, :, 0] = ((packed & 0x1F) * 255 + 15) // 31
                    rgb[:, :, 1] = (((packed >> 5) & 0x1F) * 255 + 15) // 31
                    rgb[:, :, 2] = (((packed >> 10) & 0x1F) * 255 + 15) // 31
                    return rgb

                raise RuntimeError(f"Unsupported mGBA mColor size: {pixel_size} bytes")
            except Exception as e:
                raise RuntimeError(f"Could not decode the configured mGBA video buffer: {e}") from e

        # Compatibility path for alternative bindings with direct frame/screenshot access.
        for name in ("screen_buffer", "screenBuffer", "get_screen_buffer", "getScreenBuffer"):
            source = getattr(self.core, name, None)
            if source is None:
                continue
            try:
                raw = source() if callable(source) else source
                frame = np.asarray(raw)
                if frame.ndim == 1:
                    if frame.size == 240 * 160 * 4:
                        frame = frame.reshape((160, 240, 4))
                    elif frame.size == 240 * 160 * 3:
                        frame = frame.reshape((160, 240, 3))
                if frame.ndim == 3 and frame.shape[2] >= 3 and frame.size:
                    if frame.shape[0] == 240 and frame.shape[1] == 160:
                        frame = frame.transpose(1, 0, 2)
                    return np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).copy()
            except Exception:
                continue

        screenshot = _method(self.core, "screenshot")
        if screenshot is not None:
            try:
                os.makedirs(TEMP_DIR, exist_ok=True)
                try:
                    screenshot(self._screenshot_fallback_path)
                except TypeError:
                    screenshot(filename=self._screenshot_fallback_path)
                frame_bgr = cv2.imread(self._screenshot_fallback_path, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        raise RuntimeError(
            "Could not read the mGBA framebuffer. No configured mgba.image.Image video buffer "
            "or supported direct screenshot API is available."
        )

    def _get_obs(self):
        raw_frame_rgb = self._read_framebuffer()
        self._last_raw_frame_rgb = raw_frame_rgb

        # Save a small, fixed set of real application frames so video-path regressions are easy to inspect.
        if FRAME_DIAGNOSTICS and self.step_counter in (0, 1, 30, 120, 600):
            os.makedirs(TEMP_DIR, exist_ok=True)
            debug_path = os.path.join(
                TEMP_DIR,
                f"emerald_main_step_{self.step_counter}_worker_{self.env_id}_pid_{os.getpid()}.png",
            )
            try:
                cv2.imwrite(debug_path, cv2.cvtColor(raw_frame_rgb, cv2.COLOR_RGB2BGR))
                print(
                    f"[Video Debug] step={self.step_counter}, shape={raw_frame_rgb.shape}, "
                    f"min={int(raw_frame_rgb.min())}, max={int(raw_frame_rgb.max())}, "
                    f"mean={float(raw_frame_rgb.mean()):.2f}, saved={debug_path}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Video Debug Warning] Could not save diagnostic frame: {exc}", flush=True)

        # Keep the original observation shape for existing CNN checkpoints.
        raw_height, raw_width = raw_frame_rgb.shape[:2]
        if raw_width <= 0 or raw_height <= 0:
            raise RuntimeError(f"mGBA returned an invalid frame shape: {raw_frame_rgb.shape}")
        scale = min(OBS_WIDTH / raw_width, OBS_HEIGHT / raw_height)
        scaled_width = max(1, int(round(raw_width * scale)))
        scaled_height = max(1, int(round(raw_height * scale)))
        resized = cv2.resize(
            raw_frame_rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST
        )
        obs = np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)
        x_offset = (OBS_WIDTH - scaled_width) // 2
        y_offset = (OBS_HEIGHT - scaled_height) // 2
        obs[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized
        return obs

    def _set_action_keys(self, action):
        """Set GBA input using the native bitmask API or mGBA's Python raw= keyword."""
        if action is None:
            mask = 0
            key_name = None
        else:
            action = int(action)
            if action not in ACTION_TO_KEY:
                raise ValueError(f"Invalid action {action}; expected an integer from 0 to 7.")
            key_name = ACTION_TO_KEY[action]
            mask = ACTION_KEY_MASK[action]

        set_keys_camel = _method(self.core, "setKeys")
        if set_keys_camel is not None:
            set_keys_camel(mask)
            return

        set_keys_snake = _method(self.core, "set_keys")
        if set_keys_snake is not None:
            # The official mGBA Python binding treats positional arguments as key indices;
            # raw= is the explicit escape hatch for a precomputed controller bitmask.
            try:
                set_keys_snake(raw=mask)
            except TypeError:
                if key_name is None:
                    set_keys_snake()
                else:
                    set_keys_snake(GBA_KEY_INDEX[key_name])
            return

        raise RuntimeError("The installed mGBA core does not expose setKeys() or set_keys().")

    def _read8(self, address):
        """Read one byte through the official binding's GBA Memory view or compatible aliases."""
        memory = getattr(self.core, "memory", None)
        if memory is not None:
            byte_view = getattr(memory, "u8", None)
            if byte_view is not None:
                return int(byte_view[address]) & 0xFF
            try:
                return int(memory[address]) & 0xFF
            except (TypeError, IndexError, KeyError, AttributeError):
                pass
        return int(self._core_call(("read8", "read_8", "busRead8", "bus_read_8"), address)) & 0xFF

    def _read16(self, address):
        """Read a little-endian 16-bit GBA value using the proven 8-bit bus access."""
        return self._read8(address) | (self._read8(address + 1) << 8)

    def _read32(self, address):
        """Read a little-endian 32-bit GBA value using the proven 8-bit bus access."""
        return (
            self._read8(address)
            | (self._read8(address + 1) << 8)
            | (self._read8(address + 2) << 16)
            | (self._read8(address + 3) << 24)
        )

    def _read_saveblock1_pointer(self):
        """Return Emerald's current SaveBlock1 pointer, or None if not initialized."""
        pointer = self._read32(SAVE_BLOCK1_PTR_ADDRESS)
        # EWRAM is 0x02000000..0x0203FFFF. Reject uninitialized/garbage pointers.
        if not (0x02000000 <= pointer < 0x02040000) or pointer & 1:
            return None
        return pointer

    def _read_player_position(self, saveblock1_ptr):
        """Read SaveBlock1.pos (x/y at offsets 0/2); reject values that look invalid."""
        if saveblock1_ptr is None:
            return None, None
        x = self._read16(saveblock1_ptr + SAVE_BLOCK1_POS_X_OFFSET)
        y = self._read16(saveblock1_ptr + SAVE_BLOCK1_POS_Y_OFFSET)
        # Coords16 is signed, but ordinary map coordinates should be small nonnegative ints.
        if x & 0x8000:
            x -= 0x10000
        if y & 0x8000:
            y -= 0x10000
        if not (0 <= x <= 512 and 0 <= y <= 512):
            return None, None
        return x, y

    def _read_story_flags(self, saveblock1_ptr):
        """Read curated persistent story flags from SaveBlock1.flags."""
        if saveblock1_ptr is None:
            return set()
        flags = set()
        flags_base = saveblock1_ptr + SAVE_BLOCK1_FLAGS_OFFSET
        for flag_id in STORY_FLAG_REWARDS:
            flag_byte = self._read8(flags_base + (flag_id >> 3))
            if flag_byte & (1 << (flag_id & 7)):
                flags.add(flag_id)
        return flags

    def _read_badge_flags(self, saveblock1_ptr):
        """Return an 8-bit mask from Emerald's official badge flags (0x867..0x86E)."""
        if saveblock1_ptr is None:
            return 0
        flags_base = saveblock1_ptr + SAVE_BLOCK1_FLAGS_OFFSET
        mask = 0
        for badge_index, flag_id in enumerate(BADGE_FLAG_IDS):
            flag_byte = self._read8(flags_base + (flag_id >> 3))
            if flag_byte & (1 << (flag_id & 7)):
                mask |= 1 << badge_index
        return mask

    def _read_game_state(self):
        """Read party, badges, map, coordinates and story flags for Emerald BPEE."""
        saveblock1_ptr = self._read_saveblock1_pointer()
        if saveblock1_ptr is None:
            raise ValueError("SaveBlock1 pointer is not initialized/valid yet")

        # SaveBlock1 moves at runtime, so read the party from its live pointer.
        party_count = self._read8(saveblock1_ptr + SAVE_BLOCK1_PARTY_COUNT_OFFSET)
        if party_count > 6:
            raise ValueError(
                f"Unexpected party count {party_count} from SaveBlock1; RAM mapping may be wrong."
            )

        total_party_levels = 0
        party_base = saveblock1_ptr + SAVE_BLOCK1_PARTY_OFFSET
        for i in range(party_count):
            mon_base = party_base + (i * PARTY_POKEMON_SIZE)
            level = self._read8(mon_base + PARTY_LEVEL_OFFSET)
            if 1 <= level <= 100:
                total_party_levels += level

        badge_flags = self._read_badge_flags(saveblock1_ptr)
        map_bank = self._read8(MAP_BANK_ADDRESS)
        map_number = self._read8(MAP_NUMBER_ADDRESS)
        map_id = (map_bank << 8) | map_number
        player_x, player_y = self._read_player_position(saveblock1_ptr)
        story_flags = self._read_story_flags(saveblock1_ptr)

        return {
            "party_count": party_count,
            "party_levels": total_party_levels,
            "badge_flags": badge_flags,
            "map_id": map_id,
            "player_x": player_x,
            "player_y": player_y,
            "story_flags": story_flags,
        }

    def _refresh_reward_baseline(self):
        try:
            state = self._read_game_state()
            self.prev_party_levels = state["party_levels"]
            self.prev_badges = state["badge_flags"]
            self.prev_caught_count = state["party_count"]
            self.prev_map_id = state["map_id"]
            self.visited_maps.add(state["map_id"])
            self.seen_story_flags.update(state.get("story_flags", set()))
            if state.get("player_x") is not None and state.get("player_y") is not None:
                self.visited_tiles.add((state["map_id"], state["player_x"], state["player_y"]))
            self._memory_read_warning_printed = False
        except Exception as e:
            if not self._memory_read_warning_printed:
                print(f"[RAM Warning] Reward memory baseline unavailable: {e}")
                self._memory_read_warning_printed = True

    def reset(self, seed=None, options=None):
        # If a native call wedges inside reset, faulthandler prints the Python call site.
        faulthandler.dump_traceback_later(RESET_WATCHDOG_SECONDS, repeat=False)
        reset_started = time.perf_counter()
        try:
            self._audio_accepting = False
            self._record_audio_active = False
            if self.audio_output is not None:
                self.audio_output.clear_pending()
            print(f"[Reset {self.env_id}] entered", flush=True)
            super().reset(seed=seed)

            if self.stop_requested or STOP_REQUESTED:
                obs = self._get_obs()
                return obs, {"stopped": True}

            if self._first_reset:
                restored_checkpoint = False
                if MODE == "setup":
                    # Setup deliberately ignores existing worker/game snapshots so it can
                    # boot the ROM and let a person create a clean, intentional start state.
                    print("[Setup] Ignoring existing emulator checkpoints; cold-booting for manual setup.", flush=True)
                elif os.path.isfile(TRAINING_START_STATE):
                    print(f"[Start State] Loading canonical training start: {TRAINING_START_STATE}", flush=True)
                    restored_checkpoint = self._load_state_file(TRAINING_START_STATE)
                    if not restored_checkpoint:
                        raise RuntimeError(
                            f"Training start savestate exists but could not be loaded: {TRAINING_START_STATE}. "
                            "Recreate it with --profile setup and press K in a playable state."
                        )
                elif MODE == "train" and REQUIRE_TRAINING_START_STATE:
                    raise FileNotFoundError(
                        f"Required training start savestate not found: {TRAINING_START_STATE}\n"
                        f"Run ./run.sh --profile setup and follow the instructions for stage '{TRAINING_START_STAGE}', "
                        "then press K to save the training start state."
                    )
                elif MODE != "setup" and os.path.isfile(self.checkpoint_file):
                    print(f"[Load] Attempting legacy emulator checkpoint: {self.checkpoint_file}", flush=True)
                    restored_checkpoint = self._load_state_file(self.checkpoint_file)
                    if not restored_checkpoint:
                        print("[State Warning] Checkpoint could not be loaded; starting from a clean reset.", flush=True)

                if restored_checkpoint:
                    print("[Reset] Checkpoint restored; refreshing one frame.", flush=True)
                    self._advance_frame_for_observation(1)
                    if MODE in {"train", "watch"} and os.path.isfile(TRAINING_START_STATE):
                        # Validate that the canonical savestate matches the chosen curriculum stage.
                        try:
                            start_state = self._read_game_state()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Training start state is not readable as an active Emerald game: {exc}. "
                                "Recreate it with ./run.sh --profile setup."
                            ) from exc
                        validation_error = _training_start_validation_error(start_state)
                        if validation_error:
                            raise RuntimeError(
                                f"Training start state does not satisfy stage '{TRAINING_START_STAGE}': "
                                f"{validation_error}. Recreate it with ./run.sh --profile setup and press K."
                            )
                        print(
                            f"[Start State] Validated stage={TRAINING_START_STAGE} | "
                            f"party_count={start_state['party_count']} | map_id={start_state['map_id']} | "
                            f"player=({start_state['player_x']},{start_state['player_y']})",
                            flush=True,
                        )
                else:
                    print(
                        f"[New] Cold start: advancing {INITIAL_BOOT_FRAMES} frames so the first observation is rendered.",
                        flush=True,
                    )
                    # _init_emulator() already attached the framebuffer and reset this core.
                    # Do not reset it a second time here; simply advance it as in the verified
                    # standalone mGBA probe.
                    self._advance_frame_for_observation(INITIAL_BOOT_FRAMES)

                self.episode_step_counter = 0
                print(f"[Reset {self.env_id}] building initial observation", flush=True)
                obs = self._get_obs()
                print(
                    f"[Reset {self.env_id}] observation ready: {obs.shape}, dtype={obs.dtype}, "
                    f"range={int(obs.min())}..{int(obs.max())}",
                    flush=True,
                )
                # Show the game before native snapshot or RAM-reward operations, so a later stall
                # cannot leave the window looking like video never worked.
                if GUI_MODE:
                    self._render_gui(time.perf_counter())

                print(f"[Reset {self.env_id}] capturing episode-start state", flush=True)
                self._capture_episode_start()
                print(f"[Reset {self.env_id}] episode-start state captured", flush=True)
                self._first_reset = False
            else:
                print(f"[Reset {self.env_id}] restoring episode-start state", flush=True)
                restored_episode = self._restore_episode_start()
                print(f"[Reset {self.env_id}] episode restore result: {restored_episode}", flush=True)
                self._advance_frame_for_observation(1 if restored_episode else INITIAL_BOOT_FRAMES)
                self.episode_step_counter = 0
                obs = self._get_obs()
                if GUI_MODE:
                    self._render_gui(time.perf_counter())

            # Episodes restore the same canonical start state. Novelty/story bookkeeping
            # must therefore be episode-local: the same valid route should earn its
            # feedback again in a new episode, while loops inside one episode cannot farm it.
            self.visited_maps.clear()
            self.visited_tiles.clear()
            self.seen_story_flags.clear()
            print(f"[Reset {self.env_id}] refreshing reward baseline", flush=True)
            self._refresh_reward_baseline()
            print(
                f"[Reset {self.env_id}] complete in {time.perf_counter() - reset_started:.2f}s",
                flush=True,
            )
            self._audio_accepting = bool(
                AUDIO_OUTPUT_ENABLED and GUI_MODE and self.audio_output is not None and
                self.audio_output.output_active
            )
            self._record_audio_active = bool(
                self.video_writer is not None
                and VIDEO_INCLUDE_AUDIO
                and self.audio_output is not None
                and self.audio_output.recording_active
            )
            return obs, {"episode_step": self.episode_step_counter}
        finally:
            faulthandler.cancel_dump_traceback_later()
    def step(self, action):
        if self.stop_requested or STOP_REQUESTED:
            self.stop_requested = True
            return self._get_obs(), 0.0, False, True, {"stopped": True}

        self.step_counter += 1
        self.episode_step_counter += 1
        # Setup mode is manual-only. Watch mode optionally permits one-step host overrides.
        if HOST_CONTROL_ENABLED and MODE in {"watch", "setup"}:
            if self.manual_action_next is not None:
                action = self.manual_action_next
                self.manual_action_next = None
            elif MODE == "setup":
                action = None  # neutral input until a human presses a control key

        if self.core is not None:
            run_frame = _method(self.core, "run_frame", "runFrame")
            if run_frame is None:
                raise RuntimeError("The installed mGBA core does not expose runFrame() or run_frame().")

            for _ in range(max(1, ACTION_REPEAT)):
                self._set_action_keys(action)
                frame_started = time.perf_counter()
                run_frame()
                self._pump_audio_buffer(
                    accept=self._audio_accepting,
                    record=self._record_audio_active,
                )

                # Refresh the window for each emulated frame (60 Hz) while the PPO policy
                # still chooses an action every ACTION_REPEAT frames.
                if GUI_MODE:
                    self._last_raw_frame_rgb = self._read_framebuffer()
                    self._render_gui(frame_started)
                    if self.stop_requested or STOP_REQUESTED:
                        break
                elif MODE == "watch" and WATCH_FPS_CAP > 0:
                    # Optional headless frame cap; zero means fastest possible emulation.
                    delay = (1.0 / WATCH_FPS_CAP) - (time.perf_counter() - frame_started)
                    if delay > 0:
                        time.sleep(delay)
            # Release keys after the action so button taps can be recognized by the game.
            self._set_action_keys(None)

        # Save after releasing the previous action's key, avoiding a snapshot with a held key.
        if MODE == "setup" and self._save_training_start_requested:
            self._save_training_start_requested = False
            try:
                setup_state = self._read_game_state()
                validation_error = _training_start_validation_error(setup_state)
                if validation_error:
                    print(
                        f"[Setup] Not saved for stage '{TRAINING_START_STAGE}': {validation_error}.",
                        flush=True,
                    )
                elif self._save_state_file(TRAINING_START_STATE):
                    print(
                        f"[Setup] Training start savestate saved for stage '{TRAINING_START_STAGE}': "
                        f"{TRAINING_START_STATE}",
                        flush=True,
                    )
                    if TRAINING_START_STAGE == "clock_set":
                        print(
                            "[Setup] This stage intentionally starts before the first Pokémon; tile exploration "
                            "rewards are active before obtaining a starter.", flush=True
                        )
                    print(
                        "[Setup] For a clean experiment, back up/move emerald_ppo_checkpoint.zip, "
                        "then run ./run.sh --profile fast_train.", flush=True
                    )
                else:
                    print(f"[Setup Error] Could not save training start state to {TRAINING_START_STATE}.", flush=True)
            except Exception as exc:
                print(f"[Setup] Not saved because the game state could not be validated: {exc}", flush=True)

        obs = self._get_obs()

        # --- Comprehensive Dense Reward Shaping (US v1.0 BPEE RAM Mapping) ---
        # Rewards are based on changes from the preceding step; the step penalty is constant, not cumulative.
        reward = -STEP_PENALTY
        info = {"episode_step": self.episode_step_counter}
        if self.core is not None:
            try:
                state = self._read_game_state()

                # 1. Small per-decision cost; it should not swamp progress over 20k actions.
                neg_steps_reward = -STEP_PENALTY

                # 2. Reward increases in combined party level, never decreases.
                previous_party_levels = self.prev_party_levels
                level_delta = state["party_levels"] - previous_party_levels
                level_reward = max(0, level_delta) * LEVEL_REWARD_PER_LEVEL
                if MODE == "watch" and state["party_levels"] // 10 > previous_party_levels // 10:
                    send_notification(
                        "Pokémon Training Milestone",
                        f"Combined party level reached {state['party_levels']}."
                    )
                self.prev_party_levels = state["party_levels"]

                # Badge bits come from the official persistent FLAG_BADGE01_GET..08_GET flags.
                new_badge_bits = state["badge_flags"] & (~self.prev_badges & 0xFF)
                new_badges = int(new_badge_bits).bit_count()
                badge_reward = BADGE_REWARD * new_badges
                self.prev_badges = state["badge_flags"]

                # 3. One-time reward for discovering a map.
                map_id = state["map_id"]
                new_area_discovered = map_id not in self.visited_maps
                exploration_reward = NEW_MAP_REWARD if new_area_discovered else 0.0
                if new_area_discovered:
                    self.visited_maps.add(map_id)
                    if MODE == "watch":
                        send_notification("New Pokémon Emerald Area", f"The AI discovered map ID {map_id}.")
                self.prev_map_id = map_id

                # 4. One-time tile novelty is enabled even before the first Pokémon. That is
                # essential for the clock_set curriculum, where reaching the starter is the task.
                # Skip invalid positions rather than rewarding garbage RAM values.
                player_x = state.get("player_x")
                player_y = state.get("player_y")
                tile_reward = 0.0
                new_tile_discovered = False
                if player_x is not None and player_y is not None:
                    tile_key = (map_id, int(player_x), int(player_y))
                    if tile_key not in self.visited_tiles:
                        self.visited_tiles.add(tile_key)
                        tile_reward = NEW_TILE_REWARD
                        new_tile_discovered = True

                # 5. Reward curated permanent story flags on the first transition only.
                current_story_flags = state.get("story_flags", set())
                new_story_flags = current_story_flags - self.seen_story_flags
                story_reward = sum(STORY_FLAG_REWARDS.get(flag, 0.0) for flag in new_story_flags)
                self.seen_story_flags.update(current_story_flags)

                # Party expansion is a useful intermediate milestone (receiving/catching Pokémon).
                party_members_added = max(0, state["party_count"] - self.prev_caught_count)
                party_reward = PARTY_MEMBER_REWARD * party_members_added
                if party_members_added:
                    self.last_notified_event = "party_growth"
                    if MODE == "watch":
                        send_notification(
                            "🎉 Pokémon Party Expanded",
                            f"Party size increased by {party_members_added}.",
                        )
                self.prev_caught_count = state["party_count"]

                if new_badges:
                    self.last_notified_event = "gym_badge"
                    send_notification("🏆 Gym Victory Milestone!", f"The AI appears to have acquired {new_badges} new badge(s).")

                # Aggregate reward components. Novelty/story sets are episode-local, so each
                # restarted episode can learn a route again, but loops cannot repeatedly reward
                # the same tile/flag within a single episode.
                reward = (
                    neg_steps_reward + level_reward + badge_reward + exploration_reward
                    + tile_reward + story_reward + party_reward
                )
                info.update({
                    "party_count": state["party_count"],
                    "party_levels": state["party_levels"],
                    "new_badges": new_badges,
                    "map_id": map_id,
                    "player_x": player_x if player_x is not None else -1,
                    "player_y": player_y if player_y is not None else -1,
                    "new_area_discovered": new_area_discovered,
                    "new_tile_discovered": new_tile_discovered,
                    "level_reward": level_reward,
                    "badge_reward": badge_reward,
                    "exploration_reward": exploration_reward,
                    "tile_reward": tile_reward,
                    "story_reward": story_reward,
                    "party_reward": party_reward,
                    "party_members_added": party_members_added,
                })
                self._memory_read_warning_printed = False
            except Exception as e:
                # Safe fallback reward if memory reads fluctuate; log the issue once so mapping errors are visible.
                if not self._memory_read_warning_printed:
                    print(f"[RAM Warning] Reward memory read failed; only step penalty will be used: {e}")
                    self._memory_read_warning_printed = True
                reward = -STEP_PENALTY

        # Record frame if enabled
        if self.video_writer is not None and self._last_raw_frame_rgb is not None:
            frame_rgb = cv2.resize(
                self._last_raw_frame_rgb, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

        # The GUI was rendered immediately after frame acquisition above.
        terminated = False
        truncated = MODE != "setup" and self.episode_step_counter >= MAX_EPISODE_STEPS
        if self.stop_requested or STOP_REQUESTED:
            truncated = True
            info["stopped"] = True
        elif truncated:
            info["TimeLimit.truncated"] = True

        return obs, float(reward), terminated, truncated, info


    def _decode_gui_key(self, key):
        """Handle window close/pause and return an optional manual action index."""
        if key < 0:
            return None
        normalized = key & 0xFF

        if normalized in (ord("q"), ord("Q"), 27):
            self.stop_requested = True
            print("[GUI] Quit requested; saving checkpoints and shutting down cleanly...")
            return None

        if normalized in (ord("p"), ord("P")):
            self.paused = not self.paused
            print("[GUI] Paused." if self.paused else "[GUI] Resumed.")
            return None

        if normalized in (ord("m"), ord("M")):
            self._toggle_audio_mute()
            return None

        if MODE == "setup" and normalized in (ord("k"), ord("K")):
            self._save_training_start_requested = True
            print("[Setup] Training-start capture queued; saving after the current action is released...", flush=True)
            return None

        if not (HOST_CONTROL_ENABLED and MODE in {"watch", "setup"}):
            return None

        # OpenCV's waitKeyEx arrow codes differ slightly by platform; cover common X11/Linux and waitKey codes.
        key_to_action = {
            ord("w"): 0, ord("W"): 0, 65362: 0, 82: 0,
            ord("s"): 1, ord("S"): 1, 65364: 1, 84: 1,
            ord("a"): 2, ord("A"): 2, 65361: 2, 81: 2,
            ord("d"): 3, ord("D"): 3, 65363: 3, 83: 3,
            ord("z"): 4, ord("Z"): 4, 13: 4, 10: 4,
            ord("x"): 5, ord("X"): 5, 8: 5, 127: 5,
            ord("1"): 6, ord("2"): 7,
        }
        return key_to_action.get(key, key_to_action.get(normalized))

    def _poll_gui_key(self, wait_ms=1):
        try:
            key = cv2.waitKeyEx(wait_ms)
        except Exception:
            key = cv2.waitKey(wait_ms)
        manual_action = self._decode_gui_key(key)
        if manual_action is not None:
            self.manual_action_next = manual_action

    def _render_gui(self, frame_started):
        if not self.window_created or self._last_raw_frame_rgb is None:
            return
        try:
            width, height = 240 * GUI_SCALE, 160 * GUI_SCALE
            frame = cv2.resize(
                self._last_raw_frame_rgb,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            status = "PAUSED" if self.paused else ("HUMAN SETUP" if MODE == "setup" else ("HOST CONTROL" if HOST_CONTROL_ENABLED and MODE == "watch" else "AI CONTROL"))
            cv2.putText(frame_bgr, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 240, 80), 2, cv2.LINE_AA)

            # The rectangle is both a visible status indicator and a clickable mute button.
            label = self._audio_gui_label()
            x1, y1, x2, y2 = width - 218, 8, width - 10, 45
            self._audio_button_rect = (x1, y1, x2, y2)
            if label == "AUDIO: ON":
                button_color = (40, 115, 40)
            elif label == "AUDIO: MUTED":
                button_color = (60, 60, 175)
            else:
                button_color = (70, 70, 70)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), button_color, thickness=-1)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (235, 235, 235), thickness=1)
            cv2.putText(frame_bgr, label, (x1 + 8, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            if self.audio_output is not None and self.audio_output.output_active:
                mute_hint = "M/click Audio: mute" if self._mouse_callback_available else "M: mute"
            else:
                mute_hint = "M: mute"
            if MODE == "setup":
                footer = f"Arrows/WASD move | Z/Enter=A | X/Backspace=B | 1=Start | 2=Select | K=save start | Q=quit | P=pause | {mute_hint}"
            else:
                footer = f"Q/Esc: quit | P: pause | {mute_hint}"
            cv2.putText(frame_bgr, footer, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45 if MODE == "setup" else 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(GUI_WINDOW_TITLE, frame_bgr)

            try:
                visible = cv2.getWindowProperty(GUI_WINDOW_TITLE, cv2.WND_PROP_VISIBLE)
                if visible < 1:
                    self.stop_requested = True
                    return
            except Exception:
                pass

            self._poll_gui_key(1)
            while self.paused and not self.stop_requested and not STOP_REQUESTED:
                self._poll_gui_key(50)
                try:
                    if cv2.getWindowProperty(GUI_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        self.stop_requested = True
                        break
                except Exception:
                    pass

            elapsed = time.perf_counter() - frame_started
            delay = max(0.0, (1.0 / max(1, GUI_FPS)) - elapsed)
            if delay > 0 and not self.paused:
                time.sleep(delay)
        except Exception as e:
            print(f"[GUI Warning] Could not render GUI frame: {e}", flush=True)

    def _finalize_video_recording(self):
        """Transcode temporary OpenCV video to H.264 and mux the captured WAV with FFmpeg."""
        if not self.video_writer:
            # The writer was set to None after release, so existence of the temp file
            # is the authoritative indication that this environment recorded video.
            if not os.path.isfile(VIDEO_INTERMEDIATE_PATH):
                return
        if not os.path.isfile(VIDEO_INTERMEDIATE_PATH) or os.path.getsize(VIDEO_INTERMEDIATE_PATH) == 0:
            print("[Recorder Warning] Temporary video is missing or empty; cannot finalize.", flush=True)
            return

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                os.replace(VIDEO_INTERMEDIATE_PATH, VIDEO_OUTPUT_PATH)
                print(
                    f"[Recorder Warning] ffmpeg not found; saved video without transcoding/muxing: {VIDEO_OUTPUT_PATH}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Recorder Error] Could not publish temporary video: {exc}", flush=True)
            return

        has_audio = bool(
            VIDEO_INCLUDE_AUDIO
            and os.path.isfile(VIDEO_AUDIO_TEMP_PATH)
            and os.path.getsize(VIDEO_AUDIO_TEMP_PATH) > 44
        )
        command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", VIDEO_INTERMEDIATE_PATH]
        if has_audio:
            command += ["-i", VIDEO_AUDIO_TEMP_PATH]
        command += ["-map", "0:v:0"]
        if has_audio:
            command += ["-map", "1:a:0", "-af", "apad"]
        else:
            command += ["-an"]
        command += [
            "-c:v", "libx264", "-preset", VIDEO_H264_PRESET,
            "-crf", str(VIDEO_H264_CRF), "-pix_fmt", "yuv420p",
        ]
        if has_audio:
            command += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-shortest"]
        command += ["-movflags", "+faststart", VIDEO_MUX_TEMP_PATH]

        try:
            if os.path.exists(VIDEO_MUX_TEMP_PATH):
                os.remove(VIDEO_MUX_TEMP_PATH)
            print(
                f"[Recorder] Finalizing video with H.264 CRF {VIDEO_H264_CRF}"
                f"{' + AAC audio' if has_audio else ' (no audio track)'}...",
                flush=True,
            )
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=600, check=False,
            )
            if result.returncode != 0 or not os.path.isfile(VIDEO_MUX_TEMP_PATH) or os.path.getsize(VIDEO_MUX_TEMP_PATH) == 0:
                detail = (result.stderr or result.stdout or "FFmpeg produced no output").strip()
                raise RuntimeError(detail[-3000:])
            os.replace(VIDEO_MUX_TEMP_PATH, VIDEO_OUTPUT_PATH)
            print(f"[Recorder] Final video saved: {VIDEO_OUTPUT_PATH}", flush=True)
            if os.path.isfile(VIDEO_AUDIO_TEMP_PATH):
                os.remove(VIDEO_AUDIO_TEMP_PATH)
            os.remove(VIDEO_INTERMEDIATE_PATH)
        except Exception as exc:
            print(
                f"[Recorder Warning] H.264/AAC finalization failed: {exc}. "
                "Preserving the original recorded video as fallback.",
                flush=True,
            )
            try:
                if os.path.isfile(VIDEO_MUX_TEMP_PATH):
                    os.remove(VIDEO_MUX_TEMP_PATH)
                # Preserve the raw video rather than lose the recording. If final path
                # already exists, use a fallback name so the previous file is untouched.
                fallback_path = VIDEO_OUTPUT_PATH
                if os.path.exists(fallback_path):
                    root, ext = os.path.splitext(fallback_path)
                    fallback_path = root + ".uncompressed" + ext
                os.replace(VIDEO_INTERMEDIATE_PATH, fallback_path)
                print(f"[Recorder] Fallback video preserved: {fallback_path}", flush=True)
            except Exception as fallback_exc:
                print(f"[Recorder Error] Could not preserve fallback video: {fallback_exc}", flush=True)

    def save_state(self):
        print(f"\n[Save] Saving progress at step {self.step_counter}...")
        if self.core is not None:
            if self._save_state_file(self.checkpoint_file):
                print(f"[Save] Emulator state saved to {self.checkpoint_file}")
            else:
                print("[Save Warning] Emulator state could not be saved by this mGBA binding.")
        if self.video_writer is not None:
            # The writer is released by close(); do not claim a video is finalized before that happens.
            print(f"[Recorder] Video will be finalized on close: {VIDEO_OUTPUT_PATH}")

    def close(self):
        if self.audio_output is not None:
            try:
                self.audio_output.close()
                print("[Audio] Output stream closed.", flush=True)
            except Exception as exc:
                print(f"[Audio Warning] Could not close output cleanly: {exc}", flush=True)
            self.audio_output = None

        had_video_writer = self.video_writer is not None
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                print(f"[Recorder] Temporary video writer released for {VIDEO_INTERMEDIATE_PATH}")
            except Exception as e:
                print(f"[Recorder Warning] Could not release video writer: {e}")
            self.video_writer = None
        if had_video_writer:
            self._finalize_video_recording()

        if self.window_created:
            try:
                cv2.destroyWindow(GUI_WINDOW_TITLE)
                cv2.waitKey(1)
            except Exception:
                pass
            self.window_created = False

        if self.core is not None:
            close_core = _method(self.core, "close", "deinit")
            if close_core is not None:
                try:
                    close_core()
                except Exception:
                    pass
            self.core = None

        try:
            if os.path.isfile(self._screenshot_fallback_path):
                os.remove(self._screenshot_fallback_path)
        except OSError:
            pass

        # Release bounded snapshots/CFFI buffers after shutdown to lower retained memory.
        self._episode_start_buffer = None
        self._framebuffer = None
        self._framebuffer_ffi = None
        self._last_raw_frame_rgb = None
        self.visited_maps.clear()


def atomic_save_model(model, checkpoint_base=MODEL_CHECKPOINT):
    """Save an SB3 model to an explicit .zip temp path, validate it, then publish atomically.

    Passing a filename that already ends with ``.zip`` is deliberate: Stable-Baselines3
    versions can normalize suffixes differently. The writer and validation now refer to
    the exact same path, preventing a valid archive with a surprising filename from
    being mistaken for a failed save.
    """
    final_path = f"{checkpoint_base}.zip"
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    # The temp filename has a real .zip suffix and lives outside the project root.
    temp_path = os.path.join(TEMP_DIR, os.path.basename(checkpoint_base) + ".tmp.zip")
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Give SB3 the exact archive path that will be checked below.
        model.save(temp_path)
        if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError(
                f"Stable-Baselines3 did not create the expected temporary archive: {temp_path}"
            )
        if not zipfile.is_zipfile(temp_path):
            raise RuntimeError(
                f"Temporary checkpoint is not a valid ZIP archive: {temp_path}"
            )
        with zipfile.ZipFile(temp_path, "r") as archive:
            names = set(archive.namelist())
            if not names or "data" not in names:
                raise RuntimeError(
                    f"Temporary archive does not look like an SB3 model ZIP: {temp_path}"
                )
        with open(temp_path, "rb") as f:
            os.fsync(f.fileno())

        if os.path.isfile(final_path):
            backup_path = final_path + ".bak"
            shutil.copy2(final_path, backup_path)
        os.replace(temp_path, final_path)
        _fsync_parent_dir(final_path)
        return final_path
    except Exception:
        # Keep the previous final checkpoint untouched. Remove only our known temp path.
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise


# --- Auto-Start TensorBoard Server ---
def _tcp_port_is_open(host, port, timeout=0.35):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def start_tensorboard():
    global tb_process, tb_owned, tb_log_handle
    if not (AUTO_START_TENSORBOARD and MODE == "train"):
        return
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    if _tcp_port_is_open("127.0.0.1", TENSORBOARD_PORT):
        print(
            f"[TensorBoard] Port {TENSORBOARD_PORT} already has a listener; "
            f"assuming TensorBoard is already running at http://127.0.0.1:{TENSORBOARD_PORT}. "
            "This process will not stop that external server.",
            flush=True,
        )
        return

    tb_log_path = os.path.join(LOG_DIR, "tensorboard_server.log")
    os.makedirs(os.path.dirname(tb_log_path), exist_ok=True)
    try:
        tb_log_handle = open(tb_log_path, "a", encoding="utf-8", buffering=1)
        tb_process = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard.main",
                f"--logdir={TENSORBOARD_LOG_DIR}",
                f"--port={TENSORBOARD_PORT}",
                "--host=127.0.0.1",
            ],
            stdout=tb_log_handle,
            stderr=subprocess.STDOUT,
        )
        tb_owned = True
        time.sleep(0.5)
        if tb_process.poll() is not None:
            print(
                f"[TensorBoard Warning] Server exited during startup; inspect {tb_log_path}.",
                flush=True,
            )
        else:
            print(f"[TensorBoard] Starting at http://127.0.0.1:{TENSORBOARD_PORT}", flush=True)
            print(f"[TensorBoard] Server log: {tb_log_path}", flush=True)
    except Exception as exc:
        print(f"[TensorBoard Warning] Could not start TensorBoard: {exc}", flush=True)
        if tb_log_handle is not None:
            try:
                tb_log_handle.close()
            except Exception:
                pass
            tb_log_handle = None
        tb_process = None
        tb_owned = False


# --- Graceful Exit Handler ---
def handle_exit(sig, frame):
    global STOP_REQUESTED
    # Do not call sys.exit() here: let the training/watch loop save models and emulator state in its finally block.
    STOP_REQUESTED = True
    print("\n[Interrupt] Stop requested. Saving checkpoints and cleaning up resources...")


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


class StopAndCheckpointCallback(BaseCallback):
    """Report actionable progress, log metrics to TensorBoard, and checkpoint safely."""

    def __init__(self, save_frequency=CHECKPOINT_SAVE_FREQUENCY, verbose=0):
        super().__init__(verbose)
        self.save_frequency = max(1, int(save_frequency))
        self.progress_frequency = max(1, int(PROGRESS_LOG_FREQUENCY))
        self._last_save_timestep = 0
        self._last_progress_timestep = 0
        self._run_start_timestep = 0
        self._run_started_at = 0.0
        self._recent_rewards = deque(maxlen=1000)
        self._last_step_reward = 0.0
        self._last_action = None
        self._completed_episodes = 0
        self._last_episode_reward = None
        self._last_episode_length = None
        self._episode_running_rewards = np.zeros(1, dtype=np.float64)

    def _on_training_start(self):
        # On resumed training, num_timesteps can already be very large. Anchor the
        # interval to this run's starting value so we do not save again on its first step.
        self._run_start_timestep = int(self.num_timesteps)
        self._last_save_timestep = int(self.num_timesteps)
        self._last_progress_timestep = int(self.num_timesteps)
        self._run_started_at = time.perf_counter()
        self._recent_rewards.clear()
        env_count = int(getattr(self.training_env, "num_envs", NUM_CPU))
        self._episode_running_rewards = np.zeros(max(1, env_count), dtype=np.float64)
        print(
            f"[Training] Started | mode={'GUI' if GUI_MODE else 'HEADLESS'} | "
            f"envs={env_count} | action_repeat={ACTION_REPEAT} | "
            f"SB3 verbosity={VERBOSE_LEVEL} | mGBA log mask={_MGBA_LOG_LEVEL_MASK} | "
            f"checkpoint every {self.save_frequency:,} transitions | "
            f"progress every {self.progress_frequency:,} transitions.",
            flush=True,
        )

    def _on_step(self):
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        if rewards.size:
            self._last_step_reward = float(np.mean(rewards))
            self._recent_rewards.extend(float(reward) for reward in rewards)
            if self._episode_running_rewards.size != rewards.size:
                self._episode_running_rewards = np.resize(
                    self._episode_running_rewards, rewards.size
                ).astype(np.float64, copy=False)
            self._episode_running_rewards[:rewards.size] += rewards

        actions = self.locals.get("actions")
        if actions is not None:
            try:
                action_values = np.asarray(actions).reshape(-1)
                if action_values.size:
                    self._last_action = ACTION_LABELS.get(int(action_values[-1]), str(action_values[-1]))
            except (TypeError, ValueError, OverflowError):
                pass

        infos = self.locals.get("infos", [])
        latest_info = {}
        infos_list = infos if isinstance(infos, (list, tuple)) else []
        for env_index, info in enumerate(infos_list):
            if not isinstance(info, dict):
                continue
            latest_info = info
            episode = info.get("episode")
            if isinstance(episode, dict) and "r" in episode:
                self._completed_episodes += 1
                self._last_episode_reward = float(episode["r"])
                self._last_episode_length = int(episode.get("l", 0))

        # Monitor/VecEnv marks a finished episode in dones. Reset its live return
        # after including that transition's reward, so the display tracks each new episode.
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        for env_index, done in enumerate(dones[:self._episode_running_rewards.size]):
            if done:
                self._episode_running_rewards[env_index] = 0.0

        # Custom scalars appear under TensorBoard's progress namespace. SB3 flushes
        # these alongside its normal PPO metrics at the end of each rollout/update.
        self.logger.record("progress/step_reward", self._last_step_reward)
        self.logger.record("progress/completed_episodes", self._completed_episodes)
        if self._recent_rewards:
            self.logger.record(
                "progress/recent_mean_reward_1000_steps",
                float(np.mean(self._recent_rewards)),
            )
        if self._last_episode_reward is not None:
            self.logger.record("progress/last_episode_reward", self._last_episode_reward)
        if self._last_episode_length is not None:
            self.logger.record("progress/last_episode_length", self._last_episode_length)
        live_episode_reward_mean = (
            float(np.mean(self._episode_running_rewards))
            if self._episode_running_rewards.size else 0.0
        )
        self.logger.record("progress/live_episode_reward_mean", live_episode_reward_mean)
        for env_index, running_reward in enumerate(self._episode_running_rewards):
            self.logger.record(
                f"progress/live_episode_reward_env_{env_index}", float(running_reward)
            )

        elapsed = max(1e-6, time.perf_counter() - self._run_started_at)
        steps_per_second = (int(self.num_timesteps) - self._run_start_timestep) / elapsed
        self.logger.record("progress/transitions_per_second", float(steps_per_second))
        for info_key, tb_key in (
            ("party_count", "progress/party_count"),
            ("party_levels", "progress/party_total_levels"),
            ("new_badges", "progress/new_badges_this_step"),
            ("map_id", "progress/map_id"),
            ("level_reward", "progress/level_reward"),
            ("badge_reward", "progress/badge_reward"),
            ("exploration_reward", "progress/exploration_reward"),
            ("tile_reward", "progress/tile_reward"),
            ("story_reward", "progress/story_reward"),
            ("party_reward", "progress/party_reward"),
            ("party_members_added", "progress/party_members_added"),
            ("new_tile_discovered", "progress/new_tile_discovered"),
            ("player_x", "progress/player_x"),
            ("player_y", "progress/player_y"),
        ):
            value = latest_info.get(info_key)
            if isinstance(value, (int, float, bool, np.integer, np.floating)):
                self.logger.record(tb_key, float(value))

        if int(self.num_timesteps) - self._last_progress_timestep >= self.progress_frequency:
            recent_mean = float(np.mean(self._recent_rewards)) if self._recent_rewards else 0.0
            action_text = self._last_action or "unknown"
            run_transitions = max(0, int(self.num_timesteps) - self._run_start_timestep)
            run_target = max(1, int(TOTAL_TIMESTEPS))
            progress_percent = min(100.0, 100.0 * run_transitions / run_target)
            remaining_seconds = max(0, run_target - run_transitions) / max(steps_per_second, 1e-9)
            elapsed_seconds = max(0.0, time.perf_counter() - self._run_started_at)

            env_count = int(getattr(self.training_env, "num_envs", NUM_CPU))
            processes = getattr(self.training_env, "processes", None)
            if processes is not None:
                try:
                    alive_workers = sum(1 for process in processes if process.is_alive())
                    env_status = f"envs={env_count}, workers_alive={alive_workers}/{len(processes)}"
                except Exception:
                    env_status = f"envs={env_count}, workers_alive=unknown"
            else:
                env_status = f"envs={env_count} (in-process)"

            per_env_speed = steps_per_second / max(1, env_count)
            aggregate_gba_fps = steps_per_second * max(1, ACTION_REPEAT)
            if self._last_episode_reward is not None:
                episode_text = f", last_episode_reward={self._last_episode_reward:.2f}"
            else:
                episode_text = ", last_episode_reward=n/a"
            live_values = ", ".join(
                f"env{env_index}={running_reward:.2f}"
                for env_index, running_reward in enumerate(self._episode_running_rewards)
            )
            if len(self._episode_running_rewards) > 1:
                live_reward_text = (
                    f"live_episode_reward_mean={live_episode_reward_mean:.2f} "
                    f"[{live_values}]"
                )
            else:
                live_reward_text = f"live_episode_reward={live_episode_reward_mean:.2f}"
            print(
                f"[Training] run={run_transitions:,}/{run_target:,} ({progress_percent:.2f}%) | "
                f"total_timesteps={int(self.num_timesteps):,} | "
                f"speed={steps_per_second:.1f} transitions/s aggregate "
                f"({per_env_speed:.1f}/env/s; ~{aggregate_gba_fps:.1f} GBA frames/s aggregate) | "
                f"{env_status} | elapsed={_format_duration(elapsed_seconds)} "
                f"ETA={_format_duration(remaining_seconds)} | action={action_text} | "
                f"step_reward={self._last_step_reward:.3f}, mean_reward_1k={recent_mean:.3f} | "
                f"{live_reward_text} | episodes={self._completed_episodes}{episode_text}",
                flush=True,
            )
            # Flush the custom progress scalars now instead of waiting for PPO's full
            # 2,048-step rollout. SB3 will still emit its normal training metrics later.
            try:
                self.logger.dump(int(self.num_timesteps))
            except Exception as exc:
                print(f"[TensorBoard Warning] Could not flush progress metrics: {exc}", flush=True)
            self._last_progress_timestep = int(self.num_timesteps)

        if STOP_REQUESTED:
            return False

        if GUI_MODE:
            try:
                if any(self.training_env.get_attr("stop_requested")):
                    return False
            except Exception:
                pass

        # Mark the attempt before I/O. A failure should not trigger a disk-heavy retry
        # on every single step; the next attempt is at the next configured interval.
        if int(self.num_timesteps) - self._last_save_timestep >= self.save_frequency:
            checkpoint_timestep = int(self.num_timesteps)
            self._last_save_timestep = checkpoint_timestep
            try:
                saved_model = atomic_save_model(self.model, MODEL_CHECKPOINT)
                print(
                    f"[Checkpoint] PPO model atomically saved to {saved_model} "
                    f"at timestep {checkpoint_timestep:,}.",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Checkpoint Warning] Could not save PPO model: {exc}", flush=True)
            else:
                try:
                    self.training_env.env_method("save_state")
                    print(
                        f"[Checkpoint] Emulator savestate(s) saved at timestep {checkpoint_timestep:,}.",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[Checkpoint Warning] PPO model is saved, but an emulator savestate failed: {exc}",
                        flush=True,
                    )
        return True

def _worker_state_path(env_id):
    """Give each emulator a separate savestate under runtime_data/states/."""
    env_id = int(env_id)
    if env_id == 0:
        return CHECKPOINT_STATE
    return os.path.join(STATE_DIR, f"emerald_save.worker_{env_id}.state")


def _migrate_legacy_runtime_files():
    """Move legacy root-level states/logs/temp artifacts into managed subfolders.

    Migration copies to a temporary destination, verifies the copy, atomically publishes
    it, and removes the root source only after success. Existing destination files are
    never overwritten; a non-identical legacy source is preserved under a legacy_imports
    subfolder so the project root stays clean without discarding data.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    moves = []

    if os.path.isfile(LEGACY_CHECKPOINT_STATE):
        moves.append((LEGACY_CHECKPOINT_STATE, CHECKPOINT_STATE))

    legacy_session_log = os.path.join(BASE_DIR, "session_time_log.txt")
    if os.path.isfile(legacy_session_log):
        moves.append((legacy_session_log, SESSION_LOG_PATH))

    # Support both historical worker naming conventions.
    for old_path in glob.glob(LEGACY_CHECKPOINT_STATE + ".worker_*"):
        name = os.path.basename(old_path)
        suffix = name.split(".worker_", 1)[-1]
        if suffix.isdigit():
            moves.append((old_path, os.path.join(STATE_DIR, f"emerald_save.worker_{suffix}.state")))
    for old_path in glob.glob(os.path.join(BASE_DIR, "emerald_save.worker_*.state")):
        moves.append((old_path, os.path.join(STATE_DIR, os.path.basename(old_path))))

    # File-based episode-start snapshots were only used by binding fallbacks.
    for old_path in glob.glob(LEGACY_CHECKPOINT_STATE + ".episode_start_*.state"):
        moves.append((old_path, os.path.join(STATE_DIR, os.path.basename(old_path))))
    for old_path in glob.glob(LEGACY_CHECKPOINT_STATE + ".worker_*.episode_start_*.state"):
        name = os.path.basename(old_path)
        try:
            worker_id = name.split(".worker_", 1)[1].split(".", 1)[0]
            episode_tail = name.split(".episode_start_", 1)[1]
            if worker_id.isdigit():
                new_name = f"emerald_save.worker_{worker_id}.state.episode_start_{episode_tail}"
                moves.append((old_path, os.path.join(STATE_DIR, new_name)))
        except (IndexError, ValueError):
            pass

    # Gather only recognized temporary/debug artifacts left by old revisions.
    for pattern in (
        "*.tmp",
        "*.tmp.*",
        "emerald_main_step_*.png",
        "emerald_ai_screen_*.png",
    ):
        for old_path in glob.glob(os.path.join(BASE_DIR, pattern)):
            if os.path.isfile(old_path):
                moves.append((old_path, os.path.join(TEMP_DIR, os.path.basename(old_path))))

    for source, destination in moves:
        if not os.path.isfile(source):
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        if os.path.exists(destination):
            try:
                if filecmp.cmp(source, destination, shallow=False):
                    os.remove(source)
                    print(f"[Runtime Cleanup] Removed duplicate legacy artifact {source}", flush=True)
                    continue
            except OSError:
                pass
            source_name = os.path.basename(source)
            if source == legacy_session_log:
                archive_dir = os.path.join(LOG_DIR, "legacy_imports")
            elif ".state" in source_name:
                archive_dir = os.path.join(STATE_DIR, "legacy_imports")
            else:
                archive_dir = os.path.join(TEMP_DIR, "legacy_imports")
            os.makedirs(archive_dir, exist_ok=True)
            archive_base = os.path.join(archive_dir, source_name + f".legacy-{TIMESTAMP_STR}")
            archive_path = archive_base
            index = 1
            while os.path.exists(archive_path):
                archive_path = f"{archive_base}-{index}"
                index += 1
            destination = archive_path

        migration_tmp = destination + ".migration.tmp"
        try:
            shutil.copy2(source, migration_tmp)
            if (
                os.path.getsize(migration_tmp) != os.path.getsize(source)
                or os.path.getsize(migration_tmp) == 0
            ):
                raise IOError("copied artifact failed size verification")
            os.replace(migration_tmp, destination)
            _fsync_parent_dir(destination)
            os.remove(source)
            if ".state" in os.path.basename(source):
                print(f"[State Migration] Moved legacy state into {destination}", flush=True)
            else:
                print(f"[Runtime Migration] Moved legacy artifact into {destination}", flush=True)
        except Exception as exc:
            try:
                if os.path.exists(migration_tmp):
                    os.remove(migration_tmp)
            except OSError:
                pass
            print(
                f"[Runtime Migration Warning] Could not migrate {source} to {destination}: {exc}. "
                "The original file has been preserved where possible.",
                flush=True,
            )


def _make_env(env_id):
    def _init():
        checkpoint_path = _worker_state_path(env_id)
        # On the first parallel run after an older single-state version, seed each new
        # worker from the last common state. Subsequent saves/restores remain per-worker.
        if (
            int(env_id) > 0
            and not os.path.isfile(checkpoint_path)
            and os.path.isfile(CHECKPOINT_STATE)
        ):
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            seed_temp = checkpoint_path + ".seed.tmp"
            try:
                shutil.copy2(CHECKPOINT_STATE, seed_temp)
                os.replace(seed_temp, checkpoint_path)
                print(
                    f"[State] Seeded worker {env_id} state from the existing shared checkpoint.",
                    flush=True,
                )
            except Exception as exc:
                try:
                    if os.path.exists(seed_temp):
                        os.remove(seed_temp)
                except OSError:
                    pass
                print(f"[State Warning] Could not seed worker {env_id} state: {exc}", flush=True)
        return Monitor(
            EmeraldEnv(
                rom_path=ROM_PATH,
                checkpoint_file=checkpoint_path,
                env_id=env_id,
            )
        )
    return _init


def _save_vector_env_state(vec_env):
    """Save each environment to its own path; worker paths are distinct in parallel mode."""
    if vec_env is None:
        return
    try:
        vec_env.env_method("save_state")
    except Exception as exc:
        print(f"[State Warning] Could not save all emulator states through vector environment: {exc}", flush=True)


def _shutdown_tensorboard():
    global tb_process, tb_owned, tb_log_handle
    if tb_process is not None and tb_owned:
        try:
            if tb_process.poll() is None:
                tb_process.terminate()
                try:
                    tb_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tb_process.kill()
                    tb_process.wait(timeout=2)
            print("[TensorBoard] This process's background server stopped.", flush=True)
        except Exception as exc:
            print(f"[TensorBoard Warning] Could not stop TensorBoard cleanly: {exc}", flush=True)
    tb_process = None
    tb_owned = False
    if tb_log_handle is not None:
        try:
            tb_log_handle.close()
        except Exception:
            pass
        tb_log_handle = None


# --- Main Execution Block ---
def _enable_run_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"run_{TIMESTAMP_STR}.log")
    log_handle = open(path, "a", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_stdout, log_handle, lock)
    sys.stderr = _TeeStream(old_stderr, log_handle, lock)
    print(f"[Log] Mirroring console output to {path}", flush=True)
    return log_handle, old_stdout, old_stderr


if __name__ == "__main__":
    env = None
    model = None
    model_file_path = f"{MODEL_CHECKPOINT}.zip"
    run_log_handle = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        run_log_handle, original_stdout, original_stderr = _enable_run_log()
    except Exception as exc:
        print(f"[Log Warning] Persistent console logging unavailable: {exc}", flush=True)

    try:
        _migrate_legacy_runtime_files()
        start_tensorboard()

        # Configure Vectorized Environments by mode. GUI runs use one in-process environment;
        # headless training uses subprocess workers with independent emulator savestates.
        print(
            f"[Environment Setup] Profile={ACTIVE_PROFILE} | Initializing {NUM_CPU} environment(s) "
            f"for [{MODE.upper()}] mode (GUI_MODE={GUI_MODE})...",
            flush=True,
        )
        env_fns = [_make_env(i) for i in range(NUM_CPU)]
        if NUM_CPU > 1 and not GUI_MODE:
            env = SubprocVecEnv(env_fns, start_method="spawn")
        else:
            env = DummyVecEnv(env_fns)

        if MODE == "train":
            print("--------------------------------------------------")
            print(f" [MODE] TRAINING ACTIVE ({NUM_CPU} Environment(s)). Press Ctrl+C or Q in GUI to stop and save.")
            print("--------------------------------------------------")

            if os.path.exists(model_file_path):
                print(f"[Load] Resuming training weights from {model_file_path}...")
                model = PPO.load(MODEL_CHECKPOINT, env=env, verbose=VERBOSE_LEVEL, tensorboard_log=TENSORBOARD_LOG_DIR)
            else:
                model = PPO("CnnPolicy", env, verbose=VERBOSE_LEVEL, tensorboard_log=TENSORBOARD_LOG_DIR)

            try:
                model.learn(
                    total_timesteps=TOTAL_TIMESTEPS,
                    reset_num_timesteps=False,
                    callback=StopAndCheckpointCallback(),
                )
            except KeyboardInterrupt:
                # Retained as a fallback; the signal handler normally sets STOP_REQUESTED instead.
                print("[Interrupt] KeyboardInterrupt received during training.")
            finally:
                try:
                    saved_model = atomic_save_model(model, MODEL_CHECKPOINT)
                    print(f"[Save] PPO model checkpoint atomically saved to {saved_model}", flush=True)
                except Exception as e:
                    print(f"[Save Error] Could not save PPO model checkpoint: {e}")

            if STOP_REQUESTED:
                print("[Stopped] Training stopped by user request; checkpoint saved.")
            else:
                print(f"[Complete] Training call finished. Model saved; configured timestep target was {TOTAL_TIMESTEPS}.")

        elif MODE == "setup":
            print("--------------------------------------------------")
            print(" [MODE] HUMAN SETUP. Control the game manually; this mode does not run PPO.")
            if TRAINING_START_STAGE == "clock_set":
                print(" [MODE] Set and confirm the bedroom clock, wait until the menu/dialogue closes, then press K.")
                print(" [MODE] You do not need a starter Pokémon for the clock_set stage.")
            else:
                print(" [MODE] Complete the introduction, obtain your starter, reach free movement, then press K.")
            print(f" [MODE] Selected stage={TRAINING_START_STAGE}; save path: {TRAINING_START_STATE}")
            print(" [MODE] Controls: arrows/WASD move, Z/Enter=A, X/Backspace=B, 1=Start, 2=Select, K=save, P=pause, Q/Esc=quit.")
            print("--------------------------------------------------", flush=True)
            env.reset()
            setup_steps = 0
            while not STOP_REQUESTED:
                # setup-mode step() ignores this placeholder unless a human key was pressed.
                env.step(np.zeros((NUM_CPU,), dtype=np.int64))
                setup_steps += 1
                try:
                    if any(env.get_attr("stop_requested")):
                        break
                except Exception:
                    pass
                if setup_steps % 30000 == 0:
                    print(f"[Setup] Manual setup active for {setup_steps:,} neutral/action steps.", flush=True)
            print("[Setup] Manual setup stopped. If you pressed K, the training start state is saved.", flush=True)

        elif MODE == "watch":
            print("--------------------------------------------------")
            print(f" [MODE] EVALUATION PLAYBACK ({NUM_CPU} Environment). Watching AI play.")
            print("--------------------------------------------------")

            if not os.path.exists(model_file_path):
                raise FileNotFoundError(
                    f"No trained model found at {model_file_path}. Train the AI first (MODE='train')."
                )

            model = PPO.load(MODEL_CHECKPOINT, env=env, device="cpu")
            print(f"[Load] Loaded model weights from {model_file_path}")
            send_notification("Pokémon AI Watch Mode", "Playback started. Q/Esc quits; P pauses/resumes.")

            if GUI_MODE:
                print("[Watch] GUI playback is paced by GUI_FPS; use POKEMON_AI_GUI_MODE=0 for headless playback.", flush=True)
            elif WATCH_FPS_CAP == 0:
                print(
                    f"[Watch] HEADLESS FAST-FORWARD | one emulator | no frame cap | "
                    f"video_recording={'ON' if RECORD_VIDEO else 'OFF'}.",
                    flush=True,
                )
            else:
                print(
                    f"[Watch] HEADLESS | one emulator | frame cap={WATCH_FPS_CAP:g} FPS | "
                    f"video_recording={'ON' if RECORD_VIDEO else 'OFF'}.",
                    flush=True,
                )

            obs = env.reset()
            watch_started = time.perf_counter()
            watch_last_report = watch_started
            watch_steps = 0
            watch_last_report_step = 0
            watch_completed_episodes = 0
            watch_episode_reward = 0.0
            watch_last_step_reward = 0.0
            watch_recent_rewards = deque(maxlen=PROGRESS_LOG_FREQUENCY)
            while not STOP_REQUESTED:
                action, _states = model.predict(obs, deterministic=(not STOCHASTIC_WATCH))
                obs, rewards, dones, infos = env.step(action)
                watch_steps += int(getattr(env, "num_envs", 1))
                reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
                if reward_values.size:
                    watch_last_step_reward = float(np.mean(reward_values))
                    watch_recent_rewards.extend(float(value) for value in reward_values)
                    watch_episode_reward += float(np.sum(reward_values))
                info_values = infos if isinstance(infos, (list, tuple)) else []
                done_values = np.asarray(dones, dtype=bool).reshape(-1)
                if done_values.any():
                    for i, done in enumerate(done_values):
                        if done:
                            watch_completed_episodes += 1
                            episode_info = info_values[i].get("episode") if i < len(info_values) and isinstance(info_values[i], dict) else None
                            if isinstance(episode_info, dict) and "r" in episode_info:
                                print(
                                    f"[Watch Episode] #{watch_completed_episodes} | "
                                    f"reward={float(episode_info['r']):.2f} | "
                                    f"length={int(episode_info.get('l', 0))}",
                                    flush=True,
                                )
                            watch_episode_reward = 0.0
                now = time.perf_counter()
                if watch_steps - watch_last_report_step >= PROGRESS_LOG_FREQUENCY:
                    elapsed = max(1e-6, now - watch_started)
                    speed = watch_steps / elapsed
                    recent_mean = float(np.mean(watch_recent_rewards)) if watch_recent_rewards else 0.0
                    print(
                        f"[Watch] steps={watch_steps:,} | speed={speed:.1f} transitions/s "
                        f"(~{speed * ACTION_REPEAT:.1f} GBA frames/s) | elapsed={_format_duration(elapsed)} | "
                        f"step_reward={watch_last_step_reward:.3f} | "
                        f"recent_mean_reward={recent_mean:.3f} | "
                        f"live_episode_reward={watch_episode_reward:.2f} | episodes={watch_completed_episodes}",
                        flush=True,
                    )
                    watch_last_report = now
                    watch_last_report_step = watch_steps
                if GUI_MODE:
                    try:
                        if any(env.get_attr("stop_requested")):
                            break
                    except Exception:
                        pass
            print("[Watch] Playback stopped.", flush=True)

        else:
            raise ValueError(f"Unsupported MODE={MODE!r}; choose 'train', 'watch', or 'setup'.")

    except KeyboardInterrupt:
        STOP_REQUESTED = True
        print("\n[Interrupt] KeyboardInterrupt received; continuing with cleanup.")
    except Exception as e:
        print(f"[Fatal Error] {e}")
        raise
    finally:
        # Save runtime session stopwatch log
        if env is not None:
            _save_vector_env_state(env)
            try:
                env.close()
            except Exception as e:
                print(f"[Cleanup Warning] Could not close vector environment cleanly: {e}")

        # Keep the warm-up core alive during training, then release it at shutdown.
        _close_bootstrap_core()
        session_timer.save_session()
        _shutdown_tensorboard()
        print("[Exit] Clean shutdown complete.", flush=True)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if run_log_handle is not None:
            try:
                run_log_handle.flush()
                run_log_handle.close()
            except Exception:
                pass
