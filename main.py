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
import math
from collections import Counter, deque


# Track environment provenance so the selected profile can replace stale mode settings
# from older .env files while still respecting explicitly exported shell variables.
_SHELL_ENV_KEYS_AT_START = set(os.environ)
_PROJECT_ENV_KEYS_LOADED = set()


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable without surprises."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


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
        "label": "Human-controlled gameplay setup, RAM diagnostics and training-start capture",
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
    "evaluate": {
        "label": "Deterministic fixed-start evaluation (five episodes)",
        "env": {
            "POKEMON_AI_MODE": "evaluate",
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
    "eval": "evaluate",
    "evaluation": "evaluate",
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
    """Identify routine native traces and narrowly known non-fatal mGBA noise.

    At normal verbosity, the optional known-warning filter also hides low-address
    BIOS/unmapped-memory diagnostics observed with this ROM/build and routine flash
    save-data status messages. These are console filters only; they do not change
    emulation or suppress Python exceptions. Set POKEMON_AI_VERBOSE=2, or disable
    POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS, to inspect the known diagnostics.
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
    # mGBA's RTC pak implementation prints one line for every RTC command and
    # output byte. These are routine emulation traces, not failures, and can flood
    # normal training logs. Respect the project's SB3 verbosity setting: hide only
    # these repetitive RTC lines at verbosity 0/1; verbosity 2 keeps them available
    # for native-emulator debugging. This stdout filter is separate from mGBA's
    # native severity mask and SB3's own logger.
    try:
        sb3_verbosity = int(os.environ.get("POKEMON_AI_VERBOSE", "1"))
    except (TypeError, ValueError):
        sb3_verbosity = 1
    if sb3_verbosity <= 1 and (
        payload.startswith(b"GBA Pak Hardware: Got RTC command ")
        or payload.startswith(b"GBA Pak Hardware: RTC output byte ")
    ):
        return True

    if filter_known_warnings and sb3_verbosity <= 1:
        # On this setup, absent/unmapped BIOS-space reads and a few low-address
        # invalid accesses are emitted repeatedly while the game uses BIOS routines.
        # Suppress only the known address ranges seen in the logs, leaving unrelated
        # bad-memory diagnostics available for investigation.
        bad_bios_prefix = b"GBA Memory: Bad BIOS Load"
        bad_memory_prefixes = (
            b"GBA Memory: Bad memory Load",
            b"GBA Memory: Bad memory Store",
        )
        if payload.startswith((bad_bios_prefix,) + bad_memory_prefixes):
            try:
                address_token = payload.rsplit(b"0x", 1)[1].split(None, 1)[0].rstrip(b"\r\n")
                address = int(address_token, 16)
            except (IndexError, ValueError):
                address = -1

            if payload.startswith(bad_bios_prefix) and 0 <= address < 0x2000:
                return True
            if payload.startswith(bad_memory_prefixes) and (
                0 <= address < 0x2000 or address == 0xE55EC003
            ):
                return True

        # These are routine mGBA flash-save implementation traces, not save failures.
        routine_savedata_prefixes = (
            b"GBA Savedata: Performing flash bank switch to bank ",
            b"GBA Savedata: Performing flash sector erase at ",
            b"GBA Savedata: Performing unknown sector-size erase at ",
        )
        if payload.startswith(routine_savedata_prefixes):
            return True

    if filter_known_warnings:
        # Emerald/mGBA can repeatedly emit invalid DMA-source diagnostics for
        # non-zero low addresses too (for example 0x000093AA). These are noisy
        # native emulator/game traces during ordinary runs. Hide this specific
        # diagnostic at SB3 verbosity 0/1, but retain it at verbosity 2 so it is
        # available when deliberately debugging native emulation. The zero-source
        # variant remains filtered by the existing known-warning rule at all levels.
        if payload.startswith(b"GBA DMA: Invalid DMA source address: "):
            if sb3_verbosity <= 1 or payload.startswith(
                b"GBA DMA: Invalid DMA source address: 0x00000000"
            ):
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
            "[mGBA Console] Filtering routine BIOS/DMA/RTC traces; known low-address BIOS/memory diagnostics "
            "and routine flash-save status lines are hidden at POKEMON_AI_VERBOSE=0/1 when "
            "POKEMON_AI_FILTER_KNOWN_MGBA_WARNINGS=1. Use POKEMON_AI_VERBOSE=2 for diagnostics, "
            "or POKEMON_AI_FILTER_MGBA_CHATTER=0 to show native stdout.",
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
try:
    # RecurrentPPO uses an LSTM policy so the agent can use recent observations and
    # actions to interpret multi-step dialogues, menus, and battles.
    from sb3_contrib import RecurrentPPO as PPO
except ImportError as exc:
    raise ImportError(
        "This project version uses RecurrentPPO. Install the matching SB3-Contrib package "
        "inside the project environment with: .venv/bin/pip install sb3-contrib"
    ) from exc
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor



# ==========================================
# 🛠️ CENTRAL CONFIGURATION & CONSTANTS
# ==========================================
ACTIVE_PROFILE = os.environ.get("POKEMON_AI_PROFILE", "fast_train").strip().lower()
MODE = os.environ.get("POKEMON_AI_MODE", "train").strip().lower()  # selected by profile
if MODE not in {"train", "watch", "setup", "evaluate"}:
    raise ValueError("POKEMON_AI_MODE must be 'train', 'watch', 'setup', or 'evaluate'")

# Paths are resolved relative to this script, so the project can be launched from any working directory.
# Hybrid observations and the 9-action space are intentionally stored under a separate checkpoint
# name so the older RGB-only policy and its learned weights are never loaded into this architecture.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Runtime-only artifacts stay out of the project root. Keep ROM, its cartridge .sav,
# and the established PPO model filename at the root for compatibility.
RUNTIME_DATA_DIR = os.path.join(BASE_DIR, "runtime_data")
STATE_DIR = os.path.join(RUNTIME_DATA_DIR, "states")
TEMP_DIR = os.path.join(RUNTIME_DATA_DIR, "tmp")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ROM_PATH = os.path.join(BASE_DIR, "PokemonEmerald.gba")
CARTRIDGE_SAVE_PATH = os.path.join(BASE_DIR, "PokemonEmerald.sav")
# This version uses a Dict observation (screen + structured game state + visited-tile memory)
# and an action space with WAIT. Keep it separate from the old RGB-only, 8-action checkpoint.
_MODEL_NAME = os.environ.get("POKEMON_AI_MODEL_NAME", "emerald_ppo_hybrid_v1").strip()
if not _MODEL_NAME or os.path.basename(_MODEL_NAME) != _MODEL_NAME:
    raise ValueError("POKEMON_AI_MODEL_NAME must be a simple filename without directory components")
MODEL_CHECKPOINT = os.path.join(BASE_DIR, _MODEL_NAME)
RESUME_TRAINING = os.environ.get("POKEMON_AI_RESUME", "1").strip().lower() not in {"0", "false", "no", "off"}
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
# The only supported learning objective is the original Route 101 task:
# rescue Professor Birch, choose a starter, and verify the starter is in the party.
ROUTE_101_MAP_ID = 16  # BPEE mapGroup 0, mapNum 16; verified against the project's RAM telemetry.
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
ACTION_RELEASE_FRAMES = int(os.environ.get("POKEMON_AI_ACTION_RELEASE_FRAMES", "1"))
if ACTION_REPEAT < 1:
    raise ValueError("POKEMON_AI_ACTION_REPEAT must be a positive integer")
if ACTION_RELEASE_FRAMES < 0:
    raise ValueError("POKEMON_AI_ACTION_RELEASE_FRAMES must be >= 0")
AUDIO_OUTPUT_ENABLED = True        # Audio plays in single-environment runs; parallel headless training stays silent.
AUDIO_RATE_MULTIPLIER = 2.0        # Empirical fix: this binding's captured stream sounded correct at 2x.
AUDIO_QUEUE_CHUNKS = 8             # Bounded queue prevents audio lag from growing without limit.
# Allow the GBA BIOS/game to produce a real frame on cold start before the first PPO observation.
INITIAL_BOOT_FRAMES = 600
RESET_WATCHDOG_SECONDS = 20
FRAME_DIAGNOSTICS = os.environ.get("POKEMON_AI_FRAME_DIAGNOSTICS", "0").strip().lower() not in {"0", "false", "no", "off"}
SETUP_DIAGNOSTICS_EVERY = int(os.environ.get("POKEMON_AI_SETUP_DIAGNOSTICS_EVERY", "300"))
if SETUP_DIAGNOSTICS_EVERY < 1:
    raise ValueError("POKEMON_AI_SETUP_DIAGNOSTICS_EVERY must be a positive integer")
MAX_EPISODE_STEPS = int(os.environ.get("POKEMON_AI_MAX_EPISODE_STEPS", "30000"))
if MAX_EPISODE_STEPS < 1:
    raise ValueError("POKEMON_AI_MAX_EPISODE_STEPS must be a positive integer")
EVAL_DIAGNOSTICS_EVERY = int(os.environ.get("POKEMON_AI_EVAL_DIAGNOSTICS_EVERY", "1000"))
if EVAL_DIAGNOSTICS_EVERY < 1:
    raise ValueError("POKEMON_AI_EVAL_DIAGNOSTICS_EVERY must be a positive integer")
# General, non-directional shaping. Existing milestone and progression rewards remain;
# novelty rewards decay across episodes so revisiting familiar places is less profitable.
STEP_PENALTY = float(os.environ.get("POKEMON_AI_STEP_PENALTY", "0.00005"))
LEVEL_REWARD_PER_LEVEL = float(os.environ.get("POKEMON_AI_LEVEL_REWARD", "1.0"))
BADGE_REWARD = float(os.environ.get("POKEMON_AI_BADGE_REWARD", "50.0"))
NEW_MAP_REWARD = float(os.environ.get("POKEMON_AI_NEW_MAP_REWARD", "0.75"))
NEW_TILE_REWARD = float(os.environ.get("POKEMON_AI_NEW_TILE_REWARD", "0.02"))
NEW_SCREEN_REWARD = float(os.environ.get("POKEMON_AI_NEW_SCREEN_REWARD", "0.002"))
MAX_SCREEN_NOVELTY_REWARD_PER_EPISODE = float(
    os.environ.get("POKEMON_AI_SCREEN_NOVELTY_CAP", "0.05")
)
PARTY_MEMBER_REWARD = float(os.environ.get("POKEMON_AI_PARTY_MEMBER_REWARD", "5.0"))
TRAINER_WIN_REWARD = float(os.environ.get("POKEMON_AI_TRAINER_WIN_REWARD", "0.75"))
LOOP_PENALTY = float(os.environ.get("POKEMON_AI_LOOP_PENALTY", "0.05"))
LOOP_STAGNATION_STEPS = int(os.environ.get("POKEMON_AI_LOOP_STAGNATION_STEPS", "512"))
LOOP_PENALTY_INTERVAL = int(os.environ.get("POKEMON_AI_LOOP_PENALTY_INTERVAL", "256"))
TASK_SUCCESS_REWARD = float(os.environ.get("POKEMON_AI_TASK_SUCCESS_REWARD", "30.0"))
for _reward_name, _reward_value in {
    "STEP_PENALTY": STEP_PENALTY,
    "LEVEL_REWARD_PER_LEVEL": LEVEL_REWARD_PER_LEVEL,
    "BADGE_REWARD": BADGE_REWARD,
    "NEW_MAP_REWARD": NEW_MAP_REWARD,
    "NEW_TILE_REWARD": NEW_TILE_REWARD,
    "NEW_SCREEN_REWARD": NEW_SCREEN_REWARD,
    "MAX_SCREEN_NOVELTY_REWARD_PER_EPISODE": MAX_SCREEN_NOVELTY_REWARD_PER_EPISODE,
    "PARTY_MEMBER_REWARD": PARTY_MEMBER_REWARD,
    "TRAINER_WIN_REWARD": TRAINER_WIN_REWARD,
    "LOOP_PENALTY": LOOP_PENALTY,
    "TASK_SUCCESS_REWARD": TASK_SUCCESS_REWARD,
}.items():
    if _reward_value < 0:
        raise ValueError(f"{_reward_name} must be >= 0")
if LOOP_STAGNATION_STEPS < 1 or LOOP_PENALTY_INTERVAL < 1:
    raise ValueError("Loop-detection step settings must be positive integers")
SAVE_BLOCK1_PTR_ADDRESS = 0x03005D8C
SAVE_BLOCK1_FLAGS_OFFSET = 0x1270
SAVE_BLOCK1_POS_X_OFFSET = 0x00
SAVE_BLOCK1_POS_Y_OFFSET = 0x02
SAVE_BLOCK1_LOCATION_OFFSET = 0x04  # struct WarpData: mapGroup, mapNum, warpId, x, y
TRAINER_FLAGS_START = 0x500
TRAINER_FLAGS_END = 0x85F
STARTER_OBTAINED_FLAG_ID = 0x860
POKEDEX_OBTAINED_FLAG_ID = 0x861
OBS_VISITED_GRID_SIZE = 64
# Story flags used for general, one-time task progress rewards and start-state validation.
# No route coordinates or route-script phase are fed to the policy as navigation hints.
CLOCK_SET_FLAG_ID = 0x51
BIRCH_RESCUED_FLAG_ID = 0x52

# Persistent story flags from the standard Pokémon Emerald (BPEE) decompilation.
# These are progression milestones, not arbitrary per-frame signals.
STORY_FLAG_REWARDS = {
    CLOCK_SET_FLAG_ID: 0.5,   # Set the bedroom wall clock
    BIRCH_RESCUED_FLAG_ID: 12.0,  # Rescued Professor Birch: central goal milestone
    0x860: 15.0, # FLAG_SYS_POKEMON_GET: received the first Pokémon
    0x861: 3.0,  # FLAG_SYS_POKEDEX_GET: Pokédex obtained
    0x74: 1.5,   # Adventure started / Pokédex obtained
    0x61: 1.0,   # Team Aqua Harbor event
    0x7D: 1.0,   # Rival event on Route 104
    0x82: 1.0,   # Rival event on Route 103
    0x8B: 1.5,   # Magma event on Mt. Chimney
    0x70: 2.0,   # Team Aqua submarine story event
    0x75: 2.0,   # Magma event at the Space Center
    0x6F: 2.0,   # Groudon awakened
}
# Emerald's persistent visited-town flags (FLAG_VISITED_LITTLEROOT_TOWN through
# FLAG_VISITED_EVER_GRANDE_CITY, 0x86F..0x87E). Small one-time breadcrumbs help
# the agent learn forward progression without replacing the larger story objectives.
for _town_flag in range(0x86F, 0x87F):
    STORY_FLAG_REWARDS.setdefault(_town_flag, 0.75)

OBS_STORY_FLAG_IDS = tuple(sorted(STORY_FLAG_REWARDS))
PREVIOUS_ACTION_FEATURE_NAMES = (
    "previous_action_wait", "previous_action_up", "previous_action_down",
    "previous_action_left", "previous_action_right", "previous_action_a",
    "previous_action_b", "previous_action_start", "previous_action_select",
)
STATE_FEATURE_NAMES = (
    "map_group", "map_number", "player_x", "player_y", "party_count",
    "party_total_levels", "badge_count", "visited_tile_count", "visited_map_count",
    "trainer_flag_count", "episode_progress", "steps_since_progress", "current_tile_seen",
    "clock_set", "starter_obtained", "pokedex_obtained",
) + tuple(f"badge_{i + 1}" for i in range(8)) + tuple(
    f"story_flag_{flag_id:03X}" for flag_id in OBS_STORY_FLAG_IDS
) + PREVIOUS_ACTION_FEATURE_NAMES

REWARD_COMPONENT_NAMES = (
    "step", "tile", "map", "screen", "story", "party", "level", "badge",
    "trainer", "loop", "task_success",
)


def _episode_time_limit_reached(mode, episode_steps):
    """Training episodes time out; watch/setup are continuous until user stop."""
    return str(mode).strip().lower() in {"train", "evaluate"} and int(episode_steps) >= MAX_EPISODE_STEPS


def _get_compatible_model_env(model, context):
    """Return the model's attached (possibly image-transposed) VecEnv after validating it.

    Stable-Baselines3 may wrap an environment in VecTransposeImage when the policy's
    image observations are channels-last. The wrapped environment, not the original
    VecEnv passed to PPO.load(), is the one whose observation space matches the policy
    and whose observations should be passed to model.predict().
    """
    model_env = model.get_env()
    if model_env is None:
        raise RuntimeError(f"{context}: loaded policy has no attached environment.")
    if (
        model.observation_space != model_env.observation_space
        or model.action_space != model_env.action_space
    ):
        raise RuntimeError(
            f"{context}: policy spaces do not match its attached wrapped environment. "
            "The checkpoint may have been created with a different observation/action schema."
        )
    return model_env


def _training_start_validation_error(state):
    """Validate the task's starting conditions without prescribing any route or action."""
    if state.get("player_x") is None or state.get("player_y") is None:
        return "player coordinates are invalid; save while the character is in normal gameplay"
    if int(state.get("map_id", -1)) != ROUTE_101_MAP_ID:
        return (
            f"the training state must be on Route 101 (map_id={ROUTE_101_MAP_ID}); "
            f"the current map_id is {state.get('map_id', -1)}"
        )
    if CLOCK_SET_FLAG_ID not in state.get("story_flags", set()):
        return "the bedroom clock-set flag is absent; save after setting the clock"
    if BIRCH_RESCUED_FLAG_ID in state.get("story_flags", set()):
        return "the training start must be before rescuing Birch and before receiving a starter"
    if int(state.get("party_count", 0)) != 0:
        return "the Route 101 task must start before receiving a starter Pokémon"
    if int(state.get("party_levels", 0)) != 0:
        return "the Route 101 task must start with an empty party and no party levels"
    return None


def _diminishing_novelty_reward(base_reward, prior_visits):
    """Decay novelty payoff as the same tile/map/screen is encountered across episodes."""
    base = max(0.0, float(base_reward))
    visits = max(0, int(prior_visits))
    return base / math.sqrt(1.0 + visits)


def _screen_signature(screen):
    """Return a coarse edge-signature of the image for novelty/loop diagnostics.

    It is not a classifier and does not identify menus or locations. Adjacent-pixel
    contrast detects layout changes in either brightness direction, unlike a one-sided
    difference hash that can miss a uniformly bright panel on a dark background.
    """
    if screen is None:
        return None
    frame = np.asarray(screen)
    if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
        return None
    rgb = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    contrast = np.abs(small[:, 1:] - small[:, :-1])
    bits = (contrast >= 16).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), byteorder="big", signed=False)


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
VIDEO_FPS = 60.0 / max(1, ACTION_REPEAT + ACTION_RELEASE_FRAMES)  # One saved image per policy decision
VIDEO_WIDTH = 240 * 4
VIDEO_HEIGHT = 160 * 4

# Desktop Notifications Configuration (Watch Mode Only)
ENABLE_NOTIFICATIONS = True if MODE == "watch" else False

# Environment & Performance Settings
# Keep these dimensions compatible with an existing checkpoint. The original game image is letterboxed
# inside this shape so the GBA's 3:2 aspect ratio is not distorted.
OBS_WIDTH = 120
OBS_HEIGHT = 80
TOTAL_TIMESTEPS = int(os.environ.get("POKEMON_AI_TOTAL_TIMESTEPS", "2000000"))
if TOTAL_TIMESTEPS < 1:
    raise ValueError("POKEMON_AI_TOTAL_TIMESTEPS must be a positive integer")
EVAL_EPISODES = int(os.environ.get("POKEMON_AI_EVAL_EPISODES", "5"))
if EVAL_EPISODES < 1:
    raise ValueError("POKEMON_AI_EVAL_EPISODES must be a positive integer")
VERBOSE_LEVEL = int(os.environ.get("POKEMON_AI_VERBOSE", "1"))
if VERBOSE_LEVEL not in (0, 1, 2):
    raise ValueError("POKEMON_AI_VERBOSE must be 0, 1, or 2")
STOCHASTIC_WATCH = _env_flag("POKEMON_AI_STOCHASTIC_WATCH", False)
EVAL_DETERMINISTIC = _env_flag("POKEMON_AI_EVAL_DETERMINISTIC", True)
PPO_N_STEPS = int(os.environ.get("POKEMON_AI_PPO_N_STEPS", "2048"))
PPO_BATCH_SIZE = int(os.environ.get("POKEMON_AI_PPO_BATCH_SIZE", "256"))
PPO_N_EPOCHS = int(os.environ.get("POKEMON_AI_PPO_N_EPOCHS", "10"))
PPO_GAMMA = float(os.environ.get("POKEMON_AI_PPO_GAMMA", "0.999"))
PPO_GAE_LAMBDA = float(os.environ.get("POKEMON_AI_PPO_GAE_LAMBDA", "0.98"))
PPO_LEARNING_RATE = float(os.environ.get("POKEMON_AI_PPO_LEARNING_RATE", "0.0003"))
ENTROPY_COEF_START = float(os.environ.get("POKEMON_AI_ENTROPY_COEF_START", "0.01"))
ENTROPY_COEF_END = float(os.environ.get("POKEMON_AI_ENTROPY_COEF_END", "0.001"))
if min(PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS) < 1:
    raise ValueError("PPO n_steps, batch_size, and n_epochs must be positive integers")
if not 0.0 < PPO_GAMMA <= 1.0 or not 0.0 < PPO_GAE_LAMBDA <= 1.0:
    raise ValueError("PPO gamma and GAE lambda must be in (0, 1]")
if min(PPO_LEARNING_RATE, ENTROPY_COEF_START, ENTROPY_COEF_END) < 0:
    raise ValueError("PPO learning rate and entropy coefficients must be >= 0")

# Environment count for headless training. Override with POKEMON_AI_NUM_ENVS.
# More workers can increase CPU use, but also require more RAM because every worker owns
# an emulator and its Python/PyTorch process. GUI and watch modes intentionally stay single-env.
HEADLESS_NUM_ENVS = int(os.environ.get("POKEMON_AI_NUM_ENVS", "4"))
if HEADLESS_NUM_ENVS < 1:
    raise ValueError("POKEMON_AI_NUM_ENVS must be a positive integer")
NUM_CPU = 1 if GUI_MODE or MODE in {"watch", "setup", "evaluate"} else HEADLESS_NUM_ENVS
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
    0: "WAIT",  # Explicit no-input action for dialogue animations and menu transitions.
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "A",
    6: "B",
    7: "START",
    8: "SELECT",
}
ACTION_LABELS = dict(ACTION_TO_KEY)
ACTION_KEY_MASK = {action: (0 if name == "WAIT" else 1 << GBA_KEY_INDEX[name]) for action, name in ACTION_TO_KEY.items()}


def _format_action_distribution(action_counts):
    """Format a deterministic, most-common-first action histogram for diagnostics."""
    total = sum(int(count) for count in action_counts.values())
    if total <= 0:
        return "none"
    ordered = sorted(action_counts.items(), key=lambda item: (-int(item[1]), int(item[0])))
    return ", ".join(
        f"{ACTION_LABELS.get(int(action), f'action_{action}') }={int(count)} ({int(count) / total:.1%})"
        for action, count in ordered
        if int(count) > 0
    )

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

        self.action_space = spaces.Discrete(9)  # 0:WAIT, 1..4:directions, 5:A, 6:B, 7:Start, 8:Select
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8),
            # Float mask is flattened by SB3's CombinedExtractor instead of treated as a second image CNN.
            "visited": spaces.Box(low=0.0, high=1.0, shape=(OBS_VISITED_GRID_SIZE, OBS_VISITED_GRID_SIZE), dtype=np.float32),
            "state": spaces.Box(low=0.0, high=1.0, shape=(len(STATE_FEATURE_NAMES),), dtype=np.float32),
        })

        self.rom_path = rom_path
        self.checkpoint_file = checkpoint_file
        self.env_id = env_id
        self.step_counter = 0
        self.episode_step_counter = 0
        self.last_action_id = 0  # Previous discrete action is part of the next observation.
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
        self.seen_screen_signatures = set()
        self.seen_story_flags = set()
        self.seen_trainer_flags = set()
        self.position_history = deque(maxlen=32)
        self.screen_signature_history = deque(maxlen=32)
        # Lifetime counts decay novelty across episode resets without rewarding repeats
        # every frame. They persist per emulator worker for the duration of a run.
        self.lifetime_tile_visits = Counter()
        self.lifetime_map_visits = Counter()
        self.lifetime_screen_visits = Counter()
        self.episode_screen_novelty_reward = 0.0
        self.last_novelty_step = 0
        self.episode_reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        self._last_game_state = None

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
                    temporary_save = MODE in {"train", "evaluate"} or NUM_CPU > 1
                    try:
                        load_result = load_save(CARTRIDGE_SAVE_PATH, temporary_save)
                        if load_result is False:
                            print(f"[Save Warning] mGBA reported it could not load {CARTRIDGE_SAVE_PATH}.")
                        else:
                            save_mode = "isolated temporary data during training/evaluation" if temporary_save else "persistent"
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
                    if autoload_save is not None and MODE not in {"train", "evaluate"}:
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
                        if MODE in {"train", "evaluate"}:
                            save_note = "training/evaluation uses isolated emulator save data and does not write the shared cartridge .sav file"
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
            print("[GUI] Window created. Press q/Esc to quit, P to pause/resume, M to mute/unmute audio.", flush=True)
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

    def _get_image_obs(self):
        """Return the resized RGB screen while retaining the native-sized frame for GUI/video."""
        raw_frame_rgb = self._read_framebuffer()
        self._last_raw_frame_rgb = raw_frame_rgb

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

        raw_height, raw_width = raw_frame_rgb.shape[:2]
        if raw_width <= 0 or raw_height <= 0:
            raise RuntimeError(f"mGBA returned an invalid frame shape: {raw_frame_rgb.shape}")
        scale = min(OBS_WIDTH / raw_width, OBS_HEIGHT / raw_height)
        scaled_width = max(1, int(round(raw_width * scale)))
        scaled_height = max(1, int(round(raw_height * scale)))
        resized = cv2.resize(
            raw_frame_rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST
        )
        screen = np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)
        x_offset = (OBS_WIDTH - scaled_width) // 2
        y_offset = (OBS_HEIGHT - scaled_height) // 2
        screen[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized
        return screen

    def _encode_state_features(self, state):
        """Encode validated game state as bounded float features for the policy."""
        state = state or {}
        story_flags = state.get("story_flags", set())
        badge_flags = int(state.get("badge_flags", 0) or 0)
        x = state.get("player_x")
        y = state.get("player_y")
        map_id = int(state.get("map_id", -1) if state.get("map_id") is not None else -1)
        current_tile_seen = 0.0
        if x is not None and y is not None and map_id >= 0:
            current_tile_seen = float((map_id, int(x), int(y)) in self.visited_tiles)

        values = [
            max(0.0, min(1.0, float(state.get("map_group", 0)) / 31.0)),
            max(0.0, min(1.0, float(state.get("map_number", 0)) / 100.0)),
            max(0.0, min(1.0, float(x or 0) / 512.0)),
            max(0.0, min(1.0, float(y or 0) / 512.0)),
            max(0.0, min(1.0, float(state.get("party_count", 0)) / 6.0)),
            max(0.0, min(1.0, float(state.get("party_levels", 0)) / 600.0)),
            max(0.0, min(1.0, float(badge_flags.bit_count()) / 8.0)),
            max(0.0, min(1.0, len(self.visited_tiles) / 1024.0)),
            max(0.0, min(1.0, len(self.visited_maps) / 64.0)),
            max(0.0, min(1.0, len(state.get("trainer_flags", set())) / 864.0)),
            max(0.0, min(1.0, self.episode_step_counter / float(max(1, MAX_EPISODE_STEPS)))),
            max(0.0, min(1.0, (self.episode_step_counter - getattr(self, "last_novelty_step", 0)) / float(max(1, LOOP_STAGNATION_STEPS)))),
            current_tile_seen,
            float(CLOCK_SET_FLAG_ID in story_flags),
            float(STARTER_OBTAINED_FLAG_ID in story_flags or state.get("party_count", 0) > 0),
            float(POKEDEX_OBTAINED_FLAG_ID in story_flags),
        ]
        values.extend(float(bool(badge_flags & (1 << i))) for i in range(8))
        values.extend(float(flag_id in story_flags) for flag_id in OBS_STORY_FLAG_IDS)
        previous_action = int(getattr(self, "last_action_id", 0))
        values.extend(float(previous_action == action_id) for action_id in range(9))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (len(STATE_FEATURE_NAMES),):
            raise RuntimeError(
                f"State feature vector has shape {result.shape}, expected {(len(STATE_FEATURE_NAMES),)}"
            )
        return np.clip(result, 0.0, 1.0)

    def _visited_mask(self, map_id):
        """Local map-grid memory; only positions from the current map are included."""
        mask = np.zeros((OBS_VISITED_GRID_SIZE, OBS_VISITED_GRID_SIZE), dtype=np.float32)
        if map_id is None or int(map_id) < 0:
            return mask
        # The observation memory is intentionally coarse. Emerald's map coordinates can
        # exceed 63; compress them into the fixed 64x64 mask instead of silently dropping
        # every visited tile outside the upper-left 64x64 region. Exact tile novelty still
        # uses the original (map_id, x, y) tuple, so this affects observation only.
        for visited_map, x, y in self.visited_tiles:
            if visited_map != int(map_id) or int(x) < 0 or int(y) < 0:
                continue
            gx = min(OBS_VISITED_GRID_SIZE - 1, int(x) * OBS_VISITED_GRID_SIZE // 128)
            gy = min(OBS_VISITED_GRID_SIZE - 1, int(y) * OBS_VISITED_GRID_SIZE // 128)
            mask[gy, gx] = 1.0
        return mask

    def _get_obs(self, game_state=None, read_state=True, screen=None):
        """Hybrid observation. Screens are captured in headless and GUI modes alike."""
        if screen is None:
            screen = self._get_image_obs()
        state = game_state
        if state is None and read_state:
            try:
                state = self._read_game_state()
            except Exception:
                state = {}
        state = state or {}
        map_id = state.get("map_id", -1)
        return {
            "screen": screen,
            "visited": self._visited_mask(map_id),
            "state": self._encode_state_features(state),
        }

    def _set_action_keys(self, action):
        """Set GBA input using the native bitmask API or mGBA's Python raw= keyword."""
        if action is None:
            mask = 0
            key_name = None
        else:
            action = int(action)
            if action not in ACTION_TO_KEY:
                raise ValueError(f"Invalid action {action}; expected an integer from 0 to 8.")
            key_name = ACTION_TO_KEY[action]
            mask = ACTION_KEY_MASK[action]
            if key_name == "WAIT":
                key_name = None

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

    def _read_trainer_flags(self, saveblock1_ptr):
        """Read the set trainer-defeated flags (0x500..0x85F) efficiently by byte."""
        flags_base = saveblock1_ptr + SAVE_BLOCK1_FLAGS_OFFSET
        start_byte = TRAINER_FLAGS_START >> 3
        end_byte = TRAINER_FLAGS_END >> 3
        found = set()
        for byte_index in range(start_byte, end_byte + 1):
            value = self._read8(flags_base + byte_index)
            if not value:
                continue
            first_flag = byte_index * 8
            for bit in range(8):
                flag_id = first_flag + bit
                if TRAINER_FLAGS_START <= flag_id <= TRAINER_FLAGS_END and value & (1 << bit):
                    found.add(flag_id)
        return found

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
        # SaveBlock1.location is a WarpData at offset 0x04; mapGroup/mapNum are signed bytes.
        map_group = self._read8(saveblock1_ptr + SAVE_BLOCK1_LOCATION_OFFSET)
        map_number = self._read8(saveblock1_ptr + SAVE_BLOCK1_LOCATION_OFFSET + 1)
        if map_group >= 0x80:
            map_group -= 0x100
        if map_number >= 0x80:
            map_number -= 0x100
        # Map group/number are signed bytes in WarpData. Keep the full nonnegative
        # signed-byte range rather than hardcoding an assumed number of map groups.
        if 0 <= map_group <= 127 and 0 <= map_number <= 127:
            map_id = (map_group << 8) | map_number
        else:
            map_id = -1
        player_x, player_y = self._read_player_position(saveblock1_ptr)
        story_flags = self._read_story_flags(saveblock1_ptr)
        trainer_flags = self._read_trainer_flags(saveblock1_ptr)
        return {
            "party_count": party_count,
            "party_levels": total_party_levels,
            "badge_flags": badge_flags,
            "badge_count": badge_flags.bit_count(),
            "map_id": map_id,
            "map_group": map_group,
            "map_number": map_number,
            "player_x": player_x,
            "player_y": player_y,
            "story_flags": story_flags,
            "trainer_flags": trainer_flags,
        }

    def _refresh_reward_baseline(self):
        try:
            state = self._read_game_state()
            self.prev_party_levels = state["party_levels"]
            self.prev_badges = state["badge_flags"]
            self.prev_caught_count = state["party_count"]
            self.prev_map_id = state["map_id"]
            if state["map_id"] >= 0:
                map_id = int(state["map_id"])
                self.visited_maps.add(map_id)
                self.lifetime_map_visits[map_id] += 1
            self.seen_story_flags.update(state.get("story_flags", set()))
            self.seen_trainer_flags.update(state.get("trainer_flags", set()))
            if state["map_id"] >= 0 and state.get("player_x") is not None and state.get("player_y") is not None:
                position = (int(state["map_id"]), int(state["player_x"]), int(state["player_y"]))
                self.visited_tiles.add(position)
                self.lifetime_tile_visits[position] += 1
                self.position_history.append(position)
            screen_signature = _screen_signature(self._get_image_obs())
            if screen_signature is not None:
                self.seen_screen_signatures.add(screen_signature)
                self.screen_signature_history.append(screen_signature)
                self.lifetime_screen_visits[screen_signature] += 1
            self.last_novelty_step = self.episode_step_counter
            self._last_game_state = state
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
            self.visited_maps.clear()
            self.visited_tiles.clear()
            self.seen_screen_signatures.clear()
            self.seen_story_flags.clear()
            self.seen_trainer_flags.clear()
            self.position_history.clear()
            self.screen_signature_history.clear()
            self.episode_screen_novelty_reward = 0.0
            self.last_novelty_step = 0
            self.episode_reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            self._last_game_state = None

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
                            "Recreate it with ./run.sh --profile setup and press K on Route 101 before obtaining a starter."
                        )
                elif MODE in {"train", "evaluate", "watch"} and REQUIRE_TRAINING_START_STATE:
                    raise FileNotFoundError(
                        f"Required training start savestate not found: {TRAINING_START_STATE}\n"
                        f"Run ./run.sh --profile setup and follow the instructions for Route 101 task, "
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
                    if MODE in {"train", "watch", "evaluate"} and os.path.isfile(TRAINING_START_STATE):
                        # Validate the canonical Route 101 pre-starter state.
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
                                f"Training start state is invalid: {validation_error}. "
                                "Recreate it with ./run.sh --profile setup and press K."
                            )
                        print(
                            f"[Start State] Validated Route 101 task | "
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
                    f"[Reset {self.env_id}] observation ready: screen={obs['screen'].shape}, "
                    f"state={obs['state'].shape}, visited={obs['visited'].shape}",
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

            # The canonical start is restored on train episode resets. Novelty and reward
            # component trackers are episode-local; seed them with the current start tile/flags.
            print(f"[Reset {self.env_id}] refreshing reward baseline", flush=True)
            self._refresh_reward_baseline()
            obs = self._get_obs(game_state=self._last_game_state, read_state=False)
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
            return self._get_obs(), 0.0, False, True, {
                "stopped": True,
                "episode_outcome": "STOPPED",
                "TimeLimit.truncated": True,
            }

        self.step_counter += 1
        self.episode_step_counter += 1
        # Setup mode is manual-only. Watch mode optionally permits one-step host overrides.
        if HOST_CONTROL_ENABLED and MODE in {"watch", "setup"}:
            if self.manual_action_next is not None:
                action = self.manual_action_next
                self.manual_action_next = None
            elif MODE == "setup":
                action = None  # neutral input until a human presses a control key

        try:
            self.last_action_id = int(np.asarray(action).reshape(-1)[0]) if action is not None else 0
            if not 0 <= self.last_action_id <= 8:
                self.last_action_id = 0
        except (TypeError, ValueError, IndexError, OverflowError):
            self.last_action_id = 0

        if self.core is not None:
            run_frame = _method(self.core, "run_frame", "runFrame")
            if run_frame is None:
                raise RuntimeError("The installed mGBA core does not expose runFrame() or run_frame().")

            # Insert at least one neutral emulated frame between button presses. Merely
            # calling setKeys(0) without advancing a frame doesn't reliably create a release edge.
            for frame_action, frame_count in ((action, max(1, ACTION_REPEAT)), (None, ACTION_RELEASE_FRAMES)):
                for _ in range(frame_count):
                    self._set_action_keys(frame_action)
                    frame_started = time.perf_counter()
                    run_frame()
                    self._pump_audio_buffer(
                        accept=self._audio_accepting,
                        record=self._record_audio_active,
                    )
                    if GUI_MODE:
                        self._last_raw_frame_rgb = self._read_framebuffer()
                        self._render_gui(frame_started)
                        if self.stop_requested or STOP_REQUESTED:
                            break
                    elif MODE == "watch" and WATCH_FPS_CAP > 0:
                        delay = (1.0 / WATCH_FPS_CAP) - (time.perf_counter() - frame_started)
                        if delay > 0:
                            time.sleep(delay)
                if self.stop_requested or STOP_REQUESTED:
                    break
            self._set_action_keys(None)

        # Save after releasing the previous action's key, avoiding a snapshot with a held key.
        if MODE == "setup" and self._save_training_start_requested:
            self._save_training_start_requested = False
            try:
                setup_state = self._read_game_state()
                validation_error = _training_start_validation_error(setup_state)
                if validation_error:
                    print(
                        f"[Setup] Not saved: {validation_error}.",
                        flush=True,
                    )
                elif self._save_state_file(TRAINING_START_STATE):
                    print(
                        f"[Setup] Route 101 training start savestate saved: {TRAINING_START_STATE}",
                        flush=True,
                    )
                    print(
                        "[Setup] This task starts with the clock set and an empty party, before obtaining the first Pokémon.",
                        flush=True,
                    )
                    print(
                        f"[Setup] Active PPO checkpoint path is {MODEL_CHECKPOINT}.zip. "
                        "Preserve any existing checkpoint before starting a new incompatible experiment; "
                        "then run ./run.sh --profile fast_train.", flush=True
                    )
                else:
                    print(f"[Setup Error] Could not save training start state to {TRAINING_START_STATE}.", flush=True)
            except Exception as exc:
                print(f"[Setup] Not saved because the game state could not be validated: {exc}", flush=True)

        # Read one screenshot for both generic visual novelty and the policy observation.
        # This path runs even in headless mode; GUI_MODE only controls displaying the frame.
        screen_frame = self._get_image_obs() if self.core is not None else None
        current_screen_signature = _screen_signature(screen_frame)
        state = None
        component = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        component["step"] = -STEP_PENALTY
        info = {"episode_step": self.episode_step_counter}
        terminated = False
        if current_screen_signature is not None:
            self.screen_signature_history.append(current_screen_signature)
        if self.core is not None:
            try:
                state = self._read_game_state()
                self._last_game_state = state

                # Level/party progress is delta-based and cannot be farmed by standing still.
                previous_party_levels = self.prev_party_levels
                level_delta = state["party_levels"] - previous_party_levels
                component["level"] = max(0, level_delta) * LEVEL_REWARD_PER_LEVEL
                if MODE == "watch" and state["party_levels"] // 10 > previous_party_levels // 10:
                    send_notification(
                        "Pokémon Training Milestone",
                        f"Combined party level reached {state['party_levels']}."
                    )
                self.prev_party_levels = state["party_levels"]

                new_badge_bits = state["badge_flags"] & (~self.prev_badges & 0xFF)
                new_badges = int(new_badge_bits).bit_count()
                component["badge"] = BADGE_REWARD * new_badges
                self.prev_badges = state["badge_flags"]

                map_id = int(state["map_id"])
                new_area_discovered = False
                if map_id >= 0:
                    new_area_discovered = map_id not in self.visited_maps
                    if new_area_discovered:
                        self.visited_maps.add(map_id)
                        prior_visits = self.lifetime_map_visits[map_id]
                        component["map"] = _diminishing_novelty_reward(NEW_MAP_REWARD, prior_visits)
                        self.lifetime_map_visits[map_id] += 1
                        if MODE == "watch":
                            send_notification("New Pokémon Emerald Area", f"The AI discovered map ID {map_id}.")

                player_x = state.get("player_x")
                player_y = state.get("player_y")
                new_tile_discovered = False
                if map_id >= 0 and player_x is not None and player_y is not None:
                    tile_key = (map_id, int(player_x), int(player_y))
                    if tile_key not in self.visited_tiles:
                        self.visited_tiles.add(tile_key)
                        prior_visits = self.lifetime_tile_visits[tile_key]
                        component["tile"] = _diminishing_novelty_reward(NEW_TILE_REWARD, prior_visits)
                        self.lifetime_tile_visits[tile_key] += 1
                        new_tile_discovered = True
                    self.position_history.append(tile_key)

                # Reward broad visual-state novelty at a low rate. It is capped per episode
                # and decays across resets so animated pixels/menus cannot dominate progress.
                new_screen_discovered = False
                if current_screen_signature is not None and current_screen_signature not in self.seen_screen_signatures:
                    self.seen_screen_signatures.add(current_screen_signature)
                    prior_screen_visits = self.lifetime_screen_visits[current_screen_signature]
                    potential_screen_reward = _diminishing_novelty_reward(NEW_SCREEN_REWARD, prior_screen_visits)
                    remaining_screen_budget = max(
                        0.0,
                        MAX_SCREEN_NOVELTY_REWARD_PER_EPISODE - self.episode_screen_novelty_reward,
                    )
                    component["screen"] = min(potential_screen_reward, remaining_screen_budget)
                    self.episode_screen_novelty_reward += component["screen"]
                    self.lifetime_screen_visits[current_screen_signature] += 1
                    new_screen_discovered = component["screen"] > 0.0

                # Story flag bonuses are unique within an episode; reward transitions, not
                # the absolute value of a cumulative story score.
                current_story_flags = state.get("story_flags", set())
                new_story_flags = current_story_flags - self.seen_story_flags
                component["story"] = sum(STORY_FLAG_REWARDS.get(flag, 0.0) for flag in new_story_flags)
                self.seen_story_flags.update(current_story_flags)

                party_members_added = max(0, state["party_count"] - self.prev_caught_count)
                component["party"] = PARTY_MEMBER_REWARD * party_members_added
                if party_members_added and MODE == "watch":
                    send_notification("Pokémon Party Expanded", f"Party size increased by {party_members_added}.")
                self.prev_caught_count = state["party_count"]

                # Trainer flags are a reliable, one-time proxy for trainer victories in Emerald.
                current_trainer_flags = state.get("trainer_flags", set())
                new_trainer_flags = current_trainer_flags - self.seen_trainer_flags
                component["trainer"] = min(4, len(new_trainer_flags)) * TRAINER_WIN_REWARD
                self.seen_trainer_flags.update(current_trainer_flags)

                if new_badges:
                    self.last_notified_event = "gym_badge"
                    if MODE == "watch":
                        send_notification("Gym Victory Milestone", f"The AI appears to have acquired {new_badges} new badge(s).")

                # Small periodic anti-stagnation penalty; don't punish ordinary backtracking
                # or every step. Novelty/progression resets the stagnation timer.
                novelty_event = bool(
                    new_tile_discovered or new_area_discovered or new_screen_discovered or new_story_flags or
                    party_members_added or level_delta > 0 or new_badges or new_trainer_flags
                )
                if novelty_event:
                    self.last_novelty_step = self.episode_step_counter
                elif (
                    map_id >= 0
                    and self.episode_step_counter - self.last_novelty_step >= LOOP_STAGNATION_STEPS
                    and self.episode_step_counter % LOOP_PENALTY_INTERVAL == 0
                ):
                    recent_unique_positions = len(set(self.position_history))
                    recent_unique_screens = len(set(self.screen_signature_history))
                    # A repeated position alone is insufficient: dialogue/battle/menu screens
                    # may advance while coordinates remain fixed. Penalize only a spatially
                    # and visually static loop after the grace period.
                    if (
                        len(self.position_history) >= 16
                        and recent_unique_positions <= 4
                        and len(self.screen_signature_history) >= 16
                        and recent_unique_screens <= 2
                    ):
                        component["loop"] = -LOOP_PENALTY

                # The only goal is to obtain the first Pokémon from the Route 101 start.
                task_success = state["party_count"] >= 1 and state["party_levels"] >= 1
                if task_success:
                    component["task_success"] = TASK_SUCCESS_REWARD
                    info["task_success"] = True
                    # Training/evaluation need terminal episodes for PPO success signals.
                    # Watch mode stays in the success scene until the watcher stops, so
                    # the vector wrapper cannot immediately auto-reset the screen.
                    if MODE in {"train", "evaluate"}:
                        terminated = True
                else:
                    info["task_success"] = False

                info.update({
                    "party_count": state["party_count"],
                    "party_levels": state["party_levels"],
                    "new_badges": new_badges,
                    "badge_count": state.get("badge_count", int(state["badge_flags"]).bit_count()),
                    "map_id": map_id,
                    "map_group": state.get("map_group", -1),
                    "map_number": state.get("map_number", -1),
                    "player_x": player_x if player_x is not None else -1,
                    "player_y": player_y if player_y is not None else -1,
                    "new_area_discovered": new_area_discovered,
                    "new_tile_discovered": new_tile_discovered,
                    "new_story_flags": len(new_story_flags),
                    "new_trainer_flags": len(new_trainer_flags),
                    "new_badges_this_step": new_badges,
                    "unique_tiles_episode": len(self.visited_tiles),
                    "unique_maps_episode": len(self.visited_maps),
                    "story_flags_episode": len(self.seen_story_flags),
                    "trainer_flags_episode": len(self.seen_trainer_flags),
                    # Public metric aliases: keep each component separately inspectable.
                    "step_reward": component["step"],
                    "tile_reward": component["tile"],
                    "map_reward": component["map"],
                    "exploration_reward": component["tile"] + component["map"] + component["screen"],
                    "screen_reward": component["screen"],
                    "new_screen_discovered": new_screen_discovered,
                    "unique_screens_episode": len(self.seen_screen_signatures),
                    "level_reward": component["level"],
                    "badge_reward": component["badge"],
                    "story_reward": component["story"],
                    "party_reward": component["party"],
                    "trainer_reward": component["trainer"],
                    "loop_penalty": component["loop"],
                    "task_success_reward": component["task_success"],
                })
                self._memory_read_warning_printed = False
            except Exception as e:
                if not self._memory_read_warning_printed:
                    print(f"[RAM Warning] Reward memory read failed; only step penalty will be used: {e}", flush=True)
                    self._memory_read_warning_printed = True
                state = None
                component = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
                component["step"] = -STEP_PENALTY
                self._last_game_state = None

        reward = float(sum(component.values()))
        for component_name, component_value in component.items():
            self.episode_reward_components[component_name] += float(component_value)
            info[f"reward_{component_name}"] = float(component_value)
            info[f"episode_{component_name}_sum"] = float(self.episode_reward_components[component_name])
        info["episode_reward_total_so_far"] = float(sum(self.episode_reward_components.values()))

        # The next observation sees the updated visited-tile map and decoded state.
        obs = self._get_obs(game_state=state, read_state=False, screen=screen_frame)

        # Record frame if enabled
        if self.video_writer is not None and self._last_raw_frame_rgb is not None:
            frame_rgb = cv2.resize(
                self._last_raw_frame_rgb, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

        # The GUI was rendered immediately after frame acquisition above.
        # Training is episodic. Watch mode deliberately has NO automatic time-limit reset:
        # the policy keeps playing the same game indefinitely until q/Esc/Ctrl+C.
        truncated = _episode_time_limit_reached(MODE, self.episode_step_counter)
        if self.stop_requested or STOP_REQUESTED:
            truncated = True
            info["stopped"] = True
        if truncated:
            info["TimeLimit.truncated"] = True
        if terminated or truncated:
            outcome = "SUCCESS" if terminated and info.get("task_success") else ("TIMEOUT" if truncated and not info.get("stopped") else "STOPPED")
            info["episode_outcome"] = outcome
            reward_breakdown = ", ".join(
                f"{name}={value:+.2f}"
                for name, value in self.episode_reward_components.items()
                if abs(value) >= 0.005
            ) or "no nontrivial reward components"
            print(
                f"[Episode {self.env_id}] {outcome} | steps={self.episode_step_counter:,} | "
                f"return={sum(self.episode_reward_components.values()):.3f} | "
                f"tiles={len(self.visited_tiles)} maps={len(self.visited_maps)} "
                f"story_flags={len(self.seen_story_flags)} trainers={len(self.seen_trainer_flags)} | "
                f"map={info.get('map_id', -1)} pos=({info.get('player_x', -1)},{info.get('player_y', -1)}) | "
                f"rewards: {reward_breakdown}",
                flush=True,
            )
        return obs, float(reward), terminated, truncated, info


    def _decode_gui_key(self, key):
        """Decode keyboard input without confusing OpenCV's left-arrow code with Q."""
        if key < 0:
            return None

        # On X11/OpenCV waitKeyEx, left arrow is commonly 65361. Some fallbacks expose
        # only the low byte (81), which equals ASCII 'Q'. Handle navigation codes first.
        arrow_to_action = {
            65362: 1, 82: 1,   # Up
            65364: 2, 84: 2,   # Down
            65361: 3, 81: 3,   # Left (must never be treated as quit)
            65363: 4, 83: 4,   # Right
        }
        if key in arrow_to_action:
            return arrow_to_action[key] if HOST_CONTROL_ENABLED and MODE in {"watch", "setup"} else None

        normalized = key & 0xFF
        # Lowercase q and Escape are unambiguous across OpenCV backends. Some waitKey
        # backends collapse Left to 81, which is also uppercase ASCII Q; the arrow mapping
        # above wins in that case to prevent the left-arrow key from accidentally closing the GUI.
        if key == 27 or key == ord("q"):
            self.stop_requested = True
            print("[GUI] Quit requested; saving checkpoints and shutting down cleanly...", flush=True)
            return None

        if normalized in (ord("p"), ord("P")):
            self.paused = not self.paused
            print("[GUI] Paused." if self.paused else "[GUI] Resumed.", flush=True)
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

        key_to_action = {
            ord("w"): 1, ord("W"): 1,
            ord("s"): 2, ord("S"): 2,
            ord("a"): 3, ord("A"): 3,
            ord("d"): 4, ord("D"): 4,
            ord("z"): 5, ord("Z"): 5, 13: 5, 10: 5,
            ord("x"): 6, ord("X"): 6, 8: 6, 127: 6,
            ord("1"): 7, ord("2"): 8,
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
                footer = f"Arrows/WASD move | Z/Enter=A | X/Backspace=B | 1=Start | 2=Select | K=save start | q/Esc=quit | P=pause | {mute_hint}"
            else:
                footer = f"q/Esc: quit | P: pause | {mute_hint}"
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
        self._recent_episode_success = deque(maxlen=100)
        self._task_successes = 0
        self._timeouts = 0

    def _on_training_start(self):
        # On resumed training, num_timesteps can already be very large. Anchor the
        # interval to this run's starting value so we do not save again on its first step.
        self._run_start_timestep = int(self.num_timesteps)
        self._last_save_timestep = int(self.num_timesteps)
        self._last_progress_timestep = int(self.num_timesteps)
        self._run_started_at = time.perf_counter()
        self._recent_rewards.clear()
        self._recent_episode_success.clear()
        env_count = int(getattr(self.training_env, "num_envs", NUM_CPU))
        self._episode_running_rewards = np.zeros(max(1, env_count), dtype=np.float64)
        print(
            f"[Training] Started | mode={'GUI' if GUI_MODE else 'HEADLESS'} | "
            f"envs={env_count} | action_repeat={ACTION_REPEAT}+{ACTION_RELEASE_FRAMES} release frames | "
            f"SB3 verbosity={VERBOSE_LEVEL} | mGBA log mask={_MGBA_LOG_LEVEL_MASK} | "
            f"checkpoint every {self.save_frequency:,} transitions | "
            f"progress every {self.progress_frequency:,} transitions.",
            flush=True,
        )

    def _on_rollout_end(self):
        """Anneal exploration pressure instead of keeping a random-looking policy forever."""
        elapsed = max(0, int(self.num_timesteps) - self._run_start_timestep)
        progress = min(1.0, elapsed / float(max(1, TOTAL_TIMESTEPS)))
        self.model.ent_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress
        self.logger.record("progress/entropy_coefficient", float(self.model.ent_coef))

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

        # Monitor/VecEnv marks a finished episode in dones. Track task outcomes separately
        # from reward so a positive score cannot masquerade as successful gameplay.
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        for env_index, done in enumerate(dones):
            if not done or env_index >= len(infos_list):
                continue
            terminal_info = infos_list[env_index]
            success = bool(terminal_info.get("task_success", False))
            outcome = terminal_info.get("episode_outcome", "TIMEOUT")
            self._recent_episode_success.append(1 if success else 0)
            if success:
                self._task_successes += 1
            elif outcome == "TIMEOUT":
                self._timeouts += 1

        # Monitor/VecEnv marks a finished episode in dones. Reset its live return
        # after including that transition's reward, so the display tracks each new episode.
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

        for component_name in REWARD_COMPONENT_NAMES:
            component_values = [
                float(info[f"episode_{component_name}_sum"])
                for info in infos_list
                if isinstance(info, dict)
                and isinstance(info.get(f"episode_{component_name}_sum"), (int, float, np.integer, np.floating))
            ]
            if component_values:
                # Mean cumulative reward contribution across workers; terminal episodes
                # carry their final totals in the same info dict.
                self.logger.record(
                    f"progress/episode_{component_name}_sum", float(np.mean(component_values))
                )
        if self._recent_episode_success:
            self.logger.record("progress/task_success_rate_last_100_episodes", float(np.mean(self._recent_episode_success)))
        self.logger.record("progress/task_successes_total", float(self._task_successes))
        self.logger.record("progress/timeouts_total", float(self._timeouts))

        elapsed = max(1e-6, time.perf_counter() - self._run_started_at)
        steps_per_second = (int(self.num_timesteps) - self._run_start_timestep) / elapsed
        self.logger.record("progress/transitions_per_second", float(steps_per_second))
        scalar_metric_map = (
            ("party_count", "progress/party_count"),
            ("party_levels", "progress/party_total_levels"),
            ("badge_count", "progress/badge_count"),
            ("new_badges", "progress/new_badges_this_step"),
            ("level_reward", "progress/level_reward"),
            ("badge_reward", "progress/badge_reward"),
            ("exploration_reward", "progress/exploration_reward"),
            ("map_reward", "progress/map_reward"),
            ("tile_reward", "progress/tile_reward"),
            ("screen_reward", "progress/screen_reward"),
            ("unique_screens_episode", "progress/unique_screens_episode"),
            ("story_reward", "progress/story_reward"),
            ("party_reward", "progress/party_reward"),
            ("trainer_reward", "progress/trainer_reward"),
            ("loop_penalty", "progress/loop_penalty"),
            ("task_success_reward", "progress/task_success_reward"),
            ("party_members_added", "progress/party_members_added"),
            ("new_tile_discovered", "progress/new_tile_discovered"),
            ("new_area_discovered", "progress/new_area_discovered"),
            ("new_story_flags", "progress/new_story_flags"),
            ("new_trainer_flags", "progress/new_trainer_flags"),
            ("unique_tiles_episode", "progress/unique_tiles_episode"),
            ("unique_maps_episode", "progress/unique_maps_episode"),
            ("trainer_flags_episode", "progress/trainer_flags_episode"),
        )
        for info_key, tb_key in scalar_metric_map:
            metric_values = [
                float(info[info_key])
                for info in infos_list
                if isinstance(info, dict)
                and isinstance(info.get(info_key), (int, float, bool, np.integer, np.floating))
            ]
            if metric_values:
                self.logger.record(tb_key, float(np.mean(metric_values)))

        # Map ID and coordinates are categorical/per-worker state, so do not average
        # them into a meaningless fractional location when several emulators are active.
        for info_key, tb_key in (
            ("map_id", "progress/map_id"),
            ("map_group", "progress/map_group"),
            ("map_number", "progress/map_number"),
            ("player_x", "progress/player_x"),
            ("player_y", "progress/player_y"),
        ):
            for env_index, info in enumerate(infos_list):
                if isinstance(info, dict) and isinstance(info.get(info_key), (int, float, bool, np.integer, np.floating)):
                    self.logger.record(f"{tb_key}_env_{env_index}", float(info[info_key]))
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
            aggregate_gba_fps = steps_per_second * max(1, ACTION_REPEAT + ACTION_RELEASE_FRAMES)
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
            reward_part_names = ("tile_reward", "map_reward", "screen_reward", "story_reward", "party_reward", "level_reward", "badge_reward", "trainer_reward", "loop_penalty", "task_success_reward")
            reward_part_text = ",".join(
                f"{key.removesuffix('_reward').removesuffix('_penalty').removesuffix('_success')}={np.mean([float(info[key]) for info in infos_list if isinstance(info, dict) and isinstance(info.get(key), (int, float, np.integer, np.floating))]):+.3f}"
                for key in reward_part_names
                if any(isinstance(info, dict) and isinstance(info.get(key), (int, float, np.integer, np.floating)) for info in infos_list)
            )
            success_rate = float(np.mean(self._recent_episode_success)) if self._recent_episode_success else 0.0
            print(
                f"[Training] run={run_transitions:,}/{run_target:,} ({progress_percent:.2f}%) | "
                f"total_timesteps={int(self.num_timesteps):,} | "
                f"speed={steps_per_second:.1f} transitions/s aggregate "
                f"({per_env_speed:.1f}/env/s; ~{aggregate_gba_fps:.1f} GBA frames/s aggregate) | "
                f"{env_status} | elapsed={_format_duration(elapsed_seconds)} "
                f"ETA={_format_duration(remaining_seconds)} | action={action_text} | "
                f"step_reward={self._last_step_reward:.3f}, mean_reward_1k={recent_mean:.3f} | "
                f"{live_reward_text} | episodes={self._completed_episodes}{episode_text} | "
                f"task_successes={self._task_successes}, timeouts={self._timeouts}, success_rate_100={success_rate:.1%} | "
                f"reward_parts[{reward_part_text or 'n/a'}]",
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
            print(f" [MODE] TRAINING ACTIVE ({NUM_CPU} Environment(s)). Press Ctrl+C or q/Esc in GUI to stop and save.")
            print("--------------------------------------------------")

            if os.path.exists(model_file_path) and RESUME_TRAINING:
                print(f"[Load] Resuming hybrid policy from {model_file_path}...")
                try:
                    model = PPO.load(MODEL_CHECKPOINT, env=env, device="cpu", verbose=VERBOSE_LEVEL, tensorboard_log=TENSORBOARD_LOG_DIR)
                    _get_compatible_model_env(model, "Training resume")
                except Exception as exc:
                    raise RuntimeError(
                        f"Checkpoint {model_file_path} is incompatible with the current hybrid observations/action space. "
                        f"Preserve it and set POKEMON_AI_MODEL_NAME to a new filename before training. Cause: {exc}"
                    ) from exc
            elif os.path.exists(model_file_path) and not RESUME_TRAINING:
                raise FileExistsError(
                    f"POKEMON_AI_RESUME=0 means start fresh, but {model_file_path} already exists. "
                    "To avoid overwriting learned weights, rename that checkpoint or set POKEMON_AI_MODEL_NAME "
                    "to a new unique filename, then run again."
                )
            else:
                print(
                    "[Training] Starting a fresh recurrent hybrid policy (screenshots + game state + "
                    "visited-tile memory + LSTM temporal memory).",
                    flush=True,
                )
                model = PPO(
                    "MultiInputLstmPolicy", env, device="cpu", verbose=VERBOSE_LEVEL,
                    tensorboard_log=TENSORBOARD_LOG_DIR,
                    n_steps=PPO_N_STEPS,
                    batch_size=PPO_BATCH_SIZE,
                    n_epochs=PPO_N_EPOCHS,
                    gamma=PPO_GAMMA,
                    gae_lambda=PPO_GAE_LAMBDA,
                    learning_rate=PPO_LEARNING_RATE,
                    ent_coef=ENTROPY_COEF_START,
                    policy_kwargs={
                        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                        "lstm_hidden_size": int(os.environ.get("POKEMON_AI_LSTM_HIDDEN_SIZE", "128")),
                        "n_lstm_layers": 1,
                    },
                )
                print(
                    f"[Training] RecurrentPPO: n_steps={PPO_N_STEPS}, batch={PPO_BATCH_SIZE}, "
                    f"gamma={PPO_GAMMA}, gae_lambda={PPO_GAE_LAMBDA}, "
                    f"entropy={ENTROPY_COEF_START}->{ENTROPY_COEF_END}, "
                    f"episode_limit={MAX_EPISODE_STEPS} actions.",
                    flush=True,
                )

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

        elif MODE == "evaluate":
            print("--------------------------------------------------")
            print(
                f" [MODE] {'DETERMINISTIC' if EVAL_DETERMINISTIC else 'STOCHASTIC'} EVALUATION | "
                f"episodes={EVAL_EPISODES} | max_steps={MAX_EPISODE_STEPS}."
            )
            print(" [MODE] Uses the canonical training start, no training updates, no GUI and no video.")
            print("--------------------------------------------------", flush=True)

            chosen_checkpoint = MODEL_CHECKPOINT
            chosen_file = f"{chosen_checkpoint}.zip"
            if not os.path.isfile(chosen_file):
                raise FileNotFoundError(
                    f"No evaluation checkpoint found: {chosen_file}. Train the hybrid policy first."
                )
            model = PPO.load(chosen_checkpoint, env=env, device="cpu")
            model_env = _get_compatible_model_env(model, "Evaluation")
            print(f"[Evaluation] Loaded {chosen_file}", flush=True)

            observation = model_env.reset()
            lstm_states = None
            episode_starts = np.ones((int(getattr(model_env, "num_envs", 1)),), dtype=bool)
            results = []
            for episode_index in range(1, EVAL_EPISODES + 1):
                episode_return = 0.0
                episode_steps = 0
                episode_success = False
                episode_outcome = "UNKNOWN"
                episode_action_counts = Counter()
                last_info = {}
                last_action_name = "unknown"
                while not STOP_REQUESTED and episode_steps < MAX_EPISODE_STEPS:
                    action, lstm_states = model.predict(
                        observation,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=EVAL_DETERMINISTIC,
                    )
                    action_ids = np.asarray(action).reshape(-1)
                    for action_id in action_ids:
                        episode_action_counts[int(action_id)] += 1
                    last_action_name = ",".join(
                        ACTION_LABELS.get(int(action_id), f"action_{int(action_id)}")
                        for action_id in action_ids
                    )
                    observation, rewards, dones, infos = model_env.step(action)
                    episode_starts = np.asarray(dones, dtype=bool).reshape(-1)
                    last_info = infos[0] if infos and isinstance(infos[0], dict) else {}
                    episode_return += float(np.asarray(rewards, dtype=np.float64).reshape(-1)[0])
                    episode_steps += 1

                    if episode_steps % EVAL_DIAGNOSTICS_EVERY == 0:
                        pos_x = last_info.get("player_x", "?")
                        pos_y = last_info.get("player_y", "?")
                        print(
                            f"[Evaluation Diagnostic] episode={episode_index} step={episode_steps:,} "
                            f"last_action={last_action_name} pos=({pos_x},{pos_y}) "
                            f"map={last_info.get('map_id', '?')} "
                            f"party={last_info.get('party_count', '?')} "
                            f"tiles={last_info.get('unique_tiles_episode', '?')} "
                            f"maps={last_info.get('unique_maps_episode', '?')} "
                            f"story_flags={last_info.get('story_flags_episode', '?')} "
                            f"actions=[{_format_action_distribution(episode_action_counts)}]",
                            flush=True,
                        )

                    if bool(np.asarray(dones, dtype=bool).reshape(-1)[0]):
                        episode_success = bool(last_info.get("task_success", False))
                        episode_outcome = str(last_info.get("episode_outcome", "SUCCESS" if episode_success else "TIMEOUT"))
                        if isinstance(last_info.get("episode"), dict):
                            episode_return = float(last_info["episode"].get("r", episode_return))
                            episode_steps = int(last_info["episode"].get("l", episode_steps))
                        break
                if STOP_REQUESTED:
                    print("[Evaluation] Interrupted by user.", flush=True)
                    break
                results.append((episode_success, episode_return, episode_steps, episode_outcome))
                print(
                    f"[Evaluation Episode {episode_index}/{EVAL_EPISODES}] {episode_outcome} | "
                    f"success={episode_success} | return={episode_return:.3f} | steps={episode_steps:,} | "
                    f"actions=[{_format_action_distribution(episode_action_counts)}]",
                    flush=True,
                )
                # DummyVecEnv/SubprocVecEnv already autoresets on done. If an unexpected
                # loop exit did not produce done, force a clean canonical start before next test.
                if episode_steps < MAX_EPISODE_STEPS and not episode_success and episode_outcome == "UNKNOWN":
                    observation = model_env.reset()
                    lstm_states = None
                    episode_starts = np.ones((int(getattr(model_env, "num_envs", 1)),), dtype=bool)
            if results:
                success_rate = sum(1 for success, *_ in results if success) / len(results)
                mean_return = float(np.mean([episode_return for _, episode_return, _, _ in results]))
                mean_steps = float(np.mean([episode_steps for _, _, episode_steps, _ in results]))
                timeouts = sum(1 for _, _, _, outcome in results if outcome == "TIMEOUT")
                print(
                    f"[Evaluation Summary] episodes={len(results)} | success_rate={success_rate:.1%} | "
                    f"mean_return={mean_return:.3f} | mean_steps={mean_steps:.0f} | timeouts={timeouts}",
                    flush=True,
                )
            else:
                print("[Evaluation Summary] No complete episodes were evaluated.", flush=True)

        elif MODE == "setup":
            print("--------------------------------------------------")
            print(" [MODE] HUMAN SETUP. Control the game manually; this mode does not run PPO.")
            print(" [MODE] Save only when the character is on Route 101, the clock is set, and no starter is in the party.")
            print(f" [MODE] Training start save path: {TRAINING_START_STATE}")
            print(f" [MODE] RAM diagnostic interval: every {SETUP_DIAGNOSTICS_EVERY} actions.")
            print(" [MODE] Controls: arrows/WASD move, Z/Enter=A, X/Backspace=B, 1=Start, 2=Select, K=save, P=pause, lowercase q/Esc=quit.")
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
                if setup_steps % SETUP_DIAGNOSTICS_EVERY == 0:
                    try:
                        states = env.env_method("_read_game_state")
                        for worker_index, state in enumerate(states):
                            story = state.get("story_flags", set())
                            print(
                                f"[Setup RAM] env={worker_index} step={setup_steps:,} "
                                f"map={state.get('map_id', -1)} "
                                f"group={state.get('map_group', -1)} number={state.get('map_number', -1)} "
                                f"pos=({state.get('player_x')},{state.get('player_y')}) "
                                f"party={state.get('party_count', 0)} levels={state.get('party_levels', 0)} "
                                f"badges={state.get('badge_count', 0)} "
                                f"clock_set={CLOCK_SET_FLAG_ID in story} "
                                f"starter_flag={STARTER_OBTAINED_FLAG_ID in story} "
                                f"pokedex_flag={POKEDEX_OBTAINED_FLAG_ID in story}",
                                flush=True,
                            )
                    except Exception as exc:
                        print(f"[Setup RAM Warning] Could not read diagnostics: {exc}", flush=True)
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
            model_env = _get_compatible_model_env(model, "Watch mode")
            print(f"[Load] Loaded hybrid policy weights from {model_file_path}")
            send_notification("Pokémon AI Watch Mode", "Playback started. q/Esc quits; P pauses/resumes.")

            print(
                f"[Watch] Action selection: {'STOCHASTIC' if STOCHASTIC_WATCH else 'DETERMINISTIC'}.",
                flush=True,
            )
            print("[Watch] Plays from the canonical Route 101 start and stops when the first starter is obtained.", flush=True)
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

            obs = model_env.reset()
            lstm_states = None
            episode_starts = np.ones((int(getattr(model_env, "num_envs", 1)),), dtype=bool)
            watch_started = time.perf_counter()
            watch_last_report = watch_started
            watch_steps = 0
            watch_last_report_step = 0
            watch_completed_episodes = 0
            watch_episode_reward = 0.0
            watch_last_step_reward = 0.0
            watch_recent_rewards = deque(maxlen=PROGRESS_LOG_FREQUENCY)
            watch_task_completed = False
            while not STOP_REQUESTED:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=(not STOCHASTIC_WATCH),
                )
                obs, rewards, dones, infos = model_env.step(action)
                episode_starts = np.asarray(dones, dtype=bool).reshape(-1)
                watch_steps += int(getattr(model_env, "num_envs", 1))
                reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
                if reward_values.size:
                    watch_last_step_reward = float(np.mean(reward_values))
                    watch_recent_rewards.extend(float(value) for value in reward_values)
                    watch_episode_reward += float(np.sum(reward_values))
                info_values = infos if isinstance(infos, (list, tuple)) else []
                done_values = np.asarray(dones, dtype=bool).reshape(-1)
                # Stop at the requested starter milestone instead of continuing into
                # gameplay the model was never trained to handle.
                if any(
                    isinstance(info, dict) and info.get("task_success", False)
                    for info in info_values
                ):
                    watch_task_completed = True
                    print(
                        "[Watch] TASK SUCCESS: starter obtained. Stopping at the objective; "
                        "START/SELECT remain available and are not blocked.",
                        flush=True,
                    )
                    break
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
                        f"(~{speed * (ACTION_REPEAT + ACTION_RELEASE_FRAMES):.1f} GBA frames/s) | elapsed={_format_duration(elapsed)} | "
                        f"step_reward={watch_last_step_reward:.3f} | "
                        f"recent_mean_reward={recent_mean:.3f} | "
                        f"live_episode_reward={watch_episode_reward:.2f} | episodes={watch_completed_episodes}",
                        flush=True,
                    )
                    watch_last_report = now
                    watch_last_report_step = watch_steps
                if GUI_MODE:
                    try:
                        if any(model_env.get_attr("stop_requested")):
                            break
                    except Exception:
                        pass
            if not watch_task_completed:
                print("[Watch] Playback stopped before the starter objective completed.", flush=True)

        else:
            raise ValueError(f"Unsupported MODE={MODE!r}; choose 'train', 'watch', 'setup', or 'evaluate'.")

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
