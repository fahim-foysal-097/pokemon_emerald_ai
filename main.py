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
import hashlib
import json
import textwrap
from collections import Counter, deque

from important_logging import filter_persistent_log_text, should_flush_important_log, is_exception_end
from curriculum_objectives import (
    OBJECTIVE_DEFINITIONS, OBJECTIVES_REQUIRING_FLAG_TELEMETRY, STAGES,
    ROUTE_101_MAP_ID, objective_satisfied as _objective_satisfied,
    objective_start_error, resolve_runtime_start_state_path,
)


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
    important_overrides = []
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
                elif key in {"POKEMON_AI_OBJECTIVE", "POKEMON_AI_CURRICULUM_STAGE_ONLY"}:
                    shell_value = os.environ[key].strip()
                    file_value = value.strip()
                    if shell_value.lower() != file_value.lower():
                        important_overrides.append((key, shell_value, file_value))
        if loaded_keys:
            print(
                f"[Config] Loaded {len(loaded_keys)} setting(s) from {env_path}; "
                "existing shell variables take precedence.",
                flush=True,
            )
        for key, shell_value, file_value in important_overrides:
            print(
                f"[Config Warning] Shell environment overrides .env: {key}={shell_value!r}; "
                f".env contains {file_value!r}. The shell value is active.",
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
        "label": "Fixed-start evaluation (deterministic or stochastic; five episodes by default)",
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

    if sb3_verbosity <= 1 and (
        payload.startswith(b"GBA Serial I/O: Switching mode from GPIO to MULTI")
        or payload.startswith(b"GBA I/O: Write to read-only I/O register: 006")
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
from runtime_profiler import RuntimeProfiler
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


def _format_progress_table(title, rows, width=None):
    """Render a wrapped ASCII table sized to the actual terminal when no width is supplied."""
    if width is None:
        try:
            # Keep table borders aligned instead of letting the terminal soft-wrap
            # long rows into a visually broken second line.
            width = shutil.get_terminal_size(fallback=(120, 24)).columns - 1
        except (OSError, ValueError):
            width = 119
        width = max(96, min(140, int(width)))
    inner_width = max(88, int(width) - 2)
    border = "+" + "-" * inner_width + "+"
    lines = [border, "|" + str(title).center(inner_width)[:inner_width].ljust(inner_width) + "|"]
    lines.append("|" + "-" * inner_width + "|")
    for label, value in rows:
        prefix = f" {str(label):<17} | "
        available = max(20, inner_width - len(prefix))
        chunks = textwrap.wrap(
            str(value), width=available, break_long_words=True,
            break_on_hyphens=False, replace_whitespace=True,
        ) or [""]
        for index, chunk in enumerate(chunks):
            body = (prefix if index == 0 else " " * len(prefix)) + chunk
            lines.append("|" + body[:inner_width].ljust(inner_width) + "|")
    lines.append(border)
    return "\n".join(lines)


def _numeric_info_pairs(infos, key):
    """Return finite numeric values paired with their vector-env index."""
    pairs = []
    for env_index, info in enumerate(infos if isinstance(infos, (list, tuple)) else []):
        if not isinstance(info, dict):
            continue
        value = _finite_number(info.get(key))
        if value is not None:
            pairs.append((env_index, float(value)))
    return pairs


def _verify_emerald_rom_identity(rom_path, expected_sha1=None, require_sha1=True):
    """Verify the header and optional full-image checksum before creating an mGBA core."""
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
    digest = hashlib.sha1()
    with open(rom_path, "rb") as rom_file:
        for chunk in iter(lambda: rom_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha1 = digest.hexdigest()
    if require_sha1:
        expected = str(expected_sha1 or "").strip().lower()
        if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(
                "POKEMON_AI_EXPECTED_ROM_SHA1 must be a 40-character hexadecimal SHA-1 when "
                "POKEMON_AI_REQUIRE_ROM_SHA1=1."
            )
        if actual_sha1 != expected:
            raise ValueError(
                "ROM identity check failed. This project expects the exact ROM with SHA-1 "
                f"{expected}, but {os.path.basename(rom_path)} has {actual_sha1}. "
                "No emulator was started. If this is an intentional alternate build, verify its "
                "memory map first and set POKEMON_AI_EXPECTED_ROM_SHA1 explicitly."
            )
    return {"game_code": game_code, "revision": revision, "sha1": actual_sha1}


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
        _early_expected_sha1 = os.environ.get(
            "POKEMON_AI_EXPECTED_ROM_SHA1", "f3ae088181bf583e55daf962a92bb46f4f1d07b7"
        ).strip().lower()
        _early_require_sha1 = _env_flag("POKEMON_AI_REQUIRE_ROM_SHA1", True)
        _early_rom_identity = _verify_emerald_rom_identity(
            _bootstrap_rom_path, _early_expected_sha1, _early_require_sha1
        )
        print(
            f"[ROM Preflight] Verified before mGBA bootstrap: BPEE revision "
            f"{_early_rom_identity['revision']}, SHA-1 {_early_rom_identity['sha1']}.",
            flush=True,
        )
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
    from sb3_contrib import RecurrentPPO
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
CHECKPOINT_ARCHIVE_DIR = os.path.join(RUNTIME_DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ROM_PATH = os.path.join(BASE_DIR, "PokemonEmerald.gba")
EXPECTED_ROM_SHA1 = os.environ.get(
    "POKEMON_AI_EXPECTED_ROM_SHA1", "f3ae088181bf583e55daf962a92bb46f4f1d07b7"
).strip().lower()
REQUIRE_ROM_SHA1 = _env_flag("POKEMON_AI_REQUIRE_ROM_SHA1", True)
MILESTONE_DIR = os.path.join(STATE_DIR, "milestones")
MILESTONE_SAVE_ENABLED = _env_flag("POKEMON_AI_MILESTONE_SAVE", True)
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
# This is stage-specific when explicitly configured. Resolve the stage-aware
# fallback after OBJECTIVE_NAME is known, using the shared curriculum mapping.
_configured_start_state = os.path.expanduser(
    os.environ.get("POKEMON_AI_TRAINING_START_STATE", "").strip()
)
TRAINING_START_STATE = _DEFAULT_TRAINING_START_STATE  # replaced after OBJECTIVE_NAME is resolved
REQUIRE_TRAINING_START_STATE = os.environ.get(
    "POKEMON_AI_REQUIRE_START_STATE", "1"
).strip().lower() not in {"0", "false", "no", "off"}
# Select a curriculum objective in .env; the shared recurrent policy architecture stays
# the same as the agent progresses from the starter task to later story milestones.
_configured_tensorboard_dir = os.environ.get("POKEMON_AI_TENSORBOARD_LOG_DIR", "tensorboard_logs_v2").strip()
TENSORBOARD_LOG_DIR = (
    os.path.abspath(os.path.expanduser(_configured_tensorboard_dir))
    if os.path.isabs(os.path.expanduser(_configured_tensorboard_dir))
    else os.path.join(BASE_DIR, _configured_tensorboard_dir or "tensorboard_logs_v2")
)

# GUI / headless and mode are selected together by the active profile above.
# GUI profiles use a single emulator; fast_train uses four; both watch profiles use one.
GUI_MODE = os.environ.get("POKEMON_AI_GUI_MODE", "0").strip().lower() not in {"0", "false", "no", "off"}
GUI_WINDOW_TITLE = "Pokemon Emerald PPO"
GUI_SCALE = 4                     # 240x160 native frame * 4 = 960x640 window
GUI_FPS = 60                      # GUI only: pace rendering near the real GBA refresh rate.
# Headless watch speed cap in emulated frames/sec. 0 = unlimited/fast-forward.
WATCH_FPS_CAP = float(os.environ.get("POKEMON_AI_WATCH_FPS", "0"))
if not math.isfinite(WATCH_FPS_CAP) or WATCH_FPS_CAP < 0:
    raise ValueError("POKEMON_AI_WATCH_FPS must be a finite number >= 0 (0 means unlimited speed)")
WATCH_SUCCESS_GRACE_STEPS = int(os.environ.get("POKEMON_AI_WATCH_SUCCESS_GRACE_STEPS", "300"))
if WATCH_SUCCESS_GRACE_STEPS < 0:
    raise ValueError("POKEMON_AI_WATCH_SUCCESS_GRACE_STEPS must be >= 0")

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
_configured_max_episode_steps = int(os.environ.get("POKEMON_AI_MAX_EPISODE_STEPS", "30000"))
if _configured_max_episode_steps < 1:
    raise ValueError("POKEMON_AI_MAX_EPISODE_STEPS must be a positive integer")
# Curriculum evaluation uses a shorter objective-specific cap supplied by
# run_curriculum.py. Training retains the configured episode horizon unchanged.
_eval_max_episode_steps = int(os.environ.get(
    "POKEMON_AI_EVAL_MAX_EPISODE_STEPS", str(_configured_max_episode_steps)
)) if MODE == "evaluate" else _configured_max_episode_steps
if _eval_max_episode_steps < 1:
    raise ValueError("POKEMON_AI_EVAL_MAX_EPISODE_STEPS must be a positive integer")
MAX_EPISODE_STEPS = _eval_max_episode_steps
# Soft exploration mini-episodes periodically clear only short-term exploration memory
# during longer objectives. The game state, PPO episode, reward totals and LSTM state stay live.
# Starter training remains unchanged; zero disables mini-episodes for all objectives.
MINI_EPISODE_STEPS = int(os.environ.get("POKEMON_AI_MINI_EPISODE_STEPS", "5000"))
if MINI_EPISODE_STEPS < 0:
    raise ValueError("POKEMON_AI_MINI_EPISODE_STEPS must be >= 0 (0 disables mini-episodes)")
RAM_READ_FAILURE_LIMIT = int(os.environ.get("POKEMON_AI_RAM_READ_FAILURE_LIMIT", "30"))
if RAM_READ_FAILURE_LIMIT < 1:
    raise ValueError("POKEMON_AI_RAM_READ_FAILURE_LIMIT must be a positive integer")
EVAL_DIAGNOSTICS_EVERY = int(os.environ.get("POKEMON_AI_EVAL_DIAGNOSTICS_EVERY", "5000"))
if EVAL_DIAGNOSTICS_EVERY < 1:
    raise ValueError("POKEMON_AI_EVAL_DIAGNOSTICS_EVERY must be a positive integer")
# Early-stop only the evaluation episode when repeated actions make no progress.
# The agent's action is never overridden by this diagnostic guard.
EVAL_STAGNATION_STEPS = int(os.environ.get("POKEMON_AI_EVAL_STAGNATION_STEPS", "128"))
if EVAL_STAGNATION_STEPS < 16:
    raise ValueError("POKEMON_AI_EVAL_STAGNATION_STEPS must be >= 16")
# Unlike the exact-repeat guard above, this counter detects loops where the
# policy varies its buttons but makes no measurable progress in the game.
EVAL_NO_PROGRESS_STEPS = int(os.environ.get("POKEMON_AI_EVAL_NO_PROGRESS_STEPS", "4096"))
if EVAL_NO_PROGRESS_STEPS < 128:
    raise ValueError("POKEMON_AI_EVAL_NO_PROGRESS_STEPS must be >= 128")
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
# Pokémon EXP is encrypted in the Gen III party structure. Poll it at a low cadence and
# grant a small shaping reward when EXP increases. This rewards progress after wild or trainer
# Pokémon are defeated without treating entering a battle, catching a Pokémon, or running away
# as a victory. It is an EXP-gain signal, not a perfect battle-outcome detector.
BATTLE_EXP_REWARD_PER_100 = float(os.environ.get("POKEMON_AI_BATTLE_EXP_REWARD_PER_100", "0.10"))
BATTLE_EXP_REWARD_CAP_PER_POLL = float(os.environ.get("POKEMON_AI_BATTLE_EXP_REWARD_CAP_PER_POLL", "0.50"))
BATTLE_EXP_POLL_INTERVAL = int(os.environ.get("POKEMON_AI_BATTLE_EXP_POLL_INTERVAL", "8"))
# Trainer-defeated flags are persistent in Emerald, so reading their full range every
# policy decision is unnecessary. Poll periodically and reuse the cached set between polls.
TRAINER_FLAG_POLL_INTERVAL = int(os.environ.get("POKEMON_AI_TRAINER_FLAG_POLL_INTERVAL", "8"))
TIMING_DIAGNOSTICS = _env_flag("POKEMON_AI_TIMING_DIAGNOSTICS", False)
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
    "BATTLE_EXP_REWARD_PER_100": BATTLE_EXP_REWARD_PER_100,
    "BATTLE_EXP_REWARD_CAP_PER_POLL": BATTLE_EXP_REWARD_CAP_PER_POLL,
    "LOOP_PENALTY": LOOP_PENALTY,
    "TASK_SUCCESS_REWARD": TASK_SUCCESS_REWARD,
}.items():
    if _reward_value < 0:
        raise ValueError(f"{_reward_name} must be >= 0")
if BATTLE_EXP_POLL_INTERVAL < 1:
    raise ValueError("POKEMON_AI_BATTLE_EXP_POLL_INTERVAL must be a positive integer")
if TRAINER_FLAG_POLL_INTERVAL < 1:
    raise ValueError("POKEMON_AI_TRAINER_FLAG_POLL_INTERVAL must be a positive integer")
if LOOP_STAGNATION_STEPS < 1 or LOOP_PENALTY_INTERVAL < 1:
    raise ValueError("Loop-detection step settings must be positive integers")
SAVE_BLOCK1_PTR_ADDRESS = 0x03005D8C
SAVE_BLOCK1_FLAGS_OFFSET = 0x1270
SAVE_BLOCK1_POS_X_OFFSET = 0x00
SAVE_BLOCK1_POS_Y_OFFSET = 0x02
SAVE_BLOCK1_LOCATION_OFFSET = 0x04  # struct WarpData: mapGroup, mapNum, warpId, x, y
# Vanilla Pokémon Emerald BPEE revision 0 live globals from pret/pokeemerald's symbols:
# gPlayerPartyCount and gPlayerParty. SaveBlock1's party is a saved copy and can lag during
# gameplay until SavePlayerParty() copies the live party into the save block.
PLAYER_PARTY_COUNT_ADDRESS = 0x020244E9
PLAYER_PARTY_ADDRESS = 0x020244EC
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
# FLAG_BADGE01_GET through FLAG_BADGE08_GET are 0x867..0x86E in standard Emerald.
BADGE_FLAG_IDS = tuple(range(0x867, 0x86F))
FLAG_BYTE_OFFSETS = tuple(sorted({flag_id >> 3 for flag_id in set(OBS_STORY_FLAG_IDS) | set(BADGE_FLAG_IDS)}))
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
    "trainer", "battle_exp", "loop", "task_success",
)

OBJECTIVE_NAME = os.environ.get("POKEMON_AI_OBJECTIVE", "starter").strip().lower()
if OBJECTIVE_NAME not in OBJECTIVE_DEFINITIONS:
    raise ValueError(
        "POKEMON_AI_OBJECTIVE must be one of: " + ", ".join(OBJECTIVE_DEFINITIONS)
    )
OBJECTIVE_LABEL = OBJECTIVE_DEFINITIONS[OBJECTIVE_NAME]["label"]


def _start_state_recovery_hint() -> str:
    """Give the recovery action appropriate to the selected curriculum stage."""
    stage_index = STAGES.index(OBJECTIVE_NAME)
    if stage_index == 0:
        return (
            "Recreate the pre-starter Route 101 state with ./run.sh --profile setup, "
            "then press K after setting the clock and before obtaining a starter."
        )
    previous_stage = STAGES[stage_index - 1]
    return (
        f"The required input is the verified {previous_stage} milestone. Use run_curriculum.py "
        f"to complete/evaluate {previous_stage} and publish its milestone; do not substitute "
        "the canonical pre-starter state or a different stage save."
    )


# Start-state selection is centralized with the curriculum stage definitions.
# Setup always targets the canonical Route 101 save; other modes require the
# configured stage-specific path to match the preceding verified milestone.
TRAINING_START_STATE = str(resolve_runtime_start_state_path(
    BASE_DIR,
    OBJECTIVE_NAME,
    mode=MODE,
    canonical_start_state=os.environ.get("POKEMON_AI_CANONICAL_START_STATE", "").strip() or None,
    training_start_state=_configured_start_state or None,
))

if MODE == "setup" and _configured_start_state:
    _configured_setup_state = os.path.expanduser(_configured_start_state)
    if not os.path.isabs(_configured_setup_state):
        _configured_setup_state = os.path.join(BASE_DIR, _configured_setup_state)
    if os.path.realpath(_configured_setup_state) != os.path.realpath(TRAINING_START_STATE):
        print(
            "[Setup Config] Ignoring stage-specific POKEMON_AI_TRAINING_START_STATE="
            f"{os.path.realpath(_configured_setup_state)}; setup saves only to the canonical "
            f"Route 101 state {TRAINING_START_STATE}.",
            flush=True,
        )


def _decode_gen3_growth_words(personality, ot_id, encrypted_species_item, encrypted_experience):
    """Decode the two encrypted Growth-substructure words needed for EXP reward shaping.

    Emerald encrypts each 32-bit word with personality XOR OT ID. The first word contains
    species and held item; the second stores EXP. Reject implausible species/EXP to avoid
    rewarding on an invalid pointer or a RAM read during emulator initialization.
    """
    try:
        key = (int(personality) ^ int(ot_id)) & 0xFFFFFFFF
        species_item = (int(encrypted_species_item) ^ key) & 0xFFFFFFFF
        experience = (int(encrypted_experience) ^ key) & 0xFFFFFFFF
        species = species_item & 0xFFFF
    except (TypeError, ValueError, OverflowError):
        return None
    if not (1 <= species <= 412 and 0 <= experience <= 2_000_000):
        return None
    return experience


def _finite_number(value):
    """Convert a scalar metric to a finite float; reject strings, containers and NaN/Inf."""
    if not isinstance(value, (int, float, bool, np.integer, np.floating)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _completed_episode_summaries(infos, dones, running_returns=None):
    """Return valid summaries only for genuinely completed vector episodes.

    SB3's Monitor puts exact episode return/length under ``info["episode"]`` at
    termination. Environment reward-component keys are running sums on every step,
    so they must only be consumed when the matching ``dones[i]`` is true. Episodes
    stopped by a human are deliberately excluded from training success/return windows.
    """
    infos_list = list(infos) if isinstance(infos, (list, tuple)) else []
    done_values = np.asarray(dones if dones is not None else [], dtype=bool).reshape(-1)
    returns = np.asarray(running_returns if running_returns is not None else [], dtype=np.float64).reshape(-1)
    summaries = []
    for env_index, done in enumerate(done_values):
        if not bool(done):
            continue
        info = infos_list[env_index] if env_index < len(infos_list) else {}
        if not isinstance(info, dict):
            info = {}
        outcome = str(info.get("episode_outcome", "TIMEOUT") or "TIMEOUT").strip().upper()
        if info.get("stopped", False) or outcome == "STOPPED":
            continue

        monitored = info.get("episode")
        monitored = monitored if isinstance(monitored, dict) else {}
        fallback_return = float(returns[env_index]) if env_index < returns.size else 0.0
        episode_return = _finite_number(monitored.get("r"))
        if episode_return is None:
            episode_return = _finite_number(info.get("episode_return"))
        if episode_return is None:
            episode_return = fallback_return

        episode_length = _finite_number(monitored.get("l"))
        if episode_length is None:
            episode_length = _finite_number(info.get("episode_steps"))
        if episode_length is None:
            episode_length = 0.0

        component_payload = info.get("episode_reward_components")
        component_payload = component_payload if isinstance(component_payload, dict) else {}
        component_sums = {}
        for component_name in REWARD_COMPONENT_NAMES:
            component_value = _finite_number(info.get(f"episode_{component_name}_sum"))
            if component_value is None:
                component_value = _finite_number(component_payload.get(component_name))
            if component_value is not None:
                component_sums[component_name] = component_value

        summaries.append({
            "env_index": env_index,
            "outcome": outcome,
            "success": bool(info.get("task_success", outcome == "SUCCESS")),
            "episode_return": float(episode_return),
            "episode_length": max(0, int(episode_length)),
            "component_sums": component_sums,
        })
    return summaries


def _episode_time_limit_reached(mode, episode_steps):
    """Training episodes time out; watch/setup are continuous until user stop."""
    return str(mode).strip().lower() in {"train", "evaluate"} and int(episode_steps) >= MAX_EPISODE_STEPS


def _mini_episode_boundary_due(
    mode, objective, episode_steps, interval, *, terminated=False, truncated=False,
    stopped=False, state_valid=True, task_success=False,
):
    """Return whether a non-terminal exploration-memory reset should happen now.

    These are deliberately not Gymnasium episodes: no done signal is emitted, no
    emulator savestate is restored, and the recurrent policy state remains intact.
    Only later objectives use them; the starter curriculum is unchanged.
    """
    mode = str(mode or "").strip().lower()
    objective = str(objective or "").strip().lower()
    try:
        step = int(episode_steps)
        every = int(interval)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        mode in {"train", "evaluate", "watch"}
        and objective in OBJECTIVE_DEFINITIONS and objective != "starter"
        and every > 0
        and step > 0
        and step % every == 0
        and not terminated
        and not truncated
        and not stopped
        and state_valid
        and not task_success
    )


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


def _training_start_validation_error(state, objective=None):
    """Validate a start state appropriate for the selected curriculum stage."""
    if not isinstance(state, dict):
        return "game-state decoding returned no usable state dictionary"
    objective = str(objective or OBJECTIVE_NAME).strip().lower()
    if objective not in OBJECTIVE_DEFINITIONS:
        return "unknown objective; choose one of: " + ", ".join(OBJECTIVE_DEFINITIONS)

    if state.get("saveblock_pointer_stable", True) is False:
        return "SaveBlock1 pointer changed during the state read; map/position telemetry is not trustworthy"

    coordinates = (_finite_number(state.get("player_x")), _finite_number(state.get("player_y")))
    if any(value is None or not value.is_integer() or value < 0 or value > 255 for value in coordinates):
        return "player coordinates are invalid; save while the character is in normal gameplay"
    map_id_value = _finite_number(state.get("map_id"))
    if map_id_value is None or not map_id_value.is_integer() or int(map_id_value) < 0:
        return "map ID is missing or invalid; verify the BPEE revision 0 RAM mapping"
    map_id = int(map_id_value)

    story_flags = state.get("story_flags")
    if not isinstance(story_flags, (set, frozenset, list, tuple, range)):
        return "story flags are missing or invalid; verify the SaveBlock1 pointer and RAM mapping"
    party_count = _finite_number(state.get("party_count"))
    party_levels = _finite_number(state.get("party_levels"))
    if (
        party_count is None or not party_count.is_integer() or not 0 <= party_count <= 6
        or party_levels is None or not party_levels.is_integer() or not 0 <= party_levels <= 600
    ):
        return "party count/levels are missing or invalid; verify the live-party RAM mapping"
    badge_count = _finite_number(state.get("badge_count"))
    if badge_count is None:
        badge_flags = _finite_number(state.get("badge_flags", 0))
        if badge_flags is None or badge_flags < 0 or not badge_flags.is_integer():
            return "badge telemetry is missing or invalid"
        badge_count = float(int(badge_flags).bit_count())
    if not badge_count.is_integer() or not 0 <= badge_count <= 8:
        return "badge count is outside the valid Emerald range"

    # The shared validator keeps direct training/evaluation and curriculum-stage
    # prerequisites in sync. A state may already satisfy a later objective because
    # Emerald advances a scripted transition automatically; in that case the task
    # is allowed to complete immediately instead of rejecting a valid savestate.
    return objective_start_error(
        objective,
        state,
        route_101_map_id=ROUTE_101_MAP_ID,
    )

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
EVAL_SEED = int(os.environ.get("POKEMON_AI_EVAL_SEED", "1234"))
if EVAL_SEED < 0:
    raise ValueError("POKEMON_AI_EVAL_SEED must be >= 0")
VERBOSE_LEVEL = int(os.environ.get("POKEMON_AI_VERBOSE", "1"))
if VERBOSE_LEVEL not in (0, 1, 2):
    raise ValueError("POKEMON_AI_VERBOSE must be 0, 1, or 2")


def _debug_log(*args, **kwargs):
    """Emit repetitive lifecycle diagnostics only in verbose/debug mode or setup mode."""
    if VERBOSE_LEVEL >= 2 or MODE == "setup":
        print(*args, **kwargs)
STOCHASTIC_WATCH = _env_flag("POKEMON_AI_STOCHASTIC_WATCH", False)
EVAL_DETERMINISTIC = _env_flag("POKEMON_AI_EVAL_DETERMINISTIC", True)
PPO_N_STEPS = int(os.environ.get("POKEMON_AI_PPO_N_STEPS", "2048"))
PPO_BATCH_SIZE = int(os.environ.get("POKEMON_AI_PPO_BATCH_SIZE", "256"))
PPO_N_EPOCHS = int(os.environ.get("POKEMON_AI_PPO_N_EPOCHS", "10"))
PPO_GAMMA = float(os.environ.get("POKEMON_AI_PPO_GAMMA", "0.999"))
PPO_GAE_LAMBDA = float(os.environ.get("POKEMON_AI_PPO_GAE_LAMBDA", "0.98"))
PPO_LEARNING_RATE = float(os.environ.get("POKEMON_AI_PPO_LEARNING_RATE", "0.0003"))
ENTROPY_COEF_START = float(os.environ.get("POKEMON_AI_ENTROPY_COEF_START", "0.03"))
ENTROPY_COEF_END = float(os.environ.get("POKEMON_AI_ENTROPY_COEF_END", "0.01"))
ENTROPY_COEF_COLLAPSE_FLOOR = float(os.environ.get("POKEMON_AI_ENTROPY_COEF_COLLAPSE_FLOOR", "0.03"))
ACTION_COLLAPSE_DOMINANCE = float(os.environ.get("POKEMON_AI_ACTION_COLLAPSE_DOMINANCE", "0.85"))
if min(PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS) < 1:
    raise ValueError("PPO n_steps, batch_size, and n_epochs must be positive integers")
if not 0.0 < PPO_GAMMA <= 1.0 or not 0.0 < PPO_GAE_LAMBDA <= 1.0:
    raise ValueError("PPO gamma and GAE lambda must be in (0, 1]")
if min(PPO_LEARNING_RATE, ENTROPY_COEF_START, ENTROPY_COEF_END, ENTROPY_COEF_COLLAPSE_FLOOR) < 0:
    raise ValueError("PPO learning rate and entropy coefficients must be >= 0")
if not 0.0 < ACTION_COLLAPSE_DOMINANCE <= 1.0:
    raise ValueError("POKEMON_AI_ACTION_COLLAPSE_DOMINANCE must be in (0, 1]")

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
# Default to a much less frequent checkpoint cadence. configure_training.py aligns this
# with the active vector rollout; this fallback is for direct launch without the configurator.
_REQUESTED_CHECKPOINT_SAVE_FREQUENCY = int(os.environ.get("POKEMON_AI_CHECKPOINT_EVERY", "122880"))
CHECKPOINT_ARCHIVE_KEEP = int(os.environ.get("POKEMON_AI_CHECKPOINT_KEEP", "20"))
LATEST_CHECKPOINT_BACKUP_ENABLED = _env_flag("POKEMON_AI_LATEST_BACKUP_ENABLED", True)
PROGRESS_LOG_FREQUENCY = int(os.environ.get("POKEMON_AI_PROGRESS_EVERY", "8192"))
if _REQUESTED_CHECKPOINT_SAVE_FREQUENCY < 1:
    raise ValueError("POKEMON_AI_CHECKPOINT_EVERY must be a positive integer")
# A model checkpoint must never be written from the middle of collect_rollouts().
# Round the requested cadence up to whole vector rollouts; SafeRecurrentPPO only
# publishes a due checkpoint after the matching optimizer update has completed.
VECTOR_ROLLOUT_TRANSITIONS = NUM_CPU * PPO_N_STEPS
CHECKPOINT_SAVE_FREQUENCY = (
    (_REQUESTED_CHECKPOINT_SAVE_FREQUENCY + VECTOR_ROLLOUT_TRANSITIONS - 1)
    // VECTOR_ROLLOUT_TRANSITIONS
) * VECTOR_ROLLOUT_TRANSITIONS
if CHECKPOINT_SAVE_FREQUENCY != _REQUESTED_CHECKPOINT_SAVE_FREQUENCY:
    print(
        f"[Checkpoint] Requested interval {_REQUESTED_CHECKPOINT_SAVE_FREQUENCY:,} was aligned "
        f"to {CHECKPOINT_SAVE_FREQUENCY:,} transitions ({VECTOR_ROLLOUT_TRANSITIONS:,} per PPO rollout).",
        flush=True,
    )
if CHECKPOINT_ARCHIVE_KEEP < 0:
    raise ValueError("POKEMON_AI_CHECKPOINT_KEEP must be >= 0 (0 disables rolling snapshots)")
if PROGRESS_LOG_FREQUENCY < 1:
    raise ValueError("POKEMON_AI_PROGRESS_EVERY must be a positive integer")
TERMINAL_PROGRESS_FREQUENCY = int(os.environ.get("POKEMON_AI_TERMINAL_PROGRESS_EVERY", "10000"))
if TERMINAL_PROGRESS_FREQUENCY < 1:
    raise ValueError("POKEMON_AI_TERMINAL_PROGRESS_EVERY must be a positive integer")
TERMINAL_EPISODE_EVERY = int(os.environ.get("POKEMON_AI_TERMINAL_EPISODE_EVERY", "50"))
if TERMINAL_EPISODE_EVERY < 1:
    raise ValueError("POKEMON_AI_TERMINAL_EPISODE_EVERY must be a positive integer")

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
PARTY_PERSONALITY_OFFSET = 0x00
PARTY_OT_ID_OFFSET = 0x04
PARTY_ENCRYPTED_BLOCK_OFFSET = 0x20
GEN3_SUBSTRUCT_ORDERS = (
    "GAEM", "GAME", "GEAM", "GEMA", "GMAE", "GMEA",
    "AGEM", "AGME", "AEGM", "AEMG", "AMGE", "AMEG",
    "EGAM", "EGMA", "EAGM", "EAMG", "EMGA", "EMAG",
    "MGAE", "MGEA", "MAGE", "MAEG", "MEGA", "MEAG",
)
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

def _policy_action_probability_summary(model, observation, lstm_states, episode_starts):
    """Summarize the recurrent categorical policy without advancing its live state."""
    try:
        import torch
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        if lstm_states is None:
            hidden_shape = tuple(getattr(model.policy, "lstm_hidden_state_shape"))
            zeros = torch.zeros(hidden_shape, dtype=torch.float32, device=model.device)
            state = (zeros, zeros.clone())
        else:
            if not isinstance(lstm_states, (tuple, list)) or len(lstm_states) < 2:
                return None
            state = tuple(torch.as_tensor(x, dtype=torch.float32, device=model.device) for x in lstm_states[:2])
        starts = torch.as_tensor(np.asarray(episode_starts, dtype=np.float32), dtype=torch.float32, device=model.device)
        with torch.no_grad():
            distribution, _ = model.policy.get_distribution(obs_tensor, state, starts)
            categorical = getattr(distribution, "distribution", distribution)
            probs = getattr(categorical, "probs", None)
            if probs is None:
                return None
            probs = probs.detach().float().cpu().numpy().reshape(-1)
        if probs.size != len(ACTION_LABELS) or not np.all(np.isfinite(probs)):
            return None
        probs = np.clip(probs, 0.0, 1.0)
        total = float(probs.sum())
        if total <= 0:
            return None
        probs /= total
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        top_indices = np.argsort(probs)[::-1][:4]
        top = ", ".join(f"{ACTION_LABELS.get(int(i), str(i))}={probs[i]:.1%}" for i in top_indices)
        return f"top: {top} | H={entropy:.2f}/{math.log(len(ACTION_LABELS)):.2f} | maxP={float(probs.max()):.1%}"
    except Exception:
        # Policy internals differ a little across sb3-contrib releases; diagnostics must
        # never break evaluation just because that release exposes a different API.
        return None


def _screen_observation_signature(observation):
    """Coarse image signature for detecting a static stuck loop during evaluation."""
    try:
        image = np.asarray(observation["screen"])
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            return None
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] < 3:
            return None
        image = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (16, 12), interpolation=cv2.INTER_AREA)
        quantized = (small // 16).astype(np.uint8)
        return hashlib.blake2s(quantized.tobytes(), digest_size=8).hexdigest()
    except Exception:
        return None


def _evaluation_progress_signature(info):
    """Return monotonic gameplay telemetry, excluding coordinates/action identity.

    An agent can move around an already explored room or alternate buttons forever.
    Coordinates and action identity therefore do not count as objective progress.
    New tiles/maps, changed story flags, trainer flags, party growth, or badges do.
    """
    if not isinstance(info, dict) or bool(info.get("ram_read_failed", False)):
        return None

    def scalar_int(key, default=-1):
        value = info.get(key, default)
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return number

    raw_flags = info.get("story_flags", ())
    if isinstance(raw_flags, (set, frozenset, list, tuple)):
        try:
            story_flags = tuple(sorted(int(flag) for flag in raw_flags))
        except (TypeError, ValueError, OverflowError):
            story_flags = ()
    else:
        story_flags = ()

    return (
        # map_id itself is deliberately excluded: toggling between two already
        # visited maps must not keep an otherwise stagnant evaluation alive.
        # First-time maps are represented by the cumulative unique_maps value.
        scalar_int("party_count"),
        scalar_int("party_levels"),
        scalar_int("badge_count"),
        bool(info.get("starter_flag", False)),
        bool(info.get("pokedex_flag", False)),
        story_flags,
        scalar_int("trainer_flag_count"),
        scalar_int("unique_tiles_episode"),
        scalar_int("unique_maps_episode"),
        scalar_int("story_flags_episode"),
        scalar_int("trainer_flags_episode"),
    )


def _evaluation_has_explicit_progress(info):
    """Detect cumulative progression events not yet reflected in the telemetry snapshot.

    Do not use ``new_tile_discovered`` or ``new_area_discovered`` here: the
    environment intentionally clears its *mini-episode* novelty sets every few
    thousand steps. Revisiting an old tile after that reset is reward novelty,
    not new game progress. First-time tiles/maps are already counted by the
    cumulative ``unique_tiles_episode``/``unique_maps_episode`` fields in the
    progress signature.
    """
    if not isinstance(info, dict) or bool(info.get("ram_read_failed", False)):
        return False
    for key in ("new_story_flags", "new_trainer_flags", "new_badges_this_step", "party_experience_delta"):
        try:
            if float(info.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError, OverflowError):
            continue
    return False


# mGBA Python bindings expose raw core snapshots, not the frontend saveStateFile API.
RAW_STATE_MAGIC = b"PYMGBA_RAW_STATE_V1\x00"


tb_process = None
tb_owned = False
tb_log_handle = None
STOP_REQUESTED = False
STOP_SIGNAL_NAME = None
SAFE_STOP_AFTER_UPDATE = False
ACTIVE_TRAINING_CALLBACK = None
LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP = None
LAST_SUCCESSFUL_EMULATOR_SAVE_TIMESTEP = None
LAST_COMPLETED_UPDATE_TIMESTEP = None
PPO_UPDATE_FAILED = False


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


def _atomic_write_json(path, payload):
    """Publish JSON metadata atomically so a crash cannot leave a partial report."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_dir(path)
        return True
    except Exception:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        raise


def _sha1_file(path):
    digest = hashlib.sha1()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_rom_header(rom_path):
    """Verify BPEE revision 0 and the configured exact ROM image checksum."""
    identity = _verify_emerald_rom_identity(rom_path, EXPECTED_ROM_SHA1, REQUIRE_ROM_SHA1)
    if REQUIRE_ROM_SHA1:
        print(
            f"[ROM] Exact identity verified: BPEE revision {identity['revision']}, "
            f"SHA-1 {identity['sha1']}.", flush=True,
        )
    else:
        print(
            f"[ROM Warning] Header is BPEE revision {identity['revision']}; "
            f"SHA-1={identity['sha1']}. Exact hash enforcement is disabled by "
            "POKEMON_AI_REQUIRE_ROM_SHA1=0.", flush=True,
        )
    return identity["sha1"]


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
        # Reuse a native scratch array when headless audio is drained and discarded.
        self._drain_scratch = None
        self._drain_scratch_values = 0
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
        """Drain native samples; avoid NumPy allocation when headless samples are discarded."""
        if self.buffer is None or self.ffi is None:
            return
        try:
            available = int(self.available_audio(self.buffer))
            needs_python_samples = bool(
                (record and self.record_wave is not None)
                or (accept and self.output_active and not self.muted)
            )
            # Keep the reusable native scratch allocation bounded if the ring backs up.
            max_frames_per_read = 4096
            while available > 0:
                frames_to_read = min(available, max_frames_per_read)
                values_needed = frames_to_read * self.channels
                if self._drain_scratch is None or self._drain_scratch_values < values_needed:
                    self._drain_scratch = self.ffi.new("int16_t[]", values_needed)
                    self._drain_scratch_values = values_needed
                count = int(self.read_audio(self.buffer, self._drain_scratch, frames_to_read))
                if count <= 0:
                    break
                available -= count
                self.samples_drained += count

                # fast_train/headless: remove samples from mGBA's ring without creating
                # a NumPy array or copying PCM that nobody will play or record.
                if not needs_python_samples:
                    continue

                byte_count = count * self.channels * self.ffi.sizeof("int16_t")
                samples = np.frombuffer(
                    self.ffi.buffer(self._drain_scratch, byte_count), dtype=np.int16
                ).reshape(count, self.channels).copy()

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
    """Show all console output, but persist only important events and final reports."""

    def __init__(self, console_stream, log_stream, lock):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self.lock = lock
        self._capturing_traceback = False

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    def write(self, value):
        if not value:
            return 0
        # Do not change interactive diagnostics: every message still appears on screen.
        self.console_stream.write(value)
        with self.lock:
            if "Traceback (most recent call last)" in value:
                self._capturing_traceback = True
            if self._capturing_traceback:
                persisted = value
            else:
                persisted = filter_persistent_log_text(value)
            if persisted:
                self.log_stream.write(persisted)
                if should_flush_important_log(persisted):
                    self.log_stream.flush()
            if self._capturing_traceback and is_exception_end(value):
                self._capturing_traceback = False
        return len(value)

    def flush(self):
        # Console flushes are frequent because some prints use flush=True. Do not
        # turn each one into a disk flush; selected important events flush in write().
        self.console_stream.flush()

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
        self._consecutive_ram_read_failures = 0
        # Known SaveBlock1 party-count mismatch signatures are logged once per worker,
        # not once per short successful training episode.
        self._last_party_debug_signature = None
        self._pending_reset_elapsed_ms = 0.0
        self._cached_trainer_flags = set()
        self._cached_trainer_flags_valid = False
        self._story_flag_baseline_verified = False
        self._trainer_flag_baseline_verified = False
        self._badge_baseline_verified = False
        self._last_verified_story_flags = None
        self._last_verified_badge_flags = None
        self._last_verified_flag_bytes = None
        self._last_verified_trainer_flags = None
        self._last_flag_diag_signature = None
        self.flag_telemetry_invalid_steps = 0
        self.trainer_flag_telemetry_invalid_steps = 0
        self.seen_badge_flags = 0
        self._training_success_logged_once = False
        self._terminal_episode_count = 0
        # Full Gymnasium-episode metrics remain cumulative until true termination.
        self.visited_maps = set()
        self.visited_tiles = set()
        self.episode_seen_screen_signatures = set()
        # Short-term discovery memory is what the policy sees and mini-episodes reset.
        self.mini_visited_maps = set()
        self.mini_visited_tiles = set()
        self.seen_screen_signatures = set()
        self.seen_story_flags = set()
        self.seen_trainer_flags = set()
        self.position_history = deque(maxlen=32)
        self.screen_signature_history = deque(maxlen=32)
        self.mini_episode_count = 0
        # Lifetime counts decay novelty across episode resets without rewarding repeats
        # every frame. They persist per emulator worker for the duration of a run.
        self.lifetime_tile_visits = Counter()
        self.lifetime_map_visits = Counter()
        self.lifetime_screen_visits = Counter()
        self.mini_episode_screen_novelty_reward = 0.0
        self.last_novelty_step = 0
        self.episode_reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        self._last_game_state = None
        self._task_success_rewarded = False

        # Tracking variables for reward shaping deltas
        self.prev_party_levels = 0
        self.prev_party_experience = None
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

        _debug_log(f"[Init] Pokémon Emerald environment initialized in [{MODE.upper()}] mode.")

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
                _debug_log(f"[Boot] Calling mGBA run_frame() for the first frame{suffix}...", flush=True)
            run_frame()
            self._pump_audio_buffer(accept=False)
            if frame_index == 0:
                _debug_log("[Boot] First mGBA frame completed.", flush=True)
            if total >= 100 and (frame_index + 1) % 100 == 0:
                _debug_log(f"[Boot] Advanced {frame_index + 1}/{total} GBA frames", flush=True)

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
            current_tile_seen = float((map_id, int(x), int(y)) in getattr(self, "mini_visited_tiles", self.visited_tiles))

        values = [
            max(0.0, min(1.0, float(state.get("map_group", 0)) / 31.0)),
            max(0.0, min(1.0, float(state.get("map_number", 0)) / 100.0)),
            max(0.0, min(1.0, float(x or 0) / 512.0)),
            max(0.0, min(1.0, float(y or 0) / 512.0)),
            max(0.0, min(1.0, float(state.get("party_count", 0)) / 6.0)),
            max(0.0, min(1.0, float(state.get("party_levels", 0)) / 600.0)),
            max(0.0, min(1.0, float(badge_flags.bit_count()) / 8.0)),
            max(0.0, min(1.0, len(getattr(self, "mini_visited_tiles", self.visited_tiles)) / 1024.0)),
            max(0.0, min(1.0, len(getattr(self, "mini_visited_maps", self.visited_maps)) / 64.0)),
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
        active_tiles = getattr(self, "mini_visited_tiles", self.visited_tiles)
        for visited_map, x, y in active_tiles:
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

    def _read_flag_byte_snapshot(self, saveblock1_ptr):
        """Read every byte containing a curated story or badge flag exactly once."""
        if saveblock1_ptr is None:
            raise ValueError("Cannot read SaveBlock1 flags without a valid pointer")
        flags_base = saveblock1_ptr + SAVE_BLOCK1_FLAGS_OFFSET
        return {
            int(byte_offset): self._read8(flags_base + int(byte_offset))
            for byte_offset in FLAG_BYTE_OFFSETS
        }

    @staticmethod
    def _decode_flag_set(flag_ids, byte_snapshot):
        """Decode event-flag IDs from one internally consistent byte snapshot."""
        found = set()
        for flag_id in flag_ids:
            byte_value = byte_snapshot.get(int(flag_id) >> 3)
            if byte_value is not None and byte_value & (1 << (int(flag_id) & 7)):
                found.add(int(flag_id))
        return found

    def _read_story_flags(self, saveblock1_ptr, byte_snapshot=None):
        """Read curated persistent story flags from a supplied stable snapshot."""
        if saveblock1_ptr is None:
            return set()
        if byte_snapshot is None:
            byte_snapshot = self._read_flag_byte_snapshot(saveblock1_ptr)
        return self._decode_flag_set(STORY_FLAG_REWARDS, byte_snapshot)

    def _read_badge_flags(self, saveblock1_ptr, byte_snapshot=None):
        """Return the 8-bit badge mask from the official system-flag bytes."""
        if saveblock1_ptr is None:
            return 0
        if byte_snapshot is None:
            byte_snapshot = self._read_flag_byte_snapshot(saveblock1_ptr)
        mask = 0
        for badge_index, flag_id in enumerate(BADGE_FLAG_IDS):
            byte_value = byte_snapshot.get(flag_id >> 3, 0)
            if byte_value & (1 << (flag_id & 7)):
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

    def _read_party_experience(self, party_count):
        """Read total party EXP cheaply, decoding only the encrypted Growth words.

        This is polled every few policy decisions rather than every frame/step, limiting RAM
        inspection overhead. `None` indicates invalid/unavailable telemetry.
        """
        try:
            party_count = int(party_count)
            if not 0 <= party_count <= 6:
                return None
            total = 0
            for index in range(party_count):
                mon_base = PLAYER_PARTY_ADDRESS + index * PARTY_POKEMON_SIZE
                personality = self._read32(mon_base + PARTY_PERSONALITY_OFFSET)
                ot_id = self._read32(mon_base + PARTY_OT_ID_OFFSET)
                order = GEN3_SUBSTRUCT_ORDERS[personality % 24]
                growth_base = mon_base + PARTY_ENCRYPTED_BLOCK_OFFSET + order.index("G") * 12
                encrypted_species_item = self._read32(growth_base)
                encrypted_experience = self._read32(growth_base + 4)
                experience = _decode_gen3_growth_words(
                    personality, ot_id, encrypted_species_item, encrypted_experience
                )
                if experience is None:
                    return None
                total += experience
            return total
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return None

    def _party_debug_signature_changed(self, signature):
        """Return True only when the observed live-vs-save party mismatch changes."""
        previous = getattr(self, "_last_party_debug_signature", None)
        if signature == previous:
            return False
        self._last_party_debug_signature = signature
        return True

    def _read_game_state(self, include_trainer_flags=True):
        """Read BPEE state and accept persistent flags only from a stable RAM snapshot.

        SaveBlock1 is dynamically relocated. We read its pointer before and after the
        relevant flag bytes, and read the flag bytes twice. A changing pointer or byte
        snapshot is not trusted. The most recent verified snapshot is retained for the
        current emulator episode rather than converting transient reads into rewards.
        """
        saveblock1_ptr = self._read_saveblock1_pointer()
        if saveblock1_ptr is None:
            raise ValueError("SaveBlock1 pointer is not initialized/valid yet")

        # Capture the curated story/badge flag bytes before reading the rest of the state.
        flag_bytes_first = self._read_flag_byte_snapshot(saveblock1_ptr)

        # Read both versions for diagnostics, but trust the live globals for gameplay.
        # SavePlayerParty() copies the globals into SaveBlock1 on save; its party fields
        # are not guaranteed to change immediately when the starter is given.
        saveblock_party_count = self._read8(saveblock1_ptr + SAVE_BLOCK1_PARTY_COUNT_OFFSET)
        party_count = self._read8(PLAYER_PARTY_COUNT_ADDRESS)
        if party_count > 6:
            raise ValueError(
                f"Unexpected live gPlayerPartyCount={party_count} at "
                f"0x{PLAYER_PARTY_COUNT_ADDRESS:08X}; RAM mapping/ROM revision may be wrong."
            )

        total_party_levels = 0
        party_slot_levels = []
        party_base = PLAYER_PARTY_ADDRESS
        for i in range(party_count):
            mon_base = party_base + (i * PARTY_POKEMON_SIZE)
            level = self._read8(mon_base + PARTY_LEVEL_OFFSET)
            party_slot_levels.append(level)
            if 1 <= level <= 100:
                total_party_levels += level

        # SaveBlock1.location is WarpData at offset 0x04; its map group/number
        # fields are signed bytes in the decompilation.
        map_group = self._read8(saveblock1_ptr + SAVE_BLOCK1_LOCATION_OFFSET)
        map_number = self._read8(saveblock1_ptr + SAVE_BLOCK1_LOCATION_OFFSET + 1)
        if map_group >= 0x80:
            map_group -= 0x100
        if map_number >= 0x80:
            map_number -= 0x100
        if 0 <= map_group <= 127 and 0 <= map_number <= 127:
            map_id = (map_group << 8) | map_number
        else:
            map_id = -1
        player_x, player_y = self._read_player_position(saveblock1_ptr)

        badge_flags_raw = self._read_badge_flags(saveblock1_ptr, flag_bytes_first)
        story_flags_raw = self._read_story_flags(saveblock1_ptr, flag_bytes_first)

        # Trainer flags are scanned twice on polling steps. This is a small bounded
        # cost (108 bytes per scan, normally once every eight policy decisions) and
        # prevents a torn read from being counted as hundreds of trainer victories.
        trainer_scan_performed = bool(include_trainer_flags or not hasattr(self, "_cached_trainer_flags"))
        trainer_flags_first = set()
        trainer_scan_stable = bool(getattr(self, "_cached_trainer_flags_valid", False))
        if trainer_scan_performed:
            trainer_flags_first = self._read_trainer_flags(saveblock1_ptr)
            trainer_flags_second = self._read_trainer_flags(saveblock1_ptr)
            trainer_scan_stable = trainer_flags_first == trainer_flags_second

        # Re-read the same curated bytes and pointer after all related reads. A
        # snapshot is valid only if the source remained identical throughout.
        flag_bytes_second = self._read_flag_byte_snapshot(saveblock1_ptr)
        pointer_after = self._read_saveblock1_pointer()
        saveblock_pointer_stable = pointer_after == saveblock1_ptr
        flag_bytes_stable = flag_bytes_first == flag_bytes_second
        flag_snapshot_valid = bool(saveblock_pointer_stable and flag_bytes_stable)
        trainer_snapshot_valid = bool(flag_snapshot_valid and trainer_scan_stable)

        # Preserve the raw decoded location for diagnostics, but expose it to the
        # policy/objective/reward code only when the SaveBlock1 pointer stayed fixed.
        raw_map_id = int(map_id)
        raw_player_x = player_x
        raw_player_y = player_y
        if not saveblock_pointer_stable:
            map_id = -1
            player_x = None
            player_y = None

        raw_flag_bytes = {
            f"0x{offset:03X}": int(value)
            for offset, value in flag_bytes_first.items()
        }
        # Curated Emerald event flags, badge flags and trainer-defeated flags are
        # persistent for this task. Keep verified values monotonic within an episode.
        previous_story_flags = set(getattr(self, "_last_verified_story_flags", None) or ())
        previous_badge_flags = int(getattr(self, "_last_verified_badge_flags", None) or 0)
        previous_trainer_flags = set(getattr(self, "_last_verified_trainer_flags", None) or ())
        # A regression is diagnostic only when both samples were verified. An
        # unstable/torn snapshot must not be presented as proof that a persistent
        # story, badge, or trainer flag actually disappeared.
        story_flag_regressions = (
            sorted(previous_story_flags - set(story_flags_raw))
            if flag_snapshot_valid else []
        )
        badge_flag_regressions = (
            previous_badge_flags & ~int(badge_flags_raw)
            if flag_snapshot_valid else 0
        )
        trainer_flag_regressions = (
            sorted(previous_trainer_flags - set(trainer_flags_first))
            if trainer_snapshot_valid else []
        )
        if flag_snapshot_valid:
            verified_story_flags = previous_story_flags | set(story_flags_raw)
            verified_badge_flags = previous_badge_flags | int(badge_flags_raw)
            self._last_verified_story_flags = set(verified_story_flags)
            self._last_verified_badge_flags = int(verified_badge_flags)
            self._last_verified_flag_bytes = dict(flag_bytes_first)
        else:
            verified_story_flags = previous_story_flags
            verified_badge_flags = previous_badge_flags

        if trainer_snapshot_valid:
            verified_trainer_flags = previous_trainer_flags | set(trainer_flags_first)
            self._cached_trainer_flags = set(verified_trainer_flags)
            self._cached_trainer_flags_valid = True
            self._last_verified_trainer_flags = set(verified_trainer_flags)
        last_verified_trainer_flags = getattr(self, "_last_verified_trainer_flags", None)
        cached_trainer_flags = getattr(self, "_cached_trainer_flags", set())
        trainer_flags = set(
            last_verified_trainer_flags
            if last_verified_trainer_flags is not None
            else cached_trainer_flags
        )

        # In vanilla Emerald, FLAG_SYS_POKEMON_GET should be persistent. However,
        # the live party is the stronger gameplay fact: once a Pokémon is in the
        # validated live party, expose starter_obtained as true even if the raw
        # SaveBlock1 event bit is temporarily unreadable or inconsistent.
        effective_story_flags = set(verified_story_flags)
        if party_count >= 1 and total_party_levels >= 1:
            effective_story_flags.add(STARTER_OBTAINED_FLAG_ID)
        starter_flag_raw = (
            STARTER_OBTAINED_FLAG_ID in story_flags_raw if flag_snapshot_valid else None
        )

        if not flag_snapshot_valid:
            flag_error = (
                "SaveBlock1 pointer changed during the flag read"
                if not saveblock_pointer_stable
                else "curated SaveBlock1 flag bytes changed during the double-read"
            )
        else:
            flag_error = None
        if trainer_snapshot_valid:
            trainer_error = None
        elif not flag_snapshot_valid:
            trainer_error = flag_error
        else:
            trainer_error = "trainer flag range changed during the double-read"

        badge_flags = int(verified_badge_flags)
        return {
            "party_count": party_count,
            "party_levels": total_party_levels,
            "party_slot_levels": tuple(party_slot_levels),
            "party_count_saveblock1": saveblock_party_count,
            "party_count_source": "live_gPlayerPartyCount",
            "badge_flags": badge_flags,
            "badge_count": badge_flags.bit_count(),
            "raw_badge_flags": int(badge_flags_raw),
            "map_id": map_id,
            "raw_map_id": raw_map_id,
            "map_group": map_group,
            "map_number": map_number,
            "player_x": player_x,
            "player_y": player_y,
            "raw_player_x": raw_player_x,
            "raw_player_y": raw_player_y,
            "story_flags": effective_story_flags,
            "raw_story_flags": set(story_flags_raw),
            "trainer_flags": trainer_flags,
            "raw_trainer_flag_count": len(trainer_flags_first) if trainer_scan_performed else len(trainer_flags),
            "starter_flag_raw": starter_flag_raw,
            "starter_obtained": bool(STARTER_OBTAINED_FLAG_ID in effective_story_flags),
            "flag_telemetry_valid": bool(flag_snapshot_valid),
            "trainer_flag_telemetry_valid": bool(trainer_snapshot_valid),
            "saveblock_pointer_stable": bool(saveblock_pointer_stable),
            "flag_telemetry_error": flag_error,
            "trainer_flag_telemetry_error": trainer_error,
            "story_flag_regressions": story_flag_regressions,
            "badge_flag_regressions": int(badge_flag_regressions),
            "trainer_flag_regressions": trainer_flag_regressions,
            "saveblock1_ptr": int(saveblock1_ptr),
            "raw_flag_bytes": raw_flag_bytes,
        }

    def _refresh_reward_baseline(self):
        try:
            state = self._read_game_state()
            self.prev_party_levels = state["party_levels"]
            self.prev_party_experience = self._read_party_experience(state["party_count"])
            self.prev_badges = state["badge_flags"]
            if state.get("flag_telemetry_valid", False):
                self.seen_badge_flags = int(state["badge_flags"]) & 0xFF
                self._badge_baseline_verified = True
                self.seen_story_flags.update(state.get("story_flags", set()))
                self._story_flag_baseline_verified = True
            else:
                self._badge_baseline_verified = False
                self._story_flag_baseline_verified = False
            if state.get("trainer_flag_telemetry_valid", False):
                self.seen_trainer_flags.update(state.get("trainer_flags", set()))
                self._trainer_flag_baseline_verified = True
            else:
                self._trainer_flag_baseline_verified = False
            self.prev_caught_count = state["party_count"]
            self.prev_map_id = state["map_id"]
            if state.get("saveblock_pointer_stable", False) and state["map_id"] >= 0:
                map_id = int(state["map_id"])
                self.visited_maps.add(map_id)
                self.mini_visited_maps.add(map_id)
                self.lifetime_map_visits[map_id] += 1
            if (
                state.get("saveblock_pointer_stable", False)
                and state["map_id"] >= 0
                and state.get("player_x") is not None
                and state.get("player_y") is not None
            ):
                position = (int(state["map_id"]), int(state["player_x"]), int(state["player_y"]))
                self.visited_tiles.add(position)
                self.mini_visited_tiles.add(position)
                self.lifetime_tile_visits[position] += 1
                self.position_history.append(position)
            screen_signature = _screen_signature(self._get_image_obs())
            if screen_signature is not None:
                self.seen_screen_signatures.add(screen_signature)
                self.episode_seen_screen_signatures.add(screen_signature)
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
            _debug_log(f"[Reset {self.env_id}] entered", flush=True)
            super().reset(seed=seed)
            # The previous-action one-hot is part of the next observation; clear it
            # at episode boundaries so a terminal B/A/etc. does not leak into episode 2.
            self.last_action_id = 0
            self.visited_maps.clear()
            self.visited_tiles.clear()
            self.episode_seen_screen_signatures.clear()
            self.mini_visited_maps.clear()
            self.mini_visited_tiles.clear()
            self.seen_screen_signatures.clear()
            self.seen_story_flags.clear()
            self.seen_trainer_flags.clear()
            self.seen_badge_flags = 0
            self._last_verified_story_flags = None
            self._last_verified_badge_flags = None
            self._last_verified_flag_bytes = None
            self._last_verified_trainer_flags = None
            self._cached_trainer_flags = set()
            self._cached_trainer_flags_valid = False
            self._story_flag_baseline_verified = False
            self._trainer_flag_baseline_verified = False
            self._badge_baseline_verified = False
            self.flag_telemetry_invalid_steps = 0
            self.trainer_flag_telemetry_invalid_steps = 0
            self.position_history.clear()
            self.screen_signature_history.clear()
            self.mini_episode_count = 0
            self.mini_episode_screen_novelty_reward = 0.0
            self.last_novelty_step = 0
            self.episode_reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
            self._last_game_state = None
            self._task_success_rewarded = False
            self._consecutive_ram_read_failures = 0
            # Keep _last_party_debug_signature across episodes. A static SaveBlock/live
            # mismatch should be reported once per worker, not again after every reset.

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
                    if self.env_id == 0 or VERBOSE_LEVEL >= 2:
                        print(
                            f"[Start State] Loading mapped start state for objective={OBJECTIVE_NAME}: "
                            f"{TRAINING_START_STATE}", flush=True
                        )
                    restored_checkpoint = self._load_state_file(TRAINING_START_STATE)
                    if not restored_checkpoint:
                        raise RuntimeError(
                            f"Mapped stage-start savestate exists but could not be loaded: {TRAINING_START_STATE}. "
                            f"{_start_state_recovery_hint()}"
                        )
                elif MODE in {"train", "evaluate", "watch"} and REQUIRE_TRAINING_START_STATE:
                    raise FileNotFoundError(
                        f"Required mapped stage-start savestate not found: {TRAINING_START_STATE}\n"
                        f"{_start_state_recovery_hint()}"
                    )
                elif MODE != "setup" and os.path.isfile(self.checkpoint_file):
                    print(f"[Load] Attempting legacy emulator checkpoint: {self.checkpoint_file}", flush=True)
                    restored_checkpoint = self._load_state_file(self.checkpoint_file)
                    if not restored_checkpoint:
                        print("[State Warning] Checkpoint could not be loaded; starting from a clean reset.", flush=True)

                if restored_checkpoint:
                    _debug_log("[Reset] Checkpoint restored; refreshing one frame.", flush=True)
                    self._advance_frame_for_observation(1)
                    if MODE in {"train", "watch", "evaluate"} and os.path.isfile(TRAINING_START_STATE):
                        # Validate the loaded live game state against this objective's prerequisites.
                        try:
                            start_state = self._read_game_state()
                        except Exception as exc:
                            raise RuntimeError(
                                f"Mapped stage-start state is not readable as an active Emerald game: {exc}. "
                                f"{_start_state_recovery_hint()}"
                            ) from exc
                        validation_error = _training_start_validation_error(
                            start_state, objective=OBJECTIVE_NAME
                        )
                        if validation_error:
                            raise RuntimeError(
                                f"Mapped stage-start state is invalid for objective={OBJECTIVE_NAME}: "
                                f"{validation_error}. {_start_state_recovery_hint()}"
                            )
                        if self.env_id == 0 or VERBOSE_LEVEL >= 2:
                            print(
                                f"[Start State] Validated start for objective={OBJECTIVE_NAME} | "
                                f"party_count={start_state['party_count']} | map_id={start_state['map_id']} | "
                                f"player=({start_state['player_x']},{start_state['player_y']})",
                                flush=True,
                            )
                else:
                    _debug_log(
                        f"[New] Cold start: advancing {INITIAL_BOOT_FRAMES} frames so the first observation is rendered.",
                        flush=True,
                    )
                    # _init_emulator() already attached the framebuffer and reset this core.
                    # Do not reset it a second time here; simply advance it as in the verified
                    # standalone mGBA probe.
                    self._advance_frame_for_observation(INITIAL_BOOT_FRAMES)

                self.episode_step_counter = 0
                _debug_log(f"[Reset {self.env_id}] building initial observation", flush=True)
                obs = self._get_obs()
                _debug_log(
                    f"[Reset {self.env_id}] observation ready: screen={obs['screen'].shape}, "
                    f"state={obs['state'].shape}, visited={obs['visited'].shape}",
                    flush=True,
                )
                # Show the game before native snapshot or RAM-reward operations, so a later stall
                # cannot leave the window looking like video never worked.
                if GUI_MODE:
                    self._render_gui(time.perf_counter())

                _debug_log(f"[Reset {self.env_id}] capturing episode-start state", flush=True)
                self._capture_episode_start()
                _debug_log(f"[Reset {self.env_id}] episode-start state captured", flush=True)
                self._first_reset = False
            else:
                _debug_log(f"[Reset {self.env_id}] restoring episode-start state", flush=True)
                restored_episode = self._restore_episode_start()
                _debug_log(f"[Reset {self.env_id}] episode restore result: {restored_episode}", flush=True)
                self._advance_frame_for_observation(1 if restored_episode else INITIAL_BOOT_FRAMES)
                self.episode_step_counter = 0
                obs = self._get_obs()
                if GUI_MODE:
                    self._render_gui(time.perf_counter())

            # The canonical start is restored on train episode resets. Novelty and reward
            # component trackers are episode-local; seed them with the current start tile/flags.
            _debug_log(f"[Reset {self.env_id}] refreshing reward baseline", flush=True)
            self._refresh_reward_baseline()
            obs = self._get_obs(game_state=self._last_game_state, read_state=False)
            _debug_log(
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
            if TIMING_DIAGNOSTICS and MODE == "train":
                self._pending_reset_elapsed_ms = max(
                    0.0, time.perf_counter() - reset_started
                ) * 1000.0
            faulthandler.cancel_dump_traceback_later()

    def step(self, action):
        timing_enabled = bool(TIMING_DIAGNOSTICS and MODE == "train")
        step_started = time.perf_counter() if timing_enabled else 0.0
        frame_elapsed = image_elapsed = state_elapsed = observation_elapsed = 0.0
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
                    frame_started = time.perf_counter() if (timing_enabled or GUI_MODE or WATCH_FPS_CAP > 0) else 0.0
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
                    if timing_enabled:
                        frame_elapsed += time.perf_counter() - frame_started
                if self.stop_requested or STOP_REQUESTED:
                    break
            self._set_action_keys(None)

        # Save after releasing the previous action's key, avoiding a snapshot with a held key.
        if MODE == "setup" and self._save_training_start_requested:
            self._save_training_start_requested = False
            try:
                setup_state = self._read_game_state()
                # Setup always captures the canonical pre-starter savestate, even when a
                # later curriculum objective is selected in .env.
                validation_error = _training_start_validation_error(setup_state, objective="starter")
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
        image_started = time.perf_counter() if timing_enabled else 0.0
        screen_frame = self._get_image_obs() if self.core is not None else None
        current_screen_signature = _screen_signature(screen_frame)
        if timing_enabled:
            image_elapsed = time.perf_counter() - image_started
        state = None
        component = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        component["step"] = -STEP_PENALTY
        info = {
            "episode_step": self.episode_step_counter,
            "objective_name": OBJECTIVE_NAME,
            "objective_label": OBJECTIVE_LABEL,
            "task_success": False,
            "objective_just_completed": False,
            "ram_read_failed": False,
            "mini_episode_boundary": False,
            "mini_episode_count": int(getattr(self, "mini_episode_count", 0)),
        }
        if timing_enabled:
            info["timing_reset_ms"] = float(getattr(self, "_pending_reset_elapsed_ms", 0.0))
            self._pending_reset_elapsed_ms = 0.0
        terminated = False
        if current_screen_signature is not None:
            self.screen_signature_history.append(current_screen_signature)
        if self.core is not None:
            state_started = time.perf_counter() if timing_enabled else 0.0
            try:
                poll_trainer_flags = (
                    self.episode_step_counter % TRAINER_FLAG_POLL_INTERVAL == 0
                    or not hasattr(self, "_cached_trainer_flags")
                )
                state = self._read_game_state(include_trainer_flags=poll_trainer_flags)
                self._last_game_state = state

                # Keep stale SaveBlock party data visible for diagnostics, but never use it
                # as the live party source. Emit only when the observed mismatch changes.
                live_party_count = int(state["party_count"])
                saved_party_count = int(state["party_count_saveblock1"])
                starter_flag_raw = state.get("starter_flag_raw")
                starter_flag_set = bool(
                    state.get("starter_obtained", False)
                    or (live_party_count >= 1 and int(state.get("party_levels", 0)) >= 1)
                )
                if not state.get("flag_telemetry_valid", False):
                    self.flag_telemetry_invalid_steps += 1
                if not state.get("trainer_flag_telemetry_valid", False):
                    self.trainer_flag_telemetry_invalid_steps += 1

                # Print the raw system-flag bytes when they change. This tells us
                # whether a false starter bit comes from an actual byte change or
                # from an unstable pointer/byte sample, without flooding every step.
                raw_bytes = state.get("raw_flag_bytes", {})
                flag_diag_signature = (
                    state.get("saveblock1_ptr"),
                    raw_bytes.get("0x10C"),
                    raw_bytes.get("0x10D"),
                    bool(state.get("flag_telemetry_valid", False)),
                )
                if flag_diag_signature != self._last_flag_diag_signature:
                    self._last_flag_diag_signature = flag_diag_signature
                    print(
                        f"[SaveBlock1 Flags] env={self.env_id} ptr=0x{int(state.get('saveblock1_ptr', 0)):08X} "
                        f"stable={bool(state.get('flag_telemetry_valid', False))} "
                        f"ptr_stable={bool(state.get('saveblock_pointer_stable', False))} "
                        f"bytes[0x10C/0x10D]={raw_bytes.get('0x10C', '??')}/{raw_bytes.get('0x10D', '??')} "
                        f"starter_raw={starter_flag_raw} starter_effective={starter_flag_set} "
                        f"party={live_party_count} levels={state.get('party_levels', 0)} "
                        f"error={state.get('flag_telemetry_error') or 'none'}",
                        flush=True,
                    )

                # A small reward for actual party EXP gain encourages defeating wild Pokémon
                # even when the EXP gain is too small to trigger a level-up. Party count must
                # remain stable across samples, so receiving/catching a new Pokémon is not
                # misclassified as battle EXP. This is a progress proxy, not a battle-screen
                # or gBattleOutcome detector; trainer victories still use trainer flags below.
                experience_delta = 0
                if self.episode_step_counter % BATTLE_EXP_POLL_INTERVAL == 0:
                    current_party_experience = self._read_party_experience(live_party_count)
                    previous_party_experience = getattr(self, "prev_party_experience", None)
                    if current_party_experience is not None:
                        if (
                            previous_party_experience is not None
                            and live_party_count == int(getattr(self, "prev_caught_count", live_party_count))
                            and current_party_experience > previous_party_experience
                        ):
                            experience_delta = current_party_experience - previous_party_experience
                            component["battle_exp"] = min(
                                BATTLE_EXP_REWARD_CAP_PER_POLL,
                                experience_delta * BATTLE_EXP_REWARD_PER_100 / 100.0,
                            )
                        self.prev_party_experience = current_party_experience
                    else:
                        # Re-establish a clean baseline after invalid/transient RAM telemetry.
                        self.prev_party_experience = None
                if live_party_count != saved_party_count:
                    party_signature = (
                        live_party_count, saved_party_count, int(state["party_levels"]),
                        tuple(state.get("party_slot_levels", ())),
                    )
                    if self._party_debug_signature_changed(party_signature):
                        party_rows = [
                            ("Authority", "live gPlayerPartyCount (SaveBlock1 is diagnostic only)"),
                            ("Party count", f"live={live_party_count} | SaveBlock1={saved_party_count}"),
                            ("Pokémon levels", f"slots={list(state.get('party_slot_levels', ()))!r} | total={state['party_levels']}"),
                            ("Starter flag (raw)", str(starter_flag_raw)),
                            ("Starter obtained (live party)", str(starter_flag_set)),
                            ("Flag snapshot stable", str(state.get("flag_telemetry_valid", False))),
                        ]
                        print("\n" + _format_progress_table("PARTY RAM DIAGNOSTIC — LIVE COUNT IS AUTHORITATIVE", party_rows, width=None), flush=True)
                # Keep the last reported mismatch signature even if the mismatch clears.
                # A canonical episode reset can briefly restore matching party counts; clearing
                # this guard here would print the same static mismatch again every episode.

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

                observed_badge_flags = int(state.get("badge_flags", 0)) & 0xFF
                new_badges = 0
                if state.get("flag_telemetry_valid", False):
                    if not self._badge_baseline_verified:
                        # The first reliable sample after an unavailable baseline is
                        # a baseline, not evidence that a badge was just earned.
                        self.seen_badge_flags = observed_badge_flags
                        self._badge_baseline_verified = True
                    else:
                        new_badge_bits = observed_badge_flags & (~self.seen_badge_flags & 0xFF)
                        new_badges = int(new_badge_bits).bit_count()
                        self.seen_badge_flags |= observed_badge_flags
                component["badge"] = BADGE_REWARD * new_badges
                # Latch badge bits within the episode so a later failed read cannot
                # revoke an earned badge or award it a second time.
                state["badge_flags"] = int(self.seen_badge_flags)
                state["badge_count"] = int(self.seen_badge_flags).bit_count()
                self.prev_badges = int(self.seen_badge_flags)

                # Map/position fields are read through SaveBlock1. If its pointer
                # changed during the snapshot, do not turn stale/mixed coordinates
                # into exploration rewards or policy navigation features.
                map_telemetry_valid = bool(state.get("saveblock_pointer_stable", False))
                if not map_telemetry_valid:
                    state.setdefault("raw_map_id", int(state.get("map_id", -1)))
                    state.setdefault("raw_player_x", state.get("player_x"))
                    state.setdefault("raw_player_y", state.get("player_y"))
                    state["map_id"] = -1
                    state["player_x"] = None
                    state["player_y"] = None
                map_id = int(state["map_id"])
                new_area_discovered = False
                if map_id >= 0:
                    new_area_discovered = map_id not in self.mini_visited_maps
                    self.visited_maps.add(map_id)  # Gym-episode metric stays cumulative.
                    if new_area_discovered:
                        self.mini_visited_maps.add(map_id)
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
                    if tile_key not in self.mini_visited_tiles:
                        self.mini_visited_tiles.add(tile_key)
                        self.visited_tiles.add(tile_key)  # Gym-episode metric stays cumulative.
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
                    self.episode_seen_screen_signatures.add(current_screen_signature)
                    prior_screen_visits = self.lifetime_screen_visits[current_screen_signature]
                    potential_screen_reward = _diminishing_novelty_reward(NEW_SCREEN_REWARD, prior_screen_visits)
                    remaining_screen_budget = max(
                        0.0,
                        MAX_SCREEN_NOVELTY_REWARD_PER_EPISODE - self.mini_episode_screen_novelty_reward,
                    )
                    component["screen"] = min(potential_screen_reward, remaining_screen_budget)
                    self.mini_episode_screen_novelty_reward += component["screen"]
                    self.lifetime_screen_visits[current_screen_signature] += 1
                    new_screen_discovered = component["screen"] > 0.0

                # Story flag bonuses are unique within an episode; reward transitions, not
                # the absolute value of a cumulative story score.
                current_story_flags = set(state.get("story_flags", set())) | self.seen_story_flags
                new_story_flags = set()
                if state.get("flag_telemetry_valid", False):
                    if not self._story_flag_baseline_verified:
                        # Do not reward old flags discovered after a failed baseline read.
                        self.seen_story_flags.update(current_story_flags)
                        self._story_flag_baseline_verified = True
                    else:
                        new_story_flags = current_story_flags - self.seen_story_flags
                        self.seen_story_flags.update(current_story_flags)
                component["story"] = sum(STORY_FLAG_REWARDS.get(flag, 0.0) for flag in new_story_flags)
                current_story_flags |= self.seen_story_flags
                state["story_flags"] = current_story_flags

                party_members_added = max(0, state["party_count"] - self.prev_caught_count)
                component["party"] = PARTY_MEMBER_REWARD * party_members_added
                if party_members_added and MODE == "watch":
                    send_notification("Pokémon Party Expanded", f"Party size increased by {party_members_added}.")
                self.prev_caught_count = state["party_count"]

                # Trainer flags are a reliable, one-time proxy for trainer victories in Emerald.
                current_trainer_flags = set(state.get("trainer_flags", set())) | self.seen_trainer_flags
                new_trainer_flags = set()
                if state.get("trainer_flag_telemetry_valid", False):
                    if not self._trainer_flag_baseline_verified:
                        self.seen_trainer_flags.update(current_trainer_flags)
                        self._trainer_flag_baseline_verified = True
                    else:
                        new_trainer_flags = current_trainer_flags - self.seen_trainer_flags
                        self.seen_trainer_flags.update(current_trainer_flags)
                component["trainer"] = min(4, len(new_trainer_flags)) * TRAINER_WIN_REWARD
                current_trainer_flags |= self.seen_trainer_flags
                state["trainer_flags"] = current_trainer_flags

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
                        # Escalate progressively instead of repeatedly applying a tiny
                        # flat penalty that may be too weak to disfavor button-mashing.
                        stagnant_steps = self.episode_step_counter - self.last_novelty_step
                        severity = 1 + max(0, stagnant_steps - LOOP_STAGNATION_STEPS) // LOOP_PENALTY_INTERVAL
                        component["loop"] = -LOOP_PENALTY * min(10, severity)

                # The selected curriculum goal is evaluated from the decoded live state.
                # Reward it only on the transition into success, especially in watch mode.
                task_success = _objective_satisfied(OBJECTIVE_NAME, state)
                info["task_success"] = bool(task_success)
                if task_success and not self._task_success_rewarded:
                    component["task_success"] = TASK_SUCCESS_REWARD
                    info["objective_just_completed"] = True
                    self._task_success_rewarded = True
                    success_rows = [
                        ("Status", "SUCCESS"),
                        ("Objective", f"{OBJECTIVE_NAME} — {OBJECTIVE_LABEL}"),
                        ("Step", f"{self.episode_step_counter:,}"),
                        ("Party", f"count={state.get('party_count', 0)} | total levels={state.get('party_levels', 0)} | source={state.get('party_count_source', 'unknown')}"),
                        ("Badges", str(state.get("badge_count", 0))),
                        ("Location", f"map={state.get('map_id', -1)} | pos=({state.get('player_x', -1)},{state.get('player_y', -1)})"),
                    ]
                    # During training the same short objective episode may succeed hundreds
                    # of times. Print only the first such table per worker; evaluation keeps
                    # per-episode detail because its small episode count is diagnostically useful.
                    if MODE != "train" or not self._training_success_logged_once:
                        print("\n" + _format_progress_table("OBJECTIVE COMPLETED", success_rows, width=None), flush=True)
                        if MODE == "train":
                            self._training_success_logged_once = True
                    # The curriculum requires a fresh, checkpoint-linked milestone from its
                    # evaluation process. Re-saving a ~400 KB savestate on every training
                    # success adds unnecessary I/O and floods the logs, so training defers it.
                    if MODE != "train":
                        info["milestone_saved"] = bool(self._save_objective_milestone(state))
                    if MODE == "watch":
                        try:
                            send_notification(
                                "Pokémon AI Objective Complete",
                                f"Completed objective: {OBJECTIVE_LABEL}.",
                            )
                        except Exception as notification_exc:
                            print(
                                f"[Notification Warning] Objective succeeded, but desktop notification failed: "
                                f"{notification_exc}",
                                flush=True,
                            )
                    # Training/evaluation need terminal episodes for PPO success signals.
                    # Watch is intentionally non-episodic; the outer watch loop observes the
                    # one-shot objective_just_completed event and provides a short grace window.
                    if MODE in {"train", "evaluate"}:
                        terminated = True

                info.update({
                    "party_count": state["party_count"],
                    "party_levels": state["party_levels"],
                    "party_slot_levels": tuple(state.get("party_slot_levels", ())),
                    "party_count_saveblock1": state.get("party_count_saveblock1", -1),
                    "party_count_source": state.get("party_count_source", "unknown"),
                    "starter_flag": starter_flag_set,
                    "starter_flag_raw": bool(starter_flag_raw) if starter_flag_raw is not None else None,
                    "starter_flag_inferred_from_live_party": bool(live_party_count >= 1 and int(state.get("party_levels", 0)) >= 1),
                    "pokedex_flag": POKEDEX_OBTAINED_FLAG_ID in state.get("story_flags", set()),
                    "flag_telemetry_valid": bool(state.get("flag_telemetry_valid", False)),
                    "trainer_flag_telemetry_valid": bool(state.get("trainer_flag_telemetry_valid", False)),
                    "saveblock_pointer_stable": bool(state.get("saveblock_pointer_stable", False)),
                    "map_telemetry_valid": bool(map_telemetry_valid),
                    "raw_map_id": int(state.get("raw_map_id", map_id)),
                    "raw_player_x": state.get("raw_player_x", player_x),
                    "raw_player_y": state.get("raw_player_y", player_y),
                    "story_flag_regressions": tuple(state.get("story_flag_regressions", ())),
                    "badge_flag_regressions": int(state.get("badge_flag_regressions", 0)),
                    "trainer_flag_regressions": tuple(state.get("trainer_flag_regressions", ())),
                    "raw_flag_bytes": dict(state.get("raw_flag_bytes", {})),
                    "flag_telemetry_invalid_steps": int(self.flag_telemetry_invalid_steps),
                    "trainer_flag_telemetry_invalid_steps": int(self.trainer_flag_telemetry_invalid_steps),
                    "saveblock1_ptr": int(state.get("saveblock1_ptr", 0)),
                    "raw_badge_flags": int(state.get("raw_badge_flags", 0)),
                    "story_flags": tuple(sorted(int(flag) for flag in state.get("story_flags", set()))),
                    "trainer_flag_count": len(state.get("trainer_flags", set())),
                    "new_badges": new_badges,
                    "party_members_added": int(party_members_added),
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
                    "unique_screens_episode": len(self.episode_seen_screen_signatures),
                    "level_reward": component["level"],
                    "badge_reward": component["badge"],
                    "story_reward": component["story"],
                    "party_reward": component["party"],
                    "trainer_reward": component["trainer"],
                    "battle_exp_reward": component["battle_exp"],
                    "party_experience_delta": int(experience_delta),
                    "loop_penalty": component["loop"],
                    "task_success_reward": component["task_success"],
                })
                self._consecutive_ram_read_failures = 0
                self._memory_read_warning_printed = False
            except Exception as e:
                self._consecutive_ram_read_failures += 1
                if not self._memory_read_warning_printed:
                    print(
                        f"[State/Reward Warning] Game-state or reward update failed "
                        f"({self._consecutive_ram_read_failures}/{RAM_READ_FAILURE_LIMIT} consecutive): {e}",
                        flush=True,
                    )
                    self._memory_read_warning_printed = True
                if self._consecutive_ram_read_failures >= RAM_READ_FAILURE_LIMIT:
                    raise RuntimeError(
                        f"Game-state/reward processing failed for {self._consecutive_ram_read_failures} "
                        "consecutive steps. Stopping rather than silently training with missing "
                        "state/rewards. Check the preceding error, mGBA binding, ROM revision, "
                        "and live RAM addresses."
                    ) from e
                state = None
                component = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
                component["step"] = -STEP_PENALTY
                info["ram_read_failed"] = True
                info["ram_read_failures_consecutive"] = self._consecutive_ram_read_failures
                self._last_game_state = None
            finally:
                if timing_enabled:
                    state_elapsed = time.perf_counter() - state_started

        will_truncate_at_limit = _episode_time_limit_reached(MODE, self.episode_step_counter)
        mini_boundary_due = _mini_episode_boundary_due(
            MODE, OBJECTIVE_NAME, self.episode_step_counter, MINI_EPISODE_STEPS,
            terminated=terminated,
            truncated=will_truncate_at_limit,
            stopped=bool(self.stop_requested or STOP_REQUESTED),
            state_valid=state is not None and not bool(info.get("ram_read_failed", False)),
            task_success=bool(info.get("task_success", False)),
        )
        if mini_boundary_due:
            self.mini_episode_count += 1
            info["mini_episode_boundary"] = True
            info["mini_episode_count"] = int(self.mini_episode_count)
            info["mini_episode_steps"] = int(MINI_EPISODE_STEPS)
            info["mini_episode_unique_tiles"] = len(self.mini_visited_tiles)
            info["mini_episode_unique_maps"] = len(self.mini_visited_maps)
            info["mini_episode_unique_screens"] = len(self.seen_screen_signatures)
            # Reset only exploration memory used by the observation and loop detector.
            # Do not reset the ROM, story/trainer deduplication, full-episode metrics,
            # accumulated rewards, terminal flags, or PPO's LSTM state.
            self.mini_visited_maps.clear()
            self.mini_visited_tiles.clear()
            self.seen_screen_signatures.clear()
            self.position_history.clear()
            self.screen_signature_history.clear()
            self.mini_episode_screen_novelty_reward = 0.0
            self.last_novelty_step = self.episode_step_counter
            # Seed the new window with the current state so standing still on the
            # boundary does not immediately earn novelty twice for the same tile/map.
            if state is not None:
                boundary_map_id = int(state.get("map_id", -1))
                boundary_x = state.get("player_x")
                boundary_y = state.get("player_y")
                if boundary_map_id >= 0:
                    self.mini_visited_maps.add(boundary_map_id)
                    if boundary_x is not None and boundary_y is not None:
                        boundary_tile = (boundary_map_id, int(boundary_x), int(boundary_y))
                        self.mini_visited_tiles.add(boundary_tile)
                        self.position_history.append(boundary_tile)
                if current_screen_signature is not None:
                    self.seen_screen_signatures.add(current_screen_signature)
                    self.screen_signature_history.append(current_screen_signature)
            print(
                f"[Mini-Episode] objective={OBJECTIVE_NAME} window={self.mini_episode_count} "
                f"at_step={self.episode_step_counter:,}; game state, episode totals, and LSTM preserved.",
                flush=True,
            )

        reward = float(sum(component.values()))
        for component_name, component_value in component.items():
            self.episode_reward_components[component_name] += float(component_value)
            info[f"reward_{component_name}"] = float(component_value)
            info[f"episode_{component_name}_sum"] = float(self.episode_reward_components[component_name])
        info["episode_reward_total_so_far"] = float(sum(self.episode_reward_components.values()))

        # The next observation sees the updated visited-tile map and decoded state.
        observation_started = time.perf_counter() if timing_enabled else 0.0
        obs = self._get_obs(game_state=state, read_state=False, screen=screen_frame)
        if timing_enabled:
            observation_elapsed = time.perf_counter() - observation_started

        # Record frame if enabled
        if self.video_writer is not None and self._last_raw_frame_rgb is not None:
            frame_rgb = cv2.resize(
                self._last_raw_frame_rgb, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

        # The GUI was rendered immediately after frame acquisition above.
        # Training/evaluation have a hard episode limit. Watch mode has no step-based
        # time-limit reset, but the outer playback loop can stop after objective grace.
        truncated = _episode_time_limit_reached(MODE, self.episode_step_counter)
        if self.stop_requested or STOP_REQUESTED:
            truncated = True
            info["stopped"] = True
        if truncated:
            info["TimeLimit.truncated"] = True
        if terminated or truncated:
            outcome = "SUCCESS" if terminated and info.get("task_success") else ("TIMEOUT" if truncated and not info.get("stopped") else "STOPPED")
            info["episode_outcome"] = outcome
            info["episode_steps"] = int(self.episode_step_counter)
            info["episode_return"] = float(sum(self.episode_reward_components.values()))
            info["episode_reward_components"] = {
                name: float(value) for name, value in self.episode_reward_components.items()
            }
            reward_breakdown = ", ".join(
                f"{name}={value:+.2f}"
                for name, value in self.episode_reward_components.items()
                if abs(value) >= 0.005
            ) or "no nontrivial reward components"
            episode_rows = [
                ("Environment", str(self.env_id)),
                ("Outcome", outcome),
                ("Length / return", f"{self.episode_step_counter:,} steps | {sum(self.episode_reward_components.values()):.3f}"),
                ("Exploration", f"tiles={len(self.visited_tiles)} | maps={len(self.visited_maps)} | story flags={len(self.seen_story_flags)} | trainers={len(self.seen_trainer_flags)}"),
                ("Game state", f"map={info.get('map_id', -1)} | pos=({info.get('player_x', -1)},{info.get('player_y', -1)}) | party={info.get('party_count', '?')} | levels={info.get('party_levels', '?')} | badges={info.get('badge_count', '?')}"),
                ("Reward breakdown", reward_breakdown),
            ]
            self._terminal_episode_count += 1
            should_log_episode = (
                MODE != "train"
                or outcome != "SUCCESS"
                or self._terminal_episode_count == 1
                or self._terminal_episode_count % TERMINAL_EPISODE_EVERY == 0
            )
            if should_log_episode:
                print("\n" + _format_progress_table(
                    f"EPISODE FINISHED — ENV {self.env_id} — #{self._terminal_episode_count}",
                    episode_rows, width=None
                ), flush=True)
        if timing_enabled:
            info.update({
                "timing_env_ms": (time.perf_counter() - step_started) * 1000.0,
                "timing_frames_ms": frame_elapsed * 1000.0,
                "timing_image_ms": image_elapsed * 1000.0,
                "timing_state_ms": state_elapsed * 1000.0,
                "timing_obs_ms": observation_elapsed * 1000.0,
            })
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

    def _save_objective_milestone(self, state):
        """Save a verified objective state plus a machine-readable telemetry manifest."""
        if state.get("saveblock_pointer_stable", True) is False:
            print(
                "[Milestone Warning] Objective was reached while the SaveBlock1 pointer changed; "
                "refusing to publish an unverified milestone.",
                flush=True,
            )
            return False
        if (
            OBJECTIVE_NAME in OBJECTIVES_REQUIRING_FLAG_TELEMETRY
            and state.get("flag_telemetry_valid", True) is False
        ):
            print(
                "[Milestone Warning] Objective depends on event/badge flags, but the SaveBlock1 "
                "flag snapshot was unstable; refusing to publish an unverified milestone.",
                flush=True,
            )
            return False
        if not MILESTONE_SAVE_ENABLED:
            print("[Milestone Save] Disabled by POKEMON_AI_MILESTONE_SAVE=0.", flush=True)
            return False
        os.makedirs(MILESTONE_DIR, exist_ok=True)
        suffix = "" if int(self.env_id) == 0 else f".worker_{int(self.env_id)}"
        state_path = os.path.join(MILESTONE_DIR, f"{OBJECTIVE_NAME}{suffix}.state")
        if not self._save_state_file(state_path):
            print(
                f"[Milestone Warning] Objective succeeded, but its savestate could not be saved: {state_path}",
                flush=True,
            )
            return False
        story_flags = state.get("story_flags", set())
        trainer_flags = state.get("trainer_flags", set())
        evaluation_checkpoint = {}
        if MODE == "evaluate":
            configured_checkpoint = os.environ.get("POKEMON_AI_EVAL_CHECKPOINT", "").strip()
            if configured_checkpoint:
                checkpoint_path = os.path.expanduser(configured_checkpoint)
                if not os.path.isabs(checkpoint_path):
                    checkpoint_path = os.path.join(BASE_DIR, checkpoint_path)
                if not checkpoint_path.lower().endswith(".zip"):
                    checkpoint_path += ".zip"
            else:
                checkpoint_path = f"{MODEL_CHECKPOINT}.zip"
            checkpoint_path = os.path.abspath(checkpoint_path)
            if os.path.isfile(checkpoint_path):
                try:
                    evaluation_checkpoint = {
                        "evaluation_checkpoint_path": checkpoint_path,
                        "evaluation_checkpoint_sha1": _sha1_file(checkpoint_path),
                    }
                except OSError as exc:
                    print(f"[Milestone Warning] Could not fingerprint evaluation checkpoint: {exc}", flush=True)
        metadata = {
            "schema_version": 1,
            "objective": OBJECTIVE_NAME,
            "objective_label": OBJECTIVE_LABEL,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "created_unix": time.time(),
            "rom_sha1_expected": EXPECTED_ROM_SHA1,
            **evaluation_checkpoint,
            "savestate_path": os.path.abspath(state_path),
            "env_id": int(self.env_id),
            "episode_step": int(self.episode_step_counter),
            "state": {
                "party_count": int(state.get("party_count", 0)),
                "party_levels": int(state.get("party_levels", 0)),
                "party_slot_levels": [int(x) for x in state.get("party_slot_levels", ())],
                "party_count_source": str(state.get("party_count_source", "unknown")),
                "party_count_saveblock1": int(state.get("party_count_saveblock1", -1)),
                "map_id": int(state.get("map_id", -1)),
                "map_group": int(state.get("map_group", -1)),
                "map_number": int(state.get("map_number", -1)),
                "player_x": state.get("player_x"),
                "player_y": state.get("player_y"),
                "badge_flags": int(state.get("badge_flags", 0)),
                "badge_count": int(state.get("badge_count", 0)),
                "story_flags": sorted(int(x) for x in story_flags),
                "trainer_flag_count": len(trainer_flags),
                "flag_telemetry_valid": bool(state.get("flag_telemetry_valid", True)),
                "trainer_flag_telemetry_valid": bool(state.get("trainer_flag_telemetry_valid", True)),
                "saveblock_pointer_stable": bool(state.get("saveblock_pointer_stable", True)),
                "starter_flag_raw": state.get("starter_flag_raw"),
                "starter_obtained": bool(state.get("starter_obtained", False)),
                "story_flag_regressions": [int(x) for x in state.get("story_flag_regressions", ())],
                "badge_flag_regressions": int(state.get("badge_flag_regressions", 0)),
                "trainer_flag_regressions": [int(x) for x in state.get("trainer_flag_regressions", ())],
                "saveblock1_ptr": int(state.get("saveblock1_ptr", 0)),
                "raw_flag_bytes": dict(state.get("raw_flag_bytes", {})),
            },
        }
        try:
            _atomic_write_json(state_path + ".json", metadata)
        except Exception as exc:
            print(
                f"[Milestone Warning] Savestate exists, but metadata could not be published: {exc}",
                flush=True,
            )
            return False
        print(
            f"[Milestone Save] objective={OBJECTIVE_NAME} saved={state_path} "
            f"metadata={state_path}.json | party={metadata['state']['party_count']} "
            f"levels={metadata['state']['party_levels']} badges={metadata['state']['badge_count']} "
            f"map={metadata['state']['map_id']} pos=({metadata['state']['player_x']}"
            f",{metadata['state']['player_y']})",
            flush=True,
        )
        return True

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

        if os.path.isfile(final_path) and LATEST_CHECKPOINT_BACKUP_ENABLED:
            # Same-filesystem hard links make a previous-checkpoint backup effectively metadata-only.
            # Fall back to a copy on filesystems that do not support hard links.
            backup_path = final_path + ".bak"
            backup_temp = f"{backup_path}.tmp-{os.getpid()}-{time.time_ns()}"
            try:
                try:
                    os.link(final_path, backup_temp)
                except OSError:
                    shutil.copy2(final_path, backup_temp)
                if not os.path.isfile(backup_temp) or os.path.getsize(backup_temp) != os.path.getsize(final_path):
                    raise IOError("Previous-checkpoint backup failed size validation")
                os.replace(backup_temp, backup_path)
                _fsync_parent_dir(backup_path)
            finally:
                try:
                    if os.path.exists(backup_temp):
                        os.remove(backup_temp)
                except OSError:
                    pass
        os.replace(temp_path, final_path)
        _fsync_parent_dir(final_path)
        global LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP
        try:
            LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP = int(getattr(model, "num_timesteps", -1))
        except (TypeError, ValueError, OverflowError):
            LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP = None
        return final_path
    except BaseException:
        # Keep the previous final checkpoint untouched, even on an interrupted write. Remove only our known temp path.
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise


def archive_model_checkpoint(saved_model_path, timestep):
    """Keep a bounded rolling archive of verified immutable model checkpoints.

    The latest checkpoint is saved first by atomic_save_model(). Archive snapshots are hard
    links where supported (otherwise file copies), then atomically published. A retention
    value of zero disables rolling snapshots entirely.
    """
    if CHECKPOINT_ARCHIVE_KEEP == 0:
        return None
    if not os.path.isfile(saved_model_path) or not zipfile.is_zipfile(saved_model_path):
        raise FileNotFoundError(f"Cannot archive a missing/invalid checkpoint: {saved_model_path}")

    os.makedirs(CHECKPOINT_ARCHIVE_DIR, exist_ok=True)
    destination = os.path.join(
        CHECKPOINT_ARCHIVE_DIR,
        f"{_MODEL_NAME}_step{max(0, int(timestep)):012d}.zip",
    )
    if os.path.exists(destination):
        try:
            if filecmp.cmp(saved_model_path, destination, shallow=False):
                return destination
        except OSError:
            pass
        destination = os.path.join(
            CHECKPOINT_ARCHIVE_DIR,
            f"{_MODEL_NAME}_step{max(0, int(timestep)):012d}_{time.time_ns()}.zip",
        )

    temporary = destination + f".tmp-{os.getpid()}-{time.time_ns()}"
    try:
        # Checkpoint archives are immutable snapshots. A hard link preserves the exact old
        # inode when the current checkpoint is later atomically replaced, avoiding a full
        # model-sized copy on normal same-filesystem Linux installations.
        try:
            os.link(saved_model_path, temporary)
        except OSError:
            shutil.copy2(saved_model_path, temporary)
        if (
            not os.path.isfile(temporary)
            or os.path.getsize(temporary) != os.path.getsize(saved_model_path)
            or not zipfile.is_zipfile(temporary)
        ):
            raise IOError("Checkpoint archive copy failed size or ZIP validation")
        with open(temporary, "rb") as archived:
            os.fsync(archived.fileno())
        os.replace(temporary, destination)
        _fsync_parent_dir(destination)
    except Exception:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        raise

    snapshots = sorted(
        glob.glob(os.path.join(CHECKPOINT_ARCHIVE_DIR, f"{_MODEL_NAME}_step*.zip")),
        key=lambda path: (os.stat(path).st_mtime_ns, path),
    )
    expired = snapshots[:-CHECKPOINT_ARCHIVE_KEEP]
    for old_snapshot in expired:
        try:
            if os.path.abspath(old_snapshot) != os.path.abspath(destination):
                os.remove(old_snapshot)
        except OSError as exc:
            print(f"[Checkpoint Archive Warning] Could not prune {old_snapshot}: {exc}", flush=True)
    return destination


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
    actual_port = TENSORBOARD_PORT
    if _tcp_port_is_open("127.0.0.1", actual_port):
        occupied_port = actual_port
        for candidate in range(TENSORBOARD_PORT + 1, min(TENSORBOARD_PORT + 20, 65536)):
            if not _tcp_port_is_open("127.0.0.1", candidate):
                actual_port = candidate
                break
        else:
            print(
                f"[TensorBoard Warning] Ports {TENSORBOARD_PORT}-{min(TENSORBOARD_PORT + 19, 65535)} "
                "are occupied; no new server was started.", flush=True,
            )
            return
        print(
            f"[TensorBoard] Port {occupied_port} is already occupied; starting a separate server "
            f"on {actual_port} for the current log directory.", flush=True,
        )

    try:
        # TensorBoard's verbose server output is not useful for routine runs and can
        # grow a second log indefinitely. Keep its chatter off disk; startup failures
        # are still detected below and reported by this process.
        tb_log_handle = None
        tb_process = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard.main",
                f"--logdir={TENSORBOARD_LOG_DIR}",
                f"--port={actual_port}",
                "--host=127.0.0.1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tb_owned = True
        time.sleep(0.5)
        if tb_process.poll() is not None:
            print(
                "[TensorBoard Warning] Server exited during startup; check TensorBoard installation, log directory, and port.",
                flush=True,
            )
        else:
            print(f"[TensorBoard] Starting at http://127.0.0.1:{actual_port} | logdir={TENSORBOARD_LOG_DIR}", flush=True)
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
    """Request a safe stop; never unwind the active rollout/update from a signal handler."""
    global STOP_REQUESTED, STOP_SIGNAL_NAME
    STOP_REQUESTED = True
    try:
        STOP_SIGNAL_NAME = signal.Signals(sig).name
    except (ValueError, TypeError):
        STOP_SIGNAL_NAME = str(sig)
    # The training callback observes this flag, finishes the current full PPO rollout,
    # waits for its optimizer update, and only then saves. Printing/I/O is deferred to
    # normal callback code rather than performed inside the signal handler.


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def _actor_state_numpy(recurrent_state):
    """Extract the actor hidden/cell state from sb3-contrib RNNStates or predict state."""
    try:
        if hasattr(recurrent_state, "pi"):
            values = recurrent_state.pi
        elif isinstance(recurrent_state, (tuple, list)) and len(recurrent_state) >= 2:
            values = recurrent_state[:2]
        else:
            return None
        converted = []
        for value in values[:2]:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            converted.append(np.asarray(value, dtype=np.float32))
        return tuple(converted) if len(converted) == 2 else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


class StopAndCheckpointCallback(BaseCallback):
    """Report actionable progress, log metrics to TensorBoard, and checkpoint safely."""

    def __init__(self, save_frequency=CHECKPOINT_SAVE_FREQUENCY, verbose=0, runtime_profiler=None):
        super().__init__(verbose)
        self.runtime_profiler = runtime_profiler
        self.save_frequency = max(1, int(save_frequency))
        self.progress_frequency = max(1, int(PROGRESS_LOG_FREQUENCY))
        self.terminal_progress_frequency = max(
            self.progress_frequency, TERMINAL_PROGRESS_FREQUENCY
        )
        self._last_save_timestep = 0
        self._checkpoint_due = False
        self._stop_notice_printed = False
        self._last_progress_timestep = 0
        self._last_terminal_progress_timestep = 0
        self._run_start_timestep = 0
        self._run_started_at = 0.0
        self._recent_rewards = deque(maxlen=1000)
        self._recent_actions = deque(maxlen=4096)  # actions actually sampled by PPO
        self._recent_greedy_actions = deque(maxlen=512)  # argmax actions on sampled training observations
        self._last_greedy_probe_timestep = 0
        # Greedy diagnostics run an extra CNN/LSTM inference. One probe per rollout is
        # enough for collapse monitoring and avoids repeating inference every ~625 steps.
        self._greedy_probe_interval = max(0, int(os.environ.get("POKEMON_AI_GREEDY_PROBE_EVERY", "8192")))
        self._greedy_probe_failures = 0
        self._last_collapse_warning_timestep = -1
        self._last_step_reward = 0.0
        self._last_action = None
        self._peak_metrics_path = os.path.join(RUNTIME_DATA_DIR, "training_metric_peaks.json")
        self._peak_metrics = {"party_count": 0.0, "party_total_levels": 0.0, "badge_count": 0.0,
                              "starter_seen": 0.0, "pokedex_seen": 0.0}
        self._peak_metrics_dirty = False
        self._completed_episodes = 0
        self._last_episode_reward = None
        self._last_episode_length = None
        self._episode_running_rewards = np.zeros(1, dtype=np.float64)
        self._recent_episode_success = deque(maxlen=100)
        self._recent_episode_returns = deque(maxlen=100)
        self._recent_episode_lengths = deque(maxlen=100)
        self._recent_episode_component_sums = {
            name: deque(maxlen=100) for name in REWARD_COMPONENT_NAMES
        }
        self._task_successes = 0
        self._timeouts = 0
        self._timing_sums = Counter()
        self._timing_samples = 0
        self._timing_reset_sum_ms = 0.0
        self._timing_reset_samples = 0

    def _on_training_start(self):
        global ACTIVE_TRAINING_CALLBACK, SAFE_STOP_AFTER_UPDATE
        global LAST_COMPLETED_UPDATE_TIMESTEP, LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP, PPO_UPDATE_FAILED
        # On resumed training, num_timesteps can already be very large. Anchor the
        # interval to this run's starting value so we do not save again on its first step.
        ACTIVE_TRAINING_CALLBACK = self
        SAFE_STOP_AFTER_UPDATE = False
        LAST_COMPLETED_UPDATE_TIMESTEP = int(self.num_timesteps)
        PPO_UPDATE_FAILED = False
        if (
            LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP is None
            and os.path.isfile(f"{MODEL_CHECKPOINT}.zip")
        ):
            # A resumed model's existing file is the known-good recovery baseline.
            LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP = int(self.num_timesteps)
        self._checkpoint_due = False
        self._stop_notice_printed = False
        self._run_start_timestep = int(self.num_timesteps)
        self._last_save_timestep = int(self.num_timesteps)
        self._last_progress_timestep = int(self.num_timesteps)
        self._last_terminal_progress_timestep = int(self.num_timesteps)
        self._run_started_at = time.perf_counter()
        self._recent_rewards.clear()
        self._recent_actions.clear()
        self._recent_greedy_actions.clear()
        self._last_greedy_probe_timestep = int(self.num_timesteps)
        self._last_collapse_warning_timestep = -1
        try:
            with open(self._peak_metrics_path, "r", encoding="utf-8") as stream:
                persisted = json.load(stream)
            if isinstance(persisted, dict) and persisted.get("schema_version") == 1:
                for key in self._peak_metrics:
                    value = _finite_number(persisted.get(key))
                    if value is not None and value >= 0:
                        self._peak_metrics[key] = value
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        self._recent_episode_success.clear()
        self._recent_episode_returns.clear()
        self._recent_episode_lengths.clear()
        for values in self._recent_episode_component_sums.values():
            values.clear()
        env_count = int(getattr(self.training_env, "num_envs", NUM_CPU))
        self._episode_running_rewards = np.zeros(max(1, env_count), dtype=np.float64)
        print(
            f"[Training] Started | mode={'GUI' if GUI_MODE else 'HEADLESS'} | "
            f"objective={OBJECTIVE_LABEL} | envs={env_count} | "
            f"action_repeat={ACTION_REPEAT}+{ACTION_RELEASE_FRAMES} release frames | "
            f"SB3 verbosity={VERBOSE_LEVEL} | mGBA log mask={_MGBA_LOG_LEVEL_MASK} | "
            f"checkpoint every {self.save_frequency:,} transitions | "
            f"TensorBoard flush every {self.progress_frequency:,} transitions | "
            f"terminal table every {self.terminal_progress_frequency:,} transitions.",
            flush=True,
        )

    def _on_rollout_end(self):
        """Anneal entropy, restoring exploration pressure when sampled actions collapse."""
        elapsed = max(0, int(self.num_timesteps) - self._run_start_timestep)
        progress = min(1.0, elapsed / float(max(1, TOTAL_TIMESTEPS)))
        scheduled = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress
        counts = Counter(self._recent_actions)
        count_total = sum(counts.values())
        dominance = max(counts.values()) / count_total if count_total else 0.0
        entropy = -sum((count / count_total) * math.log(max(count / count_total, 1e-12))
                       for count in counts.values()) if count_total else 0.0
        greedy_counts = Counter(self._recent_greedy_actions)
        greedy_total = sum(greedy_counts.values())
        greedy_dominance = max(greedy_counts.values()) / greedy_total if greedy_total else 0.0
        greedy_entropy = -sum((count / greedy_total) * math.log(max(count / greedy_total, 1e-12))
                              for count in greedy_counts.values()) if greedy_total else 0.0
        self.logger.record("policy/action_entropy_4k", float(entropy))
        self.logger.record("policy/action_dominance_4k", float(dominance))
        self.logger.record("policy/action_unique_4k", float(len(counts)))
        self.logger.record("policy/greedy_entropy_512", float(greedy_entropy))
        self.logger.record("policy/greedy_dominance_512", float(greedy_dominance))
        self.logger.record("policy/greedy_unique_512", float(len(greedy_counts)))
        self.model.ent_coef = scheduled
        sampled_collapse = count_total >= 512 and dominance >= ACTION_COLLAPSE_DOMINANCE
        greedy_collapse = greedy_total >= 128 and greedy_dominance >= ACTION_COLLAPSE_DOMINANCE
        if sampled_collapse or greedy_collapse:
            self.model.ent_coef = max(scheduled, ENTROPY_COEF_COLLAPSE_FLOOR)
            if int(self.num_timesteps) - self._last_collapse_warning_timestep >= max(25000, self.progress_frequency * 10):
                reason = "argmax/greedy policy" if greedy_collapse else "sampled actions"
                message_rows = [
                    ("Detection", f"{reason} action dominance exceeds {ACTION_COLLAPSE_DOMINANCE:.0%}"),
                    ("Sampled actions", _format_action_distribution(counts)),
                    ("Greedy actions", _format_action_distribution(greedy_counts)),
                    ("Exploration", f"entropy coefficient raised to {self.model.ent_coef:.4f}; next PPO update will use this value"),
                ]
                print("\n" + _format_progress_table("POLICY COLLAPSE GUARD", message_rows, width=None), flush=True)
                self._last_collapse_warning_timestep = int(self.num_timesteps)
        self.logger.record("progress/entropy_coefficient", float(self.model.ent_coef))

    def _probe_greedy_actions(self):
        """Sample deterministic argmax choices on real rollout observations without changing rollout state."""
        timestep = int(self.num_timesteps)
        if self._greedy_probe_interval <= 0:
            return
        if timestep - self._last_greedy_probe_timestep < self._greedy_probe_interval:
            return
        observations = self.locals.get("new_obs")
        if observations is None:
            return
        actor_state = _actor_state_numpy(self.locals.get("lstm_states"))
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        try:
            actions, _ = self.model.predict(
                observations,
                state=actor_state,
                episode_start=dones if dones.size else None,
                deterministic=True,
            )
            values = np.asarray(actions).reshape(-1)
            self._recent_greedy_actions.extend(
                int(value) for value in values if int(value) in ACTION_LABELS
            )
        except Exception as exc:
            self._greedy_probe_failures += 1
            if self._greedy_probe_failures in (1, 10, 100):
                print(
                    f"[Policy Probe Warning] Could not sample greedy actions "
                    f"({self._greedy_probe_failures} failures): {type(exc).__name__}: {exc}",
                    flush=True,
                )
        finally:
            self._last_greedy_probe_timestep = timestep

    def _on_step(self):
        global STOP_REQUESTED
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        self._probe_greedy_actions()
        if rewards.size:
            self._last_step_reward = float(np.mean(rewards))
            self._recent_rewards.extend(float(reward) for reward in rewards)
            if self._episode_running_rewards.size != rewards.size:
                resized = np.zeros(rewards.size, dtype=np.float64)
                keep = min(resized.size, self._episode_running_rewards.size)
                if keep:
                    resized[:keep] = self._episode_running_rewards[:keep]
                self._episode_running_rewards = resized
            self._episode_running_rewards[:rewards.size] += rewards

        actions = self.locals.get("actions")
        if actions is not None:
            try:
                action_values = np.asarray(actions).reshape(-1)
                if action_values.size:
                    self._last_action = ACTION_LABELS.get(int(action_values[-1]), str(action_values[-1]))
                    self._recent_actions.extend(
                        int(value) for value in action_values if int(value) in ACTION_LABELS
                    )
            except (TypeError, ValueError, OverflowError):
                pass

        infos = self.locals.get("infos", [])
        infos_list = infos if isinstance(infos, (list, tuple)) else []
        if TIMING_DIAGNOSTICS:
            for info in infos_list:
                if not isinstance(info, dict):
                    continue
                for timing_key in (
                    "timing_env_ms", "timing_frames_ms", "timing_image_ms",
                    "timing_state_ms", "timing_obs_ms",
                ):
                    timing_value = _finite_number(info.get(timing_key))
                    if timing_value is not None and timing_value >= 0:
                        self._timing_sums[timing_key] += timing_value
                if "timing_env_ms" in info:
                    self._timing_samples += 1
                reset_ms = _finite_number(info.get("timing_reset_ms"))
                if reset_ms is not None and reset_ms > 0:
                    self._timing_reset_sum_ms += reset_ms
                    self._timing_reset_samples += 1

        # Treat episode aggregates as terminal-only data. Per-step component sums are
        # cumulative within an episode and must never be sampled as if they were final.
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        just_completed_component_sums = {name: [] for name in REWARD_COMPONENT_NAMES}
        summaries = _completed_episode_summaries(
            infos_list, dones, running_returns=self._episode_running_rewards
        )
        for summary in summaries:
            self._completed_episodes += 1
            episode_return = float(summary["episode_return"])
            episode_length = int(summary["episode_length"])
            success = bool(summary["success"])
            outcome = str(summary["outcome"])
            self._last_episode_reward = episode_return
            self._last_episode_length = episode_length
            self._recent_episode_returns.append(episode_return)
            self._recent_episode_lengths.append(episode_length)
            self._recent_episode_success.append(1 if success else 0)
            if success:
                self._task_successes += 1
            elif outcome == "TIMEOUT":
                self._timeouts += 1

            for component_name, value in summary["component_sums"].items():
                self._recent_episode_component_sums[component_name].append(value)
                just_completed_component_sums[component_name].append(value)

        # Monitor/VecEnv marks a finished episode in dones. Reset its live return
        # after including that transition's reward, so the display tracks each new episode.
        for env_index, done in enumerate(dones[:self._episode_running_rewards.size]):
            if done:
                self._episode_running_rewards[env_index] = 0.0

        # Short, unique TensorBoard tags are intentional: SB3's default console
        # formatter truncates long keys, which used to collide for the episode-reward
        # component metrics and crash training. State metrics are emitted per worker;
        # summary metrics are explicitly mean/max, never silent averages of locations.
        self.logger.record("progress/step_reward", self._last_step_reward)
        self.logger.record("progress/episodes_done", self._completed_episodes)
        if self._recent_rewards:
            self.logger.record("progress/recent_reward_mean", float(np.mean(self._recent_rewards)))
        if self._last_episode_reward is not None:
            self.logger.record("progress/last_ep_return", self._last_episode_reward)
        if self._last_episode_length is not None:
            self.logger.record("progress/last_ep_length", self._last_episode_length)

        live_episode_reward_mean = (
            float(np.mean(self._episode_running_rewards))
            if self._episode_running_rewards.size else 0.0
        )
        self.logger.record("progress/live_return_mean", live_episode_reward_mean)
        for env_index, running_reward in enumerate(self._episode_running_rewards):
            self.logger.record(f"progress/live_return_env_{env_index}", float(running_reward))

        if self._recent_episode_returns:
            self.logger.record("progress/ep_return_mean_100", float(np.mean(self._recent_episode_returns)))
            self.logger.record("progress/ep_return_std_100", float(np.std(self._recent_episode_returns)))
            self.logger.record("progress/ep_length_mean_100", float(np.mean(self._recent_episode_lengths)))
        component_short_names = {
            "step": "step", "tile": "tile", "map": "map", "screen": "screen",
            "story": "story", "party": "party", "level": "level", "badge": "badge",
            "trainer": "trainer", "battle_exp": "bexp", "loop": "loop", "task_success": "task",
        }
        for component_name, values in self._recent_episode_component_sums.items():
            short_name = component_short_names.get(component_name, component_name[:8])
            if values:
                self.logger.record(f"progress/ep_{short_name}_mean100", float(np.mean(values)))
            if just_completed_component_sums[component_name]:
                self.logger.record(
                    f"progress/ep_{short_name}_lastdone",
                    float(np.mean(just_completed_component_sums[component_name])),
                )
        if self._recent_episode_success:
            self.logger.record("progress/success_rate_100", float(np.mean(self._recent_episode_success)))
        self.logger.record("progress/successes_run", float(self._task_successes))
        self.logger.record("progress/timeouts_run", float(self._timeouts))

        elapsed = max(1e-6, time.perf_counter() - self._run_started_at)
        steps_per_second = (int(self.num_timesteps) - self._run_start_timestep) / elapsed
        self.logger.record("progress/transitions_per_sec", float(steps_per_second))

        # Reward values are mean per environment on the current vector step. Event
        # counts are summed across workers. Cumulative per-episode counters get both
        # a descriptive cross-worker mean and exact per-worker series.
        reward_metric_map = (
            ("step_reward", "progress/step_cost"),
            ("level_reward", "progress/level_reward"),
            ("badge_reward", "progress/badge_reward"),
            ("exploration_reward", "progress/exploration_reward"),
            ("map_reward", "progress/map_reward"),
            ("tile_reward", "progress/tile_reward"),
            ("screen_reward", "progress/screen_reward"),
            ("story_reward", "progress/story_reward"),
            ("party_reward", "progress/party_reward"),
            ("trainer_reward", "progress/trainer_reward"),
            ("battle_exp_reward", "progress/battle_exp_reward"),
            ("loop_penalty", "progress/loop_penalty"),
            ("task_success_reward", "progress/task_success_reward"),
        )
        for info_key, tb_key in reward_metric_map:
            pairs = _numeric_info_pairs(infos_list, info_key)
            if pairs:
                self.logger.record(tb_key, float(np.mean([value for _, value in pairs])))

        event_metric_map = (
            ("new_badges", "progress/new_badges"),
            ("party_members_added", "progress/party_added"),
            ("new_tile_discovered", "progress/new_tiles"),
            ("new_area_discovered", "progress/new_areas"),
            ("new_story_flags", "progress/new_story_flags"),
            ("new_trainer_flags", "progress/new_trainer_flags"),
            ("new_screen_discovered", "progress/new_screens"),
            ("mini_episode_boundary", "progress/mini_ep_boundaries"),
            ("task_success", "progress/task_success_events"),
            ("ram_read_failed", "progress/ram_read_failed_envs"),
            ("starter_flag", "progress/starter_flag_workers"),
            ("starter_flag_raw", "progress/starter_flag_raw_workers"),
            ("flag_telemetry_valid", "progress/flag_telemetry_valid_workers"),
            ("trainer_flag_telemetry_valid", "progress/trainer_flag_telemetry_valid_workers"),
            ("pokedex_flag", "progress/pokedex_flag_workers"),
        )
        for info_key, tb_key in event_metric_map:
            pairs = _numeric_info_pairs(infos_list, info_key)
            if pairs:
                self.logger.record(tb_key, float(sum(value for _, value in pairs)))

        # High-water marks survive per-episode autoresets and curriculum child processes.
        # Current party values can be zero immediately after a successful episode resets;
        # *_peak charts preserve the best observed progress across those resets.
        for info_key, peak_key in (("party_count", "party_count"),
                                   ("party_levels", "party_total_levels"),
                                   ("badge_count", "badge_count")):
            values = [value for _, value in _numeric_info_pairs(infos_list, info_key)]
            if values and max(values) > self._peak_metrics[peak_key]:
                self._peak_metrics[peak_key] = max(values)
                self._peak_metrics_dirty = True
        if any(value > 0 for _, value in _numeric_info_pairs(infos_list, "party_count")) or any(
            value > 0 for _, value in _numeric_info_pairs(infos_list, "starter_flag")
        ):
            if self._peak_metrics["starter_seen"] < 1.0:
                self._peak_metrics["starter_seen"] = 1.0
                self._peak_metrics_dirty = True
        if any(value > 0 for _, value in _numeric_info_pairs(infos_list, "pokedex_flag")):
            if self._peak_metrics["pokedex_seen"] < 1.0:
                self._peak_metrics["pokedex_seen"] = 1.0
                self._peak_metrics_dirty = True
        self.logger.record("progress/party_count_peak", float(self._peak_metrics["party_count"]))
        self.logger.record("progress/party_levels_peak", float(self._peak_metrics["party_total_levels"]))
        self.logger.record("progress/badges_peak", float(self._peak_metrics["badge_count"]))
        self.logger.record("progress/starter_seen_run", float(self._peak_metrics["starter_seen"]))
        self.logger.record("progress/pokedex_seen_run", float(self._peak_metrics["pokedex_seen"]))

        # State values must not be averaged silently across different emulators.
        # Keep one exact line per worker plus clear max/mean summaries for party/badges.
        summary_state_metrics = (
            ("party_count", "party_count"),
            ("party_levels", "party_total_levels"),
            ("badge_count", "badge_count"),
        )
        for info_key, tag in summary_state_metrics:
            pairs = _numeric_info_pairs(infos_list, info_key)
            if not pairs:
                continue
            if info_key == "party_count":
                self.logger.record("progress/party_telemetry_envs", float(len(pairs)))
            values = [value for _, value in pairs]
            # The unsuffixed scalar is the maximum currently observed in any worker,
            # making the headline chart reflect progress when one of four workers succeeds.
            self.logger.record(f"progress/{tag}", float(max(values)))
            self.logger.record(f"progress/{tag}_mean", float(np.mean(values)))
            self.logger.record(f"progress/{tag}_max", float(max(values)))
            for env_index, value in pairs:
                self.logger.record(f"progress/{tag}_env_{env_index}", value)

        cumulative_metrics = (
            ("unique_tiles_episode", "tiles_seen"),
            ("unique_maps_episode", "maps_seen"),
            ("story_flags_episode", "story_flags_seen"),
            ("trainer_flags_episode", "trainers_done"),
            ("unique_screens_episode", "screens_seen"),
            ("mini_episode_count", "mini_ep_count"),
            ("trainer_flag_count", "trainer_flag_count"),
            ("episode_step", "episode_step"),
        )
        for info_key, tag in cumulative_metrics:
            pairs = _numeric_info_pairs(infos_list, info_key)
            if pairs:
                self.logger.record(f"progress/{tag}_mean", float(np.mean([value for _, value in pairs])))
                for env_index, value in pairs:
                    self.logger.record(f"progress/{tag}_env_{env_index}", value)

        # Map IDs and tile coordinates are categorical, so only per-worker series are
        # emitted. Averaging map IDs or coordinates creates locations that do not exist.
        for info_key in (
            "map_id", "map_group", "map_number", "player_x", "player_y",
            "party_count_saveblock1", "starter_flag", "starter_flag_raw", "pokedex_flag",
        ):
            tb_name = "saveblock_party_count" if info_key == "party_count_saveblock1" else info_key
            for env_index, value in _numeric_info_pairs(infos_list, info_key):
                self.logger.record(f"progress/{tb_name}_env_{env_index}", value)

        # Compare live gPlayerPartyCount (authoritative) with SaveBlock1's copy. A
        # mismatch is diagnostic only and must not replace the live party metric.
        live_party = dict(_numeric_info_pairs(infos_list, "party_count"))
        saved_party = dict(_numeric_info_pairs(infos_list, "party_count_saveblock1"))
        comparable_party_envs = sorted(set(live_party).intersection(saved_party))
        if comparable_party_envs:
            mismatch_count = sum(
                1 for env_index in comparable_party_envs
                if live_party[env_index] != saved_party[env_index]
            )
            self.logger.record("progress/party_ram_mismatch", float(mismatch_count))

        if int(self.num_timesteps) - self._last_progress_timestep >= self.progress_frequency:
            if self._peak_metrics_dirty:
                try:
                    _atomic_write_json(self._peak_metrics_path, {
                        "schema_version": 1, **self._peak_metrics,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })
                    self._peak_metrics_dirty = False
                except Exception as peak_exc:
                    print(f"[Metrics Warning] Could not persist metric high-water marks: {peak_exc}", flush=True)
            action_counts = Counter(self._recent_actions)
            action_total = sum(action_counts.values())
            action_probs = [count / action_total for count in action_counts.values()] if action_total else []
            sampled_entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in action_probs) if action_probs else 0.0
            sampled_entropy_norm = sampled_entropy / math.log(len(ACTION_LABELS)) if action_total else 0.0
            sampled_dominance = max(action_probs) if action_probs else 0.0
            greedy_counts = Counter(self._recent_greedy_actions)
            greedy_total = sum(greedy_counts.values())
            greedy_probs = [count / greedy_total for count in greedy_counts.values()] if greedy_total else []
            greedy_entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in greedy_probs) if greedy_probs else 0.0
            greedy_dominance = max(greedy_probs) if greedy_probs else 0.0
            self.logger.record("policy/action_entropy_4k", float(sampled_entropy))
            self.logger.record("policy/action_entropy_norm", float(sampled_entropy_norm))
            self.logger.record("policy/action_dominance_4k", float(sampled_dominance))
            self.logger.record("policy/action_unique_4k", float(len(action_counts)))
            self.logger.record("policy/greedy_entropy_512", float(greedy_entropy))
            self.logger.record("policy/greedy_dominance_512", float(greedy_dominance))
            self.logger.record("policy/greedy_unique_512", float(len(greedy_counts)))
            for action_id, action_name in ACTION_LABELS.items():
                self.logger.record(f"policy/act_{action_name.lower()}_pct", float(action_counts.get(action_id, 0) / action_total) if action_total else 0.0)
                self.logger.record(f"policy/greedy_{action_name.lower()}_pct", float(greedy_counts.get(action_id, 0) / greedy_total) if greedy_total else 0.0)
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
                    worker_status = f"{alive_workers}/{len(processes)} worker processes alive"
                except Exception:
                    worker_status = f"{env_count} environments; worker status unavailable"
            else:
                worker_status = f"{env_count} environment(s), in-process"

            per_env_speed = steps_per_second / max(1, env_count)
            aggregate_gba_fps = steps_per_second * max(1, ACTION_REPEAT + ACTION_RELEASE_FRAMES)
            success_rate_text = (
                f"{float(np.mean(self._recent_episode_success)):.1%} over last {len(self._recent_episode_success)} finished episodes"
                if self._recent_episode_success else "n/a (no finished episodes yet)"
            )
            infos_for_state = [info if isinstance(info, dict) else {} for info in infos_list]
            party_values = [value for _, value in _numeric_info_pairs(infos_for_state, "party_count")]
            level_values = [value for _, value in _numeric_info_pairs(infos_for_state, "party_levels")]
            badge_values = [value for _, value in _numeric_info_pairs(infos_for_state, "badge_count")]
            live_party_pairs = dict(_numeric_info_pairs(infos_for_state, "party_count"))
            saveblock_party_pairs = dict(_numeric_info_pairs(infos_for_state, "party_count_saveblock1"))
            comparable_party_envs = sorted(set(live_party_pairs).intersection(saveblock_party_pairs))
            mismatch_count = sum(
                1 for env_index in comparable_party_envs
                if live_party_pairs[env_index] != saveblock_party_pairs[env_index]
            )
            party_summary = (
                f"max={int(max(party_values))}, mean={np.mean(party_values):.2f}/env"
                if party_values else "telemetry unavailable"
            )
            level_summary = str(int(max(level_values))) if level_values else "n/a"
            badge_summary = str(int(max(badge_values))) if badge_values else "n/a"
            worker_state_parts = []
            for env_index, info in enumerate(infos_for_state):
                if not info:
                    worker_state_parts.append(f"E{env_index}: telemetry=n/a")
                    continue
                party = _finite_number(info.get("party_count"))
                levels = _finite_number(info.get("party_levels"))
                badges = _finite_number(info.get("badge_count"))
                map_id = _finite_number(info.get("map_id"))
                x = _finite_number(info.get("player_x"))
                y = _finite_number(info.get("player_y"))
                if None in (party, levels, badges, map_id, x, y):
                    worker_state_parts.append(f"E{env_index}: state partly unavailable")
                else:
                    worker_state_parts.append(
                        f"E{env_index}: P{int(party)} Lv{int(levels)} B{int(badges)} "
                        f"M{int(map_id)}@({int(x)},{int(y)})"
                    )
            worker_state_text = " | ".join(worker_state_parts) or "n/a"

            action_text = self._last_action or "unknown"
            episode_summary = (
                f"completed={self._completed_episodes} | successes={self._task_successes} | "
                f"timeouts={self._timeouts} | success rate={success_rate_text}"
            )
            last_episode_text = (
                f"return={self._last_episode_reward:.2f}, length={self._last_episode_length:,}"
                if self._last_episode_reward is not None and self._last_episode_length is not None
                else "none completed yet"
            )
            recent_reward = float(np.mean(self._recent_rewards)) if self._recent_rewards else 0.0
            live_values = ", ".join(
                f"E{i}={value:.2f}" for i, value in enumerate(self._episode_running_rewards)
            )
            component_names = (
                ("tile_reward", "tile"), ("map_reward", "map"), ("screen_reward", "screen"),
                ("story_reward", "story"), ("party_reward", "party"), ("level_reward", "level"),
                ("badge_reward", "badge"), ("trainer_reward", "trainer"),
                ("battle_exp_reward", "battle EXP"),
                ("loop_penalty", "loop"), ("task_success_reward", "task"),
            )
            component_parts = []
            for key, label in component_names:
                pairs = _numeric_info_pairs(infos_for_state, key)
                if pairs:
                    component_parts.append(f"{label}={np.mean([value for _, value in pairs]):+.3f}")
            reward_component_text = " | ".join(component_parts) or "not available"

            timing_row = None
            if TIMING_DIAGNOSTICS and self._timing_samples:
                timing_labels = (
                    ("timing_env_ms", "env"),
                    ("timing_frames_ms", "frames+audio"),
                    ("timing_image_ms", "image+signature"),
                    ("timing_state_ms", "RAM+reward"),
                    ("timing_obs_ms", "observation encode"),
                )
                timing_parts = []
                for timing_key, label in timing_labels:
                    average_ms = self._timing_sums.get(timing_key, 0.0) / self._timing_samples
                    timing_parts.append(f"{label}={average_ms:.2f}ms")
                    metric_name = timing_key.removeprefix("timing_").removesuffix("_ms")
                    self.logger.record(f"timing/{metric_name}", float(average_ms))
                if self._timing_reset_samples:
                    reset_mean = self._timing_reset_sum_ms / self._timing_reset_samples
                    timing_parts.append(f"reset={reset_mean:.1f}ms/call")
                    self.logger.record("timing/reset_mean_ms", float(reset_mean))
                timing_row = ("Step timing (worker mean)", " | ".join(timing_parts))
                self._timing_sums.clear()
                self._timing_samples = 0
                self._timing_reset_sum_ms = 0.0
                self._timing_reset_samples = 0

            rows = [
                ("Objective", f"{OBJECTIVE_LABEL} | mode={'GUI' if GUI_MODE else 'HEADLESS'} | action={action_text}"),
                ("Progress", f"{run_transitions:,} / {run_target:,} transitions ({progress_percent:.2f}%) | global timestep={int(self.num_timesteps):,}"),
                ("Runtime", f"elapsed={_format_duration(elapsed_seconds)} | ETA={_format_duration(remaining_seconds)}"),
                ("Speed", f"{steps_per_second:.1f} transitions/s total | {per_env_speed:.1f}/env/s | ~{aggregate_gba_fps:.0f} GBA frames/s aggregate"),
                ("Workers", f"{worker_status} | configured environments={env_count} | action repeat={ACTION_REPEAT}+{ACTION_RELEASE_FRAMES} release frame(s)"),
                ("Episodes", f"{episode_summary} | last episode: {last_episode_text}"),
                ("Current party / badges", f"party count {party_summary} | total party levels max={level_summary} | badges max={badge_summary}"),
                ("RAM health", f"party telemetry={len(party_values)}/{env_count} workers | live-vs-SaveBlock mismatches={mismatch_count if comparable_party_envs else 'n/a'} | unstable flag samples={sum(int(info.get('flag_telemetry_valid') is False) for info in infos_for_state)}"),
                ("Worker game state", worker_state_text),
                ("Reward summary", f"last vector-step mean={self._last_step_reward:+.4f} | recent transition mean={recent_reward:+.4f} | live episode return mean={live_episode_reward_mean:+.2f} [{live_values}]"),
                ("Reward components", f"per-environment mean for this vector step: {reward_component_text}"),
                ("Sampled actions", _format_action_distribution(Counter(self._recent_actions))),
                ("Greedy/argmax", f"{_format_action_distribution(greedy_counts)} | dominant={greedy_dominance:.1%} | entropy={greedy_entropy:.2f} | unique={len(greedy_counts)}/{len(ACTION_LABELS)}"),
                ("Exploration", f"sampled entropy={sampled_entropy:.2f}/{math.log(len(ACTION_LABELS)):.2f} "
                 f"({sampled_entropy_norm:.0%} normalized) | dominant={sampled_dominance:.1%} "
                 f"| unique={len(action_counts)}/{len(ACTION_LABELS)} | entropy coef={self.model.ent_coef:.4f}"),
                ("Historical progress", f"peak party={int(self._peak_metrics['party_count'])} | "
                 f"peak total levels={int(self._peak_metrics['party_total_levels'])} | "
                 f"peak badges={int(self._peak_metrics['badge_count'])} | "
                 f"starter ever={bool(self._peak_metrics['starter_seen'])} | "
                 f"Pokédex ever={bool(self._peak_metrics['pokedex_seen'])}"),
            ]
            if timing_row is not None:
                rows.append(timing_row)
            if TIMING_DIAGNOSTICS and self.runtime_profiler is not None:
                parent_rows = self.runtime_profiler.format_rows()
                rows.extend(parent_rows)
                for metric_name, values in self.runtime_profiler.snapshot().items():
                    short_name = metric_name[:28]
                    self.logger.record(f"timing/p_{short_name}_ms", float(values["mean_ms"]))
                    self.logger.record(f"timing/p_{short_name}_calls", float(values["count"]))
            if int(self.num_timesteps) - self._last_terminal_progress_timestep >= self.terminal_progress_frequency:
                print("\n" + _format_progress_table(
                    f"POKÉMON EMERALD RL — TRAINING PROGRESS — {run_transitions:,}/{run_target:,}", rows
                ), flush=True)
                self._last_terminal_progress_timestep = int(self.num_timesteps)

            # Flush custom metrics frequently; PPO's own train/* metrics are emitted
            # on rollout/update boundaries. Short keys avoid SB3 table truncation collisions.
            try:
                self.logger.dump(int(self.num_timesteps))
            except Exception as exc:
                print(f"[TensorBoard Warning] Could not flush progress metrics: {exc}", flush=True)
            self._last_progress_timestep = int(self.num_timesteps)

        # Never stop collect_rollouts() mid-batch. Doing so makes SB3 leave the
        # learning loop before train() and discards the rollout without a PPO update.
        # A stop flag is latched here and acted on only after the complete update.
        if GUI_MODE and not STOP_REQUESTED:
            try:
                if any(self.training_env.get_attr("stop_requested")):
                    STOP_REQUESTED = True
            except Exception:
                pass

        current_timestep = int(self.num_timesteps)
        if current_timestep - self._last_save_timestep >= self.save_frequency:
            self._checkpoint_due = True

        if STOP_REQUESTED and not self._stop_notice_printed:
            signal_label = STOP_SIGNAL_NAME or "GUI/stop request"
            print(
                f"\n[Interrupt] {signal_label} received. Finishing the current full PPO rollout "
                "and optimizer update before checkpointing; this may take several minutes.",
                flush=True,
            )
            self._stop_notice_printed = True
        return True

    def _save_after_optimizer_update(self):
        """Publish due/interrupt checkpoints only after RecurrentPPO.train() returns."""
        if not self._checkpoint_due and not STOP_REQUESTED:
            return False

        checkpoint_timestep = int(self.num_timesteps)
        reason = "interrupt" if STOP_REQUESTED else "periodic"
        try:
            saved_model = atomic_save_model(self.model, MODEL_CHECKPOINT)
        except Exception as exc:
            # Keep the request latched. SafeRecurrentPPO will continue with another
            # complete rollout/update and retry at the next safe boundary.
            print(
                f"[Checkpoint Error] Could not save the {reason} model checkpoint at "
                f"{checkpoint_timestep:,}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

        self._last_save_timestep = checkpoint_timestep
        self._checkpoint_due = False
        print(
            f"[Checkpoint] {reason.capitalize()} model atomically saved after a completed PPO update: "
            f"{saved_model} | timesteps={checkpoint_timestep:,} | updates={getattr(self.model, '_n_updates', 'n/a')}.",
            flush=True,
        )
        try:
            archived_model = archive_model_checkpoint(saved_model, checkpoint_timestep)
            if archived_model:
                print(f"[Checkpoint Archive] Preserved snapshot: {archived_model}", flush=True)
        except Exception as exc:
            # The authoritative latest checkpoint is already safe; archive failure
            # must not prevent emulator state saving or terminate training.
            print(
                f"[Checkpoint Archive Warning] Latest checkpoint is safe; snapshot failed: {exc}",
                flush=True,
            )

        emulator_state_saved = _save_vector_env_state(self.training_env, checkpoint_timestep)
        if STOP_REQUESTED:
            if emulator_state_saved:
                print(
                    f"[Interrupt] Model, optimizer, and emulator savestates checkpointed at "
                    f"timestep {checkpoint_timestep:,}. Training will now stop without another rollout.",
                    flush=True,
                )
            else:
                print(
                    f"[Interrupt Warning] Model and optimizer are safely checkpointed at "
                    f"timestep {checkpoint_timestep:,}, but one or more live emulator savestates "
                    "could not be confirmed. The canonical stage start-state remains the recovery point.",
                    flush=True,
                )
            return True
        return False

    def _on_training_end(self):
        global ACTIVE_TRAINING_CALLBACK
        if ACTIVE_TRAINING_CALLBACK is self:
            ACTIVE_TRAINING_CALLBACK = None

class SafeRecurrentPPO(RecurrentPPO):
    """RecurrentPPO that only honors stop requests after a complete update is saved.

    SB3 normally exits its learn loop if a callback returns False during rollout
    collection. That behavior discards the partial rollout before PPO.train(). This
    subclass keeps collection running to the rollout boundary, completes the optimizer
    update, then asks the next collect_rollouts() call to stop.
    """

    def train(self):
        global LAST_COMPLETED_UPDATE_TIMESTEP, PPO_UPDATE_FAILED, SAFE_STOP_AFTER_UPDATE
        try:
            result = super().train()
        except BaseException:
            # A failure halfway through PPO.train() may leave partially-applied optimizer
            # epochs in memory. Do not publish that ambiguous state from main()'s finally.
            PPO_UPDATE_FAILED = True
            raise

        # This marker advances only after all configured PPO epochs complete successfully.
        LAST_COMPLETED_UPDATE_TIMESTEP = int(self.num_timesteps)
        callback = ACTIVE_TRAINING_CALLBACK
        if callback is not None:
            stop_after_save = callback._save_after_optimizer_update()
            if stop_after_save:
                SAFE_STOP_AFTER_UPDATE = True
        return result

    def collect_rollouts(self, *args, **kwargs):
        if SAFE_STOP_AFTER_UPDATE:
            return False
        return super().collect_rollouts(*args, **kwargs)


# Keep the existing public symbol used throughout main.py and by project tests.
PPO = SafeRecurrentPPO


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


def _save_vector_env_state(vec_env, timestep=None):
    """Save each environment once at a checkpoint/shutdown boundary."""
    global LAST_SUCCESSFUL_EMULATOR_SAVE_TIMESTEP
    if vec_env is None:
        return False
    try:
        vec_env.env_method("save_state")
        if timestep is not None and int(timestep) >= 0:
            LAST_SUCCESSFUL_EMULATOR_SAVE_TIMESTEP = int(timestep)
        return True
    except Exception as exc:
        print(f"[State Warning] Could not save all emulator states through vector environment: {exc}", flush=True)
        return False


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
    log_handle = open(path, "a", encoding="utf-8", buffering=65536)
    lock = threading.RLock()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_stdout, log_handle, lock)
    sys.stderr = _TeeStream(old_stderr, log_handle, lock)
    print(f"[Log] Important events and finished reports only: {path} (routine progress tables omitted)", flush=True)
    return log_handle, old_stdout, old_stderr


if __name__ == "__main__":
    env = None
    model = None
    runtime_profiler = None
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
                # Peak metrics belong to the model's learning history. Do not allow a
                # previous model's party/badge highs to masquerade as progress for a new policy.
                peak_metrics_path = os.path.join(RUNTIME_DATA_DIR, "training_metric_peaks.json")
                try:
                    _atomic_write_json(peak_metrics_path, {
                        "schema_version": 1,
                        "party_count": 0.0,
                        "party_total_levels": 0.0,
                        "badge_count": 0.0,
                        "starter_seen": 0.0,
                        "pokedex_seen": 0.0,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })
                except Exception as reset_metrics_exc:
                    print(f"[Metrics Warning] Could not reset stale peak metrics for fresh policy: {reset_metrics_exc}", flush=True)
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
                    f"episode_limit={MAX_EPISODE_STEPS} actions, "
                    f"safe_checkpoint_every={CHECKPOINT_SAVE_FREQUENCY:,} transitions "
                    f"({CHECKPOINT_SAVE_FREQUENCY // VECTOR_ROLLOUT_TRANSITIONS} rollouts).",
                    flush=True,
                )

            runtime_profiler = RuntimeProfiler(enabled=TIMING_DIAGNOSTICS)
            if TIMING_DIAGNOSTICS:
                installed = runtime_profiler.instrument_training(model, env, globals())
                installed_names = ", ".join(name for name, active in installed.items() if active)
                print(
                    f"[Performance] Parent-process timing enabled; probes={installed_names or 'none'}. "
                    "Timings are cumulative averages and include worker wait/IPC separately from worker step time.",
                    flush=True,
                )

            try:
                model.learn(
                    total_timesteps=TOTAL_TIMESTEPS,
                    reset_num_timesteps=False,
                    callback=StopAndCheckpointCallback(runtime_profiler=runtime_profiler),
                )
            except KeyboardInterrupt:
                # Retained as a fallback; the signal handler normally sets STOP_REQUESTED instead.
                print("[Interrupt] KeyboardInterrupt received during training.")
            finally:
                observed_timestep = int(getattr(model, "num_timesteps", 0))
                completed_timestep = LAST_COMPLETED_UPDATE_TIMESTEP

                if PPO_UPDATE_FAILED:
                    # The optimizer may have been interrupted between epochs. Keep the last
                    # published checkpoint; an in-memory partial optimizer update is unsafe.
                    final_timestep = int(LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP or 0)
                    print(
                        "[Save Warning] PPO.train() failed before completing all epochs. "
                        "The in-memory update is not being published; the previous atomic "
                        "checkpoint is retained.", flush=True
                    )
                elif completed_timestep is None:
                    final_timestep = int(LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP or 0)
                    print(
                        "[Save Warning] No completed PPO update boundary was recorded. "
                        "No unverified model checkpoint will be published.", flush=True
                    )
                else:
                    final_timestep = int(completed_timestep)
                    if observed_timestep != final_timestep:
                        print(
                            f"[Save Warning] Discarding {observed_timestep - final_timestep:,} collected-but-untrained "
                            "transitions after an abnormal exit. Saving only the last completed PPO update.",
                            flush=True,
                        )
                        model.num_timesteps = final_timestep

                    if (
                        LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP == final_timestep
                        and os.path.isfile(f"{MODEL_CHECKPOINT}.zip")
                    ):
                        print(
                            f"[Save] Skipping duplicate final model/archive save; timestep {final_timestep:,} "
                            "is already checkpointed.", flush=True
                        )
                    else:
                        try:
                            saved_model = atomic_save_model(model, MODEL_CHECKPOINT)
                            print(f"[Save] PPO model checkpoint atomically saved to {saved_model}", flush=True)
                            try:
                                archived_model = archive_model_checkpoint(saved_model, final_timestep)
                                if archived_model:
                                    print(f"[Checkpoint Archive] Preserved final snapshot: {archived_model}", flush=True)
                            except Exception as archive_exc:
                                print(f"[Checkpoint Archive Warning] Latest checkpoint is safe; final snapshot failed: {archive_exc}", flush=True)
                        except Exception as e:
                            print(f"[Save Error] Could not save PPO model checkpoint: {e}")

            if STOP_REQUESTED:
                if (
                    not PPO_UPDATE_FAILED
                    and LAST_SUCCESSFUL_MODEL_SAVE_TIMESTEP == final_timestep
                    and os.path.isfile(f"{MODEL_CHECKPOINT}.zip")
                ):
                    print(f"[Stopped] Training stopped safely; checkpoint verified at timestep {final_timestep:,}.", flush=True)
                else:
                    print(
                        f"[Stopped Warning] A verified checkpoint at the last completed PPO update "
                        f"(timestep {final_timestep:,}) was not confirmed. Keep the previous checkpoint "
                        "and inspect the save error above before resuming.",
                        flush=True,
                    )
            else:
                print(f"[Complete] Training call finished. Model saved; configured timestep target was {TOTAL_TIMESTEPS}.")

        elif MODE == "evaluate":
            print("--------------------------------------------------")
            print(
                f" [MODE] {'DETERMINISTIC' if EVAL_DETERMINISTIC else 'STOCHASTIC'} EVALUATION | "
                f"objective={OBJECTIVE_LABEL} | episodes={EVAL_EPISODES} | "
                f"max_steps={MAX_EPISODE_STEPS} | no_progress_stop={EVAL_NO_PROGRESS_STEPS} | seed={EVAL_SEED}."
            )
            print(" [MODE] Uses the mapped start state for this objective; no training updates, no GUI and no video.")
            print("--------------------------------------------------", flush=True)

            eval_checkpoint_override = os.environ.get("POKEMON_AI_EVAL_CHECKPOINT", "").strip()
            if eval_checkpoint_override:
                override_path = os.path.expanduser(eval_checkpoint_override)
                if not os.path.isabs(override_path):
                    override_path = os.path.join(BASE_DIR, override_path)
                if override_path.lower().endswith(".zip"):
                    chosen_file = os.path.abspath(override_path)
                    chosen_checkpoint = chosen_file[:-4]
                else:
                    chosen_checkpoint = os.path.abspath(override_path)
                    chosen_file = chosen_checkpoint + ".zip"
            else:
                chosen_checkpoint = MODEL_CHECKPOINT
                chosen_file = f"{chosen_checkpoint}.zip"
            if not os.path.isfile(chosen_file):
                raise FileNotFoundError(
                    f"No evaluation checkpoint found: {chosen_file}. Train the hybrid policy first."
                )
            if not zipfile.is_zipfile(chosen_file):
                raise RuntimeError(f"Evaluation checkpoint is not a valid ZIP archive: {chosen_file}")
            model = PPO.load(chosen_checkpoint, env=env, device="cpu")
            model_env = _get_compatible_model_env(model, "Evaluation")
            print(f"[Evaluation] Loaded {chosen_file}", flush=True)

            # Keep stochastic comparisons reproducible without changing checkpoint weights.
            try:
                model.set_random_seed(EVAL_SEED)
            except Exception as seed_exc:
                print(f"[Evaluation Warning] Could not seed all model RNGs: {seed_exc}", flush=True)
                np.random.seed(EVAL_SEED)

            observation = model_env.reset()
            lstm_states = None
            episode_starts = np.ones((int(getattr(model_env, "num_envs", 1)),), dtype=bool)
            results = []
            detailed_results = []
            for episode_index in range(1, EVAL_EPISODES + 1):
                episode_return = 0.0
                episode_steps = 0
                episode_success = False
                episode_outcome = "UNKNOWN"
                episode_action_counts = Counter()
                episode_experience_gained = 0
                last_info = {}
                last_action_name = "unknown"
                last_policy_summary = None
                previous_action_id = None
                same_action_no_progress_steps = 0
                same_action_same_screen_steps = 0
                previous_progress_signature = None
                previous_state_signature = None
                previous_screen_signature = None
                no_progress_steps = 0
                stagnation_reason = None
                while not STOP_REQUESTED and episode_steps < MAX_EPISODE_STEPS:
                    if episode_steps == 0 or episode_steps % max(1, min(EVAL_DIAGNOSTICS_EVERY, 128)) == 0:
                        last_policy_summary = _policy_action_probability_summary(
                            model, observation, lstm_states, episode_starts
                        )
                    action, lstm_states = model.predict(
                        observation,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=EVAL_DETERMINISTIC,
                    )
                    action_ids = np.asarray(action).reshape(-1)
                    for action_value in action_ids:
                        episode_action_counts[int(action_value)] += 1
                    action_id = int(action_ids[0]) if action_ids.size else 0
                    last_action_name = ACTION_LABELS.get(action_id, f"action_{action_id}")
                    observation, rewards, dones, infos = model_env.step(action)
                    episode_starts = np.asarray(dones, dtype=bool).reshape(-1)
                    last_info = infos[0] if infos and isinstance(infos[0], dict) else {}
                    episode_return += float(np.asarray(rewards, dtype=np.float64).reshape(-1)[0])
                    episode_steps += 1
                    try:
                        episode_experience_gained += max(0, int(last_info.get("party_experience_delta", 0) or 0))
                    except (TypeError, ValueError, OverflowError):
                        pass

                    # The strict-repeat counters remain useful for diagnosing a policy
                    # that repeats one button. The independent no-progress counter catches
                    # alternating-action loops in an already explored area.
                    progress_signature = _evaluation_progress_signature(last_info)
                    # Preserve the old strict-repeat semantics: coordinate changes still
                    # count as state changes for this fast diagnostic, while the separate
                    # no-progress timer intentionally ignores coordinates.
                    state_signature = (
                        progress_signature,
                        last_info.get("player_x"),
                        last_info.get("player_y"),
                    ) if progress_signature is not None else None
                    screen_signature = _screen_observation_signature(observation)
                    if (state_signature is not None and previous_state_signature is not None
                            and action_id == previous_action_id
                            and state_signature == previous_state_signature):
                        same_action_no_progress_steps += 1
                    else:
                        same_action_no_progress_steps = 0
                    if (state_signature is not None and previous_state_signature is not None
                            and action_id == previous_action_id
                            and state_signature == previous_state_signature
                            and screen_signature is not None
                            and screen_signature == previous_screen_signature):
                        same_action_same_screen_steps += 1
                    else:
                        same_action_same_screen_steps = 0

                    if progress_signature is not None:
                        if (previous_progress_signature is None
                                or progress_signature != previous_progress_signature
                                or _evaluation_has_explicit_progress(last_info)):
                            no_progress_steps = 0
                        else:
                            no_progress_steps += 1
                    previous_action_id = action_id
                    if progress_signature is not None:
                        previous_progress_signature = progress_signature
                        previous_state_signature = state_signature
                    previous_screen_signature = screen_signature

                    def print_eval_table(title):
                        pos_x = last_info.get("player_x", "?")
                        pos_y = last_info.get("player_y", "?")
                        diagnostic_rows = [
                            ("Episode / step", f"{episode_index}/{EVAL_EPISODES} | {episode_steps:,}/{MAX_EPISODE_STEPS:,}"),
                            ("Policy mode", f"{'deterministic' if EVAL_DETERMINISTIC else 'stochastic'} | chosen={last_action_name}"),
                            ("Policy probabilities", last_policy_summary or "unavailable for this installed policy API"),
                            ("Game location", f"map={last_info.get('map_id', '?')} | group={last_info.get('map_group', '?')} "
                             f"number={last_info.get('map_number', '?')} | pos=({pos_x},{pos_y})"),
                            ("Party", f"count={last_info.get('party_count', '?')} | levels={last_info.get('party_levels', '?')} "
                             f"| slots={last_info.get('party_slot_levels', '?')} | source={last_info.get('party_count_source', '?')} "
                             f"| SaveBlock={last_info.get('party_count_saveblock1', '?')}"),
                            ("Progress", f"starter={last_info.get('starter_flag', '?')} | Pokédex={last_info.get('pokedex_flag', '?')} "
                             f"| tiles={last_info.get('unique_tiles_episode', '?')} | maps={last_info.get('unique_maps_episode', '?')} "
                             f"| story flags={last_info.get('story_flags_episode', '?')}"),
                            ("Stagnation", f"no meaningful progress={no_progress_steps}/{EVAL_NO_PROGRESS_STEPS} "
                             f"| same action/no state={same_action_no_progress_steps}/{EVAL_STAGNATION_STEPS} "
                             f"| same action + same screen={same_action_same_screen_steps}/{EVAL_STAGNATION_STEPS}"),
                            ("Action histogram", _format_action_distribution(episode_action_counts)),
                        ]
                        print("\n" + _format_progress_table(title, diagnostic_rows, width=None), flush=True)

                    if episode_steps % EVAL_DIAGNOSTICS_EVERY == 0:
                        print_eval_table("POKÉMON EMERALD RL — EVALUATION DIAGNOSTIC")

                    if bool(np.asarray(dones, dtype=bool).reshape(-1)[0]):
                        episode_success = bool(last_info.get("task_success", False))
                        episode_outcome = str(last_info.get("episode_outcome", "SUCCESS" if episode_success else "TIMEOUT"))
                        if isinstance(last_info.get("episode"), dict):
                            episode_return = float(last_info["episode"].get("r", episode_return))
                            episode_steps = int(last_info["episode"].get("l", episode_steps))
                        break

                    if not episode_success and (
                        same_action_same_screen_steps >= EVAL_STAGNATION_STEPS
                        or same_action_no_progress_steps >= EVAL_STAGNATION_STEPS * 2
                        or no_progress_steps >= EVAL_NO_PROGRESS_STEPS
                    ):
                        episode_outcome = "STAGNATED"
                        if same_action_same_screen_steps >= EVAL_STAGNATION_STEPS:
                            stagnation_reason = "same action with unchanged screen/state"
                        elif same_action_no_progress_steps >= EVAL_STAGNATION_STEPS * 2:
                            stagnation_reason = "same action with unchanged game-state telemetry"
                        else:
                            stagnation_reason = (
                                f"no meaningful game progress for {no_progress_steps:,} actions "
                                "(button changes and movement within already explored tiles do not count)"
                            )
                        print_eval_table("POKÉMON EMERALD RL — STAGNATION DETECTED")
                        print(
                            f"[Evaluation Warning] Ended this test early after {episode_steps:,} actions: {stagnation_reason}. "
                            "The evaluator did not override the policy; the greedy policy action is not making progress.",
                            flush=True,
                        )
                        break
                if STOP_REQUESTED:
                    print("[Evaluation] Interrupted by user.", flush=True)
                    break
                if episode_outcome == "UNKNOWN" and episode_steps >= MAX_EPISODE_STEPS:
                    episode_outcome = "TIMEOUT"
                if episode_outcome == "STAGNATED" and stagnation_reason:
                    last_info = dict(last_info)
                    last_info["stagnation_reason"] = stagnation_reason
                results.append((episode_success, episode_return, episode_steps, episode_outcome))
                reward_components = last_info.get("episode_reward_components", {})
                if not reward_components:
                    reward_components = {
                        name: last_info.get(f"episode_{name}_sum", 0.0)
                        for name in REWARD_COMPONENT_NAMES
                        if _finite_number(last_info.get(f"episode_{name}_sum")) is not None
                    }
                if not isinstance(reward_components, dict):
                    reward_components = {}
                terminal_story_flags = last_info.get("story_flags", [])
                if isinstance(terminal_story_flags, (set, frozenset, tuple)):
                    terminal_story_flags = sorted(int(x) for x in terminal_story_flags)
                elif not isinstance(terminal_story_flags, list):
                    terminal_story_flags = []
                detailed_results.append({
                    "episode": int(episode_index),
                    "objective": OBJECTIVE_NAME,
                    "success": bool(episode_success),
                    "outcome": str(episode_outcome),
                    "return": float(episode_return),
                    "steps": int(episode_steps),
                    "party_count": last_info.get("party_count"),
                    "party_levels": last_info.get("party_levels"),
                    "party_slot_levels": list(last_info.get("party_slot_levels", ()))
                    if isinstance(last_info.get("party_slot_levels", ()), (list, tuple)) else [],
                    "party_count_source": last_info.get("party_count_source"),
                    "party_count_saveblock1": last_info.get("party_count_saveblock1"),
                    "badge_count": last_info.get("badge_count"),
                    "map_id": last_info.get("map_id"),
                    "player_x": last_info.get("player_x"),
                    "player_y": last_info.get("player_y"),
                    "unique_tiles": last_info.get("unique_tiles_episode"),
                    "unique_maps": last_info.get("unique_maps_episode"),
                    "story_flags_episode": last_info.get("story_flags_episode"),
                    "trainer_flags_episode": last_info.get("trainer_flags_episode"),
                    "flag_telemetry_invalid_steps": int(last_info.get("flag_telemetry_invalid_steps", 0)),
                    "trainer_flag_telemetry_invalid_steps": int(last_info.get("trainer_flag_telemetry_invalid_steps", 0)),
                    "flag_telemetry_valid": bool(last_info.get("flag_telemetry_valid", False)),
                    "trainer_flag_telemetry_valid": bool(last_info.get("trainer_flag_telemetry_valid", False)),
                    "saveblock_pointer_stable": bool(last_info.get("saveblock_pointer_stable", False)),
                    "story_flag_regressions": list(last_info.get("story_flag_regressions", ())),
                    "badge_flag_regressions": int(last_info.get("badge_flag_regressions", 0)),
                    "trainer_flag_regressions": list(last_info.get("trainer_flag_regressions", ())),
                    "raw_flag_bytes": dict(last_info.get("raw_flag_bytes", {})),
                    "starter_flag_raw": last_info.get("starter_flag_raw"),
                    "starter_flag": bool(last_info.get("starter_flag", False)),
                    "party_experience_gained": int(episode_experience_gained),
                    "stagnation_reason": last_info.get("stagnation_reason"),
                    "reward_components": {
                        str(key): float(value) for key, value in reward_components.items()
                        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
                    },
                    "action_counts": {ACTION_LABELS.get(int(k), str(k)): int(v) for k, v in episode_action_counts.items()},
                })
                episode_rows = [
                    ("Result", f"{episode_outcome} | success={episode_success} | objective={OBJECTIVE_LABEL}"),
                    ("Return / length", f"{episode_return:.3f} | {episode_steps:,} actions"),
                    ("Game state", f"map={last_info.get('map_id', '?')} | pos=({last_info.get('player_x', '?')},{last_info.get('player_y', '?')}) "
                     f"| party={last_info.get('party_count', '?')} | total levels={last_info.get('party_levels', '?')} "
                     f"| badges={last_info.get('badge_count', '?')}"),
                    ("Exploration", f"tiles={last_info.get('unique_tiles_episode', '?')} | maps={last_info.get('unique_maps_episode', '?')} "
                     f"| story flags={last_info.get('story_flags_episode', '?')}"),
                    ("Flag telemetry", f"stable={last_info.get('flag_telemetry_valid', False)} | trainer stable={last_info.get('trainer_flag_telemetry_valid', False)} "
                     f"| pointer stable={last_info.get('saveblock_pointer_stable', False)} | invalid samples={last_info.get('flag_telemetry_invalid_steps', 0)} "
                     f"| starter raw/effective={last_info.get('starter_flag_raw')}/{last_info.get('starter_flag', False)} "
                     f"| regressions={len(last_info.get('story_flag_regressions', ())) + len(last_info.get('trainer_flag_regressions', ())) + int(last_info.get('badge_flag_regressions', 0)).bit_count()}"),
                    ("Actions", _format_action_distribution(episode_action_counts)),
                ]
                print("\n" + _format_progress_table(
                    f"EVALUATION EPISODE {episode_index}/{EVAL_EPISODES}", episode_rows, width=None
                ), flush=True)
                # Vector envs auto-reset only after done. A stagnation stop is evaluator-side,
                # so explicitly restore the canonical start before the next test episode.
                if episode_outcome in {"STAGNATED", "UNKNOWN"} and episode_index < EVAL_EPISODES:
                    observation = model_env.reset()
                    lstm_states = None
                    episode_starts = np.ones((int(getattr(model_env, "num_envs", 1)),), dtype=bool)
            if results:
                successes = sum(1 for success, *_ in results if success)
                success_rate = successes / len(results)
                mean_return = float(np.mean([episode_return for _, episode_return, _, _ in results]))
                mean_steps = float(np.mean([episode_steps for _, _, episode_steps, _ in results]))
                timeouts = sum(1 for _, _, _, outcome in results if outcome == "TIMEOUT")
                stagnations = sum(1 for _, _, _, outcome in results if outcome == "STAGNATED")
                summary_rows = [
                    ("Objective", OBJECTIVE_LABEL),
                    ("Episodes", f"{len(results)} completed | successes={successes} | success rate={success_rate:.1%}"),
                    ("Returns", f"mean={mean_return:.3f}"),
                    ("Episode length", f"mean={mean_steps:.0f} actions"),
                    ("Failure modes", f"timeouts={timeouts} | stagnations={stagnations}"),
                ]
                print("\n" + _format_progress_table("EVALUATION SUMMARY", summary_rows, width=None), flush=True)
            else:
                print("[Evaluation Summary] No complete episodes were evaluated.", flush=True)

            if results:
                report_successes = sum(1 for success, *_ in results if success)
                report_payload = {
                    "schema_version": 1,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "objective": OBJECTIVE_NAME,
                    "objective_label": OBJECTIVE_LABEL,
                    "mode": "deterministic" if EVAL_DETERMINISTIC else "stochastic",
                    "seed": int(EVAL_SEED),
                    "checkpoint": os.path.abspath(chosen_file),
                    "checkpoint_sha1": _sha1_file(chosen_file),
                    "start_state": os.path.abspath(TRAINING_START_STATE),
                    "rom_sha1_expected": EXPECTED_ROM_SHA1,
                    "episodes_requested": int(EVAL_EPISODES),
                    "episodes_completed": len(results),
                    "successes": int(report_successes),
                    "success_rate": float(report_successes / len(results)),
                    "mean_return": float(np.mean([r for _, r, _, _ in results])),
                    "mean_steps": float(np.mean([n for _, _, n, _ in results])),
                    "timeouts": int(sum(1 for _, _, _, outcome in results if outcome == "TIMEOUT")),
                    "stagnations": int(sum(1 for _, _, _, outcome in results if outcome == "STAGNATED")),
                    "episodes": detailed_results,
                    "flag_telemetry_invalid_steps": int(sum(item.get("flag_telemetry_invalid_steps", 0) for item in detailed_results)),
                    "trainer_flag_telemetry_invalid_steps": int(sum(item.get("trainer_flag_telemetry_invalid_steps", 0) for item in detailed_results)),
                    "flag_telemetry_integrity_ok": bool(all(item.get("flag_telemetry_invalid_steps", 0) == 0 for item in detailed_results)),
                    "trainer_flag_telemetry_integrity_ok": bool(all(item.get("trainer_flag_telemetry_invalid_steps", 0) == 0 for item in detailed_results)),
                    "saveblock_pointer_integrity_ok": bool(all(item.get("saveblock_pointer_stable", False) for item in detailed_results)),
                    "persistent_flag_regressions": int(sum(
                        len(item.get("story_flag_regressions", ()))
                        + int(item.get("badge_flag_regressions", 0)).bit_count()
                        + len(item.get("trainer_flag_regressions", ()))
                        for item in detailed_results
                    )),
                }
                report_path = os.path.join(LOG_DIR, f"evaluation_{TIMESTAMP_STR}_{os.getpid()}.json")
                try:
                    _atomic_write_json(report_path, report_payload)
                    print(f"[Evaluation Report] Machine-readable report saved to {report_path}", flush=True)
                except Exception as report_exc:
                    print(f"[Evaluation Warning] Could not save machine-readable report: {report_exc}", flush=True)

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
            print(f" [MODE] WATCH PLAYBACK ({NUM_CPU} environment). Objective: {OBJECTIVE_LABEL}.")
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
            print(
                f"[Watch] Plays from the mapped start state for {OBJECTIVE_LABEL}; after "
                f"success it continues for {WATCH_SUCCESS_GRACE_STEPS} extra decision(s) "
                "so the transition can be seen, then stops. Set "
                "POKEMON_AI_WATCH_SUCCESS_GRACE_STEPS=0 to stop immediately.",
                flush=True,
            )
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
            watch_success_start_step = None
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
                # Trigger only once on the transition into success (not every step after it).
                # A short grace window lets the viewer see a starter's battle transition without
                # letting an untrained policy wander indefinitely after its assigned objective.
                just_completed = any(
                    isinstance(info, dict) and info.get("objective_just_completed", False)
                    for info in info_values
                )
                if just_completed and watch_success_start_step is None:
                    watch_success_start_step = watch_steps
                    watch_task_completed = True
                    print(
                        f"[Watch] TASK SUCCESS: {OBJECTIVE_LABEL} objective reached. "
                        f"Showing {WATCH_SUCCESS_GRACE_STEPS} additional decision(s); "
                        "START/SELECT remain available and are not blocked.",
                        flush=True,
                    )
                if (
                    watch_success_start_step is not None
                    and watch_steps - watch_success_start_step >= WATCH_SUCCESS_GRACE_STEPS
                ):
                    print("[Watch] Objective grace window complete; stopping playback.", flush=True)
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
                print(f"[Watch] Playback stopped before the {OBJECTIVE_LABEL} objective completed.", flush=True)

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
            current_model = locals().get("model")
            current_timestep = int(getattr(current_model, "num_timesteps", -1)) if current_model is not None else -1
            if current_timestep >= 0 and LAST_SUCCESSFUL_EMULATOR_SAVE_TIMESTEP == current_timestep:
                print(
                    f"[Save] Skipping duplicate shutdown emulator savestate; timestep {current_timestep:,} "
                    "was already saved.", flush=True
                )
            else:
                _save_vector_env_state(env, timestep=current_timestep)
            if runtime_profiler is not None and TIMING_DIAGNOSTICS:
                final_profile_rows = runtime_profiler.format_rows()
                if final_profile_rows:
                    print("\n" + _format_progress_table("FINAL PARENT-PROCESS PERFORMANCE PROFILE", final_profile_rows, width=None), flush=True)
                runtime_profiler.restore()
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
