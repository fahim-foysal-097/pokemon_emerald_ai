"""Small, dependency-free filters for concise persistent Pokémon RL logs.

Console output remains complete. Persistent text logs keep important events and
finished reports while omitting routine progress tables and per-step diagnostics.
"""
from __future__ import annotations

import re

_PROGRESS_NOISE = (
    "POKÉMON EMERALD RL — TRAINING PROGRESS",
    "POKEMON EMERALD RL — TRAINING PROGRESS",
    "[Training] run=",
    "[Evaluation Diagnostic]",
    "[Mini-Episode]",
)

_REPORT_TABLE_MARKERS = (
    "OBJECTIVE COMPLETED",
    "EVALUATION SUMMARY",
    "EVALUATION REPORT",
    "POLICY COLLAPSE GUARD",
    "PARTY RAM DIAGNOSTIC",
    "PARTY RAM CONSISTENCY WARNING",
)

_IMPORTANT_MARKERS = (
    "[Log]",
    "[Config Warning]",
    "[ROM Preflight]",
    "[ROM] Exact identity verified",
    "[Environment Setup]",
    "[Start State] Validated",
    "[Training] Started",
    "[Training] Starting",
    "[Training] RecurrentPPO",
    "[Evaluation] Loaded",
    "[Evaluation Summary]",
    "[Evaluation Report]",
    "[Curriculum Evaluation]",
    "[Curriculum] STAGE",
    "[Curriculum] MASTERED:",
    "[Curriculum] NOT PROMOTED:",
    "[Curriculum] COMPLETE:",
    "[Curriculum] Interrupted.",
    "[Curriculum] Stop requested",
    "[Curriculum Error]",
    "[Save] PPO model checkpoint atomically saved",
    "[Checkpoint]",
    "[Checkpoint Archive]",
    "[Checkpoint Warning]",
    "[Milestone Save]",
    "[Milestone Warning]",
    "[TensorBoard]",
    "[TensorBoard Warning]",
    "[Interrupt]",
    "[Stopped]",
    "[Complete]",
    "[Shutdown]",
    "[Shutdown Warning]",
    "[Shutdown Error]",
    "[Cleanup Warning]",
    "[Fatal Error]",
    "[Save Error]",
    "[Exit]",
    "[Session Timer]",
    "[Recorder Error]",
    "[State Warning]",
    "[Metrics Warning]",
    "[Policy Probe Warning]",
    "[Evaluation Warning]",
    "[Evaluation Error]",
    "[Evaluation interrupted]",
    "[mGBA Logging Warning]",
    "[Runtime Cleanup Warning]",
    "[Runtime Migration Warning]",
    "[Notification Warning]",
    "[Party RAM Diagnostic]",
    "[Setup RAM Warning]",
    "[Objective] FAILED",
    "[Objective] SUCCESS",
    "MASTERED:",
    "NOT PROMOTED:",
    "COMPLETE: all configured stages",
)

_URGENT_MARKERS = (
    "Warning",
    "WARNING",
    "Error",
    "ERROR",
    "Traceback (most recent call last)",
    "Exception:",
    "[Save] PPO model checkpoint atomically saved",
    "[Checkpoint]",
    "[Checkpoint Archive]",
    "OBJECTIVE COMPLETED",
    "EVALUATION SUMMARY",
    "EVALUATION REPORT",
    "[Curriculum Evaluation]",
    "[Curriculum] MASTERED:",
    "[Curriculum] NOT PROMOTED:",
    "[Curriculum] COMPLETE:",
    "[Stopped]",
    "[Complete]",
    "[Exit]",
)

_EXCEPTION_END = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt)|SystemExit|KeyboardInterrupt)(?:\s*:\s*|$)"
)


def _is_progress_noise(text: str) -> bool:
    return any(marker in text for marker in _PROGRESS_NOISE)


def is_important_log_line(line: str) -> bool:
    """Return whether a single line is worth keeping in the compact run log."""
    if not line or _is_progress_noise(line):
        return False
    if any(marker in line for marker in _REPORT_TABLE_MARKERS):
        # Keep the report heading in the curriculum log; main.py keeps the full
        # multi-line report when it is emitted as a single formatted string.
        return True
    if "EPISODE FINISHED" in line:
        # Episode tables are noisy when obtained repeatedly during training. The
        # machine-readable evaluation report records actual evaluation outcomes.
        return False
    if any(marker in line for marker in _IMPORTANT_MARKERS):
        return True
    stripped = line.strip()
    if _EXCEPTION_END.match(stripped):
        return True
    if "Traceback (most recent call last)" in line:
        return True
    # Generic warnings/errors are useful even if a new subsystem adds its own tag.
    upper = line.upper()
    return (" WARNING]" in upper or " ERROR]" in upper or "FATAL ERROR" in upper)


def filter_persistent_log_text(text: str) -> str:
    """Filter a whole write() payload, retaining important reports in full.

    Progress tables and success episode tables are intentionally omitted. Objective
    completion and evaluation/diagnostic reports are sparse and retained in full.
    Other output is reduced to individually important lines.
    """
    if not text or not text.strip() or _is_progress_noise(text):
        return ""

    if "EPISODE FINISHED" in text:
        # Keep non-success terminal episodes (timeouts/failures); skip routine wins.
        if re.search(r"Outcome\s*\|\s*SUCCESS", text):
            return ""
        return text

    if any(marker in text for marker in _REPORT_TABLE_MARKERS):
        return text

    kept = []
    for line in text.splitlines(keepends=True):
        if is_important_log_line(line):
            kept.append(line)
    return "".join(kept)


def should_flush_important_log(text: str) -> bool:
    """Flush only for important events; ordinary progress is never persisted."""
    return any(marker in text for marker in _URGENT_MARKERS) or "Traceback" in text


def is_exception_end(text: str) -> bool:
    """Recognize the last line of a Python traceback so capture can stop."""
    return any(_EXCEPTION_END.match(line.strip()) for line in text.splitlines() if line.strip())
