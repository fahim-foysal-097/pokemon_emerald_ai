"""Low-overhead parent-process timing for the Pokémon Emerald RL training loop.

Uses only the Python standard library. Instrumentation is opt-in and is intended
for short diagnostic runs, not as a source of benchmark-quality microsecond data.
"""
from __future__ import annotations

from collections import defaultdict
import functools
import time
from typing import Any


class RuntimeProfiler:
    """Measure parent-process inference, VecEnv IPC, PPO updates and disk/log I/O.

    Wrapped methods are restored before process shutdown. The wrapper records only
    wall-clock duration and does not inspect or copy tensors/observations.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._maxima: dict[str, float] = defaultdict(float)
        self._last_values: dict[str, float] = {}
        self._patches: list[tuple[Any, ...]] = []
        self._installed = False

    @property
    def is_installed(self) -> bool:
        return self._installed

    def record(self, metric: str, seconds: float) -> None:
        if not self.enabled:
            return
        try:
            value = max(0.0, float(seconds))
        except (TypeError, ValueError, OverflowError):
            return
        self._totals[metric] += value
        self._counts[metric] += 1
        self._last_values[metric] = value
        if value > self._maxima[metric]:
            self._maxima[metric] = value

    def _wrap_attribute(self, owner: Any, name: str, metric: str) -> bool:
        if owner is None or not hasattr(owner, name):
            return False
        original = getattr(owner, name)
        if not callable(original):
            return False

        profiler = self

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                profiler.record(metric, time.perf_counter() - started)

        try:
            owner_dict = vars(owner)
            had_instance_value = name in owner_dict
            previous_instance_value = owner_dict.get(name)
        except TypeError:
            had_instance_value = False
            previous_instance_value = None
        try:
            setattr(owner, name, timed)
        except (AttributeError, TypeError):
            return False
        self._patches.append(("attribute", owner, name, had_instance_value, previous_instance_value))
        return True

    def _wrap_namespace(self, namespace: dict[str, Any], name: str, metric: str) -> bool:
        original = namespace.get(name)
        if not callable(original):
            return False
        profiler = self

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                profiler.record(metric, time.perf_counter() - started)

        namespace[name] = timed
        self._patches.append(("mapping", namespace, name, original))
        return True

    def _wrap_env_method(self, env: Any) -> bool:
        if env is None or not hasattr(env, "env_method"):
            return False
        original = env.env_method
        profiler = self

        @functools.wraps(original)
        def timed(method_name: str, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(method_name, *args, **kwargs)
            finally:
                metric = "emulator_state_save" if method_name == "save_state" else "vec_env_method"
                profiler.record(metric, time.perf_counter() - started)

        try:
            env_dict = vars(env)
            had_instance_value = "env_method" in env_dict
            previous_instance_value = env_dict.get("env_method")
        except TypeError:
            had_instance_value = False
            previous_instance_value = None
        try:
            setattr(env, "env_method", timed)
        except (AttributeError, TypeError):
            return False
        self._patches.append(("attribute", env, "env_method", had_instance_value, previous_instance_value))
        return True

    def instrument_training(self, model: Any, env: Any, namespace: dict[str, Any] | None = None) -> dict[str, bool]:
        """Wrap the main training call sites and return which probes were installed."""
        if not self.enabled:
            return {}
        if self._installed:
            raise RuntimeError("RuntimeProfiler is already installed")

        policy = getattr(model, "policy", None)
        logger = getattr(model, "logger", None)
        installed = {
            # One policy.forward call produces actions for the vectorized env batch.
            "policy_forward": self._wrap_attribute(policy, "forward", "policy_forward"),
            # SB3's predict path is used by the optional greedy/argmax probe.
            "policy_predict": self._wrap_attribute(policy, "_predict", "policy_predict"),
            "vec_send": self._wrap_attribute(env, "step_async", "vec_send"),
            # This is wall time waiting for all workers + IPC/serialization.
            "vec_wait": self._wrap_attribute(env, "step_wait", "vec_wait"),
            # Do NOT wrap model.train on the model instance. SB3 serializes the
            # model's __dict__; a closure stored there retains this profiler, the
            # SubprocVecEnv and multiprocessing AuthenticationString objects.
            # SafeRecurrentPPO.train records PPO duration explicitly instead.
            "ppo_update": False,
            "tensorboard_dump": self._wrap_attribute(logger, "dump", "tensorboard_dump"),
            "env_method": self._wrap_env_method(env),
        }
        if namespace is not None:
            installed["model_save"] = self._wrap_namespace(namespace, "atomic_save_model", "model_save")
            installed["archive_save"] = self._wrap_namespace(namespace, "archive_model_checkpoint", "archive_save")
        self._installed = True
        return installed

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        """Return cumulative counts, average/max milliseconds and total seconds."""
        result: dict[str, dict[str, float | int]] = {}
        for name, count in self._counts.items():
            total = self._totals[name]
            result[name] = {
                "count": int(count),
                "total_seconds": float(total),
                "mean_ms": float(total * 1000.0 / count) if count else 0.0,
                "last_ms": float(self._last_values.get(name, 0.0) * 1000.0),
                "max_ms": float(self._maxima[name] * 1000.0),
            }
        return result

    def restore(self) -> None:
        """Restore every wrapped attribute/function in reverse installation order."""
        for patch in reversed(self._patches):
            try:
                kind = patch[0]
                if kind == "mapping":
                    _, mapping, name, original = patch
                    mapping[name] = original
                else:
                    _, owner, name, had_instance_value, previous_instance_value = patch
                    if had_instance_value:
                        setattr(owner, name, previous_instance_value)
                    else:
                        try:
                            delattr(owner, name)
                        except AttributeError:
                            pass
            except (AttributeError, KeyError, TypeError):
                pass
        self._patches.clear()
        self._installed = False

    def format_rows(self) -> list[tuple[str, str]]:
        """Produce compact terminal rows separating per-call and per-rollout costs."""
        data = self.snapshot()
        if not data:
            return []

        def mean_ms(name: str) -> str:
            item = data.get(name)
            return f"{item['mean_ms']:.2f}ms" if item else "n/a"

        def seconds_summary(name: str) -> str:
            item = data.get(name)
            if not item or not item["count"]:
                return "n/a"
            mean_seconds_value = item["total_seconds"] / item["count"]
            last_seconds_value = item["last_ms"] / 1000.0
            max_seconds_value = item["max_ms"] / 1000.0
            return (
                f"last={last_seconds_value:.2f}s | avg={mean_seconds_value:.2f}s | "
                f"max={max_seconds_value:.2f}s | n={item['count']}"
            )

        inference = (
            f"policy forward={mean_ms('policy_forward')} | "
            f"extra predict={mean_ms('policy_predict')}"
        )
        vector = f"send={mean_ms('vec_send')} | worker wait+IPC={mean_ms('vec_wait')}"
        optimizer = seconds_summary("ppo_update")
        io = (
            f"TensorBoard dump={mean_ms('tensorboard_dump')} | "
            f"model save={seconds_summary('model_save')} | "
            f"archive={mean_ms('archive_save')} | "
            f"savestates={seconds_summary('emulator_state_save')}"
        )
        return [
            ("Parent timing — inference", inference),
            ("Parent timing — vector env", vector),
            ("Parent timing — PPO update", optimizer),
            ("Parent timing — I/O", io),
        ]
