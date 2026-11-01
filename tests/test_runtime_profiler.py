import unittest

from runtime_profiler import RuntimeProfiler


class FakePolicy:
    def forward(self, value):
        return value + 1

    def _predict(self, value):
        return value * 2


class FakeLogger:
    def dump(self, step=None):
        return step


class FakeModel:
    def __init__(self):
        self.policy = FakePolicy()
        self.logger = FakeLogger()
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        return "trained"


class FakeEnv:
    def __init__(self):
        self.actions = []

    def step_async(self, actions):
        self.actions.append(actions)
        return "sent"

    def step_wait(self):
        return "received"

    def env_method(self, method_name, *args, **kwargs):
        return (method_name, args, kwargs)


class RuntimeProfilerTests(unittest.TestCase):
    def test_instruments_parent_inference_vecenv_optimizer_tensorboard_and_saves(self):
        profiler = RuntimeProfiler(enabled=True)
        model = FakeModel()
        env = FakeEnv()
        namespace = {
            "atomic_save_model": lambda *args, **kwargs: "model.zip",
            "archive_model_checkpoint": lambda *args, **kwargs: "archive.zip",
        }
        installed = profiler.instrument_training(model, env, namespace)

        self.assertTrue(installed["policy_forward"])
        self.assertTrue(installed["policy_predict"])
        self.assertTrue(installed["vec_send"])
        self.assertTrue(installed["vec_wait"])
        self.assertFalse(installed["ppo_update"], "PPO train must be timed by SafeRecurrentPPO, not monkeypatched")
        self.assertNotIn("train", vars(model), "profiler must not attach a serializable closure to the model")
        self.assertTrue(installed["tensorboard_dump"])
        self.assertTrue(installed["model_save"])
        self.assertTrue(installed["archive_save"])

        self.assertEqual(model.policy.forward(3), 4)
        self.assertEqual(model.policy._predict(4), 8)
        self.assertEqual(env.step_async([1, 2]), "sent")
        self.assertEqual(env.step_wait(), "received")
        self.assertEqual(model.train(), "trained")
        profiler.record("ppo_update", 0.012)
        self.assertEqual(model.logger.dump(123), 123)
        self.assertEqual(namespace["atomic_save_model"](model), "model.zip")
        self.assertEqual(namespace["archive_model_checkpoint"]("model.zip", 1), "archive.zip")
        self.assertEqual(env.env_method("save_state"), ("save_state", (), {}))

        snapshot = profiler.snapshot()
        for key in (
            "policy_forward", "policy_predict", "vec_send", "vec_wait",
            "ppo_update", "tensorboard_dump", "model_save", "archive_save",
            "emulator_state_save",
        ):
            self.assertEqual(snapshot[key]["count"], 1, key)
            self.assertGreaterEqual(snapshot[key]["total_seconds"], 0.0, key)
        self.assertAlmostEqual(snapshot["ppo_update"]["last_ms"], 12.0, places=3)
        self.assertAlmostEqual(snapshot["ppo_update"]["max_ms"], 12.0, places=3)
        self.assertIn("last=0.01s", profiler.format_rows()[2][1])
        self.assertIn("n=1", profiler.format_rows()[2][1])

        self.assertEqual(len(profiler.format_rows()), 4)

    def test_restore_returns_wrapped_methods_and_namespace_functions_to_originals(self):
        profiler = RuntimeProfiler(enabled=True)
        model = FakeModel()
        env = FakeEnv()
        original_forward = model.policy.forward
        original_train = model.train
        original_step_async = env.step_async
        namespace = {"atomic_save_model": lambda: "original"}
        original_save = namespace["atomic_save_model"]
        profiler.instrument_training(model, env, namespace)
        self.assertIs(model.train.__func__, original_train.__func__)
        self.assertIs(model.train.__self__, original_train.__self__)
        profiler.restore()

        self.assertIs(model.policy.forward.__func__, original_forward.__func__)
        self.assertIs(model.policy.forward.__self__, original_forward.__self__)
        self.assertIs(model.train.__func__, original_train.__func__)
        self.assertIs(model.train.__self__, original_train.__self__)
        self.assertIs(env.step_async.__func__, original_step_async.__func__)
        self.assertIs(env.step_async.__self__, original_step_async.__self__)
        self.assertIs(namespace["atomic_save_model"], original_save)
        self.assertEqual(namespace["atomic_save_model"](), "original")
        self.assertFalse(profiler._installed)

    def test_disabled_profiler_does_not_patch_or_record(self):
        profiler = RuntimeProfiler(enabled=False)
        model = FakeModel()
        env = FakeEnv()
        original_train = model.train
        installed = profiler.instrument_training(model, env, {})
        self.assertEqual(installed, {})
        self.assertIs(model.train.__func__, original_train.__func__)
        self.assertIs(model.train.__self__, original_train.__self__)
        profiler.record("ignored", 1.0)
        self.assertEqual(profiler.snapshot(), {})

    def test_env_method_distinguishes_savestate_write_from_other_calls(self):
        profiler = RuntimeProfiler(enabled=True)
        model = FakeModel()
        env = FakeEnv()
        profiler.instrument_training(model, env)
        env.env_method("get_attr", "episode_step_counter")
        env.env_method("save_state")
        snap = profiler.snapshot()
        self.assertEqual(snap["vec_env_method"]["count"], 1)
        self.assertEqual(snap["emulator_state_save"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
