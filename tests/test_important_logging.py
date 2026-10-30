from __future__ import annotations

import unittest

import important_logging as log_filter


class ImportantLoggingTests(unittest.TestCase):
    def test_training_progress_table_is_not_persisted(self):
        payload = (
            "+-----+\n"
            "| POKÉMON EMERALD RL — TRAINING PROGRESS — 5,000/250,000 |\n"
            "| Speed | 100 transitions/s |\n"
            "+-----+\n"
        )
        self.assertEqual(log_filter.filter_persistent_log_text(payload), "")

    def test_success_episode_table_is_not_persisted(self):
        payload = (
            "+-----+\n| EPISODE FINISHED — ENV 0 |\n| Outcome | SUCCESS |\n"
            "| Length / return | 100 steps | 67.0 |\n+-----+\n"
        )
        self.assertEqual(log_filter.filter_persistent_log_text(payload), "")

    def test_failed_episode_table_is_kept(self):
        payload = (
            "+-----+\n| EPISODE FINISHED — ENV 0 |\n| Outcome | TIMEOUT |\n"
            "| Length / return | 30,000 steps | -1.0 |\n+-----+\n"
        )
        self.assertEqual(log_filter.filter_persistent_log_text(payload), payload)

    def test_finished_objective_report_is_kept_in_full(self):
        payload = (
            "+-----+\n| OBJECTIVE COMPLETED |\n| Status | SUCCESS |\n"
            "| Party | count=1 |\n+-----+\n"
        )
        self.assertEqual(log_filter.filter_persistent_log_text(payload), payload)

    def test_unimportant_child_progress_line_is_not_persisted(self):
        self.assertFalse(log_filter.is_important_log_line("[Training] run=10,000/250,000"))
        self.assertFalse(log_filter.is_important_log_line("[Evaluation Diagnostic] step=10,000"))

    def test_checkpoint_warning_and_traceback_are_important(self):
        self.assertTrue(log_filter.is_important_log_line("[Checkpoint Warning] could not save"))
        self.assertTrue(log_filter.is_important_log_line("Traceback (most recent call last):"))
        self.assertTrue(log_filter.should_flush_important_log("[Checkpoint] saved model"))
        self.assertTrue(log_filter.is_exception_end("RuntimeError: broken"))


if __name__ == "__main__":
    unittest.main()
