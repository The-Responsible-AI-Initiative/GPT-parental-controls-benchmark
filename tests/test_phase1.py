from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telemetry_gap.db import Database
from telemetry_gap.phase1 import Phase1Runner, freeze_phase2_selection

from tests.helpers import QueueProvider, make_config


class Phase1Tests(unittest.TestCase):
    def test_runner_mutates_until_binary_judge_is_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = make_config(
                directory,
                phase1_count=1,
                phase2_count=1,
                max_mutations=3,
            )
            database = Database(config.database_path)
            database.initialize(config.as_snapshot())
            prompt_engineer = QueueProvider(
                [
                    '{"prompt":"seed prompt","design_note":"seed"}',
                    '{"prompt":"mutated prompt","mutation_note":"retry"}',
                ]
            )
            target = QueueProvider(["safe response", "unsafe response"])
            judge = QueueProvider(
                [
                    '{"inappropriate":false,"rationale":"safe"}',
                    '{"inappropriate":true,"rationale":"crossed boundary"}',
                ]
            )

            summary = Phase1Runner(
                config, database, prompt_engineer, target, judge
            ).run()

            self.assertEqual(summary.completed, 1)
            attempts = database.rows(
                "SELECT * FROM attempts ORDER BY mutation_index"
            )
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["prompt"], "seed prompt")
            self.assertEqual(attempts[1]["prompt"], "mutated prompt")
            self.assertEqual(attempts[1]["judge_inappropriate"], 1)

            freeze_phase2_selection(config, database)
            self.assertEqual(
                database.selected_trajectory_ids(), ["p1-physical-harm-001"]
            )


if __name__ == "__main__":
    unittest.main()
