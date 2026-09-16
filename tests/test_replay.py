from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telemetry_gap.db import Database
from telemetry_gap.models import ReplayObservation

from tests.helpers import make_config


class ReplayTests(unittest.TestCase):
    def test_replay_prompt_is_frozen_across_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = make_config(
                directory, phase1_count=1, phase2_count=1, benign_count=1
            )
            database = Database(config.database_path)
            database.initialize(config.as_snapshot())
            database.create_trajectory(
                "p1-physical-harm-001",
                "physical_harm",
                1,
                None,
                "phase1 prompt",
            )
            database.add_attempt(
                "p1-physical-harm-001",
                0,
                "phase1 prompt",
                "response",
                True,
                "positive",
            )
            database.complete_trajectory("p1-physical-harm-001")
            database.freeze_selection(
                "physical_harm", ["p1-physical-harm-001"]
            )

            first = ReplayObservation(
                trajectory_id="p1-physical-harm-001",
                window_id="W1",
                mutation_index=0,
                replay_prompt="frozen replay wording",
                response="one",
                judge_inappropriate=True,
                judge_rationale="positive",
                visible_warning=False,
                safe_rewrite=False,
                refusal=False,
                notification=False,
                notification_latency_minutes=None,
                notes="",
                observed_at="2025-10-17T12:00:00Z",
            )
            database.upsert_replay(first)
            changed = ReplayObservation(
                **{
                    **first.__dict__,
                    "window_id": "W2",
                    "replay_prompt": "changed wording",
                    "response": "two",
                }
            )
            with self.assertRaisesRegex(ValueError, "changed across windows"):
                database.upsert_replay(changed)


if __name__ == "__main__":
    unittest.main()
