from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telemetry_gap.analysis import AnalysisError, analyze_window, wilson_interval
from telemetry_gap.db import Database
from telemetry_gap.models import BenignObservation, ReplayObservation

from tests.helpers import make_config


class AnalysisTests(unittest.TestCase):
    def test_wilson_matches_table_rounding(self) -> None:
        estimate = wilson_interval(16, 20)
        self.assertEqual(estimate.paper_cell(), "16/20 (80%) [58-92]")
        zero = wilson_interval(0, 20)
        self.assertEqual(zero.paper_cell(), "0/20 (0%) [0-16]")

    def test_session_level_metrics_use_disjoint_exact_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = make_config(directory)
            database = Database(config.database_path)
            database.initialize(config.as_snapshot())

            for ordinal in (1, 2):
                trajectory_id = f"p1-physical-harm-{ordinal:03d}"
                database.create_trajectory(
                    trajectory_id,
                    "physical_harm",
                    ordinal,
                    "10-12",
                    f"prompt {ordinal}",
                )
                database.add_attempt(
                    trajectory_id,
                    0,
                    f"prompt {ordinal}",
                    f"response {ordinal}",
                    True,
                    "positive",
                )
                database.complete_trajectory(trajectory_id)
            database.freeze_selection(
                "physical_harm",
                ["p1-physical-harm-001", "p1-physical-harm-002"],
            )

            observations = [
                ReplayObservation(
                    trajectory_id="p1-physical-harm-001",
                    window_id="W1",
                    mutation_index=0,
                    replay_prompt="prompt 1",
                    response="replay 1",
                    judge_inappropriate=True,
                    judge_rationale="positive",
                    visible_warning=True,
                    safe_rewrite=False,
                    refusal=False,
                    notification=True,
                    notification_latency_minutes=4.0,
                    notes="",
                    observed_at="2025-10-17T12:00:00Z",
                ),
                ReplayObservation(
                    trajectory_id="p1-physical-harm-002",
                    window_id="W1",
                    mutation_index=0,
                    replay_prompt="prompt 2",
                    response="replay 2",
                    judge_inappropriate=False,
                    judge_rationale="negative",
                    visible_warning=False,
                    safe_rewrite=False,
                    refusal=False,
                    notification=False,
                    notification_latency_minutes=None,
                    notes="",
                    observed_at="2025-10-17T12:05:00Z",
                ),
            ]
            for observation in observations:
                database.upsert_replay(observation)

            for ordinal in (1, 2):
                probe_id = f"benign-physical_harm-{ordinal:03d}"
                database.import_benign_probe(
                    probe_id,
                    "physical_harm",
                    ordinal,
                    "10-12",
                    f"benign {ordinal}",
                    "rater-a",
                    "",
                )
                database.upsert_benign_replay(
                    BenignObservation(
                        probe_id=probe_id,
                        window_id="BENIGN",
                        response=f"benign response {ordinal}",
                        refusal=ordinal == 1,
                        heavy_sanitization=False,
                        notes="",
                        observed_at="2025-10-17T13:00:00Z",
                    )
                )

            metrics = analyze_window(config, database, "W1")["physical_harm"]
            for metric in ("NR", "LR", "OBR", "UIR"):
                self.assertEqual(metrics[metric].positive, 1)
                self.assertEqual(metrics[metric].total, 2)

    def test_analysis_rejects_incomplete_replay_trajectories(self) -> None:
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
                "prompt",
            )
            database.add_attempt(
                "p1-physical-harm-001", 0, "prompt", "response", True, "positive"
            )
            database.complete_trajectory("p1-physical-harm-001")
            database.freeze_selection(
                "physical_harm", ["p1-physical-harm-001"]
            )
            database.import_benign_probe(
                "benign-1", "physical_harm", 1, None, "benign", "rater", ""
            )

            with self.assertRaisesRegex(AnalysisError, "incomplete"):
                analyze_window(config, database, "W1")


if __name__ == "__main__":
    unittest.main()
