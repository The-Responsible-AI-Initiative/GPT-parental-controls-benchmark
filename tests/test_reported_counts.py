from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telemetry_gap.analysis import analyze_window
from telemetry_gap.db import Database
from telemetry_gap.models import BenignObservation, ReplayObservation

from tests.helpers import make_config


REPORTED_COUNTS = {
    "pornography": {"NR": 20, "LR": 0, "OBR": 4, "UIR": 0},
    "health_consultation": {"NR": 8, "LR": 0, "OBR": 8, "UIR": 12},
    "physical_harm": {"NR": 20, "LR": 0, "OBR": 12, "UIR": 16},
    "hate_speech": {"NR": 0, "LR": 4, "OBR": 8, "UIR": 2},
    "malware": {"NR": 0, "LR": 16, "OBR": 0, "UIR": 1},
    "fraud": {"NR": 0, "LR": 12, "OBR": 8, "UIR": 1},
    "privacy_violence": {"NR": 0, "LR": 16, "OBR": 4, "UIR": 1},
}


class ReportedCountRegressionTests(unittest.TestCase):
    def test_synthetic_sessions_match_reported_table_numerators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            categories = tuple(REPORTED_COUNTS)
            config = make_config(
                directory,
                categories=categories,
                phase1_count=20,
                phase2_count=20,
                benign_count=20,
                windows=("W1",),
            )
            database = Database(config.database_path)
            database.initialize(config.as_snapshot())

            for category, expected in REPORTED_COUNTS.items():
                selected = []
                for ordinal in range(1, 21):
                    trajectory_id = f"p1-{category}-{ordinal:03d}"
                    selected.append(trajectory_id)
                    prompt = f"synthetic {category} prompt {ordinal}"
                    database.create_trajectory(
                        trajectory_id, category, ordinal, None, prompt
                    )
                    database.add_attempt(
                        trajectory_id,
                        0,
                        prompt,
                        "synthetic Phase-I response",
                        True,
                        "selection fixture",
                    )
                    database.complete_trajectory(trajectory_id)
                database.freeze_selection(category, selected)

                for ordinal, trajectory_id in enumerate(selected, start=1):
                    database.upsert_replay(
                        ReplayObservation(
                            trajectory_id=trajectory_id,
                            window_id="W1",
                            mutation_index=0,
                            replay_prompt=f"synthetic {category} prompt {ordinal}",
                            response="synthetic replay response",
                            judge_inappropriate=ordinal <= expected["LR"],
                            judge_rationale="synthetic binary label",
                            visible_warning=ordinal <= expected["UIR"],
                            safe_rewrite=False,
                            refusal=False,
                            notification=ordinal <= expected["NR"],
                            notification_latency_minutes=(
                                1.0 if ordinal <= expected["NR"] else None
                            ),
                            notes="aggregate regression fixture; not study data",
                            observed_at="2025-10-17T12:00:00Z",
                        )
                    )

                for ordinal in range(1, 21):
                    probe_id = f"benign-{category}-{ordinal:03d}"
                    database.import_benign_probe(
                        probe_id,
                        category,
                        ordinal,
                        None,
                        f"synthetic benign {category} prompt {ordinal}",
                        "synthetic fixture",
                        "not study data",
                    )
                    database.upsert_benign_replay(
                        BenignObservation(
                            probe_id=probe_id,
                            window_id="BENIGN",
                            response="synthetic benign response",
                            refusal=ordinal <= expected["OBR"],
                            heavy_sanitization=False,
                            notes="aggregate regression fixture; not study data",
                            observed_at="2025-10-17T13:00:00Z",
                        )
                    )

            results = analyze_window(config, database, "W1")
            for category, expected in REPORTED_COUNTS.items():
                for metric, positive in expected.items():
                    self.assertEqual(results[category][metric].positive, positive)
                    self.assertEqual(results[category][metric].total, 20)


if __name__ == "__main__":
    unittest.main()
