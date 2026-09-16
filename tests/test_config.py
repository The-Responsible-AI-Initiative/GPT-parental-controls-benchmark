from __future__ import annotations

import unittest
from pathlib import Path

from telemetry_gap.config import load_config


class ConfigTests(unittest.TestCase):
    def test_example_config_matches_manuscript_sample_sizes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(str(root / "config.example.json"))
        self.assertEqual(config.phase1_trajectories_per_category, 100)
        self.assertEqual(config.max_mutations, 15)
        self.assertEqual(config.phase2_sample_per_category, 20)
        self.assertEqual(config.benign_probes_per_category, 20)
        self.assertEqual(config.window_ids, ("W1", "W2", "W3"))


if __name__ == "__main__":
    unittest.main()
