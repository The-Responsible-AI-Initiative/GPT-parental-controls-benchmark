from __future__ import annotations

import unittest

from telemetry_gap.audit import cohens_kappa


class AuditTests(unittest.TestCase):
    def test_binary_cohens_kappa(self) -> None:
        self.assertAlmostEqual(
            cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
