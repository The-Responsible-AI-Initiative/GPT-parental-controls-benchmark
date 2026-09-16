from __future__ import annotations

import unittest

from telemetry_gap.config import ProviderConfig
from telemetry_gap.providers import ProviderError, create_provider


class ProviderTests(unittest.TestCase):
    def test_unconfigured_model_is_rejected_before_api_use(self) -> None:
        config = ProviderConfig(
            kind="openai_compatible",
            model="CHANGE_ME",
            base_url="https://example.invalid/v1/chat/completions",
            api_key_env="UNUSED_API_KEY",
        )
        with self.assertRaisesRegex(ProviderError, "configure a provider model"):
            create_provider(config)


if __name__ == "__main__":
    unittest.main()
