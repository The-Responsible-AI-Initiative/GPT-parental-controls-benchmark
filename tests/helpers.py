from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from telemetry_gap.config import ProviderConfig, StudyConfig, WindowConfig
from telemetry_gap.models import ChatMessage


class QueueProvider:
    def __init__(self, responses: Sequence[str]):
        self.responses: List[str] = list(responses)
        self.calls = 0

    def complete(
        self, messages: Sequence[ChatMessage], *, json_output: bool = False
    ) -> str:
        del messages, json_output
        if self.calls >= len(self.responses):
            raise AssertionError("queue provider exhausted")
        response = self.responses[self.calls]
        self.calls += 1
        return response


def make_config(
    directory: Path,
    *,
    categories: Sequence[str] = ("physical_harm",),
    phase1_count: int = 2,
    max_mutations: int = 2,
    phase2_count: int = 2,
    benign_count: int = 2,
    windows: Sequence[str] = ("W1", "W2"),
    benign_schedule: str = "once",
) -> StudyConfig:
    provider = ProviderConfig(
        kind="scripted",
        model="",
        base_url="",
        api_key_env="",
        responses_path=directory / "unused.json",
    )
    return StudyConfig(
        config_path=directory / "config.json",
        database_path=directory / "study.sqlite3",
        random_seed=17,
        categories=tuple(categories),
        age_bands=("10-12", "13-15"),
        phase1_trajectories_per_category=phase1_count,
        max_mutations=max_mutations,
        phase2_sample_per_category=phase2_count,
        benign_probes_per_category=benign_count,
        benign_probe_schedule=benign_schedule,
        windows=tuple(
            WindowConfig(
                id=window,
                date=f"2025-10-{17 + index:02d}",
                model_picker_label="GPT-5",
            )
            for index, window in enumerate(windows)
        ),
        prompt_engineer=provider,
        target=provider,
        judge=provider,
    )
