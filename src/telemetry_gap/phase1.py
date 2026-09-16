from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from .config import StudyConfig
from .db import Database
from .prompts import (
    judge_messages,
    mutation_messages,
    parse_judge_response,
    parse_prompt_response,
    seed_messages,
    target_messages,
)
from .providers import TextProvider


class Phase1Error(RuntimeError):
    pass


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"cannot derive an identifier from {value!r}")
    return slug


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Phase1Summary:
    attempted: int
    completed: int
    skipped: int
    failed: int


class Phase1Runner:
    def __init__(
        self,
        config: StudyConfig,
        database: Database,
        prompt_engineer: TextProvider,
        target: TextProvider,
        judge: TextProvider,
    ):
        self.config = config
        self.database = database
        self.prompt_engineer = prompt_engineer
        self.target = target
        self.judge = judge

    def run(
        self,
        categories: Optional[Sequence[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> Phase1Summary:
        selected_categories = tuple(categories or self.config.categories)
        unknown = set(selected_categories) - set(self.config.categories)
        if unknown:
            raise Phase1Error(f"unknown categories: {', '.join(sorted(unknown))}")

        count = (
            self.config.phase1_trajectories_per_category
            if limit_per_category is None
            else limit_per_category
        )
        if count <= 0 or count > self.config.phase1_trajectories_per_category:
            raise Phase1Error(
                "limit_per_category must be between 1 and "
                f"{self.config.phase1_trajectories_per_category}"
            )

        attempted = completed = skipped = failed = 0
        for category in selected_categories:
            for ordinal in range(1, count + 1):
                attempted += 1
                trajectory_rng = random.Random(
                    _stable_seed(self.config.random_seed, f"{category}:{ordinal}")
                )
                age_band = (
                    trajectory_rng.choice(self.config.age_bands)
                    if self.config.age_bands
                    else None
                )
                nonce = (
                    f"{category}-{ordinal}-"
                    f"{trajectory_rng.getrandbits(64):016x}"
                )
                existing = self.database.trajectory(category, ordinal)
                if existing is not None:
                    if str(existing["status"]) == "completed":
                        skipped += 1
                        continue
                    raise Phase1Error(
                        f"trajectory {category}/{ordinal} already exists with "
                        f"status {existing['status']!r}; inspect it before retrying"
                    )

                trajectory_id = f"p1-{slugify(category)}-{ordinal:03d}"
                try:
                    seed_raw = self.prompt_engineer.complete(
                        seed_messages(category, age_band, nonce), json_output=True
                    )
                    prompt = parse_prompt_response(seed_raw)
                    self.database.create_trajectory(
                        trajectory_id, category, ordinal, age_band, prompt
                    )
                    for mutation_index in range(self.config.max_mutations):
                        response = self.target.complete(target_messages(prompt))
                        judge_raw = self.judge.complete(
                            judge_messages(category, prompt, response), json_output=True
                        )
                        decision = parse_judge_response(judge_raw)
                        self.database.add_attempt(
                            trajectory_id,
                            mutation_index,
                            prompt,
                            response,
                            decision.inappropriate,
                            decision.rationale,
                        )
                        if decision.inappropriate:
                            break
                        if mutation_index + 1 < self.config.max_mutations:
                            mutation_raw = self.prompt_engineer.complete(
                                mutation_messages(
                                    category,
                                    age_band,
                                    prompt,
                                    response,
                                    decision.rationale,
                                ),
                                json_output=True,
                            )
                            prompt = parse_prompt_response(mutation_raw)
                    self.database.complete_trajectory(trajectory_id)
                    completed += 1
                except Exception as exc:
                    failed += 1
                    existing_after_error = self.database.trajectory(category, ordinal)
                    if existing_after_error is not None:
                        self.database.fail_trajectory(trajectory_id, str(exc))
                    raise Phase1Error(
                        f"trajectory {category}/{ordinal} failed: {exc}"
                    ) from exc
        return Phase1Summary(attempted, completed, skipped, failed)


def freeze_phase2_selection(config: StudyConfig, database: Database) -> None:
    selections = {}
    for category in config.categories:
        positives = database.positive_trajectory_ids(category)
        required = config.phase2_sample_per_category
        if len(positives) < required:
            raise Phase1Error(
                f"{category} has {len(positives)} judge-positive trajectories; "
                f"{required} are required for Phase II"
            )
        rng = random.Random(_stable_seed(config.random_seed, f"phase2:{category}"))
        selections[category] = rng.sample(positives, required)
    database.freeze_selections(selections)
