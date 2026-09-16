from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .db import Database


AUDIT_FIELDS = (
    "item_id",
    "category",
    "prompt",
    "response",
    "judge_label",
    "rater_a_label",
    "rater_b_label",
    "adjudicated_label",
    "notes",
)


class AuditError(ValueError):
    pass


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _attempt_rows(database: Database) -> List[Mapping[str, object]]:
    return database.rows(
        """
        SELECT
            'attempt:' || a.id AS item_id,
            t.category,
            a.prompt,
            a.response,
            a.judge_inappropriate AS judge_label
        FROM attempts a
        JOIN trajectories t ON t.id = a.trajectory_id
        WHERE t.status = 'completed'
        ORDER BY t.category, a.id
        """
    )


def _allocate_stratified(
    group_sizes: Mapping[str, int], sample_size: int
) -> Dict[str, int]:
    total = sum(group_sizes.values())
    if sample_size <= 0 or sample_size > total:
        raise AuditError(f"sample size must be between 1 and {total}")
    raw = {
        category: sample_size * size / total
        for category, size in group_sizes.items()
    }
    allocation = {
        category: min(size, int(math.floor(raw[category])))
        for category, size in group_sizes.items()
    }
    remaining = sample_size - sum(allocation.values())
    order = sorted(
        group_sizes,
        key=lambda category: (
            raw[category] - math.floor(raw[category]),
            group_sizes[category],
            category,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for category in order:
            if allocation[category] < group_sizes[category]:
                allocation[category] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise AuditError("could not allocate the requested stratified sample")
    return allocation


def _write_audit(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "judge_label": (
                        "inappropriate" if bool(row["judge_label"]) else "appropriate"
                    ),
                    "rater_a_label": "",
                    "rater_b_label": "",
                    "adjudicated_label": "",
                    "notes": "",
                }
            )


def export_calibration_sample(
    database: Database, path: Path, sample_size: int, seed: int
) -> int:
    groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in _attempt_rows(database):
        groups[str(row["category"])].append(row)
    allocation = _allocate_stratified(
        {category: len(rows) for category, rows in groups.items()}, sample_size
    )
    sample: List[Mapping[str, object]] = []
    for category in sorted(groups):
        rng = random.Random(_stable_seed(seed, f"calibration:{category}"))
        sample.extend(rng.sample(groups[category], allocation[category]))
    sample.sort(key=lambda row: (str(row["category"]), str(row["item_id"])))
    _write_audit(path, sample)
    return len(sample)


def export_targeted_audit(
    database: Database, path: Path, negative_fraction: float, seed: int
) -> int:
    if negative_fraction < 0 or negative_fraction > 1:
        raise AuditError("negative_fraction must be between 0 and 1")
    rows = _attempt_rows(database)
    positives = [row for row in rows if bool(row["judge_label"])]
    negatives_by_category: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if not bool(row["judge_label"]):
            negatives_by_category[str(row["category"])].append(row)
    sampled_negatives: List[Mapping[str, object]] = []
    for category, negatives in sorted(negatives_by_category.items()):
        count = int(math.ceil(len(negatives) * negative_fraction))
        rng = random.Random(_stable_seed(seed, f"targeted:{category}"))
        sampled_negatives.extend(rng.sample(negatives, count))
    sample = positives + sampled_negatives
    sample.sort(key=lambda row: (str(row["category"]), str(row["item_id"])))
    _write_audit(path, sample)
    return len(sample)


def _parse_label(value: str, field: str, row_number: int) -> int:
    normalized = value.strip().lower()
    if normalized in {"inappropriate", "positive", "yes", "1", "true"}:
        return 1
    if normalized in {"appropriate", "negative", "no", "0", "false"}:
        return 0
    raise AuditError(
        f"row {row_number}: {field} must be appropriate or inappropriate"
    )


def cohens_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise AuditError("kappa requires non-empty label vectors of equal length")
    if any(label not in {0, 1} for label in (*left, *right)):
        raise AuditError("kappa labels must be binary")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_positive = sum(left) / total
    right_positive = sum(right) / total
    expected = (
        left_positive * right_positive
        + (1 - left_positive) * (1 - right_positive)
    )
    if math.isclose(expected, 1.0):
        raise AuditError("kappa is undefined when both raters use one class only")
    return (observed - expected) / (1 - expected)


def score_audit(path: Path) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "category",
            "judge_label",
            "rater_a_label",
            "rater_b_label",
            "adjudicated_label",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise AuditError(
                f"audit CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                {
                    "category": row["category"].strip(),
                    "judge": _parse_label(
                        row["judge_label"], "judge_label", row_number
                    ),
                    "rater_a": _parse_label(
                        row["rater_a_label"], "rater_a_label", row_number
                    ),
                    "rater_b": _parse_label(
                        row["rater_b_label"], "rater_b_label", row_number
                    ),
                    "adjudicated": _parse_label(
                        row["adjudicated_label"], "adjudicated_label", row_number
                    ),
                }
            )
    if not rows:
        raise AuditError("audit CSV contains no rows")

    def safe_kappa(left: Sequence[int], right: Sequence[int]) -> Optional[float]:
        try:
            return cohens_kappa(left, right)
        except AuditError as exc:
            if "undefined" not in str(exc):
                raise
            return None

    def score(subset: Sequence[Mapping[str, object]]) -> Dict[str, Optional[float]]:
        return {
            "judge_vs_adjudicated": safe_kappa(
                [int(row["judge"]) for row in subset],
                [int(row["adjudicated"]) for row in subset],
            ),
            "inter_human": safe_kappa(
                [int(row["rater_a"]) for row in subset],
                [int(row["rater_b"]) for row in subset],
            ),
        }

    by_category: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    return {
        "n": len(rows),
        "overall": score(rows),
        "by_category": {
            category: {"n": len(subset), **score(subset)}
            for category, subset in sorted(by_category.items())
        },
    }


def render_audit_score(score: Mapping[str, object]) -> str:
    return json.dumps(score, indent=2, sort_keys=True) + "\n"
