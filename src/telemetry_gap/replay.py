from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .config import StudyConfig
from .db import Database
from .models import BenignObservation, ReplayObservation
from .prompts import judge_messages, parse_judge_response
from .providers import TextProvider


REPLAY_FIELDS = (
    "window_id",
    "window_date",
    "model_picker_label",
    "trajectory_id",
    "category",
    "selection_rank",
    "phase1_ordinal",
    "age_band",
    "mutation_index",
    "phase1_prompt",
    "replay_prompt",
    "response",
    "judge_inappropriate",
    "judge_rationale",
    "visible_warning",
    "safe_rewrite",
    "refusal",
    "notification",
    "notification_latency_minutes",
    "observed_at",
    "notes",
)

BENIGN_PROBE_FIELDS = (
    "probe_id",
    "category",
    "ordinal",
    "age_band",
    "prompt",
    "prevalidated_by",
    "notes",
)

BENIGN_REPLAY_FIELDS = (
    "window_id",
    "probe_id",
    "category",
    "ordinal",
    "age_band",
    "prompt",
    "prevalidated_by",
    "response",
    "refusal",
    "heavy_sanitization",
    "observed_at",
    "notes",
)


class ReplayError(ValueError):
    pass


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _require_columns(reader: csv.DictReader, required: Sequence[str]) -> None:
    actual = set(reader.fieldnames or ())
    missing = set(required) - actual
    if missing:
        raise ReplayError(f"CSV is missing columns: {', '.join(sorted(missing))}")


def _parse_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ReplayError(f"row {row_number}: {field} must be yes/no or 1/0")


def _parse_optional_bool(value: str, field: str, row_number: int) -> Optional[bool]:
    if not value.strip():
        return None
    return _parse_bool(value, field, row_number)


def _parse_optional_float(
    value: str, field: str, row_number: int
) -> Optional[float]:
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ReplayError(f"row {row_number}: {field} must be numeric") from exc
    if parsed < 0:
        raise ReplayError(f"row {row_number}: {field} cannot be negative")
    return parsed


def _validate_timestamp(value: str, field: str, row_number: int) -> None:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ReplayError(
            f"row {row_number}: {field} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ReplayError(
            f"row {row_number}: {field} must include a UTC offset or Z"
        )


def export_replay_sheet(config: StudyConfig, database: Database, path: Path) -> None:
    manifest = database.selected_manifest_rows()
    if not manifest:
        raise ReplayError("no frozen Phase II selection; run phase1-select first")
    rows: List[Dict[str, object]] = []
    for window in config.windows:
        for row in manifest:
            prompt = str(row["prompt"])
            rows.append(
                {
                    "window_id": window.id,
                    "window_date": window.date,
                    "model_picker_label": window.model_picker_label,
                    "trajectory_id": row["trajectory_id"],
                    "category": row["category"],
                    "selection_rank": row["selection_rank"],
                    "phase1_ordinal": row["ordinal"],
                    "age_band": row["age_band"] or "",
                    "mutation_index": row["mutation_index"],
                    "phase1_prompt": prompt,
                    "replay_prompt": prompt,
                    "response": "",
                    "judge_inappropriate": "",
                    "judge_rationale": "",
                    "visible_warning": "",
                    "safe_rewrite": "",
                    "refusal": "",
                    "notification": "",
                    "notification_latency_minutes": "",
                    "observed_at": "",
                    "notes": "",
                }
            )
    _write_csv(path, REPLAY_FIELDS, rows)


def import_replay_sheet(config: StudyConfig, database: Database, path: Path) -> int:
    selected = set(database.selected_trajectory_ids())
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader,
            (
                "window_id",
                "trajectory_id",
                "mutation_index",
                "replay_prompt",
                "response",
                "judge_inappropriate",
                "judge_rationale",
                "visible_warning",
                "safe_rewrite",
                "refusal",
                "notification",
                "notification_latency_minutes",
                "observed_at",
                "notes",
            ),
        )
        for row_number, row in enumerate(reader, start=2):
            trajectory_id = row["trajectory_id"].strip()
            window_id = row["window_id"].strip()
            if trajectory_id not in selected:
                raise ReplayError(
                    f"row {row_number}: trajectory {trajectory_id!r} is not selected"
                )
            if window_id not in config.window_ids:
                raise ReplayError(
                    f"row {row_number}: unknown measurement window {window_id!r}"
                )
            try:
                mutation_index = int(row["mutation_index"])
            except ValueError as exc:
                raise ReplayError(
                    f"row {row_number}: mutation_index must be an integer"
                ) from exc
            replay_prompt = row["replay_prompt"].strip()
            response = row["response"].strip()
            observed_at = row["observed_at"].strip()
            if not replay_prompt or not response or not observed_at:
                raise ReplayError(
                    f"row {row_number}: replay_prompt, response, and observed_at are required"
                )
            _validate_timestamp(observed_at, "observed_at", row_number)
            judge_inappropriate = _parse_optional_bool(
                row["judge_inappropriate"], "judge_inappropriate", row_number
            )
            judge_rationale = row["judge_rationale"].strip()
            if judge_inappropriate is not None and not judge_rationale:
                raise ReplayError(
                    f"row {row_number}: judge_rationale is required with a judge label"
                )
            notification = _parse_bool(
                row["notification"], "notification", row_number
            )
            latency = _parse_optional_float(
                row["notification_latency_minutes"],
                "notification_latency_minutes",
                row_number,
            )
            if not notification and latency is not None:
                raise ReplayError(
                    f"row {row_number}: notification latency requires notification=yes"
                )
            observation = ReplayObservation(
                trajectory_id=trajectory_id,
                window_id=window_id,
                mutation_index=mutation_index,
                replay_prompt=replay_prompt,
                response=response,
                judge_inappropriate=judge_inappropriate,
                judge_rationale=judge_rationale,
                visible_warning=_parse_bool(
                    row["visible_warning"], "visible_warning", row_number
                ),
                safe_rewrite=_parse_bool(
                    row["safe_rewrite"], "safe_rewrite", row_number
                ),
                refusal=_parse_bool(row["refusal"], "refusal", row_number),
                notification=notification,
                notification_latency_minutes=latency,
                notes=row["notes"].strip(),
                observed_at=observed_at,
            )
            database.upsert_replay(observation)
            count += 1
    return count


def judge_unlabeled_replays(
    database: Database, judge: TextProvider, window_id: Optional[str] = None
) -> int:
    query = """
        SELECT r.id, r.replay_prompt, r.response, t.category
        FROM replays r
        JOIN trajectories t ON t.id = r.trajectory_id
        WHERE r.judge_inappropriate IS NULL
    """
    parameters: Sequence[object] = ()
    if window_id is not None:
        query += " AND r.window_id = ?"
        parameters = (window_id,)
    query += " ORDER BY r.window_id, t.category, r.trajectory_id, r.mutation_index"
    rows = database.rows(query, parameters)
    for row in rows:
        raw = judge.complete(
            judge_messages(
                str(row["category"]),
                str(row["replay_prompt"]),
                str(row["response"]),
            ),
            json_output=True,
        )
        decision = parse_judge_response(raw)
        database.update_replay_judge(
            int(row["id"]), decision.inappropriate, decision.rationale
        )
    return len(rows)


def import_benign_probes(
    config: StudyConfig, database: Database, path: Path
) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, BENIGN_PROBE_FIELDS)
        for row_number, row in enumerate(reader, start=2):
            category = row["category"].strip()
            if category not in config.categories:
                raise ReplayError(f"row {row_number}: unknown category {category!r}")
            try:
                ordinal = int(row["ordinal"])
            except ValueError as exc:
                raise ReplayError(
                    f"row {row_number}: ordinal must be an integer"
                ) from exc
            prompt = row["prompt"].strip()
            prevalidated_by = row["prevalidated_by"].strip()
            if not prompt or not prevalidated_by:
                raise ReplayError(
                    f"row {row_number}: prompt and prevalidated_by are required"
                )
            probe_id = row["probe_id"].strip() or f"benign-{category}-{ordinal:03d}"
            database.import_benign_probe(
                probe_id,
                category,
                ordinal,
                row["age_band"].strip() or None,
                prompt,
                prevalidated_by,
                row["notes"].strip(),
            )
            count += 1
    return count


def export_benign_replay_sheet(
    config: StudyConfig, database: Database, path: Path
) -> None:
    probes = database.benign_manifest_rows()
    if not probes:
        raise ReplayError("no benign probes have been imported")
    window_ids = (
        ("BENIGN",)
        if config.benign_probe_schedule == "once"
        else config.window_ids
    )
    rows: List[Dict[str, object]] = []
    for window_id in window_ids:
        for row in probes:
            rows.append(
                {
                    "window_id": window_id,
                    "probe_id": row["probe_id"],
                    "category": row["category"],
                    "ordinal": row["ordinal"],
                    "age_band": row["age_band"] or "",
                    "prompt": row["prompt"],
                    "prevalidated_by": row["prevalidated_by"],
                    "response": "",
                    "refusal": "",
                    "heavy_sanitization": "",
                    "observed_at": "",
                    "notes": row["notes"],
                }
            )
    _write_csv(path, BENIGN_REPLAY_FIELDS, rows)


def import_benign_replay_sheet(
    config: StudyConfig, database: Database, path: Path
) -> int:
    valid_windows = (
        {"BENIGN"}
        if config.benign_probe_schedule == "once"
        else set(config.window_ids)
    )
    known_probes = {
        str(row["probe_id"]) for row in database.benign_manifest_rows()
    }
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader,
            (
                "window_id",
                "probe_id",
                "response",
                "refusal",
                "heavy_sanitization",
                "observed_at",
                "notes",
            ),
        )
        for row_number, row in enumerate(reader, start=2):
            window_id = row["window_id"].strip()
            probe_id = row["probe_id"].strip()
            if window_id not in valid_windows:
                raise ReplayError(
                    f"row {row_number}: invalid benign window {window_id!r}"
                )
            if probe_id not in known_probes:
                raise ReplayError(f"row {row_number}: unknown probe {probe_id!r}")
            response = row["response"].strip()
            observed_at = row["observed_at"].strip()
            if not response or not observed_at:
                raise ReplayError(
                    f"row {row_number}: response and observed_at are required"
                )
            _validate_timestamp(observed_at, "observed_at", row_number)
            database.upsert_benign_replay(
                BenignObservation(
                    probe_id=probe_id,
                    window_id=window_id,
                    response=response,
                    refusal=_parse_bool(row["refusal"], "refusal", row_number),
                    heavy_sanitization=_parse_bool(
                        row["heavy_sanitization"],
                        "heavy_sanitization",
                        row_number,
                    ),
                    notes=row["notes"].strip(),
                    observed_at=observed_at,
                )
            )
            count += 1
    return count
