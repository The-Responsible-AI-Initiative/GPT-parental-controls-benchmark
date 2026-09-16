from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .models import BenignObservation, ReplayObservation


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    age_band TEXT,
    seed_prompt TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    selected_phase2 INTEGER NOT NULL DEFAULT 0 CHECK (selected_phase2 IN (0, 1)),
    selection_rank INTEGER,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(category, ordinal)
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    mutation_index INTEGER NOT NULL CHECK (mutation_index >= 0),
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    judge_inappropriate INTEGER NOT NULL CHECK (judge_inappropriate IN (0, 1)),
    judge_rationale TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(trajectory_id, mutation_index)
);

CREATE TABLE IF NOT EXISTS replays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    window_id TEXT NOT NULL,
    mutation_index INTEGER NOT NULL CHECK (mutation_index >= 0),
    replay_prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    judge_inappropriate INTEGER CHECK (judge_inappropriate IN (0, 1)),
    judge_rationale TEXT NOT NULL DEFAULT '',
    visible_warning INTEGER NOT NULL CHECK (visible_warning IN (0, 1)),
    safe_rewrite INTEGER NOT NULL CHECK (safe_rewrite IN (0, 1)),
    refusal INTEGER NOT NULL CHECK (refusal IN (0, 1)),
    notification INTEGER NOT NULL CHECK (notification IN (0, 1)),
    notification_latency_minutes REAL,
    notes TEXT NOT NULL DEFAULT '',
    observed_at TEXT NOT NULL,
    UNIQUE(trajectory_id, window_id, mutation_index)
);

CREATE TABLE IF NOT EXISTS benign_probes (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    age_band TEXT,
    prompt TEXT NOT NULL,
    prevalidated_by TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(category, ordinal)
);

CREATE TABLE IF NOT EXISTS benign_replays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    probe_id TEXT NOT NULL REFERENCES benign_probes(id) ON DELETE CASCADE,
    window_id TEXT NOT NULL,
    response TEXT NOT NULL,
    refusal INTEGER NOT NULL CHECK (refusal IN (0, 1)),
    heavy_sanitization INTEGER NOT NULL CHECK (heavy_sanitization IN (0, 1)),
    notes TEXT NOT NULL DEFAULT '',
    observed_at TEXT NOT NULL,
    UNIQUE(probe_id, window_id)
);

CREATE INDEX IF NOT EXISTS idx_trajectories_category
    ON trajectories(category, selected_phase2);
CREATE INDEX IF NOT EXISTS idx_attempts_trajectory
    ON attempts(trajectory_id, mutation_index);
CREATE INDEX IF NOT EXISTS idx_replays_window
    ON replays(window_id, trajectory_id);
CREATE INDEX IF NOT EXISTS idx_benign_category
    ON benign_probes(category);
CREATE INDEX IF NOT EXISTS idx_benign_replays_window
    ON benign_replays(window_id, probe_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, config_snapshot: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES('config_snapshot', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (json.dumps(config_snapshot, sort_keys=True),),
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES('schema_version', '1')
                ON CONFLICT(key) DO NOTHING
                """
            )

    def create_trajectory(
        self,
        trajectory_id: str,
        category: str,
        ordinal: int,
        age_band: Optional[str],
        seed_prompt: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO trajectories(
                    id, category, ordinal, age_band, seed_prompt, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    trajectory_id,
                    category,
                    ordinal,
                    age_band,
                    seed_prompt,
                    utc_now(),
                ),
            )

    def trajectory(self, category: str, ordinal: int) -> Optional[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM trajectories WHERE category = ? AND ordinal = ?",
                (category, ordinal),
            ).fetchone()

    def complete_trajectory(self, trajectory_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE trajectories
                SET status = 'completed', completed_at = ?, failure_reason = NULL
                WHERE id = ?
                """,
                (utc_now(), trajectory_id),
            )

    def fail_trajectory(self, trajectory_id: str, reason: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE trajectories
                SET status = 'failed', completed_at = ?, failure_reason = ?
                WHERE id = ?
                """,
                (utc_now(), reason, trajectory_id),
            )

    def add_attempt(
        self,
        trajectory_id: str,
        mutation_index: int,
        prompt: str,
        response: str,
        judge_inappropriate: bool,
        judge_rationale: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO attempts(
                    trajectory_id, mutation_index, prompt, response,
                    judge_inappropriate, judge_rationale, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory_id,
                    mutation_index,
                    prompt,
                    response,
                    int(judge_inappropriate),
                    judge_rationale,
                    utc_now(),
                ),
            )

    def positive_trajectory_ids(self, category: str) -> List[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT t.id
                FROM trajectories t
                JOIN attempts a ON a.trajectory_id = t.id
                WHERE t.category = ?
                  AND t.status = 'completed'
                  AND a.judge_inappropriate = 1
                ORDER BY t.ordinal
                """,
                (category,),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def selected_trajectory_ids(self, category: Optional[str] = None) -> List[str]:
        query = """
            SELECT id FROM trajectories
            WHERE selected_phase2 = 1
        """
        parameters: Sequence[Any] = ()
        if category is not None:
            query += " AND category = ?"
            parameters = (category,)
        query += " ORDER BY category, selection_rank, ordinal"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [str(row["id"]) for row in rows]

    def freeze_selection(self, category: str, trajectory_ids: Sequence[str]) -> None:
        self.freeze_selections({category: trajectory_ids})

    def freeze_selections(
        self, selections: Mapping[str, Sequence[str]]
    ) -> None:
        with self.connect() as connection:
            for category, trajectory_ids in selections.items():
                existing = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM trajectories
                    WHERE category = ? AND selected_phase2 = 1
                    """,
                    (category,),
                ).fetchone()
                if existing is not None and int(existing["count"]) > 0:
                    raise ValueError(
                        f"Phase II selection is already frozen for category {category}"
                    )
                for trajectory_id in trajectory_ids:
                    eligible = connection.execute(
                        """
                        SELECT 1
                        FROM trajectories t
                        WHERE t.id = ? AND t.category = ? AND t.status = 'completed'
                          AND EXISTS (
                              SELECT 1 FROM attempts a
                              WHERE a.trajectory_id = t.id
                                AND a.judge_inappropriate = 1
                          )
                        """,
                        (trajectory_id, category),
                    ).fetchone()
                    if eligible is None:
                        raise ValueError(
                            f"trajectory {trajectory_id!r} is not an eligible "
                            f"judge-positive trajectory for {category!r}"
                        )
            for category, trajectory_ids in selections.items():
                for rank, trajectory_id in enumerate(trajectory_ids, start=1):
                    connection.execute(
                        """
                        UPDATE trajectories
                        SET selected_phase2 = 1, selection_rank = ?
                        WHERE id = ? AND category = ?
                        """,
                        (rank, trajectory_id, category),
                    )

    def selected_manifest_rows(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                    t.id AS trajectory_id,
                    t.category,
                    t.ordinal,
                    t.age_band,
                    t.selection_rank,
                    a.mutation_index,
                    a.prompt,
                    a.judge_inappropriate AS phase1_judge_inappropriate
                FROM trajectories t
                JOIN attempts a ON a.trajectory_id = t.id
                WHERE t.selected_phase2 = 1
                ORDER BY t.category, t.selection_rank, a.mutation_index
                """
            ).fetchall()

    def upsert_replay(self, observation: ReplayObservation) -> None:
        with self.connect() as connection:
            existing_prompt = connection.execute(
                """
                SELECT replay_prompt
                FROM replays
                WHERE trajectory_id = ? AND mutation_index = ? AND window_id <> ?
                LIMIT 1
                """,
                (
                    observation.trajectory_id,
                    observation.mutation_index,
                    observation.window_id,
                ),
            ).fetchone()
            if (
                existing_prompt is not None
                and str(existing_prompt["replay_prompt"]) != observation.replay_prompt
            ):
                raise ValueError(
                    "replay prompt changed across windows for "
                    f"{observation.trajectory_id} mutation {observation.mutation_index}"
                )
            connection.execute(
                """
                INSERT INTO replays(
                    trajectory_id, window_id, mutation_index, replay_prompt,
                    response, judge_inappropriate, judge_rationale,
                    visible_warning, safe_rewrite, refusal, notification,
                    notification_latency_minutes, notes, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trajectory_id, window_id, mutation_index) DO UPDATE SET
                    replay_prompt = excluded.replay_prompt,
                    response = excluded.response,
                    judge_inappropriate = excluded.judge_inappropriate,
                    judge_rationale = excluded.judge_rationale,
                    visible_warning = excluded.visible_warning,
                    safe_rewrite = excluded.safe_rewrite,
                    refusal = excluded.refusal,
                    notification = excluded.notification,
                    notification_latency_minutes = excluded.notification_latency_minutes,
                    notes = excluded.notes,
                    observed_at = excluded.observed_at
                """,
                (
                    observation.trajectory_id,
                    observation.window_id,
                    observation.mutation_index,
                    observation.replay_prompt,
                    observation.response,
                    (
                        None
                        if observation.judge_inappropriate is None
                        else int(observation.judge_inappropriate)
                    ),
                    observation.judge_rationale,
                    int(observation.visible_warning),
                    int(observation.safe_rewrite),
                    int(observation.refusal),
                    int(observation.notification),
                    observation.notification_latency_minutes,
                    observation.notes,
                    observation.observed_at,
                ),
            )

    def update_replay_judge(
        self,
        replay_id: int,
        judge_inappropriate: bool,
        judge_rationale: str,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE replays
                SET judge_inappropriate = ?, judge_rationale = ?
                WHERE id = ?
                """,
                (int(judge_inappropriate), judge_rationale, replay_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"unknown replay row {replay_id}")

    def import_benign_probe(
        self,
        probe_id: str,
        category: str,
        ordinal: int,
        age_band: Optional[str],
        prompt: str,
        prevalidated_by: str,
        notes: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO benign_probes(
                    id, category, ordinal, age_band, prompt, prevalidated_by, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    category = excluded.category,
                    ordinal = excluded.ordinal,
                    age_band = excluded.age_band,
                    prompt = excluded.prompt,
                    prevalidated_by = excluded.prevalidated_by,
                    notes = excluded.notes
                """,
                (
                    probe_id,
                    category,
                    ordinal,
                    age_band,
                    prompt,
                    prevalidated_by,
                    notes,
                ),
            )

    def benign_manifest_rows(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT id AS probe_id, category, ordinal, age_band, prompt,
                       prevalidated_by, notes
                FROM benign_probes
                ORDER BY category, ordinal
                """
            ).fetchall()

    def upsert_benign_replay(self, observation: BenignObservation) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO benign_replays(
                    probe_id, window_id, response, refusal, heavy_sanitization,
                    notes, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(probe_id, window_id) DO UPDATE SET
                    response = excluded.response,
                    refusal = excluded.refusal,
                    heavy_sanitization = excluded.heavy_sanitization,
                    notes = excluded.notes,
                    observed_at = excluded.observed_at
                """,
                (
                    observation.probe_id,
                    observation.window_id,
                    observation.response,
                    int(observation.refusal),
                    int(observation.heavy_sanitization),
                    observation.notes,
                    observation.observed_at,
                ),
            )

    def rows(self, query: str, parameters: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(query, parameters).fetchall()
