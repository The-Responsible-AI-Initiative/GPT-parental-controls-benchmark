from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class JudgeDecision:
    inappropriate: bool
    rationale: str


@dataclass(frozen=True)
class MetricEstimate:
    positive: int
    total: int
    rate: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "positive": self.positive,
            "total": self.total,
            "rate": self.rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }

    def paper_cell(self) -> str:
        return (
            f"{self.positive}/{self.total} ({self.rate * 100:.0f}%) "
            f"[{self.ci_low * 100:.0f}-{self.ci_high * 100:.0f}]"
        )


@dataclass(frozen=True)
class ReplayObservation:
    trajectory_id: str
    window_id: str
    mutation_index: int
    replay_prompt: str
    response: str
    judge_inappropriate: Optional[bool]
    judge_rationale: str
    visible_warning: bool
    safe_rewrite: bool
    refusal: bool
    notification: bool
    notification_latency_minutes: Optional[float]
    notes: str
    observed_at: str


@dataclass(frozen=True)
class BenignObservation:
    probe_id: str
    window_id: str
    response: str
    refusal: bool
    heavy_sanitization: bool
    notes: str
    observed_at: str
