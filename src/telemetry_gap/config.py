from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class WindowConfig:
    id: str
    date: str
    model_picker_label: str
    notes: str = ""


@dataclass(frozen=True)
class ProviderConfig:
    kind: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = 120.0
    max_retries: int = 3
    temperature: Optional[float] = 0.0
    responses_path: Optional[Path] = None


@dataclass(frozen=True)
class StudyConfig:
    config_path: Path
    database_path: Path
    random_seed: int
    categories: Tuple[str, ...]
    age_bands: Tuple[str, ...]
    phase1_trajectories_per_category: int
    max_mutations: int
    phase2_sample_per_category: int
    benign_probes_per_category: int
    benign_probe_schedule: str
    windows: Tuple[WindowConfig, ...]
    prompt_engineer: ProviderConfig
    target: ProviderConfig
    judge: ProviderConfig

    @property
    def window_ids(self) -> Tuple[str, ...]:
        return tuple(window.id for window in self.windows)

    def as_snapshot(self) -> Dict[str, Any]:
        return {
            "database_path": str(self.database_path),
            "random_seed": self.random_seed,
            "categories": list(self.categories),
            "age_bands": list(self.age_bands),
            "phase1_trajectories_per_category": self.phase1_trajectories_per_category,
            "max_mutations": self.max_mutations,
            "phase2_sample_per_category": self.phase2_sample_per_category,
            "benign_probes_per_category": self.benign_probes_per_category,
            "benign_probe_schedule": self.benign_probe_schedule,
            "windows": [
                {
                    "id": window.id,
                    "date": window.date,
                    "model_picker_label": window.model_picker_label,
                    "notes": window.notes,
                }
                for window in self.windows
            ],
            "providers": {
                "prompt_engineer": _provider_snapshot(self.prompt_engineer),
                "target": _provider_snapshot(self.target),
                "judge": _provider_snapshot(self.judge),
            },
        }


def _provider_snapshot(provider: ProviderConfig) -> Dict[str, Any]:
    return {
        "kind": provider.kind,
        "model": provider.model,
        "base_url": provider.base_url,
        "api_key_env": provider.api_key_env,
        "timeout_seconds": provider.timeout_seconds,
        "max_retries": provider.max_retries,
        "temperature": provider.temperature,
        "responses_path": (
            str(provider.responses_path) if provider.responses_path is not None else None
        ),
    }


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _require_nonempty_strings(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ConfigError(f"{field} cannot contain empty values")
    if len(set(result)) != len(result):
        raise ConfigError(f"{field} cannot contain duplicates")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_provider(
    raw: Mapping[str, Any], field: str, base: Path
) -> ProviderConfig:
    kind = str(raw.get("kind", "")).strip().lower()
    if kind not in {"openai_compatible", "anthropic", "scripted"}:
        raise ConfigError(
            f"{field}.kind must be openai_compatible, anthropic, or scripted"
        )

    model = str(raw.get("model", "")).strip()
    if kind != "scripted" and not model:
        raise ConfigError(f"{field}.model is required")

    default_url = {
        "openai_compatible": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "scripted": "",
    }[kind]
    base_url = str(raw.get("base_url", default_url)).strip()
    api_key_env = str(raw.get("api_key_env", "")).strip()
    if kind != "scripted" and not api_key_env:
        raise ConfigError(f"{field}.api_key_env is required")

    responses_path: Optional[Path] = None
    if kind == "scripted":
        responses_value = str(raw.get("responses_path", "")).strip()
        if not responses_value:
            raise ConfigError(f"{field}.responses_path is required for scripted providers")
        responses_path = _resolve_path(base, responses_value)

    timeout_seconds = float(raw.get("timeout_seconds", 120.0))
    max_retries = int(raw.get("max_retries", 3))
    temperature_raw = raw.get("temperature", 0.0)
    temperature = None if temperature_raw is None else float(temperature_raw)
    if timeout_seconds <= 0:
        raise ConfigError(f"{field}.timeout_seconds must be positive")
    if max_retries < 0:
        raise ConfigError(f"{field}.max_retries cannot be negative")

    return ProviderConfig(
        kind=kind,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        temperature=temperature,
        responses_path=responses_path,
    )


def load_config(path: str) -> StudyConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc

    root = _require_mapping(raw, "configuration")
    study = _require_mapping(root.get("study"), "study")
    providers = _require_mapping(root.get("providers"), "providers")
    base = config_path.parent

    categories = _require_nonempty_strings(study.get("categories"), "study.categories")
    age_bands_raw = study.get("age_bands", [])
    if not isinstance(age_bands_raw, list):
        raise ConfigError("study.age_bands must be an array")
    age_bands = tuple(str(item).strip() for item in age_bands_raw)
    if any(not item for item in age_bands) or len(set(age_bands)) != len(age_bands):
        raise ConfigError("study.age_bands must contain unique, non-empty strings")

    windows_raw = study.get("windows")
    if not isinstance(windows_raw, list) or not windows_raw:
        raise ConfigError("study.windows must be a non-empty array")
    windows = []
    for index, item in enumerate(windows_raw):
        window = _require_mapping(item, f"study.windows[{index}]")
        window_id = str(window.get("id", "")).strip()
        date = str(window.get("date", "")).strip()
        picker = str(window.get("model_picker_label", "")).strip()
        if not window_id or not date or not picker:
            raise ConfigError(
                f"study.windows[{index}] requires id, date, and model_picker_label"
            )
        windows.append(
            WindowConfig(
                id=window_id,
                date=date,
                model_picker_label=picker,
                notes=str(window.get("notes", "")).strip(),
            )
        )
    if len({window.id for window in windows}) != len(windows):
        raise ConfigError("study.windows IDs must be unique")

    schedule = str(study.get("benign_probe_schedule", "")).strip().lower()
    if schedule not in {"once", "every_window"}:
        raise ConfigError(
            "study.benign_probe_schedule must be 'once' or 'every_window'"
        )

    return StudyConfig(
        config_path=config_path,
        database_path=_resolve_path(
            base, str(study.get("database_path", "study.sqlite3"))
        ),
        random_seed=int(study.get("random_seed", 20251017)),
        categories=categories,
        age_bands=age_bands,
        phase1_trajectories_per_category=_positive_int(
            study.get("phase1_trajectories_per_category"),
            "study.phase1_trajectories_per_category",
        ),
        max_mutations=_positive_int(
            study.get("max_mutations"), "study.max_mutations"
        ),
        phase2_sample_per_category=_positive_int(
            study.get("phase2_sample_per_category"),
            "study.phase2_sample_per_category",
        ),
        benign_probes_per_category=_positive_int(
            study.get("benign_probes_per_category"),
            "study.benign_probes_per_category",
        ),
        benign_probe_schedule=schedule,
        windows=tuple(windows),
        prompt_engineer=_load_provider(
            _require_mapping(providers.get("prompt_engineer"), "providers.prompt_engineer"),
            "providers.prompt_engineer",
            base,
        ),
        target=_load_provider(
            _require_mapping(providers.get("target"), "providers.target"),
            "providers.target",
            base,
        ),
        judge=_load_provider(
            _require_mapping(providers.get("judge"), "providers.judge"),
            "providers.judge",
            base,
        ),
    )
