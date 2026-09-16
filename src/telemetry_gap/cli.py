from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .analysis import (
    analyze_window,
    phase1_base_rates,
    render_json,
    render_markdown,
    render_phase1_base_rates,
    write_csv,
)
from .audit import (
    export_calibration_sample,
    export_targeted_audit,
    render_audit_score,
    score_audit,
)
from .config import StudyConfig, load_config
from .db import Database
from .phase1 import Phase1Runner, freeze_phase2_selection
from .providers import create_provider
from .replay import (
    export_benign_replay_sheet,
    export_replay_sheet,
    import_benign_probes,
    import_benign_replay_sheet,
    import_replay_sheet,
    judge_unlabeled_replays,
)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, help="Path to the study JSON configuration"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="telemetry-gap",
        description="Parental telemetry-gap experiment harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Initialize the study database")
    _add_config(init)

    phase1 = subparsers.add_parser(
        "phase1-run", help="Generate and evaluate Phase-I trajectories"
    )
    _add_config(phase1)
    phase1.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Run one configured category; repeat for multiple categories",
    )
    phase1.add_argument(
        "--limit-per-category",
        type=int,
        help="Run a smaller prefix for a pilot; cannot exceed configured N",
    )

    select = subparsers.add_parser(
        "phase1-select", help="Freeze the random Phase-II success sample"
    )
    _add_config(select)

    base_rates = subparsers.add_parser(
        "phase1-rates", help="Report Phase-I success base rates"
    )
    _add_config(base_rates)
    base_rates.add_argument("--output", help="Optional Markdown output path")

    replay_export = subparsers.add_parser(
        "replay-export", help="Export a human UI-replay worksheet"
    )
    _add_config(replay_export)
    replay_export.add_argument("--output", required=True)

    replay_import = subparsers.add_parser(
        "replay-import", help="Import completed UI-replay observations"
    )
    _add_config(replay_import)
    replay_import.add_argument("--input", required=True)

    replay_judge = subparsers.add_parser(
        "replay-judge", help="Judge imported replay responses lacking labels"
    )
    _add_config(replay_judge)
    replay_judge.add_argument("--window", help="Limit judging to one window")

    benign_import = subparsers.add_parser(
        "benign-import", help="Import the pre-validated benign probe corpus"
    )
    _add_config(benign_import)
    benign_import.add_argument("--input", required=True)

    benign_export = subparsers.add_parser(
        "benign-export", help="Export a benign-probe replay worksheet"
    )
    _add_config(benign_export)
    benign_export.add_argument("--output", required=True)

    benign_results = subparsers.add_parser(
        "benign-results-import", help="Import benign-probe replay outcomes"
    )
    _add_config(benign_results)
    benign_results.add_argument("--input", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="Compute one window's session-level metrics"
    )
    _add_config(analyze)
    analyze.add_argument("--window", required=True)
    analyze.add_argument(
        "--format", choices=("markdown", "json", "csv"), default="markdown"
    )
    analyze.add_argument("--output", help="Output path; stdout if omitted")

    calibration = subparsers.add_parser(
        "audit-calibration-export", help="Export a stratified calibration sample"
    )
    _add_config(calibration)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--sample-size", type=int, default=142)

    targeted = subparsers.add_parser(
        "audit-targeted-export", help="Export all positives plus sampled negatives"
    )
    _add_config(targeted)
    targeted.add_argument("--output", required=True)
    targeted.add_argument("--negative-fraction", type=float, default=0.10)

    audit_score = subparsers.add_parser(
        "audit-score", help="Compute judge and inter-human Cohen kappa"
    )
    audit_score.add_argument("--input", required=True)
    audit_score.add_argument("--output")

    return parser


def _load_study(path: str, require_database: bool = True) -> tuple[StudyConfig, Database]:
    config = load_config(path)
    database = Database(config.database_path)
    if require_database and not database.path.exists():
        raise FileNotFoundError(
            f"study database does not exist: {database.path}; run init first"
        )
    return config, database


def _emit(text: str, output: Optional[str]) -> None:
    if output:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def run_command(args: argparse.Namespace) -> None:
    if args.command == "init":
        config, database = _load_study(args.config, require_database=False)
        database.initialize(config.as_snapshot())
        print(f"initialized {database.path}")
        return

    if args.command == "audit-score":
        score = score_audit(Path(args.input).expanduser().resolve())
        _emit(render_audit_score(score), args.output)
        return

    config, database = _load_study(args.config)

    if args.command == "phase1-run":
        runner = Phase1Runner(
            config,
            database,
            create_provider(config.prompt_engineer),
            create_provider(config.target),
            create_provider(config.judge),
        )
        summary = runner.run(args.categories, args.limit_per_category)
        print(
            f"attempted={summary.attempted} completed={summary.completed} "
            f"skipped={summary.skipped} failed={summary.failed}"
        )
    elif args.command == "phase1-select":
        freeze_phase2_selection(config, database)
        print("Phase II selection frozen")
    elif args.command == "phase1-rates":
        _emit(render_phase1_base_rates(phase1_base_rates(config, database)), args.output)
    elif args.command == "replay-export":
        export_replay_sheet(config, database, Path(args.output).expanduser().resolve())
        print(f"wrote {Path(args.output).expanduser().resolve()}")
    elif args.command == "replay-import":
        count = import_replay_sheet(
            config, database, Path(args.input).expanduser().resolve()
        )
        print(f"imported {count} replay observations")
    elif args.command == "replay-judge":
        count = judge_unlabeled_replays(
            database, create_provider(config.judge), args.window
        )
        print(f"judged {count} replay responses")
    elif args.command == "benign-import":
        count = import_benign_probes(
            config, database, Path(args.input).expanduser().resolve()
        )
        print(f"imported {count} benign probes")
    elif args.command == "benign-export":
        export_benign_replay_sheet(
            config, database, Path(args.output).expanduser().resolve()
        )
        print(f"wrote {Path(args.output).expanduser().resolve()}")
    elif args.command == "benign-results-import":
        count = import_benign_replay_sheet(
            config, database, Path(args.input).expanduser().resolve()
        )
        print(f"imported {count} benign observations")
    elif args.command == "analyze":
        results = analyze_window(config, database, args.window)
        if args.format == "markdown":
            _emit(render_markdown(args.window, results), args.output)
        elif args.format == "json":
            _emit(render_json(args.window, results), args.output)
        else:
            if not args.output:
                raise ValueError("--output is required for CSV format")
            write_csv(
                Path(args.output).expanduser().resolve(), args.window, results
            )
            print(f"wrote {Path(args.output).expanduser().resolve()}")
    elif args.command == "audit-calibration-export":
        count = export_calibration_sample(
            database,
            Path(args.output).expanduser().resolve(),
            args.sample_size,
            config.random_seed,
        )
        print(f"exported {count} calibration items")
    elif args.command == "audit-targeted-export":
        count = export_targeted_audit(
            database,
            Path(args.output).expanduser().resolve(),
            args.negative_fraction,
            config.random_seed,
        )
        print(f"exported {count} targeted audit items")
    else:
        raise AssertionError(f"unhandled command {args.command}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_command(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
