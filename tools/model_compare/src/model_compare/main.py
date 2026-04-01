from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .comparison import build_comparison, load_trace_profile
from .formatting import print_comparison_table, print_counter_detail
from .runner import RunConfig, record_all


def cmd_compare(args: argparse.Namespace) -> int:
    console = Console()

    json_paths: list[Path] = args.traces
    labels: list[str] = args.labels.split(",") if args.labels else [p.stem for p in json_paths]

    if len(labels) != len(json_paths):
        console.print(f"[red]Error: {len(labels)} labels for {len(json_paths)} traces[/red]")
        return 1

    profiles = []
    for path, label in zip(json_paths, labels):
        if not path.exists():
            console.print(f"[red]Error: {path} not found[/red]")
            return 1
        profiles.append(load_trace_profile(path, label))

    rows = build_comparison(profiles)
    print_comparison_table(profiles, rows, console, top_n=args.top)

    if args.verbose:
        print_counter_detail(profiles, rows, console, top_n=args.top)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    console = Console()

    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Error: Config not found: {config_path}[/red]")
        return 1

    config = RunConfig.from_file(config_path)
    output_dir = Path(args.output) if args.output else None

    profiles = record_all(config, output_dir=output_dir, console=console)
    rows = build_comparison(profiles)

    console.print()
    print_comparison_table(profiles, rows, console, top_n=args.top)

    if args.verbose:
        print_counter_detail(profiles, rows, console, top_n=args.top)

    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="model_compare",
        description="Compare GPU hardware counters across model variants",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare pre-recorded gpu_trace JSON files",
    )
    compare_parser.add_argument(
        "traces",
        type=Path,
        nargs="+",
        help="gpu_trace JSON export files to compare",
    )
    compare_parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels for each trace (default: filename stems)",
    )
    compare_parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top kernels to show (default: 15)",
    )
    compare_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed per-kernel counter breakdown",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Record traces for multiple models and compare",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to comparison config JSON",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for traces (default: temp dir)",
    )
    run_parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top kernels to show (default: 15)",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed per-kernel counter breakdown",
    )

    args = parser.parse_args()

    if args.command == "compare":
        sys.exit(cmd_compare(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
