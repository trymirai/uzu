from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractors import (
    compute_kernel_summary,
    extract_counter_definitions,
    extract_gpu_utilization,
    extract_kernel_dispatches,
    extract_metadata,
    extract_performance_states,
    extract_shaders,
)
from .formatting import print_comparison, print_summary
from .models import TraceExport
from .recorder import default_output_path, record_trace
from .serialization import to_json


def export_trace(trace: Path, run_number: int) -> TraceExport:
    metadata = extract_metadata(trace, run_number)
    utilization = extract_gpu_utilization(trace, run_number)
    performance_states = extract_performance_states(trace, run_number)
    counter_definitions = extract_counter_definitions(trace, run_number)
    dispatches = extract_kernel_dispatches(trace, run_number)
    shaders = extract_shaders(trace, run_number)
    kernel_summary = compute_kernel_summary(dispatches)

    return TraceExport(
        metadata=metadata,
        utilization=utilization,
        performance_states=performance_states,
        counter_definitions=counter_definitions,
        dispatches=dispatches,
        shaders=shaders,
        kernel_summary=kernel_summary,
    )


def cmd_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]

    if not command and args.attach is None:
        print("Error: Either command or --attach must be specified", file=sys.stderr)
        return 1

    output = args.output or default_output_path()

    record_rc = record_trace(
        output=output,
        command=command,
        template=args.template,
        gpu_counters=args.gpu_counters,
        time_limit=args.time_limit,
        env=args.env,
        attach_pid=args.attach,
    )

    if record_rc != 0:
        print(f"xctrace record failed (exit={record_rc})", file=sys.stderr)
        return record_rc

    print(f"Trace saved: {output}", file=sys.stderr)

    export = export_trace(output, 1)
    print_summary(export)

    if args.json:
        json_path = args.json
        json_path.write_text(to_json(export, indent=2))
        print(f"JSON exported: {json_path}", file=sys.stderr)

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    if not args.trace.exists():
        print(f"Error: Trace not found: {args.trace}", file=sys.stderr)
        return 1

    export = export_trace(args.trace, args.run)
    print_summary(export)

    if args.json:
        indent = None if args.compact else 2
        json_output = to_json(export, indent=indent)
        if args.json == Path("-"):
            print(json_output)
        else:
            args.json.write_text(json_output)
            print(f"JSON exported: {args.json}", file=sys.stderr)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if not args.trace1.exists():
        print(f"Error: Trace not found: {args.trace1}", file=sys.stderr)
        return 1
    if not args.trace2.exists():
        print(f"Error: Trace not found: {args.trace2}", file=sys.stderr)
        return 1

    export1 = export_trace(args.trace1, args.run)
    export2 = export_trace(args.trace2, args.run)

    print_comparison(export1, export2, args.label1, args.label2)

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu_trace",
        description="Record, analyze, and export Metal GPU traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run subcommand ---
    run_p = sub.add_parser(
        "run",
        help="Record a GPU trace while running a command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpu_trace run -- ./my_app arg1 arg2
  gpu_trace run --gpu-counters -- ./my_app
  gpu_trace run --attach 12345 --time-limit 5s
  gpu_trace run --json trace.json -- ./my_app
        """,
    )
    run_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .trace path (default: /tmp/gpu_trace_<timestamp>.trace)",
    )
    run_p.add_argument(
        "--template",
        default="Metal System Trace",
        help="xctrace template (default: Metal System Trace)",
    )
    run_p.add_argument(
        "--gpu-counters",
        action="store_true",
        help="Enable Metal GPU Counters for per-kernel timing",
    )
    run_p.add_argument(
        "--time-limit",
        default=None,
        help="Recording time limit (e.g., 5s, 500ms)",
    )
    run_p.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable (VAR=value)",
    )
    run_p.add_argument(
        "--attach",
        type=int,
        default=None,
        metavar="PID",
        help="Attach to running process instead of launching",
    )
    run_p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also export JSON to file",
    )
    run_p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (use -- before command)",
    )

    # --- analyze subcommand ---
    analyze_p = sub.add_parser(
        "analyze",
        help="Analyze an existing .trace bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpu_trace analyze /tmp/my.trace
  gpu_trace analyze /tmp/my.trace --json output.json
  gpu_trace analyze /tmp/my.trace --json -  # JSON to stdout
        """,
    )
    analyze_p.add_argument(
        "trace",
        type=Path,
        help="Path to .trace bundle",
    )
    analyze_p.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number to analyze (default: 1)",
    )
    analyze_p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export JSON to file (use - for stdout)",
    )
    analyze_p.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output",
    )

    # --- compare subcommand ---
    compare_p = sub.add_parser(
        "compare",
        help="Compare two traces side-by-side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpu_trace compare baseline.trace optimized.trace
  gpu_trace compare a.trace b.trace --label1 before --label2 after
        """,
    )
    compare_p.add_argument(
        "trace1",
        type=Path,
        help="First trace (baseline)",
    )
    compare_p.add_argument(
        "trace2",
        type=Path,
        help="Second trace (comparison)",
    )
    compare_p.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number to compare (default: 1)",
    )
    compare_p.add_argument(
        "--label1",
        default="baseline",
        help="Label for first trace",
    )
    compare_p.add_argument(
        "--label2",
        default="comparison",
        help="Label for second trace",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return cmd_run(args)
    elif args.cmd == "analyze":
        return cmd_analyze(args)
    elif args.cmd == "compare":
        return cmd_compare(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
