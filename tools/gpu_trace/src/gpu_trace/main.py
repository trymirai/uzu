from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractors import (
    compute_kernel_counter_stats,
    compute_kernel_summary,
    extract_counter_definitions,
    extract_counter_samples,
    extract_gpu_utilization,
    extract_kernel_dispatches,
    extract_metadata,
    extract_performance_states,
    extract_shaders,
)
from .formatting import print_comparison, print_summary
from .models import TraceExport
from .progress import status
from .recorder import default_output_path, record_trace
from .serialization import to_json


def export_trace(trace: Path, run_number: int) -> TraceExport:
    with status("Extracting trace metadata..."):
        metadata = extract_metadata(trace, run_number)

    with status("Analyzing GPU utilization..."):
        utilization = extract_gpu_utilization(trace, run_number)

    with status("Reading performance states..."):
        performance_states = extract_performance_states(trace, run_number)

    with status("Loading counter definitions..."):
        counter_definitions = extract_counter_definitions(trace, run_number)

    with status("Extracting kernel dispatches..."):
        dispatches = extract_kernel_dispatches(trace, run_number)

    with status("Loading shader info..."):
        shaders = extract_shaders(trace, run_number)

    with status("Computing kernel statistics..."):
        kernel_summary = compute_kernel_summary(dispatches)

    with status("Extracting counter samples..."):
        counter_samples = extract_counter_samples(trace, run_number)

    with status("Correlating counters with kernels..."):
        kernel_counter_stats = compute_kernel_counter_stats(
            dispatches, counter_samples, counter_definitions
        )

    return TraceExport(
        metadata=metadata,
        utilization=utilization,
        performance_states=performance_states,
        counter_definitions=counter_definitions,
        dispatches=dispatches,
        shaders=shaders,
        kernel_summary=kernel_summary,
        counter_samples=counter_samples,
        kernel_counter_stats=kernel_counter_stats,
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
    print_summary(export, verbose=args.verbose)

    json_path = args.json if args.json else output.with_suffix(".json")
    json_path.write_text(to_json(export, indent=2))
    print(f"JSON exported: {json_path}", file=sys.stderr)

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    if not args.trace.exists():
        print(f"Error: Trace not found: {args.trace}", file=sys.stderr)
        return 1

    export = export_trace(args.trace, args.run)
    print_summary(export, verbose=args.verbose)

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


MAIN_DESCRIPTION = """
Metal GPU Trace Tool - Profile GPU performance on macOS.

Record, analyze, and compare Metal GPU traces with per-kernel timing,
hardware counter data (ALU, memory bandwidth, cache utilization),
and performance bottleneck analysis.

Requires Xcode and xctrace to be installed.
""".strip()

MAIN_EPILOG = """
Quick Start:
  1. Record a trace:     gpu_trace run -- ./my_metal_app
  2. Analyze results:    gpu_trace analyze /tmp/gpu_trace_*.trace
  3. Compare traces:     gpu_trace compare before.trace after.trace

For detailed per-kernel hardware counters, use --gpu-counters:
  gpu_trace run --gpu-counters -- ./my_metal_app

Export to JSON for further analysis:
  gpu_trace analyze /tmp/my.trace --json output.json

Use -v/--verbose to show all hardware counters per kernel.
"""

RUN_DESCRIPTION = """
Record a Metal GPU trace using Xcode Instruments.

Launches the specified command and records GPU activity until the process
exits or Ctrl-C is pressed. Automatically displays a summary of kernel
timing and GPU utilization.
""".strip()

RUN_EPILOG = """
Examples:
  # Basic recording
  gpu_trace run -- ./my_app arg1 arg2

  # With hardware performance counters (ALU, memory, cache)
  gpu_trace run --gpu-counters -- ./my_app

  # Attach to running process with time limit
  gpu_trace run --attach 12345 --time-limit 5s

  # Save JSON export alongside trace
  gpu_trace run --json results.json -- ./my_app

  # Custom output path
  gpu_trace run -o /path/to/my.trace -- ./my_app

  # Verbose output with all counters
  gpu_trace run -v --gpu-counters -- ./my_app
"""

ANALYZE_DESCRIPTION = """
Analyze an existing .trace bundle created by gpu_trace or Xcode Instruments.

Displays GPU utilization, per-kernel timing, and hardware counter data
if available. Use --json to export full data for further analysis.
""".strip()

ANALYZE_EPILOG = """
Examples:
  # Basic analysis
  gpu_trace analyze /tmp/my.trace

  # Export to JSON file
  gpu_trace analyze /tmp/my.trace --json output.json

  # JSON to stdout (for piping)
  gpu_trace analyze /tmp/my.trace --json -

  # Compact JSON (no indentation)
  gpu_trace analyze /tmp/my.trace --json output.json --compact

  # Verbose output with all counters
  gpu_trace analyze /tmp/my.trace -v
"""

COMPARE_DESCRIPTION = """
Compare two GPU traces side-by-side.

Shows differences in GPU utilization, kernel timing, and performance
metrics. Useful for measuring the impact of optimizations.
""".strip()

COMPARE_EPILOG = """
Examples:
  # Basic comparison
  gpu_trace compare baseline.trace optimized.trace

  # With custom labels
  gpu_trace compare a.trace b.trace --label1 before --label2 after
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu_trace",
        description=MAIN_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=MAIN_EPILOG,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # --- run subcommand ---
    run_p = sub.add_parser(
        "run",
        help="Record a GPU trace while running a command",
        description=RUN_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RUN_EPILOG,
    )
    run_p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output .trace path (default: /tmp/gpu_trace_<timestamp>.trace)",
    )
    run_p.add_argument(
        "--template",
        default="Metal System Trace",
        metavar="NAME",
        help="Xcode Instruments template (default: Metal System Trace)",
    )
    run_p.add_argument(
        "--gpu-counters",
        action="store_true",
        help="Enable Metal GPU Counters for detailed hardware metrics",
    )
    run_p.add_argument(
        "--time-limit",
        default=None,
        metavar="DURATION",
        help="Recording time limit, e.g. 5s, 500ms (default: until exit)",
    )
    run_p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="Set environment variable for the launched process",
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
        metavar="PATH",
        help="Export trace data to JSON file",
    )
    run_p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all hardware counters per kernel",
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
        description=ANALYZE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=ANALYZE_EPILOG,
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
        metavar="N",
        help="Run number to analyze (default: 1)",
    )
    analyze_p.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Export trace data to JSON file (use - for stdout)",
    )
    analyze_p.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation)",
    )
    analyze_p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all hardware counters per kernel",
    )

    # --- compare subcommand ---
    compare_p = sub.add_parser(
        "compare",
        help="Compare two traces side-by-side",
        description=COMPARE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=COMPARE_EPILOG,
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
        metavar="N",
        help="Run number to compare (default: 1)",
    )
    compare_p.add_argument(
        "--label1",
        default="baseline",
        metavar="LABEL",
        help="Label for first trace (default: baseline)",
    )
    compare_p.add_argument(
        "--label2",
        default="comparison",
        metavar="LABEL",
        help="Label for second trace (default: comparison)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return cmd_run(args)
    elif args.cmd == "analyze":
        return cmd_analyze(args)
    elif args.cmd == "compare":
        return cmd_compare(args)

    return 1


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
