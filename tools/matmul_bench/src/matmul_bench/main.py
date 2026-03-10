import argparse
import sys
from pathlib import Path

from matmul_bench.charts import generate_charts
from matmul_bench.runner import load_results, run_benchmark


def cmd_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    json_path = run_benchmark(output_dir)
    print(f"Benchmark results: {json_path}")


def cmd_chart(args: argparse.Namespace) -> None:
    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir)

    run = load_results(json_path)
    ok_count = sum(1 for r in run.results if r.status == "ok")
    err_count = len(run.results) - ok_count
    print(f"Loaded {len(run.results)} results ({ok_count} ok, {err_count} errors) from {json_path.name}")
    print(f"Device: {run.device}")

    files = generate_charts(run, output_dir, source_filename=json_path.name)
    print(f"\nGenerated {len(files)} chart(s) in {output_dir}")


def cmd_run_and_chart(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    json_path = run_benchmark(output_dir)

    run = load_results(json_path)
    ok_count = sum(1 for r in run.results if r.status == "ok")
    print(f"Loaded {len(run.results)} results ({ok_count} ok)")

    chart_dir = output_dir / "charts"
    files = generate_charts(run, chart_dir, source_filename=json_path.name)
    print(f"\nGenerated {len(files)} chart(s) in {chart_dir}")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="matmul_bench", description="Matmul kernel benchmark runner and charter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the matmul benchmark")
    run_parser.add_argument("--output-dir", default="results", help="Directory for JSON results")
    run_parser.set_defaults(func=cmd_run)

    chart_parser = subparsers.add_parser("chart", help="Generate charts from existing JSON")
    chart_parser.add_argument("json_path", help="Path to matmul_perf.json")
    chart_parser.add_argument("--output-dir", default="charts", help="Directory for chart PNGs")
    chart_parser.set_defaults(func=cmd_chart)

    rac_parser = subparsers.add_parser("run-and-chart", help="Run benchmark then generate charts")
    rac_parser.add_argument("--output-dir", default="results", help="Directory for JSON + charts/")
    rac_parser.set_defaults(func=cmd_run_and_chart)

    args = parser.parse_args()
    args.func(args)
