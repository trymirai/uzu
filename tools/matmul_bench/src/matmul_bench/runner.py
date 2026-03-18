import json
import os
import subprocess
from pathlib import Path

import cattrs

from matmul_bench.models import BenchmarkRun, PerfResult


def _find_workspace_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Cargo.toml").exists() and (parent / "crates").exists():
            return parent
    raise RuntimeError("Could not find workspace root (no Cargo.toml with crates/ found)")


def _structure_benchmark_run(data: dict) -> BenchmarkRun:
    converter = cattrs.Converter()
    converter.register_structure_hook(
        BenchmarkRun,
        lambda d, _: BenchmarkRun(
            device=d["device"],
            results=tuple(converter.structure(r, PerfResult) for r in d["results"]),
        ),
    )
    return converter.structure(data, BenchmarkRun)


def load_results(json_path: Path | str) -> BenchmarkRun:
    with open(json_path) as f:
        data = json.load(f)
    return _structure_benchmark_run(data)


def run_benchmark(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = _find_workspace_root()

    env = {**os.environ, "UZU_TEST_RESULTS_DIR": str(output_dir)}

    cmd = [
        "cargo", "test",
        "--release",
        "-p", "uzu",
        "--test", "matmul_perf_test",
        "--",
        "--ignored",
        "--nocapture",
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Results will be written to: {output_dir}")

    result = subprocess.run(cmd, cwd=workspace, env=env, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with exit code {result.returncode}")

    json_path = output_dir / "matmul_perf.json"
    if not json_path.exists():
        raise RuntimeError(f"Expected results file not found: {json_path}")

    return json_path
