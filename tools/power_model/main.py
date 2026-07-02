from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .curve import Curve, fit_curve
from .measurements import load_block_measurements
from .prediction import (
    enumerate_block_instances,
    load_config,
    load_whole_model_measurement,
    predict_total,
)

app = typer.Typer(help="Fit per-block GPU metric curves and predict whole-model energy.")

DEFAULT_METRIC = "energy_per_iteration_joules"
TIME_METRIC = "gpu_time_per_iteration_microseconds"


def load_block_curves(
    measurements_directory: Path,
    metric: str,
) -> dict[str, Curve]:
    curves: dict[str, Curve] = {}
    for csv_path in sorted(measurements_directory.glob("*.csv")):
        if "iterations" not in pd.read_csv(csv_path, nrows=0).columns:
            continue
        measurements, parameter_columns, _ = load_block_measurements(csv_path)
        curves[csv_path.stem] = fit_curve(measurements, parameter_columns, metric, csv_path.stem)
    return curves


@app.command()
def metrics(block_csv: Path) -> None:
    _, _, metric_columns = load_block_measurements(block_csv)
    typer.echo("predictable metrics:")
    for metric in metric_columns:
        typer.echo(f"  {metric}")


@app.command()
def fit(
    block_csv: Path,
    metric: str = DEFAULT_METRIC,
) -> None:
    measurements, parameter_columns, _ = load_block_measurements(block_csv)
    curve = fit_curve(measurements, parameter_columns, metric, block_csv.stem)
    typer.echo(f"block: {curve.block_name}   metric: {curve.metric}   r_squared: {curve.r_squared:.4f}")
    typer.echo(f"compute = product of {curve.numeric_parameters}; memory = sum of leave-one-out products")
    for group_key, (overhead, compute_rate, memory_rate) in curve.coefficients_by_group.items():
        group_label = ", ".join(f"{name}={value}" for name, value in zip(curve.categorical_parameters, group_key)) or "all"
        typer.echo(f"  [{group_label}] {curve.metric} = {overhead:.3e} + {compute_rate:.3e} * compute + {memory_rate:.3e} * memory")


@app.command()
def predict(
    config: Path,
    measurements_directory: Path,
    prefill_tokens: int = 512,
    decode_tokens: int = 128,
    metric: str = DEFAULT_METRIC,
    whole_model_csv: Path | None = None,
) -> None:
    measurement = load_whole_model_measurement(whole_model_csv) if whole_model_csv is not None else None
    if measurement is not None:
        prefill_tokens = measurement["prefill_tokens"]
        decode_tokens = measurement["decode_tokens"]

    config_dict = load_config(config)
    instances = enumerate_block_instances(config_dict, prefill_tokens, decode_tokens)

    curves = load_block_curves(measurements_directory, metric)
    total, breakdown, unmodeled = predict_total(instances, curves)

    typer.echo(f"prefill_tokens={prefill_tokens}   decode_tokens={decode_tokens}")
    typer.echo(f"predicted {metric}: {total:.3e}")
    for block_name, value in sorted(breakdown.items()):
        typer.echo(f"  {block_name}: {value:.3e}")
    if unmodeled:
        typer.echo(f"  unmodeled blocks (no measurements): {unmodeled}")

    if measurement is not None:
        time_curves = load_block_curves(measurements_directory, TIME_METRIC)
        predicted_microseconds, _, _ = predict_total(instances, time_curves)
        predicted_seconds = predicted_microseconds / 1e6
        measured_seconds = measurement["total_seconds"]
        ratio = predicted_seconds / measured_seconds if measured_seconds else float("nan")
        typer.echo("")
        typer.echo(f"measured whole-model wall-clock: {measured_seconds:.3f} s (GPU-bound)")
        typer.echo(f"predicted modeled-kernel GPU execution time: {predicted_seconds:.3f} s")
        typer.echo(f"predicted GPU time / measured wall-clock: {ratio:.3f}")


def cli() -> None:
    app()
