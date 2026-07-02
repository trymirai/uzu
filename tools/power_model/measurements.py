from __future__ import annotations

from pathlib import Path

import pandas as pd

SAMPLE_METRIC_COLUMNS = [
    "cpu_watts",
    "gpu_watts",
    "gpu_sram_watts",
    "ram_watts",
    "dram_read_gigabytes_per_second",
    "dram_write_gigabytes_per_second",
]


def parameter_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.columns[: frame.columns.get_loc("iterations")])


def load_block_measurements(csv_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_csv(csv_path)
    columns = parameter_columns(frame)

    aggregations = {metric: (metric, "mean") for metric in SAMPLE_METRIC_COLUMNS}
    aggregations["gpu_time_per_iteration_microseconds"] = ("gpu_time_per_iteration_microseconds", "first")
    aggregated = frame.groupby(columns, as_index=False).agg(**aggregations)

    aggregated["gpu_power_watts"] = aggregated["gpu_watts"] + aggregated["gpu_sram_watts"]
    aggregated["energy_per_iteration_joules"] = (
        aggregated["gpu_power_watts"] * aggregated["gpu_time_per_iteration_microseconds"] * 1e-6
    )

    metric_columns = [column for column in aggregated.columns if column not in columns]
    return aggregated, columns, metric_columns
