from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass
class Curve:
    block_name: str
    metric: str
    numeric_parameters: list[str]
    categorical_parameters: list[str]
    coefficients_by_group: dict[tuple, tuple[float, float, float]]
    r_squared: float

    def predict(self, shape: dict) -> float:
        group_key = tuple(str(shape[column]) for column in self.categorical_parameters)
        overhead, compute_rate, memory_rate = self.coefficients_by_group.get(
            group_key, next(iter(self.coefficients_by_group.values()))
        )
        values = [float(shape[column]) for column in self.numeric_parameters]
        return max(0.0, overhead + compute_rate * compute_feature(values) + memory_rate * memory_feature(values))


def fit_curve(
    measurements: pd.DataFrame,
    parameter_columns: list[str],
    metric: str,
    block_name: str,
) -> Curve:
    numeric = [
        column
        for column in parameter_columns
        if pd.api.types.is_numeric_dtype(measurements[column]) and not pd.api.types.is_bool_dtype(measurements[column])
    ]
    categorical = [column for column in parameter_columns if column not in numeric]

    groups = measurements.groupby(categorical) if categorical else [((), measurements)]
    coefficients_by_group: dict[tuple, tuple[float, float, float]] = {}
    predicted = np.zeros(len(measurements))
    for group_key, group in groups:
        rows = group[numeric].to_numpy(dtype=float) if numeric else np.empty((len(group), 0))
        compute = np.array([compute_feature(row) for row in rows])
        memory = np.array([memory_feature(row) for row in rows])
        target = group[metric].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(compute), compute, memory])
        solution, _ = nnls(design, target)
        overhead, compute_rate, memory_rate = float(solution[0]), float(solution[1]), float(solution[2])
        coefficients_by_group[_as_tuple(group_key)] = (overhead, compute_rate, memory_rate)
        predicted[group.index] = np.maximum(0.0, overhead + compute_rate * compute + memory_rate * memory)

    return Curve(
        block_name=block_name,
        metric=metric,
        numeric_parameters=numeric,
        categorical_parameters=categorical,
        coefficients_by_group=coefficients_by_group,
        r_squared=_r_squared(measurements[metric].to_numpy(dtype=float), predicted),
    )


def compute_feature(values) -> float:
    values = [float(value) for value in values]
    return float(prod(values)) if values else 1.0


def memory_feature(values) -> float:
    values = [float(value) for value in values if float(value) != 0.0]
    if len(values) < 2:
        return 0.0
    total = prod(values)
    return float(sum(total / value for value in values))


def _as_tuple(group_key) -> tuple:
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    return tuple(str(value) for value in values)


def _r_squared(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    residual_sum = float(((actual - predicted) ** 2).sum())
    total_sum = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
