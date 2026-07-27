#!/usr/bin/env python3
"""Merge criterion median estimates for A8W benchmarks into a JSON accumulator.

Usage: a8w_collect.py <criterion-dir> <output.json> [<newer-than-epoch-seconds>]

Criterion overwrites target/criterion in place on every run, so a chunked sweep
has to snapshot after each chunk or earlier chunks are lost.

`newer-than` guards the opposite failure: without it, a chunk that has not run yet
still contributes whatever a *previous* sweep left in target/criterion, and stale
numbers get reported as fresh ones. Always pass it for real measurements.
"""

import json
import os
import sys


def main() -> None:
    criterion_dir, output_path = sys.argv[1], sys.argv[2]
    newer_than = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    try:
        with open(output_path) as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError):
        rows = {}

    for dirpath, _dirnames, filenames in os.walk(criterion_dir):
        if os.path.basename(dirpath) != "new" or "estimates.json" not in filenames:
            continue
        key = os.path.relpath(os.path.dirname(dirpath), criterion_dir)
        if "A8W" not in key:
            continue
        estimates_path = os.path.join(dirpath, "estimates.json")
        if os.path.getmtime(estimates_path) < newer_than:
            continue
        with open(estimates_path) as handle:
            estimates = json.load(handle)
        # Median is more robust than mean against criterion's occasional low outlier.
        rows[key] = estimates["median"]["point_estimate"] / 1000.0  # ns -> us

    with open(output_path, "w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
