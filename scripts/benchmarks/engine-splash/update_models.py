#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import catalog, upstream, write_catalog


def update():
    lines = upstream("splash", "install/suggested-models.tsv").splitlines()
    rows = csv.reader((line for line in lines if line and not line.startswith("#")), delimiter="\t")
    # Every Splash model pairs its target with a draft.
    write_catalog(
        Path(__file__).resolve().parent,
        catalog(row[0] for row in rows),
        speculation=lambda config: {"specdec": True},
        pin_revisions=True,
    )


if __name__ == "__main__":
    update()
