#!/usr/bin/env python3

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import catalog, fetch, write_catalog


def update():
    directory = Path(__file__).resolve().parent
    project = (directory / "pyproject.toml").read_text()
    version = re.findall(r"mtplx==([^\"]+)", project)[0]
    with fetch(f"https://raw.githubusercontent.com/youssofal/MTPLX/v{version}/mtplx/model_catalog.py") as response:
        source = response.read().decode()
    entries = source.split("CatalogModel(\n")[1:]
    models = {re.findall(r'hf_model_id="([^"]+)"', entry)[0]: "ar_only=True" not in entry for entry in entries}
    # MTPLX decodes with a model's MTP head unless the catalog marks the model as lacking one.
    write_catalog(
        directory, catalog(models), speculation=lambda config: {"specdec": models[config["model"]]}, pin_revisions=True
    )


if __name__ == "__main__":
    update()
