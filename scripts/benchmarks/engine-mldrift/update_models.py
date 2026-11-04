#!/usr/bin/env python3

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import upstream, write_catalog


def update():
    source = upstream("mldrift", "ml_drift/samples/llm/README.md")
    models = set(re.findall(r"https://huggingface.co/([^/)]+/[^/)]+)/tree/main", source))
    # The benchmark patch adds the upstream Qwen3 14B preset.
    models.add("Qwen/Qwen3-14B")
    result = {
        f"{model}:{precision}": {"model": model, "precision": precision}
        for model in models
        for precision in ("q4_0", "f16")
    }
    write_catalog(
        Path(__file__).resolve().parent, result, speculation=lambda config: {"specdec": False}, pin_revisions=True
    )


if __name__ == "__main__":
    update()
