import os

SILENCE = "ignore:'prefect_gcp.experimental.bundles.upload' found in sys.modules:RuntimeWarning:runpy"


def silence_experimental_bundles_warning() -> None:
    existing = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = f"{existing},{SILENCE}" if existing else SILENCE
