import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated

import typer
from bench import ModelConfig
from common import run_loop

from .engine import ServerEngine, ServerEngineType
from .engine.magnitude import MagnitudeConfig, MagnitudeServerEngine
from .engine.splash import SplashConfig, SplashServerEngine
from .engine.tensorfold import TensorFoldConfig, TensorFoldServerEngine
from .engine.uzu import UzuServerEngine

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

app = typer.Typer(add_completion=False)


@app.command(name=ServerEngineType.MAGNITUDE.value)
@app.command(name=ServerEngineType.TENSORFOLD.value)
@app.command(name=ServerEngineType.SPLASH.value)
@app.command(name=ServerEngineType.UZU.value)
def run_engine(
    ctx: typer.Context,
    model: Annotated[Path, typer.Argument(help="Engine model configuration JSON.")],
) -> None:
    engine_type = ServerEngineType(ctx.info_name)
    config = model.read_text()

    engine: ServerEngine
    if engine_type == ServerEngineType.UZU:
        engine = UzuServerEngine(DEFAULT_HOST, DEFAULT_PORT, ModelConfig.model_validate_json(config))
    elif engine_type == ServerEngineType.TENSORFOLD:
        engine = TensorFoldServerEngine(DEFAULT_HOST, DEFAULT_PORT, TensorFoldConfig.model_validate_json(config))
    elif engine_type == ServerEngineType.MAGNITUDE:
        engine = MagnitudeServerEngine(DEFAULT_HOST, DEFAULT_PORT, MagnitudeConfig.model_validate_json(config))
    elif engine_type == ServerEngineType.SPLASH:
        engine = SplashServerEngine(DEFAULT_HOST, DEFAULT_PORT, SplashConfig.model_validate_json(config))
    else:
        raise ValueError(f"Engine {engine_type} is not supported")

    try:
        typer.echo(f"Starting {engine_type.value} server...", err=True)
        with redirect_stdout(sys.stderr):
            engine.start()
        typer.echo("Server ready.", err=True)
        run_loop(engine)
    finally:
        with redirect_stdout(sys.stderr):
            engine.stop()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
