from pathlib import Path
from typing import Annotated

import requests
import tomllib
from model import Registry
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from typer import Argument, Typer
from utils import download_file_with_resume

ROOT_PATH = Path(__file__).parent.parent.parent
CARGO_TOML_PATH = ROOT_PATH / "Cargo.toml"
MODELS_PATH = ROOT_PATH / "models"


console = Console()
err_console = Console(stderr=True)
app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def get_uzu_version() -> str:
    with open(CARGO_TOML_PATH, "rb") as file:
        cargo_toml = tomllib.load(file)
        return cargo_toml["workspace"]["package"]["version"]


def load_registry() -> Registry:
    url = f"https://sdk.trymirai.com/api/v1/models/list/uzu/{get_uzu_version()}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    registry = Registry.from_dict(data)
    return registry


@app.command(help="List models")
def list_models():
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading registry...", total=None)
            registry = load_registry()
            progress.update(task, completed=True)

        table = Table()
        table.add_column("Repo", style="cyan")
        table.add_column("Name", style="green")
        for model in registry.models:
            table.add_row(model.repod_id, model.name)
        console.print(table)

    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")


@app.command(help="Download model")
def download_model(
    model_repo: Annotated[
        str,
        Argument(
            help=(
                "Hugging Face model repo. Example: [cyan]'meta-llama/Llama-3.2-1B-Instruct'[/cyan]."
            ),
        ),
    ],
):
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading registry...", total=None)
            registry = load_registry()
            progress.update(task, completed=True)

        model = None
        for current_model in registry.models:
            if current_model.repod_id == model_repo:
                model = current_model
                break
        if not model:
            err_console.print(
                f"[red]Error: Model '{model_repo}' not found in registry[/red]"
            )
            return

        engine_version = get_uzu_version()
        engine_path = MODELS_PATH / engine_version
        model_path = engine_path / model.name
        model_path.mkdir(parents=True, exist_ok=True)

        files_to_download = []
        for file in model.files:
            file_path = model_path / file.name
            if file_path.exists():
                console.print(f"[green]✓[/green] {file.name} already exists")
            else:
                files_to_download.append((file, file_path))
        if not files_to_download:
            console.print("[green]All files already downloaded![/green]")
            return

        console.print(
            f"Downloading {len(files_to_download)} file(s) for [cyan]{model.name}[/cyan]..."
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            for file, file_path in files_to_download:
                task = progress.add_task(f"[cyan]{file.name}[/cyan]", total=None)

                try:
                    download_file_with_resume(file.url, file_path, progress, task)
                    progress.update(task, description=f"[green]✓[/green] {file.name}")
                except Exception as e:
                    progress.update(task, description=f"[red]✗[/red] {file.name}")
                    raise e

        console.print(f"[green]Successfully downloaded model: {model.name}[/green]")

    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()
