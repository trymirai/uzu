from rich.console import Console
from typer import Typer

from data_exporter.export import export_bundled, export_configs, export_responses, export_tokenizers

console = Console()
err_console = Console(stderr=True)
app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.command(help="Export configs")
def configs() -> None:
    export_configs(console, err_console)


@app.command(help="Export tokenizers")
def tokenizers() -> None:
    export_tokenizers(console, err_console)


@app.command(help="Export bundled configs")
def bundled() -> None:
    export_bundled(console, err_console)


@app.command(help="Export responses")
def responses() -> None:
    export_responses(console, err_console)


if __name__ == "__main__":
    app()
