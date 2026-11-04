import typer
from benchman.pickle_by_value_hack import pickle_benchman_by_value
from benchman.python_warnings_hack import silence_experimental_bundles_warning
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@app.callback()
def benchman() -> None:
    """Benchmark orchestration."""

    silence_experimental_bundles_warning()
    pickle_benchman_by_value()
