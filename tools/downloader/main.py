import base64
import json
import os
import platform
import re
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import google_crc32c
import requests
from cattrs.errors import BaseValidationError
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
from typer import Argument, Option, Typer

from .model import (
    BenchmarkTask,
    Device,
    File,
    Message,
    Model,
    Registry,
    Role,
    parse_registry,
)
from .utils import download_file_with_resume

ROOT_PATH = Path(__file__).parent.parent.parent
CARGO_TOML_PATH = ROOT_PATH / "Cargo.toml"
TOOLCHAIN_VERSION_PATH = ROOT_PATH / "crates" / "backend-uzu" / "src" / "utils" / "version.rs"
WORKSPACE_PATH = ROOT_PATH / "workspace"
MODELS_PATH = WORKSPACE_PATH / "models"
MIRAI_API_URL = "https://sdk.trymirai.com/api/v1/fetch/models"
USER_FACING_ERRORS = (
    BaseValidationError,
    OSError,
    ValueError,
    requests.RequestException,
)


console = Console()
err_console = Console(stderr=True)
app = Typer(
    rich_markup_mode="rich",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def crc32c_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")


def get_uzu_version() -> str:
    with CARGO_TOML_PATH.open("rb") as file:
        cargo_toml = tomllib.load(file)
        return cargo_toml["workspace"]["package"]["version"]


def get_toolchain_version() -> str:
    version_source = TOOLCHAIN_VERSION_PATH.read_text()
    match = re.search(r'pub const TOOLCHAIN_VERSION: &str = "([^"]+)";', version_source)
    if match is None:
        raise ValueError(f"Unable to read TOOLCHAIN_VERSION from {TOOLCHAIN_VERSION_PATH}")
    return match.group(1)


def build_device() -> Device:
    try:
        memory_total: int | None = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        memory_total = None
    return Device(
        os_name=platform.platform(),
        cpu_name=platform.processor() or None,
        memory_total=memory_total,
    )


def load_registry() -> Registry:
    payload = {
        "device": {key: value for key, value in asdict(build_device()).items() if value is not None},
        "backends": [{"identifier": "uzu", "version": get_toolchain_version()}],
        "include_traces": True,
    }
    response = requests.post(MIRAI_API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return parse_registry(response.json())


@app.command(name="list", help="List models")
def list_models(
    repos_only: Annotated[
        bool,
        Option("--repos", help="Print only model repository IDs."),
    ] = False,
) -> None:
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

        if repos_only:
            for model in registry.models:
                console.print(model.repo_id, markup=False, highlight=False)
            return

        table = Table()
        table.add_column("Repo", style="cyan")
        table.add_column("Name", style="green")
        for model in registry.models:
            table.add_row(model.repo_id, model.name)
        console.print(table)

    except USER_FACING_ERRORS as error:
        err_console.print(f"[red]Error: {error}[/red]")


@app.command(help="Download model")
def download(
    model_repo: Annotated[
        str,
        Argument(
            help=("Hugging Face model repo. Example: [cyan]'meta-llama/Llama-3.2-1B-Instruct'[/cyan]."),
        ),
    ],
) -> None:
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

        model = next((m for m in registry.models if model_repo in {m.id, m.repo_id, m.name}), None)
        if not model:
            err_console.print(f"[red]Error: Model '{model_repo}' not found in registry[/red]")
            return

        engine_version = get_uzu_version()
        engine_path = MODELS_PATH / engine_version
        model_path = engine_path / model.name
        model_path.mkdir(parents=True, exist_ok=True)

        benchmark_task = generate_benchmark_task(model)
        benchmark_task_data = asdict(benchmark_task)
        benchmark_task_name = "benchmark_task.json"
        benchmark_task_path = model_path / benchmark_task_name
        with open(benchmark_task_path, "w") as file:
            json.dump(benchmark_task_data, file, indent=4, sort_keys=True)
        console.print(f"[green]✓[/green] {benchmark_task_name} generated")

        files_to_download = []

        def need_to_download(file: File, file_path: Path) -> bool:
            if not file_path.exists():
                return True
            if file.crc32c == crc32c_file(file_path):
                return False
            file_path.unlink()
            return True

        for file in model.files:
            file_path = model_path / file.name
            need_to_download_file = need_to_download(file, file_path)
            if not need_to_download_file:
                console.print(f"[green]✓[/green] {file.name} already exists")
            else:
                files_to_download.append((file, file_path))

        if not files_to_download:
            console.print("[green]All files already downloaded![/green]")
            return

        console.print(f"Downloading {len(files_to_download)} file(s) for [cyan]{model.name}[/cyan]...")
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
                except USER_FACING_ERRORS:
                    progress.update(task, description=f"[red]✗[/red] {file.name}")
                    raise

        console.print(f"[green]Successfully downloaded model: {model.name}[/green]")

    except USER_FACING_ERRORS as error:
        err_console.print(f"[red]Error: {error}[/red]")


def generate_benchmark_task(model: Model) -> BenchmarkTask:
    messages: list[Message] = [
        Message(role=Role.SYSTEM, content="Summarize user's input"),
        Message(
            role=Role.USER,
            content="Large language models, commonly referred to as LLMs, are a class of artificial intelligence systems designed to understand, generate, and manipulate human language at scale. They are built using deep learning techniques, most notably transformer architectures, and are trained on vast collections of text data drawn from books, articles, websites, and other written sources. Through this training process, LLMs learn statistical patterns in language, allowing them to predict likely sequences of words and produce coherent, contextually appropriate responses to prompts. At their core, LLMs operate by representing words or subword units as numerical vectors in a high-dimensional space. These representations capture semantic and syntactic relationships, such that words with similar meanings or grammatical roles tend to have similar vector representations. The transformer architecture enables the model to process entire sequences of text simultaneously rather than sequentially, using mechanisms such as self-attention to determine which parts of the input are most relevant at any given moment. This allows LLMs to handle long-range dependencies in text, such as references made many sentences earlier, more effectively than earlier generations of language models. Training an LLM typically involves two main phases. The first is pretraining, during which the model learns general language patterns by predicting missing or next tokens in large, mostly uncurated text corpora. This phase gives the model broad linguistic competence and general world knowledge as reflected in its training data. The second phase often involves fine-tuning, where the model is further trained on more specific datasets, such as question-answer pairs, instructional content, or conversational examples. Fine-tuning helps align the model’s behavior with particular tasks or desired styles of interaction. One of the most notable characteristics of LLMs is their versatility. A single model can perform a wide range of language-related tasks, including text generation, summarization, translation, classification, question answering, and code generation, often without task-specific retraining. This flexibility arises from the model’s general-purpose training and its ability to condition its outputs on the instructions or context provided in the prompt. As a result, LLMs are increasingly used as foundational models that can be adapted to many applications across different domains. Despite their impressive capabilities, LLMs do not possess true understanding or consciousness. They do not have beliefs, intentions, or awareness in the human sense. Instead, they generate responses based on learned correlations in data. This limitation can lead to errors such as producing confident-sounding but incorrect information, sometimes referred to as hallucinations. Because LLMs rely on patterns in their training data, they may also reflect biases, inaccuracies, or gaps present in that data. Addressing these issues is an ongoing area of research and development. The computational cost of training and running LLMs is another significant consideration. Training state-of-the-art models can require enormous amounts of computing power, energy, and financial investment. This has implications for environmental sustainability and for who can realistically develop and deploy such models. As a result, there is growing interest in techniques that improve efficiency, such as model compression, distillation, sparse architectures, and more efficient training algorithms, as well as in smaller models that can perform well on specific tasks. LLMs also raise important ethical and social questions. Their ability to generate human-like text can be beneficial in areas such as education, accessibility, and productivity, but it can also be misused for purposes like misinformation, plagiarism, or automated spam. Ensuring responsible use involves a combination of technical safeguards, policy decisions, and user education. Researchers and developers are actively exploring methods for making LLMs more transparent, controllable, and aligned with human values. As research continues, LLMs are likely to become more capable, more efficient, and more integrated into everyday tools and workflows. Future developments may involve better reasoning abilities, improved factual reliability, stronger multimodal integration with images, audio, and video, and more personalized interactions that adapt to individual users while respecting privacy. While LLMs are not a replacement for human judgment or creativity, they represent a powerful technology that, when used thoughtfully, can augment human capabilities and transform how people interact with information and with machines.",  # noqa: E501, RUF001
        ),
    ]

    return BenchmarkTask(
        identifier=model.name.lower(),
        repo_id=model.repo_id,
        number_of_runs=15,
        tokens_limit=512,
        messages=messages,
        greedy=False,
    )


if __name__ == "__main__":
    app()
