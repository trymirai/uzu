import asyncio
import json
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletion

from .util import await_cooldown, get_openai_base_url, load_input

app = typer.Typer(add_completion=False)


async def run(
    input: Annotated[Path, typer.Option("--input", help="Path to a JSON file.")],
    model: Annotated[str | None, typer.Option("--model", help="Model name. Omit to use the server default.")] = None,
    host: Annotated[str, typer.Option(help="Server hostname or IP address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Server HTTP port.")] = 8000,
    num_runs: Annotated[int, typer.Option(min=1, help="Number of times to send the request sequentially.")] = 1,
    wait_cooling: Annotated[
        bool, typer.Option("--wait-cooling", help="Wait before CPU and GPU are cooled enough.")
    ] = True,
) -> None:
    request: dict[str, Any] = load_input(input)
    client = AsyncOpenAI(
        base_url=get_openai_base_url(host, port),
        api_key="not-needed",
    )

    responses: list[dict[str, Any]] = []
    for _ in range(num_runs):
        await_cooldown()
        response: ChatCompletion = await client.chat.completions.create(
            model=model if model is not None else cast(Any, omit),
            messages=request["messages"],
        )
        response_json: dict[str, Any] = response.model_dump(mode="json")
        responses.append(response_json)

    typer.echo(json.dumps(responses, ensure_ascii=False, indent=2))


@app.command()
@wraps(run)
def run_cli(*args: Any, **kwargs: Any) -> None:
    asyncio.run(run(*args, **kwargs))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
