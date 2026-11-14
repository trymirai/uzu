import httpx2
import typer


def get_openai_base_url(host: str, port: int) -> httpx2.URL:
    host = host.strip().removeprefix("[").removesuffix("]")
    if not host or any(character in host for character in "/?#@"):
        raise typer.BadParameter("Use a hostname or IP address without a URL scheme or path.", param_hint="--host")

    try:
        return httpx2.URL(scheme="http", host=host, port=port, path="/v1")
    except httpx2.InvalidURL as error:
        raise typer.BadParameter(str(error), param_hint="--host") from error
