from typer import Typer

import engine_mlx
import engine_mtplx
from engine_mlx import MlxRunRequest, MlxRunResponse
from engine_mtplx import MtplxRunRequest, MtplxRunResponse

app = Typer(add_completion=False)


@app.command("mlx")
def run_mlx():
    request = MlxRunRequest(
        model="mlx-community/Qwen3.6-27B-4bit",
        prompt="Tell me about London",
        max_tokens=256,
    )
    response: MlxRunResponse = engine_mlx.run(request)
    print(response)


@app.command("mtplx")
def run_mtplx():
    request = MtplxRunRequest(
        model="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
        prompt="Tell me about London",
        speculative_depth=1,
        max_tokens=256,
    )
    response: MtplxRunResponse = engine_mtplx.run(request)
    print(response)


if __name__ == "__main__":
    app()
