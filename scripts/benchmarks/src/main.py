from typer import Typer

import engine_mlx
from engine_mlx import MlxRunRequest, MlxRunResponse

app = Typer(add_completion=False)


@app.command("mlx")
def run_mlx():
    config = MlxRunRequest(
        # model_path="~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-8bit/snapshots/e6ffd0033d03c9efa880984e611028e0da63905f",
        model_path="mlx-community/Qwen3.5-4B-8bit",
        prompt="Tell me about London",
    )
    response: MlxRunResponse = engine_mlx.run(config)
    print(response)


if __name__ == "__main__":
    app()
