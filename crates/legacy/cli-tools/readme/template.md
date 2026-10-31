<p align="center">
  <picture>
    <img alt="Mirai" src="{{ image_url }}" style="max-width: 100%;">
  </picture>
</p>

{% for badge in badges %}{{ badge }} {% endfor %}

# {{ title }}

{{ intro }}

## Quick Start

{% for language in languages %}
{% if has_multiple_languages %}
<details>
<summary>{{ language.display }}</summary>
<br>
{% endif %}
Add the dependency:

{{ language.dependency }}

Run the code below:

```{{ language.code_fence }}
{{ language.quick_start }}
```
{% if has_multiple_languages %}
</details>
{% endif %}
{% endfor %}
<br>

Everything from model downloading to inference configuration is handled automatically. Refer to the [documentation](https://docs.trymirai.com) for details on how to customize each step of the process.

## Examples

You can run any example via `cargo tools example` \<{% for language in languages %}**{{ language.name }}**{% if not loop.last %} | {% endif %}{% endfor %}\> \<{% for example in examples %}**{{ example.name }}**{% if not loop.last %} | {% endif %}{% endfor %}\>:

{% for example in examples %}{% if example.name != "quick-start" %}### {{ example.title }}

{{ example.description }}

{% for language in languages %}{% if has_multiple_languages %}<details>
<summary>{{ language.display }}</summary>

{% endif %}```{{ language.code_fence }}
{{ language.examples[example.name] }}
```
{% if has_multiple_languages %}
</details>

{% endif %}{% endfor %}
{% if example.explanation %}<br>{{ example.explanation }}

{% endif %}{% endif %}{% endfor %}{% if has_multiple_languages %}
## Development

`uzu` is a native Rust crate with bindings available for:

- `Swift` via [uniffi-rs](https://github.com/mozilla/uniffi-rs)
- `Kotlin` via [uniffi-rs](https://github.com/mozilla/uniffi-rs)
- `Python` via [pyo3](https://github.com/PyO3/pyo3)
- `TypeScript` via [napi-rs](https://github.com/napi-rs/napi-rs)

Supported backends:

- `metal`
- `vulkan`
- `cpu`

<br>
For initial setup we recommend running <code>cargo tools setup</code>, which installs all necessary dependencies (<code>rustup</code>, <code>uv</code>, <code>pnpm</code>, <code>Rust targets</code>, <code>Metal toolchain</code>, ...) if not already present.

<br>
To unify cross-language development we introduce <code>cargo tools</code>:

- Install language specific dependencies: `cargo tools install typescript`
- Build: `cargo tools build rust --targets apple`
- Test: `cargo tools test python`
- Run example: `cargo tools example swift chat`

## Model Format

`uzu` uses its own model format. You can export a model yourself with [lalamo](https://github.com/trymirai/lalamo):

```bash
git clone https://github.com/trymirai/lalamo.git
cd lalamo
uv run lalamo list-models
uv run lalamo convert meta-llama/Llama-3.2-1B-Instruct
```

## CLI

You can run `uzu` in CLI mode:

```bash
cargo run --release -p cli
```

On Linux and Windows this runs on Vulkan. The CLI names the backend it runs on when it starts.

This launches an interactive app where you can browse, download, and interact with models.

You can also preselect a model with `--model`, passing its identifier or repository id:

```bash
cargo run --release -p cli -- --model trymirai/Qwen3.5-4B-M
```

If the model is not downloaded yet, the CLI starts downloading it automatically.

On Linux and Windows, Vulkan needs:

- the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) through `VULKAN_SDK`, or Slang 2026.17 or newer through `SLANG_DIR`, or `SLANG_INCLUDE_DIR` and `SLANG_LIB_DIR`
- libclang, which Windows finds through `LIBCLANG_PATH`
- a Vulkan 1.4 driver: a GPU's, or Mesa's lavapipe, which runs Vulkan on the CPU
- `VK_LAYER_KHRONOS_validation` (part of the SDK) for the library tests

On Linux, `nix develop` provides everything but the GPU driver. To build the CLI for the CPU backend alone, without Vulkan:

```bash
cargo run --release -p cli --no-default-features --features cpu
```

A Windows release build that runs without the Visual C++ Redistributable links the C and C++ runtimes statically. Set the flag for the target alone and name the target, so build scripts keep the dynamic runtime: Windows Defender quarantines statically linked build scripts as a false positive.

```powershell
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --release -p cli --target x86_64-pc-windows-msvc  # target\x86_64-pc-windows-msvc\release\cli.exe
Remove-Item Env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS
```

## Benchmarks

To run benchmarks, pass a downloaded model path, a benchmark task file, and an output path:

```bash
cargo run --release -p cli -- bench {MODEL_PATH} {TASK_PATH} {OUTPUT_PATH}
```

## Server

You can also run `uzu` as an OpenAI-compatible HTTP server:

```bash
cargo run --release -p cli -- server --model trymirai/Qwen3.5-4B-M
```

The model is loaded on startup (and downloaded first if needed). By default the server listens on `127.0.0.1:8000`; override the address with `--host` and `--port`:

```bash
cargo run --release -p cli -- server --model trymirai/Qwen3.5-4B-M --host 0.0.0.0 --port 8080
```

It exposes the following endpoints, available both at the root and under `/v1`:

- `POST /v1/chat/completions` — chat completions, with streaming when `"stream": true`. Honors `temperature`, `top_p`, `top_k`, `max_tokens`, and `seed`. Requests without a `seed` use the one passed with `--seed`, or a random one.
- `GET /v1/models` — lists the loaded model.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "trymirai/Qwen3.5-4B-M",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    }'
```

### Decision models

When `--model` is a decision model, the server serves the [TypeSafe System One API](https://api.typesafe.ai/redoc) instead of chat completions:

```bash
cargo run --release -p cli -- server --model trymirai/d3-flash-M
```

It exposes the following endpoints, available both at the root and under `/v1`:

- `POST /v1/systemone` (or `/v1/decisions`) — answers questions about a `state` text in one request.
- `GET /v1/models` — lists the loaded model.

The request body has a `state` string and a `questions` object that maps your question names to questions. Each question has a `type`, optional `instructions`, and `criteria`:

- `choice` — picks one option. `criteria` maps 1 to 255 option names to their descriptions (or `null`). The answer has the chosen name, a probability for each name, and a confidence.
- `noul` — yes or no. `criteria` is optional and may describe the answers as `{"false": "...", "true": "..."}`. The answer `noul` is the probability of yes.
- `score` — rates on a scale. `criteria` lists 2 to 10 distinct levels from lowest to highest. The answer `score` is the expected level index, with a probability for each level index and a `legend` mapping indices to levels.

```bash
curl http://127.0.0.1:8000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
        "state": "The headphones arrived a day early and sound amazing. The battery lasts all week, and I have already recommended them to two friends.",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "What is this message about?",
                "criteria": {
                    "praise": "Compliments about a product or service",
                    "question": "Asks for information or help",
                    "complaint": "Reports a problem"
                }
            },
            "recommend": {
                "type": "noul",
                "instructions": "Would the customer recommend the product?"
            },
            "satisfaction": {
                "type": "score",
                "instructions": "How satisfied is the customer?",
                "criteria": ["Unhappy", "Neutral", "Happy", "Delighted"]
            }
        }
    }'
```

The response answers each question under its name (probabilities rounded here):

```json
{
    "model": "vllm-sr:d3-flash:9b:mirai:mirai-m:4",
    "answers": {
        "topic": {
            "type": "choice",
            "choice": "praise",
            "probabilities": {"praise": 0.9995, "question": 0.0001, "complaint": 0.0004},
            "confidence": 0.9957
        },
        "recommend": {
            "type": "noul",
            "noul": 0.9987
        },
        "satisfaction": {
            "type": "score",
            "score": 2.8875,
            "probabilities": {"0": 0.0005, "1": 0.0015, "2": 0.1080, "3": 0.8900},
            "confidence": 0.7420,
            "legend": {"0": "Unhappy", "1": "Neutral", "2": "Happy", "3": "Delighted"}
        }
    },
    "usage": {"input_tokens": 374, "output_tokens": 0}
}
```

Compared to the TypeSafe API, the server is text only: `state`, `instructions`, and criteria are strings, and there are no `assets` or `box` questions. `model` is optional and ignored, since the loaded model answers. Errors use the same OpenAI-style body as chat completions: an invalid question is rejected with status 400 and names it, for example `"param": "questions.satisfaction"`.

{% endif %}

## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
