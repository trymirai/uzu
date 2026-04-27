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
- `Python` via [pyo3](https://github.com/PyO3/pyo3)
- `TypeScript` via [napi-rs](https://github.com/napi-rs/napi-rs)

It supports:

- Backends:
{% for backend in backends %}    - `{{ backend }}`
{% endfor %}- Targets:
{% for target in targets %}    - `{{ target.name }}`{% if target.in_progress %} _(in progress)_{% endif %}
{% endfor %}
<br>
For initial setup we recommend running <code>cargo tools setup</code>, which installs all necessary dependencies (<code>rustup</code>, <code>uv</code>, <code>pnpm</code>, <code>Rust targets</code>, <code>Metal toolchain</code>, ...) if not already present.

<br>
To unify cross-language development we introduce <code>cargo tools</code>:

- Install language specific dependencies: `cargo tools install typescript`
- Build: `cargo tools build rust --targets apple`
- Test: `cargo tools test python`
- Run example: `cargo tools example swift chat`

## Model Format

`uzu` uses its own model format. You can download a test model:

```bash
./scripts/download_test_model.sh
```

Or download any supported model that has already been converted:

```bash
cd ./tools/
uv run downloader list             # show the list of supported models
uv run downloader download {REPO}  # download a specific model
```

Models downloaded for development are stored at `./workspace/models/{{ version }}/`.

You can also export a model yourself with [lalamo](https://github.com/trymirai/lalamo):

```bash
git clone https://github.com/trymirai/lalamo.git
cd lalamo
uv run lalamo list-models
uv run lalamo convert meta-llama/Llama-3.2-1B-Instruct
```

## CLI

You can run `uzu` in [CLI](https://docs.trymirai.com/overview/cli) mode:

```bash
cargo run --release -p cli -- help
```

```text
Usage: cli [COMMAND]

Commands:
  run    Run a model with the specified path
  serve  Start a server with the specified model path
  bench  Run benchmarks for the specified model
  help   Print this message or the help of the given subcommand(s)
```

## Benchmarks

To run benchmarks:

```bash
cargo run --release -p cli -- bench ./workspace/models/{{ version }}/{MODEL_NAME} ./workspace/models/{{ version }}/{MODEL_NAME}/benchmark_task.json ./workspace/models/{{ version }}/{MODEL_NAME}/benchmark_result.json
```

`benchmark_task.json` is automatically generated after the model is downloaded via `./tools/`.

{% endif %}

## Troubleshooting

If you experience any problems, please contact us via [Discord](https://discord.com/invite/trymirai) or [email](mailto:contact@getmirai.co).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
