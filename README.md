<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord" alt="Discord"></a>
<a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
<a href="https://docs.trymirai.com/overview/uzu"><img src="https://img.shields.io/badge/Read-Docs-blue" alt="Read docs"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# uzu

A high-performance inference engine for AI models on Apple Silicon. Key features:

- Simple, high-level API
- [Hybrid architecture](https://docs.trymirai.com/overview/uzu#before-we-start), where layers can be computed as GPU kernels or via MPSGraph (a low-level API beneath CoreML)
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices

## Overview

For a detailed explanation of the architecture, please refer to the [documentation](https://docs.trymirai.com/overview/uzu).

### [Models](https://trymirai.com/models)

`uzu` uses its own model format. To export a specific model, use [lalamo](https://github.com/trymirai/lalamo):

```bash
git clone https://github.com/trymirai/lalamo.git
cd lalamo
git checkout v0.6.0
```

After that, you can retrieve the list of supported models:

```bash
uv run lalamo list-models
```

Then, export the specific one:

```bash
uv run lalamo convert meta-llama/Llama-3.2-1B-Instruct
```

Alternatively, you can download a test model using the sample script:

```bash
./scripts/download_test_model.sh
```

Or you can download any supported model that has already been converted using:

```bash
cd ./tools/helpers/
uv sync # install dependencies
uv run main.py list-models # show the list of supported models
uv run main.py download-model {REPO_ID} # download a specific model using repo_id
```

After that, you can find the downloaded model at `./models/{VERSION}/`.

### Bindings

- [uzu-swift](https://github.com/trymirai/uzu-swift) - a prebuilt Swift framework, ready to use with SPM
- [uzu-ts](https://github.com/trymirai/uzu-ts) - a prebuilt TypeScript framework made for Node.js ecosystem

### CLI

You can run `uzu` in a [CLI](https://docs.trymirai.com/overview/cli) mode:

```bash
cargo run --release -p cli -- help
```

```bash
Usage: cli [COMMAND]
​
Commands:
  run    Run a model with the specified path
  serve  Start a server with the specified model path
  bench  Run benchmarks for the specified model
  help   Print this message or the help of the given subcommand(s)
```

### Compilation

For now, we only support the `Metal` backend, so to compile corresponding kernels you’ll need to install `Xcode` and run the following commands:

```bash
xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain
```

## Quick Start

First, add the `uzu` dependency to your `Cargo.toml`:

```toml
[dependencies]
uzu = { git = "https://github.com/trymirai/uzu", branch = "main", package = "uzu" }
```

Then, create an inference `Session` with a specific model and configuration:

```rust
use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    types::{Input, Output},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from("MODEL_PATH");
    let mut session = Session::new(model_path, DecodingConfig::default())?;

    let input = Input::Text(String::from("Tell about London"));
    let output = session.run(
        input,
        RunConfig::default().tokens_limit(128),
        Some(|_: Output| {
            return true;
        }),
    )?;
    println!("{}", output.text.original);
    Ok(())
}
```

## Benchmarks

To run benchmarks, you can use the following command:

```bash
cargo run --release -p cli -- bench ./models/{ENGINE_VERSION}/{MODEL_NAME} ./models/{ENGINE_VERSION}/{MODEL_NAME}/benchmark_task.json ./models/{ENGINE_VERSION}/{MODEL_NAME}/benchmark_result.json
```

`benchmark_task.json` will be automatically generated after the model is downloaded via `./tools/helpers/`, as described earlier.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
