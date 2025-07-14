<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/uzu-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://artifacts.trymirai.com/social/about_us.mp3"><img src="https://img.shields.io/badge/Listen-Podcast-red" alt="Listen to our podcast"></a>
<a href="https://docsend.com/v/76bpr/mirai2025"><img src="https://img.shields.io/badge/View-Deck-red" alt="View our deck"></a>
<a href="mailto:alexey@getmirai.co,dima@getmirai.co,aleksei@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
<a href="https://docs.trymirai.com/components/inference-engine"><img src="https://img.shields.io/badge/Read-Docs-blue" alt="Read docs"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# uzu

A high-performance inference engine for AI models on Apple Silicon. Key features:

- Simple, high-level API
- [Hybrid architecture](https://docs.trymirai.com/components/inference-engine#before-we-start), where layers can be computed as GPU kernels or via MPSGraph (a low-level API beneath CoreML with [ANE](https://trymirai.com/blog/iphone-hardware) access)
- Unified model configurations, making it easy to add support for new models
- Traceable computations to ensure correctness against the source-of-truth implementation
- Utilizes unified memory on Apple devices

## Quick Start

First, add the `uzu` dependency to your `Cargo.toml`:

```toml
[dependencies]
uzu = { git = "https://github.com/trymirai/uzu", branch = "main", package = "uzu" }
```

Then, create an inference `Session` with a specific model and configuration:

```rust
use std::path::PathBuf;
use uzu::{
    backends::metal::sampling_config::SamplingConfig,
    session::{
        session::Session, session_config::SessionConfig,
        session_input::SessionInput, session_output::SessionOutput,
        session_run_config::SessionRunConfig,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from("MODEL_PATH");
    
    let mut session = Session::new(model_path.clone())?;
    session.load_with_session_config(SessionConfig::default())?;

    let input = SessionInput::Text("Tell about London".to_string());

    let tokens_limit = 128;
    let run_config = SessionRunConfig::new_with_sampling_config(
        tokens_limit,
        Some(SamplingConfig::default())
    );

    let output = session.run(input, run_config, Some(|_: SessionOutput| {
        return true;
    }));
    println!("{}", output.text);
    Ok(())
}
```

## Overview

For a detailed explanation of the architecture, please refer to the [documentation](https://docs.trymirai.com/components/inference-engine).

### [Models](https://trymirai.com/models)

`uzu` uses its own model format. To export a specific model, use [lalamo](https://github.com/trymirai/lalamo). First, get the list of supported models:

```bash
uv run lalamo list-models
```

Then, export the specific one:

```bash
uv run lalamo convert meta-llama/Llama-3.2-1B-Instruct --precision float16
```

Alternatively, you can download a preprepared model using the sample script:

```bash
./scripts/download_test_model.sh $MODEL_PATH
```

### CLI

You can run `uzu` in a [CLI](https://docs.trymirai.com/components/cli) mode:

```bash
cargo run --release -p cli -- help
```

```bash

Usage: uzu_cli [COMMAND]
​
Commands:
  run    Run a model with the specified path
  serve  Start a server with the specified model path
  help   Print this message or the help of the given subcommand(s)
```

### Bindings

- [uzu-swift](https://github.com/trymirai/uzu-swift) - a prebuilt Swift framework, ready to use with SPM

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
