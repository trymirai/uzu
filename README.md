# Uzu

High-performance inference engine for LLM models on Apple Silicon.

---

## Usage

You can also use Uzu directly from your own Rust programs. First, add `uzu` to your `Cargo.toml` (use a path dependency while working from the repository, or the crates.io version once published):

```toml
[dependencies]
uzu = "0.1.0"
```

Then create a `Session`, load the model, and run it:

```rust
use std::path::PathBuf;
use uzu::session::{
    session::Session,
    session_config::{SessionConfig, SessionRunConfig},
    session_input::SessionInput,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Directory that contains model weights, tokenizer, etc.
    let model_dir = PathBuf::from("path/to/model/dir");

    // Create a new session bound to the model directory
    let mut session = Session::new(model_dir.clone())?;

    // Load the model with default configuration
    session.load_with_session_config(SessionConfig::default())?;

    // Prepare input (plain text or chat messages)
    let input = SessionInput::Text("Hello, world!".to_string());

    // Limit generation to 64 new tokens
    let run_config = SessionRunConfig::new(64);

    // Run inference
    let output = session.run(input, run_config, None);

    println!("Generated: {}", output.text);

    Ok(())
}
```

## CLI quick-start

For demo purposes you can fetch a test model:

```bash
./scripts/download_test_model.sh
```

### Interactive *run* mode

```bash
cargo run -p cli -- run <MODEL_DIR>
```

### Server *serve* mode

```bash
cargo run -p cli -- serve <MODEL_DIR>
```

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.
