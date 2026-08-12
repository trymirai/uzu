//! Records an activation trace of a single forward pass, in lalamo's layout.
//!
//! Takes token ids directly. For a prompt-driven version that renders the chat
//! template, use `cli trace`.
//!
//! ```text
//! cargo run -p backend-uzu --example trace -- \
//!   --model <model dir> --tokens 9707,11,1879 --output uzu-trace.safetensors
//! ```

use std::{error::Error, path::PathBuf, process::ExitCode};

use backend_uzu::trace::record_trace;
use clap::Parser;

#[derive(Parser)]
#[command(name = "trace", bin_name = "trace")]
struct Args {
    /// Model directory holding config.json and model.safetensors.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// Token ids to run the forward pass on.
    #[arg(long, value_name = "IDS", value_delimiter = ',', required = true)]
    tokens: Vec<u64>,
    /// Where to write the trace.
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let output = record_trace(&args.model, &args.tokens, &args.output, None)?;

    println!("Recorded {} arrays to {}", output.array_count, args.output.display());

    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        },
    }
}
