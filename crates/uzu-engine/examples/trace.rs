use std::{error::Error, path::PathBuf, process::ExitCode};

use clap::Parser;
use uzu_engine::trace::record_trace;

#[derive(Parser)]
#[command(name = "trace", bin_name = "trace")]
struct Args {
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    #[arg(long, value_name = "IDS", value_delimiter = ',', required = true)]
    tokens: Vec<u64>,
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
