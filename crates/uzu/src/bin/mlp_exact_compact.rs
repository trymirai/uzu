use std::path::PathBuf;

use clap::Parser;
use uzu::{ExactMlpCompactionError, compact_model_directory};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    input_model_path: PathBuf,
    #[arg(long)]
    output_model_path: PathBuf,
}

fn main() -> Result<(), ExactMlpCompactionError> {
    let args = Args::parse();
    let manifest = compact_model_directory(&args.input_model_path, &args.output_model_path)?;
    println!(
        "Compacted {} dense MLP layers ({} scanned)",
        manifest.compacted_layers.len(),
        manifest.scanned_dense_layers,
    );
    Ok(())
}
