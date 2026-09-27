mod bench;
mod common;
mod engine;
mod memory;

use clap::Parser;

use crate::{common::run_loop, engine::UzuEngine};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    model: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut engine = UzuEngine::new(args.model.as_str()).await?;
    run_loop(&mut engine).await?;
    Ok(())
}
