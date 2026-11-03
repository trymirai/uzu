use benchmarks_uzu::Args;
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    benchmarks_uzu::run(Args::parse()).await
}
