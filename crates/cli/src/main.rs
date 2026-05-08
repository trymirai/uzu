use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;

#[derive(Parser)]
#[command(name = "cli", bin_name = "cli")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Bench {
        model_path: String,
        task_path: String,
        output_path: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Bench {
            model_path,
            task_path,
            output_path,
        }) => bench::run_bench(model_path, task_path, output_path)?,
        None => run_interactive().await?,
    }

    Ok(())
}

#[cfg(feature = "capability-cli")]
async fn run_interactive() -> Result<()> {
    use uzu::{cli::CliApplication, engine::EngineConfig};

    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let application = CliApplication::create(engine_config).await?;
    application.run().await?;

    Ok(())
}

#[cfg(not(feature = "capability-cli"))]
async fn run_interactive() -> Result<()> {
    Ok(())
}
