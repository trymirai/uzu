use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
mod server;

#[derive(Parser)]
#[command(name = "cli", bin_name = "cli")]
struct Cli {
    /// Identifier of the model to start with (e.g. "Qwen/Qwen3-0.6B").
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
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
    Server {
        #[arg(long, value_name = "MODEL")]
        model: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
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
        Some(Commands::Server {
            model,
            port,
            host,
        }) => server::run_server(model, host, port).await?,
        None => run_interactive(cli.model).await?,
    }

    Ok(())
}

#[cfg(feature = "capability-cli")]
async fn run_interactive(model: Option<String>) -> Result<()> {
    use uzu::{cli::CliApplication, engine::EngineConfig};

    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let application = CliApplication::create(engine_config).await?;
    application.run_with_model(model).await?;

    Ok(())
}

#[cfg(not(feature = "capability-cli"))]
async fn run_interactive(_model: Option<String>) -> Result<()> {
    Ok(())
}
