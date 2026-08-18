use anyhow::Result;
use clap::{Parser, Subcommand};
use shoji::types::basic::ReasoningEffort;

mod bench;
mod common;
mod interactive;
mod server;
mod storage;

#[derive(Parser)]
#[command(name = "cli", bin_name = "cli")]
struct Cli {
    /// Identifier of the model to start with (e.g. "alibaba:qwen3.5:0.8b:mirai:mirai-m:4").
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
    /// Reasoning effort: disabled, default, low, medium or high.
    /// Overrides the saved preference for this run only; never persisted.
    #[arg(long, value_name = "EFFORT")]
    reasoning_effort: Option<ReasoningEffort>,
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
    ListCheckpoints {
        /// Model ID shown by `list-models`.
        #[arg(value_name = "MODEL_ID")]
        model_id: String,
    },
    ListModels,
    Server {
        #[arg(long, value_name = "MODEL")]
        model: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    Storage {
        #[arg(long, value_enum, default_value_t = storage::DownloadManagerCliType::default())]
        download_manager: storage::DownloadManagerCliType,
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
        }) => bench::run_bench(model_path, task_path, output_path).await?,
        Some(Commands::ListCheckpoints {
            model_id,
        }) => interactive::run_list_checkpoints(model_id).await?,
        Some(Commands::ListModels) => interactive::run_list_models().await?,
        Some(Commands::Server {
            model,
            port,
            host,
        }) => server::run_server(model, host, port).await?,
        Some(Commands::Storage {
            download_manager,
        }) => storage::run(download_manager).await?,
        None => interactive::run_interactive(cli.model, cli.reasoning_effort).await?,
    }

    Ok(())
}
