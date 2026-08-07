use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
mod interactive;
mod server;
mod storage;

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
        Some(Commands::ListModels {}) => interactive::run_list_models().await?,
        Some(Commands::Server {
            model,
            port,
            host,
        }) => server::run_server(model, host, port).await?,
        Some(Commands::Storage {
            download_manager,
        }) => storage::run(download_manager).await?,
        None => interactive::run_interactive(cli.model).await?,
    }

    Ok(())
}
