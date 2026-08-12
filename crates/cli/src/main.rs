use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
mod interactive;
mod server;
mod storage;
#[cfg(feature = "capability-trace")]
mod trace;

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
    /// Record an activation trace of a single forward pass, in lalamo's layout.
    #[cfg(feature = "capability-trace")]
    Trace {
        #[arg(long, value_name = "DIR")]
        model_path: String,
        /// User message to run the forward pass on.
        #[arg(long, value_name = "TEXT")]
        message: String,
        #[arg(long, value_name = "FILE")]
        output_path: String,
        /// Trace a classifier model instead of a language model.
        #[arg(long)]
        classifier: bool,
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
        #[cfg(feature = "capability-trace")]
        Some(Commands::Trace {
            model_path,
            message,
            output_path,
            classifier,
        }) => trace::run_trace(model_path, message, output_path, classifier).await?,
        None => interactive::run_interactive(cli.model).await?,
    }

    Ok(())
}
