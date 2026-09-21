use anyhow::Result;
use clap::{Parser, Subcommand};
use shoji::types::basic::ReasoningEffort;

mod bench;
mod common;
mod interactive;
mod server;
mod storage;

#[derive(Parser)]
#[command(name = "cli", bin_name = "cli", version, args_conflicts_with_subcommands = true)]
struct Cli {
    /// Model identifier, repository ID, or local model directory.
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
    /// Send one chat message, stream the reply, and exit.
    #[arg(short, long, value_name = "TEXT", requires = "model")]
    message: Option<String>,
    /// Reasoning effort: disabled, default, low, medium or high.
    /// Overrides the saved preference for this run only; never persisted.
    #[arg(long, value_name = "EFFORT")]
    reasoning_effort: Option<ReasoningEffort>,
    /// Sampling seed for chat sessions.
    #[arg(long, value_name = "SEED", allow_negative_numbers = true)]
    seed: Option<i64>,
    /// Disable built-in tools in interactive chat sessions.
    #[arg(long)]
    no_tools: bool,
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
        /// Model identifier, repository ID, or local model directory.
        #[arg(long, value_name = "MODEL")]
        model: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Reuse the previous request's state when the new prompt extends it (enabled by default).
        #[arg(long = "prefix-cache", action = clap::ArgAction::SetTrue, conflicts_with = "no_prefix_cache")]
        prefix_cache: bool,
        /// Reset the session on every request, disabling prefix cache reuse.
        #[arg(long = "no-prefix-cache", action = clap::ArgAction::SetTrue)]
        no_prefix_cache: bool,
        /// Print verbose information into the file log
        #[arg(long)]
        verbose_file_log: bool,
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
            prefix_cache,
            no_prefix_cache,
            verbose_file_log,
        }) => server::run_server(model, host, port, prefix_cache || !no_prefix_cache, verbose_file_log).await?,
        Some(Commands::Storage {
            download_manager,
        }) => storage::run(download_manager).await?,
        None => match cli.message {
            Some(message) => {
                interactive::run_non_interactive(
                    cli.model.expect("--message requires --model"),
                    message,
                    cli.reasoning_effort,
                    cli.seed,
                )
                .await?;
            },
            None => interactive::run_interactive(cli.model, cli.reasoning_effort, cli.seed, cli.no_tools).await?,
        },
    }

    Ok(())
}
