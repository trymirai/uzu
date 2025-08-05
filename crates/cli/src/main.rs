use clap::{CommandFactory, Parser, Subcommand};
use cli::handlers::{handle_run, handle_serve};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a model with the specified path
    Run {
        /// Folder with model's files
        model_path: String,
    },
    /// Start a server with the specified model path
    Serve {
        /// Folder with model's files
        model_path: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run {
            model_path,
        }) => {
            handle_run(model_path, 2048);
        },
        Some(Commands::Serve {
            model_path,
        }) => {
            handle_serve(model_path);
        },
        None => {
            let mut cmd = Cli::command();
            cmd.print_help().unwrap();
        },
    }
}
