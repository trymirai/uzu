use clap::{CommandFactory, Parser, Subcommand};
use cli::handlers::{
    handle_bench, handle_nanocodec_roundtrip, handle_run, handle_serve,
};

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
        /// Prefill step size
        prefill_step_size: Option<usize>,
        // Seed
        #[arg(long)]
        seed: Option<u64>,
        /// Non-interactive mode: run a single message and exit
        #[arg(long, short)]
        message: Option<String>,
    },
    /// Start a server with the specified model path
    Serve {
        /// Folder with model's files
        model_path: String,
        /// Prefill step size
        prefill_step_size: Option<usize>,
    },
    /// Run benchmarks for the specified model
    Bench {
        /// Folder with model's files
        model_path: String,
        /// Path to the task file
        task_path: String,
        /// Path to the output file
        output_path: String,
    },
    /// Run NanoCodec audio -> tokens -> audio roundtrip
    NanoCodecRoundtrip {
        /// Path to the `.nemo` archive
        nemo_path: String,
        /// Input WAV file path
        input_wav: String,
        /// Output WAV file path
        output_wav: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run {
            model_path,
            prefill_step_size,
            seed,
            message,
        }) => {
            handle_run(model_path, 2048, prefill_step_size, seed, message);
        },
        Some(Commands::Serve {
            model_path,
            prefill_step_size,
        }) => {
            handle_serve(model_path, prefill_step_size);
        },
        Some(Commands::Bench {
            model_path,
            task_path,
            output_path,
        }) => {
            let _ = handle_bench(model_path, task_path, output_path);
        },
        Some(Commands::NanoCodecRoundtrip {
            nemo_path,
            input_wav,
            output_wav,
        }) => {
            if let Err(e) =
                handle_nanocodec_roundtrip(nemo_path, input_wav, output_wav)
            {
                eprintln!("Error: {e}");
            }
        },
        None => {
            let mut cmd = Cli::command();
            cmd.print_help().unwrap();
        },
    }
}
